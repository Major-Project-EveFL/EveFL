"""
Secure aggregation, step 2a: the 64-bit fixed-point codec (ADR 0003, "Fixed-point arithmetic" and acceptance test 10).

Updates are scaled to 64-bit integers with 32 fractional bits and added modulo 2^64, so that pairwise masks cancel exactly.
Out-of-range values wrap silently in that arithmetic, so range is asserted before masking and the aggregate is checked after.
"""

import numpy as np
import pytest

from evefl.crypto.secagg import fixedpoint as fp

HALF_UNIT = 2.0 ** -33   # rounding error of one encoded value: half a unit of the 32-bit fractional grid
RANGE = 2.0 ** 31        # the representable sum is [-2^31, 2^31)


def _rng(seed=0):
    return np.random.default_rng(seed)


# --------------------------------------------------------------------------
# encoding
# --------------------------------------------------------------------------

def test_constants_match_the_adr():
    assert fp.FRACTIONAL_BITS == 32 and fp.MODULUS_BITS == 64 and fp.SCALE == 2 ** 32 and fp.LIMIT == 2 ** 31


def test_encode_is_twos_complement_in_uint64():
    encoded = fp.encode(np.array([1.0, -1.0, 0.0, 0.5, -0.5]))
    assert encoded.dtype == np.uint64
    assert [int(v) for v in encoded] == [2 ** 32, 2 ** 64 - 2 ** 32, 0, 2 ** 31, 2 ** 64 - 2 ** 31]


def test_round_trip_error_is_at_most_half_a_unit_of_the_grid():
    x = _rng(1).normal(scale=3.0, size=10_000)
    assert np.max(np.abs(fp.decode(fp.encode(x)) - x)) <= HALF_UNIT * (1 + 1e-12)


def test_float32_input_is_accepted_and_shape_is_preserved():
    x = _rng(2).normal(size=(3, 4, 5)).astype(np.float32)
    encoded = fp.encode(x)
    assert encoded.shape == (3, 4, 5) and fp.decode(encoded).shape == (3, 4, 5)
    assert np.max(np.abs(fp.decode(encoded) - x.astype(np.float64))) <= HALF_UNIT * (1 + 1e-12)


def test_scalar_and_empty_arrays_are_handled():
    assert fp.decode(fp.encode(np.float64(0.25))) == 0.25
    assert fp.encode(np.zeros(0)).shape == (0,)


# --------------------------------------------------------------------------
# sums modulo 2^64
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n", [3, 4, 5])
def test_sum_of_encoded_values_decodes_to_the_sum_within_n_half_units(n):
    parts = [_rng(10 + i).normal(scale=1e-2, size=5_000) for i in range(n)]
    total = fp.decode(fp.sum_encoded([fp.encode(p) for p in parts]))
    assert np.max(np.abs(total - sum(parts))) <= n * HALF_UNIT * (1 + 1e-9)


def test_sum_wraps_modulo_2_to_the_64_without_warnings():
    a = np.array([2 ** 64 - 1], dtype=np.uint64)
    b = np.array([2], dtype=np.uint64)
    with np.errstate(all="raise"):
        assert int(fp.add(a, b)[0]) == 1
        assert int(fp.sub(b, a)[0]) == 3


def test_add_and_sub_are_inverse_and_sum_of_nothing_is_refused():
    a, b = fp.encode(_rng(3).normal(size=100)), fp.encode(_rng(4).normal(size=100))
    assert np.array_equal(fp.sub(fp.add(a, b), b), a)
    with pytest.raises(ValueError):
        fp.sum_encoded([])


# --------------------------------------------------------------------------
# acceptance test 10: range assertion
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n", [3, 4, 5])
def test_an_element_just_inside_the_limit_is_accepted_and_just_outside_is_refused(n):
    limit = RANGE / n
    fp.assert_encodable(np.array([limit * (1 - 1e-9), -limit * (1 - 1e-9)]), n)
    for bad in (limit, -limit, limit * (1 + 1e-9), 2 * limit):
        with pytest.raises(fp.RangeError):
            fp.assert_encodable(np.array([0.0, bad]), n)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nan_and_infinity_are_refused_before_masking(bad):
    with pytest.raises(fp.RangeError):
        fp.assert_encodable(np.array([0.0, bad, 1.0]), 3)
    with pytest.raises(fp.RangeError):
        fp.encode(np.array([bad]))


def test_encode_refuses_a_single_value_outside_the_representable_range():
    with pytest.raises(fp.RangeError):
        fp.encode(np.array([RANGE]))
    with pytest.raises(fp.RangeError):
        fp.encode(np.array([-RANGE * 2]))
    fp.encode(np.array([-RANGE]))   # exactly -2^31 is representable


def test_the_assertion_needs_a_positive_number_of_participants():
    with pytest.raises(ValueError):
        fp.assert_encodable(np.zeros(2), 0)


def test_without_the_assertion_the_wrap_is_silent_so_the_assertion_is_not_redundant():
    values = [np.array([0.9e9])] * 3   # each is individually representable, the sum 2.7e9 is not
    encoded_sum = fp.sum_encoded([fp.encode(v) for v in values])
    wrapped = float(fp.decode(encoded_sum)[0])      # no exception, a plausible-looking number
    assert wrapped == pytest.approx(2.7e9 - 2.0 ** 32, rel=1e-9) and wrapped < 0
    with pytest.raises(fp.RangeError):               # the client-side assertion would have refused this input
        fp.assert_encodable(values[0], 3)


def test_a_deliberately_wrapped_aggregate_is_detected_and_aborts():
    wrapped = fp.sum_encoded([fp.encode(np.array([1.5e9])), fp.encode(np.array([1.5e9]))])
    assert float(fp.decode(wrapped)[0]) < 0   # silently negative
    with pytest.raises(fp.RangeError):
        fp.check_aggregate(wrapped)


def test_an_honest_aggregate_passes_the_post_check():
    honest = fp.sum_encoded([fp.encode(_rng(20 + i).normal(scale=1e-2, size=1_000)) for i in range(3)])
    fp.check_aggregate(honest)   # no exception
    assert fp.AGGREGATE_GUARD == 2.0 ** 30
    just_inside = fp.encode(np.array([fp.AGGREGATE_GUARD * (1 - 1e-6)]))
    fp.check_aggregate(just_inside)
    with pytest.raises(fp.RangeError):
        fp.check_aggregate(fp.encode(np.array([fp.AGGREGATE_GUARD])))
    with pytest.raises(fp.RangeError):
        fp.check_aggregate(fp.encode(np.array([-fp.AGGREGATE_GUARD * 1.01])))
