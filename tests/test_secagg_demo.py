"""
The secure-aggregation demonstration (`evefl/secagg_demo.py`, `dashboard/pages/6_secure_aggregation.py`).

It runs ONLY the step-2a primitives (fixed-point codec and mask PRG) in one process on synthetic updates. The pairwise seeds are fresh
random bytes standing in for the key agreement of step 2b, which is not implemented. These tests pin what the page may show: masked
updates look random, the masked sum equals the plain weighted average, a missing hospital breaks the sum, a colluding server learns
exactly what ADR 0003 says it learns, and the page says on screen that it is a simulated library not wired into the FL path.
"""

import itertools
from pathlib import Path

import numpy as np
import pytest

from evefl import secagg_demo as demo
from evefl.crypto.secagg import fixedpoint as fp

REPO = Path(__file__).resolve().parents[1]
HALF_UNIT = 2.0 ** -33


def _seeds(n):
    return {frozenset((str(i), str(j))): bytes([i * 16 + j + 1]) * 32 for i, j in itertools.combinations(range(n), 2)}


def _run(n=3, **kwargs):
    base = dict(n_clients=n, n_params=64, seed=0, pair_seeds=_seeds(n))
    base.update(kwargs)
    return demo.run_demo(demo.DemoSettings(**base))


# --------------------------------------------------------------------------
# the masked sum is the plain weighted average
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n", [3, 4, 5])
def test_masked_sum_equals_the_plain_weighted_average_within_the_quantisation_tolerance(n):
    result = _run(n)
    assert result.max_abs_error <= n * HALF_UNIT * (1 + 1e-9) and result.tolerance == pytest.approx(n * HALF_UNIT)
    plain = (result.weights_normalised[:, None] * result.updates).sum(axis=0)
    assert np.allclose(result.aggregate_plain, plain) and np.max(np.abs(result.aggregate_secure - plain)) <= result.tolerance


def test_weights_come_from_the_registry_sizes_and_are_normalised_over_the_participants():
    result = _run(3)
    assert result.registry_weights.tolist() == [46975.0, 18018.0, 25136.0]   # the audited partition sizes
    assert result.weights_normalised.sum() == pytest.approx(1.0)
    assert result.weights_normalised[0] == pytest.approx(46975 / (46975 + 18018 + 25136))


# --------------------------------------------------------------------------
# what the server sees
# --------------------------------------------------------------------------

def test_the_masked_updates_the_server_sees_look_uniform_and_unrelated_to_the_updates():
    result = _run(3, n_params=4000)
    bits = np.unpackbits(result.masked.view(np.uint8).reshape(3, -1, 8), axis=-1)   # every word of every client
    balance = bits.reshape(3, -1, 64).mean(axis=1)
    assert balance.min() > 0.45 and balance.max() < 0.55
    shifted = fp.decode(result.masked)
    for i in range(3):
        assert abs(np.corrcoef(shifted[i], result.weighted[i])[0, 1]) < 0.1
    assert not np.any(result.masked == fp.encode(result.weighted))


def test_fresh_random_seeds_by_default_and_reproducible_masks_when_seeds_are_given():
    a = demo.run_demo(demo.DemoSettings(n_clients=3, n_params=32, seed=1))
    b = demo.run_demo(demo.DemoSettings(n_clients=3, n_params=32, seed=1))
    assert np.array_equal(a.updates, b.updates) and not np.array_equal(a.masked, b.masked)   # same data, new masks
    assert np.array_equal(a.aggregate_secure, b.aggregate_secure) or a.max_abs_error <= a.tolerance   # the sum is the same either way
    c, d = _run(3), _run(3)
    assert np.array_equal(c.masked, d.masked)


def test_the_demo_takes_its_mask_seeds_from_the_operating_system_not_from_numpy():
    source = Path(demo.__file__).read_text(encoding="utf-8")
    assert "os.urandom" in source
    assert "Generator" not in source.split("def _fresh_pair_seeds")[1].split("def ")[0]


# --------------------------------------------------------------------------
# a missing hospital, a too-large update
# --------------------------------------------------------------------------

def test_without_a_participant_the_leftover_masks_make_the_sum_meaningless_and_the_round_aborts():
    result = _run(3, n_params=200, drop_client="1")
    assert result.dropout is not None and result.dropout["round_aborts"] is True
    assert result.dropout["reason"] == "secagg_dropout"
    assert result.dropout["garbage_mean_abs"] > 1e6   # random 64-bit words decode to values of the order of 2^31, not 1e-2
    assert result.dropout["garbage_mean_abs"] > 1e4 * np.mean(np.abs(result.aggregate_plain))


def test_an_update_too_large_to_sum_without_wrapping_is_refused_before_masking():
    with pytest.raises(fp.RangeError):
        _run(3, update_scale=1e9)


# --------------------------------------------------------------------------
# collusion (ADR 0003: the privacy floor with three clients)
# --------------------------------------------------------------------------

def test_a_server_with_one_colluding_hospital_learns_the_sum_of_the_others_but_not_either_alone():
    result = _run(3, colluding=("2",))
    c = result.collusion
    assert c["learns_sum_of"] == ["0", "1"] and c["individual_revealed"] is None
    assert c["sum_error"] <= 3 * HALF_UNIT * (1 + 1e-9)                        # the sum of the other two is exact
    scale = float(np.mean(np.abs(result.weighted[0])))
    assert c["single_guess_error_hospital_0"] > 1e4 * scale                   # a guess for one of them alone is garbage


def test_two_colluding_hospitals_reveal_the_third_exactly():
    result = _run(3, colluding=("0", "1"))
    c = result.collusion
    assert c["learns_sum_of"] == ["2"] and c["individual_revealed"] == "2"
    assert c["sum_error"] <= 3 * HALF_UNIT * (1 + 1e-9)


def test_with_five_hospitals_one_colluder_still_learns_only_the_sum_of_the_other_four():
    c = _run(5, colluding=("4",)).collusion
    assert c["learns_sum_of"] == ["0", "1", "2", "3"] and c["individual_revealed"] is None


def test_no_collusion_means_the_server_alone_learns_only_the_total():
    assert _run(3).collusion is None
    with pytest.raises(ValueError):
        _run(3, colluding=("0", "1", "2"))      # everybody colluding leaves nothing to learn
    with pytest.raises(ValueError):
        _run(3, colluding=("9",))


# --------------------------------------------------------------------------
# validation and the on-screen honesty label
# --------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [{"n_clients": 2}, {"n_clients": 6}, {"n_params": 0}, {"update_scale": 0.0}, {"drop_client": "9"}])
def test_invalid_settings_are_refused(bad):
    with pytest.raises(ValueError):
        demo.run_demo(demo.DemoSettings(**{"n_clients": 3, "n_params": 8, "pair_seeds": _seeds(3), **bad}))


def test_the_label_says_simulated_library_not_wired_into_fl_and_names_the_missing_steps():
    label = _run(3).label
    for phrase in ("SIMULATED", "not wired into the FL path", "step 2b", "step 2a", "not secret"):
        assert phrase in label, phrase
    assert "production" not in label.lower() and "certified" not in label.lower()


def test_the_page_renders_with_the_label_and_survives_a_change_of_clients():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(REPO / "dashboard" / "pages" / "6_secure_aggregation.py"), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    shown = " ".join(str(x.value) for x in list(at.warning) + list(at.info) + list(at.caption) + list(at.markdown))
    assert "SIMULATED" in shown and "not wired into the FL path" in shown
    clients = next(s for s in at.slider if "hospitals" in s.label.lower())
    clients.set_value(4).run()
    assert not at.exception, [e.value for e in at.exception]
