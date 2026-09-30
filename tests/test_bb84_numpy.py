"""
Tests for the channel model, the seeding scheme and the exact numpy BB84 backend.

Statistical checks are conditional on the recorded sample size m (the sifted length, hence m,
is random). Seeds are fixed, so the tests are deterministic; the significance level is set so
a correct implementation passes with large margin.
"""

import numpy as np
import pytest

from evefl.quantum.base import ChannelModel, resolve_channel
from evefl.quantum.bb84_numpy import BB84NumpyProtocol
from evefl.quantum.factory import available_backends, create_protocol
from evefl.quantum.seeding import client_id_to_int, qkd_seed_sequence, stream_generators
from evefl.quantum.validation import analytic_check, run_trials

ALPHA = 1e-3  # a correct backend passes each check with probability 1 - ALPHA


# --------------------------------------------------------------------------
# ChannelModel: the analytic formula and edge cases
# --------------------------------------------------------------------------

@pytest.mark.parametrize("alpha,e,expected", [
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.25),
    (0.44, 0.0, 0.11),
    (0.0, 0.03, 0.03),
    (1.0, 0.02, 0.02 + 0.96 * 0.25),
    (0.5, 0.01, 0.01 + 0.98 * 0.125),
])
def test_expected_qber_formula(alpha, e, expected):
    assert ChannelModel(alpha, e).expected_qber == pytest.approx(expected)


@pytest.mark.parametrize("alpha", [0.0, 0.3, 1.0])
def test_fully_noisy_channel_gives_half_whatever_eve_does(alpha):
    assert ChannelModel(alpha, 0.5).expected_qber == pytest.approx(0.5)


@pytest.mark.parametrize("kwargs", [
    {"intercept_probability": -0.1}, {"intercept_probability": 1.1},
    {"bit_flip_probability": -0.01}, {"bit_flip_probability": 0.51},
])
def test_channel_model_validates_probabilities(kwargs):
    with pytest.raises(ValueError):
        ChannelModel(**kwargs)


def test_resolve_channel_legacy_argument_and_conflict():
    assert resolve_channel(0.3, None) == ChannelModel(0.3, 0.0)
    assert resolve_channel(0.0, ChannelModel(0.2, 0.01)) == ChannelModel(0.2, 0.01)
    with pytest.raises(ValueError):
        resolve_channel(0.3, ChannelModel(0.2, 0.0))


# --------------------------------------------------------------------------
# Numpy backend vs the analytic prediction (conditional on recorded m)
# --------------------------------------------------------------------------

def _numpy_counts(channel, n_qubits=1024, n_trials=300, sample_fraction=0.25):
    return run_trials(
        lambda t: BB84NumpyProtocol.for_round(experiment_seed=11, server_round=t, cid="0",
                                              sample_fraction=sample_fraction),
        channel, n_qubits, n_trials,
    )


@pytest.mark.parametrize("alpha,e", [
    (0.0, 0.0), (1.0, 0.0), (0.5, 0.0), (0.44, 0.0),
    (0.0, 0.02), (0.5, 0.02), (1.0, 0.03), (0.3, 0.05), (0.0, 0.5), (1.0, 0.5),
])
def test_numpy_backend_matches_analytic_qber(alpha, e):
    channel = ChannelModel(alpha, e)
    check = analytic_check(_numpy_counts(channel), channel, confidence=0.999)
    assert check["qber_ci_low"] <= check["expected_qber"] <= check["qber_ci_high"], check
    assert check["qber_pvalue"] > ALPHA, check
    assert check["true_qber_pvalue"] > ALPHA, check  # ground truth over the whole sifted key
    assert check["sifted_pvalue"] > ALPHA, check  # sifted length ~ Binomial(n, 1/2)
    if 0.0 < channel.expected_qber < 1.0:
        assert check["dispersion_pvalue"] > ALPHA, check  # per-trial spread is binomial given m


def test_no_eve_no_noise_gives_exactly_zero_errors():
    for trial in range(20):
        result = BB84NumpyProtocol(seed=trial).run_exchange(1024, channel=ChannelModel(0.0, 0.0))
        assert result.qber == 0.0
        assert result.metadata["sim_only_true_error_count"] == 0
        assert result.metadata["qber_sample_errors"] == 0
        assert result.eavesdropper_active is False


def test_sample_bookkeeping_is_consistent():
    result = BB84NumpyProtocol(sample_fraction=0.25, seed=3).run_exchange(1000, channel=ChannelModel(0.7, 0.01))
    md = result.metadata
    assert md["qber_sample_size"] == int(result.n_sifted * 0.25)
    assert result.qber == pytest.approx(md["qber_sample_errors"] / md["qber_sample_size"])
    assert 0 <= md["qber_sample_errors"] <= md["qber_sample_size"]
    assert md["sim_only_true_error_count"] <= result.n_sifted
    assert len(result.sifted_key) == result.n_sifted
    assert md["backend"] == "bb84_numpy" and md["expected_qber"] == pytest.approx(0.01 + 0.98 * 0.7 / 4)


def test_true_qber_is_recorded_separately_from_the_estimate():
    """The controller sees `qber` (a sample estimate); the sim-only truth differs in general."""
    estimates, truths = [], []
    for t in range(50):
        r = BB84NumpyProtocol(seed=t).run_exchange(512, channel=ChannelModel(0.6, 0.0))
        estimates.append(r.qber)
        truths.append(r.metadata["sim_only_true_qber"])
    assert estimates != truths
    assert abs(np.mean(estimates) - np.mean(truths)) < 0.02  # unbiased on average


# --------------------------------------------------------------------------
# Streams: paired scenarios and seed dependence
# --------------------------------------------------------------------------

def test_changing_alpha_or_noise_does_not_change_alices_key_or_sifting():
    """Separate streams -> scenarios that differ only in the channel are paired comparisons."""
    base = BB84NumpyProtocol(seed=5).run_exchange(2048, channel=ChannelModel(0.0, 0.0))
    for channel in (ChannelModel(0.5, 0.0), ChannelModel(1.0, 0.03), ChannelModel(0.0, 0.05)):
        other = BB84NumpyProtocol(seed=5).run_exchange(2048, channel=channel)
        assert other.sifted_key == base.sifted_key
        assert other.n_sifted == base.n_sifted


def test_errors_grow_monotonically_with_alpha_for_the_same_seed():
    """With shared randomness, more interception can only add errors (pairing in action)."""
    counts = [
        BB84NumpyProtocol(seed=9).run_exchange(4096, channel=ChannelModel(a, 0.0)).metadata["sim_only_true_error_count"]
        for a in (0.0, 0.25, 0.5, 0.75, 1.0)
    ]
    assert counts == sorted(counts) and counts[0] == 0


def test_same_seed_is_reproducible():
    a = BB84NumpyProtocol(seed=1).run_exchange(512, channel=ChannelModel(0.4, 0.01))
    b = BB84NumpyProtocol(seed=1).run_exchange(512, channel=ChannelModel(0.4, 0.01))
    assert a.qber == b.qber and a.sifted_key == b.sifted_key and a.metadata == b.metadata


def test_repeated_calls_on_one_instance_are_independent():
    p = BB84NumpyProtocol(seed=1)
    a = p.run_exchange(512, channel=ChannelModel(0.4, 0.0))
    b = p.run_exchange(512, channel=ChannelModel(0.4, 0.0))
    assert a.sifted_key != b.sifted_key


def test_different_experiment_seeds_give_different_trajectories():
    """Regression: seeds 0..4 used to give IDENTICAL QBER trajectories (hash of round+client only)."""
    channel = ChannelModel(0.5, 0.0)
    trajectories = [
        tuple(BB84NumpyProtocol.for_round(seed, r, "0").run_exchange(1024, channel=channel).qber for r in range(1, 9))
        for seed in range(5)
    ]
    assert len(set(trajectories)) == 5


def test_round_and_client_change_the_streams():
    channel = ChannelModel(0.5, 0.0)
    keys = {
        (r, c): BB84NumpyProtocol.for_round(0, r, c).run_exchange(256, channel=channel).sifted_key
        for r in (1, 2) for c in ("0", "1")
    }
    assert len({tuple(v) for v in keys.values()}) == 4


def test_streams_are_independent_generators():
    gens = stream_generators(qkd_seed_sequence(0, 1, "0"))
    draws = [tuple(g.integers(0, 2**31, 4)) for g in gens.values()]
    assert len(set(draws)) == len(draws)


def test_client_id_mapping_is_stable():
    assert client_id_to_int("7") == 7 and client_id_to_int(7) == 7
    assert client_id_to_int("hospital_a") == client_id_to_int("hospital_a")


# --------------------------------------------------------------------------
# Interface: legacy argument, validation, registry
# --------------------------------------------------------------------------

def test_legacy_intercept_argument_equals_channel():
    a = BB84NumpyProtocol(seed=2).run_exchange(512, intercept_probability=0.6)
    b = BB84NumpyProtocol(seed=2).run_exchange(512, channel=ChannelModel(0.6, 0.0))
    assert a.qber == b.qber and a.intercept_probability == 0.6


def test_passing_both_legacy_and_channel_is_rejected():
    with pytest.raises(ValueError):
        BB84NumpyProtocol(seed=2).run_exchange(512, intercept_probability=0.6, channel=ChannelModel(0.2))


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_invalid_intercept_probability_raises(bad):
    with pytest.raises(ValueError):
        BB84NumpyProtocol(seed=7).run_exchange(50, intercept_probability=bad)


def test_invalid_sizes_raise():
    with pytest.raises(ValueError):
        BB84NumpyProtocol(seed=1).run_exchange(0)
    with pytest.raises(ValueError):
        BB84NumpyProtocol(sample_fraction=0.0)


def test_both_backends_are_registered_and_created_by_key():
    assert {"bb84", "bb84_numpy"} <= set(available_backends())
    seq = qkd_seed_sequence(0, 1, "0")
    assert create_protocol("bb84_numpy", sample_fraction=0.25, seed_sequence=seq).name == "bb84_numpy"
    assert create_protocol("bb84", sample_fraction=0.25, seed_sequence=seq).name == "bb84"
    with pytest.raises(KeyError):
        create_protocol("nope", sample_fraction=0.25, seed_sequence=seq)
