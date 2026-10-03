"""
Tests for the exact detection probabilities (restated Theorem 1) and the integer-boundary effect.
"""

import numpy as np
import pytest
from scipy import stats

from evefl.orchestration.state_machine import SecurityState, StateController, StateThresholds
from evefl.quantum.base import ChannelModel
from evefl.quantum.bb84_numpy import BB84NumpyProtocol
from evefl.quantum.sweep import sample_size
from evefl.quantum.theory import (
    detection_probability,
    effective_boundary,
    link_probabilities,
    lockdown_probability,
    minimal_alpha,
)

N, F = 1024, 0.25  # headline setting


# --------------------------------------------------------------------------
# Integer boundary effect at the boundaries the controller really uses
# --------------------------------------------------------------------------

def test_m128_lockdown_boundary_is_15_errors_about_11_7_percent():
    k, qber = effective_boundary(128, 0.11)
    assert k == 15 and qber == pytest.approx(0.1171875)
    controller = StateController()
    assert controller.classify(14 / 128) == SecurityState.CAUTION   # 10.94% is still CAUTION
    assert controller.classify(15 / 128) == SecurityState.LOCKDOWN  # 11.72% is LOCKDOWN


def test_m128_caution_boundary_is_7_errors_about_5_5_percent():
    k, qber = effective_boundary(128, 0.05)
    assert k == 7 and qber == pytest.approx(0.0546875)
    controller = StateController()
    assert controller.classify(6 / 128) == SecurityState.SECURE
    assert controller.classify(7 / 128) == SecurityState.CAUTION


@pytest.mark.parametrize("m", [100, 101, 127, 128, 129, 200, 256, 500, 1000])
def test_effective_boundary_is_the_first_crossing_for_any_sample_size(m):
    for threshold in (0.05, 0.11):
        k, qber = effective_boundary(m, threshold)
        assert qber >= threshold and (k - 1) / m < threshold


def test_effective_boundary_rejects_empty_sample():
    with pytest.raises(ValueError):
        effective_boundary(0, 0.11)


# --------------------------------------------------------------------------
# Probabilities: structure
# --------------------------------------------------------------------------

def test_no_eve_no_noise_never_alarms_exactly():
    assert detection_probability(N, F, 0.0, 0.0, links=3) == 0.0
    assert lockdown_probability(N, F, 0.0, 0.0, links=3) == 0.0


def test_probabilities_are_monotone_in_alpha_noise_and_links():
    alphas = np.linspace(0, 1, 21)
    for e in (0.0, 0.01):
        series = [lockdown_probability(N, F, a, e, links=1) for a in alphas]
        assert series == sorted(series)
    assert lockdown_probability(N, F, 0.5, 0.01) > lockdown_probability(N, F, 0.5, 0.0)
    assert lockdown_probability(N, F, 0.6, 0.0, links=3) > lockdown_probability(N, F, 0.6, 0.0, links=1)


def test_link_probabilities_lockdown_is_a_subset_of_not_secure():
    for a in (0.2, 0.4, 0.66):
        not_secure, lockdown = link_probabilities(N, F, a, 0.0)
        assert 0.0 <= lockdown <= not_secure <= 1.0


def test_old_theorem_claim_alpha_066_per_link_is_marginal_but_system_level_holds():
    """The paper's Theorem 1 claimed P(LOCKDOWN) > 0.95 for alpha >= 0.66. With the random 128-bit sample
    that is marginal for ONE link and comfortably true for the max over 3 links."""
    per_link = lockdown_probability(N, F, 0.66, 0.0, links=1)
    system = lockdown_probability(N, F, 0.66, 0.0, links=3)
    assert 0.90 < per_link < 0.97
    assert system > 0.999


# --------------------------------------------------------------------------
# Probabilities: exact vs simulation of the full protocol (averages over the random sample size)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("alpha,e", [(0.3, 0.0), (0.5, 0.0), (0.66, 0.0), (0.4, 0.01)])
def test_exact_lockdown_probability_matches_full_protocol_simulation(alpha, e):
    trials = 4000
    channel = ChannelModel(alpha, e)
    hits = sum(
        BB84NumpyProtocol.for_round(5, t, "0", sample_fraction=F).run_exchange(N, channel=channel).qber >= 0.11
        for t in range(trials)
    )
    exact = lockdown_probability(N, F, alpha, e, links=1)
    assert stats.binomtest(hits, trials, exact).pvalue > 1e-3, (hits / trials, exact)


def test_mixture_over_sample_size_differs_from_a_fixed_m_formula():
    """Averaging over the random m matters: it is not the same as plugging m = n*f/2 in."""
    alpha, p = 0.66, ChannelModel(0.66, 0.0).expected_qber
    fixed = float(stats.binom.sf(14, 128, p))  # P(K >= 15) with m fixed at 128
    mixture = lockdown_probability(N, F, alpha, 0.0, links=1)
    assert abs(fixed - mixture) > 1e-4
    # and the expected sample size really is about 128 with spread
    s = np.arange(0, N + 1)
    m = sample_size(s, F)
    assert 120 < float(np.sum(stats.binom.pmf(s, N, 0.5) * m)) < 135


# --------------------------------------------------------------------------
# minimal_alpha
# --------------------------------------------------------------------------

def test_minimal_alpha_reaches_the_target_and_is_minimal():
    for links in (1, 3):
        a = minimal_alpha(0.95, N, F, 0.0, links=links)
        assert a is not None
        assert lockdown_probability(N, F, a, 0.0, links) >= 0.95
        assert lockdown_probability(N, F, a - 2e-4, 0.0, links) < 0.95


def test_minimal_alpha_per_link_is_larger_than_for_three_links():
    assert minimal_alpha(0.95, N, F, 0.0, links=1) > minimal_alpha(0.95, N, F, 0.0, links=3)


def test_noise_lowers_the_alpha_needed():
    assert minimal_alpha(0.95, N, F, 0.01, links=1) < minimal_alpha(0.95, N, F, 0.0, links=1)


def test_minimal_alpha_edge_cases():
    assert minimal_alpha(0.5, N, F, 0.5, links=1, kind="lockdown") == 0.0  # a fully noisy channel always alarms
    assert minimal_alpha(0.95, 64, 0.25, 0.0, links=1) is None  # tiny sample can never reach 0.95
    with pytest.raises(ValueError):
        minimal_alpha(0.95, N, F, 0.0, kind="nope")


def test_detection_alpha_is_far_below_lockdown_alpha():
    assert minimal_alpha(0.95, N, F, 0.01, links=3, kind="detection") < 0.25 < \
        minimal_alpha(0.95, N, F, 0.01, links=3, kind="lockdown")


def test_default_thresholds_used_when_none_given():
    assert link_probabilities(N, F, 0.3, 0.0) == link_probabilities(N, F, 0.3, 0.0, StateThresholds())
