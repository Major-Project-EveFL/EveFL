"""Control-plane simulation: QBER trajectories from a ChannelPlan fed to the real policy classes."""

import numpy as np
import pytest
from scipy import stats

from evefl.fl.channels import ChannelPlan, LinkAttack
from evefl.fl.control_sim import (
    PolicyRun,
    client_rounds_trained,
    first_round_where,
    link_state_changes_per_round,
    rounds_discarded,
    sample_qber_trajectories,
    share_of_link_rounds,
    simulate_policy,
)
from evefl.orchestration.policy import PolicyConfig
from evefl.orchestration.state_machine import HysteresisConfig, SecurityState
from evefl.quantum.theory import lockdown_probability

S, C, L = SecurityState.SECURE, SecurityState.CAUTION, SecurityState.LOCKDOWN
LINKS = ("0", "1", "2")
KW = dict(n_qubits=1024, sample_fraction=0.25, seed=5)


def trajectories(plan, rounds=30, reps=40):
    return sample_qber_trajectories(lambda rep: plan, LINKS, rounds, reps, **KW)


def test_shape_reproducibility_and_clean_channel_is_exactly_zero():
    clean = trajectories(ChannelPlan())
    assert clean.shape == (40, 30, 3) and np.all(clean == 0.0)
    plan = ChannelPlan(attacks={"0": LinkAttack(alpha=0.5)}, default_noise=0.01)
    assert np.array_equal(trajectories(plan), trajectories(plan))


def test_the_attacked_link_has_a_higher_qber_than_the_others():
    q = trajectories(ChannelPlan(attacks={"0": LinkAttack(alpha=0.6)}), rounds=50, reps=60)
    assert q[:, :, 0].mean() == pytest.approx(0.15, abs=0.01)
    assert q[:, :, 1].mean() == 0.0 and q[:, :, 2].mean() == 0.0


def test_lockdown_share_matches_the_exact_analytic_probability():
    reps, rounds = 400, 40
    q = trajectories(ChannelPlan(attacks={"0": LinkAttack(alpha=0.5)}), rounds=rounds, reps=reps)
    hits = int(np.sum(q[:, :, 0] >= 0.11))
    exact = lockdown_probability(1024, 0.25, 0.5, 0.0, links=1)
    assert stats.binomtest(hits, reps * rounds, exact).pvalue > 1e-3


def test_intermittent_attacks_can_differ_per_rep():
    q = sample_qber_trajectories(
        lambda rep: ChannelPlan(attacks={"0": LinkAttack(alpha=0.8, kind="intermittent", probability=0.5, seed=rep)}),
        LINKS, 30, 30, **KW)
    attacked_rounds = (q[:, :, 0] > 0.05).sum(axis=1)
    assert len(set(attacked_rounds.tolist())) > 3


def test_global_policy_discards_every_attacked_round_per_client_does_not():
    q = trajectories(ChannelPlan(attacks={"0": LinkAttack(alpha=1.0)}), rounds=20, reps=10)
    glob = simulate_policy(q, LINKS, PolicyConfig(mode="global"))
    per = simulate_policy(q, LINKS, PolicyConfig(mode="per_client"))
    binary = simulate_policy(q, LINKS, PolicyConfig(mode="global_binary"))
    assert rounds_discarded(glob) == rounds_discarded(binary) == 20.0
    assert client_rounds_trained(glob) == 0.0
    assert rounds_discarded(per) == 0.0 and client_rounds_trained(per) == 2 * 20


def test_flapping_metric_counts_state_changes_per_link_round():
    def run(states):
        return PolicyRun([False] * len(states), [[]] * len(states),
                         [{cid: s for cid in LINKS} for s in states], [{}] * len(states))

    steady = [run([S, S, S, S])]
    flapping = [run([S, C, S, C])]
    assert link_state_changes_per_round(steady, LINKS) == 0.0
    assert link_state_changes_per_round(flapping, LINKS) == 1.0   # 3 changes over 3 transitions
    assert share_of_link_rounds(flapping, LINKS, C) == 0.5
    assert share_of_link_rounds(flapping, LINKS, C, slice(0, 2)) == 0.5
    assert share_of_link_rounds([], LINKS, C) == 0.0


def test_hysteresis_reduces_flapping_at_two_percent_noise():
    plan = ChannelPlan(default_noise=0.02)
    q = trajectories(plan, rounds=60, reps=60)
    plain = link_state_changes_per_round(simulate_policy(q, LINKS, PolicyConfig(mode="per_client")), LINKS)
    damped = link_state_changes_per_round(
        simulate_policy(q, LINKS, PolicyConfig(mode="per_client", hysteresis=HysteresisConfig(0.01, 3))), LINKS)
    assert plain > 0.02 and damped < plain  # about 1.6% false CAUTION per link at 2% noise


def test_recovery_takes_the_dwell_and_detection_is_not_delayed():
    plan = ChannelPlan(attacks={"0": LinkAttack(alpha=1.0, kind="window", start_round=11, end_round=20)})
    q = trajectories(plan, rounds=40, reps=20)
    for dwell in (1, 3, 5):
        runs = simulate_policy(q, LINKS, PolicyConfig(mode="per_client", hysteresis=HysteresisConfig(0.0, dwell)))
        detection = first_round_where(runs, "0", 11, lambda s: s == L)
        recovery = first_round_where(runs, "0", 21, lambda s: s == S)
        assert all(d == 0.0 for d in detection)                  # alpha = 1: LOCKDOWN in the first attacked round
        assert all(r == dwell - 1 for r in recovery)              # the link is SECURE `dwell` readings after the attack


def test_first_round_where_reports_nan_when_never():
    run = PolicyRun([False] * 3, [[]] * 3, [{"0": S}] * 3, [{}] * 3)
    assert np.isnan(first_round_where([run], "0", 1, lambda s: s == L)[0])
