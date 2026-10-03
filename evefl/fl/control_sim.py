"""
Control-plane simulation: how the round policy behaves over many rounds, without training any model.

It samples per-link QBER trajectories from a `ChannelPlan` with the EXACT count-level model validated in Phase 1
(sifted length ~ Binomial(n, 1/2); given the sample size m, sample errors ~ Binomial(m, p)) and feeds them to the REAL
policy classes (`evefl.orchestration.policy`), so what is measured is the code that ships. It is used for the
hysteresis sweep and for the effective-training-rounds comparison of the three policy modes (docs/06).

All numbers it produces are SIMULATED control-plane statistics: they count rounds and client-rounds, not accuracy.
Excluding a hospital under non-IID data has a model-quality cost that only the Phase 5 training experiments measure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

import numpy as np
from scipy import stats

from evefl.fl.channels import ChannelPlan
from evefl.orchestration.policy import ClientAction, PolicyConfig, create_policy
from evefl.orchestration.state_machine import SecurityState
from evefl.quantum.sweep import sample_size


def sample_qber_trajectories(
    plan_for_rep: Callable[[int], ChannelPlan],
    links: Sequence[str],
    rounds: int,
    reps: int,
    *,
    n_qubits: int,
    sample_fraction: float,
    seed: int,
) -> np.ndarray:
    """QBER estimates, shape [reps, rounds, links]. `plan_for_rep(rep)` lets intermittent attacks differ per rep."""
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(n_qubits), 77]))
    sifted = rng.binomial(n_qubits, 0.5, size=(reps, rounds, len(links)))
    uniforms = np.clip(rng.random((reps, rounds, len(links))), 1e-12, 1 - 1e-12)
    sizes = sample_size(sifted, sample_fraction)

    error_probability = np.zeros((reps, rounds, len(links)))
    for rep in range(reps):
        plan = plan_for_rep(rep)
        for t in range(rounds):
            for j, cid in enumerate(links):
                error_probability[rep, t, j] = plan.channel_for(cid, t + 1).expected_qber

    errors = np.where(sizes > 0, stats.binom.ppf(uniforms, sizes, error_probability), 0)
    return np.where(sizes > 0, errors / np.maximum(sizes, 1), 0.0)


@dataclass
class PolicyRun:
    """Per-round record of one simulated experiment (one rep)."""

    discarded: List[bool]
    included: List[List[str]]
    link_states: List[Dict[str, SecurityState]]
    reasons: List[Dict[str, str]]


def simulate_policy(qbers: np.ndarray, links: Sequence[str], config: PolicyConfig) -> List[PolicyRun]:
    """Run a fresh policy over each rep's trajectory."""
    runs: List[PolicyRun] = []
    for rep in range(qbers.shape[0]):
        policy = create_policy(config)
        run = PolicyRun([], [], [], [])
        for t in range(qbers.shape[1]):
            decision = policy.decide({cid: float(qbers[rep, t, j]) for j, cid in enumerate(links)})
            run.discarded.append(decision.discard_round)
            run.included.append(decision.included)
            run.link_states.append({cid: d.state for cid, d in decision.clients.items()})
            run.reasons.append({cid: d.reason for cid, d in decision.clients.items()
                                if d.action == ClientAction.EXCLUDE})
        runs.append(run)
    return runs


def link_state_changes_per_round(runs: Sequence[PolicyRun], links: Sequence[str]) -> float:
    """Mean number of state changes per link per round (flapping)."""
    changes = total = 0
    for run in runs:
        for cid in links:
            states = [s[cid] for s in run.link_states]
            changes += sum(1 for a, b in zip(states, states[1:]) if a != b)
            total += max(0, len(states) - 1)
    return changes / total if total else 0.0


def share_of_link_rounds(runs: Sequence[PolicyRun], links: Sequence[str], state: SecurityState,
                         rounds: slice = slice(None)) -> float:
    """Share of (link, round) pairs in `state`, over the selected rounds."""
    hits = total = 0
    for run in runs:
        for s in run.link_states[rounds]:
            for cid in links:
                hits += s[cid] == state
                total += 1
    return hits / total if total else 0.0


def client_rounds_trained(runs: Sequence[PolicyRun]) -> float:
    """Mean over reps of the total number of client-rounds that were included (trained and aggregated)."""
    return float(np.mean([sum(len(inc) for inc in run.included) for run in runs]))


def rounds_discarded(runs: Sequence[PolicyRun]) -> float:
    return float(np.mean([sum(run.discarded) for run in runs]))


def first_round_where(runs: Sequence[PolicyRun], cid: str, start_round: int,
                      predicate: Callable[[SecurityState], bool]) -> List[float]:
    """For each rep, the number of rounds from `start_round` (1-based) until link `cid` first satisfies `predicate`
    (0 = at start_round itself); NaN if it never does."""
    out: List[float] = []
    for run in runs:
        found = float("nan")
        for t in range(start_round - 1, len(run.link_states)):
            if predicate(run.link_states[t][cid]):
                found = float(t - (start_round - 1))
                break
        out.append(found)
    return out
