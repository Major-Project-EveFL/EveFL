"""
Sample-size sweep for the QBER-based state classifier (docs/06, "Pre-registered: QBER sample-size sweep").

Everything here follows from the exact structure of the simulation (validated against the full
protocol, see validation.py and tests):

    sifted length      s ~ Binomial(n, 1/2)
    sample size        m = min(s, max(1, int(s * f)))
    sample errors      k | m ~ Binomial(m, p),   p = e + (1 - 2e) * alpha / 4
    per-link QBER      k / m,   system QBER = max over links,   state = StateController.classify

so both the exact analytic probability of each state (binomial mixture over s, independent links) and
a fast paired Monte Carlo can be computed without simulating individual qubits. The Monte Carlo draws s
and a uniform U per (round, link) once per block size n and reuses them across alpha and e (k = the
Binomial(m, p) quantile of U), so scenarios are paired. `check_against_protocol` verifies the analytic
numbers against the full `bb84_numpy` protocol.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
from scipy import stats

from evefl.orchestration.state_machine import StateThresholds
from evefl.quantum.base import ChannelModel


def error_probability(alpha: float, e: float) -> float:
    return ChannelModel(alpha, e).expected_qber


def sample_size(s: np.ndarray, f: float) -> np.ndarray:
    """m for a sifted length s (mirrors bb84_numpy: min(s, max(1, int(s*f))), 0 if s == 0)."""
    m = np.minimum(s, np.maximum(1, np.floor(s * f).astype(np.int64)))
    return np.where(s > 0, m, 0)


def min_errors_for(m: np.ndarray, threshold: float) -> np.ndarray:
    """Smallest k with k/m >= threshold, using the same float comparison as the controller."""
    k = np.ceil(threshold * m).astype(np.int64)
    k = np.where((k - 1 >= 0) & ((k - 1) / np.maximum(m, 1) >= threshold), k - 1, k)
    k = np.where(k / np.maximum(m, 1) < threshold, k + 1, k)
    return k


def analytic_single_link(n: int, f: float, p: float, thresholds: StateThresholds) -> Dict[str, float]:
    """Exact per-link probabilities of leaving SECURE / reaching LOCKDOWN (mixture over the sifted length)."""
    s_lo, s_hi = (int(stats.binom.ppf(1e-13, n, 0.5)), int(stats.binom.isf(1e-13, n, 0.5)))
    s = np.arange(s_lo, s_hi + 1)
    w = stats.binom.pmf(s, n, 0.5)  # the tails cut at 1e-13 each lose a negligible probability mass
    m = sample_size(s, f)

    def prob_at_least(threshold: float) -> float:
        k_min = min_errors_for(m, threshold)
        # An empty sample (m = 0) gives QBER 0 in the protocol: never an alarm.
        return float(np.sum(w * np.where(m > 0, stats.binom.sf(k_min - 1, m, p), 0.0)))

    return {
        "p_not_secure": prob_at_least(thresholds.secure_max),
        "p_lockdown": prob_at_least(thresholds.caution_max),
        "expected_sample": float(np.sum(w * m)),
        "expected_leftover_per_qubit": float(np.sum(w * (s - m)) / n),
    }


def system_probability(p_link: float, links: int) -> float:
    """P(at least one of `links` independent links reaches the state)."""
    return 1.0 - (1.0 - p_link) ** links


def wilson_interval(successes: int, trials: int, z: float = 1.959964) -> Tuple[float, float]:
    if trials == 0:
        return (float("nan"), float("nan"))
    phat = successes / trials
    denom = 1 + z * z / trials
    centre = (phat + z * z / (2 * trials)) / denom
    half = z * math.sqrt(phat * (1 - phat) / trials + z * z / (4 * trials * trials)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass
class PairedDraws:
    """Random draws shared by every (f, alpha, e) cell of one block size n."""

    sifted: np.ndarray   # [rounds, links]
    uniforms: np.ndarray  # [rounds, links]

    @classmethod
    def draw(cls, n: int, rounds: int, links: int, seed: int) -> "PairedDraws":
        rng = np.random.default_rng(np.random.SeedSequence([seed, n]))
        sifted = rng.binomial(n, 0.5, size=(rounds, links))
        uniforms = np.clip(rng.random((rounds, links)), 1e-12, 1 - 1e-12)
        return cls(sifted, uniforms)


def simulate_system_rates(draws: PairedDraws, f: float, p: float, thresholds: StateThresholds) -> Dict[str, Any]:
    """Monte Carlo rates for the system state (max QBER over links) from paired draws."""
    m = sample_size(draws.sifted, f)
    k = np.where(m > 0, stats.binom.ppf(draws.uniforms, m, p), 0).astype(np.int64)
    qber = np.where(m > 0, k / np.maximum(m, 1), 0.0)
    system_qber = qber.max(axis=1)
    rounds = system_qber.shape[0]
    not_secure = int(np.sum(~(system_qber < thresholds.secure_max)))
    lockdown = int(np.sum(~(system_qber < thresholds.caution_max)))
    lo, hi = wilson_interval(not_secure, rounds)
    return {
        "p_not_secure": not_secure / rounds,
        "p_not_secure_ci95": [lo, hi],
        "p_lockdown": lockdown / rounds,
        "p_lockdown_ci95": list(wilson_interval(lockdown, rounds)),
        "rounds": rounds,
    }


def check_against_protocol(
    n: int, f: float, e: float, alpha: float, rounds: int, links: int, seed: int, thresholds: StateThresholds
) -> Dict[str, float]:
    """Run the FULL bb84_numpy protocol and compare the system-level rates with the exact analytic values."""
    from evefl.quantum.bb84_numpy import BB84NumpyProtocol

    channel = ChannelModel(alpha, e)
    not_secure = lockdown = 0
    for r in range(1, rounds + 1):
        system_qber = max(
            BB84NumpyProtocol.for_round(seed, r, link, sample_fraction=f).run_exchange(n, channel=channel).qber
            for link in range(links)
        )
        not_secure += not (system_qber < thresholds.secure_max)
        lockdown += not (system_qber < thresholds.caution_max)
    single = analytic_single_link(n, f, channel.expected_qber, thresholds)
    expected_ns = system_probability(single["p_not_secure"], links)
    expected_ld = system_probability(single["p_lockdown"], links)
    return {
        "n_qubits": n, "sample_fraction": f, "bit_flip": e, "alpha": alpha, "rounds": rounds,
        "protocol_p_not_secure": not_secure / rounds, "analytic_p_not_secure": expected_ns,
        "pvalue_not_secure": float(stats.binomtest(int(not_secure), rounds, min(max(expected_ns, 0.0), 1.0)).pvalue),
        "protocol_p_lockdown": lockdown / rounds, "analytic_p_lockdown": expected_ld,
        "pvalue_lockdown": float(stats.binomtest(int(lockdown), rounds, min(max(expected_ld, 0.0), 1.0)).pvalue),
    }
