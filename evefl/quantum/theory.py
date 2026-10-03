"""
Exact detection probabilities for the QBER-threshold classifier (restated Theorem 1).

The QBER the controller sees is K / M, where the sample size M is RANDOM (it follows the random sifted
length S ~ Binomial(n, 1/2), so M = m(S) ~ n * f / 2) and, given M, K ~ Binomial(M, p) with
p = e + (1 - 2e) * alpha / 4. The probability that one link reaches a state is therefore a mixture over
the distribution of M, not a fixed-m formula, and the threshold is an integer effect: with m sampled
bits LOCKDOWN needs k >= ceil(0.11 * m) errors (m = 128: 15 errors, an effective boundary of 11.72%).
For N independent links the system state (max QBER) is reached with probability 1 - (1 - P_link)^N.

Results from this module are EXACT analytic values (labelled "analytic"); the simulation in
scripts/table3_theorem1.py checks them.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from evefl.orchestration.state_machine import StateThresholds
from evefl.quantum.base import ChannelModel
from evefl.quantum.sweep import analytic_single_link, min_errors_for, system_probability


def link_probabilities(
    n: int, f: float, alpha: float, e: float, thresholds: StateThresholds | None = None
) -> Tuple[float, float]:
    """(P(not SECURE), P(LOCKDOWN)) for ONE link, averaged over the random sample size."""
    thresholds = thresholds or StateThresholds()
    single = analytic_single_link(n, f, ChannelModel(alpha, e).expected_qber, thresholds)
    return single["p_not_secure"], single["p_lockdown"]


def lockdown_probability(
    n: int, f: float, alpha: float, e: float, links: int = 1, thresholds: StateThresholds | None = None
) -> float:
    """P(LOCKDOWN) for the system state (max over `links` independent links)."""
    return system_probability(link_probabilities(n, f, alpha, e, thresholds)[1], links)


def detection_probability(
    n: int, f: float, alpha: float, e: float, links: int = 1, thresholds: StateThresholds | None = None
) -> float:
    """P(state != SECURE) for the system state (max over `links` independent links)."""
    return system_probability(link_probabilities(n, f, alpha, e, thresholds)[0], links)


def minimal_alpha(
    target: float,
    n: int,
    f: float,
    e: float,
    links: int = 1,
    kind: str = "lockdown",
    thresholds: StateThresholds | None = None,
    tol: float = 1e-4,
) -> Optional[float]:
    """Smallest alpha in [0, 1] whose `kind` ('lockdown' | 'detection') probability is >= `target`.

    The probability is nondecreasing in alpha, so bisection is exact up to `tol`. Returns None if
    even alpha = 1 does not reach the target.
    """
    if kind not in ("lockdown", "detection"):
        raise ValueError(f"kind must be 'lockdown' or 'detection', got {kind!r}")
    fn = lockdown_probability if kind == "lockdown" else detection_probability

    def prob(a: float) -> float:
        return fn(n, f, a, e, links, thresholds)

    if prob(1.0) < target:
        return None
    if prob(0.0) >= target:
        return 0.0
    lo, hi = 0.0, 1.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if prob(mid) >= target else (mid, hi)
    return hi


def effective_boundary(m: int, threshold: float) -> Tuple[int, float]:
    """(k_min, k_min / m): the fewest sample errors that cross `threshold`, and the QBER that implies.

    The integer effect: with m = 128 the 11% LOCKDOWN threshold really means 15 errors (11.72%), and
    the 5% CAUTION threshold means 7 errors (5.47%).
    """
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")
    k = int(min_errors_for(np.array([m]), threshold)[0])
    return k, k / m
