"""
Statistical validation of QKD backends against theory and against each other.

The QBER estimate is a count k of errors in a sample of size m, so it is DISCRETE and its
size m is random (the sifted length is about n/2). The checks therefore work on error counts,
conditional on the recorded m, not on the QBER value:

* Given the channel, every sifted bit errs independently with probability
  p = e + (1 - 2e) * alpha / 4, so a sample of m sifted bits contains Binomial(m, p) errors.
  Pooled over trials: K ~ Binomial(sum m_i, p)  -> exact binomial test + Clopper-Pearson CI.
* Per trial, k_i ~ Binomial(m_i, p): a Pearson dispersion statistic ~ chi-square(T) catches
  wrong variance even when the mean is right.
* Sifted length ~ Binomial(n, 1/2).
* Backend vs backend: a 2x2 chi-square test on pooled (errors, non-errors) counts, which is
  valid because the pooled counts are sums of binomials (no continuity assumption, unlike KS).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np
from scipy import stats

from evefl.quantum.base import ChannelModel, QKDProtocol


@dataclass
class TrialCounts:
    """Per-trial error counts and sizes recorded from QKDResult.metadata."""

    n_qubits: int
    sample_errors: List[int]
    sample_sizes: List[int]
    n_sifted: List[int]
    true_errors: List[int]

    @property
    def pooled_errors(self) -> int:
        return int(sum(self.sample_errors))

    @property
    def pooled_size(self) -> int:
        return int(sum(self.sample_sizes))


def run_trials(
    make_protocol: Callable[[int], QKDProtocol], channel: ChannelModel, n_qubits: int, n_trials: int
) -> TrialCounts:
    """Run `n_trials` independent exchanges; `make_protocol(trial_index)` builds a seeded protocol."""
    out = TrialCounts(n_qubits, [], [], [], [])
    for trial in range(n_trials):
        result = make_protocol(trial).run_exchange(n_qubits, channel=channel)
        out.sample_errors.append(int(result.metadata["qber_sample_errors"]))
        out.sample_sizes.append(int(result.metadata["qber_sample_size"]))
        out.n_sifted.append(int(result.n_sifted))
        out.true_errors.append(int(result.metadata["sim_only_true_error_count"]))
    return out


def analytic_check(counts: TrialCounts, channel: ChannelModel, confidence: float = 0.99) -> Dict[str, float]:
    """Compare recorded counts with the analytic QBER, conditional on the recorded sample sizes."""
    p = channel.expected_qber
    k, m = counts.pooled_errors, counts.pooled_size
    test = stats.binomtest(k, m, p)
    ci = test.proportion_ci(confidence_level=confidence, method="exact")

    # Sifted length: total sifted ~ Binomial(total qubits, 1/2).
    total_sifted, total_qubits = int(sum(counts.n_sifted)), counts.n_qubits * len(counts.n_sifted)
    sifted_test = stats.binomtest(total_sifted, total_qubits, 0.5)

    # Ground truth over the whole sifted key.
    truth = stats.binomtest(int(sum(counts.true_errors)), total_sifted, p)

    dispersion_p = float("nan")
    if 0.0 < p < 1.0:
        m_arr = np.asarray(counts.sample_sizes, dtype=float)
        k_arr = np.asarray(counts.sample_errors, dtype=float)
        stat = float(np.sum((k_arr - m_arr * p) ** 2 / (m_arr * p * (1.0 - p))))
        # Two-sided: too much spread (extra noise) and too little (non-independent draws) are both wrong.
        dispersion_p = float(min(1.0, 2.0 * min(stats.chi2.sf(stat, df=len(m_arr)), stats.chi2.cdf(stat, df=len(m_arr)))))

    return {
        "expected_qber": p,
        "pooled_qber": k / m if m else float("nan"),
        "pooled_errors": k,
        "pooled_sample_size": m,
        "qber_ci_low": float(ci.low),
        "qber_ci_high": float(ci.high),
        "qber_pvalue": float(test.pvalue),
        "sifted_fraction": total_sifted / total_qubits,
        "sifted_pvalue": float(sifted_test.pvalue),
        "true_qber_pvalue": float(truth.pvalue),
        "dispersion_pvalue": dispersion_p,
    }


def compare_backends(a: TrialCounts, b: TrialCounts) -> Dict[str, float]:
    """2x2 chi-square test that both backends have the same error probability."""
    table = np.array(
        [[a.pooled_errors, a.pooled_size - a.pooled_errors], [b.pooled_errors, b.pooled_size - b.pooled_errors]]
    )
    if table[:, 0].sum() == 0 or table[:, 1].sum() == 0:
        # Both backends produced only errors, or none at all: identical by construction.
        return {"chi2": 0.0, "pvalue": 1.0, "qber_a": a.pooled_errors / a.pooled_size,
                "qber_b": b.pooled_errors / b.pooled_size}
    chi2, pvalue, _, _ = stats.chi2_contingency(table, correction=False)
    return {
        "chi2": float(chi2),
        "pvalue": float(pvalue),
        "qber_a": a.pooled_errors / a.pooled_size,
        "qber_b": b.pooled_errors / b.pooled_size,
    }
