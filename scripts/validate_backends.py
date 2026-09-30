"""
Cross-validate the Qiskit BB84 backend against the exact numpy backend and the analytic QBER,
and write the evidence to a JSON file kept in the repo (docs/validation/, since results/ is
gitignored).

For each (alpha, e) it reports pooled error counts, Clopper-Pearson intervals against
QBER = e + (1 - 2e) * alpha / 4, a dispersion check, the sifted-length check, and a 2x2 chi-square
comparison of the two backends. All checks are conditional on the recorded sample sizes.

Slow (Qiskit runs one Aer job per measurement). Run by the nightly / manual workflow:

    python scripts/validate_backends.py --n-qubits 256 --trials 60
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np
import scipy

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evefl.quantum.base import ChannelModel  # noqa: E402
from evefl.quantum.bb84 import BB84Protocol  # noqa: E402
from evefl.quantum.bb84_numpy import BB84NumpyProtocol  # noqa: E402
from evefl.quantum.validation import analytic_check, compare_backends, run_trials  # noqa: E402

ALPHAS = (0.0, 0.3, 0.6, 1.0)
NOISES = (0.0, 0.02)
SIGNIFICANCE = 1e-3


def main() -> int:
    parser = argparse.ArgumentParser(description="Cross-validate BB84 backends")
    parser.add_argument("--n-qubits", type=int, default=256)
    parser.add_argument("--trials", type=int, default=60, help="Qiskit trials per case (numpy uses 10x)")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "docs" / "validation" / "backend_crosscheck.json")
    args = parser.parse_args()

    cases, failures = [], []
    for alpha in ALPHAS:
        for e in NOISES:
            channel = ChannelModel(alpha, e)
            qiskit = run_trials(lambda t: BB84Protocol(seed=args.seed + t), channel, args.n_qubits, args.trials)
            numpy_ = run_trials(
                lambda t: BB84NumpyProtocol.for_round(args.seed, t, "0"), channel, args.n_qubits, 10 * args.trials
            )
            q_check = analytic_check(qiskit, channel, confidence=0.999)
            n_check = analytic_check(numpy_, channel, confidence=0.999)
            comparison = compare_backends(qiskit, numpy_)
            case = {"alpha": alpha, "bit_flip": e, "expected_qber": channel.expected_qber,
                    "qiskit": q_check, "numpy": n_check, "qiskit_vs_numpy": comparison}
            cases.append(case)

            ok = (
                q_check["qber_ci_low"] <= channel.expected_qber <= q_check["qber_ci_high"]
                and n_check["qber_ci_low"] <= channel.expected_qber <= n_check["qber_ci_high"]
                and comparison["pvalue"] > SIGNIFICANCE
                and q_check["sifted_pvalue"] > SIGNIFICANCE
            )
            print(f"alpha={alpha:.1f} e={e:.2f} expected={channel.expected_qber:.4f} "
                  f"qiskit={q_check['pooled_qber']:.4f} numpy={n_check['pooled_qber']:.4f} "
                  f"p(qiskit~numpy)={comparison['pvalue']:.3f} {'OK' if ok else 'FAIL'}")
            if not ok:
                failures.append((alpha, e))

    report = {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "config": {"n_qubits": args.n_qubits, "qiskit_trials": args.trials, "numpy_trials": 10 * args.trials,
                   "seed": args.seed, "significance": SIGNIFICANCE},
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__},
        "passed": not failures,
        "failed_cases": failures,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {args.output}  ->  {'PASSED' if not failures else 'FAILED'}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
