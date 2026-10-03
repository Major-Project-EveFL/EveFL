"""
Regenerate Table III (QBER validation) and the restated Theorem 1 numbers at the headline setting.

Headline setting: n_qubits = 1024, sample_fraction = 0.25 (a documented deviation from the rule's output,
see docs/06). Noise: bit-flip e in {0, 1%}.

Table III is SIMULATED (full `bb84_numpy` protocol, many trials per alpha, 3 links per trial), with the exact
analytic state probabilities alongside. Theorem 1 numbers are EXACT ANALYTIC values (binomial mixture over the
random sample size, independent links). Nothing here is measured on hardware. Writes
docs/validation/table3_theorem1.json.

    python scripts/table3_theorem1.py --trials 3000
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evefl.orchestration.state_machine import StateThresholds  # noqa: E402
from evefl.quantum.base import ChannelModel  # noqa: E402
from evefl.quantum.bb84_numpy import BB84NumpyProtocol  # noqa: E402
from evefl.quantum.sweep import system_probability  # noqa: E402
from evefl.quantum.theory import (  # noqa: E402
    effective_boundary,
    link_probabilities,
    lockdown_probability,
    minimal_alpha,
)

N_QUBITS = 1024
SAMPLE_FRACTION = 0.25
LINKS = 3
SEED = 2026
THRESHOLDS = StateThresholds()
NOISES = (0.0, 0.01)
TABLE3_ALPHAS = (0.0, 0.1, 0.2, 0.3, 0.44, 0.5, 0.66, 1.0)  # the paper's rows
THEOREM_ALPHAS = tuple(round(0.3 + 0.05 * i, 2) for i in range(15))  # 0.30 .. 1.00


def simulate_cell(alpha: float, e: float, trials: int) -> dict:
    channel = ChannelModel(alpha, e)
    estimates, truths, sizes = [], [], []
    system_lockdown = system_not_secure = 0
    for t in range(1, trials + 1):
        link_q = []
        for link in range(LINKS):
            result = BB84NumpyProtocol.for_round(SEED, t, link, sample_fraction=SAMPLE_FRACTION).run_exchange(
                N_QUBITS, channel=channel
            )
            link_q.append(result.qber)
            estimates.append(result.qber)
            truths.append(result.metadata["sim_only_true_qber"])
            sizes.append(result.metadata["qber_sample_size"])
        system_lockdown += not (max(link_q) < THRESHOLDS.caution_max)
        system_not_secure += not (max(link_q) < THRESHOLDS.secure_max)

    q = np.asarray(estimates)
    secure = float(np.mean(q < THRESHOLDS.secure_max))
    lockdown = float(np.mean(~(q < THRESHOLDS.caution_max)))
    not_secure_exact, lockdown_exact = link_probabilities(N_QUBITS, SAMPLE_FRACTION, alpha, e, THRESHOLDS)
    return {
        "alpha": alpha, "bit_flip": e, "links_per_trial": LINKS, "trials": trials,
        "theoretical_qber": channel.expected_qber,
        "estimate_mean": float(q.mean()),
        "estimate_std": float(q.std(ddof=1)),
        "estimate_mean_ci95_half_width": float(1.959964 * q.std(ddof=1) / np.sqrt(len(q))),
        "true_qber_mean": float(np.mean(truths)),
        "mean_sample_size": float(np.mean(sizes)),
        "all_estimates_exactly_zero": bool(np.all(q == 0.0)),
        "link_state_shares": {"SECURE": secure, "CAUTION": 1.0 - secure - lockdown, "LOCKDOWN": lockdown},
        "link_state_shares_analytic": {"SECURE": 1.0 - not_secure_exact,
                                       "CAUTION": not_secure_exact - lockdown_exact, "LOCKDOWN": lockdown_exact},
        "system_p_lockdown_simulated": system_lockdown / trials,
        "system_p_lockdown_analytic": system_probability(lockdown_exact, LINKS),
        "system_p_not_secure_simulated": system_not_secure / trials,
        "system_p_not_secure_analytic": system_probability(not_secure_exact, LINKS),
    }


def theorem1() -> dict:
    rows = []
    for e in NOISES:
        for a in THEOREM_ALPHAS:
            rows.append({
                "bit_flip": e, "alpha": a,
                "p_lockdown_per_link": lockdown_probability(N_QUBITS, SAMPLE_FRACTION, a, e, 1),
                "p_lockdown_system_3_links": lockdown_probability(N_QUBITS, SAMPLE_FRACTION, a, e, LINKS),
            })
    minimal = {}
    for e in NOISES:
        for target in (0.95, 0.99):
            for links in (1, LINKS):
                minimal[f"e={e}|target={target}|links={links}"] = minimal_alpha(
                    target, N_QUBITS, SAMPLE_FRACTION, e, links=links
                )
    boundaries = {}
    for m in (100, 128, 150, 256, 512):
        k_l, q_l = effective_boundary(m, THRESHOLDS.caution_max)
        k_c, q_c = effective_boundary(m, THRESHOLDS.secure_max)
        boundaries[str(m)] = {"lockdown_errors": k_l, "lockdown_effective_qber": q_l,
                              "caution_errors": k_c, "caution_effective_qber": q_c}
    return {"rows": rows, "minimal_alpha": minimal, "integer_boundaries": boundaries}


def main() -> int:
    parser = argparse.ArgumentParser(description="Table III and Theorem 1 at the headline setting")
    parser.add_argument("--trials", type=int, default=3000, help="trials per (alpha, e); each runs 3 links")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "docs" / "validation" / "table3_theorem1.json")
    args = parser.parse_args()

    started = time.time()
    table3 = []
    for e in NOISES:
        for alpha in TABLE3_ALPHAS:
            cell = simulate_cell(alpha, e, args.trials)
            table3.append(cell)
            print(f"e={e:.2f} alpha={alpha:.2f} theory={cell['theoretical_qber']:.4f} "
                  f"mean={cell['estimate_mean']:.4f}+-{cell['estimate_mean_ci95_half_width']:.4f} "
                  f"sd={cell['estimate_std']:.4f} LOCKDOWN link sim={cell['link_state_shares']['LOCKDOWN']:.3f} "
                  f"exact={cell['link_state_shares_analytic']['LOCKDOWN']:.3f}", flush=True)

    zero = [c for c in table3 if c["alpha"] == 0.0 and c["bit_flip"] == 0.0][0]
    assert zero["all_estimates_exactly_zero"] and zero["estimate_mean"] == 0.0, "alpha=0, e=0 must give exactly 0"

    report = {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "labels": {"table3": "simulated (full bb84_numpy protocol)", "theorem1": "exact analytic"},
        "setting": {"n_qubits": N_QUBITS, "sample_fraction": SAMPLE_FRACTION, "links": LINKS, "seed": SEED,
                    "trials": args.trials, "secure_max": THRESHOLDS.secure_max,
                    "caution_max": THRESHOLDS.caution_max,
                    "deviation_note": "headline (1024, 0.25) deviates from the pre-registered rule's output "
                                      "(512, 0.5); see docs/06"},
        "table3": table3,
        "theorem1": theorem1(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nalpha=0, e=0: exactly 0 in every trial ({zero['trials'] * LINKS} exchanges)")
    print(f"Wrote {args.output} ({time.time() - started:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
