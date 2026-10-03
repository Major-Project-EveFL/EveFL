"""
QBER sample-size sweep (docs/06, "Pre-registered: QBER sample-size sweep").

Sweeps block size, sample fraction, baseline noise and Eve's intercept probability; reports false-alarm
and detection rates for the system state (max QBER over 3 links) as exact analytic values next to a paired
Monte Carlo, applies the PRE-REGISTERED selection rule, and writes everything to JSON under docs/validation/
(results/ is gitignored).

    python scripts/qber_sweep.py                      # full pre-registered grid
    python scripts/qber_sweep.py --quick              # tiny grid, for a smoke test only

The grid, thresholds, rule and seed are fixed by docs/06 and are NOT parameters worth tuning after seeing
results; --quick only shrinks the grid for testing and refuses to select a headline.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evefl.orchestration.state_machine import StateThresholds  # noqa: E402
from evefl.quantum.sweep import (  # noqa: E402
    PairedDraws,
    analytic_single_link,
    check_against_protocol,
    error_probability,
    simulate_system_rates,
    system_probability,
)

# ---- Pre-registered (docs/06). Do not edit after seeing results. ----------------------------------------
N_QUBITS = (256, 512, 1024, 2048, 4096, 8192, 16384)
SAMPLE_FRACTIONS = (0.10, 0.25, 0.50)
NOISES = (0.0, 0.01, 0.02, 0.03)
ALPHAS = tuple(round(0.1 * i, 1) for i in range(11))
LINKS = 3
ROUNDS = 20_000
SEED = 0
THRESHOLDS = StateThresholds()  # 0.05 / 0.11
MAX_FALSE_CAUTION = 0.01        # criterion 1, at e = 0.01
MIN_DETECTION = 0.95            # criterion 2, at alpha = 0.30 for e in {0, 0.01}
DETECTION_ALPHA = 0.3
MIN_SAMPLE = 100                # criterion 3
# ---------------------------------------------------------------------------------------------------------


def select_headline(cells: list[dict]) -> dict:
    """Apply the pre-registered rule to the analytic system-level rates."""
    index = {(c["n_qubits"], c["sample_fraction"], c["bit_flip"], c["alpha"]): c for c in cells}
    candidates = []
    for n in N_QUBITS:
        for f in SAMPLE_FRACTIONS:
            base = index[(n, f, 0.01, 0.0)]
            ok_false = base["analytic"]["system_p_not_secure"] <= MAX_FALSE_CAUTION
            ok_detect = all(
                index[(n, f, e, DETECTION_ALPHA)]["analytic"]["system_p_not_secure"] >= MIN_DETECTION
                for e in (0.0, 0.01)
            )
            ok_sample = base["expected_sample"] >= MIN_SAMPLE
            if ok_false and ok_detect and ok_sample:
                candidates.append({
                    "n_qubits": n, "sample_fraction": f,
                    "false_caution_e0.01": base["analytic"]["system_p_not_secure"],
                    "detection_a0.3_e0": index[(n, f, 0.0, DETECTION_ALPHA)]["analytic"]["system_p_not_secure"],
                    "detection_a0.3_e0.01": index[(n, f, 0.01, DETECTION_ALPHA)]["analytic"]["system_p_not_secure"],
                    "expected_sample": base["expected_sample"],
                    "expected_leftover_per_qubit": base["expected_leftover_per_qubit"],
                })
    headline = None
    if candidates:
        # smallest n_qubits; ties -> larger leftover key (i.e. smaller sample_fraction)
        headline = min(candidates, key=lambda c: (c["n_qubits"], -c["expected_leftover_per_qubit"]))
    return {"rule": "docs/06 pre-registered", "candidates": candidates, "headline": headline}


def main() -> int:
    parser = argparse.ArgumentParser(description="QBER sample-size sweep (pre-registered in docs/06)")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "docs" / "validation" / "qber_sweep.json")
    parser.add_argument("--quick", action="store_true", help="tiny grid for a smoke test; no headline is selected")
    parser.add_argument("--protocol-check-rounds", type=int, default=3000)
    args = parser.parse_args()

    n_grid = (256, 1024) if args.quick else N_QUBITS
    f_grid = (0.25,) if args.quick else SAMPLE_FRACTIONS
    rounds = 2_000 if args.quick else ROUNDS

    started = time.time()
    cells: list[dict] = []
    for n in n_grid:
        draws = PairedDraws.draw(n, rounds, LINKS, SEED)
        for f in f_grid:
            for e in NOISES:
                for alpha in ALPHAS:
                    p = error_probability(alpha, e)
                    single = analytic_single_link(n, f, p, THRESHOLDS)
                    sim = simulate_system_rates(draws, f, p, THRESHOLDS)
                    cells.append({
                        "n_qubits": n, "sample_fraction": f, "bit_flip": e, "alpha": alpha,
                        "error_probability": p,
                        "expected_sample": single["expected_sample"],
                        "expected_leftover_per_qubit": single["expected_leftover_per_qubit"],
                        "analytic": {
                            "single_p_not_secure": single["p_not_secure"],
                            "single_p_lockdown": single["p_lockdown"],
                            "system_p_not_secure": system_probability(single["p_not_secure"], LINKS),
                            "system_p_lockdown": system_probability(single["p_lockdown"], LINKS),
                        },
                        "simulated": sim,
                    })
        print(f"n_qubits={n} done ({time.time() - started:.0f}s)", flush=True)

    # Agreement between the exact analytic values and the paired Monte Carlo (Wilson 95% intervals).
    inside = sum(
        c["simulated"]["p_not_secure_ci95"][0] - 1e-12 <= c["analytic"]["system_p_not_secure"]
        <= c["simulated"]["p_not_secure_ci95"][1] + 1e-12
        for c in cells
    )
    print(f"analytic value inside the simulated 95% CI in {inside}/{len(cells)} cells "
          f"(~95% expected; cells are correlated through shared draws)")

    # Verify the count-level engine against the FULL protocol on a subset of cells.
    subset = [(1024, 0.25, 0.0, 0.0), (1024, 0.25, 0.01, 0.3), (1024, 0.25, 0.02, 0.6), (2048, 0.10, 0.01, 0.3)]
    protocol_checks = [
        check_against_protocol(n, f, e, a, args.protocol_check_rounds, LINKS, SEED, THRESHOLDS)
        for (n, f, e, a) in subset
    ]
    for c in protocol_checks:
        print(f"protocol check n={c['n_qubits']} f={c['sample_fraction']} e={c['bit_flip']} a={c['alpha']}: "
              f"P(not secure) protocol={c['protocol_p_not_secure']:.4f} analytic={c['analytic_p_not_secure']:.4f} "
              f"p={c['pvalue_not_secure']:.3f}")

    selection = select_headline(cells) if not args.quick else {"rule": "skipped (--quick)", "candidates": [],
                                                              "headline": None}
    report = {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "preregistered_in": "docs/06_EXPERIMENT_PROTOCOL.md",
        "config": {
            "n_qubits": list(n_grid), "sample_fractions": list(f_grid), "bit_flip_noise": list(NOISES),
            "alphas": list(ALPHAS), "links": LINKS, "rounds": rounds, "seed": SEED,
            "secure_max": THRESHOLDS.secure_max, "caution_max": THRESHOLDS.caution_max,
            "criteria": {"max_false_caution_at_e0.01": MAX_FALSE_CAUTION, "min_detection": MIN_DETECTION,
                         "detection_alpha": DETECTION_ALPHA, "detection_noises": [0.0, 0.01],
                         "min_expected_sample": MIN_SAMPLE},
        },
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__},
        "analytic_inside_sim_ci": {"inside": inside, "cells": len(cells)},
        "protocol_checks": protocol_checks,
        "selection": selection,
        "cells": cells,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1), encoding="utf-8")

    print("\n=== Selection (pre-registered rule) ===")
    print(f"candidates: {len(selection['candidates'])}")
    print("headline  :", selection["headline"])
    print(f"\nWrote {args.output} ({time.time() - started:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
