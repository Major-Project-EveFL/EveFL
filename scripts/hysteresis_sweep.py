"""
Hysteresis and policy-mode sweep (docs/06, "Pre-registered: hysteresis and policy-mode sweep (Phase 2)").

Control-plane only: sampled QBER trajectories (exact count-level model) through the real policy classes. Reports flapping,
false alarms, detection and recovery latency and client-rounds trained for the hysteresis grid, applies the PRE-REGISTERED
selection rule at 2% noise, and compares the three policy modes (global_binary / global / per_client) under Eve on one link.
Writes docs/validation/hysteresis_sweep.json. All numbers are SIMULATED and say nothing about model accuracy.

    python scripts/hysteresis_sweep.py            # full pre-registered grid
    python scripts/hysteresis_sweep.py --quick    # tiny grid, smoke test only (no recommendation is selected)
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evefl.fl.channels import ChannelPlan, LinkAttack  # noqa: E402
from evefl.fl.control_sim import (  # noqa: E402
    PolicyRun,
    client_rounds_trained,
    first_round_where,
    link_state_changes_per_round,
    rounds_discarded,
    sample_qber_trajectories,
    share_of_link_rounds,
    simulate_policy,
)
from evefl.orchestration.policy import PolicyConfig  # noqa: E402
from evefl.orchestration.state_machine import HysteresisConfig, SecurityState  # noqa: E402

# ---- Pre-registered (docs/06). Do not edit after seeing results. ----------------------------------------------
LINKS = ("0", "1", "2")
N_QUBITS, SAMPLE_FRACTION = 1024, 0.25
ROUNDS, REPS, SEED = 50, 1000, 0
DWELLS = (1, 2, 3, 5, 8)
MARGINS = (0.0, 0.005, 0.01, 0.02)
NOISES = (0.0, 0.01, 0.02, 0.03)
ATTACK_C = dict(alpha=0.6, kind="window", start_round=21, end_round=30)
ANCHOR_NOISE = 0.02
MAX_RECOVERY_ROUNDS = 5
MIN_CLIENT_ROUNDS_FRACTION = 0.95
COMPARISON_NOISE = 0.01
COMPARISON_STATIC_ALPHAS = (0.3, 0.44, 0.6, 1.0)
# ----------------------------------------------------------------------------------------------------------------

S, C, L = SecurityState.SECURE, SecurityState.CAUTION, SecurityState.LOCKDOWN
CODES = {S: 0, C: 1, L: 2}


def nanmean(values: List[float]) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.nanmean(arr)) if np.any(~np.isnan(arr)) else float("nan")


def state_codes(run: PolicyRun) -> np.ndarray:
    return np.array([[CODES[s[cid]] for cid in LINKS] for s in run.link_states])


def assert_escalation_never_delayed(qbers: np.ndarray, runs: List[PolicyRun]) -> None:
    for rep, run in enumerate(runs):
        codes = state_codes(run)
        assert np.all(codes[qbers[rep] >= 0.11] == 2), "LOCKDOWN was entered late"
        assert np.all(codes[qbers[rep] >= 0.05] >= 1), "CAUTION was entered late"


def trajectories(plan_for_rep: Callable[[int], ChannelPlan], reps: int, rounds: int) -> np.ndarray:
    return sample_qber_trajectories(plan_for_rep, LINKS, rounds, reps, n_qubits=N_QUBITS,
                                    sample_fraction=SAMPLE_FRACTION, seed=SEED)


def hysteresis_cell(dwell: int, margin: float, qa: np.ndarray, qb: np.ndarray, qc: np.ndarray) -> Dict:
    config = PolicyConfig(mode="per_client", min_clients=2, hysteresis=HysteresisConfig(margin=margin, dwell=dwell))
    runs_a, runs_b, runs_c = (simulate_policy(q, LINKS, config) for q in (qa, qb, qc))
    for q, runs in ((qa, runs_a), (qb, runs_b), (qc, runs_c)):
        assert_escalation_never_delayed(q, runs)
    detection = first_round_where(runs_c, "0", ATTACK_C["start_round"], lambda s: s == L)
    recovery_secure = [v + 1 for v in first_round_where(runs_c, "0", ATTACK_C["end_round"] + 1, lambda s: s == S)]
    recovery_included = [v + 1 for v in first_round_where(runs_c, "0", ATTACK_C["end_round"] + 1, lambda s: s != L)]
    return {
        "dwell": dwell, "margin": margin,
        "A_not_secure_share": 1.0 - share_of_link_rounds(runs_a, LINKS, S),
        "A_lockdown_share": share_of_link_rounds(runs_a, LINKS, L),
        "A_changes_per_link_round": link_state_changes_per_round(runs_a, LINKS),
        "B_changes_per_link_round": link_state_changes_per_round(runs_b, LINKS),
        "B_lockdown_share": share_of_link_rounds(runs_b, ("0",), L),
        "C_detection_latency": nanmean(detection),
        "C_recovery_to_secure": nanmean(recovery_secure),
        "C_recovery_to_included": nanmean(recovery_included),
        "C_client_rounds_trained": client_rounds_trained(runs_c),
        "C_rounds_discarded": rounds_discarded(runs_c),
    }


def select_recommended(cells_at_anchor: List[Dict]) -> Dict:
    baseline = next(c for c in cells_at_anchor if c["dwell"] == 1 and c["margin"] == 0.0)
    floor = MIN_CLIENT_ROUNDS_FRACTION * baseline["C_client_rounds_trained"]
    candidates = [c for c in cells_at_anchor
                  if c["C_recovery_to_secure"] <= MAX_RECOVERY_ROUNDS and c["C_client_rounds_trained"] >= floor]
    best = min(candidates, key=lambda c: (c["A_changes_per_link_round"], c["dwell"], c["margin"]), default=None)
    return {"rule": "docs/06 pre-registered", "anchor_noise": ANCHOR_NOISE,
            "baseline_client_rounds": baseline["C_client_rounds_trained"], "client_rounds_floor": floor,
            "candidates": [(c["dwell"], c["margin"]) for c in candidates],
            "recommended": None if best is None else {"dwell": best["dwell"], "margin": best["margin"]},
            "recommended_is_no_hysteresis": bool(best is not None and best["dwell"] == 1 and best["margin"] == 0.0)}


def policy_comparison(reps: int, rounds: int) -> List[Dict]:
    scenarios: Dict[str, Callable[[int], ChannelPlan]] = {}
    for alpha in COMPARISON_STATIC_ALPHAS:
        scenarios[f"static alpha={alpha}"] = (
            lambda rep, a=alpha: ChannelPlan(attacks={"0": LinkAttack(alpha=a)}, default_noise=COMPARISON_NOISE))
    scenarios["window alpha=0.6 rounds 21-30"] = (
        lambda rep: ChannelPlan(attacks={"0": LinkAttack(**ATTACK_C)}, default_noise=COMPARISON_NOISE))
    scenarios["intermittent alpha=0.6 p=0.3"] = (
        lambda rep: ChannelPlan(attacks={"0": LinkAttack(alpha=0.6, kind="intermittent", probability=0.3, seed=rep)},
                                default_noise=COMPARISON_NOISE))
    rows = []
    for name, plan_for_rep in scenarios.items():
        q = trajectories(plan_for_rep, reps, rounds)
        for mode in ("global_binary", "global", "per_client"):
            runs = simulate_policy(q, LINKS, PolicyConfig(mode=mode, min_clients=2))
            participation = {cid: float(np.mean([sum(cid in inc for inc in run.included) for run in runs]))
                             for cid in LINKS}
            rows.append({
                "scenario": name, "mode": mode,
                "rounds_discarded_share": rounds_discarded(runs) / rounds,
                "client_rounds_share": client_rounds_trained(runs) / (len(LINKS) * rounds),
                "participation_rounds": participation,
            })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Hysteresis and policy-mode sweep (pre-registered in docs/06)")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "docs" / "validation" / "hysteresis_sweep.json")
    parser.add_argument("--quick", action="store_true", help="tiny grid for a smoke test; no recommendation selected")
    args = parser.parse_args()

    reps = 40 if args.quick else REPS
    rounds = ROUNDS
    dwells = (1, 3) if args.quick else DWELLS
    margins = (0.0, 0.01) if args.quick else MARGINS
    noises = (0.02,) if args.quick else NOISES

    started = time.time()
    grid: Dict[str, List[Dict]] = {}
    for e in noises:
        qa = trajectories(lambda rep, e=e: ChannelPlan(default_noise=e), reps, rounds)
        qb = trajectories(lambda rep, e=e: ChannelPlan(attacks={"0": LinkAttack(alpha=0.44)}, default_noise=e), reps, rounds)
        qc = trajectories(lambda rep, e=e: ChannelPlan(attacks={"0": LinkAttack(**ATTACK_C)}, default_noise=e), reps, rounds)
        cells = [hysteresis_cell(n, m, qa, qb, qc) for n in dwells for m in margins]
        grid[str(e)] = cells
        latencies = {round(c["C_detection_latency"], 9) for c in cells}
        assert len(latencies) == 1, f"detection latency differs across cells at e={e}: {latencies}"
        print(f"e={e}: {len(cells)} cells done, detection latency {latencies.pop():.3f} rounds "
              f"({time.time() - started:.0f}s)", flush=True)

    selection = (select_recommended(grid[str(ANCHOR_NOISE)])
                 if (not args.quick and str(ANCHOR_NOISE) in grid) else {"rule": "skipped (--quick)"})
    comparison = policy_comparison(reps, rounds)

    report = {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "label": "simulated control-plane statistics (rounds and client-rounds); say nothing about accuracy",
        "preregistered_in": "docs/06_EXPERIMENT_PROTOCOL.md",
        "config": {"links": list(LINKS), "n_qubits": N_QUBITS, "sample_fraction": SAMPLE_FRACTION, "rounds": rounds,
                   "reps": reps, "seed": SEED, "dwells": list(dwells), "margins": list(margins),
                   "noises": list(noises), "attack_C": ATTACK_C, "anchor_noise": ANCHOR_NOISE,
                   "max_recovery_rounds": MAX_RECOVERY_ROUNDS,
                   "min_client_rounds_fraction": MIN_CLIENT_ROUNDS_FRACTION,
                   "comparison_noise": COMPARISON_NOISE},
        "selection": selection,
        "hysteresis_grid": grid,
        "policy_comparison": comparison,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1, default=lambda x: None if (isinstance(x, float) and math.isnan(x)) else x),
                           encoding="utf-8")

    print("\n=== Recommended hysteresis (pre-registered rule, e=2%) ===")
    print(json.dumps(selection, indent=1))
    print("\n=== Policy modes under Eve on ONE link (e=1%) ===")
    for row in comparison:
        print(f"{row['scenario']:<32} {row['mode']:<14} discarded={row['rounds_discarded_share']:.3f} "
              f"client_rounds={row['client_rounds_share']:.3f} participation="
              f"{ {k: round(v, 1) for k, v in row['participation_rounds'].items()} }")
    print(f"\nWrote {args.output} ({time.time() - started:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
