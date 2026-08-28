"""
EveFL controlled Eve experiment runner.

Runs all 4 attack scenarios sequentially and saves individual + combined results.

Scenarios (from handoff section 7):
    A: alpha = 0.00  (no Eve)
    B: alpha = 0.30
    C: alpha = 0.60
    D: alpha = 1.00  (full intercept-resend)

Usage:
    python scripts/run_experiments.py \
        --data-root data/chestxray14 \
        --partition-root data/chestxray14/partitions \
        --rounds 50 \
        --output-dir results/

For smoke testing:
    python scripts/run_experiments.py \
        --data-root data/smoke \
        --partition-root data/smoke/partitions \
        --rounds 2 \
        --epochs 1 \
        --batch-size 4 \
        --qubits 128 \
        --output-dir results/smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

# Ensure repo root is on path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evefl.fl.server import run_experiment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("evefl.experiments")


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    name: str
    intercept_probability: float
    description: str


SCENARIOS: List[ExperimentConfig] = [
    ExperimentConfig(
        name="A_no_eve",
        intercept_probability=0.00,
        description="No eavesdropping (baseline)",
    ),
    ExperimentConfig(
        name="B_low_eve",
        intercept_probability=0.30,
        description="Low-intensity intercept-resend",
    ),
    ExperimentConfig(
        name="C_med_eve",
        intercept_probability=0.60,
        description="Medium-intensity intercept-resend",
    ),
    ExperimentConfig(
        name="D_full_eve",
        intercept_probability=1.00,
        description="Full intercept-resend attack",
    ),
]


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_all_experiments(
    *,
    data_root: Path,
    partition_root: Path,
    num_clients: int,
    num_rounds: int,
    local_epochs: int,
    batch_size: int,
    n_qubits: int,
    output_dir: Path,
    seed: int,
) -> Dict[str, Any]:
    """
    Run all 4 Eve scenarios and save results.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries: List[Dict[str, Any]] = []

    for scenario in SCENARIOS:
        log.info("=" * 60)
        log.info("Starting scenario: %s", scenario.name)
        log.info("Description: %s", scenario.description)
        log.info("Intercept probability: %.2f", scenario.intercept_probability)
        log.info("=" * 60)

        scenario_output = output_dir / f"{scenario.name}.json"

        try:
            summary = run_experiment(
                data_root=data_root,
                partition_root=partition_root,
                num_clients=num_clients,
                num_rounds=num_rounds,
                local_epochs=local_epochs,
                batch_size=batch_size,
                n_qubits=n_qubits,
                intercept_probability=scenario.intercept_probability,
                seed=seed,
                experiment_name=scenario.name,
                output_path=scenario_output,
            )

            summary["scenario"] = asdict(scenario)
            all_summaries.append(summary)

            log.info("Scenario %s completed successfully.", scenario.name)

        except Exception:
            log.exception("Scenario %s FAILED.", scenario.name)
            all_summaries.append({
                "scenario": asdict(scenario),
                "passed": False,
                "error": "see logs",
            })

    # Combined summary
    combined = {
        "meta": {
            "num_clients": num_clients,
            "num_rounds": num_rounds,
            "local_epochs": local_epochs,
            "batch_size": batch_size,
            "n_qubits": n_qubits,
            "seed": seed,
        },
        "scenarios": all_summaries,
    }

    combined_path = output_dir / "combined_results.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    log.info("=" * 60)
    log.info("ALL SCENARIOS COMPLETE")
    log.info("Combined results: %s", combined_path)
    log.info("=" * 60)

    # Print summary table
    print()
    print("EveFL Experiment Summary")
    print("=" * 60)
    print(f"{'Scenario':<15} {'Intercept':<10} {'Rounds':<8} {'SECURE':<8} {'CAUTION':<8} {'LOCKDOWN':<8} {'Time(s)':<10}")
    print("-" * 60)
    for s in all_summaries:
        sc = s.get("scenario", {})
        print(
            f"{sc.get('name', '?'):<15} "
            f"{sc.get('intercept_probability', 0):<10.2f} "
            f"{s.get('rounds_recorded', 0):<8} "
            f"{s.get('secure_rounds', 0):<8} "
            f"{s.get('caution_rounds', 0):<8} "
            f"{s.get('lockdown_rounds', 0):<8} "
            f"{s.get('elapsed_seconds', 0):<10.1f}"
        )
    print("=" * 60)

    return combined


def main() -> int:
    parser = argparse.ArgumentParser(description="Run EveFL controlled Eve experiments (Scenarios A-D)")

    parser.add_argument("--data-root", type=Path, required=True, help="ChestX-ray14 data root")
    parser.add_argument("--partition-root", type=Path, required=True, help="Partition root with hospital_*/")
    parser.add_argument("--clients", type=int, default=3, help="Number of clients (default: 3)")
    parser.add_argument("--rounds", type=int, default=50, help="FL rounds per scenario (default: 50)")
    parser.add_argument("--epochs", type=int, default=5, help="Local epochs (default: 5)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--qubits", type=int, default=1024, help="BB84 qubits (default: 1024)")
    parser.add_argument("--output-dir", type=Path, default=Path("results"), help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")

    args = parser.parse_args()

    run_all_experiments(
        data_root=args.data_root,
        partition_root=args.partition_root,
        num_clients=args.clients,
        num_rounds=args.rounds,
        local_epochs=args.epochs,
        batch_size=args.batch_size,
        n_qubits=args.qubits,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())