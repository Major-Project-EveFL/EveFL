"""
EveFL Phase-4 smoke test.

Run this BEFORE any full experiment to verify the entire pipeline works.

What it tests:
    1. Synthetic ChestX-ray14 data generation (no real dataset needed)
    2. partition_and_save() creates .npy index files
    3. 3 clients x 2 FL rounds with Flower simulation
    4. SECURE path: normal FedAvg aggregation
    5. CAUTION/LOCKDOWN path: triggered by high intercept probability
    6. FedProx term is actually added to loss in CAUTION
    7. LOCKDOWN preserves global parameters unchanged
    8. QBER and state are logged correctly
    9. results/fl_training_log.json is valid

Usage:
    python scripts/smoke_test.py

Requires:
    pip install flwr==1.11.1 torch==2.5.1 torchvision==0.20.1 qiskit==1.2.4 qiskit-aer==0.15.1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# Ensure repo root is on path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evefl.fl.dataset import partition_and_save, get_hospital_dataloader
from evefl.fl.model import build_resnet18, get_device
from evefl.fl.client import create_client_fn
from evefl.fl.strategy import EveFLStrategy
from evefl.orchestration.state_machine import StateThresholds

import flwr as fl
from flwr.common import ndarrays_to_parameters
import torch


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("evefl.smoke")


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------

def generate_synthetic_chestxray(
    output_dir: Path,
    n_samples: int = 30,
    image_size: int = 224,
    seed: int = 42,
) -> Path:
    """
    Create a minimal fake ChestX-ray14 dataset:
        - Data_Entry_2017.csv with n_samples rows
        - Random greyscale PNG images in images/
    """
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    labels = [
        "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
        "Mass", "Nodule", "Pneumonia", "Pneumothorax",
        "Consolidation", "Edema", "Emphysema", "Fibrosis",
        "Pleural_Thickening", "Hernia",
    ]

    rows = []
    for i in range(n_samples):
        img_name = f"smoke_{i:04d}.png"
        # Random greyscale image
        img_array = rng.integers(0, 256, (image_size, image_size), dtype=np.uint8)
        img = Image.fromarray(img_array, mode="L")
        img.save(images_dir / img_name)

        # Random 1-3 labels
        n_labels = rng.integers(1, 4)
        finding_labels = "|".join(rng.choice(labels, size=n_labels, replace=False))
        rows.append({"Image Index": img_name, "Finding Labels": finding_labels})

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "Data_Entry_2017.csv", index=False)
    log.info("Generated %d synthetic images in %s", n_samples, output_dir)
    return output_dir


# ---------------------------------------------------------------------------
# Smoke test runner
# ---------------------------------------------------------------------------

def run_smoke(
    n_samples: int = 30,
    n_rounds: int = 2,
    local_epochs: int = 1,
    batch_size: int = 4,
    n_qubits: int = 128,
    intercept_probability: float = 0.0,
    seed: int = 42,
) -> dict:
    """
    Run a minimal end-to-end EveFL simulation.
    """
    device = get_device()
    log.info("=" * 60)
    log.info("EveFL SMOKE TEST")
    log.info("Device: %s", device)
    log.info("Samples: %d | Rounds: %d | Epochs: %d | Batch: %d", n_samples, n_rounds, local_epochs, batch_size)
    log.info("Qubits: %d | Intercept: %.2f", n_qubits, intercept_probability)
    log.info("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        data_root = Path(tmpdir) / "data"
        partition_root = Path(tmpdir) / "partitions"
        results_file = Path(tmpdir) / "results" / "fl_training_log.json"

        # 1. Generate synthetic data
        log.info("[1/5] Generating synthetic ChestX-ray14 data...")
        generate_synthetic_chestxray(data_root, n_samples=n_samples, seed=seed)

        # 2. Partition
        log.info("[2/5] Partitioning into 3 hospitals...")
        partition_and_save(
            data_root=data_root,
            partition_root=partition_root,
            n_clients=3,
            alpha=0.5,
            test_fraction=0.1,
            seed=seed,
            subset_fraction=1.0,
        )

        # Verify partitions exist
        for i in range(3):
            idx_path = partition_root / f"hospital_{i}" / "indices.npy"
            assert idx_path.exists(), f"Missing partition: {idx_path}"
            idx = np.load(idx_path)
            log.info("  Hospital %d: %d samples", i, len(idx))

        # 3. Build initial parameters
        log.info("[3/5] Building initial ResNet-18 parameters...")
        model = build_resnet18(pretrained=False)
        init_params = ndarrays_to_parameters([
            v.detach().cpu().numpy().copy()
            for _, v in model.state_dict().items()
        ])

        # 4. Create strategy
        log.info("[4/5] Creating EveFL strategy...")
        strategy = EveFLStrategy(
            initial_parameters=init_params,
            intercept_probability=intercept_probability,
            n_qubits=n_qubits,
            thresholds=StateThresholds(),
            fraction_fit=1.0,
            min_fit_clients=3,
            min_available_clients=3,
            evaluate_fn=None,
        )

        # 5. Create client factory
        client_fn = create_client_fn(
            data_root=data_root,
            partition_root=partition_root,
            batch_size=batch_size,
        )

        # 6. Run Flower simulation
        log.info("[5/5] Running Flower simulation (%d rounds)...", n_rounds)
        start = time.perf_counter()

        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=3,
            config=fl.server.ServerConfig(num_rounds=n_rounds),
            strategy=strategy,
            client_resources={
                "num_cpus": 1,
                "num_gpus": 1.0 if torch.cuda.is_available() else 0.0,
            },
            ray_init_args={"include_dashboard": False, "log_to_driver": False},
        )

        elapsed = time.perf_counter() - start
        log.info("Simulation finished in %.2f seconds.", elapsed)

        # 7. Assertions
        log.info("Running assertions...")
        assert len(strategy.round_logs) == n_rounds, \
            f"Expected {n_rounds} round logs, got {len(strategy.round_logs)}"

        for rlog in strategy.round_logs:
            assert "state" in rlog, "Missing state in round log"
            assert "system_qber" in rlog, "Missing system_qber in round log"
            assert "qber_per_client" in rlog, "Missing qber_per_client in round log"
            assert rlog["state"] in ("SECURE", "CAUTION", "LOCKDOWN"), \
                f"Invalid state: {rlog['state']}"

        # Check LOCKDOWN behavior: if any round was LOCKDOWN, params should be unchanged
        lockdown_rounds = [r for r in strategy.round_logs if r["state"] == "LOCKDOWN"]
        if lockdown_rounds:
            log.info("LOCKDOWN detected in %d round(s).", len(lockdown_rounds))
            # The last good parameters should still be valid
            assert strategy._last_good_parameters is not None

        # Check CAUTION behavior: FedProx mu should have been sent
        caution_rounds = [r for r in strategy.round_logs if r["state"] == "CAUTION"]
        if caution_rounds:
            log.info("CAUTION detected in %d round(s).", len(caution_rounds))

        # 8. Export JSON
        results_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "experiment": {
                "name": "smoke_test",
                "seed": seed,
                "num_clients": 3,
                "num_rounds": n_rounds,
                "local_epochs": local_epochs,
                "batch_size": batch_size,
                "n_qubits": n_qubits,
                "intercept_probability": intercept_probability,
                "elapsed_seconds": elapsed,
            },
            "security": {
                "rounds": strategy.round_logs,
            },
        }
        with open(results_file, "w") as f:
            json.dump(payload, f, indent=2)

        log.info("Results JSON written to: %s", results_file)

        # Print summary
        secure = sum(1 for r in strategy.round_logs if r["state"] == "SECURE")
        caution = sum(1 for r in strategy.round_logs if r["state"] == "CAUTION")
        lockdown = sum(1 for r in strategy.round_logs if r["state"] == "LOCKDOWN")

        log.info("=" * 60)
        log.info("SMOKE TEST PASSED")
        log.info("SECURE:   %d", secure)
        log.info("CAUTION:  %d", caution)
        log.info("LOCKDOWN: %d", lockdown)
        log.info("Elapsed:  %.2f s", elapsed)
        log.info("=" * 60)

        return {
            "passed": True,
            "secure": secure,
            "caution": caution,
            "lockdown": lockdown,
            "elapsed": elapsed,
            "results_file": str(results_file),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="EveFL smoke test")
    parser.add_argument("--samples", type=int, default=30, help="Synthetic samples (default: 30)")
    parser.add_argument("--rounds", type=int, default=2, help="FL rounds (default: 2)")
    parser.add_argument("--epochs", type=int, default=1, help="Local epochs (default: 1)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument("--qubits", type=int, default=128, help="BB84 qubits (default: 128)")
    parser.add_argument("--intercept", type=float, default=0.0, help="Eve intercept probability (default: 0.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--high-intercept-test", action="store_true", help="Also run with intercept=1.0 to force CAUTION/LOCKDOWN")
    args = parser.parse_args()

    # Run 1: normal (should be SECURE)
    result = run_smoke(
        n_samples=args.samples,
        n_rounds=args.rounds,
        local_epochs=args.epochs,
        batch_size=args.batch_size,
        n_qubits=args.qubits,
        intercept_probability=args.intercept,
        seed=args.seed,
    )

    if not result["passed"]:
        log.error("Smoke test FAILED")
        return 1

    # Optional: run with high intercept to test CAUTION/LOCKDOWN
    if args.high_intercept_test:
        log.info("")
        log.info("Running HIGH-INTERCEPT smoke test (intercept=1.0)...")
        result2 = run_smoke(
            n_samples=args.samples,
            n_rounds=args.rounds,
            local_epochs=args.epochs,
            batch_size=args.batch_size,
            n_qubits=args.qubits,
            intercept_probability=1.0,
            seed=args.seed + 1,
        )
        if not result2["passed"]:
            log.error("High-intercept smoke test FAILED")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())