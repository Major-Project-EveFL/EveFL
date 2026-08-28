"""
EveFL federated-learning server / simulation entry point.

This module is the execution layer. It does NOT contain:
    - BB84 implementation
    - QBER mathematics
    - StateController logic
    - PyTorch model definition
    - dataset implementation

Those remain in their respective modules.

Execution flow:
    server.py
        |
        | Flower simulation
        v
    EveFLStrategy  (strategy.py)
        |
        +---- BB84 per client / round
        +---- q_max
        +---- StateController
        +---- SECURE / CAUTION / LOCKDOWN
        +---- client.fit()
        +---- aggregation
        |
        v
    security log  ->  results/fl_training_log.json

Can be used for:
    1. tiny smoke tests
    2. attack experiments
    3. full ChestX-ray14 experiments
    4. Kaggle execution

Flower version target: flwr==1.11.1
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import flwr as fl
import numpy as np
import torch
from flwr.common import ndarrays_to_parameters

from evefl.fl.client import create_client_fn
from evefl.fl.model import build_resnet18, get_device
from evefl.fl.strategy import EveFLStrategy
from evefl.orchestration.state_machine import StateThresholds


# ============================================================================
# Paths
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_RESULTS_FILE = DEFAULT_RESULTS_DIR / "fl_training_log.json"


# ============================================================================
# Experiment defaults
# ============================================================================

DEFAULT_NUM_CLIENTS = 3
DEFAULT_NUM_ROUNDS = 50
DEFAULT_LOCAL_EPOCHS = 5
DEFAULT_BATCH_SIZE = 32
DEFAULT_N_QUBITS = 1024
DEFAULT_INTERCEPT_PROBABILITY = 0.0
DEFAULT_FRACTION_FIT = 1.0
DEFAULT_MIN_FIT_CLIENTS = 3
DEFAULT_MIN_AVAILABLE_CLIENTS = 3
DEFAULT_SEED = 42


# ============================================================================
# Logging
# ============================================================================

def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

log = logging.getLogger("evefl.server")


# ============================================================================
# Reproducibility
# ============================================================================

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# Initial model
# ============================================================================

def build_initial_parameters() -> fl.common.Parameters:
    """Build the initial global ResNet-18 parameters."""
    log.info("Building initial global ResNet-18 model...")
    model = build_resnet18(pretrained=False)
    model_parameters = [
        value.detach().cpu().numpy().copy()
        for _, value in model.state_dict().items()
    ]
    parameters = ndarrays_to_parameters(model_parameters)
    log.info("Initial model contains %d parameter tensors.", len(model_parameters))
    return parameters


# ============================================================================
# Strategy factory
# ============================================================================

def create_strategy(
    *,
    initial_parameters: fl.common.Parameters,
    intercept_probability: float,
    n_qubits: int,
    fraction_fit: float,
    min_fit_clients: int,
    min_available_clients: int,
) -> EveFLStrategy:
    """Construct the EveFL Flower strategy."""
    return EveFLStrategy(
        initial_parameters=initial_parameters,
        intercept_probability=intercept_probability,
        n_qubits=n_qubits,
        thresholds=StateThresholds(),
        fraction_fit=fraction_fit,
        min_fit_clients=min_fit_clients,
        min_available_clients=min_available_clients,
        evaluate_fn=None,
    )


# ============================================================================
# JSON serialization
# ============================================================================

def make_json_serializable(value: Any) -> Any:
    """Recursively convert NumPy/PyTorch values into JSON-compatible values."""
    if isinstance(value, dict):
        return {str(k): make_json_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_serializable(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    return value


# ============================================================================
# Experiment log
# ============================================================================

def build_experiment_log(
    *,
    strategy: EveFLStrategy,
    experiment_name: str,
    num_clients: int,
    num_rounds: int,
    local_epochs: int,
    batch_size: int,
    n_qubits: int,
    intercept_probability: float,
    seed: int,
    elapsed_seconds: float,
) -> Dict[str, Any]:
    """Build the final results/fl_training_log.json structure."""
    return {
        "experiment": {
            "name": experiment_name,
            "seed": seed,
            "num_clients": num_clients,
            "num_rounds": num_rounds,
            "local_epochs": local_epochs,
            "batch_size": batch_size,
            "n_qubits": n_qubits,
            "intercept_probability": intercept_probability,
            "elapsed_seconds": elapsed_seconds,
        },
        "model": {
            "architecture": "ResNet-18",
            "num_classes": 14,
            "task": "multi-label chest X-ray classification",
        },
        "dataset": {
            "name": "NIH ChestX-ray14",
            "num_hospital_clients": num_clients,
            "non_iid": True,
            "dirichlet_alpha": 0.5,
        },
        "security": {
            "qber_theoretical_relation": "QBER ~= alpha / 4",
            "states": ["SECURE", "CAUTION", "LOCKDOWN"],
            "rounds": strategy.round_logs,
        },
        "evaluation": {
            "status": "global held-out evaluation to be added",
        },
    }


def save_training_log(
    strategy: EveFLStrategy,
    *,
    output_path: Path,
    experiment_name: str,
    num_clients: int,
    num_rounds: int,
    local_epochs: int,
    batch_size: int,
    n_qubits: int,
    intercept_probability: float,
    seed: int,
    elapsed_seconds: float,
) -> None:
    """Write the FL/security log to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = build_experiment_log(
        strategy=strategy,
        experiment_name=experiment_name,
        num_clients=num_clients,
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        batch_size=batch_size,
        n_qubits=n_qubits,
        intercept_probability=intercept_probability,
        seed=seed,
        elapsed_seconds=elapsed_seconds,
    )

    payload = make_json_serializable(payload)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    log.info("Training log written to: %s", output_path)


# ============================================================================
# Main experiment runner
# ============================================================================

def run_experiment(
    *,
    data_root: Path,
    partition_root: Path,
    num_clients: int,
    num_rounds: int,
    local_epochs: int,
    batch_size: int,
    n_qubits: int,
    intercept_probability: float,
    seed: int,
    experiment_name: str,
    output_path: Path,
) -> Dict[str, Any]:
    """
    Run one complete EveFL Flower simulation.

    This is the function the Kaggle notebook should eventually call.
    """
    # Validation
    if num_clients < 1:
        raise ValueError("num_clients must be >= 1.")
    if num_rounds < 1:
        raise ValueError("num_rounds must be >= 1.")
    if local_epochs < 1:
        raise ValueError("local_epochs must be >= 1.")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")
    if n_qubits < 1:
        raise ValueError("n_qubits must be >= 1.")
    if not 0.0 <= intercept_probability <= 1.0:
        raise ValueError("intercept_probability must be between 0 and 1.")

    # Reproducibility
    set_global_seed(seed)

    device = get_device()
    log.info("=" * 60)
    log.info("Starting EveFL experiment: %s", experiment_name)
    log.info("Device: %s", device)
    log.info("Clients: %d", num_clients)
    log.info("Rounds: %d", num_rounds)
    log.info("Local epochs: %d", local_epochs)
    log.info("Batch size: %d", batch_size)
    log.info("BB84 qubits: %d", n_qubits)
    log.info("Eve interception probability: %.2f", intercept_probability)
    log.info("=" * 60)

    # Verify paths
    if not data_root.exists():
        raise FileNotFoundError(
            f"ChestX-ray14 data root does not exist: {data_root}"
        )
    if not partition_root.exists():
        raise FileNotFoundError(
            f"Partition root does not exist: {partition_root}. "
            "Run dataset.partition_and_save() first."
        )

    # Initial global model
    initial_parameters = build_initial_parameters()

    # Client factory
    client_fn = create_client_fn(
        data_root=data_root,
        partition_root=partition_root,
        batch_size=batch_size,
    )

    # Strategy
    strategy = create_strategy(
        initial_parameters=initial_parameters,
        intercept_probability=intercept_probability,
        n_qubits=n_qubits,
        fraction_fit=DEFAULT_FRACTION_FIT,
        min_fit_clients=min(num_clients, DEFAULT_MIN_FIT_CLIENTS),
        min_available_clients=max(num_clients, DEFAULT_MIN_AVAILABLE_CLIENTS),
    )

    # Run Flower
    start_time = time.perf_counter()
    log.info("Starting Flower simulation...")

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={
            "num_cpus": 1,
            "num_gpus": 1.0 if torch.cuda.is_available() else 0.0,
        },
        ray_init_args={"include_dashboard": False},
    )

    elapsed_seconds = time.perf_counter() - start_time
    log.info("Flower simulation finished in %.2f seconds.", elapsed_seconds)

    # Export results
    save_training_log(
        strategy=strategy,
        output_path=output_path,
        experiment_name=experiment_name,
        num_clients=num_clients,
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        batch_size=batch_size,
        n_qubits=n_qubits,
        intercept_probability=intercept_probability,
        seed=seed,
        elapsed_seconds=elapsed_seconds,
    )

    # Summary
    secure_rounds = sum(1 for r in strategy.round_logs if r.get("state") == "SECURE")
    caution_rounds = sum(1 for r in strategy.round_logs if r.get("state") == "CAUTION")
    lockdown_rounds = sum(1 for r in strategy.round_logs if r.get("state") == "LOCKDOWN")

    summary = {
        "experiment": experiment_name,
        "elapsed_seconds": elapsed_seconds,
        "rounds_recorded": len(strategy.round_logs),
        "secure_rounds": secure_rounds,
        "caution_rounds": caution_rounds,
        "lockdown_rounds": lockdown_rounds,
        "output": str(output_path),
        "history_available": history is not None,
    }

    log.info("=" * 60)
    log.info("EveFL experiment complete.")
    log.info("SECURE rounds: %d", secure_rounds)
    log.info("CAUTION rounds: %d", caution_rounds)
    log.info("LOCKDOWN rounds: %d", lockdown_rounds)
    log.info("Results: %s", output_path)
    log.info("=" * 60)

    return summary


# ============================================================================
# Command-line interface
# ============================================================================

def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the EveFL Phase-4 federated-learning simulation."
    )

    parser.add_argument(
        "--data-root", type=Path, required=True,
        help="Root directory containing ChestX-ray14 images/ and Data_Entry_2017.csv.",
    )
    parser.add_argument(
        "--partition-root", type=Path, required=True,
        help="Directory containing hospital_0/, hospital_1/, etc. partitions.",
    )
    parser.add_argument(
        "--clients", type=int, default=DEFAULT_NUM_CLIENTS,
        help=f"Number of federated clients (default: {DEFAULT_NUM_CLIENTS}).",
    )
    parser.add_argument(
        "--rounds", type=int, default=DEFAULT_NUM_ROUNDS,
        help=f"Number of FL rounds (default: {DEFAULT_NUM_ROUNDS}).",
    )
    parser.add_argument(
        "--epochs", type=int, default=DEFAULT_LOCAL_EPOCHS,
        help=f"Local epochs per client (default: {DEFAULT_LOCAL_EPOCHS}).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Local batch size (default: {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--qubits", type=int, default=DEFAULT_N_QUBITS,
        help=f"BB84 qubits per client per round (default: {DEFAULT_N_QUBITS}).",
    )
    parser.add_argument(
        "--intercept-probability", type=float, default=DEFAULT_INTERCEPT_PROBABILITY,
        help="Eve intercept-resend probability in [0,1]. 0 = no Eve, 1 = full intercept-resend.",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"Experiment random seed (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--name", type=str, default="evefl_phase4",
        help="Experiment name used in the results JSON.",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_RESULTS_FILE,
        help="Path for the exported FL training log.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose debug logging.",
    )

    return parser


# ============================================================================
# CLI entry point
# ============================================================================

def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()
    configure_logging(verbose=args.verbose)

    try:
        summary = run_experiment(
            data_root=args.data_root,
            partition_root=args.partition_root,
            num_clients=args.clients,
            num_rounds=args.rounds,
            local_epochs=args.epochs,
            batch_size=args.batch_size,
            n_qubits=args.qubits,
            intercept_probability=args.intercept_probability,
            seed=args.seed,
            experiment_name=args.name,
            output_path=args.output,
        )

        print()
        print("EveFL experiment finished.")
        print(f"Experiment: {summary['experiment']}")
        print(f"Rounds recorded: {summary['rounds_recorded']}")
        print(f"SECURE: {summary['secure_rounds']}")
        print(f"CAUTION: {summary['caution_rounds']}")
        print(f"LOCKDOWN: {summary['lockdown_rounds']}")
        print(f"Log: {summary['output']}")

        return 0

    except KeyboardInterrupt:
        log.warning("Experiment interrupted by user.")
        return 130
    except Exception:
        log.exception("EveFL experiment failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())