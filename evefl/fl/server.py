"""
EveFL federated-learning server / simulation entry point.

Phase 4 responsibility
----------------------

This module is the execution layer.

It does NOT contain:

    - BB84 implementation
    - QBER mathematics
    - StateController logic
    - PyTorch model definition
    - dataset implementation
    - Streamlit code

Those remain in their respective modules.

The execution flow is:

    server.py
        |
        | Flower simulation
        v
    EveFlStrategy
        |
        +---- BB84 per client / round
        |
        +---- q_max
        |
        +---- StateController
        |
        +---- SECURE / CAUTION / LOCKDOWN
        |
        +---- client.fit()
        |
        +---- aggregation
        |
        v
    security log
        |
        v
    results/fl_training_log.json

The same server runner can be used for:

    1. tiny smoke tests
    2. attack experiments
    3. full ChestX-ray14 experiments
    4. Kaggle execution

The Kaggle notebook should eventually call this file rather than
containing the actual FL logic.
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

from evefl.fl.client import create_client_fn
from evefl.fl.model import build_resnet18, get_device
from evefl.fl.strategy import EveFLStrategy


# ============================================================================
# Paths
# ============================================================================

# server.py is:
#
#     EveFL/
#         evefl/
#             fl/
#                 server.py
#
# Therefore:
#
#     PROJECT_ROOT = EveFL/
#
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"

DEFAULT_RESULTS_FILE = (
    DEFAULT_RESULTS_DIR / "fl_training_log.json"
)


# ============================================================================
# Experiment defaults
# ============================================================================

DEFAULT_NUM_CLIENTS = 3

DEFAULT_NUM_ROUNDS = 50

DEFAULT_LOCAL_EPOCHS = 5

DEFAULT_BATCH_SIZE = 32

DEFAULT_N_QUBITS = 1024

DEFAULT_INTERCEPT_PROBABILITY = 0.0

DEFAULT_PROXIMAL_MU = 0.01

DEFAULT_ANOMALY_THRESHOLD = 2.0

DEFAULT_SEED = 42

DEFAULT_FRACTION_FIT = 1.0

DEFAULT_MIN_FIT_CLIENTS = 3

DEFAULT_MIN_AVAILABLE_CLIENTS = 3


# ============================================================================
# Logging
# ============================================================================

def configure_logging(
    verbose: bool = False,
) -> None:
    """
    Configure application logging.

    INFO is sufficient for normal experiments.

    DEBUG can be enabled during implementation/debugging.
    """
    level = (
        logging.DEBUG
        if verbose
        else logging.INFO
    )

    logging.basicConfig(
        level=level,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        ),
    )


log = logging.getLogger("evefl.server")


# ============================================================================
# Reproducibility
# ============================================================================

def set_global_seed(seed: int) -> None:
    """
    Set global random seeds.

    Individual clients additionally receive deterministic
    client/round-specific seeds in client.py.
    """
    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# Flower compatibility adapter
# ============================================================================

class EveFLSimulationStrategy(EveFLStrategy):
    """
    Thin compatibility wrapper around EveFLStrategy.

    Flower's Strategy.configure_fit() receives:

        server_round
        parameters
        client_manager

    The Phase-4 strategy implementation we created earlier keeps the client
    manager on the strategy object because its internal QBER/policy code
    needs to know which clients were selected.

    This adapter bridges those two interfaces.

    Once the smoke test confirms the exact installed Flower API, this wrapper
    can be removed and the third argument can simply be added directly to
    EveFLStrategy.configure_fit().
    """

    def configure_fit(
        self,
        server_round: int,
        parameters: fl.common.Parameters,
        client_manager: fl.server.client_manager.ClientManager,
    ):
        """
        Store Flower's client manager and delegate to EveFLStrategy.
        """
        self._client_manager = client_manager

        return super().configure_fit(
            server_round,
            parameters,
        )


# ============================================================================
# Initial model
# ============================================================================

def build_initial_parameters() -> fl.common.Parameters:
    """
    Build the initial global ResNet-18 parameters.

    The architecture is the same architecture constructed by every client.

    We do NOT train here.

    The model is only used to obtain the initial state_dict that Flower
    distributes to all clients during round 1.
    """
    log.info(
        "Building initial global ResNet-18 model..."
    )

    model = build_resnet18(
        pretrained=False
    )

    model_parameters = [
        value.detach()
        .cpu()
        .numpy()
        .copy()
        for _, value
        in model.state_dict().items()
    ]

    parameters = fl.common.ndarrays_to_parameters(
        model_parameters
    )

    log.info(
        "Initial model contains %d parameter tensors.",
        len(model_parameters),
    )

    return parameters


# ============================================================================
# Evaluation
# ============================================================================

def build_evaluate_fn(
    data_root: Path,
    partition_root: Path,
    batch_size: int,
):
    """
    Build a server-side evaluation function.

    Important:

    The final EveFL experiment is supposed to report global AUC-ROC on a
    held-out IID test split.

    The exact dataset/evaluation implementation belongs in dataset.py.

    Therefore this function first attempts to use the project's dataset
    module through the expected test-loader interface.

    If that interface is not yet present, the function returns None rather
    than inventing a dataset API.

    This keeps the Phase-4 smoke test independent of the final evaluation
    implementation.
    """

    del data_root
    del partition_root
    del batch_size

    # ------------------------------------------------------------------
    # Deliberately disabled until the existing dataset.py contract is
    # verified.
    #
    # We do NOT invent:
    #
    #     get_test_dataloader()
    #
    # or any other unseen function.
    #
    # The first smoke test therefore focuses on:
    #
    #     client training
    #     QBER
    #     controller
    #     aggregation
    #
    # Global held-out AUC will be added once dataset.py is verified.
    # ------------------------------------------------------------------

    return None


# ============================================================================
# Metrics aggregation
# ============================================================================

def aggregate_fit_metrics(
    metrics: List[
        tuple[
            int,
            Dict[str, fl.common.Scalar],
        ]
    ],
) -> Dict[str, fl.common.Scalar]:
    """
    Weighted aggregation helper for client metrics.

    The weighting is based on the number of examples represented by each
    client.

    This function is currently provided for explicitness and future logging.

    Security metrics such as QBER/state are generated by the server strategy
    and should NOT be averaged blindly here.
    """
    if not metrics:
        return {}

    total_examples = sum(
        int(num_examples)
        for num_examples, _ in metrics
    )

    if total_examples <= 0:
        return {}

    aggregated: Dict[str, float] = {}

    numeric_keys = set()

    for _, client_metrics in metrics:
        for key, value in client_metrics.items():
            if isinstance(
                value,
                (int, float, np.integer, np.floating),
            ):
                numeric_keys.add(key)

    for key in numeric_keys:
        weighted_sum = 0.0

        for num_examples, client_metrics in metrics:
            value = client_metrics.get(key)

            if not isinstance(
                value,
                (int, float, np.integer, np.floating),
            ):
                continue

            weighted_sum += (
                float(value)
                * int(num_examples)
            )

        aggregated[key] = (
            weighted_sum / total_examples
        )

    return aggregated


# ============================================================================
# JSON serialization
# ============================================================================

def make_json_serializable(
    value: Any,
) -> Any:
    """
    Recursively convert NumPy/PyTorch values into JSON-compatible values.
    """
    if isinstance(
        value,
        dict,
    ):
        return {
            str(key): make_json_serializable(item)
            for key, item in value.items()
        }

    if isinstance(
        value,
        (list, tuple),
    ):
        return [
            make_json_serializable(item)
            for item in value
        ]

    if isinstance(
        value,
        np.integer,
    ):
        return int(value)

    if isinstance(
        value,
        np.floating,
    ):
        return float(value)

    if isinstance(
        value,
        np.ndarray,
    ):
        return value.tolist()

    if isinstance(
        value,
        torch.Tensor,
    ):
        return value.detach().cpu().tolist()

    if isinstance(
        value,
        Path,
    ):
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
    proximal_mu: float,
    anomaly_threshold: float,
    seed: int,
    elapsed_seconds: float,
) -> Dict[str, Any]:
    """
    Build the final results/fl_training_log.json structure.

    The dashboard can later consume this file without running any FL code.
    """
    security_records = strategy.get_security_log()

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
            "proximal_mu": proximal_mu,
            "anomaly_threshold": anomaly_threshold,
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
            "states": [
                "SECURE",
                "CAUTION",
                "LOCKDOWN",
            ],
            "rounds": security_records,
        },

        "evaluation": {
            "status": (
                "global held-out evaluation is added after "
                "dataset.py evaluation interface is verified"
            ),
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
    proximal_mu: float,
    anomaly_threshold: float,
    seed: int,
    elapsed_seconds: float,
) -> None:
    """
    Write the FL/security log to disk.
    """
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = build_experiment_log(
        strategy=strategy,
        experiment_name=experiment_name,
        num_clients=num_clients,
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        batch_size=batch_size,
        n_qubits=n_qubits,
        intercept_probability=intercept_probability,
        proximal_mu=proximal_mu,
        anomaly_threshold=anomaly_threshold,
        seed=seed,
        elapsed_seconds=elapsed_seconds,
    )

    payload = make_json_serializable(
        payload
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
        )

    log.info(
        "Training log written to: %s",
        output_path,
    )


# ============================================================================
# Strategy factory
# ============================================================================

def create_strategy(
    *,
    num_clients: int,
    num_rounds: int,
    local_epochs: int,
    n_qubits: int,
    intercept_probability: float,
    proximal_mu: float,
    anomaly_threshold: float,
    seed: int,
) -> EveFLSimulationStrategy:
    """
    Construct the EveFL Flower strategy.

    num_rounds is accepted here for experiment-level consistency even though
    Flower's strategy itself is told the current round dynamically.
    """
    del num_rounds

    strategy = EveFLSimulationStrategy(
        num_clients=num_clients,
        n_qubits=n_qubits,
        intercept_probability=intercept_probability,
        proximal_mu=proximal_mu,
        anomaly_threshold=anomaly_threshold,
        local_epochs=local_epochs,
        seed=seed,

        # --------------------------------------------------------------
        # Flower FedAvg configuration
        # --------------------------------------------------------------

        fraction_fit=DEFAULT_FRACTION_FIT,

        fraction_evaluate=0.0,

        min_fit_clients=(
            min(
                DEFAULT_MIN_FIT_CLIENTS,
                num_clients,
            )
        ),

        min_evaluate_clients=0,

        min_available_clients=(
            max(
                DEFAULT_MIN_AVAILABLE_CLIENTS,
                num_clients,
            )
        ),

        # No client-side evaluation in the first smoke test.
        evaluate_fn=None,

        # We explicitly retain client metrics returned by fit().
        fit_metrics_aggregation_fn=None,

        evaluate_metrics_aggregation_fn=None,

        accept_failures=False,
    )

    return strategy


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
    proximal_mu: float,
    anomaly_threshold: float,
    seed: int,
    experiment_name: str,
    output_path: Path,
) -> Dict[str, Any]:
    """
    Run one complete EveFL Flower simulation.

    This is the function that the Kaggle notebook should eventually call.

    The notebook itself should remain a thin execution wrapper.
    """

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    if num_clients < 1:
        raise ValueError(
            "num_clients must be >= 1."
        )

    if num_rounds < 1:
        raise ValueError(
            "num_rounds must be >= 1."
        )

    if local_epochs < 1:
        raise ValueError(
            "local_epochs must be >= 1."
        )

    if batch_size < 1:
        raise ValueError(
            "batch_size must be >= 1."
        )

    if n_qubits < 1:
        raise ValueError(
            "n_qubits must be >= 1."
        )

    if not 0.0 <= intercept_probability <= 1.0:
        raise ValueError(
            "intercept_probability must be between 0 and 1."
        )

    if proximal_mu < 0.0:
        raise ValueError(
            "proximal_mu must be >= 0."
        )

    if anomaly_threshold <= 0.0:
        raise ValueError(
            "anomaly_threshold must be > 0."
        )

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------

    set_global_seed(seed)

    # ------------------------------------------------------------------
    # Environment information
    # ------------------------------------------------------------------

    device = get_device()

    log.info(
        "============================================================"
    )

    log.info(
        "Starting EveFL experiment: %s",
        experiment_name,
    )

    log.info(
        "Device: %s",
        device,
    )

    log.info(
        "Clients: %d",
        num_clients,
    )

    log.info(
        "Rounds: %d",
        num_rounds,
    )

    log.info(
        "Local epochs: %d",
        local_epochs,
    )

    log.info(
        "Batch size: %d",
        batch_size,
    )

    log.info(
        "BB84 qubits: %d",
        n_qubits,
    )

    log.info(
        "Eve interception probability: %.2f",
        intercept_probability,
    )

    log.info(
        "FedProx μ: %.4f",
        proximal_mu,
    )

    log.info(
        "Anomaly threshold: %.4f",
        anomaly_threshold,
    )

    log.info(
        "============================================================"
    )

    # ------------------------------------------------------------------
    # Verify paths
    # ------------------------------------------------------------------

    if not data_root.exists():
        raise FileNotFoundError(
            "ChestX-ray14 data root does not exist:\n"
            f"{data_root}\n\n"
            "For the smoke test, use a tiny prepared dataset directory."
        )

    if not partition_root.exists():
        raise FileNotFoundError(
            "Partition root does not exist:\n"
            f"{partition_root}\n\n"
            "Expected the Phase-4 partitioning step to create the "
            "hospital partitions."
        )

    # ------------------------------------------------------------------
    # Initial global model
    # ------------------------------------------------------------------

    initial_parameters = (
        build_initial_parameters()
    )

    # ------------------------------------------------------------------
    # Client factory
    # ------------------------------------------------------------------

    client_fn = create_client_fn(
        data_root=data_root,
        partition_root=partition_root,
        pretrained=False,
        batch_size=batch_size,
    )

    # ------------------------------------------------------------------
    # Strategy
    # ------------------------------------------------------------------

    strategy = create_strategy(
        num_clients=num_clients,
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        n_qubits=n_qubits,
        intercept_probability=intercept_probability,
        proximal_mu=proximal_mu,
        anomaly_threshold=anomaly_threshold,
        seed=seed,
    )

    # ------------------------------------------------------------------
    # Run Flower
    # ------------------------------------------------------------------

    start_time = time.perf_counter()

    log.info(
        "Starting Flower simulation..."
    )

    history = fl.simulation.start_simulation(
        client_fn=client_fn,

        num_clients=num_clients,

        config=fl.server.ServerConfig(
            num_rounds=num_rounds
        ),

        strategy=strategy,

        client_resources={
            "num_cpus": 1,
            "num_gpus": (
                1.0
                if torch.cuda.is_available()
                else 0.0
            ),
        },

        ray_init_args={
            "include_dashboard": False,
        },
    )

    elapsed_seconds = (
        time.perf_counter()
        - start_time
    )

    log.info(
        "Flower simulation finished in %.2f seconds.",
        elapsed_seconds,
    )

    # ------------------------------------------------------------------
    # Export results
    # ------------------------------------------------------------------

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
        proximal_mu=proximal_mu,
        anomaly_threshold=anomaly_threshold,
        seed=seed,
        elapsed_seconds=elapsed_seconds,
    )

    # ------------------------------------------------------------------
    # Print concise final summary
    # ------------------------------------------------------------------

    security_records = (
        strategy.get_security_log()
    )

    secure_rounds = sum(
        record["state"] == "SECURE"
        for record in security_records
    )

    caution_rounds = sum(
        record["state"] == "CAUTION"
        for record in security_records
    )

    lockdown_rounds = sum(
        record["state"] == "LOCKDOWN"
        for record in security_records
    )

    summary = {
        "experiment": experiment_name,
        "elapsed_seconds": elapsed_seconds,
        "rounds_recorded": len(
            security_records
        ),
        "secure_rounds": secure_rounds,
        "caution_rounds": caution_rounds,
        "lockdown_rounds": lockdown_rounds,
        "output": str(output_path),
        "history_available": history is not None,
    }

    log.info(
        "============================================================"
    )

    log.info(
        "EveFL experiment complete."
    )

    log.info(
        "SECURE rounds: %d",
        secure_rounds,
    )

    log.info(
        "CAUTION rounds: %d",
        caution_rounds,
    )

    log.info(
        "LOCKDOWN rounds: %d",
        lockdown_rounds,
    )

    log.info(
        "Results: %s",
        output_path,
    )

    log.info(
        "============================================================"
    )

    return summary


# ============================================================================
# Command-line interface
# ============================================================================

def build_argument_parser() -> argparse.ArgumentParser:
    """
    Construct the command-line interface.

    Examples:

        Smoke test:

        python -m evefl.fl.server \
            --data-root data/smoke \
            --partition-root data/smoke/partitions \
            --rounds 2 \
            --clients 3 \
            --epochs 1 \
            --qubits 128

        No-Eve experiment:

        python -m evefl.fl.server \
            --data-root data/chestxray14 \
            --partition-root data/chestxray14/partitions \
            --rounds 50 \
            --clients 3 \
            --epochs 5 \
            --intercept-probability 0.0

        Full-Eve experiment:

        python -m evefl.fl.server \
            --data-root data/chestxray14 \
            --partition-root data/chestxray14/partitions \
            --rounds 50 \
            --clients 3 \
            --epochs 5 \
            --intercept-probability 1.0
    """

    parser = argparse.ArgumentParser(
        description=(
            "Run the EveFL Phase-4 federated-learning simulation."
        )
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help=(
            "Root directory containing the local ChestX-ray14 "
            "data used by the clients."
        ),
    )

    parser.add_argument(
        "--partition-root",
        type=Path,
        required=True,
        help=(
            "Directory containing the hospital/client partitions."
        ),
    )

    parser.add_argument(
        "--clients",
        type=int,
        default=DEFAULT_NUM_CLIENTS,
        help=(
            f"Number of federated clients "
            f"(default: {DEFAULT_NUM_CLIENTS})."
        ),
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=DEFAULT_NUM_ROUNDS,
        help=(
            f"Number of FL rounds "
            f"(default: {DEFAULT_NUM_ROUNDS})."
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_LOCAL_EPOCHS,
        help=(
            f"Local epochs per client "
            f"(default: {DEFAULT_LOCAL_EPOCHS})."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=(
            f"Local batch size "
            f"(default: {DEFAULT_BATCH_SIZE})."
        ),
    )

    parser.add_argument(
        "--qubits",
        type=int,
        default=DEFAULT_N_QUBITS,
        help=(
            f"BB84 qubits per client per round "
            f"(default: {DEFAULT_N_QUBITS})."
        ),
    )

    parser.add_argument(
        "--intercept-probability",
        type=float,
        default=DEFAULT_INTERCEPT_PROBABILITY,
        help=(
            "Eve intercept-resend probability in [0,1]. "
            "0 = no Eve, 1 = full intercept-resend."
        ),
    )

    parser.add_argument(
        "--proximal-mu",
        type=float,
        default=DEFAULT_PROXIMAL_MU,
        help=(
            f"FedProx proximal coefficient "
            f"(default: {DEFAULT_PROXIMAL_MU})."
        ),
    )

    parser.add_argument(
        "--anomaly-threshold",
        type=float,
        default=DEFAULT_ANOMALY_THRESHOLD,
        help=(
            f"Update anomaly weighting threshold "
            f"(default: {DEFAULT_ANOMALY_THRESHOLD})."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=(
            f"Experiment random seed "
            f"(default: {DEFAULT_SEED})."
        ),
    )

    parser.add_argument(
        "--name",
        type=str,
        default="evefl_phase4",
        help=(
            "Experiment name used in the results JSON."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RESULTS_FILE,
        help=(
            "Path for the exported FL training log."
        ),
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging.",
    )

    return parser


# ============================================================================
# CLI entry point
# ============================================================================

def main() -> int:
    """
    Command-line entry point.
    """
    parser = build_argument_parser()

    args = parser.parse_args()

    configure_logging(
        verbose=args.verbose
    )

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
            proximal_mu=args.proximal_mu,
            anomaly_threshold=args.anomaly_threshold,
            seed=args.seed,
            experiment_name=args.name,
            output_path=args.output,
        )

        print()
        print("EveFL experiment finished.")
        print(
            f"Experiment: {summary['experiment']}"
        )
        print(
            f"Rounds recorded: "
            f"{summary['rounds_recorded']}"
        )
        print(
            f"SECURE: "
            f"{summary['secure_rounds']}"
        )
        print(
            f"CAUTION: "
            f"{summary['caution_rounds']}"
        )
        print(
            f"LOCKDOWN: "
            f"{summary['lockdown_rounds']}"
        )
        print(
            f"Log: {summary['output']}"
        )

        return 0

    except KeyboardInterrupt:
        log.warning(
            "Experiment interrupted by user."
        )
        return 130

    except Exception:
        log.exception(
            "EveFL experiment failed."
        )
        return 1


if __name__ == "__main__":
    sys.exit(
        main()
    )