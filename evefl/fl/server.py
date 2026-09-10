"""
EveFL federated-learning experiment entry point.

Execution flow:

    server.py (this file)
        |
        | run_sequential_fl()      <-- runner.py: plain Python for-loop,
        v                              single process, no Ray, no gRPC
    EveFLStrategy  (strategy.py)
        |
        +---- BB84 per client / round      (evefl.quantum.bb84)
        +---- system_qber = max(...)
        +---- StateController               (evefl.orchestration.state_machine)
        +---- SECURE / CAUTION / LOCKDOWN
        +---- client.fit()                  (client.py, in-process)
        +---- FedAvg / FedProx+anomaly / discard
        |
        v
    round_logs  ->  results/<experiment_name>.json

No `fl.simulation.start_simulation`, no Ray, no `client_resources` GPU
fractioning anywhere in this file — the execution engine is the
sequential driver from the start, not a later patch. See runner.py's
module docstring for why that specifically matters on Kaggle.

Usage (Kaggle notebook cell, or CLI):

    python -m evefl.fl.server \\
        --data-root /kaggle/input/chestxray14 \\
        --partition-root /kaggle/working/partitions \\
        --num-rounds 12 \\
        --attack-schedule demo \\
        --subset-fraction 0.05 \\
        --experiment-name demo_run

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
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
from flwr.common import ndarrays_to_parameters

from evefl.fl.client import create_client_fn, get_model_parameters
from evefl.fl.dataset import partition_and_save
from evefl.fl.evaluation import make_evaluate_fn
from evefl.fl.model import build_resnet18
from evefl.fl.runner import build_local_client_proxies, run_sequential_fl
from evefl.fl.strategy import EveFLStrategy
from evefl.orchestration.state_machine import StateThresholds

log = logging.getLogger("evefl.fl.server")

DEFAULT_FRACTION_FIT = 1.0
DEFAULT_MIN_FIT_CLIENTS = 3
DEFAULT_MIN_AVAILABLE_CLIENTS = 3


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
# Attack-schedule presets
# ============================================================================

def demo_attack_schedule(server_round: int) -> float:
    """
    Quiet -> ramping attack -> full intercept-resend -> attacker backs
    off. Tuned for a ~12-round demo so a single run visibly walks
    through SECURE -> CAUTION -> LOCKDOWN -> SECURE. For longer runs,
    the last phase (SECURE) just continues, which is harmless.
    """
    if server_round <= 3:
        return 0.0
    if server_round <= 6:
        return 0.3
    if server_round <= 9:
        return 0.7
    return 0.0


ATTACK_SCHEDULES: Dict[str, Callable[[int], float]] = {
    "demo": demo_attack_schedule,
}


# ============================================================================
# Building blocks
# ============================================================================

def build_initial_parameters():
    model = build_resnet18(pretrained=True)
    return ndarrays_to_parameters(get_model_parameters(model))


def create_strategy(
    *,
    initial_parameters,
    intercept_probability: float,
    intercept_probability_schedule: Optional[Callable[[int], float]],
    n_qubits: int,
    num_clients: int,
    evaluate_fn=None,
) -> EveFLStrategy:
    return EveFLStrategy(
        initial_parameters=initial_parameters,
        intercept_probability=intercept_probability,
        intercept_probability_schedule=intercept_probability_schedule,
        n_qubits=n_qubits,
        thresholds=StateThresholds(),
        fraction_fit=DEFAULT_FRACTION_FIT,
        min_fit_clients=min(num_clients, DEFAULT_MIN_FIT_CLIENTS),
        min_available_clients=max(num_clients, DEFAULT_MIN_AVAILABLE_CLIENTS),
        evaluate_fn=evaluate_fn,
    )


# ============================================================================
# Experiment runner
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
    intercept_probability_schedule: Optional[Callable[[int], float]] = None,
    eval_every_n_rounds: int = 1,
    run_federated_evaluate: bool = True,
) -> Dict[str, Any]:
    """
    Run one complete EveFL federated-learning experiment on the
    Ray-free sequential driver.

    If `intercept_probability_schedule` is given, it overrides the
    fixed `intercept_probability` and Eve's activity varies round to
    round — see `ATTACK_SCHEDULES["demo"]` for a schedule tuned to
    visibly hit all three security states in one short run.
    """
    set_global_seed(seed)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    initial_parameters = build_initial_parameters()

    client_fn = create_client_fn(
        data_root, partition_root,
        batch_size=batch_size, local_epochs=local_epochs,
    )
    client_proxies = build_local_client_proxies(client_fn, num_clients=num_clients)

    evaluate_fn = make_evaluate_fn(
        data_root=data_root,
        partition_root=partition_root,
        batch_size=max(batch_size, 64),
        every_n_rounds=eval_every_n_rounds,
        num_rounds=num_rounds,
    )

    strategy = create_strategy(
        initial_parameters=initial_parameters,
        intercept_probability=intercept_probability,
        intercept_probability_schedule=intercept_probability_schedule,
        n_qubits=n_qubits,
        num_clients=num_clients,
        evaluate_fn=evaluate_fn,
    )

    start_time = time.perf_counter()
    log.info("Starting sequential (Ray-free) EveFL run: %d round(s), %d client(s)...",
              num_rounds, num_clients)

    final_parameters, history = run_sequential_fl(
        client_proxies=client_proxies,
        strategy=strategy,
        num_rounds=num_rounds,
        initial_parameters=initial_parameters,
        run_federated_evaluate=run_federated_evaluate,
    )

    elapsed_seconds = time.perf_counter() - start_time
    log.info("EveFL run finished in %.2f seconds (%d failure(s)).",
              elapsed_seconds, len(history.failures))

    experiment_log = build_experiment_log(
        experiment_name=experiment_name,
        seed=seed,
        num_clients=num_clients,
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        batch_size=batch_size,
        n_qubits=n_qubits,
        intercept_probability=intercept_probability,
        elapsed_seconds=elapsed_seconds,
        round_logs=strategy.round_logs,
        n_failures=len(history.failures),
    )

    with open(output_path, "w") as f:
        json.dump(experiment_log, f, indent=2)
    log.info("Results written to %s", output_path)

    return {
        "output": str(output_path),
        "n_failures": len(history.failures),
        "final_state": strategy.round_logs[-1]["state"] if strategy.round_logs else None,
    }


def build_experiment_log(
    *,
    experiment_name: str,
    seed: int,
    num_clients: int,
    num_rounds: int,
    local_epochs: int,
    batch_size: int,
    n_qubits: int,
    intercept_probability: float,
    elapsed_seconds: float,
    round_logs: list,
    n_failures: int,
) -> Dict[str, Any]:
    state_counts: Dict[str, int] = {}
    for r in round_logs:
        state_counts[r["state"]] = state_counts.get(r["state"], 0) + 1

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
            "execution_engine": "sequential (Ray-free)",
            "n_failures": n_failures,
            "state_counts": state_counts,
        },
        "rounds": round_logs,
    }


# ============================================================================
# CLI
# ============================================================================

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run an EveFL federated-learning experiment.")
    p.add_argument("--data-root", type=Path, required=True,
                    help="Folder containing images/ and Data_Entry_2017.csv")
    p.add_argument("--partition-root", type=Path, required=True,
                    help="Where to read/write hospital_*/indices.npy partitions")
    p.add_argument("--num-clients", type=int, default=3)
    p.add_argument("--num-rounds", type=int, default=12)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--qubits", type=int, default=512, dest="n_qubits")
    p.add_argument("--intercept-probability", type=float, default=0.0,
                    help="Fixed Eve intercept-resend probability. Ignored if --attack-schedule is set.")
    p.add_argument("--attack-schedule", type=str, default=None, choices=list(ATTACK_SCHEDULES.keys()),
                    help="Use a preset per-round schedule instead of a fixed intercept-probability.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--experiment-name", type=str, default="evefl_experiment")
    p.add_argument("--output", type=Path, default=Path("results/fl_training_log.json"))
    p.add_argument("--eval-every-n-rounds", type=int, default=1)
    p.add_argument("--subset-fraction", type=float, default=1.0,
                    help="If the partitions at --partition-root don't exist yet, "
                         "(re-)partition using only this fraction of the dataset "
                         "(e.g. 0.05 for a quick demo run).")
    p.add_argument("--partition-alpha", type=float, default=0.5,
                    help="Dirichlet alpha for non-IID partitioning; lower = more skewed.")
    p.add_argument("--no-federated-evaluate", action="store_true",
                    help="Skip per-client evaluate() calls (server-side evaluate_fn still runs).")
    return p


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    args = _build_arg_parser().parse_args(argv)

    partition_meta = args.partition_root / "partition_meta.json"
    if not partition_meta.exists():
        log.info("No existing partition at %s — partitioning now (subset_fraction=%.3f)...",
                  args.partition_root, args.subset_fraction)
        partition_and_save(
            args.data_root, args.partition_root,
            n_clients=args.num_clients,
            alpha=args.partition_alpha,
            seed=args.seed,
            subset_fraction=args.subset_fraction,
        )
    else:
        log.info("Using existing partition at %s", args.partition_root)

    schedule = ATTACK_SCHEDULES[args.attack_schedule] if args.attack_schedule else None

    summary = run_experiment(
        data_root=args.data_root,
        partition_root=args.partition_root,
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        n_qubits=args.n_qubits,
        intercept_probability=args.intercept_probability,
        intercept_probability_schedule=schedule,
        seed=args.seed,
        experiment_name=args.experiment_name,
        output_path=args.output,
        eval_every_n_rounds=args.eval_every_n_rounds,
        run_federated_evaluate=not args.no_federated_evaluate,
    )

    log.info("Done: %s", summary)


if __name__ == "__main__":
    main()
