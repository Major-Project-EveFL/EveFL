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

Flower version target: flwr==1.13.0 (flwr==1.11.1 also works functionally,
but pins numpy<2.0, which fights Kaggle's numpy-2.x-native base image and
causes repeated, hard-to-diagnose "numpy.dtype size changed" ABI errors
in unrelated imports like pandas. 1.13.0+ relaxed that constraint to
numpy>=1.26.0,<3.0.0 with no other API changes affecting this codebase
-- verified by running the full test suite against both versions and
diffing the output byte-for-byte.)
"""

from __future__ import annotations

import argparse
import hashlib
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

from evefl.fl.client import DEFAULT_LR, DEFAULT_WEIGHT_DECAY, create_client_fn, get_model_parameters
from evefl.fl.dataset import partition_and_save
from evefl.fl.evaluation import make_evaluate_fn
from evefl.fl.model import build_resnet18, describe_initialisation
from evefl.fl.runner import build_local_client_proxies, run_sequential_fl
from evefl.fl.strategy import DEFAULT_CAUTION_LR_MULTIPLIER, EveFLStrategy
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

def build_initial_parameters(pretrained: bool, pretrained_weights: Optional[Path] = None):
    """Global model initialisation. `pretrained` is an explicit choice: a pretrained run and a
    from-scratch run are not comparable, so it is recorded in every results file."""
    model = build_resnet18(pretrained=pretrained, weights_path=pretrained_weights)
    return ndarrays_to_parameters(get_model_parameters(model))


def create_strategy(
    *,
    initial_parameters,
    intercept_probability: float,
    intercept_probability_schedule: Optional[Callable[[int], float]],
    n_qubits: int,
    num_clients: int,
    evaluate_fn=None,
    base_lr: Optional[float] = None,
    num_rounds: Optional[int] = None,
    caution_lr_multiplier: float = DEFAULT_CAUTION_LR_MULTIPLIER,
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
        base_lr=base_lr,
        num_rounds=num_rounds,
        caution_lr_multiplier=caution_lr_multiplier,
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
    pretrained: bool,
    pretrained_weights: Optional[Path] = None,
    lr: float = DEFAULT_LR,
    caution_lr_multiplier: float = DEFAULT_CAUTION_LR_MULTIPLIER,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
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

    initial_parameters = build_initial_parameters(pretrained, pretrained_weights)
    init_weights = describe_initialisation(pretrained, pretrained_weights)

    client_fn = create_client_fn(
        data_root, partition_root,
        batch_size=batch_size, local_epochs=local_epochs, lr=lr, weight_decay=weight_decay,
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
        base_lr=lr,
        num_rounds=num_rounds,
        caution_lr_multiplier=caution_lr_multiplier,
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
        init_weights=init_weights,
        optimizer={
            "name": "adamw",
            "lr": lr,
            "schedule": "cosine_by_global_round",
            "caution_lr_multiplier": caution_lr_multiplier,
            "weight_decay": weight_decay,
        },
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
    init_weights: Dict[str, Any],
    optimizer: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    state_counts: Dict[str, int] = {}
    for r in round_logs:
        state_counts[r["state"]] = state_counts.get(r["state"], 0) + 1

    config = {
        "name": experiment_name,
        "seed": seed,
        "num_clients": num_clients,
        "num_rounds": num_rounds,
        "local_epochs": local_epochs,
        "batch_size": batch_size,
        "n_qubits": n_qubits,
        "intercept_probability": intercept_probability,
        "pretrained": init_weights["pretrained"],
        "init_weights_sha256": init_weights["sha256"],
        "optimizer": optimizer,
    }
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    return {
        "experiment": {
            **config,
            "config_hash": config_hash,
            "init_weights": init_weights,
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
    p.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True,
                    help="Initialise the global ResNet-18 from ImageNet weights (default, as in the paper). "
                         "Use --no-pretrained for a from-scratch run; the choice is written into the results JSON.")
    p.add_argument("--pretrained-weights", type=Path, default=None,
                    help="Local ImageNet ResNet-18 state-dict file (for offline Kaggle notebooks; "
                         "see scripts/cache_pretrained_weights.py). Without it, torchvision downloads the weights.")
    p.add_argument("--lr", type=float, default=DEFAULT_LR,
                    help="Base (round-1) learning rate for AdamW; cosine-annealed over the GLOBAL rounds.")
    p.add_argument("--caution-lr-multiplier", type=float, default=DEFAULT_CAUTION_LR_MULTIPLIER,
                    help="Learning-rate multiplier applied in CAUTION rounds (0.5: 1e-3 -> 5e-4).")
    p.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
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
        pretrained=args.pretrained,
        pretrained_weights=args.pretrained_weights,
        lr=args.lr,
        caution_lr_multiplier=args.caution_lr_multiplier,
        weight_decay=args.weight_decay,
        eval_every_n_rounds=args.eval_every_n_rounds,
        run_federated_evaluate=not args.no_federated_evaluate,
    )

    log.info("Done: %s", summary)


if __name__ == "__main__":
    main()
