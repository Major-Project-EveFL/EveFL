"""
Learning-rate schedule that follows the GLOBAL federated round (P1-8).

Clients build a fresh optimiser every round, so a scheduler living inside the
client would restart from the base rate every round. Instead the server computes
the rate for the current round and sends it in the round config
(`config["learning_rate"]`); clients just use it.

    lr_t = eta_min + (base - eta_min) * 0.5 * (1 + cos(pi * (t - 1) / T))     t = 1..T

Round 1 uses the base rate and later rounds decay along a cosine over the run
(the standard CosineAnnealingLR stepped once per round). In CAUTION the rate is
additionally multiplied by `caution_lr_multiplier` (0.5, i.e. 1e-3 -> 5e-4).
"""

from __future__ import annotations

import math


def cosine_lr(base_lr: float, server_round: int, total_rounds: int, eta_min: float = 0.0) -> float:
    if base_lr <= 0:
        raise ValueError(f"base_lr must be positive, got {base_lr}")
    if total_rounds < 1:
        raise ValueError(f"total_rounds must be >= 1, got {total_rounds}")
    if not 1 <= server_round <= total_rounds:
        raise ValueError(f"server_round must be in [1, {total_rounds}], got {server_round}")
    if not 0.0 <= eta_min <= base_lr:
        raise ValueError(f"eta_min must be in [0, base_lr], got {eta_min}")
    return eta_min + (base_lr - eta_min) * 0.5 * (1.0 + math.cos(math.pi * (server_round - 1) / total_rounds))
