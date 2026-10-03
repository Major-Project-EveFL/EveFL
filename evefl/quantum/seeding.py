"""
Independent random streams for the QKD simulation.

Two requirements drive this module:

1. Different experiment seeds must give different QBER trajectories. Earlier code hashed
   only (round, client), so seeds 0..4 produced identical BB84 draws and would have faked
   confidence intervals. Streams are now derived from
   `numpy.random.SeedSequence(entropy=[experiment_seed, round, client_id])`.

2. Alpha scenarios should be PAIRED comparisons. Alice's bits and bases, Eve's decision to
   intercept, Eve's basis, Bob's bases, the random outcomes of wrong-basis measurements and
   Bob's noise flips each get their OWN stream, and each stream is always consumed for all
   n qubits regardless of the channel. Changing alpha (or the noise) therefore changes only
   what it should, not Alice's bits.
"""

from __future__ import annotations

import zlib
from typing import Dict

import numpy as np

# Stream ids: part of the reproducibility contract, never renumber.
STREAM_NAMES = (
    "alice_bits",
    "alice_bases",
    "eve_intercept",
    "eve_basis",
    "eve_result",   # Eve's outcome when she guesses the wrong basis
    "bob_bases",
    "bob_result",   # Bob's outcome when he measures in the wrong basis
    "bob_flips",
    "sample",       # which sifted positions are publicly compared
)
_STREAM_ID = {name: i for i, name in enumerate(STREAM_NAMES)}


def client_id_to_int(cid: str | int) -> int:
    """Stable integer for a Flower client id ("0", "1", ... or any string)."""
    text = str(cid)
    return int(text) if text.isdigit() else zlib.crc32(text.encode("utf-8"))


def qkd_seed_sequence(
    experiment_seed: int, server_round: int, cid: str | int, purpose: str = "controller"
) -> np.random.SeedSequence:
    """Root SeedSequence for one (experiment, round, client) exchange.

    `purpose` separates the two exchanges a round may run: the "controller" exchange (the QBER signal that drives
    SECURE/CAUTION/LOCKDOWN, entropy unchanged so every validated result is reproduced) and the "key" exchange
    (a separate, larger block for key generation, docs/adr/0001). They never share randomness.
    """
    entropy = [int(experiment_seed), int(server_round), client_id_to_int(cid)]
    if purpose == "key":
        entropy.append(1)
    elif purpose != "controller":
        raise ValueError(f"purpose must be 'controller' or 'key', got {purpose!r}")
    return np.random.SeedSequence(entropy=entropy)


def stream_generators(root: np.random.SeedSequence, call_index: int = 0) -> Dict[str, np.random.Generator]:
    """One independent Generator per named stream, for the `call_index`-th exchange on `root`."""
    base = tuple(root.spawn_key) + (int(call_index),)
    return {
        name: np.random.default_rng(np.random.SeedSequence(entropy=root.entropy, spawn_key=base + (sid,)))
        for name, sid in _STREAM_ID.items()
    }


def int_seed(root: np.random.SeedSequence) -> int:
    """A 31-bit integer seed derived from `root`, for backends that want a plain int (Qiskit)."""
    return int(root.generate_state(1)[0]) & 0x7FFFFFFF  # default dtype is uint32
