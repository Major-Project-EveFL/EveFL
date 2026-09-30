"""Create a QKD backend by registry key (`settings.yaml: quantum.active_protocol`, `--qkd-backend`)."""

from __future__ import annotations

import numpy as np

from evefl.quantum.base import QKDProtocol, quantum_registry
from evefl.quantum.seeding import int_seed

DEFAULT_BACKEND = "bb84_numpy"


def available_backends() -> list[str]:
    _load_backends()
    return quantum_registry.list_keys()


def _load_backends() -> None:
    # Importing registers the classes. bb84 pulls in Qiskit (slow import), so only on demand.
    from evefl.quantum import bb84_numpy  # noqa: F401
    try:
        from evefl.quantum import bb84  # noqa: F401
    except ImportError:  # qiskit not installed: the numpy backend still works
        pass


def create_protocol(
    backend: str,
    *,
    sample_fraction: float,
    seed_sequence: np.random.SeedSequence,
) -> QKDProtocol:
    """Instantiate `backend`, seeded from `seed_sequence` (see seeding.qkd_seed_sequence)."""
    _load_backends()
    cls = quantum_registry.get(backend)
    if backend == "bb84":  # Qiskit backend takes a plain int seed
        return cls(sample_fraction=sample_fraction, seed=int_seed(seed_sequence))
    return cls(sample_fraction=sample_fraction, seed=seed_sequence)
