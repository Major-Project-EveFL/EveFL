"""
Abstract interface for QKD protocols.

BB84 is the first implementation (see bb84.py). Any future protocol
(E91, B92, six-state, etc.) implements this same interface so the rest
of EveFL (state controller, Flower strategy) never needs to know which
QKD scheme is running underneath.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class QKDResult:
    """Outcome of a single QKD exchange (i.e. one 'round' of key agreement)."""

    sifted_key: list[int]          # the raw sifted key bits (post basis-reconciliation)
    qber: float                    # estimated quantum bit error rate, 0.0-1.0
    n_qubits_sent: int
    n_sifted: int                  # number of bits surviving basis sifting
    intercept_probability: float      # ground truth flag: fraction of qubits intercepted by Eve (0.0-1.0)
    eavesdropper_active: bool      # convenience flag: true if intercept_probability > 0.0, false otherwise
    metadata: dict = field(default_factory=dict)  # any extra info the protocol wants to

class QKDProtocol(ABC):
    """
    Base class every QKD protocol implementation must satisfy.

    Implementations are expected to be simulation-only for now (Qiskit),
    but the interface makes no assumption about that — a real hardware
    backend could implement the same methods later.
    """

    @abstractmethod
    def run_exchange(self, n_qubits: int, intercept_probability: float = 0.0) -> QKDResult:
        """
        Simulate (or perform) one full QKD exchange and return the result.

        Args:
            n_qubits: number of qubits to send before sifting.
            intercept_probability: fraction of qubits intercepted by Eve (0.0-1.0).
            eavesdropper_active: whether to simulate an intercept-resend
                (or other) eavesdropping attack on this exchange.

        Returns:
            QKDResult with the sifted key and measured QBER.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError
