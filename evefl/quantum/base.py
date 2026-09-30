"""
Abstract interface for QKD protocols.

BB84 is the first implementation (see bb84.py, Qiskit; bb84_numpy.py, fast and
exact). Any future protocol (E91, B92, six-state, etc.) implements this same
interface so the rest of EveFL (state controller, Flower strategy) never needs
to know which QKD scheme is running underneath.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from evefl.registry import Registry

# One registry for every QKD backend ("bb84" = Qiskit, "bb84_numpy" = fast exact numpy, ...).
quantum_registry: Registry = Registry("quantum_protocol")


@dataclass(frozen=True)
class ChannelModel:
    """The quantum channel between Alice and Bob for one exchange.

    intercept_probability: Eve measures-and-resends each qubit independently with
        this probability (intercept-resend attack, random basis).
    bit_flip_probability: baseline channel noise, modelled as an independent bit
        flip with this probability on Bob's measurement result, applied AFTER
        Eve. It is NOT depolarizing noise (that would be a separate field).

    Expected QBER on the sifted key, with alpha = intercept_probability and
    e = bit_flip_probability (see `expected_qber`):

        QBER = e + (1 - 2e) * alpha / 4

    Eve alone causes an error on a sifted bit with probability alpha/4 (she
    intercepts with probability alpha, picks the wrong basis half the time, and
    a wrong-basis resend errs half the time); the noise flip is independent.
    """

    intercept_probability: float = 0.0
    bit_flip_probability: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.intercept_probability <= 1.0:
            raise ValueError(f"intercept_probability must be in [0, 1], got {self.intercept_probability}")
        if not 0.0 <= self.bit_flip_probability <= 0.5:
            raise ValueError(
                f"bit_flip_probability must be in [0, 0.5] (0.5 = a channel that carries no information), "
                f"got {self.bit_flip_probability}"
            )

    @property
    def expected_qber(self) -> float:
        e, alpha = self.bit_flip_probability, self.intercept_probability
        return e + (1.0 - 2.0 * e) * alpha / 4.0


def resolve_channel(intercept_probability: float, channel: Optional[ChannelModel]) -> ChannelModel:
    """Combine the legacy `intercept_probability` argument with the new `channel` argument.

    Callers may use either; passing a non-zero legacy value together with `channel` is
    ambiguous and rejected.
    """
    if channel is None:
        return ChannelModel(intercept_probability=intercept_probability)
    if intercept_probability != 0.0:
        raise ValueError("Pass either intercept_probability or channel, not both.")
    return channel


@dataclass
class QKDResult:
    """Outcome of a single QKD exchange (i.e. one 'round' of key agreement).

    `qber` is the ESTIMATE from a public sample of the sifted key; it is the only quantity
    the controller may use. Simulation-only ground truth (e.g. the true error rate over the
    whole sifted key) is stored in `metadata` under keys prefixed `sim_only_`: a real QKD
    system cannot observe it.
    """

    sifted_key: list[int]          # the raw sifted key bits (post basis-reconciliation)
    qber: float                    # estimated quantum bit error rate, 0.0-1.0
    n_qubits_sent: int
    n_sifted: int                  # number of bits surviving basis sifting
    intercept_probability: float   # ground truth flag: fraction of qubits intercepted by Eve (0.0-1.0)
    eavesdropper_active: bool      # convenience flag: true if intercept_probability > 0.0, false otherwise
    metadata: dict = field(default_factory=dict)  # any extra info the protocol wants to report


class QKDProtocol(ABC):
    """
    Base class every QKD protocol implementation must satisfy.

    Implementations are expected to be simulation-only for now, but the
    interface makes no assumption about that: a real hardware backend could
    implement the same methods later.
    """

    @abstractmethod
    def run_exchange(
        self,
        n_qubits: int,
        intercept_probability: float = 0.0,
        *,
        channel: Optional[ChannelModel] = None,
    ) -> QKDResult:
        """
        Simulate (or perform) one full QKD exchange and return the result.

        Args:
            n_qubits: number of qubits to send before sifting.
            intercept_probability: legacy shorthand for
                `ChannelModel(intercept_probability=...)`. Kept until every caller
                has moved to `channel`.
            channel: the channel (Eve and baseline noise) for this exchange.

        Returns:
            QKDResult with the sifted key and the sampled QBER estimate.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError
