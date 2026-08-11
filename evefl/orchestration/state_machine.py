"""
Three-state QBER-driven aggregation controller.

Pure logic, no dependency on Flower/PyTorch/Qiskit — takes a QBER
value and config thresholds, returns a SecurityState. This isolation
is deliberate: it's the piece most likely to grow (more states, more
nuanced transitions) as additional security algorithms are added, and
it should be trivially unit-testable without spinning up FL or a
quantum simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SecurityState(str, Enum):
    SECURE = "SECURE"
    CAUTION = "CAUTION"
    LOCKDOWN = "LOCKDOWN"


@dataclass(frozen=True)
class StateThresholds:
    """QBER thresholds, as fractions (0.05 = 5%), not percentages."""

    secure_max: float = 0.05      # QBER < secure_max -> SECURE
    caution_max: float = 0.11     # secure_max <= QBER < caution_max -> CAUTION
    # QBER >= caution_max -> LOCKDOWN


@dataclass
class StateTransition:
    qber: float
    previous_state: SecurityState | None
    new_state: SecurityState
    changed: bool


class StateController:
    """
    Stateful wrapper around the classification logic — tracks the
    previous state so callers can detect transitions (useful for
    logging state changes to the dashboard / W&B rather than every
    per-round state value).
    """

    def __init__(self, thresholds: StateThresholds | None = None):
        self.thresholds = thresholds or StateThresholds()
        self._previous_state: SecurityState | None = None

    def classify(self, qber: float) -> SecurityState:
        if not 0.0 <= qber <= 1.0:
            raise ValueError(f"QBER must be in [0, 1], got {qber}")

        if qber < self.thresholds.secure_max:
            return SecurityState.SECURE
        elif qber < self.thresholds.caution_max:
            return SecurityState.CAUTION
        else:
            return SecurityState.LOCKDOWN

    def update(self, qber: float) -> StateTransition:
        new_state = self.classify(qber)
        transition = StateTransition(
            qber=qber,
            previous_state=self._previous_state,
            new_state=new_state,
            changed=new_state != self._previous_state,
        )
        self._previous_state = new_state
        return transition

    @property
    def current_state(self) -> SecurityState | None:
        return self._previous_state
