"""
Three-state QBER-driven aggregation controller.

Pure logic, no dependency on Flower/PyTorch/Qiskit — takes a QBER
value and config thresholds, returns a SecurityState. This isolation
is deliberate: it's the piece most likely to grow (more states, more
nuanced transitions) as additional security algorithms are added, and
it should be trivially unit-testable without spinning up FL or a
quantum simulator.

Hysteresis (optional, off by default): ESCALATION IS IMMEDIATE, only the way OUT is slowed.
- A reading whose memoryless state is higher than the current state switches to it at once, so crossing 0.05
  enters CAUTION and reaching 0.11 enters LOCKDOWN with no delay (the 0.11 constant is never softened).
- The state drops only after `dwell` consecutive readings that are below the current state's entry
  boundary by at least `margin` (i.e. `classify(q + margin)` is lower), and then to the highest such level
  seen in that window. With margin = 0 and dwell = 1 the controller is exactly the memoryless classifier.
Hysteresis buys stability (less flapping from sampling noise), not accuracy, and costs rounds spent in the
higher state.
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

    # The LOCKDOWN boundary is the asymptotic BB84 bound (Shor-Preskill 2000); it may be raised for
    # experiments but never lowered. The CAUTION boundary is a configurable heuristic.
    MIN_LOCKDOWN_THRESHOLD = 0.11

    def __post_init__(self) -> None:
        if not 0.0 < self.secure_max < self.caution_max <= 1.0:
            raise ValueError(
                f"Need 0 < secure_max < caution_max <= 1, got secure_max={self.secure_max}, "
                f"caution_max={self.caution_max}."
            )
        if self.caution_max < self.MIN_LOCKDOWN_THRESHOLD:
            raise ValueError(
                f"caution_max (the LOCKDOWN threshold) may not be configured below "
                f"{self.MIN_LOCKDOWN_THRESHOLD}, got {self.caution_max}."
            )


@dataclass(frozen=True)
class HysteresisConfig:
    """How slowly the controller de-escalates. margin and QBER are fractions (0.01 = 1 percentage point)."""

    margin: float = 0.0   # a reading must be this far below the boundary to count towards leaving a state
    dwell: int = 1        # number of consecutive qualifying readings needed to de-escalate

    def __post_init__(self) -> None:
        if self.margin < 0.0:
            raise ValueError(f"margin must be >= 0, got {self.margin}")
        if self.dwell < 1:
            raise ValueError(f"dwell must be >= 1, got {self.dwell}")


_RANK = {SecurityState.SECURE: 0, SecurityState.CAUTION: 1, SecurityState.LOCKDOWN: 2}


@dataclass
class StateTransition:
    qber: float
    previous_state: SecurityState | None
    new_state: SecurityState
    changed: bool
    raw_state: SecurityState | None = None  # what the memoryless classifier said for this reading
    held: bool = False                      # True if hysteresis kept a higher state than raw_state


class StateController:
    """
    Stateful wrapper around the classification logic — tracks the
    previous state so callers can detect transitions (useful for
    logging state changes to the dashboard / W&B rather than every
    per-round state value). Optionally applies hysteresis on the way out
    of a state (see the module docstring).
    """

    def __init__(self, thresholds: StateThresholds | None = None, hysteresis: HysteresisConfig | None = None):
        self.thresholds = thresholds or StateThresholds()
        self.hysteresis = hysteresis or HysteresisConfig()
        if self.hysteresis.margin >= self.thresholds.secure_max:
            raise ValueError(
                f"hysteresis margin {self.hysteresis.margin} must be smaller than secure_max "
                f"{self.thresholds.secure_max}, otherwise a link could never return to SECURE."
            )
        self._previous_state: SecurityState | None = None
        self._pending: list[SecurityState] = []  # consecutive de-escalation candidates

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
        raw = self.classify(qber)
        previous = self._previous_state

        if previous is None or _RANK[raw] >= _RANK[previous]:
            # First reading, same state, or ESCALATION: take the memoryless state immediately.
            new_state = raw
            self._pending.clear()
        else:
            # The reading is below the current state; count it only if it is below by the margin.
            candidate = self.classify(min(1.0, qber + self.hysteresis.margin))
            if _RANK[candidate] < _RANK[previous]:
                self._pending.append(candidate)
            else:
                self._pending.clear()  # not clearly below: the streak is broken
            if len(self._pending) >= self.hysteresis.dwell:
                recent = self._pending[-self.hysteresis.dwell:]
                new_state = max(recent, key=lambda state: _RANK[state])
                self._pending.clear()
            else:
                new_state = previous

        transition = StateTransition(
            qber=qber,
            previous_state=previous,
            new_state=new_state,
            changed=new_state != previous,
            raw_state=raw,
            held=_RANK[new_state] > _RANK[raw],
        )
        self._previous_state = new_state
        return transition

    def reset(self) -> None:
        """Forget all history (as if no reading had been seen)."""
        self._previous_state = None
        self._pending.clear()

    @property
    def current_state(self) -> SecurityState | None:
        return self._previous_state
