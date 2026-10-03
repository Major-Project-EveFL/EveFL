"""
Round policy: turn per-link QBER readings (and key outcomes) into a decision for the round.

Pure logic, no imports from the quantum, ML or FL stacks (the controller must stay unaware of key length).
Inputs are plain numbers and strings, so this can run anywhere.

Three modes, selected by name (registry) and configured from YAML (`orchestration.policy`):

  global_binary  QKDFL-style baseline (docs/06 B2): pause the round when the WORST link reaches the LOCKDOWN
                 threshold, no CAUTION response, no hysteresis.
  global         EveFL with one global three-state controller on the worst link (B3): LOCKDOWN discards the
                 round, CAUTION puts every client on FedProx.
  per_client     Each link has its own controller and state (B4): a LOCKDOWN link is excluded and the rest
                 aggregate with renormalised weights; below `min_clients` the round is discarded.

Key outcomes (docs/adr/0001): a client whose key generation failed cannot protect its update, so it is excluded
in `per_client` mode (reason `no_key` / `key_abort`) and the round is discarded in the global modes. These reasons
are kept separate from QBER exclusions in every decision, for the logs.

Selective exclusion is an attack lever: in `per_client` mode an adversary who disturbs one link chooses which
hospital drops out of training (docs/03). Participation counts are kept so experiments can report them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Tuple

from evefl.orchestration.state_machine import (
    HysteresisConfig,
    SecurityState,
    StateController,
    StateThresholds,
)
from evefl.registry import Registry

KEY_OK = "ok"
MIN_CLIENTS_FOR_SCREENING = 3  # median/MAD screening is inactive below this many updates (evefl.fl.screening)

REASON_QBER = "qber_lockdown"            # the whole round discarded because of QBER
REASON_QBER_EXCLUSION = "qber_exclusion"  # one client excluded because of its own QBER
REASON_NO_KEY = "no_key"
REASON_KEY_ABORT = "key_abort"
REASON_TOO_FEW = "too_few_clients"       # fewer than min_clients left after exclusions
REASON_NONE = "none"

policy_registry: Registry = Registry("round_policy")


class ClientAction(str, Enum):
    TRAIN = "TRAIN"              # plain local training (SECURE)
    TRAIN_PROX = "TRAIN_PROX"    # FedProx and screening (CAUTION)
    EXCLUDE = "EXCLUDE"          # not sent the model, not aggregated, even if an update arrives


@dataclass(frozen=True)
class ClientDecision:
    state: SecurityState         # the client's own link state (after hysteresis)
    action: ClientAction
    reason: str = REASON_NONE    # why it is excluded, if it is


@dataclass(frozen=True)
class RoundDecision:
    mode: str
    global_state: SecurityState                  # the most severe link state this round
    clients: Mapping[str, ClientDecision]
    discard_round: bool                          # keep the last good model, aggregate nothing
    reason: str                                  # round-level reason (REASON_NONE if the round proceeds)
    detail: str = ""
    participation: Mapping[str, int] = field(default_factory=dict)  # rounds each client has been included so far

    @property
    def included(self) -> List[str]:
        """Clients that train and are aggregated (empty if the round is discarded)."""
        if self.discard_round:
            return []
        return sorted(cid for cid, d in self.clients.items() if d.action != ClientAction.EXCLUDE)

    @property
    def excluded(self) -> List[str]:
        return sorted(cid for cid, d in self.clients.items() if d.action == ClientAction.EXCLUDE)

    @property
    def exclusion_reasons(self) -> Dict[str, List[str]]:
        """reason -> client ids, so QBER exclusions and key failures stay distinguishable in the logs."""
        out: Dict[str, List[str]] = {}
        for cid in self.excluded:
            out.setdefault(self.clients[cid].reason, []).append(cid)
        return out

    @property
    def screening_possible(self) -> bool:
        """Median/MAD screening needs at least 3 updates; per-client exclusion can push it below that."""
        return len(self.included) >= MIN_CLIENTS_FOR_SCREENING

    @property
    def clip_clients(self) -> Tuple[str, ...]:
        """Clients whose update may be clipped: only included clients whose OWN state is CAUTION, and only if
        screening is possible (the statistics are computed over all included updates)."""
        if not self.screening_possible:
            return ()
        return tuple(cid for cid in self.included if self.clients[cid].state == SecurityState.CAUTION)


@dataclass(frozen=True)
class PolicyConfig:
    mode: str = "global"
    thresholds: StateThresholds = field(default_factory=StateThresholds)
    hysteresis: HysteresisConfig = field(default_factory=HysteresisConfig)
    min_clients: int = 2

    def __post_init__(self) -> None:
        if self.min_clients < 1:
            raise ValueError(f"min_clients must be >= 1, got {self.min_clients}")

    @classmethod
    def from_mapping(cls, raw: Optional[Mapping[str, Any]]) -> "PolicyConfig":
        """Build from the `orchestration` section of settings.yaml (missing keys use the defaults)."""
        raw = dict(raw or {})
        thresholds = StateThresholds(**dict(raw.get("thresholds") or {}))
        hysteresis = HysteresisConfig(**dict(raw.get("hysteresis") or {}))
        return cls(
            mode=str(raw.get("mode", "global")),
            thresholds=thresholds,
            hysteresis=hysteresis,
            min_clients=int(raw.get("min_clients", 2)),
        )


def _combine_key_failures(key_status: Mapping[str, str]) -> Tuple[List[str], str]:
    """Failed client ids and a round-level reason (`no_key` only if every failure is a plain no_key)."""
    failed = sorted(cid for cid, status in key_status.items() if status != KEY_OK)
    statuses = {key_status[cid] for cid in failed}
    reason = REASON_NO_KEY if statuses == {"no_key"} else REASON_KEY_ABORT
    return failed, reason


def _severity(state: SecurityState) -> int:
    return list(SecurityState).index(state)


def _key_reason(status: str) -> str:
    return REASON_NO_KEY if status == "no_key" else REASON_KEY_ABORT


class RoundPolicy(ABC):
    """Stateful per-experiment policy: call `decide` once per round with every link's QBER."""

    mode: str = ""

    def __init__(self, config: PolicyConfig):
        self.config = config
        self.participation: Dict[str, int] = {}

    @abstractmethod
    def decide(
        self, qbers: Mapping[str, float], key_status: Optional[Mapping[str, str]] = None
    ) -> RoundDecision:
        """`qbers`: QBER of EVERY client's link (excluded clients too: they must keep being measured to rejoin).
        `key_status`: client id -> post-processing status ("ok", "no_key", "abort_*"), or None if keys are not
        required this round."""

    def _finish(self, decision: RoundDecision) -> RoundDecision:
        if not decision.discard_round:
            for cid in decision.included:
                self.participation[cid] = self.participation.get(cid, 0) + 1
        return RoundDecision(
            decision.mode, decision.global_state, decision.clients, decision.discard_round, decision.reason,
            decision.detail, dict(self.participation),
        )

    @staticmethod
    def _require_readings(qbers: Mapping[str, float]) -> None:
        if not qbers:
            raise ValueError("decide() needs at least one client reading")


@policy_registry.register("global_binary")
class GlobalBinaryPolicy(RoundPolicy):
    """QKDFL-style: pause at the LOCKDOWN threshold, otherwise train everyone normally (no CAUTION response)."""

    mode = "global_binary"

    def decide(
        self, qbers: Mapping[str, float], key_status: Optional[Mapping[str, str]] = None
    ) -> RoundDecision:
        self._require_readings(qbers)
        ks: Mapping[str, str] = key_status or {}
        worst = max(qbers.values())
        paused = worst >= self.config.thresholds.caution_max
        state = SecurityState.LOCKDOWN if paused else SecurityState.SECURE
        action = ClientAction.EXCLUDE if paused else ClientAction.TRAIN
        clients = {cid: ClientDecision(state, action, REASON_QBER if paused else REASON_NONE) for cid in qbers}
        failed, reason = _combine_key_failures(ks)
        if paused:
            return self._finish(RoundDecision(self.mode, state, clients, True, REASON_QBER,
                                              f"worst QBER {worst:.4f} >= {self.config.thresholds.caution_max}"))
        if failed:
            return self._finish(RoundDecision(self.mode, SecurityState.LOCKDOWN, clients, True, reason,
                                              ", ".join(f"{cid}:{ks[cid]}" for cid in failed)))
        return self._finish(RoundDecision(self.mode, state, clients, False, REASON_NONE))


@policy_registry.register("global")
class GlobalPolicy(RoundPolicy):
    """One three-state controller on the worst link (with hysteresis if configured)."""

    mode = "global"

    def __init__(self, config: PolicyConfig):
        super().__init__(config)
        self._controller = StateController(config.thresholds, config.hysteresis)

    def decide(
        self, qbers: Mapping[str, float], key_status: Optional[Mapping[str, str]] = None
    ) -> RoundDecision:
        self._require_readings(qbers)
        ks: Mapping[str, str] = key_status or {}
        state = self._controller.update(max(qbers.values())).new_state
        failed, reason = _combine_key_failures(ks)

        if state == SecurityState.LOCKDOWN:
            clients = {cid: ClientDecision(state, ClientAction.EXCLUDE, REASON_QBER) for cid in qbers}
            return self._finish(RoundDecision(self.mode, state, clients, True, REASON_QBER,
                                              "QBER at or above the LOCKDOWN threshold"))
        action = ClientAction.TRAIN_PROX if state == SecurityState.CAUTION else ClientAction.TRAIN
        if failed:
            clients = {cid: ClientDecision(state, ClientAction.EXCLUDE, _key_reason(ks[cid]))
                       if cid in failed else ClientDecision(state, action) for cid in qbers}
            return self._finish(RoundDecision(self.mode, SecurityState.LOCKDOWN, clients, True, reason,
                                              ", ".join(f"{cid}:{ks[cid]}" for cid in failed)))
        return self._finish(RoundDecision(self.mode, state, {cid: ClientDecision(state, action) for cid in qbers},
                                          False, REASON_NONE))


@policy_registry.register("per_client")
class PerClientPolicy(RoundPolicy):
    """One controller per link: exclude LOCKDOWN links, keep the rest; discard below `min_clients`."""

    mode = "per_client"

    def __init__(self, config: PolicyConfig):
        super().__init__(config)
        self._controllers: Dict[str, StateController] = {}

    def decide(
        self, qbers: Mapping[str, float], key_status: Optional[Mapping[str, str]] = None
    ) -> RoundDecision:
        self._require_readings(qbers)
        ks: Mapping[str, str] = key_status or {}
        clients: Dict[str, ClientDecision] = {}
        for cid, q in qbers.items():
            controller = self._controllers.setdefault(
                cid, StateController(self.config.thresholds, self.config.hysteresis))
            state = controller.update(q).new_state  # every link is measured every round, excluded ones too
            if state == SecurityState.LOCKDOWN:
                clients[cid] = ClientDecision(state, ClientAction.EXCLUDE, REASON_QBER_EXCLUSION)
            elif ks.get(cid, KEY_OK) != KEY_OK:
                clients[cid] = ClientDecision(state, ClientAction.EXCLUDE, _key_reason(ks[cid]))
            elif state == SecurityState.CAUTION:
                clients[cid] = ClientDecision(state, ClientAction.TRAIN_PROX)
            else:
                clients[cid] = ClientDecision(state, ClientAction.TRAIN)

        worst = max((d.state for d in clients.values()), key=_severity)
        remaining = sum(1 for d in clients.values() if d.action != ClientAction.EXCLUDE)
        if remaining < self.config.min_clients:
            return self._finish(RoundDecision(
                self.mode, worst, clients, True, REASON_TOO_FEW,
                f"{remaining} client(s) left, need {self.config.min_clients}"))
        return self._finish(RoundDecision(self.mode, worst, clients, False, REASON_NONE))


def create_policy(config: PolicyConfig) -> RoundPolicy:
    """Instantiate the policy named by `config.mode`."""
    return policy_registry.get(config.mode)(config)


# ---------------------------------------------------------------------------
# Simple two-input helper kept from the first version (QBER state + key status -> effective state)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PolicyDecision:
    state: SecurityState
    reason: str                    # one of the REASON_* constants
    failed_clients: List[str] = field(default_factory=list)
    detail: str = ""

    @property
    def discard_round(self) -> bool:
        return self.state == SecurityState.LOCKDOWN


def apply_key_policy(state: SecurityState, key_status: Mapping[str, str]) -> PolicyDecision:
    """Effective state for a GLOBAL round, given the QBER state and per-client key statuses.

    A round in which any client could not obtain a key is treated like LOCKDOWN (discard, keep the last good
    model) with a reason separate from a QBER lockdown. An empty mapping means keys are not required.
    """
    failed = sorted(cid for cid, status in key_status.items() if status != KEY_OK)

    if state == SecurityState.LOCKDOWN:
        return PolicyDecision(state, REASON_QBER, failed, "QBER at or above the LOCKDOWN threshold")
    if not failed:
        return PolicyDecision(state, REASON_NONE)

    statuses = {key_status[cid] for cid in failed}
    reason = REASON_NO_KEY if statuses == {"no_key"} else REASON_KEY_ABORT
    return PolicyDecision(
        SecurityState.LOCKDOWN, reason, failed,
        ", ".join(f"{cid}:{key_status[cid]}" for cid in failed),
    )
