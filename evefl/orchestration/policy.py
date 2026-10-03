"""
Round policy: combine the QBER state with the outcome of key generation.

Pure logic, no imports from the quantum, ML or FL stacks (the controller must stay unaware of key length).
Inputs are plain strings, so this can run anywhere.

Rule (decided with the team): a round in which any client could not obtain a key (`no_key`, or an
error-correction / authentication abort) is treated like LOCKDOWN: the round is discarded and the last good
model kept. The reason is recorded SEPARATELY from a QBER-driven lockdown so the logs and the paper can tell
"the channel looked attacked" apart from "the block was too small / post-processing failed".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Mapping

from evefl.orchestration.state_machine import SecurityState

KEY_OK = "ok"

REASON_QBER = "qber_lockdown"
REASON_NO_KEY = "no_key"
REASON_KEY_ABORT = "key_abort"
REASON_NONE = "none"


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
    """Effective state for the round, given the QBER state and per-client key statuses.

    `key_status` maps client id -> status string from the post-processing ("ok", "no_key",
    "abort_qber_high", "abort_ec_verification", "abort_auth", ...). An empty mapping means keys are not
    required this round (the policy then leaves the QBER state unchanged).
    """
    failed = sorted(cid for cid, status in key_status.items() if status != KEY_OK)

    if state == SecurityState.LOCKDOWN:
        # Already discarded because of QBER; keep that as the reason even if keys also failed.
        return PolicyDecision(state, REASON_QBER, failed, "QBER at or above the LOCKDOWN threshold")
    if not failed:
        return PolicyDecision(state, REASON_NONE)

    statuses = {key_status[cid] for cid in failed}
    reason = REASON_NO_KEY if statuses == {"no_key"} else REASON_KEY_ABORT
    return PolicyDecision(
        SecurityState.LOCKDOWN, reason, failed,
        ", ".join(f"{cid}:{key_status[cid]}" for cid in failed),
    )
