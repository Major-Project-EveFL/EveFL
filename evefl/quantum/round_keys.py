"""
Per-client, per-round key generation on a SEPARATE exchange from the controller's QBER signal (docs/adr/0001).

Under the chosen design the controller keeps its small headline block (n_qubits = 1024, sample fraction 0.25;
detection only) while the key comes from its own, larger block (default 2^17 qubits; the finite-key bound needs
about 10^4 qubits or more, see docs/validation/key_rates.json). The two exchanges use independent random streams
(`purpose`), so the controller's QBER and the key block's QBER are different measurements: the controller can say
SECURE while the key exchange aborts or yields no key, which `evefl.orchestration.policy.apply_key_policy` turns into
a discarded round with reason `no_key` / `key_abort`.

The keys are simulated and seed-determined: NOT secret (see postprocess.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from evefl.quantum.base import ChannelModel
from evefl.quantum.bb84_numpy import BB84NumpyProtocol
from evefl.quantum.postprocess import PostprocessResult, postprocess
from evefl.quantum.seeding import qkd_seed_sequence

DEFAULT_KEY_BLOCK_QUBITS = 2 ** 17
DEFAULT_KEY_SAMPLE_FRACTION = 0.1  # the fraction the key-rate table was measured with


@dataclass(frozen=True)
class KeyBlockConfig:
    n_qubits: int = DEFAULT_KEY_BLOCK_QUBITS
    sample_fraction: float = DEFAULT_KEY_SAMPLE_FRACTION

    def __post_init__(self) -> None:
        if self.n_qubits < 1:
            raise ValueError(f"n_qubits must be >= 1, got {self.n_qubits}")
        if not 0.0 < self.sample_fraction < 1.0:
            raise ValueError(f"sample_fraction must be in (0, 1), got {self.sample_fraction}")


def generate_round_key(
    *,
    experiment_seed: int,
    server_round: int,
    client_id: str,
    channel: ChannelModel,
    auth_key: bytes,
    config: Optional[KeyBlockConfig] = None,
    ec_seed: Optional[int] = None,
) -> PostprocessResult:
    """Run one key exchange for (client, round) on `channel` and post-process it.

    The exchange is independent of the controller's exchange for the same round and client.
    """
    config = config or KeyBlockConfig()
    protocol = BB84NumpyProtocol(
        sample_fraction=config.sample_fraction,
        seed=qkd_seed_sequence(experiment_seed, server_round, client_id, purpose="key"),
    )
    result = protocol.run_exchange(config.n_qubits, channel=channel)
    return postprocess(result, client_id=client_id, round_id=server_round, auth_key=auth_key, ec_seed=ec_seed)
