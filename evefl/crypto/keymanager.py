"""
KeyManager: one link key per client per round, handed out through handles that can be released (docs/adr/0003, step 1).

- A `KeyProvider` produces the raw round key (today the simulated QKD pipeline of docs/adr/0001; a real QKD system with
  an ETSI GS QKD 014 key-delivery API would be another provider, registered by name and chosen in settings.yaml).
- `KeyManager.issue(client, round)` asks the provider at most once per (client, round), even after a failure or a release
  (CLAUDE.md rule 2: never reuse). Its outcome is a status the round policy understands ("ok", "no_key", "abort_*").
- `KeyHandle.release()` overwrites the round key and every derived link key (best effort, see linkkeys.py).
- `ProtectedSender` / `ProtectedReceiver` send and open one message per (client, round, direction). Without a key the
  sender raises BEFORE anything reaches the transport: there is no unprotected fallback.

The simulated keys are seed-determined and NOT secret. Nothing here is a hybrid or post-quantum construction.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional, Protocol, Set, Tuple

from evefl.crypto.base import CipherSuite
from evefl.crypto.classical import crypto_registry
from evefl.crypto.linkkeys import (
    ROLES,
    Direction,
    KeyReuseError,
    LinkKey,
    LinkKeyError,
    SealedMessage,
    components_for,
    derive_link_key,
    open_sealed,
    provenance_tag,
    seal,
)
from evefl.quantum.base import ChannelModel
from evefl.quantum.postprocess import STATUS_OK, PostprocessResult, RoundKey
from evefl.quantum.round_keys import KeyBlockConfig, generate_round_key
from evefl.registry import Registry


class NoKeyError(LinkKeyError):
    """No usable key exists for this (client, round): never issued, the exchange failed, or it was released."""


# --------------------------------------------------------------------------
# Providers
# --------------------------------------------------------------------------

class KeyProvider(ABC):
    """Source of per-client, per-round key material. Must return a fresh key for every call."""

    @abstractmethod
    def generate(self, client_id: str, round_id: int) -> PostprocessResult:
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError


key_provider_registry: Registry = Registry("key_provider")


@key_provider_registry.register("simulated_qkd")
class SimulatedQKDKeyProvider(KeyProvider):
    """The simulated BB84 key pipeline (separate large exchange, docs/adr/0001). Keys are NOT secret."""

    def __init__(
        self,
        *,
        experiment_seed: int,
        channel_for: Callable[[str, int], ChannelModel],
        auth_key: Optional[bytes] = None,
        config: Optional[KeyBlockConfig] = None,
        ec_seed: Optional[int] = None,
    ) -> None:
        self._experiment_seed = experiment_seed
        self._channel_for = channel_for
        # Stand-in for the pre-shared MAC key of the classical channel: random per provider, never a fixed value.
        self._auth_key = auth_key if auth_key is not None else os.urandom(32)
        self.config = config or KeyBlockConfig()
        self._ec_seed = ec_seed

    @property
    def name(self) -> str:
        return "simulated_qkd"

    def generate(self, client_id: str, round_id: int) -> PostprocessResult:
        return generate_round_key(
            experiment_seed=self._experiment_seed, server_round=round_id, client_id=client_id,
            channel=self._channel_for(client_id, round_id), auth_key=self._auth_key, config=self.config,
            ec_seed=self._ec_seed,
        )


# --------------------------------------------------------------------------
# Handles and the manager
# --------------------------------------------------------------------------

class KeyHandle:
    """The round key of one (client, round) and the link keys derived from it, each at most once."""

    def __init__(self, round_key: RoundKey, suite: CipherSuite) -> None:
        self.client_id = round_key.client_id
        self.round_id = round_key.round_id
        self.simulated = round_key.simulated
        self.provenance = provenance_tag(components_for(round_key))
        self._round_key = round_key
        self._suite = suite
        self._taken: Set[Tuple[Direction, str]] = set()
        self._link_keys: list[LinkKey] = []
        self.released = False

    def link_key(self, direction: Direction, role: str) -> LinkKey:
        """The `role` end ("client" or "server") of the key for `direction`; each can be taken once."""
        if self.released:
            raise NoKeyError(f"the key for client {self.client_id!r}, round {self.round_id} was released")
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}, got {role!r}")
        if (direction, role) in self._taken:
            raise KeyReuseError(f"the {role} key for {direction.value} of client {self.client_id!r}, "
                                f"round {self.round_id} was already taken")
        key = derive_link_key(self._round_key, direction=direction, role=role, suite=self._suite)
        self._taken.add((direction, role))
        self._link_keys.append(key)
        return key

    def release(self) -> None:
        """Overwrite the round key and every derived link key (best effort). Idempotent."""
        for key in self._link_keys:
            key.zeroize()
        self._round_key.zeroize()
        self.released = True

    def __repr__(self) -> str:
        return (f"KeyHandle(client={self.client_id!r}, round={self.round_id}, provenance={self.provenance!r}, "
                f"simulated={self.simulated}, released={self.released})")


@dataclass(frozen=True)
class KeyIssue:
    client_id: str
    round_id: int
    status: str            # "ok", "no_key", "abort_*": what apply_key_policy / the round policy consume
    reason: str
    handle: Optional[KeyHandle]

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK and self.handle is not None


class KeyManager:
    def __init__(self, provider: KeyProvider, suite: Optional[CipherSuite] = None) -> None:
        self.provider = provider
        self.suite: CipherSuite = suite if suite is not None else crypto_registry.get("classical")()
        self._issued: Set[Tuple[str, int]] = set()
        self._handles: Dict[Tuple[str, int], KeyHandle] = {}

    def issue(self, client_id: str, round_id: int) -> KeyIssue:
        """Ask the provider for the key of this (client, round). A second request is refused, whatever happened before."""
        pair = (client_id, round_id)
        if pair in self._issued:
            raise KeyReuseError(f"a key was already requested for client {client_id!r}, round {round_id}")
        self._issued.add(pair)  # recorded before the call: a provider failure still uses the pair up
        result = self.provider.generate(client_id, round_id)
        if result.ok and result.round_key is not None:
            handle = KeyHandle(result.round_key, self.suite)
            self._handles[pair] = handle
            return KeyIssue(client_id, round_id, result.status, result.reason, handle)
        return KeyIssue(client_id, round_id, result.status, result.reason, None)

    def handle(self, client_id: str, round_id: int) -> KeyHandle:
        handle = self._handles.get((client_id, round_id))
        if handle is None or handle.released:
            raise NoKeyError(f"no usable key for client {client_id!r}, round {round_id}")
        return handle

    def key_status_fn(self, server_round: int, cid: str) -> str:
        """The `key_status_fn` hook of EveFLStrategy: issues the key for (cid, round) and returns its status."""
        return self.issue(cid, server_round).status

    def release_round(self, round_id: int) -> None:
        for (_cid, rnd), handle in self._handles.items():
            if rnd == round_id:
                handle.release()


# --------------------------------------------------------------------------
# Sending and receiving without a plaintext fallback
# --------------------------------------------------------------------------

class Transport(Protocol):
    def send(self, data: bytes) -> None: ...


class ProtectedSender:
    """Seals and sends one message per (client, round, direction). Raises before the transport is touched if no key exists."""

    def __init__(self, manager: KeyManager, transport: Transport, role: str = "client") -> None:
        self._manager = manager
        self._transport = transport
        self._role = role

    def send(
        self, client_id: str, round_id: int, plaintext: bytes, *, direction: Direction = Direction.CLIENT_TO_SERVER,
        state: str, model_hash: bytes, shapes_digest: bytes,
    ) -> SealedMessage:
        handle = self._manager.handle(client_id, round_id)       # NoKeyError: nothing is sent
        key = handle.link_key(direction, self._role)              # KeyReuseError on a second send
        message = seal(key, plaintext, state=state, model_hash=model_hash, shapes_digest=shapes_digest)
        self._transport.send(message.to_bytes())
        return message


class ProtectedReceiver:
    """Opens link messages. Each (client, round, direction) key is taken once and kept, so a failed attempt does not burn it."""

    def __init__(self, manager: KeyManager, role: str = "server") -> None:
        self._manager = manager
        self._role = role
        self._keys: Dict[Tuple[str, int, Direction], LinkKey] = {}

    def receive(self, data: bytes, *, state: str, model_hash: bytes, shapes_digest: bytes) -> bytes:
        message = SealedMessage.from_bytes(data)
        slot = (message.client_id, message.round_id, message.direction)
        key = self._keys.get(slot)
        if key is None:
            key = self._manager.handle(message.client_id, message.round_id).link_key(message.direction, self._role)
            self._keys[slot] = key
        return open_sealed(key, message, state=state, model_hash=model_hash, shapes_digest=shapes_digest)


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

def create_key_manager(
    settings: Mapping[str, object],
    *,
    experiment_seed: int,
    channel_for: Callable[[str, int], ChannelModel],
    auth_key: Optional[bytes] = None,
) -> KeyManager:
    """Build a KeyManager from the `crypto` section of settings.yaml (`key_provider`, `key_block`, `active_suite`)."""
    section = dict(settings.get("crypto", {}) or {})  # type: ignore[call-overload]
    provider_cls = key_provider_registry.get(str(section.get("key_provider", "simulated_qkd")))
    block = dict(section.get("key_block", {}) or {})
    defaults = KeyBlockConfig()
    config = KeyBlockConfig(
        n_qubits=int(block.get("n_qubits", defaults.n_qubits)),
        sample_fraction=float(block.get("sample_fraction", defaults.sample_fraction)),
    )
    provider = provider_cls(experiment_seed=experiment_seed, channel_for=channel_for, auth_key=auth_key, config=config)
    suite = crypto_registry.get(str(section.get("active_suite", "classical")))()
    return KeyManager(provider, suite)
