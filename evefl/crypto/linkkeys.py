"""
Link protection between one client and the server, one key per client, round and direction (docs/adr/0003, layer 1).

What this layer is and is not
- It protects the client <-> server LINK against outsiders: confidentiality and integrity of a message, bound to the
  round, the client, the direction, the security state and the model it refers to. The server holds the other end of
  every key, so this layer alone does NOT make the server blind to an update (that is the aggregation layer).
- In this repository the QKD key comes from a simulator: it is seed-determined and NOT secret. Nothing here claims to
  protect real data, and no hybrid or post-quantum property is claimed. The combiner is built so that a KEM shared
  secret can be added later as another labelled component without touching any caller.
- Zeroization is best effort: our own key buffers are `bytearray`s that are overwritten in place. Python cannot
  guarantee that no other copy exists (immutable `bytes` made inside library calls, interpreter caches), and key
  objects held inside OpenSSL cannot be overwritten from Python.

Encodings
Every variable-length field is prefixed with its 4-byte big-endian length and every integer has a fixed width, so the
`info` string of the key derivation and the associated data of the cipher are injective: two different
(label, direction, round, client, ...) tuples can never produce the same bytes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np
from cryptography.exceptions import InvalidTag

from evefl.crypto.base import CipherSuite, EncryptedPayload
from evefl.quantum.postprocess import RoundKey

INFO_LABEL = b"evefl/v1/aes-gcm"
AAD_LABEL = b"evefl/v1/aad"
WIRE_LABEL = b"evefl/v1/message"
KEY_LENGTH_BYTES = 32
ROLES = ("client", "server")


class Direction(str, Enum):
    CLIENT_TO_SERVER = "client_to_server"
    SERVER_TO_CLIENT = "server_to_client"


class LinkKeyError(Exception):
    """Base class of every error raised by the link-key layer."""


class KeyReuseError(LinkKeyError):
    """A second key, or a second message, for the same (client, round, direction)."""


class KeyZeroizedError(LinkKeyError):
    """The key material was already overwritten."""


class LinkAuthError(LinkKeyError):
    """Authentication failed: tampered ciphertext or nonce, or wrong associated data or key."""


class HeaderMismatchError(LinkKeyError):
    """The message header names another round, client or direction than the key it is opened with."""


# --------------------------------------------------------------------------
# Injective encodings
# --------------------------------------------------------------------------

def _lp(data: bytes) -> bytes:
    if len(data) >= 2 ** 32:
        raise ValueError("field too long to encode")
    return len(data).to_bytes(4, "big") + data


def _u64(value: int) -> bytes:
    if not 0 <= value < 2 ** 64:
        raise ValueError(f"integer out of range for a 64-bit field: {value}")
    return value.to_bytes(8, "big")


def link_info(direction: Direction, round_id: int, client_id: str) -> bytes:
    """HKDF `info`: label, direction, round id and client id (CLAUDE.md rule 2)."""
    return _lp(INFO_LABEL) + _lp(direction.value.encode("ascii")) + _u64(round_id) + _lp(client_id.encode("utf-8"))


def link_aad(
    *, round_id: int, client_id: str, direction: Direction, state: str, model_hash: bytes, shapes_digest: bytes
) -> bytes:
    """AEAD associated data: round, client, direction, security state, model hash and tensor-shape digest."""
    return (_lp(AAD_LABEL) + _u64(round_id) + _lp(client_id.encode("utf-8")) + _lp(direction.value.encode("ascii"))
            + _lp(state.encode("utf-8")) + _lp(model_hash) + _lp(shapes_digest))


def shapes_digest(arrays: Sequence[np.ndarray]) -> bytes:
    """SHA-256 over the number of tensors and each tensor's dtype and shape (not its values)."""
    digest = hashlib.sha256()
    digest.update(_u64(len(arrays)))
    for array in arrays:
        digest.update(_lp(array.dtype.str.encode("ascii")))
        digest.update(_u64(array.ndim))
        for dim in array.shape:
            digest.update(_u64(dim))
    return digest.digest()


def model_digest(arrays: Sequence[np.ndarray]) -> bytes:
    """SHA-256 over dtypes, shapes and values of the tensors: identifies the model an update refers to."""
    digest = hashlib.sha256()
    digest.update(shapes_digest(arrays))
    for array in arrays:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.digest()


# --------------------------------------------------------------------------
# Key combiner (KEM-ready)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class KeyComponent:
    """One labelled secret that feeds the key derivation (today only "qkd"; a KEM secret could be added later)."""

    label: str
    secret: bytes | bytearray


def _wipe(buffer: bytearray) -> None:
    """Overwrite a bytearray in place (same length, so the same memory is reused)."""
    buffer[:] = bytes(len(buffer))


def combine_key_material(components: Sequence[KeyComponent]) -> bytearray:
    """Length-prefixed, labelled concatenation of the components, as a `bytearray` the caller must wipe."""
    if len(components) < 1:
        raise ValueError("combine_key_material needs at least one component")
    labels = [c.label for c in components]
    if len(set(labels)) != len(labels):
        raise ValueError(f"duplicate component label in {labels}")
    out = bytearray(_u64(len(components)))
    for component in components:
        if len(component.secret) == 0:
            raise ValueError(f"empty secret for component {component.label!r}")
        out += _lp(component.label.encode("ascii"))
        out += len(component.secret).to_bytes(4, "big")
        out += component.secret
    return out


def provenance_tag(components: Sequence[KeyComponent]) -> str:
    """"qkd-only" for the single-component case, otherwise the labels joined by "+". A label, not a security claim."""
    labels = [c.label for c in components]
    return "qkd-only" if labels == ["qkd"] else "+".join(labels)


def components_for(round_key: RoundKey) -> list[KeyComponent]:
    return [KeyComponent("qkd", round_key.key)]


# --------------------------------------------------------------------------
# Link keys and sealed messages
# --------------------------------------------------------------------------

class LinkKey:
    """One end (`role`) of the key for one (client, round, direction). Seals at most one message and opens at most one."""

    def __init__(
        self, *, client_id: str, round_id: int, direction: Direction, role: str, key: bytearray, suite: CipherSuite,
        provenance: str, simulated: bool,
    ) -> None:
        self.client_id = client_id
        self.round_id = round_id
        self.direction = direction
        self.role = role
        self.suite = suite
        self.provenance = provenance
        self.simulated = simulated
        self._key = key
        self._sealed = False
        self._opened = False
        self.zeroized = False

    def is_zero(self) -> bool:
        return not any(self._key)

    def zeroize(self) -> None:
        """Best effort: overwrite our own buffer in place."""
        _wipe(self._key)
        self.zeroized = True

    def __repr__(self) -> str:  # never print key material
        return (f"LinkKey(client={self.client_id!r}, round={self.round_id}, direction={self.direction.value!r}, "
                f"role={self.role!r}, provenance={self.provenance!r}, simulated={self.simulated}, "
                f"zeroized={self.zeroized})")


def derive_link_key(round_key: RoundKey, *, direction: Direction, role: str, suite: CipherSuite) -> LinkKey:
    """HKDF-SHA256 over the combined key material, salted with the transcript hash and bound to direction, round, client."""
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}, got {role!r}")
    if round_key.zeroized:
        raise KeyZeroizedError("the round key was already zeroized")
    components = components_for(round_key)
    ikm = combine_key_material(components)
    try:
        derived = suite.derive_key(
            ikm, info=link_info(direction, round_key.round_id, round_key.client_id), length=KEY_LENGTH_BYTES,
            salt=round_key.transcript_sha256 or None,
        )
    finally:
        _wipe(ikm)
    return LinkKey(
        client_id=round_key.client_id, round_id=round_key.round_id, direction=direction, role=role,
        key=bytearray(derived), suite=suite, provenance=provenance_tag(components), simulated=round_key.simulated,
    )


@dataclass(frozen=True)
class SealedMessage:
    """What travels on the link: a visible header (bound into the associated data) plus nonce and ciphertext."""

    round_id: int
    client_id: str
    direction: Direction
    nonce: bytes
    ciphertext: bytes

    def to_bytes(self) -> bytes:
        return (_lp(WIRE_LABEL) + _u64(self.round_id) + _lp(self.client_id.encode("utf-8"))
                + _lp(self.direction.value.encode("ascii")) + _lp(self.nonce) + _lp(self.ciphertext))

    @classmethod
    def from_bytes(cls, data: bytes) -> SealedMessage:
        offset = 0

        def take(n: int) -> bytes:
            nonlocal offset
            if n < 0 or offset + n > len(data):
                raise ValueError("truncated message")
            chunk = data[offset:offset + n]
            offset += n
            return chunk

        def take_lp() -> bytes:
            return take(int.from_bytes(take(4), "big"))

        if take_lp() != WIRE_LABEL:
            raise ValueError("not an EveFL link message")
        round_id = int.from_bytes(take(8), "big")
        client_id = take_lp().decode("utf-8")
        direction = Direction(take_lp().decode("ascii"))
        nonce, ciphertext = take_lp(), take_lp()
        if offset != len(data):
            raise ValueError("trailing bytes after the message")
        return cls(round_id, client_id, direction, nonce, ciphertext)


def _require_live(key: LinkKey) -> None:
    if key.zeroized:
        raise KeyZeroizedError("this link key was zeroized")


def seal(key: LinkKey, plaintext: bytes, *, state: str, model_hash: bytes, shapes_digest: bytes) -> SealedMessage:
    """Encrypt one message. A key seals at most once; a second call raises KeyReuseError."""
    _require_live(key)
    if key._sealed:
        raise KeyReuseError(f"{key!r} already sealed a message")
    aad = link_aad(round_id=key.round_id, client_id=key.client_id, direction=key.direction, state=state,
                   model_hash=model_hash, shapes_digest=shapes_digest)
    payload = key.suite.encrypt(plaintext, key._key, associated_data=aad)
    key._sealed = True
    return SealedMessage(key.round_id, key.client_id, key.direction, payload.nonce, payload.ciphertext)


def open_sealed(
    key: LinkKey, message: SealedMessage, *, state: str, model_hash: bytes, shapes_digest: bytes
) -> bytes:
    """Decrypt one message. The header must match the key; a key opens at most one message (a replay raises KeyReuseError)."""
    _require_live(key)
    if (message.round_id, message.client_id, message.direction) != (key.round_id, key.client_id, key.direction):
        raise HeaderMismatchError(
            f"message header (round {message.round_id}, client {message.client_id!r}, {message.direction.value}) "
            f"does not match {key!r}")
    if key._opened:
        raise KeyReuseError(f"{key!r} already opened a message")
    aad = link_aad(round_id=key.round_id, client_id=key.client_id, direction=key.direction, state=state,
                   model_hash=model_hash, shapes_digest=shapes_digest)
    try:
        plaintext = key.suite.decrypt(EncryptedPayload(ciphertext=message.ciphertext, nonce=message.nonce), key._key,
                                      associated_data=aad)
    except (InvalidTag, ValueError) as exc:
        raise LinkAuthError("authentication failed: tampered message, wrong key or wrong associated data") from exc
    key._opened = True  # only a message that authenticated uses the key up
    return plaintext
