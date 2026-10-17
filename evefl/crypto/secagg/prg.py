"""
Mask PRG for pairwise-masked secure aggregation (ADR 0003, amendment 4: no counter or nonce is ever reused).

For each unordered pair {i, j}, round r and floating-point tensor index t:

    K = HKDF-SHA256(ikm = seed_ij, salt = none,
                    info = lp("evefl/v1/mask-stream") || u64(r) || lp(min(i, j)) || lp(max(i, j)) || u32(t), length = 32)

and the stream is AES-256 in CTR mode under K with an all-zero 128-bit initial counter block; element k of the tensor is the
little-endian uint64 in keystream bytes [8k, 8k+8). Different (pair, round, tensor) give different keys and `seed_ij` is new every
round, so no (key, counter) pair occurs twice and the fixed zero start is safe. Ids are ordered by their UTF-8 bytes.

The two partners expand the SAME stream on purpose (that is what makes the masks cancel): the smaller id adds the mask, the larger
subtracts it. The stream encrypts nothing; each partner uses it once, for one tensor of one round.

Enforced in code: a `MaskStreams` object serves each tensor index once (a second request raises `StreamReuseError`), and a
`RoundGuard` lets a round id begin at most once and never resumes an aborted attempt (`RoundReuseError`). Only `cryptography`
primitives are used; there is no non-cryptographic randomness in this package.
"""

from __future__ import annotations

from typing import Dict, Optional, Set

import numpy as np
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from evefl.crypto.secagg import fixedpoint as fp
from evefl.crypto.secagg.encoding import lp, u32, u64

STREAM_LABEL = b"evefl/v1/mask-stream"
SEED_BYTES = 32
KEY_BYTES = 32


class MaskStreamError(Exception):
    """Base class for mask-stream misuse."""


class StreamReuseError(MaskStreamError):
    """The same (pair, round, tensor) stream was requested twice."""


class RoundReuseError(MaskStreamError):
    """A round id was begun twice, an aborted round was resumed, or a round was ended that never began."""


def _ordered(client_a: str, client_b: str) -> tuple[bytes, bytes]:
    a, b = client_a.encode("utf-8"), client_b.encode("utf-8")
    if a == b:
        raise ValueError("a mask pair needs two different clients")
    return (a, b) if a < b else (b, a)


def stream_info(*, round_id: int, client_a: str, client_b: str, tensor_index: int) -> bytes:
    """The HKDF `info` of one stream (symmetric in the pair)."""
    low, high = _ordered(client_a, client_b)
    return lp(STREAM_LABEL) + u64(round_id) + lp(low) + lp(high) + u32(tensor_index)


def stream_key(seed: bytes, *, round_id: int, client_a: str, client_b: str, tensor_index: int) -> bytes:
    """The 32-byte AES key of one (pair, round, tensor) stream, derived from the 32-byte pairwise seed."""
    if len(seed) != SEED_BYTES:
        raise ValueError(f"the pairwise seed must be {SEED_BYTES} bytes")
    info = stream_info(round_id=round_id, client_a=client_a, client_b=client_b, tensor_index=tensor_index)
    return HKDF(algorithm=hashes.SHA256(), length=KEY_BYTES, salt=None, info=info).derive(seed)


def expand(key: bytes, n_elements: int) -> np.ndarray:
    """`n_elements` uint64 words of the AES-256-CTR keystream under `key`, counter starting at zero (little-endian words)."""
    if len(key) != KEY_BYTES:
        raise ValueError(f"the stream key must be {KEY_BYTES} bytes")
    if n_elements < 0:
        raise ValueError("n_elements must not be negative")
    if n_elements == 0:
        return np.zeros(0, dtype=np.uint64)
    encryptor = Cipher(algorithms.AES(key), modes.CTR(bytes(16))).encryptor()
    keystream = encryptor.update(bytes(8 * n_elements)) + encryptor.finalize()
    return np.frombuffer(keystream, dtype="<u8").astype(np.uint64)


class MaskStreams:
    """The streams one participant `me` shares with one `other` in one round; each tensor index is served at most once."""

    def __init__(self, seed: bytes, *, round_id: int, me: str, other: str):
        _ordered(me, other)   # validates that the two ids differ
        if len(seed) != SEED_BYTES:
            raise ValueError(f"the pairwise seed must be {SEED_BYTES} bytes")
        self._seed = bytes(seed)
        self.round_id = round_id
        self.me, self.other = me, other
        self._taken: Set[int] = set()

    def _consume(self, tensor_index: int, n_elements: int) -> np.ndarray:
        if tensor_index in self._taken:
            raise StreamReuseError(
                f"stream (round {self.round_id}, pair {self.me}/{self.other}, tensor {tensor_index}) was already consumed")
        key = stream_key(self._seed, round_id=self.round_id, client_a=self.me, client_b=self.other, tensor_index=tensor_index)
        words = expand(key, n_elements)
        self._taken.add(tensor_index)
        return words

    def take(self, tensor_index: int, n_elements: int) -> np.ndarray:
        """The raw mask words for one tensor (the same on both partners)."""
        return self._consume(tensor_index, n_elements)

    def signed_mask(self, tensor_index: int, n_elements: int) -> np.ndarray:
        """The mask this participant adds: +mask if `me` is the smaller id, -mask (mod 2^64) otherwise."""
        words = self._consume(tensor_index, n_elements)
        if self.me.encode("utf-8") < self.other.encode("utf-8"):
            return words
        return fp.sub(np.zeros_like(words), words)


class RoundGuard:
    """A round id may begin once; an aborted attempt is never resumed (a retry is a new round with new ephemeral keys)."""

    def __init__(self) -> None:
        self._state: Dict[int, str] = {}

    def begin(self, round_id: int) -> None:
        if not 0 <= round_id < 2 ** 64:
            raise ValueError("round id out of range")
        if round_id in self._state:
            raise RoundReuseError(f"round {round_id} was already used for a secure-aggregation attempt ({self._state[round_id]})")
        self._state[round_id] = "begun"

    def abort(self, round_id: int) -> None:
        if self._state.get(round_id) != "begun":
            raise RoundReuseError(f"round {round_id} has not begun, so it cannot be aborted")
        self._state[round_id] = "aborted"

    def state(self, round_id: int) -> Optional[str]:
        return self._state.get(round_id)
