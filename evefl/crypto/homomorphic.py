"""
Abstract interface for homomorphic aggregation engines.

This is deliberately a separate interface from CipherSuite (base.py):
CipherSuite protects a message in transit (encrypt -> send -> decrypt
the *same* message). A HomomorphicAggregator protects a value while
letting the server compute *on* the ciphertexts themselves (encrypt N
gradients -> sum the ciphertexts -> decrypt only the sum) without ever
seeing the individual plaintext gradients. Different job, different
interface, kept swappable the same way as everything else in evefl.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class HomomorphicContext:
    """Holds whatever key material / scheme parameters an implementation
    needs. Opaque outside the implementation that created it."""

    public_material: Any
    secret_material: Any | None  # None if this context is server-side (public-key only)


class HomomorphicAggregator(ABC):
    """Base class every homomorphic aggregation implementation must satisfy."""

    @abstractmethod
    def create_context(self) -> HomomorphicContext:
        """Generate fresh key material / scheme parameters. Called once
        per FL session (or per round, depending on rekeying policy)."""
        raise NotImplementedError

    @abstractmethod
    def encrypt_vector(self, vector: list[float], context: HomomorphicContext) -> bytes:
        """Encrypt a flattened gradient vector, returning a serialized
        ciphertext the client can transmit to the server."""
        raise NotImplementedError

    @abstractmethod
    def aggregate(self, ciphertexts: list[bytes], weights: list[float], context: HomomorphicContext) -> bytes:
        """Server-side: homomorphically compute the weighted sum of
        ciphertexts, without decrypting any individual one. Returns a
        serialized aggregate ciphertext."""
        raise NotImplementedError

    @abstractmethod
    def decrypt_vector(self, ciphertext: bytes, context: HomomorphicContext) -> list[float]:
        """Decrypt an (aggregate) ciphertext back to a plaintext vector.
        Requires secret_material in context — i.e. only the party
        holding the secret key can call this, never the server."""
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError