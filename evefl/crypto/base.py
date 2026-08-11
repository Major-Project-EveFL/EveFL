"""
Abstract interface for cipher suites used to protect model updates.

classical.py implements the current stack (SHA-256 -> HKDF-SHA256 ->
AES-256-GCM). The stretch-goal Kyber/ML-KEM fallback implements the
same interface so orchestration code never branches on which crypto
backend is active.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class EncryptedPayload:
    ciphertext: bytes
    nonce: bytes
    tag: bytes | None = None  # None if the cipher embeds the tag in ciphertext


class CipherSuite(ABC):
    """Base class every cipher suite implementation must satisfy."""

    @abstractmethod
    def derive_key(self, key_material: bytes, *, info: bytes = b"", length: int = 32) -> bytes:
        """Derive a symmetric key from raw key material (e.g. a QKD sifted key)."""
        raise NotImplementedError

    @abstractmethod
    def encrypt(self, plaintext: bytes, key: bytes, *, associated_data: bytes = b"") -> EncryptedPayload:
        raise NotImplementedError

    @abstractmethod
    def decrypt(self, payload: EncryptedPayload, key: bytes, *, associated_data: bytes = b"") -> bytes:
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError
