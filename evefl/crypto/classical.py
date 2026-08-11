"""
Classical cipher suite: HKDF-SHA256 key derivation + AES-256-GCM AEAD.

This is the default cipher suite for EveFL. Key material typically
comes from a BB84 sifted key (see evefl.quantum), converted to bytes
upstream before being passed to derive_key().
"""

from __future__ import annotations

import os

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from evefl.crypto.base import CipherSuite, EncryptedPayload
from evefl.registry import Registry

crypto_registry: Registry = Registry("cipher_suite")

NONCE_SIZE_BYTES = 12  # standard for AES-GCM


@crypto_registry.register("classical")
class ClassicalCipherSuite(CipherSuite):
    @property
    def name(self) -> str:
        return "classical-sha256-hkdf-aesgcm"

    def derive_key(self, key_material: bytes, *, info: bytes = b"evefl-round-key", length: int = 32) -> bytes:
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=length,
            salt=None,  # no salt: key_material (QKD-derived) is already high-entropy
            info=info,
        )
        return hkdf.derive(key_material)

    def encrypt(self, plaintext: bytes, key: bytes, *, associated_data: bytes = b"") -> EncryptedPayload:
        aesgcm = AESGCM(key)
        nonce = os.urandom(NONCE_SIZE_BYTES)
        ciphertext = aesgcm.encrypt(nonce, plaintext, associated_data or None)
        # AESGCM appends the 16-byte tag to ciphertext; we keep it combined
        # and just track nonce separately for clarity/interface consistency.
        return EncryptedPayload(ciphertext=ciphertext, nonce=nonce, tag=None)

    def decrypt(self, payload: EncryptedPayload, key: bytes, *, associated_data: bytes = b"") -> bytes:
        aesgcm = AESGCM(key)
        return aesgcm.decrypt(payload.nonce, payload.ciphertext, associated_data or None)
