"""
CKKS homomorphic aggregation via TenSEAL (Microsoft SEAL Python bindings).

Real implementation, not a mock: this actually performs RLWE-based
encryption, homomorphic ciphertext summation, and decryption. Scoped
deliberately to gradient-sized float vectors (tested on dummy vectors
first — see tests/test_ckks.py) rather than full ResNet-18 gradients,
since serialized ciphertext size and encryption time both scale with
vector length and poly_modulus_degree; full-model-scale benchmarking
is a follow-up once this integrates into the FL pipeline.
"""

from __future__ import annotations

import tenseal as ts

from evefl.crypto.homomorphic import HomomorphicAggregator, HomomorphicContext
from evefl.registry import Registry

homomorphic_registry: Registry = Registry("homomorphic_aggregator")


@homomorphic_registry.register("ckks")
class CKKSAggregator(HomomorphicAggregator):
    def __init__(
        self,
        poly_modulus_degree: int = 8192,
        coeff_mod_bit_sizes: list[int] | None = None,
        global_scale_bits: int = 40,
    ):
        self._poly_modulus_degree = poly_modulus_degree
        self._coeff_mod_bit_sizes = coeff_mod_bit_sizes or [60, 40, 40, 60]
        self._global_scale = 2 ** global_scale_bits

    @property
    def name(self) -> str:
        return "ckks"

    def create_context(self) -> HomomorphicContext:
        ctx = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=self._poly_modulus_degree,
            coeff_mod_bit_sizes=self._coeff_mod_bit_sizes,
        )
        ctx.generate_galois_keys()
        ctx.global_scale = self._global_scale

        # Full context (has secret key) — stays with whoever will decrypt
        # (e.g. the FL coordinator that ultimately needs the aggregate),
        # never sent to the server.
        secret_material = ctx.serialize(save_secret_key=True)

        # Public-only context — safe to hand to the server: it can
        # encrypt/aggregate ciphertexts but cannot decrypt anything.
        ctx.make_context_public()
        public_material = ctx.serialize()

        return HomomorphicContext(public_material=public_material, secret_material=secret_material)

    def encrypt_vector(self, vector: list[float], context: HomomorphicContext) -> bytes:
        ctx = ts.context_from(context.public_material)
        enc = ts.ckks_vector(ctx, vector)
        return enc.serialize()

    def aggregate(self, ciphertexts: list[bytes], weights: list[float], context: HomomorphicContext) -> bytes:
        if len(ciphertexts) != len(weights):
            raise ValueError("ciphertexts and weights must be the same length")
        if not ciphertexts:
            raise ValueError("aggregate() requires at least one ciphertext")

        ctx = ts.context_from(context.public_material)
        vectors = [ts.ckks_vector_from(ctx, ct) for ct in ciphertexts]

        # Weighted homomorphic sum — computed entirely on ciphertexts,
        # server never sees plaintext values of any individual vector.
        agg = vectors[0] * weights[0]
        for v, w in zip(vectors[1:], weights[1:]):
            agg += v * w

        return agg.serialize()

    def decrypt_vector(self, ciphertext: bytes, context: HomomorphicContext) -> list[float]:
        if context.secret_material is None:
            raise ValueError("decrypt_vector() requires a context with secret_material (secret key)")
        ctx = ts.context_from(context.secret_material)
        enc = ts.ckks_vector_from(ctx, ciphertext)
        return enc.decrypt()