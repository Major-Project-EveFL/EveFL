"""
Fast, EXACT numpy simulation of BB84 with intercept-resend Eve and a bit-flip channel.

Single-qubit BB84 is fully described by a few classical probabilities, so this backend
simulates every qubit exactly (no approximations, no sampling of the QBER from a binomial):

    Alice picks a bit and a basis.
    Eve (with probability alpha) measures in a random basis: in Alice's basis she reads the
        bit, in the other basis she gets a uniformly random bit; she resends her result in
        her own basis.
    Bob measures in a random basis: if the state he receives is in his basis he reads its
        bit, otherwise he gets a uniformly random bit.
    The channel then flips Bob's result with probability e (bit_flip_probability).
    Sifting keeps positions where Alice's and Bob's bases match; the QBER is estimated on a
    random public sample of the sifted key.

Streams are independent per role (see seeding.py), so scenarios that differ only in alpha or
e share Alice's bits, bases and Bob's bases (paired comparisons). It is cross-validated
against the Qiskit backend (bb84.py) by scripts/validate_backends.py.

Vectorised: a 1024-qubit exchange takes well under a millisecond, versus seconds for Qiskit.
"""

from __future__ import annotations

import numpy as np

from evefl.quantum.base import ChannelModel, QKDProtocol, QKDResult, quantum_registry, resolve_channel
from evefl.quantum.seeding import qkd_seed_sequence, stream_generators


@quantum_registry.register("bb84_numpy")
class BB84NumpyProtocol(QKDProtocol):
    def __init__(
        self,
        sample_fraction: float = 0.25,
        seed: int | np.random.SeedSequence | None = None,
    ):
        if not 0.0 < sample_fraction <= 1.0:
            raise ValueError(f"sample_fraction must be in (0, 1], got {sample_fraction}")
        self._sample_fraction = sample_fraction
        if isinstance(seed, np.random.SeedSequence):
            self._root = seed
        else:
            self._root = np.random.SeedSequence(seed) if seed is not None else np.random.SeedSequence()
        self._calls = 0

    @classmethod
    def for_round(
        cls, experiment_seed: int, server_round: int, cid: str | int, sample_fraction: float = 0.25
    ) -> "BB84NumpyProtocol":
        """Protocol whose streams depend on (experiment seed, round, client)."""
        return cls(sample_fraction=sample_fraction, seed=qkd_seed_sequence(experiment_seed, server_round, cid))

    @property
    def name(self) -> str:
        return "bb84_numpy"

    def run_exchange(
        self,
        n_qubits: int,
        intercept_probability: float = 0.0,
        *,
        channel: ChannelModel | None = None,
    ) -> QKDResult:
        channel = resolve_channel(intercept_probability, channel)
        if n_qubits < 1:
            raise ValueError(f"n_qubits must be >= 1, got {n_qubits}")

        rng = stream_generators(self._root, self._calls)
        self._calls += 1
        n = int(n_qubits)

        # Every stream is consumed for all n qubits whatever the channel, so scenarios pair up.
        alice_bits = rng["alice_bits"].integers(0, 2, n, dtype=np.int8)
        alice_bases = rng["alice_bases"].integers(0, 2, n, dtype=np.int8)
        intercepted = rng["eve_intercept"].random(n) < channel.intercept_probability
        eve_bases = rng["eve_basis"].integers(0, 2, n, dtype=np.int8)
        eve_random = rng["eve_result"].integers(0, 2, n, dtype=np.int8)
        bob_bases = rng["bob_bases"].integers(0, 2, n, dtype=np.int8)
        bob_random = rng["bob_result"].integers(0, 2, n, dtype=np.int8)
        flips = rng["bob_flips"].random(n) < channel.bit_flip_probability

        # State on the wire: Alice's, or Eve's resend if she intercepted.
        eve_bits = np.where(eve_bases == alice_bases, alice_bits, eve_random)
        wire_bits = np.where(intercepted, eve_bits, alice_bits)
        wire_bases = np.where(intercepted, eve_bases, alice_bases)

        bob_bits = np.where(bob_bases == wire_bases, wire_bits, bob_random)
        bob_bits = np.where(flips, 1 - bob_bits, bob_bits).astype(np.int8)

        sifted = alice_bases == bob_bases
        sifted_alice = alice_bits[sifted]
        sifted_bob = bob_bits[sifted]
        n_sifted = int(sifted.sum())

        errors = sifted_alice != sifted_bob
        true_errors = int(errors.sum())

        if n_sifted > 0:
            sample_size = min(n_sifted, max(1, int(n_sifted * self._sample_fraction)))
            sample_idx = rng["sample"].choice(n_sifted, size=sample_size, replace=False)
            sample_errors = int(errors[sample_idx].sum())
            qber = sample_errors / sample_size
        else:
            sample_size, sample_errors, qber = 0, 0, 0.0

        return QKDResult(
            sifted_key=sifted_alice.tolist(),
            qber=qber,
            n_qubits_sent=n,
            n_sifted=n_sifted,
            intercept_probability=channel.intercept_probability,
            eavesdropper_active=channel.intercept_probability > 0.0,
            metadata={
                "backend": self.name,
                "sample_fraction": self._sample_fraction,
                "qber_sample_size": sample_size,
                "qber_sample_errors": sample_errors,
                "bit_flip_probability": channel.bit_flip_probability,
                "expected_qber": channel.expected_qber,
                "n_intercepted": int(intercepted.sum()),
                # Simulation-only ground truth: unavailable to a real QKD system. The controller
                # must only ever use `qber` (the sampled estimate).
                "sim_only_true_error_count": true_errors,
                "sim_only_true_qber": true_errors / n_sifted if n_sifted else 0.0,
            },
        )
