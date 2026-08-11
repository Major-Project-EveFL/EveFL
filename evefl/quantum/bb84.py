"""
BB84 QKD simulation using Qiskit.

Simulates Alice -> Bob qubit transmission with random bases, optional
Eve intercept-resend attack, basis sifting, and QBER estimation via
public comparison of a sample subset of the sifted key.

This is a *simulation* — qubits are represented as single-qubit Qiskit
circuits measured via AerSimulator. No real quantum hardware involved.
"""

from __future__ import annotations

import random

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from evefl.quantum.base import QKDProtocol, QKDResult
from evefl.registry import Registry

quantum_registry: Registry = Registry("quantum_protocol")

# 0 = Z-basis (rectilinear), 1 = X-basis (diagonal)
Z_BASIS = 0
X_BASIS = 1


def _prepare_and_measure(bit: int, send_basis: int, measure_basis: int, simulator: AerSimulator) -> int:
    """Encode `bit` in `send_basis`, measure in `measure_basis`, return the outcome bit."""
    qc = QuantumCircuit(1, 1)

    # Encode
    if bit == 1:
        qc.x(0)
    if send_basis == X_BASIS:
        qc.h(0)

    # Change to measurement basis
    if measure_basis == X_BASIS:
        qc.h(0)

    qc.measure(0, 0)

    result = simulator.run(qc, shots=1, memory=True).result()
    outcome = int(result.get_memory()[0])
    return outcome


@quantum_registry.register("bb84")
class BB84Protocol(QKDProtocol):
    def __init__(self, sample_fraction: float = 0.25, seed: int | None = None):
        """
        Args:
            sample_fraction: fraction of the sifted key publicly compared
                to estimate QBER (standard BB84 practice; these bits are
                discarded from the final key).
            seed: optional RNG seed for reproducibility in tests.
        """
        self._sample_fraction = sample_fraction
        self._simulator = AerSimulator()
        self._rng = random.Random(seed)

    @property
    def name(self) -> str:
        return "bb84"

    def run_exchange(self, n_qubits: int, eavesdropper_active: bool = False) -> QKDResult:
        alice_bits = [self._rng.randint(0, 1) for _ in range(n_qubits)]
        alice_bases = [self._rng.randint(0, 1) for _ in range(n_qubits)]
        bob_bases = [self._rng.randint(0, 1) for _ in range(n_qubits)]

        bob_results = []
        for i in range(n_qubits):
            bit, send_basis = alice_bits[i], alice_bases[i]

            if eavesdropper_active:
                # Intercept-resend: Eve measures in a random basis, then
                # resends a fresh qubit prepared in her measured bit/basis.
                eve_basis = self._rng.randint(0, 1)
                eve_bit = _prepare_and_measure(bit, send_basis, eve_basis, self._simulator)
                bit, send_basis = eve_bit, eve_basis

            outcome = _prepare_and_measure(bit, send_basis, bob_bases[i], self._simulator)
            bob_results.append(outcome)

        # Basis sifting: keep only positions where Alice's and Bob's bases matched
        sifted_alice = []
        sifted_bob = []
        for i in range(n_qubits):
            if alice_bases[i] == bob_bases[i]:
                sifted_alice.append(alice_bits[i])
                sifted_bob.append(bob_results[i])

        n_sifted = len(sifted_alice)
        qber = self._estimate_qber(sifted_alice, sifted_bob)

        return QKDResult(
            sifted_key=sifted_alice,  # final key material (post-sample-removal happens upstream if desired)
            qber=qber,
            n_qubits_sent=n_qubits,
            n_sifted=n_sifted,
            eavesdropper_active=eavesdropper_active,
            metadata={"sample_fraction": self._sample_fraction},
        )

    def _estimate_qber(self, sifted_alice: list[int], sifted_bob: list[int]) -> float:
        n_sifted = len(sifted_alice)
        if n_sifted == 0:
            return 0.0

        sample_size = max(1, int(n_sifted * self._sample_fraction))
        sample_size = min(sample_size, n_sifted)
        sample_indices = self._rng.sample(range(n_sifted), sample_size)

        mismatches = sum(
            1 for i in sample_indices if sifted_alice[i] != sifted_bob[i]
        )
        return mismatches / sample_size
