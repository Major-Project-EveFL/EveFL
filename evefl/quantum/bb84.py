from __future__ import annotations

import random

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from evefl.quantum.base import QKDProtocol, QKDResult
from evefl.registry import Registry

quantum_registry: Registry = Registry("quantum_protocol")

Z_BASIS = 0
X_BASIS = 1


def _prepare_and_measure(
    bit: int, send_basis: int, measure_basis: int, simulator: AerSimulator, seed_simulator: int
) -> int:
    qc = QuantumCircuit(1, 1)
    if bit == 1:
        qc.x(0)
    if send_basis == X_BASIS:
        qc.h(0)
    if measure_basis == X_BASIS:
        qc.h(0)
    qc.measure(0, 0)
    # seed_simulator must be a FRESH value per call (see run_exchange) —
    # reusing one fixed seed across many calls makes every superposition
    # measurement return the identical outcome instead of an independent
    # coin flip, which silently breaks QBER statistics.
    result = simulator.run(qc, shots=1, memory=True, seed_simulator=seed_simulator).result()
    return int(result.get_memory()[0])


@quantum_registry.register("bb84")
class BB84Protocol(QKDProtocol):
    def __init__(self, sample_fraction: float = 0.25, seed: int | None = None):
        self._sample_fraction = sample_fraction
        self._simulator = AerSimulator()
        self._rng = random.Random(seed)

    @property
    def name(self) -> str:
        return "bb84"

    def run_exchange(self, n_qubits: int, intercept_probability: float = 0.0) -> QKDResult:
        if not 0.0 <= intercept_probability <= 1.0:
            raise ValueError(f"intercept_probability must be in [0, 1], got {intercept_probability}")

        alice_bits = [self._rng.randint(0, 1) for _ in range(n_qubits)]
        alice_bases = [self._rng.randint(0, 1) for _ in range(n_qubits)]
        bob_bases = [self._rng.randint(0, 1) for _ in range(n_qubits)]

        bob_results = []
        for i in range(n_qubits):
            bit, send_basis = alice_bits[i], alice_bases[i]
            if self._rng.random() < intercept_probability:
                eve_basis = self._rng.randint(0, 1)
                # Fresh seed per measurement, drawn from the already-seeded
                # self._rng stream — keeps the whole exchange reproducible
                # from one top-level `seed` while still giving each qubit
                # its own independent quantum measurement outcome.
                eve_seed = self._rng.randint(0, 2**31 - 1)
                eve_bit = _prepare_and_measure(bit, send_basis, eve_basis, self._simulator, eve_seed)
                bit, send_basis = eve_bit, eve_basis
            bob_seed = self._rng.randint(0, 2**31 - 1)
            outcome = _prepare_and_measure(bit, send_basis, bob_bases[i], self._simulator, bob_seed)
            bob_results.append(outcome)

        sifted_alice, sifted_bob = [], []
        for i in range(n_qubits):
            if alice_bases[i] == bob_bases[i]:
                sifted_alice.append(alice_bits[i])
                sifted_bob.append(bob_results[i])

        n_sifted = len(sifted_alice)
        qber = self._estimate_qber(sifted_alice, sifted_bob)

        return QKDResult(
            sifted_key=sifted_alice,
            qber=qber,
            n_qubits_sent=n_qubits,
            n_sifted=n_sifted,
            intercept_probability=intercept_probability,
            eavesdropper_active=intercept_probability > 0.0,
            metadata={"sample_fraction": self._sample_fraction},
        )

    def _estimate_qber(self, sifted_alice: list[int], sifted_bob: list[int]) -> float:
        n_sifted = len(sifted_alice)
        if n_sifted == 0:
            return 0.0
        sample_size = max(1, int(n_sifted * self._sample_fraction))
        sample_size = min(sample_size, n_sifted)
        sample_indices = self._rng.sample(range(n_sifted), sample_size)
        mismatches = sum(1 for i in sample_indices if sifted_alice[i] != sifted_bob[i])
        return mismatches / sample_size
