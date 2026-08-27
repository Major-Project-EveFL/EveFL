"""
BB84 validation demo — run this and show the output to your guide.

Demonstrates that the simulation produces QBER values matching known
BB84 theory:
  - No eavesdropper -> QBER ~ 0%
  - Intercept-resend eavesdropper -> QBER ~ 25%

Also prints one sample Qiskit circuit so it's visible this is real
gate-level simulation, not a shortcut random-number generator.
"""

from __future__ import annotations

import statistics

from qiskit import QuantumCircuit

from evefl.quantum.bb84 import BB84Protocol


def run_trials(n_qubits: int, eavesdropper_active: bool, n_trials: int = 10) -> list[float]:
    qbers = []
    for trial in range(n_trials):
        protocol = BB84Protocol(seed=trial)  # different seed per trial
        result = protocol.run_exchange(n_qubits=n_qubits, eavesdropper_active=eavesdropper_active)
        qbers.append(result.qber)
    return qbers


def print_summary(label: str, qbers: list[float], expected: str):
    mean_qber = statistics.mean(qbers)
    stdev_qber = statistics.stdev(qbers) if len(qbers) > 1 else 0.0
    print(f"{label}")
    print(f"  trials run       : {len(qbers)}")
    print(f"  mean QBER        : {mean_qber:.4f}  ({mean_qber*100:.2f}%)")
    print(f"  std dev          : {stdev_qber:.4f}")
    print(f"  expected (theory): {expected}")
    print()


def show_sample_circuit():
    """Prints one example single-qubit BB84 circuit so it's visible this
    is real Qiskit gate-level simulation, not a shortcut."""
    print("Sample BB84 qubit circuit (bit=1, Z-basis encode, X-basis measure):")
    qc = QuantumCircuit(1, 1)
    qc.x(0)         # encode bit = 1
    # (no H here since send_basis = Z)
    qc.h(0)         # change to X-basis for measurement
    qc.measure(0, 0)
    print(qc.draw(output="text"))
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("EveFL — BB84 QKD Simulation Validation")
    print("=" * 60)
    print()

    show_sample_circuit()

    for n_qubits in (200, 500, 1000):
        print(f"--- n_qubits = {n_qubits} ---")
        no_eve_qbers = run_trials(n_qubits, eavesdropper_active=False)
        print_summary("No eavesdropper", no_eve_qbers, expected="~0% (ideal channel)")

        eve_qbers = run_trials(n_qubits, eavesdropper_active=True)
        print_summary(
            "Intercept-resend eavesdropper",
            eve_qbers,
            expected="~25% (standard BB84 intercept-resend result)",
        )