import pytest

from evefl.quantum.bb84 import BB84Protocol


def test_no_eavesdropper_low_qber():
    protocol = BB84Protocol(seed=42)
    result = protocol.run_exchange(n_qubits=400, eavesdropper_active=False)
    # No Eve => ideal channel => QBER should be ~0 (allow small sampling noise)
    assert result.qber < 0.03
    assert result.n_sifted > 0
    assert result.eavesdropper_active is False


def test_intercept_resend_eavesdropper_raises_qber():
    protocol = BB84Protocol(seed=42)
    result = protocol.run_exchange(n_qubits=400, eavesdropper_active=True)
    # Intercept-resend attack theoretically introduces ~25% QBER
    assert result.qber > 0.15
    assert result.eavesdropper_active is True


def test_sifted_key_shorter_than_sent():
    protocol = BB84Protocol(seed=1)
    result = protocol.run_exchange(n_qubits=200, eavesdropper_active=False)
    assert result.n_sifted <= result.n_qubits_sent
    assert result.n_sifted == len(result.sifted_key)


def test_qber_bounds():
    protocol = BB84Protocol(seed=7)
    for eve in (True, False):
        result = protocol.run_exchange(n_qubits=100, eavesdropper_active=eve)
        assert 0.0 <= result.qber <= 1.0
