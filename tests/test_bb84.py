import pytest
from evefl.quantum.bb84 import BB84Protocol

def test_no_eavesdropper_low_qber():
    protocol = BB84Protocol(seed=42)
    result = protocol.run_exchange(n_qubits=400, intercept_probability=0.0)
    # No Eve => ideal channel => QBER should be ~0 (allow small sampling noise)
    assert result.qber < 0.03
    assert result.n_sifted > 0
    assert result.eavesdropper_active is False
    assert result.intercept_probability == 0.0


def test_full_intercept_resend_eavesdropper_raises_qber():
    protocol = BB84Protocol(seed=42)
    result = protocol.run_exchange(n_qubits=400, intercept_probability=1.0)
    # Full intercept-resend attack theoretically introduces ~25% QBER
    assert result.qber > 0.15
    assert result.eavesdropper_active is True
    assert result.intercept_probability == 1.0

def test_prtial_inteception_scales_qber():
    """QBER should scale roughly linearly with intercept_probability,
    since each intercepted qubit independently contributes ~25% erroe risk. This is
    what lets CAUTION (5-11%) actually be reachable, instead if only ever seeing ~0% or ~25%"""
    probabilities = [0.1, 0.3, 0.5, 0.8]
    results=[]
    for p in probabilities:
        protocol = BB84Protocol(seed=123)
        result = protocol.run_exchange(n_qubits=2000, intercept_probability=p)
        results.append(result.qber)
    assert results[0]<results[-1]
    assert results[0]<0.10
    assert results[-1]>0.10


def test_sifted_key_shorter_than_sent():
    protocol = BB84Protocol(seed=1)
    result = protocol.run_exchange(n_qubits=200, intercept_probability=0.0)
    assert result.n_sifted <= result.n_qubits_sent
    assert result.n_sifted == len(result.sifted_key)


def test_qber_bounds():
    protocol = BB84Protocol(seed=7)
    for p in (0.0, 0.5, 1.0):
        result = protocol.run_exchange(n_qubits=100, intercept_probability=p)
        assert 0.0 <= result.qber <= 1.0

def test_invalid_intercept_probability_raises():
    protocol= BB84Protocol(seed=7)
    with pytest.raises(ValueError):
        protocol.run_exchange(n_qubits=50, intercept_probability=-0.1)
    with pytest.raises(ValueError):
        protocol.run_exchange(n_qubits=50, intercept_probability=1.5)
