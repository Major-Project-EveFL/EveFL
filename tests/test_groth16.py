"""
Tests for Groth16NormBoundProver. These invoke real circom/snarkjs
subprocesses -- not mocks. setup() is expensive (trusted setup), so
it's scoped to module level and reused across tests in this file.
"""

import pytest

from evefl.crypto.groth16 import Groth16NormBoundProver
from evefl.crypto.zkproof import ProofArtifacts


@pytest.fixture(scope="module")
def prover():
    return Groth16NormBoundProver()


@pytest.fixture(scope="module")
def setup(prover):
    return prover.setup()


def test_valid_gradient_proof_verifies(prover, setup):
    gradient = [0.1, 0.2, -0.1, 0.05, -0.05, 0.0, 0.15, -0.2]
    proof_artifacts = prover.prove({"gradient": gradient}, setup)
    assert prover.verify(proof_artifacts, setup) is True


def test_public_signal_reveals_only_norm_not_gradient(prover, setup):
    """The core privacy property: public_signals must contain exactly
    one value (the norm), never the individual gradient components."""
    gradient = [0.12, -0.05, 0.33, -0.21, 0.08, -0.14, 0.02, 0.19]
    proof_artifacts = prover.prove({"gradient": gradient}, setup)
    assert len(proof_artifacts.public_signals) == 1


def test_norm_bound_check_accepts_small_gradient(prover, setup):
    gradient = [0.01] * 8
    proof_artifacts = prover.prove({"gradient": gradient}, setup)
    assert prover.check_norm_bound(proof_artifacts.public_signals, tau=1.0) is True


def test_norm_bound_check_rejects_oversized_gradient(prover, setup):
    """This is the actual Byzantine-poisoning defense: the ZK proof
    itself is valid (it correctly proves the norm), but the norm value
    fails the policy bound and should be rejected at that layer."""
    gradient = [5.0, -5.0, 5.0, -5.0, 5.0, -5.0, 5.0, -5.0]
    proof_artifacts = prover.prove({"gradient": gradient}, setup)
    assert prover.verify(proof_artifacts, setup) is True  # proof itself is honest
    assert prover.check_norm_bound(proof_artifacts.public_signals, tau=1.0) is False  # but norm too big


def test_forged_public_signal_fails_verification(prover, setup):
    gradient = [0.1] * 8
    proof_artifacts = prover.prove({"gradient": gradient}, setup)
    forged = ProofArtifacts(proof=proof_artifacts.proof, public_signals=["999999"])
    assert prover.verify(forged, setup) is False


def test_wrong_length_gradient_raises(prover, setup):
    with pytest.raises(ValueError):
        prover.prove({"gradient": [0.1, 0.2]}, setup)  # not length 8