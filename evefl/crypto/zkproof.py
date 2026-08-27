"""
Abstract interface for zero-knowledge proof systems.

Groth16 (via circom + snarkjs, wrapped as a subprocess) is the first
implementation. Kept as its own interface — separate from CipherSuite
and HomomorphicAggregator — since "prove a statement about hidden data
without revealing it" is a distinct job from encryption or homomorphic
computation, even though all three compose together in EveFL's pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ProofArtifacts:
    proof: dict
    public_signals: list[str]


@dataclass
class SetupArtifacts:
    """Paths/handles to the trusted-setup output needed to prove and
    verify against a specific circuit. Opaque outside the implementation."""

    proving_key_path: str
    verification_key: dict
    metadata: dict


class ZKProver(ABC):
    """Base class every zero-knowledge proof system implementation must satisfy."""

    @abstractmethod
    def setup(self) -> SetupArtifacts:
        """Run (or load a cached) trusted setup for the circuit this
        prover wraps. Expensive — expected to be called once, not per-proof."""
        raise NotImplementedError

    @abstractmethod
    def prove(self, private_inputs: dict[str, Any], setup: SetupArtifacts) -> ProofArtifacts:
        """Generate a proof that the prover knows private_inputs
        satisfying the circuit's constraints, revealing only the
        circuit's declared public outputs."""
        raise NotImplementedError

    @abstractmethod
    def verify(self, proof_artifacts: ProofArtifacts, setup: SetupArtifacts) -> bool:
        """Verify a proof against its claimed public signals. Returns
        False for both malformed and dishonest proofs — never raises
        on an invalid-but-well-formed proof."""
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError