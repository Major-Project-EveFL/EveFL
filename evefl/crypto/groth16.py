"""
Groth16 zero-knowledge proofs via circom (legacy 0.5.x, JS-based —
avoids requiring Rust/cargo) + snarkjs, invoked as subprocesses.

This wraps the norm_bound.circom circuit: proves knowledge of a hidden
gradient vector g (length FIXED_N) whose sum-of-squares equals a
publicly-revealed scalar S, without revealing any individual g_i.
Callers compare S against tau^2 in plain Python after verification —
the norm-bound *check* is not itself zero-knowledge, only the
underlying gradient values are hidden.

Requires `circom` and `snarkjs` on PATH (or reachable via `npx`). See
evefl/crypto/circuits/README.md for setup instructions.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from evefl.crypto.zkproof import ProofArtifacts, SetupArtifacts, ZKProver
from evefl.registry import Registry

zk_registry: Registry = Registry("zk_prover")

CIRCUIT_DIR = Path(__file__).parent / "circuits"
CIRCUIT_FILE = CIRCUIT_DIR / "norm_bound.circom"
FIXED_N = 8  # must match `component main = NormBound(N)` in the .circom file
FIXED_POINT_SCALE = 1000  # gradient floats are scaled to integers by this factor


class SnarkjsError(RuntimeError):
    """Raised when a circom/snarkjs subprocess call fails unexpectedly
    (distinct from a proof simply failing to verify, which returns
    False rather than raising)."""


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0 and "groth16 verify" not in " ".join(cmd):
        raise SnarkjsError(f"Command failed: {' '.join(cmd)}\n{result.stderr}")
    return result


@zk_registry.register("groth16")
class Groth16NormBoundProver(ZKProver):
    def __init__(self, work_dir: str | None = None):
        # Persistent work dir so setup artifacts (zkey, vk) can be reused
        # across prove/verify calls instead of re-running the expensive
        # trusted setup every time.
        self._work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="evefl_zk_"))
        self._work_dir.mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str:
        return "groth16-norm-bound"

    def _compile_circuit_if_needed(self) -> None:
        r1cs = self._work_dir / "norm_bound.r1cs"
        wasm = self._work_dir / "norm_bound.wasm"
        if r1cs.exists() and wasm.exists():
            return
        _run(
            ["circom", str(CIRCUIT_FILE), "-r", "norm_bound.r1cs", "-w", "norm_bound.wasm", "-s", "norm_bound.sym"],
            cwd=self._work_dir,
        )

    def setup(self) -> SetupArtifacts:
        self._compile_circuit_if_needed()

        ptau_final = self._work_dir / "pot_final.ptau"
        zkey = self._work_dir / "norm_bound_final.zkey"
        vk_path = self._work_dir / "verification_key.json"

        if not ptau_final.exists():
            ptau0 = self._work_dir / "pot_0000.ptau"
            ptau1 = self._work_dir / "pot_0001.ptau"
            _run(["snarkjs", "powersoftau", "new", "bn128", "12", str(ptau0)], cwd=self._work_dir)
            _run(
                ["snarkjs", "powersoftau", "contribute", str(ptau0), str(ptau1), "--name=evefl", "-e=evefl-entropy"],
                cwd=self._work_dir,
            )
            _run(["snarkjs", "powersoftau", "prepare", "phase2", str(ptau1), str(ptau_final)], cwd=self._work_dir)

        if not zkey.exists():
            _run(
                ["snarkjs", "groth16", "setup", "norm_bound.r1cs", str(ptau_final), str(zkey)],
                cwd=self._work_dir,
            )

        if not vk_path.exists():
            _run(["snarkjs", "zkey", "export", "verificationkey", str(zkey), str(vk_path)], cwd=self._work_dir)

        with open(vk_path) as f:
            vk = json.load(f)

        return SetupArtifacts(
            proving_key_path=str(zkey),
            verification_key=vk,
            metadata={"circuit": "norm_bound", "n": FIXED_N, "scale": FIXED_POINT_SCALE},
        )

    def prove(self, private_inputs: dict[str, Any], setup: SetupArtifacts) -> ProofArtifacts:
        gradient: list[float] = private_inputs["gradient"]
        if len(gradient) != FIXED_N:
            raise ValueError(f"gradient must have exactly {FIXED_N} elements (got {len(gradient)})")

        scaled = [str(round(g * FIXED_POINT_SCALE)) for g in gradient]
        input_path = self._work_dir / "input.json"
        with open(input_path, "w") as f:
            json.dump({"g": scaled}, f)

        witness_path = self._work_dir / "witness.wtns"
        _run(["snarkjs", "wtns", "calculate", "norm_bound.wasm", "input.json", "witness.wtns"], cwd=self._work_dir)

        proof_path = self._work_dir / "proof.json"
        public_path = self._work_dir / "public.json"
        _run(
            ["snarkjs", "groth16", "prove", setup.proving_key_path, str(witness_path), str(proof_path), str(public_path)],
            cwd=self._work_dir,
        )

        with open(proof_path) as f:
            proof = json.load(f)
        with open(public_path) as f:
            public_signals = json.load(f)

        return ProofArtifacts(proof=proof, public_signals=public_signals)

    def verify(self, proof_artifacts: ProofArtifacts, setup: SetupArtifacts) -> bool:
        vk_path = self._work_dir / "verification_key.json"
        proof_path = self._work_dir / "_verify_proof.json"
        public_path = self._work_dir / "_verify_public.json"

        with open(proof_path, "w") as f:
            json.dump(proof_artifacts.proof, f)
        with open(public_path, "w") as f:
            json.dump(proof_artifacts.public_signals, f)

        result = _run(
            ["snarkjs", "groth16", "verify", str(vk_path), str(public_path), str(proof_path)],
            cwd=self._work_dir,
        )
        return result.returncode == 0

    def check_norm_bound(self, public_signals: list[str], tau: float) -> bool:
        """Convenience: the ZK proof only guarantees S = sum(g_i^2) for
        SOME hidden g the prover knows. The actual bound check (is that
        S within budget) happens here, in plaintext, since S itself is
        not sensitive."""
        s_scaled = int(public_signals[0])
        s = s_scaled / (FIXED_POINT_SCALE ** 2)
        return s <= tau ** 2