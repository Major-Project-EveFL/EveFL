# 02 Target Architecture

## Principles
Interfaces first, simulation and hardware behind the same ABC, controller isolated from ML and
quantum code, everything configurable, every round auditable.

## Target package layout
```
evefl/
  core/          types, errors, pydantic config, logging, registry
  quantum/       base.py (QKDProtocol, ChannelModel), bb84_numpy.py (fast), bb84_qiskit.py
                 (cross-validation), etsi014.py (real KME client), postprocess.py
                 (sifting, param estimation, error correction, privacy amplification, auth)
  keys/          key_manager.py (per-round per-client keys, hybrid combiner, zeroization),
                 kem.py (ML-KEM-768 via liboqs)
  crypto/        cipher (AEAD), secure_agg/ (plaintext, ckks_threshold, pairwise_mask),
                 zk/ (prover + verifier interfaces, commitment-bound norm proof)
  orchestration/ state_machine.py (hysteresis, per-client), policy.py
  fl/            model, data (patient-level split), client, strategy, aggregation rules,
                 runner.py (single-process), deploy/ (Flower gRPC + TLS), evaluation.py
  optimize/      nsga2.py (pymoo, offline)
  telemetry/     structured logs, Prometheus metrics, W&B, hash-chained audit log
  cli/           evefl run | sweep | eval | report
dashboard/  scripts/  tests/  docs/  docker/  .github/workflows/
```

## Key interfaces (sketches, refine in plan mode)
```python
@dataclass(frozen=True)
class ChannelModel:
    intercept_probability: float = 0.0   # Eve alpha
    bit_flip_probability: float = 0.0     # baseline noise: independent flip on Bob's result, after Eve

class QKDProtocol(ABC):
    def run_exchange(self, n_qubits: int, channel: ChannelModel, *, seed: int) -> QKDResult: ...

@dataclass
class RoundKey:
    client_id: str; round_id: int; qber: float; key: bytes   # post-processed, PA length l
    provenance: Literal["bb84-sim", "etsi014"]; leaked_bits: int

class KeyProvider(ABC):
    def get_round_key(self, client_id: str, round_id: int) -> RoundKey: ...

@dataclass(frozen=True)
class Decision:
    state: SecurityState                 # global
    per_client: dict[str, ClientAction]  # TRAIN | TRAIN_PROX | EXCLUDE
    aggregation: AggregationPolicy       # rule, mu, lr multiplier, screening params
    reason: str

class SecureAggregator(ABC):             # plaintext | ckks_threshold | pairwise_mask
    def client_protect(self, update: Update, ctx) -> Protected: ...
    def server_aggregate(self, items: list[Protected], weights: list[float]) -> AggregateHandle: ...
    def open(self, handle) -> Update: ...      # who can call this is a key-custody decision

class UpdateVerifier(ABC):               # plain norm check | zk norm proof
    def attest(self, update) -> Attestation: ...
    def verify(self, att: Attestation, tau: float) -> bool: ...
```

## Per-round sequence
1. KeyProvider gets a RoundKey per client (sim or ETSI 014) with its QBER.
2. Controller consumes {client: qber} and returns a Decision (global state and per-client action).
3. LOCKDOWN. Discard the round, keep last good model, discard that round's keys, log it.
   Otherwise the server sends FitIns with the policy (state, mu, lr multiplier, local_epochs
   from config, not hardcoded).
4. Clients train (FedProx if instructed) and compute the update delta.
5. Client protects the update. AEAD on the wire with a key from the hybrid KDF (AAD = round,
   client, state, model hash). Optionally SecureAggregator encryption and an attestation.
6. Server verifies attestations, screens updates (robust statistics on delta norms), aggregates,
   and updates the global model.
7. Server evaluates on the held-out test set (macro AUC-ROC) and logs metrics.
8. Telemetry and audit record for the round (hash-chained).

## Round policy modes (Phase 2)
`evefl/orchestration/policy.py` (pure logic, registry + YAML `orchestration.mode`) offers three modes: `global_binary`
(QKDFL-style baseline B2: pause at the worst link's QBER >= 0.11, no CAUTION response), `global` (EveFL three states on the
worst link, B3) and `per_client` (one controller per link, B4: a LOCKDOWN link is excluded, the others continue with
renormalised weights, fewer than `min_clients` discards the round). Hysteresis (`orchestration.hysteresis`) slows only the
way out of a state; entering CAUTION at 0.05 and LOCKDOWN at 0.11 is immediate. A client with no key is excluded in
`per_client` mode (reason `no_key` / `key_abort`), which is logged apart from `qber_exclusion`; in the global modes it
discards the round. Excluded links are still measured every round (otherwise they could never rejoin), an excluded client is
not sent the model, and its update is never aggregated even if it arrives. Screening (median/MAD) is computed over all
included updates but clips only clients whose own state is CAUTION, and is inactive below 3 updates (a cost of exclusion).

## Decision (ADR 0001): separate key block
Step 1 of the per-round sequence runs two exchanges per client: the controller's small QBER block and a larger key block with
independent random streams (`evefl/quantum/round_keys.py`). A failed key exchange discards the round like LOCKDOWN with a
separate reason (`evefl/orchestration/policy.py`). See docs/adr/0001-key-block-separate-from-controller-block.md.

## Design constraints that must be resolved
- Anomaly screening needs update norms. Under server-blind aggregation the server cannot compute
  them, so norms come from the verified proof (or from a secure norm-check protocol), not from
  plaintext. Design this explicitly.
- Global max-QBER lockdown penalises everyone for one bad link. Implement per-client exclusion
  as the primary mode and keep global lockdown as a baseline (it is also the QKDFL-style baseline).
- State machine gets hysteresis (paper Eq. 27 promises it, code has none) and a minimum dwell.

## Deployment modes (YAML profiles)
- `research-sim`. Single process, simulated QKD, plaintext aggregation.
- `crypto-full`. Adds key manager, AEAD, secure aggregation, attestations.
- `deploy`. Flower SuperLink/SuperNode over gRPC with mutual TLS, containers, real KME endpoint.

## Non-functional requirements
Typing (mypy strict on core), ruff, pytest with hypothesis for the controller and crypto
round-trips, coverage of core at or above 85%, GitHub Actions CI (lint, type, test, pip-audit,
secret scan, SBOM), multi-stage non-root Dockerfile and compose (server plus 3 clients),
structured JSON logs, Prometheus metrics, semantic versioning and CHANGELOG, SECURITY.md,
CONTRIBUTING.md, license decision by the team (Apache-2.0 suggested), docs site (mkdocs).