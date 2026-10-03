# ADR 0003: Key custody and secure aggregation (pairwise masking is the baseline, CKKS is optional)

Status: PROPOSED (2026-10-03). Awaiting the team's review. No Phase 4 code is written until this is accepted, and nothing
below is implemented: every "will" is a design commitment, not a result.

## Context
- ADR 0001 gives every client, every round, its own QKD link key, produced by a separate large exchange (2^17 qubits,
  thousands of bits). That key is shared by exactly two parties: **client i and the server**. In this repository it is also
  simulated and seed-determined: it is NOT secret and must not be described as protecting anything real.
- docs/05 Phase 4 acceptance: the server verifiably cannot decrypt an individual update; a tampered ciphertext is
  rejected; keys never repeat; overhead is measured per layer.
- Adversaries (docs/03): (1) an eavesdropper on the links who records traffic (harvest now, decrypt later), (2) an
  honest-but-curious aggregation server, (3) a strict minority of Byzantine clients.

**The central point.** A key that the server holds cannot make the server blind. The link key therefore gives
confidentiality and integrity **against outsiders only**; the server decrypts every update it receives. Server blindness
must come from the aggregation layer, and the QKD keys cannot supply the secrets that layer needs (they all terminate at
the server). This ADR separates the two layers and says who holds what.

## Decision

### Layer 1: link protection (client <-> server), AEAD under the QKD key
- One fresh key per client per round and per direction. `K = HKDF-SHA256(ikm, salt = H(transcript), info)` with an
  injective, length-prefixed `info = "evefl/v1/aes-gcm" || direction || round_id || client_id`. The salt `H(transcript)` is
  the SHA-256 of the classical post-processing transcript, which the key pipeline already authenticates with HMAC (docs/03
  step 6, `evefl/quantum/postprocess.py`).
- Cipher: AES-256-GCM, 96-bit random nonce, associated data = `round_id || client_id || direction || state ||
  model_hash` (the hash of the global model the update is based on) and a digest of the tensor shapes. A change to any of
  these fields must make decryption fail.
- Replay: the server and the client each refuse a second key or a second message for the same `(client, round,
  direction)`. Key material is never stored; only the fact that a key was issued.
- **Key combiner, KEM-ready.** `ikm` is built by a function that takes a list of labelled secrets, each length-prefixed,
  so a KEM shared secret can be appended later without touching any caller. Today the list holds the QKD key only and the
  key is tagged `qkd-only`. No hybrid or post-quantum claim is made anywhere; Kyber/ML-KEM is out of scope.
- **Zeroization (best effort, stated as such).** Our own key buffers are `bytearray` and are overwritten with zeros after
  use (`RoundKey.zeroize()` already does this). Python cannot guarantee that no copies exist (immutable `bytes` from
  library calls, interpreter caches), and key objects held inside the `cryptography`/OpenSSL layer cannot be overwritten
  from Python. Tests check our buffers only; the docs say "best effort", never "erased".
- **No fallback.** If a client has no key (`no_key`, `key_abort`) it is excluded or the round is discarded by the policy
  that already exists. Nothing is ever transmitted unprotected to make up for a missing key; a test with a spying
  transport enforces it.
- The key source sits behind a `KeyProvider` interface so a real QKD system (ETSI GS QKD 014 key delivery) can replace the
  simulator without touching FL code (docs/03).

### Layer 2: server blindness, pairwise-masked secure aggregation (mandatory baseline)
Each client i in the round's participant set S adds, to its (pre-weighted, quantised) update, one pseudorandom mask for
every other participant j: `+m_ij` if i < j, `-m_ij` otherwise. The masks cancel in the sum, so the server learns the
aggregate and only the aggregate. `m_ij` is expanded from a pairwise seed by a cryptographic PRG (AES-CTR keystream from
`cryptography`; numpy's `Generator` and `random` are forbidden here, CLAUDE.md rule 8).

**How the pairwise seeds are agreed (the part the QKD keys cannot do).**
- Options considered:
  - A. *Client-to-client Diffie-Hellman, server relays the public keys* (Bonawitz et al. style).
  - B. *The server generates each pair's seed and sends it to both clients under their link keys.* Rejected: it uses the
    QKD keys, which is attractive, but then the server knows every seed and can remove every mask. It would not be blind.
  - C. *Pairwise secrets agreed out of band between hospitals* (for example QKD links between hospital pairs, or a key
    ceremony). The strongest option, but it needs a hospital-to-hospital topology the simulator does not have.
- Decision: **A as the default implementation, behind a `PairwiseSeedProvider` ABC** (registered, selected by YAML) so C can
  be added later.
- Protocol per round: the server announces S (decided before any mask exists, see "Dropout"); every client generates a
  **fresh ephemeral X25519 key pair for this round** and sends its public key together with an **Ed25519 signature** over
  `(round_id, client_id, public_key)` made with its long-term identity key; the server forwards the signed keys; each client
  verifies the others' signatures, computes the shared secret and derives
  `seed_ij = HKDF-SHA256(shared_ij, info = "evefl/v1/mask" || round_id || min(i,j) || max(i,j))`. A fresh ephemeral key per
  round keeps CLAUDE.md rule 2 literally true (no seed reused across rounds) and leaves no long-term DH secret to steal.
- **Identity keys are a trusted-setup assumption.** Without authenticated public keys the server could substitute its own and
  read everything (a man-in-the-middle on the relay). The registry of identity public keys must reach clients by a path that
  does not go through the server alone. In tests the registry is generated from `os.urandom` at run time and never committed.
- Primitives come from `cryptography` (already a dependency): no new package.
- **Limit that must be stated:** X25519 is classical. A recorded transcript could be broken later by a quantum adversary, so
  the masking layer has computational, classical security and **no post-quantum guarantee**, whatever the QKD link does. The
  QKD key protects the link; it does not strengthen the masks.
- Fixed-point arithmetic: updates are scaled to 64-bit integers (proposed: 32 fractional bits, arithmetic modulo 2^64) so that
  masks cancel exactly. A coarser 32-bit grid would erase the very small late-training updates. Cost: 8 bytes per parameter
  on the wire instead of 4. The quantisation error against plaintext FedAvg is an acceptance criterion and is measured.

**Dropout.** No Shamir-share recovery. The participant set S is fixed by the policy before any key is advertised, using the
QBER and key status already known in `configure_fit`. If a participant fails after masking began, the round is discarded
(logged with reason `secagg_dropout`) and the model is not updated. Recovery from dropout is out of scope.

**Privacy floor with 3 clients.**
- The server necessarily learns the aggregate. If the server colludes with one client, it can subtract that client's update and
  learns the **sum** of the remaining honest updates, but not any one of them, because their shared mask is unknown to it.
- With 3 clients and one colluding client, the honest anonymity set is therefore **2**. Any side information about one honest
  update (or an inference attack on a sum of two) collapses that. With two colluding clients the third update is revealed in
  full. Neither is defended; both must be stated in the paper.
- With only 2 participants in a round, one colluding client reveals the other's update exactly, and there is no tolerance at all.
- Rule: a configurable `min_secagg_clients`, **default 3**, enforced as `max(min_clients, min_secagg_clients)`. Below it the
  round is discarded (reason `secagg_too_few`). The pure controller does not change: the floor is a number handed to the policy.

### Consequence of the floor for the per-client policy (stated now, measured in Phase 5)
With 3 hospitals and a floor of 3, **any exclusion discards the round**, so `per_client` mode can no longer keep the two clean
hospitals training once secure aggregation is on. The Phase 2 advantage (ADR 0002, simulated, plaintext aggregation: with Eve
on one link at alpha 0.44 to 0.6, per_client trains 69 to 80% of client-rounds where the global modes train 7 to 41%) therefore does not carry over to the server-blind setting at
N = 3. This follows from the floor; the rounds lost will be measured, not assumed. Options in the open questions.

### Update screening cannot run on masked updates: interim rule
The server sees only `x_i + masks`, so the median/MAD screening of update-delta norms (P0-2), the CAUTION down-weight and the
held clip-to-bound branch are impossible per client. While secure aggregation is on:
1. Per-update screening is **disabled** and every round log says `screening: unavailable_secagg`. This weakens adversary 3
   (Byzantine clients); the paper must say so, not hide it.
2. Each client clips the L2 norm of its own update to a public bound before masking. **This is not an enforcement**: a malicious
   client can skip it, and the server cannot verify it until a norm proof is bound to the submitted update (the ZK redesign, last
   item of Phase 4).
3. Client-side effects still work: FedProx mu in CAUTION, the lower learning rate the server sends, and the FedAvg weights
   (clients scale their update by `n_i / sum n` before masking; the example counts `n_i` are public by design).
4. An **aggregate-level check**: the norm of the aggregated delta is compared with the median of previously accepted aggregate
   norms; above `k_agg` times that median the round is discarded (reason `aggregate_norm`), without attributing it to a client.
   `k_agg` and the warm-up rule are pre-registered in docs/06 before any run uses them.
5. A plaintext (not server-blind) mode with full screening stays available and is the default, so the existing results stay
   reproducible. The paper's baselines include both modes and never present one as the other.

### Optional: CKKS behind `HomomorphicAggregator`
Stays optional and slow (docs/03 estimates about 1 GB per client per round for a full ResNet-18; to be measured, not quoted).
A CKKS result may not be called server-blind without naming the secret-key holder. Proposed holder: a **non-aggregating
decryptor party** that sees only the aggregate; a server colluding with that party breaks it. Threshold CKKS is not planned.

## Who holds what
| Secret | Held by | Never held by |
|---|---|---|
| Link key `K(i, round, direction)` | client i and the server | other clients |
| Pairwise seed `seed_ij` and mask `m_ij` | clients i and j | the server (if the relayed keys are authenticated) |
| Ephemeral X25519 private key | the client, dropped after deriving the seeds | anyone else |
| Ed25519 identity private key | the client | the server |
| CKKS secret key (optional) | the decryptor party | the aggregating server |

## What may be claimed once implemented, and what may not
- May claim, after code + passing tests + a result file: tampered or replayed link messages are rejected; no key is reused
  across clients or rounds; the aggregation server API receives only masked updates and recovers only their sum; the cost of
  each layer in time and bytes (labelled measured, with the library versions the results JSON now records).
- May claim, as an assumption-dependent statement: against an honest-but-curious server, and one colluding client among at
  least three participants, individual updates are hidden up to the sum of the honest ones, computationally and classically.
- May not claim: post-quantum or hybrid security; protection against a malicious server that lies about S (it can make masks
  fail to cancel; a signed consistency round is out of scope); protection against membership or inversion attacks on the
  aggregate; Byzantine robustness under secure aggregation; that simulated keys are secret.
- The server still sees metadata: who participates, `n_i`, per-client QBER and key status, round timing.

## Acceptance tests (written before the code, so they cannot be fitted to it)
1. The server-side aggregator only accepts a `MaskedUpdate` type; a test shows the plaintext update never reaches it.
2. Masked-sum aggregate equals plaintext FedAvg within the quantisation tolerance, for |S| in {3, 4, 5}.
3. A missing participant raises or discards; a wrong aggregate is never returned silently.
4. Flipping a bit of the ciphertext, or changing `round_id`, `client_id`, direction or model hash in the associated data, makes
   decryption fail; a replayed `(client, round)` is rejected; a key of another client does not decrypt.
5. Over N clients x R rounds every derived key is distinct, and `info` encodings differ for any change in round or client
   (injective encoding tested directly).
6. After use, our key buffers are all zero (best effort; the test says what it covers).
7. With a missing key nothing leaves the client in the clear (spy transport).
8. A demonstration, not a proof: with the server's view plus client 3's secrets, the sum `x_1 + x_2` is recovered exactly and
   `x_1` alone is not.
9. Overhead (time, bytes) per layer measured for plaintext, AEAD, masking and, if installed, CKKS, saved as a result file.

## Open questions for the team
1. **Floor vs per-client policy.** Keep the privacy floor at 3 (so exclusions discard rounds at N = 3) or allow a documented
   degraded mode with 2 participants (no collusion tolerance, flagged in every log line)? Proposal: floor 3 by default, and a
   Phase 5 experiment with more simulated hospitals (for example 5) so per-client exclusion is meaningful under secure
   aggregation. That changes the documented "three-client scale" limitation, so it is the team's call.
2. **Interim screening rule.** Accept the aggregate-norm check (item 4) as the only server-side defence, or run server-blind
   mode with no screening at all and state it?
3. **Identity-key registry.** Accept a trusted-setup registry (simulated by a test fixture) as the assumption for
   authenticating the ephemeral keys?
4. **Fixed-point grid.** 64-bit with 32 fractional bits (proposed, 2x the bytes of float32) or a coarser grid with a
   measured accuracy cost?

## Documents to update when this is accepted
docs/03 (custody table and the screening limit), docs/05 (Phase 4 steps in the order below), docs/06 (pre-register `k_agg` and
the overhead protocol), docs/04 (new known limitations), and the paper patches (a short patch with the claims above).

## Proposed implementation order (each its own branch, tests first)
1. `KeyManager` plus the key combiner, bound HKDF and AEAD, with acceptance tests 4 to 7.
2. `SecureAggregator` ABC and registry with `plaintext` and `pairwise_mask`, `PairwiseSeedProvider`, tests 1 to 3 and 8.
3. Wire both into `EveFLStrategy` and the runner behind config; screening-unavailable logging; `k_agg` pre-registration first.
4. Overhead measurement (test 9) and a results file.
5. Optional CKKS adapter; ZK norm-proof redesign last.
