# ADR 0003: Key custody and secure aggregation (pairwise masking is the baseline, CKKS is optional)

Status: ACCEPTED by the project lead's decisions of 2026-10-04 (see "Decisions"), pending review of this revision; it becomes
final when this branch is merged. No Phase 4 code is written before that, and nothing below is implemented: every "will" is a
design commitment, not a result.

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
- **Identity keys are a trusted-setup assumption, simulated by a test fixture.** Without authenticated public keys the server
  could substitute its own and read everything (a man-in-the-middle on the relay). Rule: the identity public keys reach the
  clients **out of band and never through the server**: the fixture hands the registry to each client object directly, and the
  server object is never given a code path that serves or modifies it. The per-round ephemeral public keys are different: they
  do pass through the server (they cannot be out of band), which is safe only because each is signed with an identity key that
  the receiving client checks against its out-of-band registry. A test makes the server substitute an ephemeral key and shows
  the clients reject it. In tests the registry is generated from `os.urandom` at run time and never committed (CLAUDE.md rule 8).
- Primitives come from `cryptography` (already a dependency): no new package.
- **Limit that must be stated:** X25519 is classical. A recorded transcript could be broken later by a quantum adversary, so
  the masking layer has computational, classical security and **no post-quantum guarantee**, whatever the QKD link does. The
  QKD key protects the link; it does not strengthen the masks.
- Fixed-point arithmetic (decided): updates are scaled to 64-bit integers with 32 fractional bits, arithmetic modulo 2^64, so
  that masks cancel exactly. A coarser 32-bit grid would erase the very small late-training updates. Cost: 8 bytes per
  parameter on the wire instead of 4. The quantisation error against plaintext FedAvg is an acceptance criterion and is
  measured. Three rules go with it:
  1. **Weights are applied client-side, before masking, and come from the enrollment registry, not from the clients.** At
     enrollment each hospital's registry entry holds its identity public key and its aggregation weight `w_i` (its
     training-example count at enrollment, recorded out of band with the key and under the same trust assumption). In every
     round client i sends the masked fixed-point form of `(w_i / sum_{j in S} w_j) * delta_i`; the server only sums, and it
     **ignores any example count a client reports** (in secure mode that field is not read). Self-declared counts cannot be
     trusted: a Byzantine client that inflated its count would inflate its share of the average, and the server could not see
     it inside the masked update. Registry weights are public, fixed for the whole experiment, and renormalised over S each
     round, so excluding a client renormalises exactly as plaintext FedAvg does. A hospital whose dataset changes needs a new
     enrollment (a new registry version), never a per-round claim.
     *Effect on the comparison with plaintext FedAvg.* Plaintext FedAvg today weights by the `num_examples` each client
     reports every round (`len(train_loader.dataset)`; `EveFLStrategy.aggregate_fit`), renormalised over the included clients.
     In this simulation a hospital's training set never changes, so with an honest enrollment (`w_i` equal to that count)
     the weights of the two modes are numerically identical and secure aggregation must reproduce plaintext FedAvg within the
     quantisation tolerance (acceptance test 11). The comparison stops being like for like in three cases, which are reported,
     not hidden: (a) if a registry weight differs from the real count (stale or wrong enrollment), the secure result differs
     from plaintext-with-reported-counts by exactly that reweighting, not by any cryptographic effect; (b) a dishonest count
     moves plaintext FedAvg and cannot move secure aggregation, so under a count-inflation attack the two modes differ by
     design; (c) the CAUTION per-update 0.5 down-weight exists only in plaintext (see the screening rule). The paper's baselines
     therefore include three arms: plaintext with reported counts (the current code), plaintext with registry weights, and
     secure with registry weights. The second isolates the effect of the weighting rule; the third differs from the second
     only by masking and quantisation, so that is the pair a "secure aggregation costs X accuracy" statement must compare.
  2. **Out-of-range values wrap silently, so range is asserted, not assumed.** In two's-complement arithmetic modulo 2^64 with
     32 fractional bits the representable sum is `[-2^31, 2^31)`; a value outside it wraps to a plausible-looking wrong
     number. Each client therefore asserts, before masking, that every element is finite and that
     `|element| * |S| < 2^31` (so the sum cannot wrap), and raises instead of sending otherwise. The aggregator checks that
     the unmasked aggregate lies in range and that no element sits suspiciously close to the boundary, and aborts the round if
     it does. The norm clip in item 2 of the screening rule below keeps honest clients far from the limit.
  3. **Any dropout aborts the round; there is no recovery.** See "Dropout".

**Dropout.** No Shamir-share recovery. The participant set S is fixed by the policy before any key is advertised, using the
QBER and key status already known in `configure_fit`. If any participant in S fails after masking began (an exception, a
timeout, a bad status, a message that fails AEAD authentication, a range-assertion failure), the **whole round is aborted**
(logged with reason `secagg_dropout`), no partial aggregate is formed, and the model is not updated. Because a single client can
trigger this, repeated aborts are a denial-of-service lever of the same kind as selective exclusion (docs/03): the abort count
per client is logged and reported with the participation counts.

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
N = 3. This follows from the floor; the rounds lost will be measured, not assumed. Decision: the floor stays at 3, and a
**Phase 5 experiment with 5 simulated hospitals** is added so per-client exclusion is meaningful under secure aggregation
(exclusions of one or two hospitals leave at least 3). Every number from it is labelled **simulated**: the five hospitals are
five disjoint patient-level partitions of the same ChestX-ray14 data, not five institutions. It needs a partition audit for
5 clients, and it is pre-registered in docs/06 (settings, scenarios, reported quantities) before it runs. The headline
three-hospital results and the documented "three-client scale" limitation do not change.

### Update screening cannot run on masked updates: interim rule
The server sees only `x_i + masks`, so the median/MAD screening of update-delta norms (P0-2), the CAUTION down-weight and the
held clip-to-bound branch are impossible per client. While secure aggregation is on:
1. Per-update screening is **disabled** and every round log says `screening: unavailable_secagg`. This weakens adversary 3
   (Byzantine clients); the paper must say so, not hide it.
2. Each client clips the L2 norm of its own update to a public bound before masking. **This is not an enforcement**: a malicious
   client can skip it, and the server cannot verify it until a norm proof is bound to the submitted update (the ZK redesign, last
   item of Phase 4).
3. Client-side effects still work: FedProx mu in CAUTION, the lower learning rate the server sends, and the FedAvg weights
   (clients scale their update by `w_i / sum w` before masking, with `w_i` from the enrollment registry, rule 1 above).
4. An **aggregate-level check**, accepted as the **only server-side defence** in secure mode: the norm of the aggregated
   delta is compared with the median of previously accepted aggregate norms; above `k_agg` times that median the round is
   discarded (reason `aggregate_norm`), without attributing it to a client. `k_agg` and the warm-up rule are pre-registered in
   docs/06 before any run uses them. **What it cannot catch** (stated plainly, and to be repeated in the paper):
   - any poisoning whose aggregate norm stays in the normal band: a small or "stealthy" update, a backdoor or targeted
     label-flip scaled to look ordinary, a slow drift below `k_agg`;
   - a large malicious update cancelled by another large update (colluding Byzantine clients, or one Byzantine client and a
     non-IID honest one): only the norm of the sum is seen, not the norms of the parts;
   - an attacker who knows `k_agg` and the history (the global model is public to the participants) and stays just under it;
   - anything during warm-up, before enough accepted rounds exist to form the median;
   - which client misbehaved: the check cannot attribute, so it cannot exclude, and it can only throw the round away. A single
     Byzantine client can therefore force repeated discards (a denial-of-service lever), and a legitimate large non-IID update
     can trigger a false discard;
   - a wrong enrolled weight `w_i` (a registry error): clients cannot inflate their weight per round, but nothing in secure
     mode detects a mistake made at enrollment.
   In short: in secure mode the guarantee against Byzantine clients (adversary 3) is "gross blow-ups of the aggregate are
   discarded", and nothing more.
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
| Aggregation weight `w_i` (not secret) | the enrollment registry; everyone reads it | clients: none may set or change it per round |
| Ephemeral X25519 private key | the client, dropped after deriving the seeds | anyone else |
| Ed25519 identity private key | the client | the server |
| CKKS secret key (optional) | the decryptor party | the aggregating server |

## What may be claimed once implemented, and what may not
- May claim, after code + passing tests + a result file: tampered or replayed link messages are rejected; no key is reused
  across clients or rounds; the aggregation server API receives only masked updates and recovers only their sum; the cost of
  each layer in time and bytes (labelled measured, with the library versions the results JSON now records).
- May claim, as an assumption-dependent statement: against an honest-but-curious server, and one colluding client among at
  least three participants, individual updates are hidden up to the sum of the honest ones, computationally and classically.
- May not claim: Theorem 3 in secure mode; post-quantum or hybrid security; protection against a malicious server that lies about S (it can make masks
  fail to cancel; a signed consistency round is out of scope); protection against membership or inversion attacks on the
  aggregate; Byzantine robustness under secure aggregation; that simulated keys are secret.
- The server still sees metadata: who participates, the public registry weights `w_i`, per-client QBER and key status, round
  timing.

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
10. Range assertion: an element just inside `2^31 / |S|` is accepted; one just outside, a NaN and an infinity are refused before
    masking; a deliberately wrapped aggregate is detected and aborts the round. A test also shows that without the assertion the
    wrap is silent (so the assertion is not redundant).
11. Client-side weighting from the registry: the masked sum equals the plaintext weighted average computed with the same
    registry weights within the quantisation tolerance; the server object never receives an unweighted update; a client that
    reports a different example count changes nothing in the secure result (and does change plaintext FedAvg, shown as the
    contrast); excluding a client renormalises the weights over S exactly as plaintext FedAvg does.
12. Dropout: removing any one participant, or corrupting one message, aborts the round with `secagg_dropout`, forms no partial
    aggregate and leaves the model unchanged.
13. A server that substitutes an ephemeral public key is detected (signature check against the out-of-band registry) and the
    round aborts.

## Decisions (project lead, 2026-10-04)
1. Privacy floor stays at 3. Phase 5 adds a run with 5 simulated hospitals, labelled simulated (see above).
2. The aggregate-norm check is the only server-side defence in secure mode; its limits are stated above.
3. Theorem 3 is removed from, or heavily qualified in, the secure-mode claims (see "Paper impact").
4. The identity-key registry is a test fixture; identity public keys reach clients out of band and never through the server.
5. 64-bit fixed point with 32 fractional bits, with the three rules above (client-side weights **fixed at enrollment from the
   registry**, range assertion, any dropout aborts the round). Self-declared example counts are not trusted in secure mode.
6. After this ADR is reviewed and merged, implementation starts with the `KeyManager` (step 1 of the order below).

## Paper impact: Theorems 3 and 4
- **Theorem 3 (Byzantine Robustness under Groth16-Verified Norm Bounds)** claims that in CAUTION, if every client proves
  `||W_i|| <= tau` and the server verifies each proof, a Byzantine client over the bound is rejected with probability 1. It
  cannot stand in secure mode: the server never sees an individual update, only the masked sum, and no proof is bound to the
  masked vector that is actually submitted (docs/03 lists this gap, the missing range checks and the fixed setup entropy for the
  current circuit). Required wording: remove it from the secure-mode claims. If it is kept at all, restrict it to the
  plaintext path, state that it is conditional on a proof bound to the submitted update, and mark it as not yet implemented.
  For secure mode the paper states the aggregate-norm check and its limits instead.
- **Theorem 4 (Server-Blind Aggregation)** assumes CKKS plus a Groth16 norm proof; docs/03 already flags its wording (the
  server learns the exact norm). With pairwise masking as the baseline it needs a restatement naming the adversary (honest-but-
  curious server, at most one colluding client among at least three participants), the classical-only assumption and the
  sum-only leakage. Those edits are the authors'; this ADR only fixes what the code will and will not support.

## Documents to update when this is accepted
docs/03 (custody table and the screening limit), docs/05 (Phase 4 steps in the order below, and the 5-hospital Phase 5 run),
docs/06 (pre-register `k_agg`, the overhead protocol and the 5-hospital run), docs/04 (new known limitations), and the paper
patches (a short patch with the claims above, including Theorems 3 and 4).

## Proposed implementation order (each its own branch, tests first)
1. `KeyManager` plus the key combiner, bound HKDF and AEAD, with acceptance tests 4 to 7.
2. `SecureAggregator` ABC and registry with `plaintext` and `pairwise_mask`, `PairwiseSeedProvider`, tests 1 to 3 and 8.
3. Wire both into `EveFLStrategy` and the runner behind config; screening-unavailable logging; `k_agg` pre-registration first.
4. Overhead measurement (test 9) and a results file.
5. Optional CKKS adapter; ZK norm-proof redesign last.
