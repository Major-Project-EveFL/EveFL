# ADR 0003: Key custody and secure aggregation (pairwise masking is the baseline, CKKS is optional)

Status: ACCEPTED, with the amendments of 2026-10-10 (see "Decisions"); the amendments were merged on 2026-10-11. Step 1
(KeyManager) and step 2a (fixed-point codec and mask PRG) are implemented and tested as a library; everything else below is a
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
- Protocol per round: the server announces S (decided before any mask exists, see "Dropout"); every client first checks S
  (next bullet), then generates a **fresh ephemeral X25519 key pair for this round** and sends its public key together with
  an **Ed25519 signature** over `(round_id, client_id, public_key, H(S))` made with its long-term identity key (below); the
  server forwards the signed keys; each client verifies the others' signatures and the hash of S, computes the shared secret
  and derives `seed_ij = HKDF-SHA256(shared_ij, info = "evefl/v1/mask" || round_id || min(i,j) || max(i,j))`. A fresh
  ephemeral key per round keeps CLAUDE.md rule 2 literally true (no seed reused across rounds) and leaves no long-term DH
  secret to steal.
- **Each client checks the announced set before it does anything else (amendment 1).** S comes from the server, so no client
  trusts it. Client i refuses to advertise a key, compute a mask or send a masked update unless (a) `|S| >= min_secagg_clients`
  (default 3), (b) every id in S is in its out-of-band registry, (c) i itself is in S, and (d) the round id is the one it
  expects (greater than the last round it took part in). On refusal it sends nothing and the round aborts with reason
  `secagg_bad_set`. The server-side floor protects nobody when the server is the adversary: a server that announced
  `S = {i, colluding client}` would otherwise obtain i's update from the sum. (Acceptance test 14.)
- **Signed advertisements carry a hash of S (amendment 2).** The signed message is
  `lp("evefl/v1/advert") || u64(round_id) || lp(client_id) || lp(ephemeral_public_key) || H(S)`, where
  `H(S) = SHA-256(lp("evefl/v1/set") || u64(round_id) || u32(|S|) || lp(id_1) || ... || lp(id_n))` over the ids in sorted
  order. A client accepts the relayed advertisements only if every signature verifies against its registry, the round id and
  client id match, and **every advertisement carries the same `H(S)` as the hash the client computed itself from the
  announcement it received**; otherwise it aborts with `secagg_set_mismatch` before computing any mask. A server that tells
  client A one set and client B another is therefore caught, because A sees B's signature over a different hash and cannot be
  made to see a forged one. This replaces the earlier "signed consistency round is out of scope" statement. It does not stop a
  server from withholding or dropping advertisements (that is a denial of service: the round aborts). (Acceptance test 15.)
- **PRG layout: no counter or nonce is ever reused (amendment 4).**
  - For each unordered pair {i, j}, round r and floating-point tensor index t (position in the state-dict order) the stream key
    is `K = HKDF-SHA256(ikm = seed_ij, salt = none, info = lp("evefl/v1/mask-stream") || u64(r) || lp(min(i,j)) ||
    lp(max(i,j)) || u32(t), length = 32)`. Different (pair, round, tensor) give different keys, and `seed_ij` is itself new
    every round because the ephemeral keys are.
  - The stream is AES-256 in CTR mode under `K` with an all-zero 128-bit initial counter block; element k of tensor t is the
    little-endian uint64 in keystream bytes `[8k, 8k+8)`. Every stream therefore starts at counter 0 and is consumed from its
    start for that one tensor only. Because no two streams share a key, no (key, counter) pair can occur twice, so the fixed
    zero start is safe; no counter or nonce is shared, carried over or chosen by a caller.
  - The two partners expand the **same** stream on purpose (that is what makes the masks cancel): i adds, j subtracts. This is
    not nonce reuse in the encryption sense: the stream encrypts nothing and is used once by each partner for one tensor of
    one round, and never for anything else.
  - Enforced in code, not by convention: a stream object can be consumed once (a second request for the same
    (pair, round, tensor) raises); a round id is used for at most one secure-aggregation attempt, and an aborted attempt is never
    resumed or retried under the same round id (a retry is a new round with new ephemeral keys); integer tensors have no stream
    (they are not masked, see "What exactly is aggregated"). (Acceptance test 17.)
  - Rejected alternative: one key per pair and a counter/nonce space partitioned among tensors and rounds. It needs bookkeeping
    across tensors and rounds that is easy to get wrong; per-stream keys remove the need.
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
     *Registry weight versus the count the plaintext path reports (checked against the code, 2026-10-10).* The count the
     plaintext path reports is `len(train_loader.dataset)` (`EveFLClient.fit`), the number of images in the hospital's partition,
     which is `hospital_sizes[i]` in `partition_meta.json`. The training loader uses `drop_last=True` whenever the dataset has more
     than one batch, but `drop_last` only changes which samples are iterated in an epoch, not the dataset length that is
     reported. The registry weight must therefore be the **dataset length, not the number of samples actually trained**:
     for the audited partition 46,975, 18,018 and 25,136, while an epoch with batch size 32 iterates 46,944, 18,016 and 25,120
     samples (31, 2 and 16 dropped per epoch, a different set each epoch because the loader shuffles). With that registry the
     two weights are equal integers, and a test asserts the equality from the partition metadata and from a real loader. Weighting
     by samples actually trained would be a different rule and a different arm, not this one.
  2. **Out-of-range values wrap silently, so range is asserted, not assumed.** In two's-complement arithmetic modulo 2^64 with
     32 fractional bits the representable sum is `[-2^31, 2^31)`; a value outside it wraps to a plausible-looking wrong
     number. Each client therefore asserts, before masking, that every element is finite and that
     `|element| * |S| < 2^31` (so the sum cannot wrap), and raises instead of sending otherwise. The aggregator checks that
     the unmasked aggregate lies in range and that no element sits suspiciously close to the boundary, and aborts the round if
     it does. The norm clip in item 2 of the screening rule below keeps honest clients far from the limit.
  3. **Any dropout aborts the round; there is no recovery.** See "Dropout".

**What exactly is aggregated (amendment 3; matches the plaintext path).**
- The unit is the model's whole `state_dict()` in its fixed order, which is what `get_model_parameters` returns and what
  plaintext `aggregate_fit` averages today: for ResNet-18 with the 14-way head, **122 tensors**, of which **102 are floating
  point (11,193,294 elements)** and **20 are integer** (`num_batches_tracked`, one int64 scalar per BatchNorm layer). The
  floating-point elements include the convolution and linear weights and biases, the BatchNorm weight and bias, and the
  BatchNorm `running_mean` and `running_var` (9,600 elements).
- **Floating-point tensors, including BatchNorm running statistics, are masked and summed**, exactly like the weights, because
  the plaintext path averages them too. Client i sends the masked fixed-point form of `(w_i / sum_{j in S} w_j) * (theta_i -
  theta_g)`, where `theta_g` is the broadcast global model and `theta_i` the local model after training. The server decodes the
  sum `Delta` and sets `theta' = theta_g + Delta`; since the weights sum to 1 this equals the FedAvg result `sum_j w_hat_j theta_j`
  the plaintext path computes. Deltas are used because they are small (they fit the fixed-point range, rule 2) and are what the
  aggregate-norm check measures. The BatchNorm running statistics are computed from local activations, so the aggregate (in
  both modes) carries some information about local data; masking hides each client's own, not the aggregate. The paper states it.
- **Integer tensors are not masked and not sent.** The plaintext path takes a float64 weighted mean of them and casts back with
  `astype(int64)` (truncation). Every honest client's counter equals the broadcast value plus `E * b_i` (E local epochs, `b_i`
  training batches per epoch), and `b_i` follows from public inputs: `floor(n_i / B)` when the dataset has more than B
  samples (`drop_last`), otherwise 1. The server computes the counters from the registry weights `n_i`, B and E and applies the
  same float64 weighted mean and truncation, which reproduces the plaintext value for honest clients. A client's own counter is
  ignored, like its example count, so a lie changes nothing. The counters cannot influence model outputs: the BatchNorm layers
  use momentum 0.1, and `num_batches_tracked` is read only when the momentum is None (checked: evaluation outputs are
  bit-identical when the counters are changed). Masking exact integers would need a separate modulus domain for no benefit.
- **Not part of the aggregate:** optimiser state (a fresh AdamW per round), the FedProx reference copy (client-local), training
  metrics and any example count.
- Wire size, arithmetic only (not measured): 11,193,294 x 8 bytes is about 89.5 MB per client per round, against about 44.8 MB
  for the same tensors as float32.

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
- **What masking buys at N = 3 (amendment 6).** Every client already holds the broadcast global model and, after the round,
  receives the new global model, so every client can compute the sum of the *other* clients' weighted updates as the new model
  minus the old one minus its own contribution. A server colluding with one client therefore learns about the honest clients no
  more than that client learns anyway, and the benefit of masking at N = 3 is **against the server alone** (and, through layer 1,
  against an eavesdropper). Masking does not hide the aggregate from any participant, and the paper must not suggest that it does.
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
4. An **aggregate-level check**, accepted as the **only server-side defence** in secure mode, with a trailing window and a
   learning-rate normalisation (amendment 5), all pre-registered in docs/06 before any run uses them:
   - *Statistic.* `a_t = ||Delta_t||_2 / m_t`, where `Delta_t` is the decoded aggregate over the floating-point tensors (the same
     definition as `update_delta_norm`) and `m_t = sum_{i in S} w_hat_i * lr_{i,t} / lr_base` is the weighted mean learning-rate
     multiplier the server itself sent this round (cosine schedule and the CAUTION factor, `schedule.py`). An update shrinks with
     the rate, so without the division a decaying schedule would make every late round look small and every CAUTION round tiny.
   - *Window.* The last `W = 5` **accepted** rounds that had the **same set S**; fewer than 3 such rounds is warm-up and the
     check does not fire (a stated limit). Discarded rounds never enter the window.
   - *Rule.* Discard the round (reason `aggregate_norm`, no client attributed) if
     `a_t > median(window) + k_agg * max(1.4826 * MAD(window), rel_floor * median(window))` with `k_agg = 3` and
     `rel_floor = 0.25`: the form and constants of the pre-registered P0-2 screening rule, applied across time instead of across
     clients, not tuned on any secure run (none exists).
   - *Caveat.* The normalisation assumes the update scales roughly linearly with the rate (true for small steps; FedProx and
     the optimiser make it approximate). Plaintext runs log `a_t`, so the discard rate this rule would have caused on honest runs
     is measured first; if it exceeds the pre-registered 5% it is reported as unfit, not retuned.
   **What it cannot catch** (stated plainly, and to be repeated in the paper):
   - any poisoning whose aggregate norm stays in the normal band: a small or "stealthy" update, a backdoor or targeted
     label-flip scaled to look ordinary, a slow drift below `k_agg`;
   - a large malicious update cancelled by another large update (colluding Byzantine clients, or one Byzantine client and a
     non-IID honest one): only the norm of the sum is seen, not the norms of the parts;
   - an attacker who knows `k_agg` and the history (the global model is public to the participants) and stays just under it;
   - anything during warm-up (fewer than 3 accepted rounds with the same S), and anything after a change of S until a new
     window of the same S exists;
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
- May claim, after the corresponding tests pass: a server that announces different sets to different clients, or a set that is
  too small or contains unregistered ids, is detected by the clients and the round aborts (amendments 1 and 2).
- May not claim: Theorem 3 in secure mode; post-quantum or hybrid security; availability against a malicious server (it can
  withhold or drop messages and force aborts); protection against membership or inversion attacks on the aggregate; that
  masking hides the aggregate from the participants; Byzantine robustness under secure aggregation; that simulated keys are secret.
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
11. Client-side weighting from the registry: the registry weight equals the example count the plaintext path reports (the
    dataset length, asserted from `partition_meta.json` and from a real loader, with `drop_last` active). The masked sum equals
    the plaintext weighted average computed with the same weights within a stated tolerance, per element and for floating-point
    tensors: `|secure - plaintext| <= |S| * 2^-33 + 1 ulp(float32)`. The first term is the rounding of each client's weighted
    delta to the 32-bit fractional grid (half a unit, `2^-33`, per client); the second is the float64-to-float32 cast, which can
    differ by one unit in the last place at a rounding boundary. Integer tensors must be **exactly equal** (rule 3 of "What
    exactly is aggregated"). The server object never receives an unweighted update; a client that reports a different example
    count changes nothing in the secure result (and does change plaintext FedAvg, shown as the contrast); excluding a client
    renormalises the weights over S exactly as plaintext FedAvg does.
12. Dropout: removing any one participant, or corrupting one message, aborts the round with `secagg_dropout`, forms no partial
    aggregate and leaves the model unchanged.
13. A server that substitutes an ephemeral public key is detected (signature check against the out-of-band registry) and the
    round aborts.
14. Set checks (amendment 1): a client refuses, and a spy transport receives nothing, when `|S|` is below the floor, when an id of
    S is not in the registry, when the client itself is not in S, and when the round id is not the expected one.
15. Set consistency (amendment 2): when the server shows two clients two different (individually valid) sets, both abort with
    `secagg_set_mismatch` and no mask has been computed (a spy on the PRG); a flipped bit in `H(S)` or in a signature, or a
    replayed advertisement from another round, is rejected.
16. State-dict handling (amendment 3): the secure aggregate has the same tensor count, order, shapes and dtypes as the plaintext
    aggregate; BatchNorm running statistics are included; integer buffers equal the plaintext values exactly on real client
    outputs of a synthetic run; evaluation outputs are unchanged when only the counters differ.
17. PRG layout (amendment 4): over a domain of pairs, rounds and tensors every stream key is distinct and every stream starts at
    counter 0; the two partners derive identical masks for the same (pair, round, tensor) and different ones for any change of
    pair, round or tensor; a second consumption of the same stream raises; a second secure-aggregation attempt under the same
    round id is refused.
18. Aggregate-norm rule (amendment 5): only accepted rounds with the same S enter the window; the first rounds are unchecked;
    scaling every delta and every learning rate by the same constant leaves the decision unchanged; a ten-fold blow-up is
    discarded and an honest trace is not; changing S empties the effective window.

## Decisions (project lead, 2026-10-04)
1. Privacy floor stays at 3. Phase 5 adds a run with 5 simulated hospitals, labelled simulated (see above).
2. The aggregate-norm check is the only server-side defence in secure mode; its limits are stated above.
3. Theorem 3 is removed from, or heavily qualified in, the secure-mode claims (see "Paper impact").
4. The identity-key registry is a test fixture; identity public keys reach clients out of band and never through the server.
5. 64-bit fixed point with 32 fractional bits, with the three rules above (client-side weights **fixed at enrollment from the
   registry**, range assertion, any dropout aborts the round). Self-declared example counts are not trusted in secure mode.
6. After this ADR is reviewed and merged, implementation starts with the `KeyManager` (step 1 of the order below).

## Decisions (project lead, 2026-10-10: amendments, all in this document)
1. Each client refuses to send a masked update unless the announced set has at least `min_secagg_clients` members and every id
   is in the registry (with a test).
2. Advertisements are signed over round id, client id, public key and a hash of S, and every client checks that all signatures
   carry the same hash as its own.
3. The exact aggregated object is specified, including integer buffers and BatchNorm running statistics, matching the
   plaintext path; the registry weight equals the reported count (checked against `drop_last`) and the tolerance of
   acceptance test 11 is stated.
4. The PRG layout is specified so that no counter or nonce is reused.
5. The aggregate-norm check uses a trailing window of accepted rounds with the same S, normalised by the round's learning-rate
   multiplier; both are pre-registered with `k_agg` in docs/06.
6. The sentence on what masking buys at N = 3 (every client can already derive the sum of the others) is part of the privacy
   floor section.
7. No SecureAggregator code is started until this amended ADR is merged.

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
docs/06 (the aggregate-norm rule is pre-registered there by this amendment; the overhead protocol and the 5-hospital run
still to be), docs/04 (new known limitations), and the paper
patches (a short patch with the claims above, including Theorems 3 and 4).

## Proposed implementation order (each its own branch, tests first)
1. `KeyManager` plus the key combiner, bound HKDF and AEAD, with acceptance tests 4 to 7.
2. `SecureAggregator` ABC and registry with `plaintext` and `pairwise_mask`, `PairwiseSeedProvider`, tests 1 to 3 and 8.
3. Wire both into `EveFLStrategy` and the runner behind config; screening-unavailable logging; `k_agg` pre-registration first.
4. Overhead measurement (test 9) and a results file.
5. Optional CKKS adapter; ZK norm-proof redesign last.

## Implementation status
- **Step 1 (KeyManager, key combiner, bound HKDF, AEAD, single use, zeroization): implemented and merged to `main`
  (merge `c9ca64c`, 2026-10-10).** Code: `evefl/crypto/linkkeys.py` (injective `info` and associated-data encodings, labelled
  length-prefixed key combiner that a KEM secret could join later, HKDF salted with the transcript hash that
  `RoundKey.transcript_sha256` now carries, AES-256-GCM with the bound associated data, one seal and one open per key,
  best-effort overwrite of our own `bytearray` buffers) and `evefl/crypto/keymanager.py` (`KeyProvider` ABC with registry and
  a `simulated_qkd` provider selected from `settings.yaml`, `KeyManager`, `KeyHandle.release()`, `ProtectedSender` and
  `ProtectedReceiver`). `CipherSuite.derive_key` gained an optional `salt`.
- Tests: `tests/test_link_keys.py` and `tests/test_key_manager.py` cover acceptance tests 4, 5 and 6 and test 7 at the level
  of the sender object (a spy transport receives nothing when there is no key). Eleven deliberate breakages of the security
  properties (dropping the client or round from `info`, a field from the associated data, the salt, the header check, the
  single-use checks, the wipe, the no-key guard) were each caught by at least one test.
- **Not done, and not claimed:** nothing is wired into the FL path yet, so P1-1 stays open; the keys are simulated,
  seed-determined and not secret; no secure aggregation exists; the "nothing in the clear" property is shown for the sender
  object, not yet for the whole round. Steps 2b, 2c and 3 to 5 are not started.
- **Step 2a (fixed-point codec and mask PRG): implemented as a library, 2026-10-17, branch `feat/secagg-primitives`.** Code:
  `evefl/crypto/secagg/fixedpoint.py` (64-bit fixed point with 32 fractional bits, modular add and subtract, the client-side
  range assertion `|x| * n < 2^31` with NaN and infinity refused, and an aggregator post-check that aborts when a decoded
  element reaches 2^30), `evefl/crypto/secagg/prg.py` (per-(pair, round, tensor) HKDF key, AES-256-CTR from counter zero,
  little-endian uint64 words, ids ordered by UTF-8 bytes, single-use streams and one attempt per round id enforced in code) and
  `encoding.py`. Tests: `tests/test_secagg_fixedpoint.py` (acceptance test 10) and `tests/test_secagg_prg.py` (acceptance
  test 17), written and seen to fail before the code. **Not done, and not claimed:** no seed provider, no aggregator class,
  nothing wired into the FL path; the aggregate post-check cannot detect a wrap that lands near zero, which is why the
  client-side assertion is the primary defence.
