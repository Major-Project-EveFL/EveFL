# 03 Security Model and Crypto Requirements

## Assets
Patient images (never leave hospital), model updates, global model, session keys, audit log.

## Adversaries
1. External eavesdropper on the quantum key-exchange link (intercept-resend modelled; PNS and
   detector blinding out of scope of the simulator). Can record classical traffic (HNDL).
2. Honest-but-curious aggregation server.
3. Byzantine clients, a strict minority.
Out of scope: DoS, brute force, endpoint compromise, side channels in the simulator.

## What QBER does and does not tell you
- Rises when the quantum link is disturbed (Eve or noise). It cannot tell them apart.
- Says nothing about ciphertext sniffing on the classical channel.
- Blinding attacks can leave QBER unchanged. Say so plainly.
- 11% is the asymptotic Shor-Preskill bound for ideal single-photon BB84 with one-way
  post-processing. With finite samples the tolerable QBER is lower. Key fraction is roughly
  1 - 2h(e), which is about 0.66 at 2.5%, 0.43 at 5% and about 0 at 11%.
  Consequence: in CAUTION the secret key must be compressed accordingly or the hybrid Kyber
  secret must carry the security. Do not derive a full 256-bit key from a barely-secret string.

## Required key pipeline (replace the current shortcut)
1. Sift on matching bases.
2. Parameter estimation. Publicly compare a random sample, compute QBER, then DISCARD the sampled
   bits (current code leaves them in the key).
3. Error correction (Cascade first, LDPC later). Count every leaked bit.
4. Privacy amplification with universal hashing (Toeplitz). Output length
   l = n_remaining - leaked_bits - f(qber, finite-key terms). SHA-256 is a KDF, not PA.
5. Authenticate the classical channel (Wegman-Carter MAC with a pre-shared initial key, HMAC in sim).
6. Combine: K = HKDF-SHA256(ikm = k_qkd || s_kyber, salt = H(transcript), info =
   "evefl/v1/aes-gcm" || round || client). If Kyber is off, use only k_qkd and tag the key
   `qkd-only`.
7. AEAD: AES-256-GCM, 96-bit random nonce, AAD = round, client, state, model hash. One key per
   round, zeroize afterwards. Reject replayed rounds.
8. LOCKDOWN: discard that exchange's key material entirely and run a new exchange next round.

## Selective exclusion as an attack lever (per_client mode)
Out of scope otherwise (denial of service), but in `per_client` mode an adversary who disturbs one link CHOOSES which
hospital's data drops out of training, biasing the model under non-IID data and, below `min_clients`, halting training.
Mitigations are reporting, not prevention: per-client participation counts are logged every round and must be reported
with per-client performance in the Phase 5 experiments. Excluding a hospital is never free; its cost is measured, not
assumed. With only 2 updates left, update screening is inactive (needs >= 3), a direct cost of exclusion.

## Implemented pipeline and finite-key accounting (Phase 1, step 3)
`evefl/quantum/postprocess.py` implements steps 1 to 5 above with the Tomamichel et al. 2012 finite-key bound
(Supplementary Theorem 2, verified against the paper; assumptions in the module docstring and
`docs/paper_patches/post_processing.md`). Authentication is HMAC-SHA256 in the simulator, not Wegman-Carter. Rounds in
which any client gets no key are treated like LOCKDOWN by `evefl/orchestration/policy.py` with reason `no_key`.
Findings: blocks of about 10^4 qubits or more are needed for any key at eps_sec = 1e-10; the 1024-qubit controller
setting yields none. Steps 6 to 8 (HKDF binding, AEAD, zeroization in the FL path) are Phase 4.

## Simulation caveat
The simulator is not QKD security. The key provider interface must accept keys from an ETSI GS QKD
014 key-delivery API so a real system can replace it without touching FL code.

## Homomorphic aggregation requirements
- TenSEAL CKKS at N=8192 packs 4096 slots per ciphertext. Full ResNet-18 (about 11.2M params)
  needs roughly 2,700 chunks, so chunking is mandatory. Expect on the order of 1 GB per client
  per round unless optimised. Measure it (test_ckks already has the measurement test).
- Key custody. One shared secret key held by the aggregator's context breaks server-blindness.
  Options, decide with the team: (a) key held by a non-aggregating party, (b) threshold or
  multiparty CKKS (OpenFHE or Lattigo), (c) pairwise-masked secure aggregation with dropout
  tolerance (Bonawitz-style), which is far cheaper. Keep (c) as a mandatory baseline in
  experiments. Server-blind claims must name who holds the secret key.

## Verifiable norm bound requirements
Current circuit proves "I know some g with sum g_i^2 = S" for n=8. Gaps:
- No range checks on g_i, so field wrap-around lets a huge vector look small. Add Num2Bits per element.
- The proof is not bound to the encrypted or submitted update. Bind via a commitment
  (Poseidon or Pedersen) to the same vector that is sent, or use a secure-aggregation input
  validation protocol (see RoFL and ACORN in the literature, verify before citing).
- Exact S is public, so the server learns the norm, not just "norm is within bound". Fix the
  paper's Theorem 4 wording or hide S behind an in-circuit comparison.
- Setup uses fixed entropy ("evefl-entropy"), so anyone can forge proofs. Tests only. For
  deployment use a real multi-party ceremony or a transparent proof system.
- circom 0.5 is unmaintained (`signal private input`). Move to circom 2.x.
- n=8 cannot cover 11M parameters. Prove over chunked or randomly projected norms, and state
  the approximation. Groth16 proof is about 128 bytes compressed in binary, the "805 bytes"
  figure is snarkjs JSON size.

## Known accepted limitations (must appear in the paper)
Detector blinding, unauthenticated classical channel until step 5 lands, trusted setup, CKKS
communication cost, simulation only, three-client scale.