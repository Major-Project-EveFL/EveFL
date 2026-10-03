# ADR 0001: Key generation uses its own block, separate from the controller's QBER block

Status: accepted (2026-10). Decided by the team after the Phase 1 results.

## Context
- The controller classifies SECURE / CAUTION / LOCKDOWN from a QBER estimated on a small block: the headline setting is
  n_qubits = 1024 with a 25% sample (about 128 bits), chosen by the pre-registered sweep (docs/06). Sweep, Table III and
  Theorem 1 are all computed for this block.
- Phase 1 step 3 showed that the finite-key bound (Tomamichel et al. 2012, eps_sec = 1e-10, eps_cor = 1e-15) leaves
  **no secret key at 1024 qubits**, at any error rate. A key needs about 10^4 qubits or more
  (`docs/validation/key_rates.json`: 0.24 key bits per qubit at 2^17 qubits with 1% noise, 0.31 at 2^20).
- AES-256 needs 256 key bits per client per round; a 2^17-qubit block gives thousands.

## Options
1. **Separate blocks.** The controller keeps the 1024-qubit exchange for the QBER signal; the key comes from its own,
   larger exchange (default 2^17 qubits, sample fraction 0.1) with independent random streams.
2. **One large block for both.** Estimate the QBER on the same large block that produces the key (as real QKD systems
   do), with a sample of the order of 10^4 bits.

## Decision
Option 1.

Reasons: nothing validated moves (sweep, Table III, Theorem 1 and the headline setting all stay valid); the key block is
cheap (about 0.5 s per client-round for post-processing at 2^17); only the key path changes.

## Consequences
- **Two different exchanges per client per round.** The controller's QBER and the key block's QBER are different
  measurements (separate seed streams, `qkd_seed_sequence(..., purpose="controller" | "key")`; the controller stream is
  unchanged so earlier results reproduce). They can disagree.
- **The controller can say SECURE while the key exchange fails.** Examples (tests in `tests/test_round_keys.py`): Eve
  attacks only the key block and it aborts on high QBER; or a QBER the 128-bit sample cannot resolve is punished by the
  finite-key bound and leaves no key. The orchestration policy (`evefl/orchestration/policy.py`) then discards the round
  like LOCKDOWN with a separate reason, `no_key` or `key_abort`, so the logs can tell "the channel looked attacked"
  from "key generation failed". The pure controller stays unaware of key length.
- **Threat-model wording.** The controller's detection claims are about the 1024-qubit exchange only; they say nothing
  about the key block. Paper text must not imply the two are the same measurement.
- **Cost.** One extra exchange and post-processing per client-round (simulated: about 0.3 to 0.9 s at 2^17 qubits).
- Still true: keys are simulated and seed-determined (not secret), authentication is HMAC (computational), and the
  finite-key bound assumes basis-symmetric channels (docs/paper_patches/post_processing.md).

## Option 2 as a later ablation
Real QKD systems monitor QBER on the same large blocks that make the key. A sample of the order of 10^4 bits would cut
the standard error of the QBER estimate by roughly a factor of ten and is expected to remove most of the false CAUTION
seen at 3% baseline noise (the noise-range limitation of the sweep). That is an expectation, not a result; it would need
the sweep and Table III rerun and is therefore deferred until the Phase 4 baseline exists.
