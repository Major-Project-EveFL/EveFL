# 04 Known Issues (audited)

Priority. P0 blocks correct results. P1 blocks a defensible paper. P2 blocks production quality.
Fix in order. Each fix needs a test.

## P0 (correctness)
- P0-1 [FIXED, tests in tests/test_fedprox.py + tests/test_caution_e2e.py] evefl/fl/client.py `calculate_fedprox_term`. Zips `model.parameters()` (62 tensors) with a
  list built from `state_dict()` (122 tensors, includes BatchNorm buffers) -> ValueError in
  CAUTION. Clients fail, strategy gets no results, `aggregate_fit` returns None, so CAUTION rounds
  are silent no-ops. smoke_test never exercises CAUTION. Fix by mapping global tensors by parameter
  name, add a unit test, and add a smoke run at alpha=0.3 that asserts `fedprox_active == 1`.
- P0-2 [FIXED: evefl/fl/screening.py, median + floored MAD on delta norms, tests/test_screening.py] evefl/fl/strategy.py `_caution_aggregate`. With N=3, mean+2*std can never be exceeded
  (max z = (n-1)/sqrt(n) = 1.155). It also measures full-parameter norm, not update-delta norm
  (paper says gradient norm). Use delta norms with robust statistics (median and MAD) or a
  bound relative to the median.
- P0-3 [FIXED: structured per-round macro/per-class AUC in results JSON via the sequential runner, LOCKDOWN rounds included, skipped classes listed; tests/test_evaluation.py] No evaluation. `evaluate_fn=None`, client `evaluate()` returns NaN. Add server-side
  macro AUC-ROC on the held-out test split (skip classes with no positives), plus per-class AUC.
- P0-4 evefl/fl/dataset.py `_dirichlet_partition`. A multi-label image is added to every client
  that receives any of its positive classes, so partitions overlap. Splitting is per image, but
  ChestX-ray14 has multiple images per patient, so patients leak across train, test and clients.
  Also the code (class-wise Dirichlet) differs from the paper's description (per-sample p~Dir).
  Split by Patient ID, make partitions disjoint, document the exact scheme, fix the paper text.
- P0-5 [FIXED: pretrained is an explicit --pretrained/--no-pretrained flag, offline weights via --pretrained-weights (scripts/cache_pretrained_weights.py), recorded with SHA-256 in every results JSON; clients no longer download weights] Model init. `pretrained=False` on server and clients, but the paper says ImageNet
  pretrained. Decide, apply to both, and make weights available offline on Kaggle.
- P0-6 [FIXED: _stable_seed uses SHA-256] evefl/fl/strategy.py seed uses builtin `hash(client.cid)`, which changes per process
  (PYTHONHASHSEED). Runs are not reproducible. Use `int(cid)` or zlib.crc32.
- P0-7 `local_epochs` hardcoded to 5 in `configure_fit`, config ignored. Also `set_global_seed`
  CUDA init deadlock on Kaggle (see CLAUDE.md gotchas).
- P0-8 [FIXED: shared per-root image index cache] evefl/fl/dataset.py `__getitem__` falls back to per-image `rglob`. Build a filename index
  once. The dataset also re-reads the CSV and rebuilds labels on every client instantiation.

## P1 (scientific validity)
- P1-1 The BB84 key never protects the FL traffic. AES-GCM is not called anywhere in the FL path.
  README ("encrypted gradients") and dashboard footer ("only AES-256-GCM is active") are wrong
  until wired. Wire it or fix the claims.
- P1-2 [FIXED in code: evefl/quantum/postprocess.py (sample discarded, Cascade with counted leakage, verification hash, Toeplitz privacy amplification with the finite-key bound, HMAC authentication), tests/test_postprocess.py, docs/paper_patches/post_processing.md. NOT wired into the FL path yet (Phase 4). Headline 1024-qubit blocks yield no key.] BB84 post-processing. `sifted_key` includes publicly compared sample bits, and there is no
  error correction, no real privacy amplification and no authentication (see docs/03).
- P1-3 [PARTLY FIXED: ChannelModel with bit-flip noise, expected QBER e+(1-2e)alpha/4 tested; false-alarm rates still to be measured by the sample-size sweep, docs/06] Simulator has no channel noise. Add a depolarizing or bit-flip baseline. Without it,
  false-alarm behaviour and SECURE-band statistics are meaningless.
- P1-4 [FIXED in code/data: exact restatement and checks in evefl/quantum/theory.py, tests/test_theory.py, docs/paper_patches/theorem1_table3.md; paper text pending] QBER estimated on a 25% sample (about 128 bits at n=1024). Statistical noise is large. At
  alpha=0.10 (mean 2.5%), a single client exceeds 5% about 3.6% of the time, and about 10% for
  max over 3 clients (false CAUTION). Theorem 1 in the paper: its sigma (~0.031) matches a
  128-bit sample, but the text says |I|=512 and Eq. 28 gives sigma about 0.016 for 512. Restate
  the theorem for the actual sample size m and use the exact binomial tail. At m=128 and
  alpha=0.66, single-client miss probability is about 4.7% (marginal), and about 1e-4 for max
  over three independent links.
- P1-5 [CODE AND CONTROL-PLANE RESULTS DONE: per-client exclusion + per-link attacks (Phase 2), docs/adr/0002, docs/paper_patches/phase2_per_client.md; the ACCURACY comparison (B2 vs B3 vs B4 training runs) is Phase 5] "Graduated beats binary" claim. Both EveFL and QKDFL lock at 11%, so for static alpha>=0.44
  EveFL also halts almost every round. The paper's "about 18 of 50 rounds skipped, AUC 0.773 at
  alpha=0.6" and "36% / 64% rounds discarded" are not derivable from the design. Redesign the
  experiment (per-client links, intermittent or step Eve) or remove the claim.
- P1-6 [FIXED: regenerated, simulated, docs/validation/table3_theorem1.json; alpha=0,e=0 is exactly 0; paper text pending] Paper Table III (0.8% at alpha=0, 100 trials, std 0.5-1.0%) cannot come from this
  noiseless simulator, and demo_validation uses 10 trials. Regenerate from real runs.
- P1-7 Every Section VIII number (AUC, MIA, overhead, Table IV) is projected. "17.3x" CKKS
  inflation is called fabricated in tests/test_ckks.py. QKDFL comparison numbers are invented.
  Replace with measured results and reimplement QKDFL as a baseline. MIA numbers must be measured
  on our models.
- P1-8 [FIXED: AdamW, cosine over the GLOBAL round sent in the round config, x0.5 in CAUTION, base lr 1e-3 (old code actually used Adam 1e-4), all recorded in the results JSON; tests/test_schedule.py] Optimiser and schedule. Paper says AdamW, cosine annealing, lr 5e-4 in CAUTION. Code uses
  Adam 1e-3 constant.
- P1-9 CKKS and Groth16 are not integrated. See docs/03 for the design gaps.
- P1-10 Citations. 11% attributed inconsistently (Lo et al. vs Shor-Preskill; cite Shor and
  Preskill, PRL 85, 441, 2000). Lydersen is 2023 in the lit review and 2010 in the paper. Paper
  has an unresolved "[?]" (Sebert et al.), says "seven clusters" (there are five) and says no
  external citations while its reference list differs from the lit review. Verify every entry,
  especially QKDFL, FedSec, "FedMed" (Roth et al.) and Kaissis et al. (Lancet Digital Health).

## P2 (engineering and docs)
- P2-1 [PARTLY FIXED: demo_validation.py and the README quick-start now use the real API; README status table still stale] evefl/demo_validation.py is broken (`eavesdropper_active` kwarg no longer exists). README
  quick-start uses `eve_intercept_rate` (nonexistent) and shows a wrong example output. README
  status table is stale. quantum/base.py docstring is stale.
- P2-2 bb84.py ValueError message missing f-string prefix.
- P2-3 [FIXED: exact vectorised backend bb84_numpy (default), Qiskit kept as reference and cross-checked by tests/test_backend_crosscheck.py + scripts/validate_backends.py] BB84 runs one Aer job per qubit (up to two), taking seconds per client-round, not the
  paper's 180 ms. Add a vectorised numpy backend and keep Qiskit for cross-validation.
- P2-4 [FIXED: HysteresisConfig (immediate escalation, slow de-escalation), property tests, 100% branch coverage of the controller and policy modules] State machine has no hysteresis. Add it plus property tests.
- P2-5 `min_available_clients=max(num_clients,3)` can hang for num_clients<3. Flower
  `start_simulation` is deprecated in newer versions. Plan the migration.
- P2-6 Missing tests. Strategy, client, dataset, aggregation, integration, docs build.
- P2-7 No CI, Docker, telemetry, lockfile or SBOM.
- P2-8 Team metadata inconsistent across documents (guide name, one USN). Confirm and unify.
- P2-9 FedAvg averages BatchNorm buffers including num_batches_tracked (int64 cast after
  float average). Handle explicitly.