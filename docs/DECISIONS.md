# EveFL: decisions, architecture, results and limits

A single place to answer "what did you decide, why, and what is the evidence". It is a living log: a change that decides
something, or a result that arrives, adds a line here in the same pull request. Last reviewed: 2026-10-18.

How to read it
- **Evidence** points to a file in this repository. "Decided in chat" means the project lead decided it in conversation and no
  repository file records the reasoning yet; those lines are the ones to move into an ADR or docs/06 when they matter.
- **Labels.** Every number is *measured* (a run on real data or hardware that produced a result file), *simulated* (a model or
  simulator we wrote), or *projected* (arithmetic from measured rates; not a run). Literature numbers are never ours.
- The authority for each issue's status is [docs/04_KNOWN_ISSUES.md](04_KNOWN_ISSUES.md), for what to build next
  [docs/05_ROADMAP.md](05_ROADMAP.md), for how numbers are produced [docs/06_EXPERIMENT_PROTOCOL.md](06_EXPERIMENT_PROTOCOL.md).
  This file indexes them; it does not replace them.

## 1. What EveFL is, in one paragraph
Federated learning of a chest X-ray classifier (ResNet-18, NIH ChestX-ray14, 3 simulated hospitals) in which a **simulated BB84
quantum-key-distribution link** per hospital supplies two things: a QBER reading that drives a three-state controller
(SECURE / CAUTION / LOCKDOWN) deciding whether a hospital's update is used, and key material that protects the update on the
wire. The quantum layer is a simulator: the keys are seed-determined and **not secret**, and nothing here has been measured on
QKD hardware. The project is a research prototype with tests and recorded results; it is not a certified or deployed system.
**No model-accuracy baseline has been produced yet** (there is one single-seed pilot at one local epoch); the lite runs are
pipeline checks.

## 2. Architecture at a glance
| Layer | What it does | Status | Evidence |
|---|---|---|---|
| QBER controller | three states on QBER, thresholds 0.05 and 0.11, hysteresis, three policy modes | implemented, simulated, 100% branch coverage of the controller and policy modules | [ADR 0002](adr/0002-per-client-decisions-and-hysteresis.md), `evefl/orchestration/` |
| QKD simulation | BB84 with a channel model (noise, intercept-resend); exact NumPy backend, Qiskit reference | implemented, simulated | `docs/validation/backend_crosscheck.json` |
| Key layer | key from its own 2^17-qubit block, Cascade, verification hash, Toeplitz privacy amplification, finite-key bound | implemented, simulated | [ADR 0001](adr/0001-key-block-separate-from-controller-block.md), `docs/validation/key_rates.json` |
| Link protection | HKDF-SHA256 bound to round, client and direction; AES-256-GCM with bound associated data; single use; best-effort zeroization | implemented as a library and tested; **not yet wired into the FL path** | [ADR 0003](adr/0003-key-custody-and-secure-aggregation.md) step 1, `evefl/crypto/linkkeys.py`, `keymanager.py` |
| Secure aggregation | pairwise masking so the server learns only the sum | **designed and accepted; step 2a (fixed-point codec, mask PRG) implemented and tested as a library; no seed provider or aggregator yet, nothing wired in** | ADR 0003, layer 2 |
| FL runner | single-process sequential driver around Flower's strategy and client | implemented | `evefl/fl/runner.py` |
| Data | patient-level disjoint train, validation and test splits, Dirichlet(0.5) across hospitals | implemented, verified on the real metadata | `docs/validation/partition_audit_real.json` |
| Image backends | PNG decode (default, reference) or pre-resized uint8 cache | implemented, accepted | docs/06, `docs/validation/image_cache_build_2026-10-12.json` |
| CKKS, Groth16 norm proof | modules exist, optional | **not integrated** | docs/04 P1-9 |
| Kyber / ML-KEM, NSGA-II supervisor | not implemented; the key combiner is KEM-ready | out of scope for now | ADR 0003, decided in chat |

## 3. Decision log
Dates are the merge or decision dates in git history where a file records them.

### Security and cryptography
| ID | Date | Decision | Why | Rejected | Evidence |
|---|---|---|---|---|---|
| S1 | standing | LOCKDOWN threshold is 0.11 (the BB84 asymptotic bound), compared with `>=`, never configurable below 0.11; 5% CAUTION is a configurable heuristic | the 11% bound is the security-relevant constant; the 5% boundary is not | a tunable lockdown threshold | CLAUDE.md rule 1, `evefl/orchestration/state_machine.py`, tests |
| S2 | 2026-10-03 | The key comes from a separate, larger exchange (2^17 qubits), not from the controller's 1024-qubit block | at 1024 qubits the finite-key bound leaves no secret key at any error rate; nothing validated has to move | one large block for both | ADR 0001, `docs/validation/key_rates.json` (simulated) |
| S3 | 2026-10-03 | Three policy modes behind an ABC and registry (`global_binary` = B2, `global` = B3, `per_client` = B4); hysteresis slows only the way out; recommended dwell 3, margin 0.01 | tests the claim that a graduated, per-link response beats a binary pause; pre-registered sweep | a single global mode | ADR 0002, `docs/validation/hysteresis_sweep.json` (simulated) |
| S4 | 2026-10-04 | Link protection: HKDF with injective length-prefixed `info`, salt = authenticated transcript hash, AES-256-GCM with associated data binding round, client, direction, state, model hash and shapes; one seal and one open per key | rule 2 of CLAUDE.md: one fresh key per client per round, bound to round and client | reusing keys; non-injective encodings | ADR 0003 layer 1, `tests/test_link_keys.py`, `tests/test_key_manager.py` |
| S5 | 2026-10-04 | Secure aggregation is **pairwise masking** (ephemeral X25519 per round, Ed25519 identities from an out-of-band registry, AES-CTR streams); CKKS stays optional behind an ABC | server-generated seeds from the QKD keys would let the server remove every mask, so it would not be blind | server-generated seeds; mandatory CKKS | ADR 0003 layer 2 |
| S6 | 2026-10-04 | 64-bit fixed point (32 fractional bits); weights fixed at enrollment from the registry, applied client-side; range assertion; any dropout aborts the round (no Shamir recovery) | self-declared example counts cannot be trusted; silent wrap-around must not happen | 32-bit grid; per-round weights from clients | ADR 0003 |
| S7 | 2026-10-04 | Privacy floor of 3 clients; a Phase 5 run with 5 *simulated* hospitals so per-client exclusion stays meaningful under secure aggregation | with 3 hospitals any exclusion discards the round | lowering the floor | ADR 0003, docs/06 |
| S8 | 2026-10-04, amended 2026-10-10 | The aggregate-norm check is the only server-side defence in secure mode: trailing window of 5 accepted rounds with the same set, learning-rate normalised, `k_agg` = 3, `rel_floor` = 0.25; reported unfit if it would discard more than 5% of honest rounds | per-update screening is impossible on masked updates; constants are those of the pre-registered P0-2 rule | tuning on secure runs | ADR 0003, docs/06 (pre-registered) |
| S9 | 2026-10-04 | Theorem 3 is removed from, or heavily qualified in, secure-mode claims | it assumes a server that sees plaintext updates | keeping it as stated | ADR 0003, "Paper impact" |
| S10 | 2026-10-10 | Each client refuses to take part unless the announced set has at least the floor, every id is in its registry and it is in the set; advertisements are signed over a hash of the set | the server-side floor protects nobody when the server is the adversary | trusting the announced set | ADR 0003 amendments 1 and 2 |
| S11 | 2026-10 | Kyber is out of scope for now; the key combiner is labelled and KEM-ready; no hybrid or post-quantum claim is made | keeps the design honest about what is implemented | claiming a hybrid scheme | ADR 0003, decided in chat |
| S14 | 2026-10-17 | Secure aggregation 2a: 64-bit fixed point with a client-side range assertion (`\|x\| * n < 2^31`, NaN and infinity refused) and an aggregator post-check that aborts at `\|sum\|` of 2^30; one HKDF key per (pair, round, tensor) and AES-256-CTR from counter zero; ids ordered by UTF-8 bytes; single-use streams and one attempt per round id enforced in code | masks must cancel exactly and a wrap must not be silent; no (key, counter) pair may repeat | a partitioned counter space under one key per pair (bookkeeping error-prone) | ADR 0003 amendment 4, `evefl/crypto/secagg/`, `tests/test_secagg_fixedpoint.py`, `tests/test_secagg_prg.py` |
| S12 | 2026-10 | Simulated keys are labelled seed-determined and not secret wherever they appear; zeroization is "best effort" on our own buffers | Python cannot guarantee memory erasure | calling it secure erasure | ADR 0003, [docs/03_SECURITY_MODEL.md](03_SECURITY_MODEL.md) |
| S13 | 2026-10-16 | SecureAggregator step 2 is built as three branches, tests first: 2a primitives (fixed-point codec, mask PRG), 2b seed provider (ABC, X25519/Ed25519, registry, set checks), 2c aggregator (ABC, `plaintext` and `pairwise_mask`); no new dependency; wiring into the strategy and runner is step 3 | each branch is small enough to review and tests precede code; the ADR's acceptance tests 1 to 3, 8 and 10 to 17 map onto them | one large branch | ADR 0003 (order and tests), decided in chat |

### Data and method
| ID | Date | Decision | Why | Evidence |
|---|---|---|---|---|
| M1 | 2026-10-03 | Splits are by patient ID, disjoint, with a 10% test and a 10% validation split; hospitals are non-IID Dirichlet(0.5); verified on the real metadata | the earlier image-level split leaked patients across train and test (P0-4) | `docs/validation/partition_audit_real.json` (measured on the metadata), docs/04 |
| M2 | 2026-10-03 | Model selection uses the validation split's macro AUC over classes with at least 20 positives; the test split is for the final report only | thin classes swing AUC by several hundredths | docs/06 (pre-registered), `evefl/fl/evaluation.py` |
| M3 | 2026-09 to 10 | The controller's QBER block is 1024 qubits with a 25% sample (about 128 bits); samples under about 100 bits are refused | pre-registered sample-size sweep | docs/06, `docs/validation/qber_sweep.json` (simulated) |
| M4 | 2026-09-29 | AdamW, learning rate 1e-3 with cosine annealing over the global round, x0.5 in CAUTION | the older code actually used Adam at 1e-4; the paper text said otherwise (P1-8) | docs/04 P1-8 |
| M5 | 2026-09-29 | Pretrained weights are an explicit flag with a recorded SHA-256; offline weights supported | silent downloads made results non-reproducible (P0-5) | docs/04 P0-5, `docs/KAGGLE_RUNBOOK.md` |
| M6 | 2026-10-10 | Federated evaluate is off by default; test evaluated every N rounds and on the final round; validation every round | the second pass fed neither selection nor the controller and cost about 43% of a round | `evefl/fl/server.py`, `experiment.evaluation` in results |
| M7 | 2026-10-15 | Comparisons use several seeds (0 to 4, minimum 3) with mean and 95% CI; one lite-scale run is never used to rank methods; runs compared with each other use the same image backend | GPU runs are not bit-reproducible: three runs with the same seed differ by up to 0.004 in loss and 0.14 in AUC at lite scale | docs/06 addendum, `docs/validation/image_cache_lite_run_2026-10-15.json` |
| M8 | standing | Every number in the paper traces to a result file with a config hash and seeds; otherwise it is labelled projected or removed | honesty rule | CLAUDE.md rule 6, docs/06 |
| M9 | 2026-10-17 | The first full-data baseline is a **single-seed pilot** of B0 (unsecured FedAvg): full data, partition seed 42, run seed 0, 224 px, fp32, cache backend, 50 rounds, validation every round, test every 5 rounds and last; **E = 1, a recorded deviation from the E = 5 specification** because E = 5 is projected at about 18 h per run against about 3.8 h; the primary result is the round-50 test AUC with no selection | the first measured accuracy figure and a full-scale check of the compute projection, without spending quota on a configuration that cannot be repeated for several seeds | E = 5 pilot; AMP in the pilot (learning effect not yet measured) | docs/06 (pre-registered 2026-10-17) |

### Engineering and performance
| ID | Date | Decision | Why | Evidence |
|---|---|---|---|---|
| E1 | 2026-09 | A single-process sequential runner drives the strategy and clients directly | Ray/Flower simulation deadlocked after CUDA initialisation on Kaggle | CLAUDE.md gotchas, `evefl/fl/runner.py` |
| E2 | 2026-10-03 | Kaggle installs with `--no-deps` from `requirements-kaggle.txt` (Python 3.13, torch 2.11); Kaggle runs are a different environment from CI and are labelled so | the pinned set has no cp313 wheels | `docs/KAGGLE_RUNBOOK.md` |
| E3 | 2026-10-10 | Library versions are recorded in every results JSON and the NumPy version in the partition metadata | partition hashes are comparable only under the same NumPy | `experiment.environment` |
| E4 | 2026-10-10 | Profiling instrumentation with a pre-registered rule: build a preprocessing cache only if loader wait is at least 50% of the loaded phases, the single-process decode rate is under half the compute rate, and the run was on a GPU | decide on evidence, not on a hunch | docs/06, `docs/validation/kaggle_profile_2026-10-11.json` (measured) |
| E5 | 2026-10-11 | uint8 cache: stored with the same `Resize(256)` call on the RGB-converted image, one channel kept after checking equality, sharded by 4096, resumable, an image that cannot be cached fails the build and is listed (no silent PNG fallback), PNG stays the default backend, the cache hash is in every results JSON and in `config_hash` only when the backend is not PNG, normalisation moves to the device | the data path must be provably the same input | docs/06 (pre-registered acceptance), `evefl/fl/imagecache.py` |
| E6 | 2026-10-12 to 15 | The cache was **accepted**: items 1 to 4 of the rule hold; item 5 is report-only | see section 4 | `docs/validation/image_cache_build_2026-10-12.json`, `image_cache_lite_run_2026-10-15.json` |
| E7 | 2026-10-15 | A deterministic-GPU mode is **not built** for now | not needed for correctness; costs throughput; PyTorch documents the adaptive-average-pool backward on CUDA as non-deterministic, so it may need model changes; seeds with confidence intervals are the protocol anyway | decided in chat |
| E8 | 2026-10-15 | AMP and `channels_last` were benchmarked (throughput only): AMP + `channels_last` trains 2.66x faster at 224 px, `channels_last` alone is slower (0.79x); wiring AMP into training awaits a decision and a learning-effect measurement against the run-to-run spread | the largest remaining compute lever; a speed-up says nothing about accuracy | `scripts/profile_compute.py`, `docs/validation/compute_benchmark_2026-10-15.json` (measured, throughput only) |
| E9 | 2026-10-18 | `--save-model DIR` writes two checkpoints at the end of a run: the final-round global model and the model of the round with the highest **validation** macro AUC over non-thin classes (ties to the later round); the test split is never used for the choice; the files stay out of git and the manifest is recorded in the results JSON; saving changes nothing that is trained or hashed | a trained model is needed for any inference demo, and the choice of which model is shown must be fixed before test numbers are seen | selecting on test AUC; keeping only the last round | `evefl/fl/checkpoint.py`, `tests/test_checkpoint.py` |

### Process and repository
| ID | Date | Decision | Why | Evidence |
|---|---|---|---|---|
| P1 | standing | One branch per task; small one-concern commits; explicit `git add` paths; merge to `main` only when the project lead confirms Actions is green; push only to the `parineeta` remote, never to the read-only `origin`; no `push --all` or `--mirror` | keeps `main` always green and the shared repository untouched | CLAUDE.md |
| P2 | 2026-10 | The repository is private; reference material (`handoff/`, `docs/reference/`) was purged from history with a targeted `filter-branch` that keeps signed commits; the mentor's reference thesis stays out of the repository | not ours to publish | git history, `.gitignore` |
| P3 | standing | Destructive steps (history rewrites, force pushes, deleting backups or old folders, deleting remote branches) happen only on an explicit instruction, after a backup and a shown verification | irreversibility | working agreement |
| P4 | standing | Ask before adding a dependency or an improvement; pre-register selection rules in docs/06 before running anything | prevents fitting rules to results | CLAUDE.md |
| P5 | 2026-10-15 | This file is the single index of decisions, results and limits for reviewers and funders; every decision or result adds one line to it in the same pull request, and a test fails if a cited path disappears | a funder will ask about every decision, architecture choice and result | `tests/test_decisions_log.py`, CLAUDE.md |
| P6 | 2026-10-17 | A project tracker lists every roadmap item with a status (DONE, PARTIAL, TODO, WAITING, STRETCH), the evidence and what is left; a test keeps its counts and cited paths true | reviewers and funders ask what is finished and what is not, and the roadmap alone does not say | [PROJECT_TRACKER.md](PROJECT_TRACKER.md), `tests/test_project_tracker.py` |

## 4. Results to date
| Result | Label | Value | File |
|---|---|---|---|
| Split has no patient overlap (real ChestX-ray14 metadata, seed 42) | measured (metadata) | every overlap 0 | `docs/validation/partition_audit_real.json` |
| Kaggle rerun: the partition reproduces; a lite pipeline run | measured, Kaggle (not CI) | counts reproduce; lite AUC is a pipeline check, not a baseline | `docs/validation/kaggle_rerun_2026-10-04.json` |
| QBER sample-size sweep, Table III, Theorem 1 restatement | simulated | regenerated; alpha = 0, e = 0 is exactly 0 | `qber_sweep.json`, `table3_theorem1.json` |
| Fast NumPy backend equals the Qiskit reference | simulated | all cross-check cases pass | `backend_crosscheck.json` |
| Key rate vs block size | simulated (keys not secret) | no key at 1024 qubits; about 0.24 key bits per qubit at 2^17 with 1% noise | `key_rates.json` |
| Policy modes under Eve on one link | simulated, control plane only | per_client trains 69 to 80% of client-rounds where the global modes train 7 to 41% (alpha 0.44 to 0.6); says nothing about accuracy | `hysteresis_sweep.json`, ADR 0002 |
| Loader wait in the lite run | measured, Kaggle GPU | 92% of the loaded phases; GPU utilisation 7%; 45 img/s single-process decode against 366 img/s training compute | `kaggle_profile_2026-10-11.json` |
| Image cache build and verification | measured, Kaggle CPU | 112,120 images, 0 failures, 7.36 GB, 22.3 min, 0 pixel mismatches on 1,517 rows including all 519 RGBA sources; `cache_sha256` `db497ea6...` | `image_cache_build_2026-10-12.json` |
| Same lite run, cache against PNG | measured, Kaggle GPU, pipeline check | 78 s against 936 s; data wait 0.048 from round 2 against 0.92; GPU utilisation in the fit phase 94% against 16% | `image_cache_lite_run_2026-10-15.json` |
| Run-to-run variation, 3 cache-backend runs with the same seed | measured, Kaggle GPU, pipeline check | largest per-round spread: validation loss 0.0040, test loss 0.0034, validation macro AUC (non-thin) 0.141, test 0.098; the cache-versus-PNG differences are of the same size, and the PNG values fall inside the cache runs' range in 9 of 20 comparisons | `image_cache_lite_run_2026-10-15.json` (`run_to_run_variation`) |
| First full-data B0 pilot: single seed, E = 1, fp32, cache backend, 50 rounds | measured, Kaggle GPU, single-seed pilot (not a baseline) | test macro AUC **0.8086** at round 50 (primary, no selection); 0.8316 at the validation-selected round 25 (secondary); validation AUC peaked at round 26 and fell by 0.020 by round 50; 16,078 s, 17.5% above the projection | `pilot_b0_full_e1_seed0_2026-10-17.json` |
| Compute-only throughput of ResNet-18 on synthetic tensors, batch 32, one GPU | measured, Kaggle, one run per cell | 224 px training: fp32 362 img/s, `channels_last` 287, AMP 721, AMP + `channels_last` 962 (2.66x); 128 px fp32 941 (2.60x the 224 px rate, not the 3.06x assumed); peak memory 897 to 563 MiB with AMP | `compute_benchmark_2026-10-15.json` |

Observed but not yet recorded in a result file: a second independent cache build produced the same `cache_sha256`; folder-scan
time varied between 60 s and 560 s across sessions and build speed by about 30%.

## 5. Projections (projected, not runs)
From the measured lite-run compute rates (342 images/s training, fp32, one T4) and the measured speed-up ratios of the
compute benchmark; model in docs/06 (50 rounds, loader not limiting, one GPU). One full-data run at 224 px and 5 local
epochs is about 18 h in fp32; about 7 h with AMP and `channels_last` (throughput only, learning effect not measured). A 36-run
headline set at that profile is far beyond a few weeks of Kaggle quota, which is why a reduced profile for sweeps is planned.
The 2026-10-17 pilot took 17.5% longer than the projection for its profile (16,078 s against 13,680 s), so the hours here are low
by roughly that factor. The full table is in docs/06; the benchmark is `docs/validation/compute_benchmark_2026-10-15.json`. The measured 128 px ratio
(2.60x, not 3.06x) corrected the earlier 128 px hours upwards by about 15%.

## 6. What we do not claim
- Keys are simulated and seed-determined; no statement of key secrecy is made. The finite-key bound assumes basis-symmetric channels.
- QBER measures disturbance on the quantum link; it is not an attack classifier and does not observe interception of classical ciphertext.
- Secure aggregation does not exist yet as a protocol: only its primitives (fixed-point codec, mask PRG) are implemented and tested;
  "nothing in the clear" is shown for the sender object only.
- Masking is classical (X25519): no post-quantum guarantee, whatever the QKD link does.
- With 3 hospitals masking protects against the server alone; every client can already derive the sum of the others.
- In secure mode only gross blow-ups of the aggregate are discarded; stealthy, cancelling and just-under-threshold updates are not caught.
- **Dropout aborts the round and there is no secret-share recovery.** The design is pairwise masking, not the full Bonawitz et al.
  secure-aggregation protocol, which recovers from dropouts with secret sharing. A single failing or malicious client can force
  repeated aborts, which is logged per client (ADR 0003, "Dropout").
- **With 3 clients a server that colludes with one client learns the sum of the other two clients' updates** (the honest
  anonymity set is 2); with two colluding clients the third update is revealed in full. Neither case is defended.
- **Under secure aggregation the server cannot compute per-client update norms,** so the median/MAD screening that scores
  anomalies in CAUTION is unavailable (each round log says `screening: unavailable_secagg`), and **the Groth16 Byzantine claim is
  not supported:** the circuit is a test-size n = 8 and is not bound to the submitted update (the redesign is the last item of
  Phase 4). The only server-side defence in secure mode is the aggregate-norm check, with the limits listed above.
- The only accuracy figure is a single-seed pilot at one local epoch (test macro AUC 0.8086); there is no baseline with seeds and
  confidence intervals, no membership-inference result and no overhead table exist yet; Section VIII numbers of the paper draft are projected.
- The 5-hospital run is five disjoint patient-level partitions of one dataset, not five institutions.
- Novelty rests on prior work cited as QKDFL and FedSec; both citations still have to be verified by DOI (docs/01).
- No clinical validation and no regulatory claim of any kind.

## 7. Open decisions and next steps
| Item | Owner | State |
|---|---|---|
| SecureAggregator step 2 (primitives, seed provider, aggregator; ADR 0003 tests 1 to 3, 8, 10 to 17) | engineering | plan approved 2026-10-16 (S13); 2a in progress |
| First full-data baseline pilot (B0, seed 0, E = 1, fp32, cache backend) | engineering and project lead | done: run 2026-10-17, result recorded 2026-10-18 (section 4) |
| Number of rounds for the sweep profile (the pilot's validation AUC peaked at round 26 of 50; one seed) | project lead | open |
| Wire AMP and `channels_last` into training behind flags, then measure the learning effect against the run-to-run spread | project lead | awaiting a decision |
| Reduced profile for sweeps (25% of patients, 1 epoch); whether to define a 128 px pipeline and cache | project lead | proposed |
| How many full-profile headline runs and which epochs | project lead | open |
| Partition audit for 5 clients and docs/06 pre-registration of the 5-hospital run | engineering | not started |
| Paper text patches; verification of citations by DOI; team metadata (guide name) | team | open (docs/04 P1-10, P2-8) |

## 8. Questions a reviewer or investor will ask
**What is new here?** Using a live QBER reading to decide, per hospital and per round, whether a federated update is used, with a
graduated response instead of a binary pause. Implemented and tested in simulation. The advantage over a binary pause exists only
when Eve is on some links and not others, and it is a count of client-rounds, not accuracy (ADR 0002). The prior-work citations
are not yet verified.

**Is the quantum part real?** It is a simulator behind an interface designed so a real key-delivery system (ETSI GS QKD 014)
could replace it. No hardware has been used and the simulated keys are not secret.

**What accuracy do you get?** One measured figure, labelled a single-seed pilot: unsecured FedAvg across 3 simulated hospitals, full
data, one local epoch per round, 50 rounds, test macro AUC 0.8086 over the 14 pathologies (patient-level split, one seed, no
confidence interval). It is not the paper baseline and cannot be compared directly with published ChestX-ray14 numbers, which use
other splits. The protocol produces mean and 95% CI over at least 3 seeds (docs/06); the lite runs only check the pipeline.

**Could patient data leak between your train and test sets?** The split is by patient and was audited on the real metadata with
every overlap 0 (`partition_audit_real.json`). The earlier image-level split did leak, which is why it was replaced (docs/04 P0-4).

**Is the server blind to individual hospitals' updates?** Not yet. The design is accepted (pairwise masking, a registry of
identity keys, signed advertisements, client-side weights) and the first step, link protection, is implemented as a library.
When built, with 3 hospitals it protects against the server alone and its Byzantine defence is limited to gross blow-ups (section 6).

**Is it post-quantum secure?** No claim is made. The link key comes from QKD, whose security does not rest on computational
assumptions, but the masking layer uses X25519 and is classical. The combiner is ready for a post-quantum KEM that is not
implemented.

**Can you reproduce your numbers?** Seeds, config hashes, library versions and the data and cache hashes are recorded in every
results JSON. GPU runs with the same seed are not bit-identical (three lite runs differed by up to 0.004 in loss), so the protocol
uses several seeds. CI covers the code; Kaggle runs are a different software environment and are labelled.

**What will the full experiments cost?** Projected: about 7 GPU hours per full-data run at 224 px and 5 epochs with AMP, about 18
in fp32; a reduced profile for the sweeps is about 0.4 h per run. Rates are measured at lite scale; full-data runs have not been
made.

**What did you get wrong and fix?** Examples recorded in docs/04: FedProx crashed on a parameter mismatch (P0-1), the screening
rule was ineffective with 3 clients (P0-2), no global AUC was evaluated (P0-3), the split leaked patients (P0-4), Table III's
alpha = 0 entry was not exactly 0 (P1-6), and older sections of the paper draft are projected, not measured (P1-7).

**How do you protect patient data in this project?** The dataset is public and de-identified; images never leave the simulated
hospital in the design, and only model parameters are exchanged. Nothing here has been evaluated for clinical or regulatory use.
