# EveFL project tracker

What is done, what is left, and what each item is waiting on. Last updated: 2026-10-18. The decisions behind these items are in
[DECISIONS.md](DECISIONS.md); the phase definitions and acceptance criteria are in [05_ROADMAP.md](05_ROADMAP.md); the
authority for each audited bug is [04_KNOWN_ISSUES.md](04_KNOWN_ISSUES.md).

Status words: **DONE** (code, a passing test and, for numbers, a result file exist), **PARTIAL** (part exists, the rest is named),
**TODO** (not started), **WAITING** (blocked on a run or a decision by the project lead), **STRETCH** (optional; not needed for the
core claim). Sizes are relative effort, not dates: S = a day or less, M = a few days, L = a week or more, C = compute-bound
(hours on a GPU, not hours of typing).

Counts: DONE 25, PARTIAL 6, TODO 28, WAITING 5, STRETCH 4 (68 items)

## Phase 0: stabilise
| ID | Item | Status | Evidence / what is left |
|---|---|---|---|
| 0.1 | Audited bugs P0-1 to P0-6 and P0-8 fixed with tests | DONE | `docs/04_KNOWN_ISSUES.md` |
| 0.2 | P0-7 (`local_epochs` hardcoded, CUDA seeding): the epochs flag is honoured and the runner is single-process; the entry in docs/04 is stale | PARTIAL | S: update the docs/04 entry with the evidence (`evefl/fl/client.py`) |
| 0.3 | ruff, mypy, CI (lint, tests, Groth16 jobs, nightly validation) | DONE | `.github/workflows/ci.yml`, `.github/workflows/nightly-validation.yml` |
| 0.4 | pre-commit configuration | DONE | `.pre-commit-config.yaml` (present; not re-verified) |
| 0.5 | README matches reality (P2-1) | DONE | `README.md`: status table with measured / simulated / library labels, current architecture tree, the seven dashboard pages, and what is not wired into training |
| 0.6 | Small bugs P2-2 (missing f-string) and P2-5 (client floor hang) | TODO | S: each with a test first |
| 0.7 | Tests for strategy, client, dataset, aggregation, integration (P2-6) | PARTIAL | M: 46 test files; the dashboard has no tests |
| 0.8 | Team metadata consistent across documents (P2-8: guide name, one USN) | WAITING | S: needs the team to confirm |

## Phase 1: quantum layer (all simulated)
| ID | Item | Status | Evidence / what is left |
|---|---|---|---|
| 1.1 | Channel model with Eve and noise; fast NumPy backend cross-checked against Qiskit | DONE | `docs/validation/backend_crosscheck.json` |
| 1.2 | Post-processing: Cascade, verification hash, Toeplitz privacy amplification, finite-key bound | DONE | `docs/validation/key_rates.json`, docs/04 P1-2 |
| 1.3 | Table III regenerated and Theorem 1 restated and checked by simulation | DONE | `docs/validation/table3_theorem1.json`; paper text still to be patched |
| 1.4 | Pre-registered QBER sample-size sweep | DONE | `docs/validation/qber_sweep.json` |
| 1.5 | False-alarm rates measured under noise (P1-3) | PARTIAL | M: needs the noise sweep of docs/06 |
| 1.6 | ETSI GS QKD 014 client with a mock key-management server | TODO | M: the key-provider interface is the seam (`evefl/crypto/keymanager.py`) |

## Phase 2: controller
| ID | Item | Status | Evidence / what is left |
|---|---|---|---|
| 2.1 | Three-state controller, hysteresis, three policy modes, property tests, 100% branch coverage, forbidden-import test | DONE | `evefl/orchestration/state_machine.py`, ADR 0002 |
| 2.2 | Policy-mode and hysteresis sweep (control-plane statistics) | DONE | `docs/validation/hysteresis_sweep.json` |
| 2.3 | Accuracy cost of excluding a hospital under non-IID data | TODO | C: needs the Phase 5 training runs |

## Phase 3: federated-learning core
| ID | Item | Status | Evidence / what is left |
|---|---|---|---|
| 3.1 | Patient-level split verified on the real metadata; server-side AUC evaluation; robust screening; configurable epochs and optimiser; pretrained init | DONE | `docs/validation/partition_audit_real.json` |
| 3.2 | Profiling instrumentation and the uint8 image cache (11.9x faster lite run) | DONE | `docs/validation/image_cache_build_2026-10-12.json`, `docs/validation/image_cache_lite_run_2026-10-15.json` |
| 3.3 | First measured accuracy: single-seed pilot of B0 (pre-registered), test macro AUC 0.8086 at round 50, E = 1 | DONE | `docs/validation/pilot_b0_full_e1_seed0_2026-10-17.json`; a single seed, not the baseline |
| 3.4 | Measured baseline: unsecured FedAvg, at least 3 seeds, mean and 95% CI, with config hash | TODO | C: needs a decision on the profile (epochs, data fraction) |
| 3.6 | Second single-seed run with checkpoints (seed 1), pre-registered | DONE | `docs/validation/pilot_b0_full_e1_seed1_2026-10-24.json`, `tests/test_seed1_record.py`: test AUC* 0.8114 at round 50; the served round-27 checkpoint scores 0.8301 in one separate test evaluation |
| 3.5 | Checkpoint saving (`--save-model`: final and best-validation checkpoints); resume of a long run is still missing | PARTIAL | `evefl/fl/checkpoint.py`, `tests/test_checkpoint.py`; no resume |

## Phase 4: crypto integration
| ID | Item | Status | Evidence / what is left |
|---|---|---|---|
| 4.1 | Key manager and link encryption as a library (step 1) | DONE | `evefl/crypto/linkkeys.py`, `evefl/crypto/keymanager.py`, ADR 0003 |
| 4.2 | Link encryption wired into the FL path (P1-1) | TODO | M: part of step 3 |
| 4.3 | Secure aggregation 2a: fixed-point codec and mask PRG | DONE | `evefl/crypto/secagg/`, `tests/test_secagg_fixedpoint.py`, `tests/test_secagg_prg.py` |
| 4.4 | Secure aggregation 2b: seed provider (X25519, Ed25519, registry, signed advertisements, set checks) | TODO | M: next, tests first |
| 4.5 | Secure aggregation 2c: aggregator classes, masked-update type, server-blindness and dropout tests | TODO | M |
| 4.6 | Step 3: wire both into the strategy and runner; two-phase secure round; aggregate-norm check | TODO | L |
| 4.7 | Step 4: overhead of each layer (time and bytes) measured | TODO | M, C |
| 4.8 | End-to-end acceptance: the server verifiably cannot read an individual update | TODO | follows 4.6 |
| 4.9 | CKKS (chunked) behind the aggregator interface | STRETCH | optional per ADR 0003 |
| 4.10 | Zero-knowledge norm-proof redesign (the circuit is a test-size n = 8, not bound to the update) | TODO | L: last item of the phase |
| 4.11 | Post-quantum KEM (Kyber) | STRETCH | out of scope by decision S11; the key combiner is ready for it |

## Phase 5: experiments
| ID | Item | Status | Evidence / what is left |
|---|---|---|---|
| 5.1 | Experiment protocol, pre-registered rules and compute projection | DONE | `docs/06_EXPERIMENT_PROTOCOL.md` |
| 5.2 | Decide the reduced profile for sweeps (data fraction, epochs; whether to define a 128 px pipeline and cache) | WAITING | decision by the project lead |
| 5.3 | Decide whether to wire AMP and `channels_last` into training (2.66x throughput measured, learning effect not) | WAITING | S/M plus a small Kaggle check |
| 5.4 | Baselines B0 to B5 under the attack settings, several seeds | TODO | C: far beyond a few weeks of quota at the full profile (see the projection in docs/06) |
| 5.5 | Five-hospital simulated run and its partition audit | TODO | M, C |
| 5.6 | Membership-inference measurement on our own models | TODO | M, C |
| 5.7 | `evefl report`: every table and figure regenerated from result files | TODO | M |
| 5.8 | Every number in the paper traces to a result file and config hash | TODO | follows 5.4 to 5.7 |
| 5.9 | Decide the number of rounds for sweep runs (the pilot's validation AUC peaked at round 26 of 50; one seed, a hypothesis) | WAITING | decision by the project lead; a short run at 25 rounds would test it |

## Phase 6: productisation and demo
| ID | Item | Status | Evidence / what is left |
|---|---|---|---|
| 6.1 | Dashboard: four pages that render | PARTIAL | `dashboard/app.py`, `dashboard/pages/`: the results page now reads current files; the QBER page uses the Qiskit reference backend instead of the fast NumPy one, the controller page predates the policy modes, and the crypto page is an honest placeholder for CKKS and Groth16 |
| 6.2 | Dashboard reads the current results files (per-round state, QBER, exclusions, AUC curves, timing) | DONE | `evefl/results_view.py`, `dashboard/pages/4_fl_results.py`, `tests/test_results_view.py` (checked on a real tiny run) |
| 6.3 | Interactive attack lab on the real controller and policy code (Eve slider, per-link, hysteresis) | DONE | `evefl/attack_lab.py`, `dashboard/pages/5_attack_lab.py`, `tests/test_attack_lab.py`; simulated control-plane counts, not accuracy |
| 6.4 | Secure-aggregation demo page (masked updates look random; the sum is exact; dropout; the N = 3 collusion case) | DONE | `evefl/secagg_demo.py`, `dashboard/pages/6_secure_aggregation.py`, `tests/test_secagg_demo.py`: a simulated library demonstration, not wired into the FL path |
| 6.5 | Inference page: upload one chest X-ray, get 14 scores, clearly marked research-only | DONE | `evefl/inference.py`, `dashboard/pages/7_upload_xray.py`, `tests/test_inference.py`; checked with the real round-27 checkpoint and the 12 exported test images. The checkpoints and images are local and git-ignored, so a new machine needs them copied in |
| 6.6 | Hospital client and server as separate processes: Flower gRPC with mutual TLS | TODO | L |
| 6.7 | Local data ingestion for a real hospital (DICOM to the model's 224 px input, label mapping) | TODO | L: the dataset class expects the NIH folder and CSV layout |
| 6.8 | Docker and compose, Prometheus and W&B, hash-chained audit log | TODO | L |
| 6.9 | mkdocs site, SECURITY.md, SBOM, lockfile, release v0.1.0 | TODO | M |
| 6.10 | A fresh clone reproduces one scenario from the README alone | TODO | M: depends on 0.5 |
| 6.11 | Demo guide for 3 November: run order, quotable numbers, claims not to make, fallbacks | DONE | `docs/DEMO.md`; the backup screen recording is still to be made by the team |
| 6.12 | Design system, grouped navigation and Mission Control home with a live system check that runs the real code | DONE | `dashboard/theme.py`, `dashboard/app.py`, `dashboard/pages/0_mission_control.py`, `evefl/selfcheck.py`, `tests/test_dashboard_home.py` |
| 6.13 | Live network page: three hospitals, the server and Eve on any link, QBER gauges and states changing, driven by the real policy code | TODO | M |
| 6.14 | Evidence board: tests, modules, ADRs and validation records counted from the repository, both seeds' curves overlaid | TODO | M |
| 6.15 | Demo preflight script (go / no-go before the demo) and friendly error pages instead of tracebacks | TODO | S/M |
| 6.16 | Guided tour mode that walks the run order of docs/DEMO.md | TODO | S/M |

## Phase 7: stretch
| ID | Item | Status | Evidence / what is left |
|---|---|---|---|
| 7.1 | NSGA-II threshold supervisor, QBER forecasting, PNS attack model | STRETCH | |
| 7.2 | External security review of the threat model | STRETCH | |

## Paper and funding readiness
| ID | Item | Status | Evidence / what is left |
|---|---|---|---|
| 8.1 | One log of decisions, results and limits, with reviewer questions | DONE | [DECISIONS.md](DECISIONS.md) |
| 8.2 | Paper text patches for the changed claims | PARTIAL | `docs/paper_patches/`: Table III, Theorem 1, partition, post-processing, per-client policy are written; the rest waits for results |
| 8.3 | Citations verified by DOI (QKDFL, FedSec and the others; P1-10) | WAITING | S/M: the team must check each entry |
| 8.4 | Paper Section VIII numbers replaced by measured ones (P1-7) | TODO | follows Phase 5 |
| 8.5 | Mentor reference thesis and handoff material kept out of the repository | DONE | `.gitignore` |
| 8.6 | AFTER NOV 3: replace the paper's projected numbers (baseline AUC 0.820, convergence by round 38, the basis of the 17.5% time estimate) with measured values | TODO | numbers as listed by the project lead; the paper draft is not in the repository. Measured so far: a single-seed pilot at 0.8086 (E = 1) and a second single-seed run at 0.8114, see `docs/validation/pilot_b0_full_e1_seed0_2026-10-17.json` and `docs/validation/pilot_b0_full_e1_seed1_2026-10-24.json` |
| 8.7 | AFTER NOV 3: correct the other hour estimates in docs/06 (low by roughly 6 to 18% after two runs: 16,078 s and 14,485 s measured against 13,680 s projected) | TODO | S |

## What a hospital or a company can do today, and what they cannot
- **Today:** nothing with their own images. The three "hospitals" are three patient-level partitions of the public ChestX-ray14
  dataset on one machine. There is no upload path, no ingestion of DICOM, no hospital client program, no network deployment and no
  saved model to run on a new image.
- **In the design:** each hospital keeps its images in its own environment and trains locally; only parameter updates leave. That is
  items 6.6 and 6.7 plus secure aggregation (4.4 to 4.6), none of which exist yet.
- **For a demo day:** items 6.2, 6.3 and 6.4 are the honest core (real controller, real secure-aggregation library, real result
  files); item 6.5 adds an upload-an-X-ray moment but needs a saved model and must say that it is a research prototype.
