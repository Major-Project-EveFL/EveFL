# Demo guide: 3 November 2026, college major-project evaluators

A run order, the numbers that may be quoted, and what must not be claimed. Every number below is labelled **measured**,
**simulated** or **projected**; the evidence for each is in [DECISIONS.md](DECISIONS.md) (sections 4 and 6) and the Q&A in its
section 8 is the place to rehearse answers. Nothing here is a clinical or security certification claim.

## The day before (about 30 minutes)
1. Fresh clone on the demo laptop, then `python3.12 -m venv .venv`, `pip install -r requirements.txt`, `pip install -e .`
   (setup in the [README](../README.md)). Run `pytest -q -x` once at home; it takes several minutes, so not on the day.
2. Copy the two git-ignored folders from the development machine: `checkpoints/` (the two `.pt` files, about 90 MB) and
   `demo_samples/` (12 public test-split images and `samples.json`). Without them "Try the model" shows a "no checkpoint" message.
3. `streamlit run dashboard/app.py`, then **open every page once**. The first load of a page takes about 15 to 20 seconds on the
   development laptop (imports); later loads are quick.
4. Turn the wifi off and click through once to confirm nothing needs the internet. Plug in the charger, set the display to never sleep.
5. Record a 5 to 8 minute screen capture of the full run order below as a backup, and keep it on the laptop.

## Run order (about 12 minutes)
| # | Page | Do | Say (one line) | Do not say |
|---|------|----|----------------|------------|
| 1 | Home | show the page list | "A simulated quantum channel monitor decides which hospital updates a federated model may use." | that the quantum part is real hardware |
| 2 | BB84 / QBER Explorer | raise Eve's interception and watch QBER rise | "Eve's interception raises the error rate by about a quarter of her interception probability." | that the keys are secret (they are seed-determined) |
| 3 | 3-State Security Controller | step through SECURE, CAUTION, LOCKDOWN | "Below 5% we train normally; 5 to 11% we train with care; at 11% or more we stop using that link. The 11% is the BB84 bound and is fixed." | that 5% is a proven bound (it is a configurable heuristic) |
| 4 | Attack lab | put Eve on one link, compare the three policies | "With Eve on one link, the per-client policy keeps training the clean hospitals; a binary pause stops the whole round." | that this improves accuracy: it counts client-rounds, simulated, accuracy was not measured |
| 5 | Secure aggregation | show masked updates, the exact sum, then the dropout and the colluding server | "The server sees only masked values and still gets the exact sum. This is a library demonstration; training does not use it yet." | that training is secured by it; that dropout is handled (it aborts the round) |
| 6 | Results explorer | open the seed-1 and seed-0 records | "Two single-seed runs on the real data: the validation curve peaks near round 26 to 27 and then declines." | that these are a baseline or comparable with published numbers |
| 7 | Try the model | pick a sample, then upload one image | "A research prototype: it gives 14 independent scores per picture. It is not diagnostic." | a diagnosis; that a high score means disease; cherry-picked successes |
| 8 | (close) | go back to the Results explorer or the README status table | "What is measured, what is simulated, and what is not built yet." | anything on the lists below |

On "Try the model", show a sample where the top score differs from the recorded label as well as one that agrees. The model's
test AUC is about 0.83, which means individual pictures are often wrong; saying so is part of the honest demonstration.

## Numbers that may be quoted
| Number | Label | Where it comes from |
|--------|-------|---------------------|
| Test macro AUC 0.8086 (seed 0) and 0.8114 (seed 1) at round 50, 14 pathologies | **measured**, Kaggle, single seed each, no confidence interval | `docs/validation/pilot_b0_full_e1_seed0_2026-10-17.json`, `docs/validation/pilot_b0_full_e1_seed1_2026-10-24.json` |
| The served model (round 27, best validation) scores test AUC 0.8301 | **measured**, one separate evaluation, selects nothing | the seed-1 record |
| Test AUC of the two seeds at the ten tested rounds differs by at most 0.0036 | **measured** | the seed-1 record |
| Setup: 3 simulated hospitals, unsecured FedAvg, 1 local epoch per round, 50 rounds, patient-level split, Dirichlet(0.5) | stated | docs/06 |
| Per-client policy trains 69 to 80% of client-rounds against 7 to 41% for the global modes (Eve on one link, alpha 0.44 to 0.6) | **simulated**, control plane only | ADR 0002, `docs/validation/hysteresis_sweep.json` |
| Data split verified on the real metadata: every patient overlap is 0 | **measured** | `docs/validation/partition_audit_real.json` |

## Never say
- "production ready", "certified", "FIPS compliant", "information-theoretically secure end to end".
- That the model diagnoses, screens or detects disease, or that it is clinically validated.
- That the paper's projected figures (baseline AUC 0.820, convergence by round 38) are results. They are projections and are to be replaced by measured values after the demo.
- That secure aggregation, link encryption, CKKS or the Groth16 proof protect the training run. They are libraries or stubs, not wired in.
- That the quantum keys are secret, or that anything ran on QKD hardware.
- That two seeds prove anything statistically.

## If something goes wrong
| Symptom | Fix |
|---------|-----|
| "No checkpoint found" on Try the model | `checkpoints/` is missing or empty: copy the two `.pt` files back, then reload the page |
| A page is slow to appear | it is the first load; wait 20 seconds. If it hangs, stop and restart `streamlit run dashboard/app.py` |
| A page shows an error | restart the app; if it persists, play the backup video and say so |
| An uploaded image is rejected | the page accepts PNG or JPEG up to 20 MB; use a sample image instead |
| The laptop fails | use the backup video; the repository and `docs/DECISIONS.md` are the written record |
