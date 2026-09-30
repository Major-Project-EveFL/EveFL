# CLAUDE.md — EveFL

Quantum-aware adaptive orchestration for federated learning (FL) security. Two-person student
major project. Hospitals train ResNet-18 on NIH ChestX-ray14 locally; a simulated BB84 QKD link
gives per-client, per-round QBER, which drives a three-state aggregation policy.

## Read these first (`docs/`)
- `01_PROJECT_CONTEXT.md` — problem, framing (crypto-first, QBER orchestrates), status table, glossary
- `02_ARCHITECTURE.md` — target package layout, interfaces, per-round sequence
- `03_SECURITY_MODEL.md` — threat model, what QBER does and does not show, key pipeline
- `04_KNOWN_ISSUES.md` — audited P0/P1/P2 bug list (fix in order, each fix needs a test)
- `05_ROADMAP.md` — phases with acceptance criteria, kickoff prompts
- `06_EXPERIMENT_PROTOCOL.md` — baselines, attacks, metrics
- `docs/reference/` — paper, report and lit review (.docx/.pdf). Not source of truth for numbers.

The docs are an audit snapshot. Some P0 items may already be fixed in code (e.g. `_stable_seed`
and `evaluation.py` now exist). Verify against the code before treating an issue as open, and
update `docs/04` when you fix one.

## Honesty rules (important)
- Every number in the paper must come from a saved result file with config hash and seeds.
  Anything else is "projected" or removed. Never invent results, AUCs or overheads.
- QBER is a link-integrity signal on the QKD link, NOT a detector of gradient sniffing.
- The simulator is not real QKD security. AES-GCM, CKKS and Groth16 are NOT yet wired into the FL path;
  do not write docs/README/UI text implying they are.
- ChestX-ray14 has no COVID-19 label. No COVID claims.
- Novelty citations (QKDFL, FedSec) are unverified. Do not rely on them without checking the DOI.

## Layout
- `evefl/quantum/` BB84 (Qiskit Aer) + Eve intercept-resend (`base.py` QKDProtocol ABC)
- `evefl/crypto/` `classical.py` (HKDF + AES-256-GCM), `ckks.py` (TenSEAL), `groth16.py` + `circuits/` (norm proof)
- `evefl/orchestration/state_machine.py` — pure logic, must NOT import ML or quantum code
- `evefl/fl/` `model.py`, `dataset.py` (Dirichlet partition), `client.py`, `strategy.py` (`EveFLStrategy`),
  `runner.py` (Ray-free sequential driver), `server.py` (CLI entry), `evaluation.py`
- `evefl/config/settings.yaml` — thresholds/backends (secure_max 0.05, caution_max 0.11)
- `evefl/registry.py` — plug-in registry
- `dashboard/` Streamlit app (4 pages); `scripts/` `smoke_test.py`, `run_experiments.py`; `tests/`
- Notebooks (`evefl*.ipynb`) are Kaggle runs.

## States
QBER < 5% SECURE (FedAvg) · 5–<11% CAUTION (FedProx mu=0.01 + update screening) · >=11% LOCKDOWN
(discard round, keep last good model, re-key). System QBER = max over clients.

## Commands
Python 3.12 is pinned (Qiskit/Pydantic break on 3.14). Windows dev machine, Kaggle for GPU runs.
```bash
python -m venv .venv && source .venv/Scripts/activate   # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt && pip install -e .
pytest -v                                # tests/ (bb84, crypto, ckks, groth16, state machine)
python scripts/smoke_test.py             # synthetic-data FL smoke test; add --intercept 0.3, --high-intercept-test
python -m evefl.fl.server --data-root <cxr14> --partition-root <partitions> --num-rounds 12
python scripts/run_experiments.py --data-root <cxr14> --partition-root <partitions>   # scenarios A-D
streamlit run dashboard/app.py
```
`results/`, `data/`, `*.png`, checkpoints are gitignored. Never commit datasets or keys.

## Local environment (this Windows machine)
- `.venv` in the repo is a junction to `C:\Users\Parin\evefl-env\venv` (C: has the space). pip cache and temp live in
  `C:\Users\Parin\evefl-env\`. Use `.venv\Scripts\python.exe`; set `PIP_CACHE_DIR`/`TMP` there when installing.
- CPU-only torch 2.5.1. tenseal installs fine. `circom`/`snarkjs` are NOT installed, so `zk` tests auto-skip
  (markers `ckks`, `zk` in `tests/conftest.py`). Run the full suite on Kaggle/WSL.
- Baseline (2026-09-29, before any fixes): 22 passed, 6 errors (Groth16, no circom/snarkjs). With markers: 22 passed, 6 skipped.
- CI: `.github/workflows/ci.yml` runs two jobs on Linux with `--require-optional-tools` (skips become failures): tests, and Groth16 (circom@0.5.46 + snarkjs@0.4.27). Check runs via the public API (no `gh`).
- Tooling: `pip install -r requirements-dev.txt`, then `ruff check .` and `mypy` (typed core listed in pyproject). Both run in CI (lint job). Optional: `pre-commit install`. Ruff is lint-only; no auto-formatter yet (would rewrite every file).
- No `gh` CLI. Workflow: one branch per task, push to `parineeta`, open the PR on GitHub web. Tooling/CI PR comes after the P0 fixes.

## Gotchas
- Do not use `fl.simulation.start_simulation` (Ray deadlocks against the CUDA context on Kaggle).
  Use the sequential driver in `evefl/fl/runner.py`.
- Seed CUDA lazily; `set_global_seed` calling `torch.cuda` early caused a Kaggle deadlock.
- Do not use builtin `hash()` for seeds (varies per process). Use a stable hash.
- Pinned versions: flwr 1.13.0 (settled; 1.11.1 pinned numpy<2 and clashed with Kaggle), cryptography <43 for flwr, torch 2.5.1.
- FedAvg over BatchNorm buffers (`num_batches_tracked`) needs explicit handling.
- Split ChestX-ray14 by Patient ID, not image, to avoid leakage.

## Working conventions
- Work one docs/05 task at a time, in order. Add a test with every fix. Run pytest before finishing.
- Keep `state_machine.py` free of ML/quantum imports.
- Match existing style: typed, docstring-heavy modules that explain *why*.
- Confirm before committing or pushing.

## Team
Uzma Taheen Khan and Parineeta Rana. Guide: Dr. Pradeep Kumar (confirm). Paper is LaTeX IEEEtran.
Original remote: github.com/Major-Project-EveFL/EveFL. Personal repo:
github.com/Parineeta-2307/EveFL_Federated_Learning.
