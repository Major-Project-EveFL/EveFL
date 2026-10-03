# EveFL

Quantum channel-aware federated learning (FL) for multi-hospital chest X-ray classification.
Atria Institute of Technology (VTU), Dept. of AI&ML, final-year major project. SDGs 3, 9, 17.
Team: Parineeta Rana, Uzma Taheen Khan. Guide: Dr. Pradeep Kumar.

Goal of this codebase: a reproducible, tested, security-reviewed library that an industry or
government team could evaluate and adopt. Today it is a research prototype. Your job is to close
that gap honestly, never by overclaiming.

## Read order (read only what the task needs)
1. docs/01_PROJECT_CONTEXT.md  what and why, what is real vs claimed
2. docs/04_KNOWN_ISSUES.md     audited bugs and paper/code mismatches, prioritized
3. docs/02_ARCHITECTURE.md     target design and interfaces
4. docs/03_SECURITY_MODEL.md   threat model and crypto requirements
5. docs/05_ROADMAP.md          phases and acceptance criteria
6. docs/06_EXPERIMENT_PROTOCOL.md  how paper numbers get produced

## Non-negotiable rules
1. LOCKDOWN threshold is 0.11 (BB84 asymptotic bound, Shor-Preskill 2000). Compare with >=.
   It may never be configured below 0.11. The 5% CAUTION boundary is a configurable heuristic.
2. One fresh key per client per round. Never reuse. Bind round id and client id into HKDF info
   and AEAD associated data. Zeroize key material after use.
3. Data is always non-IID Dirichlet(alpha=0.5), partitions are disjoint, and splits are by
   patient ID (never by image).
4. The QBER state controller is pure logic. It must not import qiskit, flwr or torch.
5. Every pluggable component sits behind an ABC, is registered in a Registry, and is selected
   through YAML config.
6. Honesty rule. Never say something is implemented, secure or measured unless code, a passing
   test and (for numbers) a result file exist. Every reported number is labelled measured,
   simulated or projected. Never present literature numbers as our results.
7. Never write "production ready", "certified", "FIPS compliant" or "information-theoretically
   secure end to end" in code, docs or paper text.
8. No secrets or datasets in git. No fixed entropy for trusted setup outside tests. Use
   `secrets` or `os.urandom` for anything cryptographic, never `random`.
9. Reproducibility. Explicit seeds, no builtin `hash()` for seeding, and no CUDA initialisation
   before process fork (see gotchas).
10. Ask before adding a dependency. Pin every version.

## Working agreement
- Use plan mode first for anything touching quantum, crypto, the controller or aggregation.
- For bugs, write the failing test first, then fix.
- Before saying "done", run `ruff check`, `mypy evefl`, `pytest -q` and report the result.
- Small commits, one concern each, conventional commit messages.
- When a change closes an item in docs/04_KNOWN_ISSUES.md, update that file in the same commit.
- Code is the truth for what exists. docs/05_ROADMAP.md is the truth for what to build next.
  If they conflict, flag it and ask. Do not silently choose.
- If blocked, ask one clear question. Do not invent requirements.

## Commands
```
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
pytest -q
python scripts/smoke_test.py --high-intercept-test
python scripts/run_experiments.py --data-root D --partition-root P --rounds 50 --output-dir results/
streamlit run dashboard/app.py
```

## Environment gotchas
- Python is pinned to 3.12 (pyproject). flwr==1.13.0 (1.11.1 pins numpy<2 and clashes with Kaggle's
  numpy 2 image), torch==2.5.1, qiskit==1.2.4, qiskit-aer==0.15.1, scipy==1.17.1,
  cryptography>=42.0.4,<43 (flwr constraint).
- Kaggle deadlock. `set_global_seed()` calls `torch.cuda.manual_seed_all()`, which creates a CUDA
  context in the parent before Flower/Ray forks workers, and the children deadlock. Workaround is
  a single-process manual FL loop that drives EveFLStrategy and EveFLClient directly. Make that
  runner first-class (`evefl/fl/runner.py`) and seed CUDA lazily.
- Circom is legacy 0.5.x (JS) plus snarkjs via subprocess. Needs `circom` and `snarkjs` on PATH.
- Kaggle's image is Python 3.13 / torch 2.11, not the pinned 3.12 environment. `requirements.txt` does not install there
  (no cp313 wheels for torchvision 0.20.1, qiskit-aer 0.15.1, or grpcio<=1.64.3 which flwr 1.13.0 requires). Use
  `pip install --no-deps -r requirements-kaggle.txt` and `pip install --no-deps -e .` (docs/KAGGLE_RUNBOOK.md). Runs there
  are a different software environment from CI: say so when quoting their numbers.
- Dataset images may sit in nested folders on Kaggle. Build a filename-to-path index once. Never
  `rglob` per image.
- tenseal==0.3.17 in requirements.txt must be verified to exist for py3.12.

## Local environment (Parineeta's Windows machine)
- `.venv` in the repo is a junction to `C:\Users\Parin\evefl-env\venv` (C: has the space). pip cache and temp
  live in `C:\Users\Parin\evefl-env\`; set `PIP_CACHE_DIR` and `TMP` there when installing. Use
  `.venv\Scripts\python.exe`. CPU-only torch 2.5.1. `circom`/`snarkjs` are not installed locally, so the `zk`
  tests auto-skip here (markers `ckks`, `zk`, `slow` in tests/conftest.py and pyproject); CI runs them.
- Property tests use hypothesis: CI runs the derandomised profile (`HYPOTHESIS_PROFILE=ci`, fixed examples); locally the default
  profile explores randomly. CI enforces 100% branch coverage on `orchestration/state_machine.py` and `policy.py` only.
- Experiments: `--policy-mode global_binary|global|per_client` (B2/B3/B4), `--link-attack 0:0.6[:step@N|window@A-B|intermittent@P]`
  (Eve on ONE link), `--link-noise`, `--hysteresis-margin/--hysteresis-dwell`, `--min-clients`; the full channel plan and policy are
  written into the results JSON.
- Dev tools: `pip install -r requirements-dev.txt`; `ruff check .`; `mypy` (typed core listed in pyproject; the
  whole-package `mypy evefl` still has ~17 errors, mostly flwr typing, tracked as a separate PR).
- No `gh` CLI. Workflow: one branch per task, push to the `parineeta` remote
  (github.com/Parineeta-2307/EveFL_Federated_Learning), merge to main only when CI is green. Read CI results
  through the public API (`.github/workflows/ci.yml`: lint, tests, Groth16 jobs with `--require-optional-tools`,
  which turns skipped tests into failures). `origin` (Major-Project-EveFL/EveFL) is a read-only reference.
- Slow Qiskit-vs-numpy cross-check: `pytest -m slow` or the nightly workflow; evidence goes in `docs/validation/`
  (`results/` is gitignored).

## Repo map (current)
evefl/quantum (base: QKDProtocol + ChannelModel + shared registry | bb84_numpy: fast exact default | bb84:
Qiskit reference | seeding: independent streams from experiment seed/round/client | config: refuses QBER
samples under ~100 bits | factory | validation, sweep, theory: exact detection probabilities) |
evefl/crypto (base, classical, homomorphic, ckks, zkproof, groth16, circuits/norm_bound.circom) |
evefl/orchestration/state_machine.py (pure logic; LOCKDOWN threshold guarded >= 0.11) |
evefl/fl (model, dataset, partition: patient-level train/val/test split, client, strategy, screening, schedule,
runner, evaluation, server) | evefl/config/settings.yaml | evefl/registry.py | dashboard/ (Streamlit, 4 pages) |
scripts/ (smoke_test, run_experiments, audit_partition, qber_sweep, table3_theorem1, validate_backends,
summarize_run, cache_pretrained_weights) | docs/ (01-06, paper_patches/, validation/, KAGGLE_RUNBOOK.md) | tests/
Only `qber` (a sampled estimate) may drive the controller; `sim_only_*` metadata is simulation ground truth.
Data split: P0-4 is verified on the real ChestX-ray14 metadata (`docs/validation/partition_audit_real.json`: seed 42, all
overlaps 0). `partition_meta.json` stores `index_sha256` for every index file (`scripts/audit_partition.py` verifies it).
Model selection uses the VALIDATION split's `macro_auc_roc_excl_thin` (classes with < 20 positives are "thin"; rule in
docs/06); the test split is for the final report only.

## Style
Type hints everywhere, dataclasses or pydantic for data, no bare `except`, structured logging
via `logging` (no print in library code), docstrings state invariants and units (QBER is a
fraction in [0,1], never a percent).