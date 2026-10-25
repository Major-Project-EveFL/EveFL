# EveFL

**Quantum channel-aware federated learning for multi-hospital chest X-ray classification.**

EveFL reads the quantum bit error rate (QBER) of a **simulated** BB84 key exchange on each hospital's link to the server, and uses it to decide, round by round and link by link, whether a hospital's model update is used. The response is graduated (use it, use it with care, set it aside) instead of a binary pause.

> **Final-year major project**, Atria Institute of Technology (VTU), Dept. of AI&ML (SDGs 3, 9, 17). Its limits are part of the result: the quantum layer is a simulator, its keys are seed-determined and not secret, nothing here has been run on QKD hardware, and the classifier is not diagnostic.

---

## What EveFL does (and does not do)

> **The quantum layer monitors the communication channel, not the data.**

Hospitals train a ResNet-18 on their own chest X-ray images and send model updates to a central server. Before each round, a simulated BB84 exchange runs on each hospital's link. If an eavesdropper (Eve) intercepts photons she raises the QBER, and the controller reacts:

| QBER | State | What the code does |
|------|-------|--------------------|
| < 5% (configurable heuristic) | **SECURE** | plain weighted FedAvg |
| 5% up to 11% | **CAUTION** | FedProx local training (mu = 0.01), learning rate halved, median/MAD anomaly down-weighting of flagged updates (needs at least 3 updates) |
| >= 11% (fixed, never configurable below 0.11) | **LOCKDOWN** | in the global modes the whole round is discarded: the last good model is kept and re-keying is flagged as recommended; in `per_client` mode only the affected link's client is excluded (not sent the model, update never aggregated), and the round is discarded only below `min_clients` |

The 11% constant is the BB84 asymptotic bound (Shor-Preskill 2000). Three policy modes exist behind one interface: `global_binary`, `global` and `per_client` (see `docs/adr/0002-per-client-decisions-and-hysteresis.md`). The X-ray images never leave a hospital and the quantum simulation never touches them.

**Not yet part of the training path:** link encryption (HKDF + AES-256-GCM) and secure aggregation exist as tested libraries and demonstrations, but are not wired into the federated-learning loop. CKKS and the Groth16 norm proof are optional modules that are not integrated.

---

## Status

Labels: **measured** = a run that produced a result file; **simulated** = a model or simulator we wrote; **library** = implemented and tested, not wired into training.

| Area | State | Label / evidence |
|------|-------|------------------|
| BB84 simulation with a channel model (noise, intercept-resend) | implemented | simulated; exact NumPy backend (default) and a Qiskit reference, cross-checked |
| QBER controller, three policy modes, hysteresis | implemented, 100% branch coverage of the controller and policy modules | simulated |
| Key layer (separate 2^17-qubit block, Cascade, Toeplitz privacy amplification, finite-key bound) | implemented | simulated |
| Link protection (bound HKDF, AES-256-GCM, single use, zeroization) | tested library | library; not wired into the FL path |
| Secure aggregation (pairwise masking) | fixed-point codec and mask generator built; seed provider and aggregator not built | library; demonstration page only |
| Federated pipeline (Flower strategy and client, sequential runner, patient-level Dirichlet(0.5) partition, uint8 image cache) | implemented | audited on the real metadata |
| Accuracy | two single-seed runs, unsecured FedAvg, 3 simulated hospitals, 1 local epoch, 50 rounds: test macro AUC 0.8086 (seed 0) and 0.8114 (seed 1) at round 50 | **measured**, Kaggle; no confidence interval, not a baseline, not comparable with published ChestX-ray14 numbers |
| Dashboard | 7 pages | see below |
| Docker packaging, mutual-TLS client/server processes, post-quantum key exchange | not started | |

The full decision log, results and limits (with a reviewer Q&A) are in [docs/DECISIONS.md](docs/DECISIONS.md); the item-by-item status is in [docs/PROJECT_TRACKER.md](docs/PROJECT_TRACKER.md).

---

## Demo app

```bash
python -m evefl.webapp          # opens http://localhost:8765
```

A small web app (standard library server, hand-written front end, no internet needed) that binds to the loopback interface only. Today it has **Mission Control**: the headline numbers read from the recorded results and a live system check that runs the real code (controller thresholds, BB84 simulator, link protection, secure-aggregation sum, recorded audit and runs, model load). The remaining pages are being ported from the Streamlit dashboard below, which stays as the fallback.

## Streamlit dashboard (fallback)

```bash
streamlit run dashboard/app.py
```

| Page | What it shows | Label |
|------|---------------|-------|
| BB84 / QBER Explorer | a key exchange with a configurable Eve | simulated |
| 3-State Security Controller | how QBER maps to SECURE / CAUTION / LOCKDOWN | simulated |
| Crypto Pipeline Demo | AES-256-GCM on an update (CKKS and Groth16 are stubs) | AES real, others stub |
| Results explorer | recorded runs: validation and test AUC per round, per-class AUC, settings | recorded results only |
| Attack lab | Eve on one link, the three policies compared | simulated control-plane counts, not accuracy |
| Secure aggregation | masked updates, the exact sum, a dropout, a colluding server | library demonstration, not wired into training |
| Try the model | score a sample or an uploaded chest X-ray | research prototype, **not diagnostic** |

"Try the model" needs a saved checkpoint in `checkpoints/` and (for samples) images in `demo_samples/`. Both are git-ignored; see [docs/DEMO.md](docs/DEMO.md) and [docs/KAGGLE_RUNBOOK.md](docs/KAGGLE_RUNBOOK.md).

---

## Architecture

```
evefl/
├── quantum/          # BB84 simulation: base interfaces, bb84_numpy (default), bb84 (Qiskit reference),
│                     #   key layer (postprocess, round_keys), seeding, theory, sweeps, validation
├── crypto/           # linkkeys + keymanager (bound HKDF, AES-256-GCM), secagg/ (fixed point, mask PRG),
│                     #   classical, optional ckks / homomorphic / zkproof / groth16
├── orchestration/    # state_machine + policy: pure logic, imports no qiskit, flwr or torch
├── fl/               # model, dataset, partition, imagecache, strategy, client, screening, schedule,
│                     #   channels, control_sim, runner (sequential), evaluation, checkpoint, server
├── webapp/           # the demo web app: stdlib server, JSON data layer, static front end
├── selfcheck.py      # the live system check (real library calls) and showcase.py (headline numbers)
├── inference.py      # loads a saved checkpoint and scores one image (used by the dashboard)
├── results_view.py   # reads recorded results for the explorer
├── attack_lab.py     # the attack lab's simulation
├── secagg_demo.py    # the secure-aggregation demonstration
├── config/settings.yaml
└── registry.py       # every pluggable component sits behind an ABC and is registered here
dashboard/            # Streamlit app and pages
scripts/              # smoke test, experiment runner, partition audit, image cache, checkpoint evaluation, sweeps
docs/                 # context, architecture, security model, known issues, roadmap, experiment protocol, decisions
tests/
```

**Per-round flow (global mode):**

```
configure_fit()
│
├─ BB84 exchange per client  →  QBER estimate per link (a sampled estimate, never the simulation's ground truth)
│
├─ policy  →  SECURE / CAUTION / LOCKDOWN decision (per round, or per client in per_client mode)
│
└─ each client is told: train, train with a FedProx term, or sit out

aggregate_fit()
│
├─ SECURE   → weighted FedAvg
├─ CAUTION  → FedAvg with anomaly down-weighting of flagged updates
└─ LOCKDOWN → global modes: discard the round, keep the last good model; per_client mode: exclude that link only
```

---

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Python 3.12 is pinned and every dependency version is pinned. Kaggle runs use a different environment (Python 3.13); see [docs/KAGGLE_RUNBOOK.md](docs/KAGGLE_RUNBOOK.md). `circom` and `snarkjs` are needed only for the Groth16 tests, which skip when they are missing.

## Quick start

```bash
pytest -q
python scripts/smoke_test.py --high-intercept-test
streamlit run dashboard/app.py
```

The full test suite takes several minutes. The QBER controller on its own:

```python
from evefl.orchestration.state_machine import StateController
from evefl.quantum.base import ChannelModel
from evefl.quantum.bb84_numpy import BB84NumpyProtocol   # fast and exact; "bb84" is the slow Qiskit reference

bb84       = BB84NumpyProtocol(seed=42)
controller = StateController()

# One exchange with 30% Eve interception (expected QBER = alpha/4 = 7.5%, plus sampling noise)
channel    = ChannelModel(intercept_probability=0.3, bit_flip_probability=0.0)
result     = bb84.run_exchange(n_qubits=1024, channel=channel)
transition = controller.update(result.qber)

print(f"QBER: {result.qber:.3f}  ->  State: {transition.new_state.value}")
# a 128-bit sample, so the estimate is noisy
```

Training on the real dataset (NIH ChestX-ray14, downloaded separately from [NIH Box](https://nihcc.app.box.com/v/ChestXray-NIHCC); no data is stored in this repository) is described, with the exact commands used for the measured runs, in [docs/KAGGLE_RUNBOOK.md](docs/KAGGLE_RUNBOOK.md). How numbers are produced is fixed in advance in [docs/06_EXPERIMENT_PROTOCOL.md](docs/06_EXPERIMENT_PROTOCOL.md).

---

## Configuration

Thresholds and active backends are in `evefl/config/settings.yaml`. The 5% CAUTION boundary is a configurable heuristic; the 11% LOCKDOWN threshold is guarded in code and cannot be set below 0.11.

```yaml
orchestration:
  thresholds:
    secure_max:  0.05   # QBER < 5%  -> SECURE
    caution_max: 0.11   # 5% <= QBER < 11% -> CAUTION; QBER >= 11% -> LOCKDOWN (never lower than 0.11)
```

---

## Let's connect

| | |
|:---|:---|
| **Uzma Taheen Khan** | 📧 uztaheenkhan05@gmail.com · [LinkedIn](https://www.linkedin.com/in/utk05) · [GitHub](https://github.com/utk05) |
| **Parineeta Rana** | 📧 parineetarana1@gmail.com · [LinkedIn](https://www.linkedin.com/in/parineeta-rana/) · [GitHub](https://github.com/Parineeta-2307) |
