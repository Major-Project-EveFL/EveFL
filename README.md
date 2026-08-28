# EveFL

**Quantum channel-aware adaptive orchestration for Federated Learning security.**

EveFL monitors the quantum bit error rate (QBER) of a simulated BB84 key exchange to detect eavesdropping in real time and respond with a graduated three-state aggregation policy — without ever halting training unnecessarily.

---

## What EveFL actually does (and does not do)

> **The quantum layer monitors the communication channel, not the data.**

In standard FL, hospitals train ResNet-18 on their local chest X-ray images and send gradient updates to a central server. Those gradients travel over a network channel that classical TLS encrypts but *cannot* **monitor** for active interception.

EveFL adds a parallel BB84 quantum key distribution simulation. Before every FL round, Alice (client) and Bob (server) exchange simulated photons. If Eve intercepts photons, she introduces errors in Bob's measurements — a signal that appears as elevated Quantum Bit Error Rate (QBER). EveFL uses this QBER as a live orchestration signal:

| QBER | State | What EveFL does |
|------|-------|-----------------|
| < 5% | **SECURE** | Standard FedAvg with AES-256-GCM encrypted gradients |
| 5 – 11% | **CAUTION** | FedProx (proximal term μ=0.01) + gradient norm anomaly scoring |
| ≥ 11% | **LOCKDOWN** | Discard round, keep previous model, trigger BB84 re-keying |

The X-ray images never leave their hospital. ResNet-18 trains on them locally. The quantum simulation never touches the image data.

---

## Architecture

```
evefl/
├── quantum/              # BB84 QKD simulation (Qiskit) — QBER signal generator
│   ├── base.py           # QKDProtocol abstract interface
│   └── bb84.py           # BB84 + Eve intercept-resend with configurable alpha
├── crypto/               # Gradient encryption
│   ├── base.py           # CipherSuite abstract interface
│   └── classical.py      # SHA-256 → HKDF-SHA256 → AES-256-GCM
├── orchestration/        # Three-state controller (pure logic, no ML deps)
│   └── state_machine.py
├── fl/                   # Flower FL integration (Phase 4)
│   ├── model.py          # ResNet-18 with 14-output sigmoid head
│   ├── dataset.py        # ChestX-ray14 loader + Dirichlet(α=0.5) partitioner
│   ├── strategy.py       # QBER-aware Flower Strategy ← wires everything together
│   ├── client.py         # Flower NumPyClient (upcoming)
│   └── server.py         # Entry point (upcoming)
├── dashboard/            # Streamlit UI (Phase 5)
├── tracking/             # W&B logging (Phase 5)
├── config/
│   └── settings.yaml     # Thresholds, active backends — tune without touching code
└── registry.py           # Plug-in registry used by all subsystems
```

**Per-round flow:**

```
configure_fit()
│
├─ BB84 engine × N clients → [QBER_0, QBER_1, QBER_2]
│
├─ max(QBER) → StateController → SECURE / CAUTION / LOCKDOWN
│
└─ FitIns injected with {state, proximal_mu, round_id} → clients

aggregate_fit()
│
├─ SECURE  → weighted FedAvg
├─ CAUTION → anomaly scoring + downweight suspicious gradients
└─ LOCKDOWN → return last good model, trigger BB84 re-key
```

---

## Status

| Phase | Component | Status |
|-------|-----------|--------|
| 1 | BB84 QKD simulation (Qiskit) + configurable Eve alpha | ✅ Done |
| 2 | Crypto stack (SHA-256 / HKDF / AES-256-GCM) | ✅ Done |
| 3 | Three-state QBER controller | ✅ Done |
| 4 | Flower FL integration | ✅ Done |
| 4a | ResNet-18 model (`fl/model.py`) | ✅ Done |
| 4b | ChestX-ray14 dataset + Dirichlet partitioner (`fl/dataset.py`) | ✅ Done |
| 4c | QBER-aware Strategy (`fl/strategy.py`) | ✅ Done |
| 4d | Flower client (`fl/client.py`) | ✅ Done |
| 4e | Server entry point (`fl/server.py`) | ✅ Done |
| 5 | Streamlit dashboard + W&B tracking | 🔄 In progress |
| 6 | Docker packaging | 🔄 In progress |
| Stretch | Post-quantum fallback (CRYSTALS-Kyber) | ⏳ Not started |

---

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

> Python 3.12 is pinned — Qiskit and Pydantic have dependency conflicts on 3.14.

Uncomment the Phase 4+ dependencies in `requirements.txt` before running FL:

```
flwr==1.11.1
torch==2.5.1
torchvision==0.20.1
```

---

## Quick start

**Run tests (Phases 1–3):**

```bash
pytest -v
```

**Partition the dataset once before FL training:**

```python
from evefl.fl.dataset import partition_and_save

partition_and_save(
    data_root      = "/path/to/chestxray14",
    partition_root = "/path/to/chestxray14/partitions",
    subset_fraction = 0.15,   # use 0.15 for fast debugging, 1.0 for full eval
)
```

**Try the QBER controller standalone:**

```python
from evefl.quantum.bb84 import BB84Protocol
from evefl.orchestration.state_machine import StateController

bb84       = BB84Protocol(seed=42)
controller = StateController()

# Simulate a round with 44% Eve interception
result     = bb84.run_exchange(n_qubits=1024, eve_intercept_rate=0.44)
transition = controller.update(result.qber)

print(f"QBER: {result.qber:.3f}  →  State: {transition.new_state.value}")
# QBER: 0.107  →  State: CAUTION
```

---

## Configuration

All thresholds and active backends are in `evefl/config/settings.yaml`. No code changes needed to retune the state machine or swap the cipher suite.

```yaml
orchestration:
  thresholds:
    secure_max:  0.05   # QBER < 5%  → SECURE
    caution_max: 0.11   # 5% ≤ QBER < 11% → CAUTION
                        # QBER ≥ 11% → LOCKDOWN
```

---

## Requirements

- Python 3.12 (pinned)
- NIH ChestX-ray14 dataset — download from [NIH Box](https://nihcc.app.box.com/v/ChestXray-NIHCC)

---

## Let's connect

| | |
|:---|:---|
| **Uzma Taheen Khan** | 📧 uztaheenkhan05@gmail.com · [LinkedIn](https://www.linkedin.com/in/utk05) · [GitHub](https://github.com/utk05) |
| **Parineeta Rana** | 📧 parineetarana1@gmail.com · [LinkedIn](https://www.linkedin.com/in/parineeta-rana/) · [GitHub](https://github.com/Parineeta-2307) |
