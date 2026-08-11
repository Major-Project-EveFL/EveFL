# EveFL

Quantum-aware adaptive orchestration for Federated Learning security.

EveFL uses parallel BB84 QKD simulation to extract real-time Quantum
Bit Error Rate (QBER) telemetry, driving a three-state adaptive
aggregation controller that protects a federated learning system from
eavesdropping and poisoning attacks — a transport-layer blind spot
that classical FL security (TLS, differential privacy) doesn't cover.

## Status

| Phase | Component | Status |
|---|---|---|
| 1 | BB84 QKD simulation (Qiskit) | ✅ implemented |
| 2 | Crypto stack (SHA-256 / HKDF / AES-256-GCM) | ✅ implemented |
| 3 | Three-state orchestration controller | ✅ implemented |
| 4 | Flower FL integration | ⏳ not started |
| 5 | Streamlit dashboard + W&B tracking | ⏳ not started |
| 6 | Docker packaging | ⏳ not started |
| stretch | Post-quantum fallback (CRYSTALS-Kyber) | ⏳ not started |

## Architecture

Each subsystem is built behind an abstract interface + registry so new
algorithms can be added without touching existing code:

```
evefl/
├── quantum/        # QKDProtocol interface + BB84 implementation
├── crypto/         # CipherSuite interface + classical (SHA256/HKDF/AES-GCM)
├── orchestration/  # Three-state QBER controller (pure logic, no ML deps)
├── fl/             # Flower client/server, ResNet-18, ChestX-ray14 (Phase 4)
├── dashboard/       # Streamlit UI (Phase 5)
├── tracking/        # W&B logging (Phase 5)
├── config/
│   └── settings.yaml   # thresholds, active backends — tune without touching code
└── registry.py      # generic plug-in registry used by all subsystems above
```

To add a new QKD protocol, cipher suite, or state-classification
scheme: implement the relevant base class, register it under a new
key, and flip `active_protocol` / `active_suite` in `settings.yaml`.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Running tests

```bash
pytest -v
```

## Requirements

- Python 3.12 (pinned — Qiskit/pydantic dependency conflicts observed on 3.14)
