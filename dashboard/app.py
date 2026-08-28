"""
EveFL Streamlit Dashboard — Main Entry Point.

Phase 5: Visualization and interactive demos.

Navigation:
    - LIVE: BB84 / QBER Explorer
    - LIVE: 3-State Security Controller
    - LIVE: Crypto Pipeline Demo (AES-GCM real, CKKS/Groth16 STUB)
    - STATIC: FL Results (reads results/fl_training_log.json)

The FL Results panel does NOT run training.
It only reads the JSON exported by the Kaggle notebook / server.py.

Usage:
    streamlit run dashboard/app.py
"""

import streamlit as st

st.set_page_config(
    page_title="EveFL Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔬 EveFL Dashboard")

st.markdown("""
**EveFL** — Quantum channel-aware adaptive orchestration for Federated Learning security.

This dashboard is divided into two areas:

### 🔴 LIVE Panels
These compute on-the-fly using the EveFL library:
- **BB84 / QBER Explorer** — Simulate quantum key exchange with configurable Eve attack
- **Security Controller** — See how QBER maps to SECURE / CAUTION / LOCKDOWN
- **Crypto Pipeline** — AES-256-GCM live demo (CKKS & Groth16 are stubs)

### 📊 FL RESULTS Panel
Reads static experiment output only:
- **FL Results** — Visualize `results/fl_training_log.json` without running training
""")

st.divider()

st.info("""
💡 **Tip:** Use the sidebar to navigate between pages.
The FL Results page expects `results/fl_training_log.json` to exist
(run the experiment first via `python scripts/run_experiments.py`).
""")

st.caption("EveFL — Privacy/security-aware federated learning for medical imaging.")