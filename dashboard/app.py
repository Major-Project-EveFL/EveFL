"""
EveFL Streamlit Dashboard — Main Entry Point.

Phase 5: Visualization and interactive demos.

Navigation:
    - LIVE: BB84 / QBER Explorer
    - LIVE: 3-State Security Controller
    - LIVE: Attack lab (Eve on a link, three policies compared; simulated control-plane counts)
    - LIVE: Crypto Pipeline Demo (AES-GCM real, CKKS/Groth16 STUB)
    - STATIC: Results explorer (the committed pilot record and any results JSON)

The Results explorer does NOT run training.
It only reads recorded results (docs/validation) and the JSON written by server.py.

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
- **Attack lab** — Put Eve on a link and compare the three round policies (simulated control-plane counts, not accuracy)
- **Crypto Pipeline** — AES-256-GCM live demo (CKKS & Groth16 are stubs)

### 📊 RESULTS Panel
Reads recorded experiment output only:
- **Results explorer** — the single-seed pilot and any results JSON, with the label and caveats each was recorded with
""")

st.divider()

st.info("""
💡 **Tip:** Use the sidebar to navigate between pages.
The Results explorer opens the committed pilot record by default; choose another results file, or upload one, in its sidebar.
""")

st.caption("EveFL — Privacy/security-aware federated learning for medical imaging.")
