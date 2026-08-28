"""
LIVE: 3-State Security Controller + Policy Demo

Interactive demo of how EveFL maps QBER to security states and policies.
"""

import sys
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from evefl.orchestration.state_machine import StateController, StateThresholds, SecurityState

st.set_page_config(page_title="Security Controller", page_icon="🛡️")

st.title("🛡️ 3-State Security Controller + Policy Demo")

st.markdown("""
EveFL's **StateController** takes the maximum QBER across all clients
and decides the system security state. This page lets you explore
that mapping interactively.

| QBER Range | State | Policy |
|------------|-------|--------|
| < 5% | **SECURE** | Normal FedAvg |
| 5% – 11% | **CAUTION** | FedProx (μ=0.01) + anomaly scoring |
| ≥ 11% | **LOCKDOWN** | Reject round, keep previous model |
""")

# ---------------------------------------------------------------------------
# Interactive QBER input
# ---------------------------------------------------------------------------

st.divider()

qber_input = st.slider(
    "System QBER (q_max)",
    0.0, 0.50, 0.03, step=0.005,
    format="%.3f",
    help="Drag to see how different QBER values change the security state.",
)

controller = StateController()
transition = controller.update(qber_input)
state = transition.new_state

# ---------------------------------------------------------------------------
# State display
# ---------------------------------------------------------------------------

col_state, col_policy = st.columns(2)

with col_state:
    st.subheader("Security State")

    if state == SecurityState.SECURE:
        st.success(f"### ✅ {state.value}")
        st.write("The quantum channel is clean. Proceed with standard federated learning.")
    elif state == SecurityState.CAUTION:
        st.warning(f"### ⚠️ {state.value}")
        st.write("Elevated QBER detected. Activating FedProx and anomaly scoring.")
    else:
        st.error(f"### 🚨 {state.value}")
        st.write("Critical QBER! Discarding this round and triggering re-keying.")

    if transition.changed:
        prev = transition.previous_state.value if transition.previous_state else "None"
        st.info(f"**State transition:** {prev} → {state.value}")

with col_policy:
    st.subheader("Applied Policy")

    if state == SecurityState.SECURE:
        st.markdown("""
        - **Aggregation:** Standard FedAvg
        - **Client loss:** BCEWithLogitsLoss only
        - **Proximal μ:** 0.0
        - **Anomaly scoring:** Disabled
        """)
    elif state == SecurityState.CAUTION:
        st.markdown("""
        - **Aggregation:** FedAvg with gradient-norm anomaly downweighting
        - **Client loss:** BCEWithLogitsLoss + FedProx term
        - **Proximal μ:** 0.01
        - **Anomaly scoring:** Enabled (flag if ||W|| > mean + 2σ)
        """)
    else:
        st.markdown("""
        - **Aggregation:** **REJECTED** — round discarded
        - **Global model:** Unchanged from previous accepted round
        - **Client loss:** N/A (training skipped)
        - **Post-action:** Fresh BB84 re-keying before resuming
        """)

# ---------------------------------------------------------------------------
# Threshold visualization
# ---------------------------------------------------------------------------

st.divider()
st.subheader("📊 Threshold Visualization")

import numpy as np
import pandas as pd

thresholds = StateThresholds()
qber_range = np.linspace(0.0, 0.30, 300)
states = []
for q in qber_range:
    s = controller.classify(q)
    states.append(1 if s == SecurityState.SECURE else (2 if s == SecurityState.CAUTION else 3))

df_thresh = pd.DataFrame({"QBER": qber_range, "State code": states})
st.line_chart(df_thresh.set_index("QBER"), use_container_width=True)

st.caption("""
**Legend:** 1 = SECURE, 2 = CAUTION, 3 = LOCKDOWN.
Vertical boundaries at 5% and 11%.
""")

st.metric("Secure threshold (max)", f"{thresholds.secure_max * 100:.1f}%")
st.metric("Caution threshold (max)", f"{thresholds.caution_max * 100:.1f}%")