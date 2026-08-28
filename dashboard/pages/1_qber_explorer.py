"""
LIVE: BB84 / QBER Explorer

Interactive simulation of the BB84 quantum key distribution protocol.
Adjust Eve's intercept probability and watch QBER respond.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from evefl.quantum.bb84 import BB84Protocol

st.set_page_config(page_title="QBER Explorer", page_icon="🔐")

st.title("🔐 BB84 / QBER Explorer")

st.markdown("""
Simulate a BB84 quantum key exchange and observe how eavesdropping
introduces errors (QBER). This is the **ground-truth signal** that
drives EveFL's security controller.
""")

# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    n_qubits = st.slider("Number of qubits", 64, 4096, 1024, step=64)

with col2:
    intercept_probability = st.slider(
        "Eve intercept probability (α)",
        0.0, 1.0, 0.0, step=0.05,
        help="0.0 = no Eve, 1.0 = full intercept-resend",
    )

with col3:
    seed = st.number_input("RNG seed", min_value=0, max_value=99999, value=42)

run_button = st.button("▶️ Run BB84 Simulation", type="primary")

# ---------------------------------------------------------------------------
# Single simulation
# ---------------------------------------------------------------------------

if run_button:
    with st.spinner("Running BB84 simulation..."):
        protocol = BB84Protocol(seed=seed)
        result = protocol.run_exchange(
            n_qubits=n_qubits,
            intercept_probability=intercept_probability,
        )

    st.success("Simulation complete!")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("QBER", f"{result.qber:.4f}", delta=f"{result.qber * 100:.2f}%")
    c2.metric("Qubits sent", result.n_qubits_sent)
    c3.metric("Sifted bits", result.n_sifted)
    c4.metric("Eve active?", "Yes" if result.eavesdropper_active else "No")

    st.json({
        "qber": result.qber,
        "n_qubits_sent": result.n_qubits_sent,
        "n_sifted": result.n_sifted,
        "intercept_probability": result.intercept_probability,
        "eavesdropper_active": result.eavesdropper_active,
        "sample_fraction": result.metadata.get("sample_fraction"),
    })

# ---------------------------------------------------------------------------
# QBER sweep
# ---------------------------------------------------------------------------

st.divider()
st.subheader("📈 QBER vs Intercept Probability Sweep")

st.markdown("""
Run multiple simulations across a range of Eve intercept probabilities
to see the theoretical relationship: **QBER ≈ α / 4**.
""")

sweep_qubits = st.slider("Sweep qubits", 64, 4096, 512, step=64, key="sweep_q")
sweep_seed = st.number_input("Sweep seed", 0, 99999, 123, key="sweep_seed")

if st.button("▶️ Run Sweep", key="sweep_btn"):
    alphas = np.linspace(0.0, 1.0, 21)
    qbers = []

    progress = st.progress(0.0)
    for i, alpha in enumerate(alphas):
        protocol = BB84Protocol(seed=sweep_seed + i)
        res = protocol.run_exchange(n_qubits=sweep_qubits, intercept_probability=float(alpha))
        qbers.append(res.qber)
        progress.progress((i + 1) / len(alphas))

    df = pd.DataFrame({"alpha (intercept probability)": alphas, "QBER": qbers})
    df["theoretical QBER = α/4"] = alphas / 4.0

    st.line_chart(df.set_index("alpha (intercept probability)"))

    st.dataframe(df.style.format({"QBER": "{:.4f}", "theoretical QBER = α/4": "{:.4f}"}), use_container_width=True)