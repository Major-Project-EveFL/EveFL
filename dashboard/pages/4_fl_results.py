"""
STATIC: FL Results Visualization

Reads results/fl_training_log.json (or any uploaded JSON) and renders
charts for QBER, security states, training metrics, and aggregation actions.

This page does NOT run training. It only visualizes pre-computed results.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

st.set_page_config(page_title="FL Results", page_icon="📊")

st.title("📊 FL Results")

st.markdown("""
Visualize EveFL experiment output. Upload a `fl_training_log.json`
or use the default path.
""")

# ---------------------------------------------------------------------------
# Load JSON
# ---------------------------------------------------------------------------

json_path = st.text_input(
    "Results JSON path",
    str(REPO_ROOT / "results" / "fl_training_log.json"),
)

uploaded = st.file_uploader("Or upload a JSON file", type=["json"])

data = None

if uploaded is not None:
    try:
        data = json.load(uploaded)
        st.success("Uploaded JSON loaded successfully.")
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON: {e}")
else:
    p = Path(json_path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        st.success(f"Loaded: {p}")
    else:
        st.warning(f"File not found: {p}")
        st.info("Run an experiment first: `python scripts/run_experiments.py`")

if data is None:
    st.stop()

# ---------------------------------------------------------------------------
# Experiment metadata
# ---------------------------------------------------------------------------

st.divider()
st.header("📋 Experiment Metadata")

meta = data.get("experiment", {})
cols = st.columns(4)
cols[0].metric("Name", meta.get("name", "N/A"))
cols[1].metric("Clients", meta.get("num_clients", "N/A"))
cols[2].metric("Rounds", meta.get("num_rounds", "N/A"))
cols[3].metric("Seed", meta.get("seed", "N/A"))

st.json(meta)

# ---------------------------------------------------------------------------
# Security rounds table
# ---------------------------------------------------------------------------

st.divider()
st.header("🛡️ Security Timeline")

rounds = data.get("security", {}).get("rounds", [])

if not rounds:
    st.warning("No round data found in JSON.")
    st.stop()

df = pd.DataFrame(rounds)

# State distribution pie chart
state_counts = df["state"].value_counts().to_dict()
state_df = pd.DataFrame({
    "State": list(state_counts.keys()),
    "Count": list(state_counts.values()),
})

col_chart, col_table = st.columns([1, 2])

with col_chart:
    st.subheader("State Distribution")
    st.bar_chart(state_df.set_index("State"))

with col_table:
    st.subheader("Round Details")
    display_df = df[["round", "state", "system_qber", "n_results", "aggregation"]].copy()
    st.dataframe(display_df, use_container_width=True)

# ---------------------------------------------------------------------------
# QBER over rounds
# ---------------------------------------------------------------------------

st.divider()
st.header("📈 QBER Over Rounds")

if "system_qber" in df.columns:
    qber_df = df[["round", "system_qber"]].copy()
    qber_df = qber_df.set_index("round")
    st.line_chart(qber_df)

# Per-client QBER if available
if "qber_per_client" in df.columns and df["qber_per_client"].notna().any():
    st.subheader("Per-Client QBER")
    qber_per = df["qber_per_client"].tolist()
    if qber_per and isinstance(qber_per[0], dict):
        qber_client_df = pd.DataFrame(qber_per)
        qber_client_df["round"] = df["round"].values
        qber_client_df = qber_client_df.set_index("round")
        st.line_chart(qber_client_df)

# ---------------------------------------------------------------------------
# Metrics over rounds
# ---------------------------------------------------------------------------

st.divider()
st.header("📉 Metrics Over Rounds")

if "metrics" in df.columns and df["metrics"].notna().any():
    metrics_list = df["metrics"].tolist()
    if metrics_list and isinstance(metrics_list[0], dict):
        metrics_df = pd.DataFrame(metrics_list)
        metrics_df["round"] = df["round"].values
        metrics_df = metrics_df.set_index("round")

        numeric_cols = metrics_df.select_dtypes(include=["number"]).columns.tolist()
        if numeric_cols:
            selected = st.multiselect("Select metrics to plot", numeric_cols, default=numeric_cols[:3])
            if selected:
                st.line_chart(metrics_df[selected])
        else:
            st.info("No numeric metrics available for plotting.")

# ---------------------------------------------------------------------------
# Raw JSON
# ---------------------------------------------------------------------------

st.divider()
st.header("📝 Raw JSON")
with st.expander("Click to expand full JSON"):
    st.json(data)