"""
Results explorer: measured results that already exist, shown with the label and caveats they were recorded with.

Reads the committed pilot record under docs/validation/ and any results JSON found under results/ (or uploaded). It does NOT run
training. Parsing lives in `evefl/results_view.py` (tested on the real file schema); this page only draws.
"""

import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from evefl import results_view as rv  # noqa: E402

st.set_page_config(page_title="Results explorer", page_icon="📊", layout="wide")
st.title("📊 Results explorer")
st.caption("Shows measured results that already exist. It does not run training. Every number carries the label it was recorded with.")

UPLOAD = "Upload a results file..."
options = {p.relative_to(REPO_ROOT).as_posix(): p for p in rv.find_runs(REPO_ROOT)}
choice = st.sidebar.selectbox("Result file", [*options, UPLOAD]) if options else UPLOAD

view = None
if choice == UPLOAD:
    uploaded = st.sidebar.file_uploader("A results JSON or pilot record", type=["json"])
    if uploaded is not None:
        try:
            view = rv.load_run(json.load(uploaded))
        except (ValueError, json.JSONDecodeError) as exc:
            st.error(f"Cannot read this file: {exc}")
    else:
        st.info("No result file found under docs/validation or results. Upload one in the sidebar, or run an experiment first.")
else:
    view = rv.load_run(options[choice])
if view is None:
    st.stop()

st.header(view.title)
st.warning(f"**{view.label}**")
for caveat in view.caveats:
    st.caption(f"• {caveat}")
for warning in view.warnings:
    st.warning(warning)

if view.headline:
    columns = st.columns(len(view.headline))
    for column, (name, value) in zip(columns, view.headline):
        column.metric(name, value)
    st.caption("AUC = macro AUC over the classes with at least 20 positives (non-thin classes); see the settings table below.")

rounds = view.rounds.set_index("round")


def two_series_chart(frame: pd.DataFrame, validation: str, test: str, y_title: str) -> alt.Chart:
    """Validation every round (line) and test at the rounds it was evaluated (line with markers); y axis not forced to zero."""
    val = frame[["round", validation]].rename(columns={validation: "value"}).assign(series="validation (every round)")
    tst = frame.dropna(subset=[test])[["round", test]].rename(columns={test: "value"}).assign(series="test (evaluated rounds only)")
    return (alt.Chart(pd.concat([val.dropna(subset=["value"]), tst]))
            .mark_line(point=True).encode(x=alt.X("round:Q", title="round"),
                                          y=alt.Y("value:Q", title=y_title, scale=alt.Scale(zero=False)),
                                          color=alt.Color("series:N", title=None)))

st.subheader("AUC per round")
st.caption("Macro AUC over classes with at least 20 positives (non-thin). Validation is evaluated every round and alone drives "
           "selection; the test split is evaluated only at the rounds shown.")
st.altair_chart(two_series_chart(view.rounds, "val_auc_star", "test_auc_star", "macro AUC (non-thin)"), use_container_width=True)

st.subheader("Loss per round")
st.altair_chart(two_series_chart(view.rounds, "val_loss", "test_loss", "loss"), use_container_width=True)

if rounds["system_qber"].notna().any() or rounds["state"].notna().any():
    st.subheader("Channel state per round")
    if rounds["system_qber"].notna().any():
        st.line_chart(rounds[["system_qber"]].rename(columns={"system_qber": "system QBER (fraction)"}))
    if rounds["state"].notna().any():
        st.write("Rounds per state:", rounds["state"].value_counts().to_dict())
        flags = [v for v in rounds["model_updated"].tolist() if isinstance(v, bool)]
        if flags:
            st.write("Rounds with the model updated:", sum(flags), "of", len(flags),
                     "| client-rounds excluded:", int(rounds["n_excluded"].fillna(0).sum()))

if view.per_class_auc:
    st.subheader("Per-class test AUC (last test evaluation)")
    st.bar_chart(pd.Series(view.per_class_auc, name="AUC").sort_values())
    if view.thin_classes:
        st.caption(f"Classes too thin to score reliably: {', '.join(view.thin_classes)}")

if view.timing:
    st.subheader("Time")
    st.bar_chart(pd.Series(view.timing, name="seconds"))

if view.checkpoints:
    with st.expander("Saved checkpoints (kept out of git)"):
        st.json(view.checkpoints)

st.subheader("Settings")
st.table(pd.DataFrame({"setting": list(view.settings), "value": [str(v) for v in view.settings.values()]}))

with st.expander("All rounds"):
    st.dataframe(view.rounds, use_container_width=True)
