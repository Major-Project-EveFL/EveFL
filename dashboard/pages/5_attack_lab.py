"""
LIVE: Attack lab. Put Eve on a link and watch the real controller and the three round policies respond.

Runs `evefl.attack_lab` (the real policy code on the exact count-level QBER model). Every number is a SIMULATED control-plane
statistic: rounds and client-rounds, not accuracy. No model is trained here.
"""

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from evefl import attack_lab as lab  # noqa: E402

try:
    st.set_page_config(page_title="Attack lab", page_icon="🛡️", layout="wide")
except st.errors.StreamlitAPIException:
    pass  # dashboard/app.py has already set it for the whole app (set_page_config is allowed once per run)
st.title("🛡️ Attack lab")
st.warning(lab.LABEL)
st.caption("QBER on a link under an intercept-resend Eve with intercept probability alpha and baseline noise e: "
           "expected QBER = e + (1 - 2e) * alpha / 4. SECURE below 5%, CAUTION from 5%, LOCKDOWN from 11% (the BB84 bound, "
           "never lowered). The 3 hospitals are simulated links.")

# ---------------------------------------------------------------- controls
st.sidebar.header("Eve")
target = st.sidebar.selectbox("Attacked link (hospital)", ["none", "0", "1", "2"], index=1)
alpha = st.sidebar.slider("Eve's intercept probability (alpha)", 0.0, 1.0, 0.6, 0.05)
kind = st.sidebar.selectbox("Attack pattern", ["static", "window", "step", "intermittent"],
                            help="static: every round. window: only between two rounds. step: from a round onward. "
                                 "intermittent: each round with some probability.")
rounds = st.sidebar.slider("Rounds", 10, 100, 50, 5)
start = st.sidebar.slider("First attacked round", 1, rounds, min(21, rounds)) if kind in ("window", "step") else 1
end = st.sidebar.slider("Last attacked round", start, rounds, min(max(30, start), rounds)) if kind == "window" else rounds
probability = st.sidebar.slider("Chance Eve is active in a round", 0.05, 1.0, 0.3, 0.05) if kind == "intermittent" else 1.0
noise = st.sidebar.slider("Baseline channel noise (bit-flip probability)", 0.0, 0.05, 0.01, 0.005)

st.sidebar.header("Controller")
modes = st.sidebar.multiselect("Policies to compare", list(lab.DEFAULT_MODES), default=list(lab.DEFAULT_MODES),
                               help="global_binary = pause at the worst link's 11% (the QKDFL-style baseline B2); "
                                    "global = three states for the whole system (B3); per_client = exclude only the attacked link (B4).")
margin = st.sidebar.slider("Hysteresis margin (QBER fraction)", 0.0, 0.03, 0.0, 0.005,
                           help="How far below a boundary a reading must be to count towards leaving a state. 0 = off.")
dwell = st.sidebar.slider("Hysteresis dwell (rounds)", 1, 5, 1,
                          help="Consecutive qualifying readings needed to de-escalate. 1 = off. Escalation is never delayed.")
min_clients = st.sidebar.slider("Minimum clients to aggregate", 1, 3, 2)

st.sidebar.header("Simulation")
reps = st.sidebar.slider("Simulated runs behind the table", 50, 1000, 300, 50)
seed = int(st.sidebar.number_input("Seed", min_value=0, max_value=10_000, value=0, step=1))

if not modes:
    st.info("Choose at least one policy in the sidebar.")
    st.stop()

settings = lab.LabSettings(
    attacked_link=None if target == "none" else target, alpha=alpha, attack_kind=kind, attack_start=start, attack_end=end,
    attack_probability=probability, noise=noise, rounds=rounds, reps=reps, seed=seed, hysteresis_margin=margin,
    hysteresis_dwell=dwell, min_clients=min_clients, modes=tuple(modes))


@st.cache_data(show_spinner="Simulating...")
def cached_run(s: lab.LabSettings) -> lab.LabResult:
    return lab.run_lab(s)


result = cached_run(settings)

if settings.attacked_link is None or alpha == 0.0:
    st.info("No attack is active: this is the baseline. With 1% noise a false LOCKDOWN is very rare, so every policy trains every round.")

# ---------------------------------------------------------------- comparison
st.header("What happens to training")
st.caption(f"Mean over {result.reps} simulated runs of {rounds} rounds, same sampled channels for every policy. "
           "Client-rounds = hospital-rounds that were trained and aggregated (3 hospitals x rounds is the maximum).")
nice = result.summary.rename(columns={
    "rounds_discarded_pct": "rounds discarded (%)", "client_rounds_trained_pct": "client-rounds trained (%)",
    "attacked_client_rounds_trained": "attacked hospital: rounds trained", "caution_share_pct": "link-rounds in CAUTION (%)",
    "lockdown_share_pct": "link-rounds in LOCKDOWN (%)", "state_changes_per_link_round": "state changes per link-round"})
st.dataframe(nice.round(2), use_container_width=True)
chart_data = nice[["client-rounds trained (%)", "rounds discarded (%)"]].reset_index().melt("policy mode", var_name="measure")
st.altair_chart(alt.Chart(chart_data).mark_bar().encode(
    x=alt.X("policy mode:N", title=None, sort=list(modes)), y=alt.Y("value:Q", title="percent", scale=alt.Scale(domain=[0, 100])),
    color=alt.Color("measure:N", title=None), xOffset="measure:N"), use_container_width=True)
for note in result.notes:
    st.caption(f"• {note}")

# ---------------------------------------------------------------- one run
st.header("One simulated run, round by round")
shown_mode = st.selectbox("Policy", list(modes), index=len(modes) - 1)
trace = result.trace[shown_mode]
qber = alt.Chart(trace).mark_line(point=True).encode(
    x=alt.X("round:Q", title="round"), y=alt.Y("qber:Q", title="estimated QBER (fraction)", scale=alt.Scale(domain=[0, 0.3])),
    color=alt.Color("link:N", title="link"))
rules = alt.Chart(pd.DataFrame({"y": [0.05, 0.11], "label": ["CAUTION from 5%", "LOCKDOWN from 11%"]})).mark_rule(
    strokeDash=[6, 4], color="gray").encode(y="y:Q")
st.altair_chart(qber + rules, use_container_width=True)

colors = alt.Scale(domain=["SECURE", "CAUTION", "LOCKDOWN"], range=["#2e9d5b", "#e0a800", "#c0392b"])
strip = alt.Chart(trace).mark_rect().encode(
    x=alt.X("round:O", title="round"), y=alt.Y("link:N", title="link"), color=alt.Color("state:N", scale=colors, title="state"),
    opacity=alt.condition(alt.datum.included, alt.value(1.0), alt.value(0.45)),
    tooltip=["round", "link", "state", "qber", "included", "discarded"])
st.altair_chart(strip, use_container_width=True)
st.caption("Faded cells: that hospital was not trained in that round (excluded, or the whole round was discarded).")
discarded = int(trace.groupby("round")["discarded"].first().sum())
st.write(f"In this run **{discarded} of {rounds} rounds were discarded** under `{shown_mode}`.")
with st.expander("Table of this run"):
    st.dataframe(trace, use_container_width=True)
