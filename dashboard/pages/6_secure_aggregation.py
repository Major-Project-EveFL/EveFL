"""
LIVE: Secure aggregation, a library demonstration.

Runs the step-2a primitives of ADR 0003 (fixed-point codec and mask generator) in one process on synthetic updates. It is NOT wired
into the federated-learning path; the pairwise seeds are fresh random bytes standing in for the step-2b key agreement.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from evefl import secagg_demo as demo  # noqa: E402
from evefl.crypto.secagg import fixedpoint as fp  # noqa: E402

st.set_page_config(page_title="Secure aggregation", page_icon="🔐", layout="wide")
st.title("🔐 Secure aggregation: a library demonstration")
st.warning(demo.LABEL)
st.markdown(
    "**The idea.** Each hospital adds one random-looking mask for every other hospital to its (weighted) update: it **adds** the mask "
    "shared with a hospital whose number is higher and **subtracts** the one shared with a lower number. Every mask appears once "
    "with `+` and once with `-`, so the masks cancel in the sum. The server learns the sum and, from one masked update, nothing about "
    "that hospital's own numbers. Arithmetic is on 64-bit integers with 32 fractional bits, so cancellation is exact.")

# ---------------------------------------------------------------- controls
st.sidebar.header("Demo")
n_clients = st.sidebar.slider("Number of hospitals", 3, 5, 3)
n_params = st.sidebar.slider("Parameters per update", 8, 64, 16,
                             help="A tiny stand-in: ResNet-18 has 11,193,294 floating-point values per hospital per round.")
scale = st.sidebar.select_slider("Size of the synthetic updates", options=[0.001, 0.01, 0.1, 1.0], value=0.01)
st.sidebar.caption("Weights are registry weights: the audited hospital sizes 46,975 / 18,018 / 25,136 images, then two illustrative sizes.")

if "mask_round" not in st.session_state:
    st.session_state.mask_round = 0
if st.sidebar.button("Draw new random masks"):
    st.session_state.mask_round += 1


@st.cache_data
def seeds_for(n: int, token: int):
    """Fresh seeds for this number of hospitals; the token makes a button press draw new ones."""
    return demo.new_pair_seeds(n)


ids = [str(i) for i in range(n_clients)]
dropped = st.sidebar.selectbox("A hospital's masked update never arrives", ["none", *ids])
colluding = st.sidebar.multiselect("Hospitals that give their pairwise seeds to the server", ids, max_selections=n_clients - 1)

try:
    result = demo.run_demo(demo.DemoSettings(
        n_clients=n_clients, n_params=n_params, update_scale=scale, seed=0, round_id=1,
        drop_client=None if dropped == "none" else dropped, colluding=tuple(colluding),
        pair_seeds=seeds_for(n_clients, st.session_state.mask_round)))
except ValueError as exc:
    st.error(f"This setting is refused before any masking: {exc}")
    st.stop()

preview = min(6, n_params)

# ---------------------------------------------------------------- step 1
st.header("1. What each hospital trained")
st.caption("Synthetic numbers standing in for the change in the model after local training. Never sent in this form.")
table = pd.DataFrame(result.updates[:, :preview], index=[f"hospital {i}" for i in result.client_ids],
                     columns=[f"param {k}" for k in range(preview)])
table.insert(0, "weight share", result.weights_normalised.round(4))
st.dataframe(table.round(5), use_container_width=True)

# ---------------------------------------------------------------- step 2
st.header("2. What the server receives")
st.caption("Each hospital's weighted update plus its masks, as raw 64-bit words (hexadecimal). Without the seeds these are "
           "indistinguishable from random numbers.")
hexed = pd.DataFrame([[f"{int(v):016x}" for v in row[:preview]] for row in result.masked],
                     index=[f"hospital {i}" for i in result.client_ids], columns=[f"param {k}" for k in range(preview)])
st.dataframe(hexed, use_container_width=True)
decoded = np.abs(result.masked.view(np.int64).astype(float) / fp.SCALE)
st.caption(f"Read as numbers, these masked words are of the order of {np.median(decoded):.2e}, while the true weighted updates "
           f"are of the order of {np.median(np.abs(result.weighted)):.2e}.")

# ---------------------------------------------------------------- step 3
st.header("3. The sum")
left, middle, right = st.columns(3)
left.metric("Largest difference from the plain weighted average", f"{result.max_abs_error:.1e}")
middle.metric("Quantisation tolerance (hospitals x 2^-33)", f"{result.tolerance:.1e}")
right.metric("Within tolerance", "yes" if result.max_abs_error <= result.tolerance else "NO")
compare = pd.DataFrame({"plain weighted average": result.aggregate_plain[:preview],
                        "sum of the masked updates": result.aggregate_secure[:preview]},
                       index=[f"param {k}" for k in range(preview)])
st.dataframe(compare.round(8), use_container_width=True)
st.caption("The masks cancelled exactly; the only difference is rounding each update to the 2^-32 grid.")

# ---------------------------------------------------------------- what-ifs
st.header("4. If a hospital drops out")
if result.dropout is None:
    st.info("Choose a hospital in the sidebar whose masked update never arrives.")
else:
    d = result.dropout
    st.error(f"Hospital {d['dropped']} is missing. Adding what arrived gives numbers of the order of "
             f"{d['garbage_mean_abs']:.2e}, not {np.mean(np.abs(result.aggregate_plain)):.1e}. Round aborted: `{d['reason']}`.")
    st.caption(d["note"])

st.header("5. If the server colludes with hospitals")
if result.collusion is None:
    st.info("Alone, the server learns only the total. Choose hospitals in the sidebar that give their pairwise seeds to the server.")
else:
    c = result.collusion
    if c["individual_revealed"] is not None:
        st.error(f"With hospitals {', '.join(c['colluding'])} colluding, hospital {c['individual_revealed']}'s update is revealed "
                 f"exactly (error {c['sum_error']:.1e}).")
    else:
        first = c["learns_sum_of"][0]
        guess_error = c[f"single_guess_error_hospital_{first}"]
        st.warning(f"With hospital(s) {', '.join(c['colluding'])} colluding, the server learns the exact **sum** of hospitals "
                   f"{', '.join(c['learns_sum_of'])} (error {c['sum_error']:.1e}) but not any one of them: a guess for "
                   f"hospital {first} alone is off by {guess_error:.2e}.")
    st.caption("This is the privacy floor stated in ADR 0003: with 3 hospitals, one colluding hospital leaves an anonymity set of 2, "
               "and two colluding hospitals reveal the third. Masking protects against the server alone; every hospital can already "
               "compute the sum of the others from the new global model.")

st.divider()
st.caption("Not shown, because not built yet: the key agreement and signed advertisements (step 2b), the aggregator class with "
           "its set checks (step 2c), the norm check, and wiring into the training loop (step 3). The masking is classical (X25519 in "
           "the planned design), so it carries no post-quantum guarantee.")
