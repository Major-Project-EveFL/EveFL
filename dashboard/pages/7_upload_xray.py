"""
LIVE: Try the model on a chest X-ray.

A research prototype trained on a public dataset in a simulation of three hospitals. It is NOT diagnostic. The model is read from a
local `checkpoints/` folder (written by `--save-model`); samples come from a local `demo_samples/` folder (public test-split images
exported by `scripts/export_demo_samples.py`); neither is in git. Uploaded images are scored in memory and not stored.
"""

import json
import os
import sys
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from PIL import Image  # noqa: E402

from evefl import inference as inf  # noqa: E402

CHECKPOINT_DIR = Path(os.environ.get("EVEFL_CHECKPOINT_DIR", REPO_ROOT / "checkpoints"))
SAMPLES_DIR = Path(os.environ.get("EVEFL_DEMO_SAMPLES", REPO_ROOT / "demo_samples"))

try:
    st.set_page_config(page_title="Try the model", page_icon="🩻", layout="wide")
except st.errors.StreamlitAPIException:
    pass  # dashboard/app.py has already set it for the whole app (set_page_config is allowed once per run)
st.title("🩻 Try the model on a chest X-ray")
st.warning("**This is a research prototype and is not diagnostic.** It scores a picture with a model trained on a public dataset "
           "(NIH ChestX-ray14) in a simulation of three hospitals. The numbers are model scores, not diagnoses, are not calibrated "
           "probabilities of disease, and must not be used for any clinical decision. Uploaded images are scored in memory and not stored.")

# ---------------------------------------------------------------- the model
checkpoints = inf.find_checkpoints(CHECKPOINT_DIR)
if not checkpoints:
    st.info(f"No checkpoint found in `{CHECKPOINT_DIR}`. Train with `--save-model` (see the runbook) and copy the `.pt` files "
            "there, or set EVEFL_CHECKPOINT_DIR. The page needs a model to score a picture.")
    st.stop()

choice = st.sidebar.selectbox("Model", [p.name for p in checkpoints],
                              help="The best-validation checkpoint is the one chosen before any test result was seen.")
path = next(p for p in checkpoints if p.name == choice)


@st.cache_resource(show_spinner="Loading the model...")
def load(file: str, modified: float) -> inf.LoadedModel:
    return inf.load_checkpoint(file)


try:
    loaded = load(str(path), path.stat().st_mtime)
except ValueError as exc:
    st.error(f"Cannot use this checkpoint: {exc}")
    st.stop()

auc = loaded.val_auc_excl_thin
st.caption(f"Model: `{loaded.file}` ({loaded.kind.replace('_', ' ')}), round {loaded.round}"
           + (f", validation AUC {auc:.4f}" if isinstance(auc, float) else "")
           + f", run config hash {loaded.metadata.get('config_hash', 'unknown')}, SHA-256 {loaded.sha256[:16]}.")


def show(image: Image.Image, caption: str, recorded=None) -> None:
    left, right = st.columns([1, 1])
    left.image(image, caption=caption, use_column_width=True)
    table = inf.predict(loaded, image)
    with right:
        if recorded is not None:
            st.write("Labels recorded in the dataset for this picture (text-mined, not a diagnosis):", ", ".join(recorded))
        st.dataframe(table.assign(probability=table["probability"].round(4)), use_container_width=True, hide_index=True)
        st.bar_chart(table.set_index("finding")["probability"])
    st.caption("Fourteen independent scores, one per finding; they do not sum to 1. Scores of an untested single run on one picture "
               "say nothing about how the model performs overall: see the Results explorer for the measured evaluation.")


samples = []
if (SAMPLES_DIR / "samples.json").is_file():
    samples = json.loads((SAMPLES_DIR / "samples.json").read_text(encoding="utf-8")).get("samples", [])

tab_samples, tab_upload = st.tabs(["Sample images", "Upload your own"])
with tab_samples:
    if not samples:
        st.info(f"No sample images found in `{SAMPLES_DIR}`. Create them with `scripts/export_demo_samples.py` (public test-split "
                "images, kept out of git).")
    else:
        names = [s["file"] for s in samples]
        pick = st.selectbox("Sample image (public test split)", names)
        entry = next(s for s in samples if s["file"] == pick)
        show(Image.open(SAMPLES_DIR / pick), pick, recorded=entry["labels"])

with tab_upload:
    uploaded = st.file_uploader("A chest X-ray (PNG or JPEG)", type=["png", "jpg", "jpeg"])
    if uploaded is not None:
        try:
            show(inf.open_upload(uploaded.getvalue()), uploaded.name)
        except ValueError as exc:
            st.error(str(exc))
