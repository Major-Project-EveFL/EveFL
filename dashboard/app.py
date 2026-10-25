"""
EveFL dashboard: entry point and navigation.

    streamlit run dashboard/app.py

The pages are grouped in the order of the demo run (docs/DEMO.md). Every page is labelled with what kind of evidence it shows:
real code, simulated, recorded (measured) or a research prototype. Nothing here trains a model; the Results explorer only reads
recorded results and "Try the model" needs a saved checkpoint.
"""

import sys
import threading
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dashboard import theme  # noqa: E402
from evefl import selfcheck  # noqa: E402

st.set_page_config(page_title="EveFL", page_icon="🛰️", layout="wide", initial_sidebar_state="expanded")
theme.inject_css()


@st.cache_resource
def _warm_up_once() -> bool:
    """Import PyTorch and friends in the background once per server process, so the first click on a page is quick."""
    threading.Thread(target=selfcheck.warm_up, name="evefl-warm-up", daemon=True).start()
    return True


_warm_up_once()

navigation = st.navigation({
    "Start here": [
        st.Page("pages/0_mission_control.py", title="Mission Control", icon="🛰️", default=True),
    ],
    "Live demos": [
        st.Page("pages/1_qber_explorer.py", title="QBER Explorer", icon="🔐"),
        st.Page("pages/2_security_controller.py", title="Security Controller", icon="🛡️"),
        st.Page("pages/5_attack_lab.py", title="Attack Lab", icon="⚔️"),
        st.Page("pages/6_secure_aggregation.py", title="Secure Aggregation", icon="🧩"),
    ],
    "Evidence": [
        st.Page("pages/4_fl_results.py", title="Results Explorer", icon="📊"),
        st.Page("pages/3_crypto_demo.py", title="Crypto Pipeline", icon="🔒"),
    ],
    "Try it": [
        st.Page("pages/7_upload_xray.py", title="Try the Model", icon="🩻"),
    ],
})
theme.sidebar_brand()
navigation.run()
