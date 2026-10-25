"""
Mission Control: the home page. What EveFL is in one screen, the measured numbers (read from the recorded files), and a live system
check that runs the real library code and shows each result as it finishes.
"""

import sys
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from dashboard import theme  # noqa: E402
from evefl import selfcheck, showcase  # noqa: E402

h = showcase.headline_numbers()
seed0, seed1 = h["runs"]

theme.hero("EveFL",
           "Hospitals train a chest X-ray AI together without sharing a single image. A simulated quantum link watches each "
           "hospital's connection, and a controller decides, round by round, whose update can be trusted.",
           ["prototype", "simulated", "measured"])

theme.kpi_row([
    {"label": "Served model, test AUC", "value": f"{h['served_model_test_auc']:.3f}", "kind": "measured",
     "note": f"Round {h['served_model_round']} checkpoint, chosen on validation, one test evaluation, 14 chest findings"},
    {"label": "Two training runs", "value": f"{seed0['test_auc_round_50']:.3f} / {seed1['test_auc_round_50']:.3f}", "kind": "measured",
     "note": "Test AUC at round 50 for seeds 0 and 1; two runs, no confidence interval"},
    {"label": "Hospitals", "value": str(h["n_hospitals"]), "kind": "measured",
     "note": "Patient-level split, non-IID Dirichlet(0.5); the audit finds no shared patient or image"},
    {"label": "LOCKDOWN bound", "value": f"{h['lockdown_bound'] * 100:.0f}%", "kind": "real code",
     "note": "The BB84 asymptotic limit; the code refuses any lower setting"},
])

theme.section("How it works", "One federated round, in four steps")
theme.step_cards([
    {"title": "Hospitals train locally", "text": "Each hospital trains a ResNet-18 on its own X-rays. Images never leave the hospital."},
    {"title": "A quantum link is checked", "text": "A simulated BB84 exchange on each connection reads the error rate (QBER). An eavesdropper raises it."},
    {"title": "A controller decides", "text": "Below 5% the link is SECURE, up to 11% CAUTION, and from 11% LOCKDOWN. The response is graduated, not a blunt pause."},
    {"title": "The server aggregates", "text": "Only trusted updates are averaged into the shared model. A suspicious link is set aside, the others keep training."},
])

theme.section("Is it working right now?", "These checks run the real code on this machine, in front of you")
st.caption("Nothing here is a recording: each line is produced by calling the library when you press the button. "
           "Keys used in the checks are random test bytes; the simulator's own keys are not secret.")

if "system_check" not in st.session_state:
    st.session_state["system_check"] = None

ran_now = False
if st.button("▶  Run live system check", type="primary"):
    ran_now = True
    rows, collected = st.container(), []
    progress = st.progress(0.0, text="Starting...")
    checks = selfcheck.selected_checks()
    with rows:
        slot = st.empty()
        for i, check in enumerate(checks):
            progress.progress(i / len(checks), text=f"Running: {check.title}...")
            result = selfcheck.run_check(check)
            collected.append(result)
            slot.markdown("".join(theme.check_row(r.title, r.detail, r.status, r.seconds, r.kind, r.what) for r in collected),
                          unsafe_allow_html=True)
    progress.progress(1.0, text="Done")
    st.session_state["system_check"] = collected

results = st.session_state["system_check"]
if results is None:
    theme.banner("Not run yet. Press the button; it takes a few seconds.", "warn")
else:
    summary = selfcheck.summarise(results)
    if summary["healthy"]:
        extra = f", {summary['skipped']} skipped" if summary["skipped"] else ""
        theme.banner(f"{summary['passed']} of {summary['total']} checks passed{extra} in {summary['seconds']:.1f} s", "ok" if not summary["skipped"] else "warn")
    else:
        theme.banner(f"{summary['failed']} of {summary['total']} checks FAILED: see the red lines below", "bad")
    if not ran_now:
        st.markdown("".join(theme.check_row(r.title, r.detail, r.status, r.seconds, r.kind, r.what) for r in results), unsafe_allow_html=True)

theme.section("Where to go next", "The demo path, in order")
for target, label, icon in [("pages/1_qber_explorer.py", "BB84 / QBER Explorer: put Eve on the line", "🔐"),
                            ("pages/2_security_controller.py", "Security Controller: SECURE, CAUTION, LOCKDOWN", "🛡️"),
                            ("pages/5_attack_lab.py", "Attack lab: three policies against one eavesdropper", "⚔️"),
                            ("pages/6_secure_aggregation.py", "Secure aggregation: the server sees only masked updates", "🧩"),
                            ("pages/4_fl_results.py", "Results explorer: the measured training runs", "📊"),
                            ("pages/7_upload_xray.py", "Try the model: score a chest X-ray", "🩻")]:
    theme.page_link(target, label, icon)

theme.honest_footer()
