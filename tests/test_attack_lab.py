"""
The attack lab (`evefl/attack_lab.py`, `dashboard/pages/5_attack_lab.py`): Eve on a chosen link, the three policy modes side by side.

It must run the real policy classes on the exact count-level QBER model (`evefl.fl.control_sim`), report only control-plane
counts labelled simulated, and reproduce the numbers already recorded in ADR 0002 (docs/validation/hysteresis_sweep.json).
"""

from pathlib import Path

import numpy as np
import pytest

from evefl import attack_lab as lab
from evefl.orchestration.state_machine import SecurityState

REPO = Path(__file__).resolve().parents[1]


def _run(**kwargs):
    base = dict(reps=200, rounds=50, seed=0)
    base.update(kwargs)
    return lab.run_lab(lab.LabSettings(**base))


# --------------------------------------------------------------------------
# agrees with the recorded Phase 2 results (ADR 0002, simulated)
# --------------------------------------------------------------------------

def test_static_eve_on_one_link_reproduces_the_adr_0002_table_for_alpha_0_6():
    summary = _run(attacked_link="0", alpha=0.6).summary
    # ADR 0002 (1000 reps): global modes discard 93.1% of rounds and train 6.9% of client-rounds; per_client discards 0% and trains 69.0%
    for mode in ("global_binary", "global"):
        assert abs(summary.loc[mode, "rounds_discarded_pct"] - 93.1) < 4.0
        assert abs(summary.loc[mode, "client_rounds_trained_pct"] - 6.9) < 4.0
    assert summary.loc["per_client", "rounds_discarded_pct"] < 1.0
    assert abs(summary.loc["per_client", "client_rounds_trained_pct"] - 69.0) < 4.0


def test_global_binary_and_global_discard_exactly_the_same_rounds():
    summary = _run(attacked_link="0", alpha=0.44).summary
    assert summary.loc["global_binary", "rounds_discarded_pct"] == summary.loc["global", "rounds_discarded_pct"]


def test_with_no_attack_nothing_is_discarded_and_every_client_trains():
    for settings in ({"attacked_link": None}, {"attacked_link": "0", "alpha": 0.0}):
        summary = _run(**settings).summary
        assert (summary["rounds_discarded_pct"] < 1.0).all() and (summary["client_rounds_trained_pct"] > 99.0).all()


def test_the_attacked_clients_own_rounds_are_reported_and_drop_under_per_client_exclusion():
    summary = _run(attacked_link="1", alpha=0.6).summary
    assert summary.loc["per_client", "attacked_client_rounds_trained"] < 10.0
    quiet = _run(attacked_link=None).summary
    assert np.isnan(quiet.loc["per_client", "attacked_client_rounds_trained"])


# --------------------------------------------------------------------------
# the controller rules are visible in the trace
# --------------------------------------------------------------------------

@pytest.mark.parametrize("margin,dwell", [(0.0, 1), (0.01, 3)])
def test_lockdown_at_011_and_escalation_is_immediate_even_with_hysteresis(margin, dwell):
    result = _run(attacked_link="0", alpha=0.7, hysteresis_margin=margin, hysteresis_dwell=dwell, reps=20)
    for trace in result.trace.values():
        high = trace[trace["qber"] >= 0.11]
        mid = trace[(trace["qber"] >= 0.05) & (trace["qber"] < 0.11)]
        assert len(high) > 0 and (high["state"] == SecurityState.LOCKDOWN.value).all()
        assert (mid["state"] != SecurityState.SECURE.value).all()


def test_the_lockdown_threshold_cannot_be_lowered_through_the_lab():
    with pytest.raises(TypeError):
        lab.LabSettings(caution_max=0.05)   # not a setting: the 0.11 boundary is not exposed


def test_trace_has_one_row_per_round_and_link_and_the_same_qber_for_every_mode():
    result = _run(reps=10, rounds=12)
    assert set(result.trace) == {"global_binary", "global", "per_client"}
    for trace in result.trace.values():
        assert len(trace) == 12 * 3 and set(trace["link"]) == {"0", "1", "2"}
        assert {"round", "link", "qber", "state", "included", "discarded"} <= set(trace.columns)
    qbers = [t.sort_values(["round", "link"])["qber"].tolist() for t in result.trace.values()]
    assert qbers[0] == qbers[1] == qbers[2]   # paired comparison: one trajectory, three policies


# --------------------------------------------------------------------------
# determinism, validation, honesty label
# --------------------------------------------------------------------------

def test_same_settings_give_the_same_result_and_a_different_seed_a_different_one():
    a, b = _run(reps=50, seed=3), _run(reps=50, seed=3)
    assert a.summary.equals(b.summary) and a.trace["per_client"].equals(b.trace["per_client"])
    c = _run(reps=50, seed=4)
    assert not a.trace["per_client"]["qber"].equals(c.trace["per_client"]["qber"])


@pytest.mark.parametrize("bad", [{"alpha": 1.5}, {"alpha": -0.1}, {"noise": 0.6}, {"rounds": 0}, {"reps": 0},
                                 {"attacked_link": "9"}, {"modes": ("nope",)}, {"attack_kind": "teleport"}, {"min_clients": 0}])
def test_invalid_settings_are_refused(bad):
    with pytest.raises(ValueError):
        lab.run_lab(lab.LabSettings(**bad))


def test_window_step_and_intermittent_attacks_are_supported():
    window = _run(attacked_link="0", alpha=0.7, attack_kind="window", attack_start=21, attack_end=30, reps=50)
    quiet = window.trace["global"]
    before = quiet[(quiet["round"] <= 20)]["discarded"].mean()
    during = quiet[(quiet["round"] >= 21) & (quiet["round"] <= 30)]["discarded"].mean()
    assert during > 0.8 and before < 0.1
    for kind in ("step", "intermittent"):
        assert _run(attacked_link="0", alpha=0.7, attack_kind=kind, attack_start=10, attack_probability=0.5, reps=30).summary is not None


def test_the_result_is_labelled_simulated_control_plane_counts_not_accuracy():
    result = _run(reps=10)
    assert "SIMULATED" in result.label and "control-plane" in result.label and "not accuracy" in result.label
    assert "client-rounds" in result.label


# --------------------------------------------------------------------------
# the page
# --------------------------------------------------------------------------

def test_the_page_renders_with_its_label_and_survives_an_alpha_change():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(REPO / "dashboard" / "pages" / "5_attack_lab.py"), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    shown = " ".join(str(x.value) for x in list(at.warning) + list(at.info) + list(at.caption) + list(at.markdown))
    assert "SIMULATED" in shown and "not accuracy" in shown
    alpha = next(s for s in at.slider if "alpha" in s.label.lower())
    alpha.set_value(0.0).run()
    assert not at.exception, [e.value for e in at.exception]
