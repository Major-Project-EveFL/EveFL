"""The selection rule and the escalation check of scripts/hysteresis_sweep.py (pre-registered in docs/06)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "hysteresis_sweep.py"
    spec = importlib.util.spec_from_file_location("hysteresis_sweep_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cell(dwell, margin, flap, recovery, rounds):
    return {"dwell": dwell, "margin": margin, "A_changes_per_link_round": flap,
            "C_recovery_to_secure": recovery, "C_client_rounds_trained": rounds}


def test_the_rule_minimises_flapping_subject_to_recovery_and_client_rounds():
    module = _load()
    cells = [
        cell(1, 0.0, 0.031, 1, 140.0),     # baseline
        cell(3, 0.01, 0.012, 3, 138.0),    # candidate, fewer flaps
        cell(5, 0.02, 0.006, 5, 136.0),    # candidate, fewest flaps, still within 5 rounds and 95% (133)
        cell(8, 0.02, 0.002, 8, 133.0),    # excluded: recovery 8 > 5
        cell(5, 0.005, 0.001, 5, 130.0),   # excluded: client-rounds 130 < 0.95 * 140 = 133
    ]
    selection = module.select_recommended(cells)
    assert selection["recommended"] == {"dwell": 5, "margin": 0.02}
    assert sorted(selection["candidates"]) == [(1, 0.0), (3, 0.01), (5, 0.02)]
    assert selection["client_rounds_floor"] == pytest.approx(133.0) and not selection["recommended_is_no_hysteresis"]


def test_ties_go_to_the_smaller_dwell_then_the_smaller_margin():
    module = _load()
    cells = [cell(1, 0.0, 0.03, 1, 140.0), cell(3, 0.01, 0.01, 3, 139.0), cell(2, 0.02, 0.01, 2, 139.0),
             cell(2, 0.01, 0.01, 2, 139.0)]
    assert module.select_recommended(cells)["recommended"] == {"dwell": 2, "margin": 0.01}


def test_no_hysteresis_can_be_the_answer():
    module = _load()
    cells = [cell(1, 0.0, 0.0, 1, 140.0), cell(3, 0.01, 0.01, 3, 139.0)]
    selection = module.select_recommended(cells)
    assert selection["recommended"] == {"dwell": 1, "margin": 0.0} and selection["recommended_is_no_hysteresis"]


def test_no_candidate_gives_no_recommendation():
    module = _load()
    cells = [cell(1, 0.0, 0.03, 9, 140.0), cell(3, 0.01, 0.01, 9, 139.0)]  # even the baseline recovers too slowly
    assert module.select_recommended(cells)["recommended"] is None


def test_escalation_check_passes_for_the_real_policy_and_catches_a_late_lockdown():
    module = _load()
    from evefl.fl.control_sim import PolicyRun, simulate_policy
    from evefl.orchestration.policy import PolicyConfig
    from evefl.orchestration.state_machine import HysteresisConfig, SecurityState

    q = np.zeros((1, 6, 3))
    q[0, 2:4, 0] = 0.2                     # link 0 attacked in rounds 3 and 4
    runs = simulate_policy(q, module.LINKS, PolicyConfig(mode="per_client", hysteresis=HysteresisConfig(0.01, 3)))
    module.assert_escalation_never_delayed(q, runs)  # the real controller escalates immediately

    states = [{cid: SecurityState.SECURE for cid in module.LINKS} for _ in range(6)]
    late = PolicyRun([False] * 6, [[]] * 6, states, [{}] * 6)  # LOCKDOWN never entered despite q >= 0.11
    with pytest.raises(AssertionError, match="late"):
        module.assert_escalation_never_delayed(q, [late])
