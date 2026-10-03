"""
Hysteresis in the QBER controller: escalation is immediate, only the way out is slowed.

Example tests pin the behaviour; hypothesis property tests check the invariants over arbitrary QBER sequences
(profile "ci" in conftest: derandomised, fixed number of examples).
"""

import copy

import pytest

from evefl.orchestration.state_machine import (
    HysteresisConfig,
    SecurityState,
    StateController,
    StateThresholds,
)

S, C, L = SecurityState.SECURE, SecurityState.CAUTION, SecurityState.LOCKDOWN
RANK = {S: 0, C: 1, L: 2}


def make(margin=0.0, dwell=1, thresholds=None):
    return StateController(thresholds, HysteresisConfig(margin=margin, dwell=dwell))


def run(controller, readings):
    return [controller.update(q).new_state for q in readings]


# --------------------------------------------------------------------------
# Config validation
# --------------------------------------------------------------------------

def test_default_hysteresis_is_off():
    assert HysteresisConfig() == HysteresisConfig(margin=0.0, dwell=1)


@pytest.mark.parametrize("kwargs", [{"margin": -0.001}, {"dwell": 0}, {"dwell": -3}])
def test_hysteresis_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        HysteresisConfig(**kwargs)


def test_margin_must_leave_room_to_return_to_secure():
    with pytest.raises(ValueError, match="margin"):
        make(margin=0.05)  # equals secure_max: SECURE could never be re-entered
    make(margin=0.049)  # just below is fine


def test_hysteresis_cannot_soften_the_lockdown_threshold():
    """The 0.11 guard still applies; hysteresis only acts on the way out."""
    with pytest.raises(ValueError, match="0.11"):
        StateController(StateThresholds(secure_max=0.05, caution_max=0.10), HysteresisConfig(0.01, 3))


# --------------------------------------------------------------------------
# Escalation is immediate
# --------------------------------------------------------------------------

@pytest.mark.parametrize("margin,dwell", [(0.0, 1), (0.01, 3), (0.02, 8)])
def test_crossing_005_enters_caution_and_011_enters_lockdown_at_once(margin, dwell):
    c = make(margin, dwell)
    assert run(c, [0.0, 0.049999]) == [S, S]
    assert c.update(0.05).new_state == C       # >= 0.05: immediately CAUTION
    assert c.update(0.1099999).new_state == C
    assert c.update(0.11).new_state == L       # >= 0.11: immediately LOCKDOWN, no delay


def test_lockdown_entry_is_not_delayed_even_in_the_middle_of_a_deescalation_window():
    c = make(margin=0.01, dwell=4)
    run(c, [0.2])                              # LOCKDOWN
    run(c, [0.01, 0.01])                       # two qualifying readings, still LOCKDOWN
    assert c.current_state == L
    assert c.update(0.2).new_state == L        # attack resumes: stays, window reset
    assert run(c, [0.01, 0.01, 0.01]) == [L, L, L]  # the earlier two readings no longer count
    assert c.update(0.01).new_state == S


# --------------------------------------------------------------------------
# De-escalation is slow
# --------------------------------------------------------------------------

def test_deescalation_takes_exactly_dwell_qualifying_readings():
    for dwell in (1, 2, 3, 5):
        c = make(margin=0.01, dwell=dwell)
        c.update(0.2)
        states = run(c, [0.0] * (dwell + 1))
        assert states[:dwell - 1] == [L] * (dwell - 1)
        assert states[dwell - 1] == S  # dropped on the dwell-th qualifying reading


def test_margin_is_required_below_the_boundary():
    c = make(margin=0.01, dwell=2)
    c.update(0.06)                              # CAUTION
    # 0.045 is below 0.05 but within the margin (0.045 + 0.01 >= 0.05): does not count
    assert run(c, [0.045, 0.045, 0.045]) == [C, C, C]
    # 0.039 + 0.01 < 0.05: counts
    assert run(c, [0.039, 0.039]) == [C, S]


def test_a_reading_that_is_not_clearly_below_breaks_the_streak():
    c = make(margin=0.01, dwell=3)
    c.update(0.06)
    assert run(c, [0.01, 0.01, 0.045, 0.01, 0.01]) == [C, C, C, C, C]  # 0.045 reset the count
    assert c.update(0.01).new_state == S


def test_deescalation_drops_to_the_highest_level_seen_in_the_window():
    c = make(margin=0.01, dwell=2)
    c.update(0.3)                               # LOCKDOWN
    # candidates: classify(0.06 + 0.01) = CAUTION, classify(0.0 + 0.01) = SECURE -> drop to CAUTION (the higher)
    assert run(c, [0.06, 0.0]) == [L, C]
    assert run(c, [0.0, 0.0]) == [C, S]


def test_transition_reports_raw_state_and_held():
    c = make(margin=0.01, dwell=3)
    c.update(0.2)
    t = c.update(0.01)
    assert t.raw_state == S and t.new_state == L and t.held is True and t.changed is False
    c.update(0.01)
    done = c.update(0.01)
    assert done.new_state == S and done.held is False and done.changed is True


def test_first_reading_has_no_previous_state():
    t = make(0.01, 3).update(0.2)
    assert t.previous_state is None and t.changed is True and t.new_state == L


def test_reset_forgets_history():
    c = make(0.01, 3)
    c.update(0.2)
    c.update(0.0)
    c.reset()
    assert c.current_state is None
    assert c.update(0.0).new_state == S  # no leftover LOCKDOWN or pending readings


def test_qber_outside_unit_interval_is_rejected():
    with pytest.raises(ValueError):
        make().update(1.5)


def test_margin_pushing_qber_over_one_is_clamped():
    c = make(margin=0.04, dwell=1)
    c.update(0.99)
    assert c.update(1.0).new_state == L  # q + margin > 1 must not raise


# --------------------------------------------------------------------------
# Hysteresis off == memoryless classifier
# --------------------------------------------------------------------------

def test_default_controller_equals_the_memoryless_classifier():
    c = StateController()
    for q in (0.0, 0.049, 0.05, 0.08, 0.109, 0.11, 0.3, 0.02, 0.0):
        assert c.update(q).new_state == c.classify(q)


# --------------------------------------------------------------------------
# Property tests (hypothesis)
# --------------------------------------------------------------------------

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

QBER = st.one_of(
    st.floats(min_value=0.0, max_value=0.4, allow_nan=False),
    st.sampled_from([0.0, 0.0499999, 0.05, 0.0500001, 0.0999, 0.1099999, 0.11, 0.1100001, 0.2, 1.0]),
)
SEQUENCES = st.lists(QBER, min_size=1, max_size=60)
MARGINS = st.sampled_from([0.0, 0.002, 0.005, 0.01, 0.02, 0.049])
DWELLS = st.integers(min_value=1, max_value=8)


@given(SEQUENCES, MARGINS, DWELLS)
def test_state_is_never_lower_than_the_memoryless_state(readings, margin, dwell):
    c = make(margin, dwell)
    for q in readings:
        assert RANK[c.update(q).new_state] >= RANK[c.classify(q)]


@given(SEQUENCES, MARGINS, DWELLS)
def test_lockdown_is_never_entered_late_and_caution_floor_holds(readings, margin, dwell):
    c = make(margin, dwell)
    for q in readings:
        state = c.update(q).new_state
        if q >= 0.11:
            assert state == L          # compared with >=, no delay, whatever the hysteresis
        if q >= 0.05:
            assert RANK[state] >= RANK[C]


@given(SEQUENCES, st.floats(min_value=0.0, max_value=0.4, allow_nan=False),
       st.floats(min_value=0.0, max_value=0.4, allow_nan=False), MARGINS, DWELLS)
def test_escalation_is_monotone_in_the_latest_reading(prefix, a, b, margin, dwell):
    base = make(margin, dwell)
    run(base, prefix)
    low, high = copy.deepcopy(base), copy.deepcopy(base)
    lo, hi = min(a, b), max(a, b)
    assert RANK[low.update(lo).new_state] <= RANK[high.update(hi).new_state]


@given(SEQUENCES, MARGINS, DWELLS)
def test_a_downward_move_needs_dwell_qualifying_readings_and_goes_to_the_highest_candidate(readings, margin, dwell):
    c = make(margin, dwell)
    history = []
    previous = None
    for q in readings:
        history.append(q)
        state = c.update(q).new_state
        if previous is not None and RANK[state] < RANK[previous]:
            assert len(history) >= dwell
            window = history[-dwell:]
            candidates = [c.classify(min(1.0, w + margin)) for w in window]
            assert all(RANK[cand] < RANK[previous] for cand in candidates)
            assert state == max(candidates, key=lambda st_: RANK[st_])
        previous = state


@given(SEQUENCES)
def test_without_hysteresis_the_controller_is_the_memoryless_classifier(readings):
    c = StateController()
    for q in readings:
        assert c.update(q).new_state == c.classify(q)


@given(SEQUENCES, MARGINS, DWELLS)
def test_the_controller_is_deterministic_and_reset_restores_a_fresh_one(readings, margin, dwell):
    a, b = make(margin, dwell), make(margin, dwell)
    assert run(a, readings) == run(b, readings)
    a.reset()
    assert run(a, readings) == run(make(margin, dwell), readings)


@given(st.floats(min_value=0.0, max_value=0.1099999, allow_nan=False), MARGINS, DWELLS)
def test_the_lockdown_threshold_cannot_be_configured_below_011(caution_max, margin, dwell):
    if not 0.0 < 0.001 < caution_max:
        return  # not a valid ordering anyway
    with pytest.raises(ValueError):
        StateController(StateThresholds(secure_max=0.001, caution_max=caution_max), HysteresisConfig(0.0, dwell))
