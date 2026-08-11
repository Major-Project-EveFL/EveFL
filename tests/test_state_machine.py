import pytest

from evefl.orchestration.state_machine import (
    SecurityState,
    StateController,
    StateThresholds,
)


@pytest.fixture
def controller():
    return StateController(StateThresholds(secure_max=0.05, caution_max=0.11))


def test_secure_below_threshold(controller):
    assert controller.classify(0.0) == SecurityState.SECURE
    assert controller.classify(0.049) == SecurityState.SECURE


def test_caution_boundary(controller):
    assert controller.classify(0.05) == SecurityState.CAUTION
    assert controller.classify(0.10) == SecurityState.CAUTION


def test_lockdown_boundary(controller):
    assert controller.classify(0.11) == SecurityState.LOCKDOWN
    assert controller.classify(0.5) == SecurityState.LOCKDOWN


def test_invalid_qber_raises(controller):
    with pytest.raises(ValueError):
        controller.classify(-0.1)
    with pytest.raises(ValueError):
        controller.classify(1.1)


def test_update_tracks_transitions(controller):
    t1 = controller.update(0.01)  # SECURE
    assert t1.changed is True
    assert t1.new_state == SecurityState.SECURE

    t2 = controller.update(0.02)  # still SECURE
    assert t2.changed is False

    t3 = controller.update(0.15)  # LOCKDOWN
    assert t3.changed is True
    assert t3.previous_state == SecurityState.SECURE
    assert t3.new_state == SecurityState.LOCKDOWN

    assert controller.current_state == SecurityState.LOCKDOWN
