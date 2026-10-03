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

# --------------------------------------------------------------------------
# Boundaries and the LOCKDOWN-threshold guard
# --------------------------------------------------------------------------

def test_boundaries_use_greater_or_equal(controller):
    assert controller.classify(0.0499999) == SecurityState.SECURE
    assert controller.classify(0.05) == SecurityState.CAUTION
    assert controller.classify(0.1099999) == SecurityState.CAUTION
    assert controller.classify(0.11) == SecurityState.LOCKDOWN


@pytest.mark.parametrize("below", [0.10, 0.05, 0.0999])
def test_lockdown_threshold_may_not_be_configured_below_0_11(below):
    with pytest.raises(ValueError, match="0.11"):
        StateThresholds(secure_max=0.02, caution_max=below)


def test_lockdown_threshold_may_be_raised_and_stays_ge():
    c = StateController(StateThresholds(secure_max=0.05, caution_max=0.15))
    assert c.classify(0.11) == SecurityState.CAUTION
    assert c.classify(0.15) == SecurityState.LOCKDOWN


@pytest.mark.parametrize("secure,caution", [(0.11, 0.11), (0.2, 0.15), (0.0, 0.11), (0.05, 1.5)])
def test_threshold_ordering_is_validated(secure, caution):
    with pytest.raises(ValueError):
        StateThresholds(secure_max=secure, caution_max=caution)


def test_controller_module_does_not_import_ml_or_quantum_stacks():
    import evefl.orchestration.state_machine as module

    source = open(module.__file__, encoding="utf-8").read()
    for forbidden in ("qiskit", "flwr", "torch"):
        assert f"import {forbidden}" not in source and f"from {forbidden}" not in source
