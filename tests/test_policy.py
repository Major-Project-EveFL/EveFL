from evefl.orchestration.policy import (
    REASON_KEY_ABORT,
    REASON_NO_KEY,
    REASON_NONE,
    REASON_QBER,
    apply_key_policy,
)
from evefl.orchestration.state_machine import SecurityState

S, C, L = SecurityState.SECURE, SecurityState.CAUTION, SecurityState.LOCKDOWN


def test_all_keys_ok_leaves_the_qber_state_unchanged():
    for state in (S, C):
        d = apply_key_policy(state, {"0": "ok", "1": "ok", "2": "ok"})
        assert d.state == state and d.reason == REASON_NONE and not d.discard_round and d.failed_clients == []


def test_no_keys_required_means_no_change():
    assert apply_key_policy(C, {}).state == C


def test_a_client_without_a_key_turns_the_round_into_lockdown_with_its_own_reason():
    d = apply_key_policy(S, {"0": "ok", "1": "no_key", "2": "ok"})
    assert d.state == L and d.discard_round and d.reason == REASON_NO_KEY and d.failed_clients == ["1"]


def test_aborts_are_logged_separately_from_no_key():
    d = apply_key_policy(C, {"0": "abort_ec_verification", "1": "no_key"})
    assert d.state == L and d.reason == REASON_KEY_ABORT and d.failed_clients == ["0", "1"]
    assert apply_key_policy(S, {"0": "abort_auth"}).reason == REASON_KEY_ABORT


def test_qber_lockdown_keeps_its_own_reason_even_if_keys_also_failed():
    d = apply_key_policy(L, {"0": "no_key", "1": "abort_qber_high"})
    assert d.state == L and d.reason == REASON_QBER and d.failed_clients == ["0", "1"]
    assert apply_key_policy(L, {"0": "ok"}).reason == REASON_QBER


def test_policy_module_is_pure_logic():
    import evefl.orchestration.policy as module

    source = open(module.__file__, encoding="utf-8").read()
    for forbidden in ("qiskit", "flwr", "torch", "evefl.quantum", "evefl.fl"):
        assert f"import {forbidden}" not in source and f"from {forbidden}" not in source
