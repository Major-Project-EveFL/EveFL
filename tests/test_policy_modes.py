"""
Round policy modes: global_binary (QKDFL-style baseline), global (EveFL three-state on the worst link) and
per_client (one controller per link, exclusion of LOCKDOWN links). Pure logic.
"""

import pytest

from evefl.fl.screening import MIN_CLIENTS_FOR_SCREENING as FL_MIN_FOR_SCREENING
from evefl.orchestration.policy import (
    MIN_CLIENTS_FOR_SCREENING,
    REASON_KEY_ABORT,
    REASON_NO_KEY,
    REASON_NONE,
    REASON_QBER,
    REASON_QBER_EXCLUSION,
    REASON_TOO_FEW,
    ClientAction,
    GlobalBinaryPolicy,
    GlobalPolicy,
    PerClientPolicy,
    PolicyConfig,
    create_policy,
    policy_registry,
)
from evefl.orchestration.state_machine import HysteresisConfig, SecurityState, StateThresholds

S, C, L = SecurityState.SECURE, SecurityState.CAUTION, SecurityState.LOCKDOWN
T, TP, X = ClientAction.TRAIN, ClientAction.TRAIN_PROX, ClientAction.EXCLUDE

CLEAN = {"0": 0.01, "1": 0.01, "2": 0.01}


def q(a=0.01, b=0.01, c=0.01):
    return {"0": a, "1": b, "2": c}


def per_client(**kwargs):
    return PerClientPolicy(PolicyConfig(mode="per_client", **kwargs))


# --------------------------------------------------------------------------
# Config and registry
# --------------------------------------------------------------------------

def test_the_three_modes_are_registered_and_created_by_name():
    assert set(policy_registry.list_keys()) >= {"global_binary", "global", "per_client"}
    assert isinstance(create_policy(PolicyConfig(mode="global_binary")), GlobalBinaryPolicy)
    assert isinstance(create_policy(PolicyConfig(mode="global")), GlobalPolicy)
    assert isinstance(create_policy(PolicyConfig(mode="per_client")), PerClientPolicy)
    with pytest.raises(KeyError):
        create_policy(PolicyConfig(mode="nope"))


def test_policy_config_validation_and_yaml_mapping():
    with pytest.raises(ValueError):
        PolicyConfig(min_clients=0)
    cfg = PolicyConfig.from_mapping({
        "mode": "per_client", "min_clients": 3,
        "thresholds": {"secure_max": 0.04, "caution_max": 0.12},
        "hysteresis": {"margin": 0.01, "dwell": 3},
    })
    assert cfg.mode == "per_client" and cfg.min_clients == 3
    assert cfg.thresholds == StateThresholds(0.04, 0.12) and cfg.hysteresis == HysteresisConfig(0.01, 3)
    default = PolicyConfig.from_mapping(None)
    assert default.mode == "global" and default.min_clients == 2 and default.hysteresis == HysteresisConfig()


def test_empty_readings_are_rejected():
    for policy in (GlobalBinaryPolicy, GlobalPolicy, PerClientPolicy):
        with pytest.raises(ValueError):
            policy(PolicyConfig()).decide({})


def test_screening_constant_matches_the_fl_module():
    assert MIN_CLIENTS_FOR_SCREENING == FL_MIN_FOR_SCREENING == 3


# --------------------------------------------------------------------------
# global_binary: QKDFL-style baseline
# --------------------------------------------------------------------------

def test_binary_pauses_at_the_lockdown_threshold_with_ge_and_has_no_caution_response():
    p = GlobalBinaryPolicy(PolicyConfig(mode="global_binary"))
    d = p.decide(q(0.06, 0.09, 0.1099))              # CAUTION range for EveFL: binary just trains everyone
    assert not d.discard_round and d.global_state == S
    assert all(c.action == T for c in d.clients.values())
    d = p.decide(q(0.0, 0.0, 0.11))                  # worst link at exactly 0.11: pause
    assert d.discard_round and d.reason == REASON_QBER and d.included == []
    assert all(c.action == X for c in d.clients.values())


def test_binary_discards_on_any_key_failure_with_its_own_reason():
    p = GlobalBinaryPolicy(PolicyConfig(mode="global_binary"))
    d = p.decide(CLEAN, {"0": "ok", "1": "no_key", "2": "ok"})
    assert d.discard_round and d.reason == REASON_NO_KEY and d.global_state == L
    d = p.decide(CLEAN, {"0": "abort_auth"})
    assert d.reason == REASON_KEY_ABORT
    assert p.decide(CLEAN, {"0": "ok"}).reason == REASON_NONE


def test_binary_qber_pause_keeps_the_qber_reason_even_if_keys_failed():
    d = GlobalBinaryPolicy(PolicyConfig(mode="global_binary")).decide(q(0.2), {"1": "no_key"})
    assert d.reason == REASON_QBER


# --------------------------------------------------------------------------
# global: EveFL on the worst link
# --------------------------------------------------------------------------

def test_global_uses_the_worst_link_for_everyone():
    p = GlobalPolicy(PolicyConfig(mode="global"))
    assert p.decide(CLEAN).global_state == S
    d = p.decide(q(0.01, 0.06, 0.01))                # one link in CAUTION puts EVERY client on FedProx
    assert d.global_state == C and not d.discard_round
    assert all(c.action == TP for c in d.clients.values())
    d = p.decide(q(0.01, 0.01, 0.2))                 # one link in LOCKDOWN stops the whole round
    assert d.discard_round and d.reason == REASON_QBER and d.included == []


def test_global_key_failure_discards_the_round_with_a_separate_reason():
    p = GlobalPolicy(PolicyConfig(mode="global"))
    d = p.decide(CLEAN, {"0": "ok", "1": "no_key", "2": "ok"})
    assert d.discard_round and d.reason == REASON_NO_KEY
    assert d.exclusion_reasons == {REASON_NO_KEY: ["1"]}
    d = p.decide(q(0.06, 0.01, 0.01), {"2": "abort_ec_verification"})
    assert d.reason == REASON_KEY_ABORT and d.exclusion_reasons == {REASON_KEY_ABORT: ["2"]}
    assert not p.decide(CLEAN, {"0": "ok", "1": "ok", "2": "ok"}).discard_round


def test_global_qber_lockdown_beats_key_failure_for_the_reason():
    assert GlobalPolicy(PolicyConfig(mode="global")).decide(q(0.3), {"0": "no_key"}).reason == REASON_QBER


def test_global_applies_hysteresis_to_the_worst_link():
    p = GlobalPolicy(PolicyConfig(mode="global", hysteresis=HysteresisConfig(0.01, 3)))
    assert p.decide(q(0.3)).discard_round
    assert [p.decide(CLEAN).discard_round for _ in range(3)] == [True, True, False]


# --------------------------------------------------------------------------
# per_client
# --------------------------------------------------------------------------

def test_eve_on_one_link_excludes_only_that_client_and_the_rest_continue():
    d = per_client().decide(q(0.3, 0.01, 0.01))
    assert not d.discard_round and d.included == ["1", "2"] and d.excluded == ["0"]
    assert d.clients["0"] == d.clients["0"].__class__(L, X, REASON_QBER_EXCLUSION)
    assert d.clients["1"].action == T and d.clients["2"].action == T
    assert d.exclusion_reasons == {REASON_QBER_EXCLUSION: ["0"]} and d.global_state == L


def test_per_client_caution_applies_only_to_that_client():
    d = per_client().decide(q(0.06, 0.01, 0.01))
    assert d.clients["0"].action == TP and d.clients["1"].action == T and d.clients["2"].action == T
    assert d.global_state == C and not d.discard_round and d.excluded == []


def test_below_min_clients_the_round_is_discarded():
    p = per_client(min_clients=2)
    d = p.decide(q(0.3, 0.3, 0.01))
    assert d.discard_round and d.reason == REASON_TOO_FEW and d.included == []
    assert "1 client(s) left" in d.detail
    assert not p.decide(q(0.3, 0.01, 0.01)).discard_round
    assert per_client(min_clients=3).decide(q(0.3, 0.01, 0.01)).discard_round  # needs all three
    assert not per_client(min_clients=1).decide(q(0.3, 0.3, 0.01)).discard_round
    assert per_client(min_clients=1).decide(q(0.3, 0.3, 0.3)).reason == REASON_TOO_FEW


def test_a_key_failure_excludes_only_that_client_with_its_own_reason():
    p = per_client()
    d = p.decide(CLEAN, {"0": "ok", "1": "no_key", "2": "ok"})
    assert not d.discard_round and d.excluded == ["1"] and d.included == ["0", "2"]
    assert d.clients["1"].reason == REASON_NO_KEY and d.clients["1"].action == X
    assert d.exclusion_reasons == {REASON_NO_KEY: ["1"]}
    d = p.decide(CLEAN, {"2": "abort_auth"})
    assert d.clients["2"].reason == REASON_KEY_ABORT and d.exclusion_reasons == {REASON_KEY_ABORT: ["2"]}


def test_qber_and_key_exclusions_are_logged_apart():
    d = per_client(min_clients=1).decide(q(0.3, 0.01, 0.01), {"1": "no_key"})
    assert d.exclusion_reasons == {REASON_QBER_EXCLUSION: ["0"], REASON_NO_KEY: ["1"]}


def test_a_client_that_is_both_qber_locked_down_and_keyless_keeps_the_qber_reason():
    d = per_client(min_clients=1).decide(q(0.3, 0.01, 0.01), {"0": "no_key"})
    assert d.clients["0"].reason == REASON_QBER_EXCLUSION


def test_too_many_key_failures_discard_the_round():
    d = per_client().decide(CLEAN, {"0": "no_key", "1": "no_key"})
    assert d.discard_round and d.reason == REASON_TOO_FEW


def test_excluded_clients_keep_being_measured_and_rejoin_after_the_dwell():
    p = per_client(hysteresis=HysteresisConfig(margin=0.01, dwell=3))
    assert p.decide(q(0.3)).excluded == ["0"]
    rounds = [p.decide(CLEAN).excluded for _ in range(4)]
    assert rounds == [["0"], ["0"], [], []]      # the link was measured all along, so it can rejoin on its own
    assert p.decide(CLEAN).clients["0"].action == T


def test_without_hysteresis_a_client_rejoins_the_round_its_link_recovers():
    p = per_client()
    assert p.decide(q(0.3)).excluded == ["0"]
    assert p.decide(CLEAN).excluded == []


def test_participation_counts_only_included_clients_in_rounds_that_proceed():
    p = per_client()
    p.decide(CLEAN)
    p.decide(q(0.3, 0.01, 0.01))
    p.decide(q(0.3, 0.3, 0.01))                   # discarded: nobody participates
    d = p.decide(CLEAN)
    assert d.participation == {"0": 2, "1": 3, "2": 3}


def test_screening_flags_follow_the_included_count_and_the_clients_own_state():
    three = per_client().decide(q(0.06, 0.01, 0.01))
    assert three.screening_possible and three.clip_clients == ("0",)  # only the CAUTION client may be clipped
    two = per_client().decide(q(0.3, 0.06, 0.01))                       # one excluded: only 2 updates remain
    assert not two.screening_possible and two.clip_clients == ()
    assert per_client().decide(CLEAN).clip_clients == ()                # SECURE clients are never clipped
    discarded = per_client().decide(q(0.3, 0.3, 0.01))
    assert discarded.included == [] and not discarded.screening_possible


def test_decisions_are_immutable_records():
    d = per_client().decide(CLEAN)
    with pytest.raises(AttributeError):
        d.discard_round = True  # type: ignore[misc]


# --------------------------------------------------------------------------
# Property tests (hypothesis)
# --------------------------------------------------------------------------

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

QBERS = st.floats(min_value=0.0, max_value=0.3, allow_nan=False)
ROUNDS = st.lists(st.tuples(QBERS, QBERS, QBERS), min_size=1, max_size=25)
RANK = {S: 0, C: 1, L: 2}


@given(ROUNDS, st.floats(min_value=0.0, max_value=0.3, allow_nan=False))
def test_a_clients_decision_depends_only_on_its_own_history(rounds, other):
    """Changing the OTHER links never changes client 0's state or action in per_client mode."""
    a, b = per_client(), per_client()
    for r in rounds:
        da = a.decide(q(r[0], r[1], r[2]))
        db = b.decide(q(r[0], other, other))
        assert da.clients["0"].state == db.clients["0"].state and da.clients["0"].action == db.clients["0"].action


@given(ROUNDS)
def test_no_lockdown_client_is_ever_included_and_discard_iff_too_few(rounds):
    p = per_client(min_clients=2)
    for r in rounds:
        d = p.decide(q(*r))
        for cid, client in d.clients.items():
            if client.state == L:
                assert client.action == X and cid not in d.included
        remaining = sum(1 for c in d.clients.values() if c.action != X)
        assert d.discard_round == (remaining < 2)


@given(ROUNDS)
def test_global_modes_depend_only_on_the_worst_link(rounds):
    for cls, mode in ((GlobalPolicy, "global"), (GlobalBinaryPolicy, "global_binary")):
        a, b = cls(PolicyConfig(mode=mode)), cls(PolicyConfig(mode=mode))
        for r in rounds:
            worst = max(r)
            da, db = a.decide(q(*r)), b.decide(q(worst, 0.0, 0.0))
            assert da.discard_round == db.discard_round and da.global_state == db.global_state


@given(ROUNDS)
def test_per_client_never_discards_more_rounds_than_the_global_mode(rounds):
    """The graduated response can only keep training at least as often as the global one."""
    per, glob = per_client(min_clients=1), GlobalPolicy(PolicyConfig(mode="global"))
    for r in rounds:
        d_per, d_glob = per.decide(q(*r)), glob.decide(q(*r))
        assert not (d_glob.discard_round is False and d_per.discard_round is True)


@given(st.floats(min_value=0.0, max_value=0.3, allow_nan=False), st.floats(min_value=0.0, max_value=0.3, allow_nan=False))
def test_a_higher_qber_never_gives_a_milder_action(low, high):
    lo, hi = sorted((low, high))
    a = per_client().decide(q(lo)).clients["0"]
    b = per_client().decide(q(hi)).clients["0"]
    order = {T: 0, TP: 1, X: 2}
    assert order[b.action] >= order[a.action]
