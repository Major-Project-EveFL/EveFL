"""
Per-client decisions inside EveFLStrategy: Eve on one link of three, exclusion, renormalised weights, rejoining,
key failures, screening scope, and the rule that an excluded client's update is never aggregated.
"""

import numpy as np
import pytest
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays

from evefl.fl import client as client_module
from evefl.fl import strategy as strategy_module
from evefl.fl.channels import ChannelPlan, LinkAttack, build_plan
from evefl.fl.client import create_client_fn, get_model_parameters
from evefl.fl.model import build_resnet18
from evefl.fl.runner import build_local_client_proxies, run_sequential_fl
from evefl.fl.strategy import EveFLStrategy
from evefl.orchestration.policy import PolicyConfig
from evefl.orchestration.state_machine import HysteresisConfig
from evefl.quantum.base import QKDResult
from tests.test_screening import _Manager, _Proxy

ATTACK_0 = ChannelPlan(attacks={"0": LinkAttack(alpha=0.8)})  # expected QBER 0.2: LOCKDOWN on link 0 only


class _ExpectedQber:
    """QKD stand-in: reports exactly the channel's expected QBER, so scenarios are deterministic."""

    def __init__(self, *args, **kwargs):
        pass

    def run_exchange(self, n_qubits, intercept_probability=0.0, *, channel=None):
        return QKDResult([], channel.expected_qber, n_qubits, 0, channel.intercept_probability,
                         channel.intercept_probability > 0)


@pytest.fixture(autouse=True)
def stub_qkd(monkeypatch):
    monkeypatch.setattr(strategy_module, "create_protocol", lambda *args, **kwargs: _ExpectedQber())


GLOBAL_PARAMS = [np.zeros((4,), dtype=np.float32)]


def make_strategy(mode="per_client", channels=ATTACK_0, **kwargs):
    policy = kwargs.pop("policy_config", PolicyConfig(mode=mode))
    return EveFLStrategy(initial_parameters=ndarrays_to_parameters(GLOBAL_PARAMS), n_qubits=8,
                         channels=channels, policy_config=policy, **kwargs)


def proxies_and_manager():
    proxies = [_Proxy(str(i)) for i in range(3)]
    return proxies, _Manager(proxies)


def fit_res(value, n):
    return FitRes(status=Status(Code.OK, ""),
                  parameters=ndarrays_to_parameters([np.full(4, value, dtype=np.float32)]),
                  num_examples=n, metrics={"fedprox_mu": 0.0})


def run_round(strategy, server_round, results_by_cid, manager, proxies):
    """configure_fit, then aggregate whatever the given clients 'returned'."""
    instructions = strategy.configure_fit(server_round, ndarrays_to_parameters(GLOBAL_PARAMS), manager)
    results = [(proxies[int(cid)], fit_res(v, n)) for cid, (v, n) in results_by_cid.items()]
    params, log = strategy.aggregate_fit(server_round, results, [])
    return instructions, params, log


# --------------------------------------------------------------------------
# Eve on one link of three
# --------------------------------------------------------------------------

def test_eve_on_one_link_excludes_only_that_client_and_renormalises_the_weights():
    proxies, manager = proxies_and_manager()
    strategy = make_strategy()
    instructions, params, log = run_round(strategy, 1, {"1": (1.0, 10), "2": (3.0, 30)}, manager, proxies)

    assert [p.cid for p, _ in instructions] == ["1", "2"]            # client 0 is not sent the model
    assert log["state"] == "LOCKDOWN" and log["round_discarded"] is False
    assert log["excluded_clients"] == ["0"] and log["included_clients"] == ["1", "2"]
    assert log["exclusion_reasons"] == {"qber_exclusion": ["0"]}
    assert log["qber_per_client"]["0"] == pytest.approx(0.2)       # still measured while excluded
    assert log["model_updated"] is True
    # weights renormalised over the two included clients: (10*1 + 30*3) / 40
    assert parameters_to_ndarrays(params)[0][0] == pytest.approx(2.5)
    assert log["aggregation_weights"] == {"1": pytest.approx(0.25), "2": pytest.approx(0.75)}
    assert log["participation_actual"] == {"1": 1, "2": 1}


def test_global_mode_discards_the_same_attack_round():
    proxies, manager = proxies_and_manager()
    strategy = make_strategy(mode="global")
    instructions, params, log = run_round(strategy, 1, {}, manager, proxies)
    assert instructions == [] and log["round_discarded"] is True and log["discard_reason"] == "qber_lockdown"
    assert log["model_updated"] is False
    assert parameters_to_ndarrays(params)[0][0] == 0.0


def test_global_binary_baseline_also_pauses():
    proxies, manager = proxies_and_manager()
    strategy = make_strategy(mode="global_binary")
    instructions, _, log = run_round(strategy, 1, {}, manager, proxies)
    assert instructions == [] and log["discard_reason"] == "qber_lockdown" and log["policy_mode"] == "global_binary"


def test_two_attacked_links_leave_too_few_clients_and_the_round_is_discarded():
    plan = ChannelPlan(attacks={"0": LinkAttack(alpha=0.8), "1": LinkAttack(alpha=0.8)})
    proxies, manager = proxies_and_manager()
    instructions, params, log = run_round(make_strategy(channels=plan), 1, {}, manager, proxies)
    assert instructions == [] and log["discard_reason"] == "too_few_clients" and log["model_updated"] is False
    assert parameters_to_ndarrays(params)[0][0] == 0.0


def test_an_excluded_clients_update_is_never_aggregated_even_if_it_arrives():
    proxies, manager = proxies_and_manager()
    strategy = make_strategy()
    _, params, log = run_round(
        strategy, 1, {"0": (1e6, 1000), "1": (1.0, 10), "2": (3.0, 30)}, manager, proxies)
    assert log["ignored_updates_from"] == ["0"]
    assert parameters_to_ndarrays(params)[0][0] == pytest.approx(2.5)
    assert "0" not in log["aggregation_weights"] and "0" not in log["participation_actual"]


def test_the_model_is_not_sent_to_an_excluded_client_for_evaluation_either():
    proxies, manager = proxies_and_manager()
    strategy = make_strategy()
    strategy.configure_fit(1, ndarrays_to_parameters(GLOBAL_PARAMS), manager)
    evaluate = strategy.configure_evaluate(1, ndarrays_to_parameters(GLOBAL_PARAMS), manager)
    assert [p.cid for p, _ in evaluate] == ["1", "2"]
    discarded = make_strategy(mode="global")
    discarded.configure_fit(1, ndarrays_to_parameters(GLOBAL_PARAMS), manager)
    assert discarded.configure_evaluate(1, ndarrays_to_parameters(GLOBAL_PARAMS), manager) == []


# --------------------------------------------------------------------------
# Measuring excluded links every round, rejoining
# --------------------------------------------------------------------------

def test_an_excluded_link_keeps_being_measured_and_rejoins_after_the_hysteresis_dwell():
    plan = build_plan(["0:0.8:window@1-2"])
    policy = PolicyConfig(mode="per_client", hysteresis=HysteresisConfig(margin=0.01, dwell=2))
    proxies, manager = proxies_and_manager()
    strategy = make_strategy(channels=plan, policy_config=policy)
    sent_to_zero, measured = [], []
    for r in range(1, 6):
        instructions = strategy.configure_fit(r, ndarrays_to_parameters(GLOBAL_PARAMS), manager)
        sent_to_zero.append("0" in [p.cid for p, _ in instructions])
        measured.append("0" in strategy._round_qber_per_client)
        strategy.aggregate_fit(r, [(p, fit_res(1.0, 10)) for p, _ in instructions], [])
    assert measured == [True] * 5                                     # never skipped, or it could not rejoin
    assert sent_to_zero == [False, False, False, True, True]          # attack ends after round 2; dwell 2


def test_without_hysteresis_the_client_is_back_the_round_after_the_attack_ends():
    plan = build_plan(["0:0.8:window@1-2"])
    proxies, manager = proxies_and_manager()
    strategy = make_strategy(channels=plan)
    flags = []
    for r in range(1, 5):
        instructions = strategy.configure_fit(r, ndarrays_to_parameters(GLOBAL_PARAMS), manager)
        flags.append("0" in [p.cid for p, _ in instructions])
        strategy.aggregate_fit(r, [(p, fit_res(1.0, 10)) for p, _ in instructions], [])
    assert flags == [False, False, True, True]


# --------------------------------------------------------------------------
# Key failures
# --------------------------------------------------------------------------

def test_a_key_failure_excludes_only_that_client_in_per_client_mode_and_is_logged_apart():
    proxies, manager = proxies_and_manager()
    strategy = make_strategy(channels=ChannelPlan(), key_status_fn=lambda r, cid: "no_key" if cid == "2" else "ok")
    instructions, params, log = run_round(
        strategy, 1, {"0": (1.0, 10), "1": (3.0, 10), "2": (100.0, 10)}, manager, proxies)
    assert [p.cid for p, _ in instructions] == ["0", "1"]            # no fallback: client 2 is simply out
    assert log["exclusion_reasons"] == {"no_key": ["2"]} and log["key_status"]["2"] == "no_key"
    assert log["ignored_updates_from"] == ["2"]                       # its update is not aggregated if it arrives
    assert parameters_to_ndarrays(params)[0][0] == pytest.approx(2.0)


def test_a_key_failure_discards_the_round_in_the_global_modes():
    proxies, manager = proxies_and_manager()
    for mode in ("global", "global_binary"):
        strategy = make_strategy(mode=mode, channels=ChannelPlan(),
                                 key_status_fn=lambda r, cid: "abort_auth" if cid == "1" else "ok")
        instructions, _, log = run_round(strategy, 1, {}, manager, proxies)
        assert instructions == [] and log["discard_reason"] == "key_abort" and log["model_updated"] is False


# --------------------------------------------------------------------------
# Screening scope
# --------------------------------------------------------------------------

CAUTION_0 = ChannelPlan(attacks={"0": LinkAttack(alpha=0.24)})   # QBER 0.06: CAUTION on link 0, SECURE on 1 and 2


def test_only_clients_in_caution_are_screened_and_secure_clients_stay_plain_fedavg():
    proxies, manager = proxies_and_manager()
    # client 1 is SECURE with a huge update: must NOT be touched. Client 0 (CAUTION) is normal.
    log = run_round(make_strategy(channels=CAUTION_0), 1,
                    {"0": (0.01, 10), "1": (5.0, 10), "2": (0.0102, 10)}, manager, proxies)[2]
    assert log["screening_active"] is True and log["anomalous_clients"] == []
    assert log["aggregation_weights"]["1"] == pytest.approx(1 / 3)

    # now the CAUTION client is the outlier: it is flagged and down-weighted
    log = run_round(make_strategy(channels=CAUTION_0), 1,
                    {"0": (5.0, 10), "1": (0.01, 10), "2": (0.0102, 10)}, manager, proxies)[2]
    assert log["anomalous_clients"] == ["0"] and log["aggregation_weights"]["0"] < 1 / 3


def test_screening_is_inactive_when_exclusion_leaves_fewer_than_three_updates():
    plan = ChannelPlan(attacks={"0": LinkAttack(alpha=0.8), "1": LinkAttack(alpha=0.24)})
    proxies, manager = proxies_and_manager()
    log = run_round(make_strategy(channels=plan), 1, {"1": (50.0, 10), "2": (0.0, 10)}, manager, proxies)[2]
    assert log["included_clients"] == ["1", "2"]
    assert log["screening_active"] is False and "fewer than 3" in log["screening_inactive_reason"]
    assert log["anomalous_clients"] == []


def test_a_clean_round_has_no_screening_and_no_reason():
    proxies, manager = proxies_and_manager()
    log = run_round(make_strategy(channels=ChannelPlan()), 1,
                    {"0": (1.0, 10), "1": (1.0, 10), "2": (1.0, 10)}, manager, proxies)[2]
    assert log["screening_active"] is False and "screening_inactive_reason" not in log


# --------------------------------------------------------------------------
# Per-client FitIns config
# --------------------------------------------------------------------------

def test_each_client_gets_its_own_state_fedprox_mu_and_learning_rate():
    proxies, manager = proxies_and_manager()
    strategy = make_strategy(channels=CAUTION_0, base_lr=1e-3, num_rounds=10)
    instructions = strategy.configure_fit(1, ndarrays_to_parameters(GLOBAL_PARAMS), manager)
    configs = {p.cid: ins.config for p, ins in instructions}
    assert configs["0"]["state"] == "CAUTION" and configs["0"]["fedprox_mu"] > 0
    assert configs["0"]["learning_rate"] == pytest.approx(5e-4)
    assert configs["1"]["state"] == "SECURE" and configs["1"]["fedprox_mu"] == 0.0
    assert configs["1"]["learning_rate"] == pytest.approx(1e-3)
    assert configs["0"]["qber"] == pytest.approx(0.06) and configs["1"]["qber"] == 0.0


# --------------------------------------------------------------------------
# Construction rules
# --------------------------------------------------------------------------

def test_channels_cannot_be_combined_with_the_legacy_attack_arguments():
    init = ndarrays_to_parameters(GLOBAL_PARAMS)
    with pytest.raises(ValueError, match="channels"):
        EveFLStrategy(init, n_qubits=8, channels=ATTACK_0, intercept_probability=0.3)
    with pytest.raises(ValueError, match="channels"):
        EveFLStrategy(init, n_qubits=8, channels=ATTACK_0, bit_flip_probability=0.01)
    with pytest.raises(ValueError, match="thresholds"):
        from evefl.orchestration.state_machine import StateThresholds
        EveFLStrategy(init, n_qubits=8, policy_config=PolicyConfig(), thresholds=StateThresholds())


def test_legacy_arguments_still_mean_the_same_eve_on_every_link():
    proxies, manager = proxies_and_manager()
    strategy = EveFLStrategy(ndarrays_to_parameters(GLOBAL_PARAMS), n_qubits=8, intercept_probability=0.8)
    strategy.configure_fit(1, ndarrays_to_parameters(GLOBAL_PARAMS), manager)
    assert strategy._round_decision.discard_round and strategy._round_intercept_probability == 0.8


# --------------------------------------------------------------------------
# End to end on synthetic data with the real clients and the sequential runner
# --------------------------------------------------------------------------

def _end_to_end(monkeypatch, synthetic_data, mode, rounds=2):
    data_root, partition_root = synthetic_data
    monkeypatch.setattr(client_module, "build_resnet18", lambda pretrained=True: build_resnet18(pretrained=False))
    init = ndarrays_to_parameters(get_model_parameters(build_resnet18(pretrained=False)))
    strategy = EveFLStrategy(init, n_qubits=8, channels=ATTACK_0, policy_config=PolicyConfig(mode=mode))
    proxies = build_local_client_proxies(create_client_fn(data_root, partition_root, batch_size=4), 3)
    fit_calls = {p.cid: 0 for p in proxies}
    for proxy in proxies:
        original = proxy.fit

        def counted(ins, timeout, group_id, _orig=original, _cid=proxy.cid):
            fit_calls[_cid] += 1
            return _orig(ins, timeout, group_id)

        proxy.fit = counted
    final, history = run_sequential_fl(client_proxies=proxies, strategy=strategy, num_rounds=rounds,
                                       initial_parameters=init, run_federated_evaluate=False)
    return init, final, strategy, history, fit_calls


def test_end_to_end_one_client_excluded_and_the_global_model_still_updates(monkeypatch, synthetic_data):
    init, final, strategy, history, fit_calls = _end_to_end(monkeypatch, synthetic_data, "per_client")
    assert history.failures == []
    assert fit_calls == {"0": 0, "1": 2, "2": 2}                      # the attacked client never trains
    for log in strategy.round_logs:
        assert log["excluded_clients"] == ["0"] and log["model_updated"] is True and log["n_results"] == 2
    changed = any(not np.allclose(a, b) for a, b in zip(parameters_to_ndarrays(init), parameters_to_ndarrays(final)))
    assert changed
    assert strategy.participation_actual == {"1": 2, "2": 2}


def test_end_to_end_global_mode_stops_training_under_the_same_attack(monkeypatch, synthetic_data):
    init, final, strategy, history, fit_calls = _end_to_end(monkeypatch, synthetic_data, "global")
    assert fit_calls == {"0": 0, "1": 0, "2": 0}
    assert all(log["round_discarded"] for log in strategy.round_logs)
    for a, b in zip(parameters_to_ndarrays(init), parameters_to_ndarrays(final)):
        np.testing.assert_array_equal(a, b)
