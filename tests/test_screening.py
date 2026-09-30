"""
P0-2 tests: robust screening of update-delta norms (median + floored MAD).

The old rule (`mean + 2*std` over full-parameter norms) could never fire with 3
clients (max z-score is 1.155) and looked at the wrong quantity. These tests pin
the replacement, with emphasis on the exactly-3-client case, where two near
identical honest updates make the raw MAD ~ 0.
"""

import numpy as np
import pytest
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy

from evefl.fl import strategy as strategy_module
from evefl.fl.screening import flag_anomalous_updates, screening_threshold, update_delta_norm
from evefl.fl.strategy import EveFLStrategy
from evefl.quantum.base import QKDResult

# --------------------------------------------------------------------------
# update_delta_norm
# --------------------------------------------------------------------------

def test_delta_norm_is_about_the_change_not_the_weights():
    global_ = [np.full((100,), 1000.0)]
    client = [np.full((100,), 1000.0) + 0.1]
    assert update_delta_norm(client, global_) == pytest.approx(1.0)  # sqrt(100 * 0.01)


def test_delta_norm_is_zero_when_unchanged():
    w = [np.ones((3, 3)), np.arange(4.0)]
    assert update_delta_norm(w, w) == 0.0


def test_delta_norm_ignores_integer_counters():
    global_ = [np.zeros(4), np.array(0, dtype=np.int64)]
    client = [np.zeros(4), np.array(500, dtype=np.int64)]  # num_batches_tracked style counter
    assert update_delta_norm(client, global_) == 0.0


def test_delta_norm_rejects_mismatched_tensor_counts():
    with pytest.raises(ValueError):
        update_delta_norm([np.zeros(2)], [np.zeros(2), np.zeros(2)])


# --------------------------------------------------------------------------
# flag_anomalous_updates: exactly 3 clients
# --------------------------------------------------------------------------

def test_three_clients_flags_the_outlier_only():
    assert flag_anomalous_updates([1.00, 1.05, 10.0]) == [False, False, True]


def test_three_clients_two_near_identical_honest_plus_outlier():
    """MAD is ~0 here; the floor must still let the real outlier through."""
    assert flag_anomalous_updates([1.000, 1.001, 8.0]) == [False, False, True]


def test_three_clients_near_identical_honest_does_not_flag_a_slightly_larger_honest_one():
    """The false positive the MAD floor exists for: raw MAD is ~0.001, so without
    the floor a 1.15x honest client would look like a huge outlier."""
    norms = [1.000, 1.001, 1.15]
    assert flag_anomalous_updates(norms) == [False, False, False]
    # Confirm the floor is what protects it: with no floor it would be flagged.
    assert flag_anomalous_updates(norms, rel_floor=0.0) == [False, False, True]


def test_three_identical_updates_flag_nothing():
    assert flag_anomalous_updates([2.0, 2.0, 2.0]) == [False, False, False]


def test_three_clients_ordinary_non_iid_spread_flags_nothing():
    assert flag_anomalous_updates([0.8, 1.0, 1.3]) == [False, False, False]


def test_outlier_position_does_not_matter():
    assert flag_anomalous_updates([10.0, 1.0, 1.05]) == [True, False, False]
    assert flag_anomalous_updates([1.0, 10.0, 1.05]) == [False, True, False]


def test_all_zero_norms_flag_nothing():
    assert flag_anomalous_updates([0.0, 0.0, 0.0]) == [False, False, False]


def test_too_few_clients_flags_nothing():
    assert flag_anomalous_updates([1.0, 100.0]) == [False, False]
    assert flag_anomalous_updates([5.0]) == [False]
    assert flag_anomalous_updates([]) == []


def test_only_large_updates_are_flagged_not_small_ones():
    assert flag_anomalous_updates([5.0, 5.1, 0.001]) == [False, False, False]


def test_larger_federation_flags_outlier_via_mad():
    norms = [1.0, 1.1, 0.9, 1.05, 0.95, 1.02, 9.0]
    assert flag_anomalous_updates(norms) == [False] * 6 + [True]


def test_threshold_uses_floor_when_mad_is_tiny():
    assert screening_threshold([1.0, 1.0, 1.0], k=3.0, rel_floor=0.25) == pytest.approx(1.75)


@pytest.mark.parametrize("bad", [{"k": 0.0}, {"k": -1.0}, {"rel_floor": -0.1}])
def test_invalid_parameters_rejected(bad):
    with pytest.raises(ValueError):
        flag_anomalous_updates([1.0, 1.0, 1.0], **bad)


# --------------------------------------------------------------------------
# Wired into EveFLStrategy (CAUTION)
# --------------------------------------------------------------------------

class _Proxy(ClientProxy):
    def get_properties(self, ins, timeout, group_id): ...
    def get_parameters(self, ins, timeout, group_id): ...
    def fit(self, ins, timeout, group_id): ...
    def evaluate(self, ins, timeout, group_id): ...
    def reconnect(self, ins, timeout, group_id): ...


class _Manager:
    def __init__(self, proxies):
        self.proxies = proxies

    def num_available(self):
        return len(self.proxies)

    def sample(self, num_clients, min_num_clients=None, criterion=None):
        return self.proxies[:num_clients]


class _CautionProtocol:
    def __init__(self, *args, **kwargs):
        pass

    def run_exchange(self, n_qubits, intercept_probability=0.0, *, channel=None):
        intercept_probability = channel.intercept_probability if channel else intercept_probability
        return QKDResult([], 0.08, n_qubits, 0, intercept_probability, intercept_probability > 0)


def _fit_res(weights, n=10):
    return FitRes(status=Status(Code.OK, ""), parameters=ndarrays_to_parameters(weights),
                  num_examples=n, metrics={"fedprox_mu": 0.01})


def _caution_round(monkeypatch, global_, updates):
    monkeypatch.setattr(strategy_module, "create_protocol", lambda *args, **kwargs: _CautionProtocol())
    strategy = EveFLStrategy(initial_parameters=ndarrays_to_parameters(global_), n_qubits=8)
    proxies = [_Proxy(str(i)) for i in range(3)]
    fit_ins = strategy.configure_fit(1, ndarrays_to_parameters(global_), _Manager(proxies))
    assert strategy._round_state.value == "CAUTION" and len(fit_ins) == 3
    results = [(p, _fit_res([u])) for p, u in zip(proxies, updates)]
    params, _ = strategy.aggregate_fit(1, results, [])
    return strategy.round_logs[0], parameters_to_ndarrays(params)[0]


def test_strategy_flags_scaled_update_that_full_norm_rule_would_miss(monkeypatch):
    """Global weights are large, so full-parameter norms are nearly identical
    (the old rule's blind spot); only the delta norm reveals the outlier."""
    global_ = [np.full((1000,), 1000.0, dtype=np.float32)]
    updates = [global_[0] + d for d in (0.01, 0.0102, 0.5)]
    log, _ = _caution_round(monkeypatch, global_, updates)
    assert log["anomalous_clients"] == ["2"]
    assert log["update_delta_norms"]["2"] > 10 * log["update_delta_norms"]["0"]


def test_strategy_flags_nothing_for_two_near_identical_honest_plus_slightly_larger(monkeypatch):
    global_ = [np.full((1000,), 1000.0, dtype=np.float32)]
    updates = [global_[0] + d for d in (0.0100, 0.0101, 0.0115)]
    log, _ = _caution_round(monkeypatch, global_, updates)
    assert log["anomalous_clients"] == []


def test_flagged_client_is_downweighted_in_aggregate(monkeypatch):
    global_ = [np.zeros((4,), dtype=np.float32)]
    updates = [np.full(4, 0.01, np.float32), np.full(4, 0.0102, np.float32), np.full(4, 5.0, np.float32)]
    log, aggregated = _caution_round(monkeypatch, global_, updates)
    assert log["anomalous_clients"] == ["2"]
    # equal example counts, outlier weight halved -> weights 0.4 / 0.4 / 0.2
    assert aggregated[0] == pytest.approx(0.4 * 0.01 + 0.4 * 0.0102 + 0.2 * 5.0, rel=1e-4)
