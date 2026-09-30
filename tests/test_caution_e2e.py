"""
End-to-end CAUTION round with real clients, the real strategy and the
sequential runner, on a tiny synthetic ChestX-ray14-shaped dataset.

BB84 is stubbed to a fixed QBER so the state is deterministic and the test
is fast; the QBER -> state mapping and the BB84 statistics are covered in
test_bb84.py / test_state_machine.py. Before the P0-1 fix, every client crashed
here and CAUTION rounds returned no aggregate.
"""

import numpy as np
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

from evefl.fl import client as client_module
from evefl.fl import strategy as strategy_module
from evefl.fl.client import create_client_fn, get_model_parameters
from evefl.fl.model import build_resnet18
from evefl.fl.runner import build_local_client_proxies, run_sequential_fl
from evefl.fl.strategy import EveFLStrategy
from evefl.quantum.base import QKDResult


class _FixedQberProtocol:
    """Stand-in for BB84Protocol that returns a fixed QBER instantly."""

    qber = 0.0

    def __init__(self, *args, **kwargs):
        pass

    def run_exchange(self, n_qubits, intercept_probability=0.0, *, channel=None):
        intercept_probability = channel.intercept_probability if channel else intercept_probability
        return QKDResult(sifted_key=[], qber=type(self).qber, n_qubits_sent=n_qubits, n_sifted=0,
                         intercept_probability=intercept_probability,
                         eavesdropper_active=intercept_probability > 0)


def _run(monkeypatch, synthetic_data, qber, rounds=1):
    data_root, partition_root = synthetic_data
    _FixedQberProtocol.qber = qber
    monkeypatch.setattr(strategy_module, "create_protocol", lambda *args, **kwargs: _FixedQberProtocol())
    # Avoid downloading ImageNet weights in tests.
    monkeypatch.setattr(client_module, "build_resnet18", lambda pretrained=True: build_resnet18(pretrained=False))

    init = ndarrays_to_parameters(get_model_parameters(build_resnet18(pretrained=False)))
    strategy = EveFLStrategy(initial_parameters=init, n_qubits=8)
    proxies = build_local_client_proxies(create_client_fn(data_root, partition_root, batch_size=4), 3)
    final, history = run_sequential_fl(client_proxies=proxies, strategy=strategy, num_rounds=rounds,
                                       initial_parameters=init, run_federated_evaluate=False)
    return init, final, strategy, history


def test_caution_round_runs_fedprox_and_aggregates(monkeypatch, synthetic_data):
    init, final, strategy, history = _run(monkeypatch, synthetic_data, qber=0.08)

    assert history.failures == []
    log = strategy.round_logs[0]
    assert log["state"] == "CAUTION"
    assert log["n_failures"] == 0 and log["n_results"] == 3
    assert log["fedprox_active_clients"] == 3
    assert log["aggregation"] == "fedprox_anomaly_weighted"
    changed = any(not np.allclose(a, b) for a, b in zip(parameters_to_ndarrays(init), parameters_to_ndarrays(final)))
    assert changed, "CAUTION round must produce an updated global model"


def test_secure_round_does_not_use_fedprox(monkeypatch, synthetic_data):
    _, _, strategy, history = _run(monkeypatch, synthetic_data, qber=0.01)
    assert history.failures == []
    assert strategy.round_logs[0]["state"] == "SECURE"
    assert strategy.round_logs[0]["fedprox_active_clients"] == 0


def test_lockdown_round_keeps_global_model_unchanged(monkeypatch, synthetic_data):
    init, final, strategy, _ = _run(monkeypatch, synthetic_data, qber=0.20)
    assert strategy.round_logs[0]["state"] == "LOCKDOWN"
    for a, b in zip(parameters_to_ndarrays(init), parameters_to_ndarrays(final)):
        np.testing.assert_array_equal(a, b)
