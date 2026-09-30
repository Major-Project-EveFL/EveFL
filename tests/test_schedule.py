"""
P1-8 tests: AdamW + cosine annealing that follows the GLOBAL round (schedule.py), with a
reduced rate in CAUTION, sent to clients in the round config.
"""


import numpy as np
import pytest
import torch
from flwr.common import ndarrays_to_parameters

from evefl.fl import client as client_module
from evefl.fl import strategy as strategy_module
from evefl.fl.client import EveFLClient, _train_one_client
from evefl.fl.model import build_resnet18
from evefl.fl.schedule import cosine_lr
from evefl.fl.strategy import EveFLStrategy
from evefl.quantum.base import QKDResult
from tests.test_screening import _Manager, _Proxy

# --------------------------------------------------------------------------
# cosine_lr
# --------------------------------------------------------------------------

def test_first_round_uses_the_base_rate():
    assert cosine_lr(1e-3, 1, 50) == pytest.approx(1e-3)


def test_rate_decreases_every_round_and_stays_positive():
    rates = [cosine_lr(1e-3, r, 50) for r in range(1, 51)]
    assert all(a > b for a, b in zip(rates, rates[1:]))
    assert rates[-1] > 0


def test_midpoint_is_about_half_the_base_rate():
    assert cosine_lr(1.0, 26, 50) == pytest.approx(0.5, abs=1e-9)  # (t-1)/T = 0.5 -> cos(pi/2) = 0


def test_matches_torch_cosine_annealing_stepped_once_per_round():
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([param], lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)
    for round_ in range(1, 21):
        assert opt.param_groups[0]["lr"] == pytest.approx(cosine_lr(1e-3, round_, 20))
        opt.step()
        sched.step()


def test_eta_min_is_respected():
    assert cosine_lr(1e-3, 50, 50, eta_min=1e-4) > 1e-4
    assert cosine_lr(1e-3, 1, 50, eta_min=1e-4) == pytest.approx(1e-3)


@pytest.mark.parametrize("args", [(0.0, 1, 10), (1e-3, 0, 10), (1e-3, 11, 10), (1e-3, 1, 0)])
def test_invalid_arguments_rejected(args):
    with pytest.raises(ValueError):
        cosine_lr(*args)


# --------------------------------------------------------------------------
# Strategy sends the per-round rate (does not restart each round)
# --------------------------------------------------------------------------

class _QberEqualsAlpha:
    def __init__(self, *args, **kwargs):
        pass

    def run_exchange(self, n_qubits, intercept_probability=0.0, *, channel=None):
        intercept_probability = channel.intercept_probability if channel else intercept_probability
        return QKDResult([], intercept_probability, n_qubits, 0, intercept_probability, intercept_probability > 0)


def _round_config(monkeypatch, strategy_kwargs, server_round, alpha):
    monkeypatch.setattr(strategy_module, "create_protocol", lambda *args, **kwargs: _QberEqualsAlpha())
    init = ndarrays_to_parameters([np.zeros(2, np.float32)])
    strategy = EveFLStrategy(initial_parameters=init, n_qubits=8, intercept_probability=alpha, **strategy_kwargs)
    proxies = [_Proxy(str(i)) for i in range(3)]
    (_, fit_ins), *_ = strategy.configure_fit(server_round, init, _Manager(proxies))
    return fit_ins.config, strategy


def test_config_carries_the_scheduled_rate_for_the_global_round(monkeypatch):
    kwargs = {"base_lr": 1e-3, "num_rounds": 10}
    lrs = [_round_config(monkeypatch, kwargs, r, alpha=0.0)[0]["learning_rate"] for r in (1, 5, 10)]
    assert lrs == pytest.approx([cosine_lr(1e-3, r, 10) for r in (1, 5, 10)])
    assert lrs[0] > lrs[1] > lrs[2]  # decays with the global round instead of restarting at 1e-3


def test_caution_uses_the_reduced_rate(monkeypatch):
    kwargs = {"base_lr": 1e-3, "num_rounds": 10}
    secure, _ = _round_config(monkeypatch, kwargs, 1, alpha=0.0)
    caution, strategy = _round_config(monkeypatch, kwargs, 1, alpha=0.08)
    assert secure["learning_rate"] == pytest.approx(1e-3)
    assert caution["learning_rate"] == pytest.approx(5e-4)
    assert caution["state"] == "CAUTION"
    assert strategy._round_learning_rate == pytest.approx(5e-4)


def test_no_base_lr_means_no_learning_rate_in_config(monkeypatch):
    config, _ = _round_config(monkeypatch, {}, 1, alpha=0.0)
    assert "learning_rate" not in config


def test_without_num_rounds_the_base_rate_is_constant(monkeypatch):
    a, _ = _round_config(monkeypatch, {"base_lr": 1e-3}, 1, alpha=0.0)
    b, _ = _round_config(monkeypatch, {"base_lr": 1e-3}, 7, alpha=0.0)
    assert a["learning_rate"] == b["learning_rate"] == 1e-3


# --------------------------------------------------------------------------
# Client: AdamW, and it uses the rate it is sent
# --------------------------------------------------------------------------

def test_training_uses_adamw_with_the_given_rate(monkeypatch):
    created = []
    real_adamw = torch.optim.AdamW

    def spy(*args, **kwargs):
        opt = real_adamw(*args, **kwargs)
        created.append(opt)
        return opt

    monkeypatch.setattr(torch.optim, "AdamW", spy)
    loader = [(torch.randn(2, 3, 32, 32), torch.randint(0, 2, (2, 14)).float())]
    _train_one_client(build_resnet18(pretrained=False), loader, device=torch.device("cpu"),
                      epochs=1, lr=7e-4, fedprox_mu=0.0, weight_decay=0.02)
    assert len(created) == 1
    assert created[0].param_groups[0]["lr"] == pytest.approx(7e-4)
    assert created[0].param_groups[0]["weight_decay"] == pytest.approx(0.02)


def test_client_fit_uses_config_learning_rate_over_its_own(monkeypatch, synthetic_data):
    data_root, partition_root = synthetic_data
    seen = {}

    def fake_train(model, loader, *, device, epochs, lr, fedprox_mu, weight_decay):
        seen["lr"] = lr
        return {"train_loss": 0.0, "n_batches": 1}

    monkeypatch.setattr(client_module, "_train_one_client", fake_train)
    client = EveFLClient("0", data_root, partition_root, batch_size=4, lr=1e-3)
    params = client_module.get_model_parameters(client.model)

    client.fit(params, {"state": "SECURE", "learning_rate": 2.5e-4})
    assert seen["lr"] == pytest.approx(2.5e-4)

    client.fit(params, {"state": "SECURE"})  # no scheduled rate -> falls back to its own base rate
    assert seen["lr"] == pytest.approx(1e-3)
