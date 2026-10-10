"""
Profiling and evaluation-frequency wiring: timing in client metrics, evaluation, round logs and the results JSON;
federated evaluate off by default; test evaluation every N rounds plus the final round, validation every round.
"""

import json

import numpy as np
import pytest
import torch
from flwr.common import ndarrays_to_parameters
from torch.utils.data import DataLoader, TensorDataset

from evefl.fl import client as client_module
from evefl.fl import evaluation as evaluation_module
from evefl.fl import server as server_module
from evefl.fl import strategy as strategy_module
from evefl.fl.client import _evaluate_one_client, _train_one_client, create_client_fn, get_model_parameters
from evefl.fl.evaluation import evaluate_global_model, make_evaluate_fn
from evefl.fl.model import NUM_CLASSES, build_resnet18
from evefl.fl.runner import build_local_client_proxies, run_sequential_fl
from evefl.fl.server import _build_arg_parser, build_experiment_log
from evefl.fl.strategy import EveFLStrategy
from evefl.profiling import LoaderMeter
from evefl.quantum.base import QKDResult

INIT = {"pretrained": False, "source": "random_init", "weights_file": None, "sha256": None}


def tiny_loader(n=8, batch=4):
    images = torch.randn(n, 3, 32, 32)
    targets = (torch.rand(n, NUM_CLASSES) > 0.7).float()
    return DataLoader(TensorDataset(images, targets), batch_size=batch)


class QberStub:
    def __init__(self, *args, **kwargs):
        pass

    def run_exchange(self, n_qubits, intercept_probability=0.0, *, channel=None):
        return QKDResult([], 0.0, n_qubits, 0, 0.0, False)


# --------------------------------------------------------------------------
# Client-level timing
# --------------------------------------------------------------------------

def test_train_one_client_reports_wait_compute_and_image_counts_and_keeps_its_old_keys():
    metrics = _train_one_client(build_resnet18(pretrained=False), tiny_loader(8, 4), device=torch.device("cpu"),
                                epochs=2, lr=1e-3, fedprox_mu=0.0)
    assert {"train_loss", "n_batches"} <= set(metrics)
    assert metrics["n_images"] == 16 and metrics["n_batches"] == 4
    assert metrics["phase_s"] > 0 and metrics["compute_s"] > 0
    assert metrics["data_wait_s"] + metrics["compute_s"] == pytest.approx(metrics["phase_s"])


def test_evaluate_one_client_fills_the_meter_when_given_one_and_still_returns_loss_and_count():
    meter = LoaderMeter()
    loss, n = _evaluate_one_client(build_resnet18(pretrained=False), tiny_loader(8, 4),
                                   device=torch.device("cpu"), meter=meter)
    assert n == 8 and loss == loss and meter.n_items == 8 and meter.n_batches == 2
    loss2, n2 = _evaluate_one_client(build_resnet18(pretrained=False), tiny_loader(8, 4), device=torch.device("cpu"))
    assert n2 == 8  # the meter is optional


def test_server_side_evaluation_carries_a_timing_block(synthetic_data):
    data_root, partition_root = synthetic_data
    _, metrics = evaluate_global_model(get_model_parameters(build_resnet18(pretrained=False)), data_root=data_root,
                                       partition_root=partition_root, batch_size=4)
    timing = metrics["timing"]
    assert {"phase_s", "data_wait_s", "compute_s", "n_images", "images_per_s", "metric_s"} <= set(timing)
    assert timing["n_images"] == metrics["n_test_examples"]
    json.dumps(metrics)


# --------------------------------------------------------------------------
# Runner: per-round timing block
# --------------------------------------------------------------------------

def _run_one_round(monkeypatch, synthetic_data, *, federated):
    data_root, partition_root = synthetic_data
    monkeypatch.setattr(strategy_module, "create_protocol", lambda *args, **kwargs: QberStub())
    monkeypatch.setattr(client_module, "build_resnet18", lambda pretrained=True: build_resnet18(pretrained=False))
    init = ndarrays_to_parameters(get_model_parameters(build_resnet18(pretrained=False)))
    strategy = EveFLStrategy(
        initial_parameters=init, n_qubits=8,
        evaluate_fn=make_evaluate_fn(data_root=data_root, partition_root=partition_root, batch_size=4),
        val_evaluate_fn=make_evaluate_fn(data_root=data_root, partition_root=partition_root, batch_size=4, split="val"),
    )
    proxies = build_local_client_proxies(create_client_fn(data_root, partition_root, batch_size=4), 3)
    run_sequential_fl(client_proxies=proxies, strategy=strategy, num_rounds=1, initial_parameters=init,
                      run_federated_evaluate=federated)
    return strategy.round_logs[0], partition_root


def test_round_log_has_timing_for_fit_and_both_server_evaluations(monkeypatch, synthetic_data):
    log, partition_root = _run_one_round(monkeypatch, synthetic_data, federated=False)
    timing = log["timing"]
    assert timing["round_s"] > 0 and timing["t_end"] > timing["t_start"] >= 0
    phases = timing["phases"]
    assert {"fit", "server_eval_test", "server_eval_val"} <= set(phases) and "federated_eval" not in phases
    train_images = sum(len(np.load(partition_root / f"hospital_{i}" / "indices.npy")) for i in range(3))
    assert phases["fit"]["n_images"] <= train_images  # drop_last may drop a few per client
    assert phases["fit"]["n_images"] > 0 and phases["fit"]["data_wait_s"] >= 0
    for name in ("fit", "server_eval_test", "server_eval_val"):
        assert phases[name]["t_end"] >= phases[name]["t_start"]
    json.dumps(timing)


def test_federated_evaluation_adds_its_own_phase_only_when_enabled(monkeypatch, synthetic_data):
    log, _ = _run_one_round(monkeypatch, synthetic_data, federated=True)
    assert "federated_eval" in log["timing"]["phases"]
    assert log["timing"]["phases"]["federated_eval"]["n_images"] > 0


# --------------------------------------------------------------------------
# Evaluation frequency and defaults
# --------------------------------------------------------------------------

def test_test_split_is_evaluated_every_n_rounds_plus_the_final_round(monkeypatch):
    seen = []

    def stub(ndarrays, **kwargs):
        seen.append(kwargs["split"])
        return 0.0, {}

    monkeypatch.setattr(evaluation_module, "evaluate_global_model", stub)
    fn = make_evaluate_fn(data_root=".", partition_root=".", every_n_rounds=3, num_rounds=7, split="test")
    evaluated = [r for r in range(1, 8) if fn(r, [], {}) is not None]
    assert evaluated == [3, 6, 7]  # every 3rd round and the final one


def test_build_eval_fns_validates_every_round_while_test_follows_its_own_frequency(monkeypatch, tmp_path):
    calls = []

    def fake_make(**kwargs):
        calls.append(kwargs)
        return lambda *a, **k: None

    monkeypatch.setattr(server_module, "make_evaluate_fn", fake_make)
    (tmp_path / "val").mkdir()
    np.save(tmp_path / "val" / "indices.npy", np.arange(3))
    test_fn, val_fn = server_module.build_eval_fns(
        data_root=tmp_path, partition_root=tmp_path, batch_size=8, num_rounds=10, test_eval_every=5)
    by_split = {c.get("split", "test"): c for c in calls}
    assert by_split["test"]["every_n_rounds"] == 5 and by_split["val"]["every_n_rounds"] == 1
    assert val_fn is not None and test_fn is not None
    assert by_split["test"]["num_rounds"] == by_split["val"]["num_rounds"] == 10


def test_build_eval_fns_has_no_validation_function_without_a_val_partition(monkeypatch, tmp_path):
    monkeypatch.setattr(server_module, "make_evaluate_fn", lambda **kw: (lambda *a, **k: None))
    _, val_fn = server_module.build_eval_fns(data_root=tmp_path, partition_root=tmp_path, batch_size=8, num_rounds=3)
    assert val_fn is None


def _parse(*extra):
    return _build_arg_parser().parse_args(["--data-root", "d", "--partition-root", "p", *extra])


def test_federated_evaluate_is_off_by_default_and_the_flag_turns_it_on():
    assert _parse().federated_evaluate is False and _parse("--federated-evaluate").federated_evaluate is True
    assert _parse().test_eval_every == 1 and _parse("--test-eval-every", "4").test_eval_every == 4


def test_old_flags_are_still_accepted_with_their_documented_meaning():
    assert _parse("--no-federated-evaluate").federated_evaluate is False  # deprecated no-op: it is the default now
    assert _parse("--eval-every-n-rounds", "3").test_eval_every == 3      # deprecated alias, applies to the test split only


# --------------------------------------------------------------------------
# Results JSON
# --------------------------------------------------------------------------

def _log(**kw):
    return build_experiment_log(
        experiment_name="t", seed=0, num_clients=3, num_rounds=1, local_epochs=1, batch_size=4, n_qubits=8,
        intercept_probability=0.0, elapsed_seconds=0.0, round_logs=[], n_failures=0, init_weights=INIT, **kw)


def test_results_json_records_profile_and_evaluation_settings_outside_the_hash():
    plain = _log()["experiment"]
    profiled = _log(profile={"device": {"default_device": "cuda"}, "summary": {"data_wait_share_loaded": 0.7}},
                    evaluation={"federated_evaluate": False, "test_eval_every": 5, "val_eval_every": 1})["experiment"]
    assert profiled["profile"]["device"]["default_device"] == "cuda"
    assert profiled["evaluation"] == {"federated_evaluate": False, "test_eval_every": 5, "val_eval_every": 1}
    assert plain["config_hash"] == profiled["config_hash"]
    assert "profile" not in plain and "evaluation" not in plain
