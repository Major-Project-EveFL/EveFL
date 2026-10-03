"""
P0-3 tests: server-side held-out evaluation inside the sequential runner.

Checks that per-round macro / per-class AUC reach the results JSON on every round
(LOCKDOWN included, so the AUC curve shows flat stretches), and that classes with
no positives in the test split are skipped and named rather than scored.
"""

import json

import numpy as np
import pytest
from flwr.common import ndarrays_to_parameters

from evefl.fl import client as client_module
from evefl.fl import strategy as strategy_module
from evefl.fl.client import create_client_fn, get_model_parameters
from evefl.fl.evaluation import (
    MIN_POSITIVES_FOR_SELECTION,
    _per_class_auc_roc,
    _support_summary,
    evaluate_global_model,
    make_evaluate_fn,
)
from evefl.fl.model import CHESTXRAY_LABELS, build_resnet18
from evefl.fl.runner import build_local_client_proxies, run_sequential_fl
from evefl.fl.server import build_experiment_log
from evefl.fl.strategy import EveFLStrategy
from evefl.quantum.base import QKDResult


class _QberEqualsAlphaProtocol:
    """BB84 stand-in: reports QBER == the round's intercept probability, so a schedule of
    alphas directly scripts the states (0.0 SECURE, 0.08 CAUTION, 0.20 LOCKDOWN)."""

    def __init__(self, *args, **kwargs):
        pass

    def run_exchange(self, n_qubits, intercept_probability=0.0, *, channel=None):
        intercept_probability = channel.intercept_probability if channel else intercept_probability
        return QKDResult([], intercept_probability, n_qubits, 0, intercept_probability, intercept_probability > 0)


# --------------------------------------------------------------------------
# _per_class_auc_roc: skipping
# --------------------------------------------------------------------------

def test_classes_without_positives_or_negatives_are_skipped_and_named():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=(40, len(CHESTXRAY_LABELS))).astype(np.float32)
    y_true[:, 0] = 0.0   # Atelectasis: no positives
    y_true[:, 13] = 1.0  # Hernia: no negatives
    per_class, skipped = _per_class_auc_roc(y_true, rng.random(y_true.shape))
    assert skipped == ["Atelectasis", "Hernia"]
    assert "Atelectasis" not in per_class and "Hernia" not in per_class
    assert len(per_class) == len(CHESTXRAY_LABELS) - 2


def test_perfect_scores_give_auc_one():
    y_true = np.tile(np.array([[0.0], [1.0], [0.0], [1.0]], dtype=np.float32), (1, len(CHESTXRAY_LABELS)))
    per_class, skipped = _per_class_auc_roc(y_true, y_true.copy())
    assert skipped == [] and all(v == 1.0 for v in per_class.values())


# --------------------------------------------------------------------------
# evaluate_global_model on the synthetic hold-out
# --------------------------------------------------------------------------

def test_evaluate_global_model_returns_json_ready_structured_metrics(synthetic_data):
    data_root, partition_root = synthetic_data
    model = build_resnet18(pretrained=False)
    loss, metrics = evaluate_global_model(get_model_parameters(model), data_root=data_root,
                                          partition_root=partition_root, batch_size=4)
    assert loss == loss  # not NaN
    assert set(metrics) == {"macro_auc_roc", "per_class_auc", "skipped_classes", "n_scored_classes", "n_test_examples",
                            "n_positives_per_class", "thin_classes", "min_positives", "macro_auc_roc_excl_thin"}
    assert metrics["min_positives"] == MIN_POSITIVES_FOR_SELECTION
    assert set(metrics["n_positives_per_class"]) == set(CHESTXRAY_LABELS)
    assert metrics["n_test_examples"] > 0
    # The scored and skipped lists partition the 14 classes.
    assert set(metrics["per_class_auc"]) | set(metrics["skipped_classes"]) == set(CHESTXRAY_LABELS)
    assert not set(metrics["per_class_auc"]) & set(metrics["skipped_classes"])
    assert metrics["n_scored_classes"] == len(metrics["per_class_auc"])
    if metrics["per_class_auc"]:
        assert metrics["macro_auc_roc"] == pytest.approx(np.mean(list(metrics["per_class_auc"].values())))
    json.dumps(metrics, allow_nan=True)  # serialisable


# --------------------------------------------------------------------------
# Inside the sequential runner
# --------------------------------------------------------------------------

def test_runner_evaluates_every_round_including_lockdown_and_writes_json(monkeypatch, synthetic_data, tmp_path):
    data_root, partition_root = synthetic_data
    monkeypatch.setattr(strategy_module, "create_protocol", lambda *args, **kwargs: _QberEqualsAlphaProtocol())
    monkeypatch.setattr(client_module, "build_resnet18", lambda pretrained=True: build_resnet18(pretrained=False))

    alphas = {1: 0.0, 2: 0.20, 3: 0.0}  # SECURE, LOCKDOWN, SECURE
    init = ndarrays_to_parameters(get_model_parameters(build_resnet18(pretrained=False)))
    strategy = EveFLStrategy(
        initial_parameters=init, n_qubits=8, intercept_probability_schedule=lambda r: alphas[r],
        evaluate_fn=make_evaluate_fn(data_root=data_root, partition_root=partition_root, batch_size=4, num_rounds=3),
    )
    proxies = build_local_client_proxies(create_client_fn(data_root, partition_root, batch_size=4), 3)
    run_sequential_fl(client_proxies=proxies, strategy=strategy, num_rounds=3, initial_parameters=init,
                      run_federated_evaluate=False)

    logs = strategy.round_logs
    assert [r["state"] for r in logs] == ["SECURE", "LOCKDOWN", "SECURE"]
    assert all("server_eval" in r for r in logs), "every round, LOCKDOWN included, must be evaluated"
    assert [r["model_updated"] for r in logs] == [True, False, True]

    # The LOCKDOWN round evaluates the unchanged model -> identical numbers to round 1 (flat stretch).
    assert logs[1]["server_eval"]["loss"] == pytest.approx(logs[0]["server_eval"]["loss"])
    assert logs[1]["server_eval"]["metrics"]["per_class_auc"] == logs[0]["server_eval"]["metrics"]["per_class_auc"]
    # ... and a real update changes it.
    assert logs[2]["server_eval"]["loss"] != pytest.approx(logs[1]["server_eval"]["loss"])

    # Round-trips through the results JSON with per-round macro AUC, per-class AUC and skipped classes.
    experiment_log = build_experiment_log(
        experiment_name="t", seed=0, num_clients=3, num_rounds=3, local_epochs=1, batch_size=4, n_qubits=8,
        intercept_probability=0.0, elapsed_seconds=0.0, round_logs=logs, n_failures=0,
        init_weights={"pretrained": False, "source": "random_init", "weights_file": None, "sha256": None},
    )
    path = tmp_path / "results.json"
    path.write_text(json.dumps(experiment_log))
    loaded = json.loads(path.read_text())
    for r in loaded["rounds"]:
        m = r["server_eval"]["metrics"]
        assert {"macro_auc_roc", "per_class_auc", "skipped_classes"} <= set(m)


# --------------------------------------------------------------------------
# Thin-class flag (pre-registered model-selection rule, docs/06): classes with fewer than 20 positives
# --------------------------------------------------------------------------

def _labels_with_support(positives_per_class, n=400):
    y = np.zeros((n, len(CHESTXRAY_LABELS)), dtype=np.float32)
    for i, k in enumerate(positives_per_class):
        y[:k, i] = 1.0
    return y


def test_min_positives_threshold_is_twenty():
    assert MIN_POSITIVES_FOR_SELECTION == 20


def test_thin_class_is_flagged_and_left_out_of_the_selection_macro_only():
    support = [100] * 13 + [10]  # Hernia has 10 positives, as in the real validation split
    y_true = _labels_with_support(support)
    rng = np.random.default_rng(1)
    y_score = np.clip(y_true * 0.5 + rng.random(y_true.shape) * 0.6, 0, 1)
    per_class, skipped = _per_class_auc_roc(y_true, y_score)
    summary = _support_summary(y_true, per_class)
    assert skipped == [] and summary["thin_classes"] == ["Hernia"]
    assert summary["n_positives_per_class"]["Hernia"] == 10
    others = [v for k, v in per_class.items() if k != "Hernia"]
    assert summary["macro_auc_roc_excl_thin"] == pytest.approx(np.mean(others))
    assert summary["macro_auc_roc_excl_thin"] != pytest.approx(np.mean(list(per_class.values())))
    assert "Hernia" in per_class  # still scored and reported per class


def test_exactly_twenty_positives_is_not_thin_and_nineteen_is():
    y_true = _labels_with_support([20] + [19] + [100] * 12)
    per_class = {label: 0.7 for label in CHESTXRAY_LABELS}
    summary = _support_summary(y_true, per_class)
    assert summary["thin_classes"] == [CHESTXRAY_LABELS[1]]


def test_too_few_negatives_is_also_thin():
    y_true = _labels_with_support([100] * 14, n=400)
    y_true[:, 0] = 1.0
    y_true[:15, 0] = 0.0  # 15 negatives
    summary = _support_summary(y_true, {label: 0.7 for label in CHESTXRAY_LABELS})
    assert summary["thin_classes"] == [CHESTXRAY_LABELS[0]]


def test_skipped_classes_are_not_counted_as_thin_and_all_thin_gives_nan():
    y_true = _labels_with_support([5] * 14)
    per_class = {label: 0.7 for label in CHESTXRAY_LABELS}
    summary = _support_summary(y_true, per_class)
    assert len(summary["thin_classes"]) == 14 and summary["macro_auc_roc_excl_thin"] != summary["macro_auc_roc_excl_thin"]
    per_class.pop(CHESTXRAY_LABELS[0])  # skipped by the AUC code: absent from per_class
    assert CHESTXRAY_LABELS[0] not in _support_summary(y_true, per_class)["thin_classes"]
