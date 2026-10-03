"""
Validation split by patient: a third held-out set (disjoint from the hospitals and the test set) for
model selection, so the test set is used for the final report only.
"""

import json

import numpy as np
import pytest
from flwr.common import ndarrays_to_parameters

from evefl.fl import client as client_module
from evefl.fl import strategy as strategy_module
from evefl.fl.client import create_client_fn, get_model_parameters
from evefl.fl.dataset import audit_saved_partition, get_test_dataloader, partition_and_save
from evefl.fl.evaluation import make_evaluate_fn
from evefl.fl.model import build_resnet18
from evefl.fl.partition import audit_partition, split_by_patient
from evefl.fl.runner import build_local_client_proxies, run_sequential_fl
from evefl.fl.strategy import EveFLStrategy
from evefl.quantum.base import QKDResult
from tests.helpers_data import make_metadata, make_metadata_frame


@pytest.fixture(scope="module")
def dataset():
    return make_metadata(1500, max_images=6, seed=21)


@pytest.fixture(scope="module")
def split(dataset):
    ids, labels = dataset
    return split_by_patient(ids, labels, n_clients=3, alpha=0.5, test_fraction=0.1, val_fraction=0.1, seed=3)


# --------------------------------------------------------------------------
# The split itself
# --------------------------------------------------------------------------

def test_hospitals_test_and_validation_are_pairwise_disjoint_by_patient_and_image(dataset, split):
    ids, labels = dataset
    audit = audit_partition(ids, labels, split.client_indices, split.test_indices, split.val_indices)
    assert audit["ok"], audit["overlaps"]
    assert "val" in audit["stats"]
    for pair, counts in audit["overlaps"].items():
        assert counts == {"patients": 0, "images": 0}, pair
    assert any("val" in pair for pair in audit["overlaps"])  # validation really took part in the audit


def test_validation_holds_about_val_fraction_of_patients(dataset, split):
    assert split.n_patients_val == round(split.n_patients_used * 0.1)
    ids, _ = dataset
    assert len(np.unique(ids[split.val_indices])) == split.n_patients_val


def test_every_patient_is_in_exactly_one_of_hospitals_test_val(dataset, split):
    ids, _ = dataset
    groups = list(split.client_indices) + [split.test_indices, split.val_indices]
    everything = np.concatenate(groups)
    assert sorted(everything.tolist()) == list(range(len(ids)))
    homes = {}
    for g, idx in enumerate(groups):
        for pid in np.unique(ids[idx]):
            assert pid not in homes
            homes[pid] = g
    assert sum(split.n_patients_per_client) + split.n_patients_test + split.n_patients_val == split.n_patients_used


def test_without_validation_fraction_the_split_is_unchanged_and_val_is_empty(dataset):
    ids, labels = dataset
    plain = split_by_patient(ids, labels, n_clients=3, alpha=0.5, test_fraction=0.1, seed=3)
    zero = split_by_patient(ids, labels, n_clients=3, alpha=0.5, test_fraction=0.1, val_fraction=0.0, seed=3)
    assert len(plain.val_indices) == 0 and plain.n_patients_val == 0
    assert all(np.array_equal(a, b) for a, b in zip(plain.client_indices, zero.client_indices))
    assert np.array_equal(plain.test_indices, zero.test_indices)


def test_validation_split_is_deterministic(dataset):
    ids, labels = dataset
    kwargs = dict(n_clients=3, alpha=0.5, test_fraction=0.1, val_fraction=0.1, seed=9)
    a, b = split_by_patient(ids, labels, **kwargs), split_by_patient(ids, labels, **kwargs)
    assert np.array_equal(a.val_indices, b.val_indices)


@pytest.mark.parametrize("kwargs", [{"val_fraction": -0.1}, {"val_fraction": 1.0},
                                    {"val_fraction": 0.5, "test_fraction": 0.5}])
def test_invalid_validation_fractions_rejected(dataset, kwargs):
    ids, labels = dataset
    params = {"test_fraction": 0.1, "val_fraction": 0.1, **kwargs}
    with pytest.raises(ValueError):
        split_by_patient(ids, labels, n_clients=3, alpha=0.5, seed=0, **params)


def test_audit_detects_patient_leakage_between_test_and_validation(dataset, split):
    ids, labels = dataset
    leaked_image = split.test_indices[0]
    val_with_leak = np.append(split.val_indices, np.flatnonzero(ids == ids[leaked_image])[0])
    audit = audit_partition(ids, labels, split.client_indices, split.test_indices, val_with_leak)
    assert audit["ok"] is False and audit["overlaps"]["test~val"]["patients"] >= 1


def test_audit_detects_training_patient_in_validation(dataset, split):
    ids, labels = dataset
    stolen = np.flatnonzero(ids == ids[split.client_indices[1][0]])
    audit = audit_partition(ids, labels, split.client_indices, split.test_indices,
                            np.concatenate([split.val_indices, stolen]))
    assert audit["ok"] is False and audit["overlaps"]["hospital_1~val"]["patients"] >= 1


# --------------------------------------------------------------------------
# partition_and_save and loaders
# --------------------------------------------------------------------------

def test_partition_and_save_writes_a_validation_split_by_default(tmp_path):
    make_metadata_frame(400, seed=5).to_csv(tmp_path / "Data_Entry_2017.csv", index=False)
    out = tmp_path / "partitions"
    partition_and_save(tmp_path, out, n_clients=3, alpha=0.5, test_fraction=0.1, seed=7)
    assert (out / "val" / "indices.npy").exists()
    meta = json.loads((out / "partition_meta.json").read_text())
    assert meta["val_fraction"] == 0.1 and meta["n_patients_val"] > 0 and meta["audit"]["ok"] is True
    audit = audit_saved_partition(tmp_path, out)
    assert audit["ok"] is True and "val" in audit["stats"]


def test_partition_and_save_can_skip_validation(tmp_path):
    make_metadata_frame(200, seed=6).to_csv(tmp_path / "Data_Entry_2017.csv", index=False)
    out = tmp_path / "partitions"
    partition_and_save(tmp_path, out, n_clients=3, seed=1, val_fraction=0.0)
    assert not (out / "val").exists()
    assert "val" not in audit_saved_partition(tmp_path, out)["stats"]


def test_loader_split_argument_is_validated_and_needs_the_files(tmp_path):
    with pytest.raises(ValueError, match="split must be"):
        get_test_dataloader(tmp_path, tmp_path, split="train")
    with pytest.raises(FileNotFoundError, match="val partition not found"):
        get_test_dataloader(tmp_path, tmp_path, split="val")


# --------------------------------------------------------------------------
# Runner logs a validation evaluation next to the test evaluation
# --------------------------------------------------------------------------

class _QberStub:
    def __init__(self, *args, **kwargs):
        pass

    def run_exchange(self, n_qubits, intercept_probability=0.0, *, channel=None):
        return QKDResult([], 0.0, n_qubits, 0, 0.0, False)


def test_runner_logs_val_eval_separately_from_the_test_eval(monkeypatch, synthetic_data):
    data_root, partition_root = synthetic_data
    assert (partition_root / "val" / "indices.npy").exists()
    monkeypatch.setattr(strategy_module, "create_protocol", lambda *args, **kwargs: _QberStub())
    monkeypatch.setattr(client_module, "build_resnet18", lambda pretrained=True: build_resnet18(pretrained=False))

    init = ndarrays_to_parameters(get_model_parameters(build_resnet18(pretrained=False)))
    strategy = EveFLStrategy(
        initial_parameters=init, n_qubits=8,
        evaluate_fn=make_evaluate_fn(data_root=data_root, partition_root=partition_root, batch_size=4),
        val_evaluate_fn=make_evaluate_fn(data_root=data_root, partition_root=partition_root, batch_size=4,
                                         split="val"),
    )
    proxies = build_local_client_proxies(create_client_fn(data_root, partition_root, batch_size=4), 3)
    run_sequential_fl(client_proxies=proxies, strategy=strategy, num_rounds=1, initial_parameters=init,
                      run_federated_evaluate=False)

    log = strategy.round_logs[0]
    assert "server_eval" in log and "val_eval" in log
    n_test = len(np.load(partition_root / "test" / "indices.npy"))
    n_val = len(np.load(partition_root / "val" / "indices.npy"))
    assert log["server_eval"]["metrics"]["n_test_examples"] == n_test
    assert log["val_eval"]["metrics"]["n_test_examples"] == n_val  # the field name is shared; the set differs


def test_strategy_without_val_fn_has_no_validation_evaluation():
    init = ndarrays_to_parameters([np.zeros(1, np.float32)])
    assert EveFLStrategy(initial_parameters=init, n_qubits=8).evaluate_validation(1, init) is None
