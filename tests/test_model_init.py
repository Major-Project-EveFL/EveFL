"""
P0-5 tests: pretrained vs from-scratch initialisation is explicit, loadable offline,
and recorded in the results JSON.
"""

import json

import pytest
import torch
from flwr.common import parameters_to_ndarrays
from torchvision import models

from evefl.fl.model import NUM_CLASSES, build_resnet18, describe_initialisation, file_sha256
from evefl.fl.server import build_experiment_log, build_initial_parameters


@pytest.fixture
def fake_imagenet_file(tmp_path):
    """A stand-in for the cached torchvision ImageNet state dict (1000-class head)."""
    torch.manual_seed(123)
    path = tmp_path / "resnet18_imagenet1k_v1.pth"
    torch.save(models.resnet18(weights=None).state_dict(), path)
    return path


def test_weights_load_offline_from_local_file(fake_imagenet_file):
    reference = torch.load(fake_imagenet_file, weights_only=True)
    model = build_resnet18(pretrained=True, weights_path=fake_imagenet_file)
    assert torch.equal(model.conv1.weight, reference["conv1.weight"])
    assert torch.equal(model.layer4[1].bn2.running_var, reference["layer4.1.bn2.running_var"])
    assert model.fc.out_features == NUM_CLASSES  # head replaced after loading the 1000-class file


def test_from_scratch_differs_from_pretrained_file(fake_imagenet_file):
    torch.manual_seed(0)
    scratch = build_resnet18(pretrained=False)
    loaded = build_resnet18(pretrained=True, weights_path=fake_imagenet_file)
    assert not torch.equal(scratch.conv1.weight, loaded.conv1.weight)


def test_missing_weights_file_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Pretrained weights file not found"):
        build_resnet18(pretrained=True, weights_path=tmp_path / "nope.pth")


def test_weights_path_with_pretrained_false_is_rejected(fake_imagenet_file):
    with pytest.raises(ValueError):
        build_resnet18(pretrained=False, weights_path=fake_imagenet_file)


def test_download_failure_gives_actionable_error(monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("no internet")

    monkeypatch.setattr(models, "resnet18", boom)
    with pytest.raises(RuntimeError, match="pretrained-weights"):
        build_resnet18(pretrained=True)


def test_describe_initialisation_records_source_and_hash(fake_imagenet_file):
    assert describe_initialisation(False)["source"] == "random_init"
    info = describe_initialisation(True, fake_imagenet_file)
    assert info["pretrained"] is True and info["weights_file"] == fake_imagenet_file.name
    assert info["sha256"] == file_sha256(fake_imagenet_file)
    assert describe_initialisation(True)["sha256"] is None


def test_initial_parameters_reflect_the_pretrained_choice(fake_imagenet_file):
    a = parameters_to_ndarrays(build_initial_parameters(True, fake_imagenet_file))
    b = parameters_to_ndarrays(build_initial_parameters(True, fake_imagenet_file))
    c = parameters_to_ndarrays(build_initial_parameters(False))
    assert (a[0] == b[0]).all()  # loading is deterministic
    assert not (a[0] == c[0]).all()


def test_results_json_records_pretrained_and_changes_config_hash(fake_imagenet_file):
    def log_for(pretrained, path=None):
        return build_experiment_log(
            experiment_name="x", seed=1, num_clients=3, num_rounds=2, local_epochs=1, batch_size=4, n_qubits=8,
            intercept_probability=0.0, elapsed_seconds=0.0, round_logs=[], n_failures=0,
            init_weights=describe_initialisation(pretrained, path),
        )["experiment"]

    pre, scratch = log_for(True, fake_imagenet_file), log_for(False)
    assert pre["pretrained"] is True and scratch["pretrained"] is False
    assert pre["init_weights"]["sha256"] == file_sha256(fake_imagenet_file)
    assert pre["config_hash"] != scratch["config_hash"]  # cannot be compared by accident
    assert log_for(True, fake_imagenet_file)["config_hash"] == pre["config_hash"]  # stable
    json.dumps(pre)
