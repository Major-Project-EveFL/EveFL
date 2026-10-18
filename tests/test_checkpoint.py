"""
Model checkpoints (`--save-model`): the final-round model and the round with the best VALIDATION AUC, never selected on the test split.

The saver keeps a copy of the best parameters in memory, writes two files at the end of the run plus a manifest recorded in the
results JSON, and does not touch training, so the config hash of a run is unchanged by saving.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from flwr.common import ndarrays_to_parameters

from evefl.fl import checkpoint as ck
from evefl.fl.client import get_model_parameters
from evefl.fl.model import build_resnet18
from evefl.fl.runner import run_sequential_fl
from evefl.fl.server import _build_arg_parser, run_experiment

AUC = "macro_auc_roc_excl_thin"


def _arrays(value: float):
    """A real ResNet-18 parameter list whose first float tensor is shifted by `value` (so rounds are distinguishable)."""
    torch.manual_seed(0)   # the same random initialisation every call, so only `value` distinguishes two lists
    arrays = get_model_parameters(build_resnet18(pretrained=False))
    first = next(i for i, a in enumerate(arrays) if a.dtype.kind == "f")
    arrays[first] = arrays[first] + np.float32(value)
    return arrays, first


def _val(auc: float):
    return 0.2, {AUC: auc}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# selection: best validation AUC, ties to the later round
# --------------------------------------------------------------------------

def test_best_checkpoint_is_the_highest_validation_auc_and_ties_go_to_the_later_round(tmp_path):
    saver = ck.CheckpointSaver(tmp_path, name="run", metadata={})
    for round_no, auc in [(1, 0.5), (2, 0.7), (3, 0.7), (4, 0.6)]:
        saver.observe(round_no, ndarrays_to_parameters(_arrays(float(round_no))[0]), _val(auc))
    manifest = saver.finalize(ndarrays_to_parameters(_arrays(4.0)[0]), final_round=4)
    assert manifest["best_validation"]["round"] == 3 and manifest["best_validation"]["val_auc_excl_thin"] == 0.7
    assert manifest["final"]["round"] == 4


def test_nan_and_missing_validation_results_never_select_a_checkpoint(tmp_path):
    saver = ck.CheckpointSaver(tmp_path, name="run", metadata={})
    saver.observe(1, ndarrays_to_parameters(_arrays(1.0)[0]), _val(float("nan")))
    saver.observe(2, ndarrays_to_parameters(_arrays(2.0)[0]), None)
    saver.observe(3, ndarrays_to_parameters(_arrays(3.0)[0]), (0.2, {"loss_only": 1.0}))
    manifest = saver.finalize(ndarrays_to_parameters(_arrays(3.0)[0]), final_round=3)
    assert manifest["best_validation"] is None and "no validation" in manifest["note"].lower()
    assert (tmp_path / "run_final.pt").exists() and not (tmp_path / "run_best_val.pt").exists()


def test_the_best_parameters_are_copied_so_later_changes_do_not_alter_them(tmp_path):
    saver = ck.CheckpointSaver(tmp_path, name="run", metadata={})
    arrays, first = _arrays(5.0)
    saver.observe(1, ndarrays_to_parameters(arrays), _val(0.9))
    arrays[first][...] = 0.0   # the caller reuses its buffers
    saver.finalize(ndarrays_to_parameters(_arrays(0.0)[0]), final_round=2)
    saved = torch.load(tmp_path / "run_best_val.pt", weights_only=True)
    reference = _arrays(5.0)[0][first]
    assert np.array_equal(list(saved["state_dict"].values())[first].numpy(), reference)


# --------------------------------------------------------------------------
# files and manifest
# --------------------------------------------------------------------------

def test_files_load_with_weights_only_into_a_resnet18_and_carry_round_and_metadata(tmp_path):
    saver = ck.CheckpointSaver(tmp_path, name="run", metadata={"config_hash": "abc", "cache_sha256": "d" * 64})
    saver.observe(7, ndarrays_to_parameters(_arrays(1.0)[0]), _val(0.8))
    manifest = saver.finalize(ndarrays_to_parameters(_arrays(2.0)[0]), final_round=9)
    for key, expected_round in (("final", 9), ("best_validation", 7)):
        payload = torch.load(tmp_path / manifest[key]["file"], weights_only=True)
        assert payload["format"] == "evefl-checkpoint-v1" and payload["round"] == expected_round
        assert payload["metadata"]["config_hash"] == "abc"
        model = build_resnet18(pretrained=False)
        model.load_state_dict(payload["state_dict"], strict=True)
    assert torch.load(tmp_path / manifest["best_validation"]["file"], weights_only=True)["val_auc_excl_thin"] == 0.8


def test_manifest_hashes_match_the_files_and_it_is_json_ready(tmp_path):
    saver = ck.CheckpointSaver(tmp_path, name="run", metadata={})
    saver.observe(1, ndarrays_to_parameters(_arrays(1.0)[0]), _val(0.6))
    manifest = saver.finalize(ndarrays_to_parameters(_arrays(2.0)[0]), final_round=2)
    json.dumps(manifest)
    for key in ("final", "best_validation"):
        assert manifest[key]["sha256"] == _sha(tmp_path / manifest[key]["file"])
        assert manifest[key]["bytes"] == (tmp_path / manifest[key]["file"]).stat().st_size
    assert manifest["selection"].startswith("highest validation") and "test" in manifest["selection"].lower()
    assert (tmp_path / "run_checkpoints.json").exists()


def test_the_selection_code_never_reads_the_test_split():
    source = Path(ck.__file__).read_text(encoding="utf-8")
    assert "server_eval" not in source and "test_eval" not in source


def test_a_wrong_number_of_tensors_is_refused(tmp_path):
    saver = ck.CheckpointSaver(tmp_path, name="run", metadata={})
    with pytest.raises(ValueError):
        saver.observe(1, ndarrays_to_parameters(_arrays(1.0)[0][:-1]), _val(0.9))


# --------------------------------------------------------------------------
# wiring: the runner hook and the CLI flag
# --------------------------------------------------------------------------

def test_the_runner_calls_the_round_end_hook_once_per_round_in_order(synthetic_data, tmp_path):
    from evefl.fl import server as server_module

    seen = []
    original = server_module.run_sequential_fl

    def spy(**kwargs):
        user_hook = kwargs.get("on_round_end")
        assert user_hook is not None

        def hook(round_no, parameters, val_eval):
            seen.append((round_no, val_eval is not None))
            user_hook(round_no, parameters, val_eval)

        kwargs["on_round_end"] = hook
        return original(**kwargs)

    data_root, partition_root = synthetic_data
    server_module.run_sequential_fl = spy
    try:
        run_experiment(data_root=data_root, partition_root=partition_root, num_clients=3, num_rounds=2, local_epochs=1,
                       batch_size=4, n_qubits=1024, intercept_probability=0.0, seed=0, experiment_name="hook",
                       output_path=tmp_path / "r.json", pretrained=False, save_model_dir=tmp_path / "ck")
    finally:
        server_module.run_sequential_fl = original
    assert seen == [(1, True), (2, True)]
    assert run_sequential_fl is original


def test_save_model_writes_two_checkpoints_records_them_and_leaves_the_config_hash_alone(synthetic_data, tmp_path):
    data_root, partition_root = synthetic_data

    def run(name, **extra):
        out = tmp_path / f"{name}.json"
        run_experiment(data_root=data_root, partition_root=partition_root, num_clients=3, num_rounds=2, local_epochs=1,
                       batch_size=4, n_qubits=1024, intercept_probability=0.0, seed=0, experiment_name="same",
                       output_path=out, pretrained=False, **extra)
        return json.loads(out.read_text())

    plain = run("plain")
    saved = run("saved", save_model_dir=tmp_path / "ck")
    assert saved["experiment"]["config_hash"] == plain["experiment"]["config_hash"]
    assert "checkpoints" not in plain["experiment"]
    manifest = saved["experiment"]["checkpoints"]
    assert manifest["final"]["round"] == 2
    val_auc = [r["val_eval"]["metrics"][AUC] for r in saved["rounds"]]
    # On this tiny synthetic split every class is "thin", so the validation AUC is NaN in every round: no round qualifies and no
    # best-validation file is written (the selection itself is covered by the unit tests above).
    assert all(v != v for v in val_auc)
    assert manifest["best_validation"] is None and "no validation" in manifest["note"].lower()
    path = tmp_path / "ck" / manifest["final"]["file"]
    assert path.exists() and manifest["final"]["sha256"] == _sha(path)
    assert not (tmp_path / "ck" / "same_best_val.pt").exists()
    assert manifest["metadata"]["config_hash"] == saved["experiment"]["config_hash"]


def test_cli_has_an_optional_save_model_flag_and_checkpoints_are_git_ignored():
    parser = _build_arg_parser()
    assert parser.parse_args(["--data-root", "d", "--partition-root", "p"]).save_model is None
    args = parser.parse_args(["--data-root", "d", "--partition-root", "p", "--save-model", "ck"])
    assert args.save_model == Path("ck")
    ignore = (Path(__file__).resolve().parents[1] / ".gitignore").read_text(encoding="utf-8")
    assert "*.pt" in ignore and "*.pth" in ignore
