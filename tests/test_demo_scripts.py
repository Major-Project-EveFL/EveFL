"""
Two small scripts for the demonstration:
- `scripts/export_demo_samples.py` copies a few public test-split chest X-rays (seeded) with their recorded labels, outside git;
- `scripts/eval_checkpoint.py` evaluates a saved checkpoint once on the test split, after its selection by validation.
"""

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from flwr.common import ndarrays_to_parameters

from evefl.fl.checkpoint import CheckpointSaver
from evefl.fl.client import get_model_parameters
from evefl.fl.model import build_resnet18

REPO = Path(__file__).resolve().parents[1]
AUC = "macro_auc_roc_excl_thin"


def _load(name):
    path = REPO / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# export_demo_samples
# --------------------------------------------------------------------------

def test_export_picks_only_test_split_images_seeded_and_writes_labels_and_copies(synthetic_data, tmp_path):
    import numpy as np

    data_root, partition_root = synthetic_data
    out = tmp_path / "samples"
    module = _load("export_demo_samples")
    summary = module.main(["--data-root", str(data_root), "--partition-root", str(partition_root), "--output", str(out),
                           "--n", "3", "--seed", "5"])
    meta = json.loads((out / "samples.json").read_text(encoding="utf-8"))
    assert len(meta["samples"]) == len(summary["samples"]) == 3
    test_rows = set(np.load(partition_root / "test" / "indices.npy").tolist())
    import pandas as pd

    names = pd.read_csv(data_root / "Data_Entry_2017.csv")["Image Index"].tolist()
    test_names = {names[i] for i in test_rows}
    for entry in meta["samples"]:
        assert entry["file"] in test_names and (out / entry["file"]).exists() and entry["split"] == "test"
        assert isinstance(entry["labels"], list) and entry["labels"]
    assert "public" in meta["note"].lower() and "not diagnostic" in meta["note"].lower()
    again = module.main(["--data-root", str(data_root), "--partition-root", str(partition_root),
                         "--output", str(tmp_path / "again"), "--n", "3", "--seed", "5"])
    assert [s["file"] for s in again["samples"]] == [s["file"] for s in summary["samples"]]


def test_export_refuses_more_samples_than_the_test_split_holds(synthetic_data, tmp_path):
    data_root, partition_root = synthetic_data
    with pytest.raises(SystemExit):
        _load("export_demo_samples").main(["--data-root", str(data_root), "--partition-root", str(partition_root),
                                           "--output", str(tmp_path / "s"), "--n", "9999"])


# --------------------------------------------------------------------------
# eval_checkpoint
# --------------------------------------------------------------------------

def _saved(tmp_path):
    torch.manual_seed(0)
    params = ndarrays_to_parameters(get_model_parameters(build_resnet18(pretrained=False)))
    saver = CheckpointSaver(tmp_path, name="run", metadata={"config_hash": "abc123"})
    saver.observe(4, params, (0.2, {AUC: 0.6}))
    saver.finalize(params, final_round=6)
    return tmp_path / "run_best_val.pt"


def test_eval_checkpoint_scores_the_test_split_once_and_records_how_the_checkpoint_was_chosen(synthetic_data, tmp_path):
    data_root, partition_root = synthetic_data
    checkpoint = _saved(tmp_path)
    out = tmp_path / "eval.json"
    result = _load("eval_checkpoint").main(["--checkpoint", str(checkpoint), "--data-root", str(data_root),
                                            "--partition-root", str(partition_root), "--batch-size", "4", "--output", str(out)])
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["split"] == "test" and saved["checkpoint"]["round"] == 4 and saved["checkpoint"]["kind"] == "best_val"
    assert len(saved["checkpoint"]["sha256"]) == 64
    assert "evaluated once on the test split" in saved["label"] and "after selection by validation" in saved["label"]
    assert "macro_auc_roc" in saved["metrics"] and saved["metrics"]["n_test_examples"] > 0
    assert result["metrics"]["n_test_examples"] == saved["metrics"]["n_test_examples"]


def test_eval_checkpoint_refuses_the_validation_split_flag_and_a_missing_file(synthetic_data, tmp_path):
    data_root, partition_root = synthetic_data
    module = _load("eval_checkpoint")
    with pytest.raises(SystemExit):
        module.main(["--checkpoint", str(tmp_path / "nope.pt"), "--data-root", str(data_root), "--partition-root", str(partition_root),
                     "--output", str(tmp_path / "o.json")])
    with pytest.raises(SystemExit):
        module.main(["--checkpoint", str(_saved(tmp_path)), "--data-root", str(data_root), "--partition-root", str(partition_root),
                     "--output", str(tmp_path / "o.json"), "--split", "val"])
