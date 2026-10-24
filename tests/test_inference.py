"""
Inference for the upload page (`evefl/inference.py`) and the page itself (`dashboard/pages/7_upload_xray.py`).

A trained checkpoint (`--save-model`) is loaded only from a local folder, with `torch.load(..., weights_only=True)`; an uploaded
image goes through exactly the evaluation preprocessing the model was scored with (RGB, resize 256, centre crop 224, ImageNet
normalisation). The page is a research prototype and must say, on screen, that it is not diagnostic.
"""

import hashlib
import io
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from flwr.common import ndarrays_to_parameters
from PIL import Image

from evefl import inference as inf
from evefl.fl.checkpoint import CheckpointSaver
from evefl.fl.client import get_model_parameters
from evefl.fl.dataset import PngBackend
from evefl.fl.model import CHESTXRAY_LABELS, build_resnet18

REPO = Path(__file__).resolve().parents[1]
AUC = "macro_auc_roc_excl_thin"


def _checkpoint_dir(tmp_path, *, name="run", best_auc=0.7, seed=0):
    torch.manual_seed(seed)
    params = ndarrays_to_parameters(get_model_parameters(build_resnet18(pretrained=False)))
    saver = CheckpointSaver(tmp_path, name=name, metadata={"config_hash": "abc123", "experiment": name})
    saver.observe(7, params, (0.2, {AUC: best_auc}))
    saver.finalize(params, final_round=9)
    return tmp_path


def _png_bytes(mode="L", size=300, seed=0):
    rng = np.random.default_rng(seed)
    grey = rng.integers(0, 256, (size, size), dtype=np.uint8)
    arrays = {"L": grey, "RGB": np.stack([grey] * 3, -1), "RGBA": np.stack([grey, grey, grey, np.full_like(grey, 255)], -1)}
    buffer = io.BytesIO()
    Image.fromarray(arrays[mode], mode=mode).save(buffer, format="PNG")
    return buffer.getvalue()


# --------------------------------------------------------------------------
# finding and loading checkpoints
# --------------------------------------------------------------------------

def test_find_checkpoints_lists_best_validation_files_first_then_final_and_ignores_other_files(tmp_path):
    _checkpoint_dir(tmp_path)
    (tmp_path / "notes.txt").write_text("x")
    found = inf.find_checkpoints(tmp_path)
    assert [p.name for p in found] == ["run_best_val.pt", "run_final.pt"]
    assert inf.find_checkpoints(tmp_path / "missing") == []


def test_a_checkpoint_loads_into_an_eval_mode_resnet_with_its_round_hash_and_provenance(tmp_path):
    _checkpoint_dir(tmp_path, best_auc=0.7)
    loaded = inf.load_checkpoint(tmp_path / "run_best_val.pt")
    assert not loaded.model.training and loaded.round == 7 and loaded.val_auc_excl_thin == 0.7
    assert loaded.sha256 == hashlib.sha256((tmp_path / "run_best_val.pt").read_bytes()).hexdigest()
    assert loaded.metadata["config_hash"] == "abc123" and loaded.kind == "best_val"
    assert inf.load_checkpoint(tmp_path / "run_final.pt").kind == "final"


def test_a_file_that_is_not_one_of_our_checkpoints_is_refused(tmp_path):
    bad = tmp_path / "bad_best_val.pt"
    torch.save({"format": "something-else", "state_dict": {}}, bad)
    with pytest.raises(ValueError, match="not an EveFL checkpoint"):
        inf.load_checkpoint(bad)
    wrong = tmp_path / "wrong_best_val.pt"
    torch.save({"format": "evefl-checkpoint-v1", "round": 1, "val_auc_excl_thin": None, "state_dict": {"fc.weight": torch.zeros(1)},
                "metadata": {}}, wrong)
    with pytest.raises((ValueError, RuntimeError)):
        inf.load_checkpoint(wrong)
    notorch = tmp_path / "junk.pt"
    notorch.write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError):
        inf.load_checkpoint(notorch)


# --------------------------------------------------------------------------
# prediction
# --------------------------------------------------------------------------

def test_prediction_gives_fourteen_probabilities_sorted_and_deterministic(tmp_path):
    _checkpoint_dir(tmp_path)
    loaded = inf.load_checkpoint(tmp_path / "run_best_val.pt")
    table = inf.predict(loaded, Image.open(io.BytesIO(_png_bytes())))
    assert list(table.columns) == ["finding", "probability"] and len(table) == 14
    assert set(table["finding"]) == set(CHESTXRAY_LABELS)
    assert table["probability"].between(0.0, 1.0).all() and table["probability"].is_monotonic_decreasing
    again = inf.predict(loaded, Image.open(io.BytesIO(_png_bytes())))
    assert table.equals(again)


def test_grey_rgb_and_rgba_versions_of_the_same_picture_give_the_same_prediction(tmp_path):
    _checkpoint_dir(tmp_path)
    loaded = inf.load_checkpoint(tmp_path / "run_best_val.pt")
    tables = [inf.predict(loaded, Image.open(io.BytesIO(_png_bytes(mode)))).set_index("finding")["probability"] for mode in ("L", "RGB", "RGBA")]
    assert np.allclose(tables[0], tables[1], atol=1e-6) and np.allclose(tables[0], tables[2].reindex(tables[0].index), atol=1e-6)


def test_prediction_uses_exactly_the_evaluation_preprocessing_of_training_time(synthetic_data, tmp_path):
    data_root, _ = synthetic_data
    _checkpoint_dir(tmp_path)
    loaded = inf.load_checkpoint(tmp_path / "run_best_val.pt")
    dataset = PngBackend().make_dataset(data_root, None, train=False)
    image_path = data_root / "images" / "synthetic_0000.png"
    with torch.no_grad():
        expected = torch.sigmoid(loaded.model(dataset[0][0].unsqueeze(0)))[0].numpy()
    got = inf.predict(loaded, Image.open(image_path)).set_index("finding")["probability"].reindex(CHESTXRAY_LABELS).to_numpy()
    assert np.allclose(got, expected, atol=1e-6)


def test_oversized_or_unreadable_uploads_are_refused_with_a_clear_message():
    with pytest.raises(ValueError, match="could not read"):
        inf.open_upload(b"definitely not an image")
    with pytest.raises(ValueError, match="too large"):
        inf.open_upload(b"x" * (inf.MAX_UPLOAD_BYTES + 1))
    assert inf.open_upload(_png_bytes()).size == (300, 300)


def test_the_module_never_loads_pickled_objects_without_the_weights_only_guard():
    source = Path(inf.__file__).read_text(encoding="utf-8")
    assert "weights_only=True" in source and "pickle" not in source


# --------------------------------------------------------------------------
# the page
# --------------------------------------------------------------------------

def _page(env):
    from streamlit.testing.v1 import AppTest

    for key, value in env.items():
        os.environ[key] = str(value)
    try:
        return AppTest.from_file(str(REPO / "dashboard" / "pages" / "7_upload_xray.py"), default_timeout=120).run()
    finally:
        for key in env:
            os.environ.pop(key, None)


def test_the_page_without_a_checkpoint_explains_what_is_missing_and_does_not_crash(tmp_path):
    at = _page({"EVEFL_CHECKPOINT_DIR": tmp_path / "none", "EVEFL_DEMO_SAMPLES": tmp_path / "nosamples"})
    assert not at.exception, [e.value for e in at.exception]
    shown = " ".join(str(x.value) for x in list(at.warning) + list(at.info) + list(at.markdown))
    assert "research prototype" in shown and "not diagnostic" in shown and "no checkpoint" in shown.lower()


def test_the_page_with_a_checkpoint_and_samples_shows_the_banner_provenance_and_a_prediction(tmp_path):
    ck = _checkpoint_dir(tmp_path / "ck", name="seed1")
    samples = tmp_path / "samples"
    samples.mkdir()
    (samples / "a.png").write_bytes(_png_bytes(seed=1))
    (samples / "samples.json").write_text('{"samples": [{"file": "a.png", "labels": ["Effusion"], "split": "test"}]}', encoding="utf-8")
    at = _page({"EVEFL_CHECKPOINT_DIR": ck, "EVEFL_DEMO_SAMPLES": samples})
    assert not at.exception, [e.value for e in at.exception]
    shown = " ".join(str(x.value) for x in list(at.warning) + list(at.info) + list(at.caption) + list(at.markdown))
    assert "research prototype" in shown and "not diagnostic" in shown
    assert "round 7" in shown and "abc123" in shown          # which model is being shown
    assert len(at.dataframe) >= 1                            # the default sample is predicted straight away
