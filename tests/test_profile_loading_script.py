"""scripts/profile_loading.py: stage times per image, DataLoader throughput per worker count, compute ceiling, criterion."""

import importlib.util
import json
from pathlib import Path

import pytest


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "profile_loading.py"
    spec = importlib.util.spec_from_file_location("profile_loading_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    # module-scoped copy of the synthetic dataset (the shared fixture is function-scoped)
    import numpy as np
    import pandas as pd
    from PIL import Image

    from evefl.fl.model import CHESTXRAY_LABELS

    rng = np.random.default_rng(0)
    root = tmp_path_factory.mktemp("bench")
    (root / "images").mkdir()
    rows = []
    for i in range(24):
        name = f"b_{i:04d}.png"
        Image.fromarray(rng.integers(0, 256, (300, 300), dtype=np.uint8), mode="L").save(root / "images" / name)
        rows.append({"Image Index": name, "Finding Labels": CHESTXRAY_LABELS[i % 14], "Patient ID": i})
    pd.DataFrame(rows).to_csv(root / "Data_Entry_2017.csv", index=False)
    out = root / "profile.json"
    module = _load()
    summary = module.main(["--data-root", str(root), "--n-images", "16", "--workers", "0,2", "--batch-size", "4",
                           "--compute-batch-size", "4", "--compute-iters", "2", "--compute-warmup", "1",
                           "--seed", "1", "--output", str(out)])
    return summary, out


def test_output_file_is_written_and_json_ready(result):
    summary, out = result
    assert out.exists() and json.loads(out.read_text())["n_images"] == 16
    json.dumps(summary)


def test_every_decode_stage_is_timed_per_image(result):
    stages = result[0]["stage_ms_per_image"]
    assert {"read_bytes", "decode", "convert_rgb", "resize", "center_crop", "to_tensor", "normalize", "total"} <= set(stages)
    assert all(v >= 0 for v in stages.values())
    assert stages["total"] == pytest.approx(sum(v for k, v in stages.items() if k != "total"), rel=0.05)
    assert result[0]["single_process_images_per_s"] > 0


def test_dataloader_throughput_is_reported_for_each_worker_count(result):
    rows = result[0]["dataloader"]
    assert [r["num_workers"] for r in rows] == [0, 2]
    assert all(r["images_per_s"] > 0 and r["n_images"] == 16 for r in rows)


def test_compute_ceiling_has_forward_and_training_rates_and_names_the_device(result):
    compute = result[0]["compute"]
    assert compute["forward_images_per_s"] > 0 and compute["train_images_per_s"] > 0
    assert compute["device"] in ("cuda", "cpu") and compute["batch_size"] == 4


def test_criterion_compares_decode_rate_with_half_the_training_rate(result):
    c = result[0]["criterion"]
    assert c["threshold_fraction"] == 0.5
    assert c["decode_below_half_of_compute"] == (
        result[0]["single_process_images_per_s"] < 0.5 * result[0]["compute"]["train_images_per_s"])
    assert "device" in c["note"].lower()


def test_environment_and_device_are_recorded(result):
    assert result[0]["device"]["default_device"] in ("cuda", "cpu")
    assert "numpy" in result[0]["environment"]["libraries"]
    assert result[0]["dataset"]["data_root"].endswith("bench0") or result[0]["dataset"]["data_root"]
