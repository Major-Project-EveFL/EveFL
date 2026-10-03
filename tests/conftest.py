"""
Optional-tool gating for the test suite.

Some tests need tooling that is not available everywhere (TenSEAL on some
platforms, circom + snarkjs which need Node). Rather than failing local runs,
those tests are marked and skipped automatically when the tool is missing, so
plain `pytest` stays green locally while Kaggle / WSL / CI run the full suite.

    ckks  -> needs the `tenseal` package
    zk    -> needs `circom` and `snarkjs` on PATH

CI runs with `--require-optional-tools`, which turns those skips into FAILURES, so a
green CI run really did execute every test.

Select or exclude explicitly with e.g. `pytest -m "not zk"`.
"""

from __future__ import annotations

import importlib.util
import shutil

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from evefl.fl.dataset import partition_and_save
from evefl.fl.model import CHESTXRAY_LABELS

# Test modules that are tagged automatically (so the tests themselves stay untouched).
_MODULE_MARKERS = {
    "test_ckks": "ckks",
    "test_groth16": "zk",
}


def _missing_reason(marker: str) -> str | None:
    if marker == "ckks":
        if importlib.util.find_spec("tenseal") is None:
            return "tenseal is not installed"
    elif marker == "zk":
        missing = [tool for tool in ("circom", "snarkjs") if shutil.which(tool) is None]
        if missing:
            return f"{' and '.join(missing)} not found on PATH"
    return None


def pytest_addoption(parser):
    parser.addoption(
        "--require-optional-tools",
        action="store_true",
        default=False,
        help="Fail (instead of skip) ckks/zk tests when tenseal / circom / snarkjs are missing.",
    )


def pytest_collection_modifyitems(config, items):
    strict = config.getoption("--require-optional-tools")
    for item in items:
        marker = _MODULE_MARKERS.get(item.module.__name__.rsplit(".", 1)[-1])
        if marker is None:
            continue
        item.add_marker(getattr(pytest.mark, marker))
        reason = _missing_reason(marker)
        if not reason:
            continue
        if strict:
            item.stash_missing_tool = reason  # picked up by the autouse fixture below
        else:
            item.add_marker(pytest.mark.skip(reason=reason))


@pytest.fixture(autouse=True)
def _fail_if_required_tool_missing(request):
    reason = getattr(request.node, "stash_missing_tool", None)
    if reason:
        pytest.fail(f"required optional tool missing (--require-optional-tools): {reason}", pytrace=False)


@pytest.fixture
def synthetic_data(tmp_path):
    rng = np.random.default_rng(0)
    data_root, partition_root = tmp_path / "data", tmp_path / "partitions"
    (data_root / "images").mkdir(parents=True)
    rows = []
    for i in range(36):
        name = f"synthetic_{i:04d}.png"
        Image.fromarray(rng.integers(0, 256, (64, 64), dtype=np.uint8), mode="L").save(data_root / "images" / name)
        labels = "|".join(rng.choice(CHESTXRAY_LABELS, size=rng.integers(1, 3), replace=False))
        rows.append({"Image Index": name, "Finding Labels": labels, "Patient ID": i // 3})  # 12 patients x 3 images
    pd.DataFrame(rows).to_csv(data_root / "Data_Entry_2017.csv", index=False)
    partition_and_save(data_root, partition_root, n_clients=3, alpha=0.5, seed=0)
    return data_root, partition_root
