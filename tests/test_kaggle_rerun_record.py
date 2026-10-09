"""Consistency of the committed Kaggle rerun record (docs/validation/kaggle_rerun_2026-10-04.json): typo guard only."""

import json
import re
from pathlib import Path

RECORD = Path(__file__).resolve().parents[1] / "docs" / "validation" / "kaggle_rerun_2026-10-04.json"


def _record():
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_index_hashes_are_five_sha256_values():
    hashes = _record()["partition_full_rerun"]["index_sha256"]
    assert set(hashes) == {"hospital_0", "hospital_1", "hospital_2", "test", "val"}
    assert all(re.fullmatch(r"[0-9a-f]{64}", h) for h in hashes.values())
    assert len(set(hashes.values())) == 5


def test_weights_hash_matches_the_value_in_the_runbook():
    runbook = (RECORD.parents[1] / "KAGGLE_RUNBOOK.md").read_text(encoding="utf-8")
    assert _record()["weights"]["sha256"] in runbook


def test_lite_run_is_five_secure_rounds_with_zero_qber_and_is_labelled_not_a_baseline():
    run = _record()["lite_baseline_alpha0"]
    assert [r["round"] for r in run["rounds"]] == [1, 2, 3, 4, 5]
    assert all(r["state"] == "SECURE" and r["system_qber"] == 0.0 for r in run["rounds"])
    assert run["state_counts"] == {"SECURE": 5} and run["failures"] == 0 and run["rounds_discarded"] == 0
    caveats = " ".join(_record()["caveats"])
    assert "not a baseline" in caveats and "must not be quoted" in caveats


def test_zero_qber_explanation_matches_the_formula_inputs():
    setting = _record()["lite_baseline_alpha0"]["setting"]
    assert setting["intercept_probability"] == 0.0 and setting["bit_flip_probability"] == 0.0
