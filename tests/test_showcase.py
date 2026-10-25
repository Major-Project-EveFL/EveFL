"""`evefl/showcase.py`: the numbers the dashboard's home page shows are read from the recorded files, never typed into the page."""

import json
from pathlib import Path

from evefl import showcase

ROOT = Path(__file__).resolve().parents[1]
V = ROOT / "docs" / "validation"


def test_headline_numbers_come_from_the_recorded_files():
    h = showcase.headline_numbers()
    seed1 = json.loads((V / "pilot_b0_full_e1_seed1_2026-10-24.json").read_text(encoding="utf-8"))
    seed0 = json.loads((V / "pilot_b0_full_e1_seed0_2026-10-17.json").read_text(encoding="utf-8"))
    assert h["served_model_test_auc"] == seed1["demonstration_model"]["test_evaluation"]["test_auc_excl_thin"]
    assert h["served_model_round"] == seed1["demonstration_model"]["round"] == 27
    assert [s["seed"] for s in h["runs"]] == [0, 1]
    assert h["runs"][0]["test_auc_round_50"] == seed0["results"]["primary"]["test_auc_excl_thin"]
    assert h["runs"][1]["test_auc_round_50"] == seed1["results"]["primary"]["test_auc_excl_thin"]
    assert h["n_hospitals"] == 3 and h["lockdown_bound"] == 0.11 and h["partition_overlaps_all_zero"] is True


def test_every_headline_number_carries_its_label_and_a_source_file():
    h = showcase.headline_numbers()
    for key in ("served_model_test_auc", "runs", "lockdown_bound", "partition_overlaps_all_zero"):
        assert h["labels"][key] in ("measured", "simulated", "real code"), key
    for path in h["sources"].values():
        assert (ROOT / path).exists(), path


def test_a_missing_record_gives_a_clear_error_instead_of_wrong_numbers(tmp_path):
    try:
        showcase.headline_numbers(tmp_path)
    except FileNotFoundError as exc:
        assert "pilot_b0_full_e1_seed1" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError")
