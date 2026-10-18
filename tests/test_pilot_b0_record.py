"""Consistency of docs/validation/pilot_b0_full_e1_seed0_2026-10-17.json (the single-seed B0 pilot): typo guard only."""

import json
import re
from pathlib import Path

VALIDATION = Path(__file__).resolve().parents[1] / "docs" / "validation"
RECORD = VALIDATION / "pilot_b0_full_e1_seed0_2026-10-17.json"
BUILD = VALIDATION / "image_cache_build_2026-10-12.json"
DOC06 = Path(__file__).resolve().parents[1] / "docs" / "06_EXPERIMENT_PROTOCOL.md"


def _record():
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_record_is_labelled_a_single_seed_pilot_with_e_equal_one():
    record = _record()
    assert "single-seed pilot" in record["label"]
    caveats = " ".join(record["caveats"])
    assert "single-seed pilot, E = 1" in caveats and "not the paper baseline" in caveats and "no confidence interval" in caveats
    assert record["run"]["local_epochs"] == 1 and record["run"]["seed"] == 0 and record["run"]["num_rounds"] == 50


def test_settings_equal_the_pre_registration_and_the_recorded_hashes():
    run = _record()["run"]
    build = json.loads(BUILD.read_text(encoding="utf-8"))
    assert run["cache_sha256"] == build["full_build"]["cache_sha256"]
    assert re.fullmatch(r"[0-9a-f]{64}", run["weights_sha256"]) and run["weights_sha256"].startswith("9803814b")
    assert run["image_backend"] == "uint8_cache" and run["arithmetic"] == "fp32" and run["partition_seed"] == 42
    assert run["evaluation"] == {"federated_evaluate": False, "test_eval_every": 5, "val_eval_every": 1}
    assert run["state_counts"] == {"SECURE": 50} and run["failures"] == 0
    text = DOC06.read_text(encoding="utf-8")
    assert "Pre-registered: first full-data baseline pilot" in text and "9803814b" in text


def test_all_pre_registered_checks_passed_before_the_result_was_read():
    checks = _record()["checks_before_reading_any_auc"]
    assert checks and all(checks.values())


def test_fifty_rounds_and_test_evaluated_at_every_fifth_round_only():
    rounds = _record()["rounds"]
    assert [r["round"] for r in rounds] == list(range(1, 51))
    assert [r["round"] for r in rounds if "test_auc_excl_thin" in r] == list(range(5, 51, 5))
    assert all(0.5 < r["val_auc_excl_thin"] < 1.0 and 0.0 < r["val_loss"] < 1.0 for r in rounds)


def test_primary_and_secondary_results_follow_the_pre_registered_rules():
    record = _record()
    rounds = record["rounds"]
    final = rounds[-1]
    primary = record["results"]["primary"]
    assert primary["round"] == 50 and primary["test_auc_excl_thin"] == final["test_auc_excl_thin"] == 0.8086
    tested = [r for r in rounds if "test_auc_excl_thin" in r]
    chosen = max(tested, key=lambda r: (r["val_auc_excl_thin"], r["round"]))
    secondary = record["results"]["secondary"]
    assert secondary["round"] == chosen["round"] == 25
    assert secondary["test_auc_excl_thin"] == chosen["test_auc_excl_thin"]
    assert secondary["val_auc_excl_thin"] == chosen["val_auc_excl_thin"]


def test_observations_are_recomputed_from_the_rounds_and_stay_a_hypothesis():
    record = _record()
    rounds = record["rounds"]
    best = max(rounds, key=lambda r: (r["val_auc_excl_thin"], r["round"]))
    obs = record["observations"]
    assert obs["best_validation_auc_round"] == best["round"] == 26
    assert abs(obs["validation_auc_drop_peak_to_final"] - (best["val_auc_excl_thin"] - rounds[-1]["val_auc_excl_thin"])) < 1e-9
    low = min(rounds, key=lambda r: (r["val_loss"], -r["round"]))
    assert obs["min_validation_loss_round"] == low["round"] and obs["final_validation_loss"] - obs["min_validation_loss"] > 0.01
    assert "hypothesis to test, not a result" in obs["reading"]


def test_per_class_auc_has_the_fourteen_pathologies_and_their_mean_is_the_primary_result():
    results = _record()["results"]
    per_class = results["per_class_test_auc_round_50"]
    assert len(per_class) == 14 and results["thin_classes_in_test_at_round_50"] == []
    assert abs(sum(per_class.values()) / 14 - results["primary"]["test_auc_excl_thin"]) < 0.002   # per-class values are rounded to 3 places


def test_timing_ratio_and_the_projection_statement():
    timing = _record()["timing"]
    assert abs(timing["elapsed_seconds"] / timing["projected_seconds"] - timing["measured_over_projected"]) < 0.001
    assert 1.1 < timing["measured_over_projected"] < 1.3
    assert timing["data_wait_share_from_round_2"] < 0.05 and timing["gpu0_mean_util_pct_all_samples"] > 90
    assert "low by roughly this factor" in timing["reading"]
