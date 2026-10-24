"""Consistency of docs/validation/pilot_b0_full_e1_seed1_2026-10-24.json (the second single-seed run with checkpoints): typo guard only."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "docs" / "validation"
RECORD = VALIDATION / "pilot_b0_full_e1_seed1_2026-10-24.json"
PILOT = VALIDATION / "pilot_b0_full_e1_seed0_2026-10-17.json"
DOC06 = ROOT / "docs" / "06_EXPERIMENT_PROTOCOL.md"


def _record():
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_record_is_labelled_a_single_seed_run_and_makes_no_significance_claim():
    record = _record()
    assert "single-seed run" in record["label"] and "measured" in record["label"]
    caveats = " ".join(record["caveats"])
    assert "no confidence interval" in caveats and "not the paper baseline" in caveats and "differs from the CI environment" in caveats
    assert record["run"]["seed"] == 1 and record["run"]["local_epochs"] == 1 and record["run"]["num_rounds"] == 50
    assert "no significance statement" in record["side_by_side_with_pilot"]["note"]


def test_settings_equal_the_pilot_apart_from_the_seed_and_the_preregistration_exists():
    run, pilot = _record()["run"], json.loads(PILOT.read_text(encoding="utf-8"))["run"]
    for key in ("weights_sha256", "cache_sha256", "image_backend", "arithmetic", "partition_seed", "num_rounds", "num_clients",
                "local_epochs", "batch_size", "image_size", "evaluation", "optimizer", "policy", "intercept_probability"):
        assert run[key] == pilot[key], key
    assert run["config_hash"] != pilot["config_hash"] and run["state_counts"] == {"SECURE": 50} and run["failures"] == 0
    assert "Pre-registered: second single-seed run with checkpoints" in DOC06.read_text(encoding="utf-8")


def test_all_pre_registered_checks_passed_before_the_result_was_read():
    checks = _record()["checks_before_reading_any_auc"]
    assert all(v is True for v in checks.values() if isinstance(v, bool)) and checks["val_auc_star_round_50_above_round_1"]


def test_fifty_rounds_and_test_evaluated_at_every_fifth_round_only():
    rounds = _record()["rounds"]
    assert [r["round"] for r in rounds] == list(range(1, 51))
    assert [r["round"] for r in rounds if "test_auc_excl_thin" in r] == list(range(5, 51, 5))


def test_primary_secondary_and_the_served_checkpoint_follow_the_preregistered_rules():
    record = _record()
    rounds = record["rounds"]
    assert record["results"]["primary"]["round"] == 50
    assert record["results"]["primary"]["test_auc_excl_thin"] == rounds[-1]["test_auc_excl_thin"]
    tested = [r for r in rounds if "test_auc_excl_thin" in r]
    # 4-decimal values; the record chose with the unrounded ones, so allow either neighbour at a tie
    top = max(r["val_auc_excl_thin"] for r in tested)
    assert record["results"]["secondary"]["val_auc_excl_thin"] == top
    best = max(rounds, key=lambda r: (r["val_auc_excl_thin"], r["round"]))
    model = record["demonstration_model"]
    assert model["round"] == 27 and best["round"] in (27, 30) and model["file"].endswith("_best_val.pt")
    assert abs(model["val_auc_excl_thin"] - next(r for r in rounds if r["round"] == 27)["val_auc_excl_thin"]) < 1e-9
    assert re.fullmatch(r"[0-9a-f]{64}", model["sha256"]) and model["bytes"] > 40_000_000


def test_the_demonstration_model_test_result_is_one_evaluation_that_selects_nothing():
    ev = _record()["demonstration_model"]["test_evaluation"]
    assert "evaluated once" in ev["how"] and "selects nothing" in ev["how"] and ev["no_confidence_interval"]
    assert ev["n_test_examples"] == 10889 and ev["thin_classes"] == []
    per_class = ev["per_class_test_auc"]
    assert len(per_class) == 14 and abs(sum(per_class.values()) / 14 - ev["test_auc_excl_thin"]) < 0.002


def test_observations_are_recomputed_from_the_rounds_and_stay_a_hypothesis():
    record = _record()
    rounds = record["rounds"]
    obs = record["observations"]
    assert obs["best_validation_auc_round"] == 27
    assert abs(obs["validation_auc_drop_peak_to_final"] - (obs["best_validation_auc"] - rounds[-1]["val_auc_excl_thin"])) < 1e-3
    assert "still a hypothesis to test, not a result" in obs["reading"]


def test_side_by_side_numbers_match_the_pilot_record_and_this_one():
    record = _record()
    pilot = json.loads(PILOT.read_text(encoding="utf-8"))
    side = record["side_by_side_with_pilot"]
    assert side["primary_test_auc_round_50"] == {"seed_0_pilot": pilot["results"]["primary"]["test_auc_excl_thin"],
                                                 "seed_1": record["results"]["primary"]["test_auc_excl_thin"]}
    by_round = {r["round"]: r["test_auc_excl_thin"] for r in record["rounds"] if "test_auc_excl_thin" in r}
    pilot_by_round = {r["round"]: r["test_auc_excl_thin"] for r in pilot["rounds"] if "test_auc_excl_thin" in r}
    assert abs(side["max_abs_test_auc_difference_over_tested_rounds"] - max(abs(by_round[k] - pilot_by_round[k]) for k in by_round)) < 1e-9


def test_the_file_hash_and_timing_are_recorded():
    record = _record()
    v = record["verified_against_results_json"]
    assert re.fullmatch(r"[0-9a-f]{64}", v["sha256"]) and v["bytes"] > 1_000_000 and "line endings" in v["saved_copy_note"]
    assert abs(v["elapsed_seconds_exact"] - record["timing"]["elapsed_seconds"]) < 0.1
    assert abs(record["timing"]["elapsed_seconds"] / record["timing"]["pilot_elapsed_seconds"] - record["timing"]["measured_over_pilot"]) < 0.001
    assert "not established and is not claimed" in record["timing"]["reading"]
