"""Consistency of docs/validation/image_cache_lite_run_2026-10-15.json (cache vs png lite run): typo guard only."""

import json
from pathlib import Path

VALIDATION = Path(__file__).resolve().parents[1] / "docs" / "validation"
RECORD = VALIDATION / "image_cache_lite_run_2026-10-15.json"
BUILD = VALIDATION / "image_cache_build_2026-10-12.json"
KEYS = ("val_loss", "val_auc_excl_thin", "test_loss", "test_auc_excl_thin")


def _record():
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_record_is_a_labelled_pipeline_check_and_does_not_claim_the_runs_are_identical():
    caveats = " ".join(_record()["caveats"])
    assert "Pipeline check" in caveats and "Neither run is a baseline" in caveats
    assert "NOT identical" in caveats and "not attributed" in caveats
    assert _record()["acceptance_against_docs_06"]["5_side_by_side"].startswith("reported above; report-only")


def test_cache_identity_matches_the_build_record_and_the_config_hashes_differ():
    record = _record()
    build = json.loads(BUILD.read_text(encoding="utf-8"))
    assert record["runs"]["uint8_cache"]["data_backend"]["cache_sha256"] == build["full_build"]["cache_sha256"]
    assert record["runs"]["png"]["config_hash"] != record["runs"]["uint8_cache"]["config_hash"]
    assert record["runs"]["png"]["data_backend"]["cache_sha256"] is None


def test_round_two_onward_share_follows_from_the_per_round_timings_and_is_below_the_threshold():
    record = _record()
    wait = wall = 0.0
    for row in record["rounds"]:
        if row["round"] < 2:
            continue
        c = row["uint8_cache"]
        wait += c["fit"]["data_wait_s"]
        wall += c["fit"]["phase_s"] + c["server_eval_test_phase_s"] + c["server_eval_val_phase_s"]
    # the per-phase wait for the two evaluations is not stored; the fit phase alone dominates (95% of the loaded time)
    assert wait / wall < 0.5
    stated = record["runs"]["uint8_cache"]["data_wait_share_from_round_2"]
    assert stated < 0.5 and record["acceptance_against_docs_06"]["4_data_wait_share_from_round_2_below_0_5"] is True
    assert abs(wait / wall - stated) < 0.03


def test_differences_and_their_maxima_are_recomputed_from_the_two_runs():
    record = _record()
    for row in record["rounds"]:
        for key in KEYS:
            diff = row["uint8_cache"][key] - row["png"][key]
            assert abs(diff - row["difference_cache_minus_png"][key]) < 1.5e-4, (row["round"], key)
    for key in KEYS:
        biggest = max(abs(row["difference_cache_minus_png"][key]) for row in record["rounds"])
        assert abs(biggest - record["max_abs_difference_cache_vs_png"][key]) < 1e-9
    assert any(record["max_abs_difference_cache_vs_png"][k] > 0 for k in KEYS)  # not claimed to be identical


def test_speed_summary_matches_the_elapsed_times():
    speed = _record()["speed"]
    runs = _record()["runs"]
    assert speed["elapsed_seconds_png"] == runs["png"]["elapsed_seconds"]
    assert speed["elapsed_seconds_cache"] == runs["uint8_cache"]["elapsed_seconds"]
    assert speed["elapsed_seconds_png"] / speed["elapsed_seconds_cache"] > 10
    assert speed["fit_phase_seconds_round_2_cache"] < speed["fit_phase_seconds_round_2_png"] / 5
