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
    assert "NOT identical" in caveats and "run-to-run variation" in caveats and "hypothesis" in caveats
    assert "report-only" in _record()["acceptance_against_docs_06"]["5_side_by_side"]


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


def test_run_to_run_variation_is_recomputed_from_the_three_cache_runs_and_the_png_run():
    record = _record()
    variation = record["run_to_run_variation"]
    runs = variation["runs"]
    assert set(runs) == {"run1", "rep2", "rep3"} and all(len(rows) == 5 for rows in runs.values())
    inside = below = above = 0
    for key in KEYS:
        spread = 0.0
        for i, row in enumerate(record["rounds"]):
            values = [runs[name][i][key] for name in runs]
            lo, hi = min(values), max(values)
            spread = max(spread, hi - lo)
            png = row["png"][key]
            inside += lo <= png <= hi
            below += png < lo
            above += png > hi
        assert abs(spread - variation["max_spread_across_cache_runs"][key]) < 1e-9, key
    position = variation["png_position_against_range_of_cache_runs"]
    assert (position["inside"], position["below"], position["above"]) == (inside, below, above)
    assert position["comparisons"] == inside + below + above == 20
    # the first cache run is the one recorded in the side-by-side
    for i, row in enumerate(record["rounds"]):
        assert all(runs["run1"][i][key] == row["uint8_cache"][key] for key in KEYS)
    # "of the same size": no cache-vs-png maximum exceeds 1.5 times the largest spread among the cache runs themselves
    for key in KEYS:
        assert record["max_abs_difference_cache_vs_png"][key] <= 1.5 * variation["max_spread_across_cache_runs"][key], key
