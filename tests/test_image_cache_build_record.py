"""Consistency of the committed image-cache build record (docs/validation/image_cache_build_2026-10-12.json): typo guard only."""

import json
import math
import re
from pathlib import Path

RECORD = Path(__file__).resolve().parents[1] / "docs" / "validation" / "image_cache_build_2026-10-12.json"
IMAGE_BYTES = 256 * 256  # one uint8 channel per image


def _record():
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_hashes_are_sha256_values_and_the_trial_is_a_different_cache():
    record = _record()
    values = [record["metadata_csv_sha256"], record["trial"]["cache_sha256"], record["full_build"]["cache_sha256"],
              record["verification"]["cache_sha256"]]
    assert all(re.fullmatch(r"[0-9a-f]{64}", v) for v in values)
    assert record["verification"]["cache_sha256"] == record["full_build"]["cache_sha256"]
    assert record["trial"]["cache_sha256"] != record["full_build"]["cache_sha256"]


def test_full_build_counts_are_consistent():
    build = _record()["full_build"]
    assert build["n_images"] == 112120 == sum(build["source_png_modes"].values())
    assert build["n_shards"] == math.ceil(build["n_images"] / build["shard_size"]) == 28
    assert build["failures"] == 0
    assert build["rows_with_non_l_source"] == build["source_png_modes"]["RGBA"] == 519
    assert abs(build["n_images"] / build["build_seconds"] - build["images_per_s"]) < 0.1


def test_folder_size_matches_one_byte_per_pixel_plus_small_files():
    build = _record()["full_build"]
    pixel_bytes = build["n_images"] * IMAGE_BYTES
    assert pixel_bytes == 7_347_896_320
    assert pixel_bytes / 1e9 < build["folder_bytes_decimal_gb"] < (pixel_bytes + 5e7) / 1e9


def test_verification_sample_includes_every_non_l_row_and_found_no_mismatch():
    record = _record()
    chk = record["verification"]["pixel_check"]
    non_l = record["full_build"]["rows_with_non_l_source"]
    assert chk["non_l_rows_included"] == non_l
    # 1000 random rows plus all non-L rows, minus the rows drawn by both
    assert chk["random_rows_requested"] <= chk["rows_compared"] <= chk["random_rows_requested"] + non_l
    assert chk["rows_compared"] >= non_l
    assert chk["stored_array_mismatches"] == chk["eval_pipeline_mismatches"] == 0
    assert record["verification"]["shard_hash_mismatches"] == 0


def test_acceptance_items_four_and_five_are_not_claimed_and_the_session_caveat_is_present():
    record = _record()
    acceptance = record["acceptance_against_docs_06"]
    assert acceptance["1_zero_failures"] is True
    assert acceptance["2_pixel_equality_on_sample_and_all_non_l"] is True
    assert acceptance["3_shard_hashes_reproduce_index"] is True
    assert str(acceptance["4_data_wait_share_from_round_2_below_0_5"]).startswith("not yet measured")
    assert str(acceptance["5_side_by_side_lite_metrics"]).startswith("not yet measured")
    caveats = " ".join(record["caveats"])
    assert "same notebook session" in caveats and "after upload" in caveats


def test_trial_pixel_check_found_no_mismatch_and_ran_one_process():
    trial = _record()["trial"]
    assert trial["n_shards"] == 1 and trial["failures"] == 0
    assert trial["pixel_check"]["stored_array_mismatches"] == trial["pixel_check"]["eval_pipeline_mismatches"] == 0
    assert "one-process rate" in " ".join(_record()["caveats"])
