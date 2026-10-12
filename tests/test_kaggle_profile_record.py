"""Consistency of the committed profiling record (docs/validation/kaggle_profile_2026-10-11.json): typo guard only.

The record holds counts and timings of a 5% pipeline check. These tests keep its numbers consistent with each other and
keep any AUC or loss value out of it; they do not re-measure anything.
"""

import json
import re
from pathlib import Path

RECORD = Path(__file__).resolve().parents[1] / "docs" / "validation" / "kaggle_profile_2026-10-11.json"
DROP_LAST_BATCH = 32


def _record():
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _walk_keys(node):
    if isinstance(node, dict):
        for key, value in node.items():
            yield key
            yield from _walk_keys(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk_keys(item)


def test_record_is_labelled_a_pipeline_check_and_holds_no_auc_or_loss_values():
    record = _record()
    text = " ".join([record["label"], *record["caveats"]])
    assert "Pipeline check" in text and "not a baseline" in text and "no AUC and no loss" in text
    forbidden = [k for k in _walk_keys(record) if {"auc", "loss", "roc"} & set(re.split(r"[^a-z0-9]+", k.lower()))]
    assert forbidden == []


def test_source_file_hashes_are_sha256_values():
    sources = _record()["source_files"]
    assert set(sources) == {"lite_alpha0_seed0_profile.json", "profile_loading.json"}
    assert all(re.fullmatch(r"[0-9a-f]{64}", v["sha256"]) and v["bytes"] > 0 for v in sources.values())


def test_run_is_five_secure_rounds_on_gpu_with_federated_evaluate_off():
    record = _record()
    run = record["run"]
    assert (run["num_rounds"], run["num_clients"], run["local_epochs"], run["batch_size"]) == (5, 3, 1, 32)
    assert run["state_counts"] == {"SECURE": 5} and run["failures"] == 0 and run["rounds_discarded"] == 0
    assert run["evaluation"]["federated_evaluate"] is False
    assert run["dataloader"]["num_workers"] == 0
    assert record["device"]["default_device"] == "cuda" and record["device"]["device_count"] == 2
    assert [r["round"] for r in record["rounds"]] == [1, 2, 3, 4, 5]


def test_fit_image_count_is_the_drop_last_total_of_the_three_hospitals():
    record = _record()
    sizes = record["partition_lite"]["hospital_images"]
    expected = sum((n // DROP_LAST_BATCH) * DROP_LAST_BATCH for n in sizes)
    for rd in record["rounds"]:
        assert rd["phases"]["fit"]["n_images"] == expected == 4256
        assert rd["phases"]["server_eval_test"]["n_images"] == record["partition_lite"]["test_images"]
        assert rd["phases"]["server_eval_val"]["n_images"] == record["partition_lite"]["val_images"]


def test_phase_times_are_internally_consistent():
    for rd in _record()["rounds"]:
        for name, ph in rd["phases"].items():
            # phase_s is the wall time of the whole phase; wait and compute are the loop times, so the phase exceeds their
            # sum by setup (model build, dataset construction) and metric code: about a second, never the other way round.
            gap = ph["phase_s"] - ph["data_wait_s"] - ph["compute_s"]
            assert -0.06 <= gap <= 0.01 * ph["phase_s"] + 1.0, (rd["round"], name, gap)


def test_data_wait_share_and_the_pre_registered_rule_follow_from_the_per_round_numbers():
    record = _record()
    wait = sum(ph["data_wait_s"] for rd in record["rounds"] for ph in rd["phases"].values())
    wall = sum(ph["phase_s"] for rd in record["rounds"] for ph in rd["phases"].values())
    share = record["summary"]["data_wait_share_loaded"]
    assert abs(wait / wall - share) < 0.002
    rule = record["summary"]["rule_50_percent"]
    assert rule["threshold"] == 0.5 and rule["data_wait_exceeds_threshold"] is (share >= 0.5) is True
    pre = record["pre_registered_rule"]
    assert pre["run_part_data_wait_ge_0_5"] and pre["micro_benchmark_part"] and pre["device_condition_cuda"]


def test_micro_benchmark_criterion_matches_its_own_rates():
    bench = _record()["micro_benchmark"]
    single = bench["single_process_images_per_s"]
    train = bench["compute_ceiling_synthetic"]["train_images_per_s"]
    assert bench["criterion_decode_below_half_of_compute"] is (single < 0.5 * train) is True
    stages = bench["stage_ms_per_image_second_pass"]
    assert abs(sum(v for k, v in stages.items() if k != "total") - stages["total"]) < 0.05
    assert abs(1000.0 / stages["total"] - single) < 0.5
    rates = bench["dataloader_images_per_s"]
    assert rates["0"] < rates["2"] < rates["4"] < train


def test_derived_rates_are_the_ratios_of_the_round_2_numbers():
    record = _record()
    fit = record["rounds"][1]["phases"]["fit"]
    derived = record["derived_round_2"]
    assert abs(derived["fit_compute_only_images_per_s"] - fit["n_images"] / fit["compute_s"]) < 0.2
    assert abs(derived["fit_delivered_images_per_s"] - fit["n_images"] / fit["phase_s"]) < 0.2
    assert derived["fit_compute_only_images_per_s"] > 5 * derived["fit_delivered_images_per_s"]
