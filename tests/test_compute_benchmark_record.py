"""Consistency of docs/validation/compute_benchmark_2026-10-15.json (AMP / channels_last / 128 px throughput): typo guard."""

import json
from pathlib import Path

RECORD = Path(__file__).resolve().parents[1] / "docs" / "validation" / "compute_benchmark_2026-10-15.json"


def _record():
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _row(size, variant):
    return next(r for r in _record()["results"] if r["image_size"] == size and r["variant"] == variant)


def test_record_is_labelled_synthetic_compute_only_and_not_an_accuracy_claim():
    record = _record()
    assert "synthetic tensors" in record["label"] and "no training run" in record["label"]
    caveats = " ".join(record["caveats"])
    assert "says nothing about what AMP does to accuracy" in caveats and "One run per cell" in caveats


def test_every_size_and_variant_is_present_once():
    pairs = [(r["image_size"], r["variant"]) for r in _record()["results"]]
    assert sorted(pairs) == sorted((s, v) for s in (224, 128) for v in ("fp32", "channels_last", "amp", "amp_channels_last"))


def test_speedups_are_recomputed_from_the_rates():
    for r in _record()["results"]:
        base = _row(r["image_size"], "fp32")
        assert abs(r["train_speedup_vs_fp32_same_size"] - r["train_images_per_s"] / base["train_images_per_s"]) < 0.006
        assert abs(r["forward_speedup_vs_fp32_same_size"] - r["forward_images_per_s"] / base["forward_images_per_s"]) < 0.006
        assert abs(r["train_speedup_vs_fp32_224"] - r["train_images_per_s"] / _row(224, "fp32")["train_images_per_s"]) < 0.006


def test_the_findings_quoted_in_docs_06_hold():
    assert _row(224, "channels_last")["train_speedup_vs_fp32_same_size"] < 1.0
    assert _row(128, "channels_last")["train_speedup_vs_fp32_same_size"] < 1.0
    assert 1.9 < _row(224, "amp")["train_speedup_vs_fp32_same_size"] < 2.1
    assert 2.6 < _row(224, "amp_channels_last")["train_speedup_vs_fp32_same_size"] < 2.7
    assert _row(224, "amp")["peak_memory_mib_train"] < _row(224, "fp32")["peak_memory_mib_train"]
    ratio = _record()["derived"]["fp32_train_ratio_128_over_224"]
    assert abs(ratio - _row(128, "fp32")["train_images_per_s"] / _row(224, "fp32")["train_images_per_s"]) < 0.006
    assert ratio < 3.06  # the pixel-count assumption of the earlier projection was optimistic


def test_amp_logit_difference_is_small_for_one_inference_batch_and_zero_for_layout_only_variants():
    for size in (224, 128):
        assert _row(size, "fp32")["max_abs_logit_difference_vs_fp32"] == 0.0
        assert _row(size, "channels_last")["max_abs_logit_difference_vs_fp32"] == 0.0
        assert 0 < _row(size, "amp")["max_abs_logit_difference_vs_fp32"] < 0.01
