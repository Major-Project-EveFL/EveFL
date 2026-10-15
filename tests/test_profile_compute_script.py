"""scripts/profile_compute.py: fp32 / channels_last / AMP throughput on synthetic tensors (CPU run checks the script only)."""

import importlib.util
import json
import math
from pathlib import Path

import pytest


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "profile_compute.py"
    spec = importlib.util.spec_from_file_location("profile_compute_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    out = tmp_path_factory.mktemp("compute") / "profile_compute.json"
    summary = _load().main(["--batch-size", "2", "--image-sizes", "64,32", "--warmup", "1", "--iters", "1",
                            "--output", str(out)])
    return summary, out


def test_every_variant_is_measured_at_every_size_and_the_file_is_written(result):
    summary, out = result
    assert json.loads(out.read_text())["batch_size"] == 2
    pairs = {(r["image_size"], r["variant"]) for r in summary["results"]}
    assert pairs == {(s, v) for s in (64, 32) for v in ("fp32", "channels_last", "amp", "amp_channels_last")}
    json.dumps(summary)


def test_rates_are_positive_and_speedups_are_relative_to_fp32(result):
    for r in result[0]["results"]:
        assert r["forward_images_per_s"] > 0 and r["train_images_per_s"] > 0
        assert r["train_speedup_vs_fp32"] > 0 and r["forward_speedup_vs_fp32"] > 0
        if r["variant"] == "fp32":
            assert r["train_speedup_vs_fp32"] == pytest.approx(1.0)
            assert r["amp"] is False and r["channels_last"] is False and r["autocast_dtype"] is None


def test_variant_flags_match_their_names(result):
    flags = {r["variant"]: (r["amp"], r["channels_last"]) for r in result[0]["results"]}
    assert flags == {"fp32": (False, False), "channels_last": (False, True), "amp": (True, False),
                     "amp_channels_last": (True, True)}


def test_logit_difference_is_reported_finite_and_zero_for_the_layout_only_variant(result):
    for r in result[0]["results"]:
        d = r["logit_difference_vs_fp32"]
        assert all(math.isfinite(d[k]) for k in ("max_abs", "mean_abs", "reference_abs_mean"))
        if r["variant"] == "fp32":
            assert d["max_abs"] == 0.0
        if r["variant"] == "channels_last":
            assert d["max_abs"] < 1e-3   # same arithmetic, different memory layout


def test_device_and_environment_and_the_not_an_accuracy_statement_are_recorded(result):
    summary = result[0]
    assert summary["device"]["default_device"] in ("cuda", "cpu")
    assert "torch" in summary["environment"]["libraries"]
    assert "says nothing about accuracy" in summary["note"]


def test_unknown_variant_is_refused(tmp_path):
    with pytest.raises(SystemExit):
        _load().main(["--variants", "int4", "--output", str(tmp_path / "x.json")])
