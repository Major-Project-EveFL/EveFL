"""
Profiling instrumentation (evefl/profiling.py): time per phase (data loading, compute, evaluation), device and GPU
utilisation samples. These are measurements about the software and hardware, not about the model: they never enter the
config hash and never change what is trained.
"""

import json
import time

import numpy as np
import pytest

from evefl import profiling
from evefl.profiling import (
    GpuSampler,
    LoaderMeter,
    begin_phase,
    device_info,
    end_phase,
    parse_nvidia_smi,
    summarize_gpu,
    summarize_rounds,
    timed_iter,
)


class SlowLoader:
    """Yields `n` batches, sleeping `delay` seconds before each one (a loader that is slow to produce data)."""

    def __init__(self, n, delay, batch_size=4):
        self.n, self.delay, self.batch_size = n, delay, batch_size

    def __iter__(self):
        for _ in range(self.n):
            time.sleep(self.delay)
            yield np.zeros((self.batch_size, 3)), np.zeros((self.batch_size, 14))


# --------------------------------------------------------------------------
# Loader wait time and phase accounting
# --------------------------------------------------------------------------

def test_timed_iter_counts_batches_and_items_and_measures_the_wait_inside_the_loader():
    meter = LoaderMeter()
    batches = list(timed_iter(SlowLoader(5, 0.02, batch_size=4), meter))
    assert len(batches) == 5 and meter.n_batches == 5 and meter.n_items == 20
    assert 0.08 <= meter.data_wait_s <= 0.5


def test_time_spent_by_the_consumer_is_not_counted_as_data_wait():
    meter = LoaderMeter()
    for _ in timed_iter(SlowLoader(4, 0.0), meter):
        time.sleep(0.03)  # the consumer works; the loader is instant
    assert meter.data_wait_s < 0.05 and meter.n_batches == 4


def test_phase_splits_wall_time_into_wait_and_compute():
    meter = LoaderMeter()
    t0 = begin_phase()
    for _ in timed_iter(SlowLoader(4, 0.02), meter):
        time.sleep(0.01)
    phase = end_phase(meter, t0)
    assert phase["phase_s"] >= phase["data_wait_s"] > 0.0
    assert phase["compute_s"] == pytest.approx(phase["phase_s"] - phase["data_wait_s"])
    assert phase["n_images"] == 16 and phase["n_batches"] == 4
    assert phase["images_per_s"] == pytest.approx(16 / phase["phase_s"])
    assert 0.0 < phase["data_wait_share"] < 1.0
    json.dumps(phase)


def test_empty_phase_has_no_division_by_zero():
    phase = end_phase(LoaderMeter(), begin_phase())
    assert phase["n_images"] == 0 and phase["images_per_s"] == 0.0 and phase["data_wait_share"] == 0.0


# --------------------------------------------------------------------------
# Device information and GPU samples
# --------------------------------------------------------------------------

def test_device_info_is_json_ready_and_says_whether_cuda_is_used():
    info = device_info()
    assert {"cuda_available", "device_count", "devices", "torch_cuda_version", "cpu_count", "torch_num_threads",
            "default_device"} <= set(info)
    assert info["default_device"] in ("cuda", "cpu") and isinstance(info["devices"], list)
    json.dumps(info)


NVIDIA_SMI = "0, 87, 4321\n1, 3, 15\n"


def test_nvidia_smi_csv_is_parsed_per_gpu():
    assert parse_nvidia_smi(NVIDIA_SMI) == [
        {"index": 0, "util_pct": 87.0, "mem_mib": 4321.0}, {"index": 1, "util_pct": 3.0, "mem_mib": 15.0}]
    assert parse_nvidia_smi("") == [] and parse_nvidia_smi("garbage line\n") == []


def test_sampler_without_nvidia_smi_is_unavailable_not_an_error():
    def missing(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    sampler = GpuSampler(interval_s=0.01, runner=missing)
    sampler.start()
    time.sleep(0.05)
    sampler.stop()
    assert sampler.available is False and sampler.samples == []
    assert sampler.to_dict()["available"] is False


def test_sampler_collects_timestamped_samples_from_the_runner():
    calls = []

    def fake(*args, **kwargs):
        calls.append(1)
        return NVIDIA_SMI

    sampler = GpuSampler(interval_s=0.01, runner=fake)
    sampler.start()
    time.sleep(0.12)
    sampler.stop()
    assert sampler.available is True and len(sampler.samples) >= 3
    t, gpus = sampler.samples[0]
    assert t >= 0.0 and gpus[0]["util_pct"] == 87.0
    stamps = [s[0] for s in sampler.samples]
    assert stamps == sorted(stamps)


def test_gpu_summary_per_window_uses_only_the_samples_inside_it():
    samples = [(0.5, [{"index": 0, "util_pct": 10.0, "mem_mib": 1.0}]),
               (1.5, [{"index": 0, "util_pct": 20.0, "mem_mib": 1.0}]),
               (2.5, [{"index": 0, "util_pct": 90.0, "mem_mib": 1.0}])]
    out = summarize_gpu(samples, {"early": (0.0, 2.0), "late": (2.0, 3.0), "none": (5.0, 6.0)})
    assert out["early"]["n_samples"] == 2 and out["early"]["mean_util_pct"] == {"0": 15.0}
    assert out["late"]["mean_util_pct"] == {"0": 90.0}
    assert out["none"]["n_samples"] == 0 and out["none"]["mean_util_pct"] == {}


# --------------------------------------------------------------------------
# Whole-run summary and the 50 percent rule
# --------------------------------------------------------------------------

def _round(fit_wait, fit_compute, fed=None, i=1):
    phases = {"fit": {"phase_s": fit_wait + fit_compute, "data_wait_s": fit_wait, "compute_s": fit_compute,
                      "n_images": 100, "t_start": 0.0, "t_end": fit_wait + fit_compute},
              "server_eval_test": {"phase_s": 2.0, "data_wait_s": 1.5, "compute_s": 0.5, "n_images": 50,
                                   "t_start": 10.0, "t_end": 12.0}}
    if fed:
        phases["federated_eval"] = fed
    return {"round": i, "timing": {"phases": phases, "round_s": 20.0}}


def test_summary_computes_the_data_wait_share_over_the_loaded_phases_only():
    logs = [_round(8.0, 2.0)]
    s = summarize_rounds(logs)
    # loaded phases: fit (10 s, 8 wait) and server_eval_test (2 s, 1.5 wait) -> 9.5 / 12
    assert s["data_wait_share_loaded"] == pytest.approx(9.5 / 12.0)
    assert s["rule_50_percent"]["threshold"] == 0.5 and s["rule_50_percent"]["data_wait_exceeds_threshold"] is True
    assert s["phase_seconds"]["fit"] == pytest.approx(10.0)
    assert s["images_per_s"]["fit"] == pytest.approx(10.0)


def test_summary_below_threshold_and_rounds_without_timing_are_tolerated():
    s = summarize_rounds([_round(1.0, 9.0), {"round": 2}])
    assert s["rule_50_percent"]["data_wait_exceeds_threshold"] is False and s["n_rounds_with_timing"] == 1
    assert summarize_rounds([])["data_wait_share_loaded"] is None


def test_rule_states_that_it_needs_the_microbenchmark_and_the_device():
    rule = summarize_rounds([_round(8.0, 2.0)])["rule_50_percent"]
    assert "micro-benchmark" in rule["note"] and "gpu" in rule["note"].lower()


def test_module_has_no_side_effects_on_import():
    assert hasattr(profiling, "LoaderMeter") and not profiling.__dict__.get("_sampler_running", False)
