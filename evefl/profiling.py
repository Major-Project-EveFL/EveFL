"""
Profiling instrumentation: where the wall time of a run goes (data loading, compute, evaluation) and what hardware did it.

These are measurements about the software and the machine, not about the model. They are written under `experiment.profile`
and `rounds[i].timing` of the results JSON, they never enter the config hash, and they never change what is trained.

Method
- A phase is timed with `time.perf_counter()` around the whole phase. The time the consumer spent BLOCKED waiting for the
  next batch from the DataLoader is measured separately (`timed_iter`); compute is the rest of the phase (host-to-device copy,
  forward, backward, optimiser, metric computation). On CUDA the phase ends with one `torch.cuda.synchronize()`, so
  asynchronous kernels are counted in the phase that launched them without a synchronisation per batch.
- GPU utilisation is sampled by a background thread that calls `nvidia-smi` every `interval_s` seconds. Samples are
  timestamped on the same clock as the phase windows, so utilisation can be reported per phase. If `nvidia-smi` is missing
  the sampler reports `available: false`; that, together with `device_info()`, is how a CPU-only run is told apart from a GPU
  run (a data-loading diagnosis is meaningless if the run was on a CPU).

Units: seconds, images per second, percent (0 to 100), MiB.
"""

from __future__ import annotations

import os
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

DATA_WAIT_THRESHOLD = 0.5   # pre-registered rule: loading is the bottleneck if data wait >= 50% of the loaded phases
NVIDIA_SMI_QUERY = ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader,nounits"]
LOADED_PHASES = ("fit", "server_eval_test", "server_eval_val", "federated_eval")


# --------------------------------------------------------------------------
# Loader wait time and phases
# --------------------------------------------------------------------------

@dataclass
class LoaderMeter:
    """Accumulates the time the consumer spent blocked on the loader, and how many batches and items it delivered."""

    data_wait_s: float = 0.0
    n_batches: int = 0
    n_items: int = 0


def _batch_size(batch: Any) -> int:
    try:
        return int(len(batch[0]))
    except (TypeError, IndexError, KeyError):
        return 0


def timed_iter(iterable: Iterable[Any], meter: LoaderMeter) -> Iterator[Any]:
    """Yield the batches of `iterable`, adding the time spent inside the loader's `next()` to `meter.data_wait_s`."""
    iterator = iter(iterable)
    while True:
        started = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            meter.data_wait_s += time.perf_counter() - started
            return
        meter.data_wait_s += time.perf_counter() - started
        meter.n_batches += 1
        meter.n_items += _batch_size(batch)
        yield batch


def begin_phase() -> float:
    return time.perf_counter()


def _cuda_synchronize() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:  # noqa: BLE001 - profiling must never break a run
        pass


def end_phase(meter: LoaderMeter, started: float, *, synchronize: bool = True) -> Dict[str, float]:
    """Close a phase: wall time, loader wait, compute (the rest), counts and throughput."""
    if synchronize:
        _cuda_synchronize()
    phase_s = time.perf_counter() - started
    wait = min(meter.data_wait_s, phase_s)
    compute = max(phase_s - wait, 0.0)
    return {
        "phase_s": phase_s,
        "data_wait_s": wait,
        "compute_s": compute,
        "n_batches": meter.n_batches,
        "n_images": meter.n_items,
        "images_per_s": (meter.n_items / phase_s) if phase_s > 0 and meter.n_items else 0.0,
        "data_wait_share": (wait / phase_s) if phase_s > 0 and meter.n_items else 0.0,
    }


# --------------------------------------------------------------------------
# Device and GPU samples
# --------------------------------------------------------------------------

def device_info() -> Dict[str, Any]:
    """Which device the code will use and what the machine has (JSON-ready; never raises)."""
    info: Dict[str, Any] = {
        "cuda_available": False, "device_count": 0, "devices": [], "torch_cuda_version": None, "cudnn_version": None,
        "cpu_count": os.cpu_count(), "torch_num_threads": None, "default_device": "cpu",
    }
    try:
        import torch

        info["torch_num_threads"] = torch.get_num_threads()
        info["torch_cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["default_device"] = "cuda"
            info["device_count"] = torch.cuda.device_count()
            info["cudnn_version"] = torch.backends.cudnn.version()
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                info["devices"].append({
                    "index": i, "name": props.name, "capability": f"{props.major}.{props.minor}",
                    "total_memory_mib": round(props.total_memory / 2 ** 20),
                })
    except Exception as exc:  # noqa: BLE001
        info["error"] = repr(exc)
    return info


def parse_nvidia_smi(text: str) -> List[Dict[str, float]]:
    """Parse `index, utilization.gpu, memory.used` CSV lines (no header, no units); malformed lines are skipped."""
    gpus: List[Dict[str, float]] = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            gpus.append({"index": int(parts[0]), "util_pct": float(parts[1]), "mem_mib": float(parts[2])})
        except ValueError:
            continue
    return gpus


def _run_nvidia_smi(*_: Any, **__: Any) -> str:
    return subprocess.run(NVIDIA_SMI_QUERY, capture_output=True, text=True, timeout=5, check=True).stdout


class GpuSampler:
    """Background `nvidia-smi` sampler. `samples` is a list of (seconds since start(), [per-GPU dicts])."""

    def __init__(self, interval_s: float = 2.0, runner: Callable[..., str] = _run_nvidia_smi) -> None:
        self.interval_s = interval_s
        self._runner = runner
        self.samples: List[Tuple[float, List[Dict[str, float]]]] = []
        self.available = False
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._t0 = 0.0

    def start(self, clock_start: Optional[float] = None) -> None:
        self._t0 = time.perf_counter() if clock_start is None else clock_start
        self._thread = threading.Thread(target=self._loop, name="gpu-sampler", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                gpus = parse_nvidia_smi(self._runner())
            except (FileNotFoundError, subprocess.SubprocessError, OSError):
                return  # no nvidia-smi: stay unavailable, stop sampling
            if gpus:
                self.available = True
                self.samples.append((time.perf_counter() - self._t0, gpus))
            self._stop.wait(self.interval_s)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)

    def to_dict(self, max_samples: int = 2000) -> Dict[str, Any]:
        step = max(1, len(self.samples) // max_samples)
        return {
            "available": self.available, "interval_s": self.interval_s, "n_samples": len(self.samples),
            "samples": [[round(t, 3), g] for t, g in self.samples[::step]],
        }


def summarize_gpu(
    samples: Sequence[Tuple[float, List[Dict[str, float]]]], windows: Mapping[str, Tuple[float, float]]
) -> Dict[str, Dict[str, Any]]:
    """Mean utilisation and memory per GPU inside each (start, end) window, same clock as the samples."""
    out: Dict[str, Dict[str, Any]] = {}
    for name, (t0, t1) in windows.items():
        inside = [gpus for t, gpus in samples if t0 <= t < t1]
        util: Dict[str, List[float]] = {}
        mem: Dict[str, List[float]] = {}
        for gpus in inside:
            for g in gpus:
                util.setdefault(str(g["index"]), []).append(g["util_pct"])
                mem.setdefault(str(g["index"]), []).append(g["mem_mib"])
        out[name] = {
            "n_samples": len(inside),
            "mean_util_pct": {k: statistics.fmean(v) for k, v in util.items()},
            "mean_mem_mib": {k: statistics.fmean(v) for k, v in mem.items()},
        }
    return out


# --------------------------------------------------------------------------
# Whole-run summary and the pre-registered 50 percent rule
# --------------------------------------------------------------------------

def summarize_rounds(round_logs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Totals per phase over all rounds that carry a `timing` block, and the data-wait share of the loaded phases."""
    seconds: Dict[str, float] = {}
    images: Dict[str, float] = {}
    wait_loaded = 0.0
    wall_loaded = 0.0
    wait_r2 = 0.0   # rounds >= 2 only: round 1 holds the one-time path index and cold file reads (docs/06, image cache)
    wall_r2 = 0.0
    n_rounds = 0
    for log in round_logs:
        timing = log.get("timing")
        if not timing:
            continue
        n_rounds += 1
        for name, phase in (timing.get("phases") or {}).items():
            wall = float(phase.get("phase_s", phase.get("wall_s", 0.0)))
            seconds[name] = seconds.get(name, 0.0) + wall
            images[name] = images.get(name, 0.0) + float(phase.get("n_images", 0))
            if name in LOADED_PHASES and "data_wait_s" in phase:
                wait_loaded += float(phase["data_wait_s"])
                wall_loaded += wall
                if int(log.get("round", 0)) >= 2:
                    wait_r2 += float(phase["data_wait_s"])
                    wall_r2 += wall
    share = (wait_loaded / wall_loaded) if wall_loaded > 0 else None
    share_r2 = (wait_r2 / wall_r2) if wall_r2 > 0 else None
    return {
        "n_rounds_with_timing": n_rounds,
        "phase_seconds": seconds,
        "images_per_s": {k: (images[k] / seconds[k]) for k in seconds if seconds[k] > 0 and images.get(k)},
        "data_wait_share_loaded": share,
        "data_wait_share_loaded_from_round_2": share_r2,
        "rule_50_percent": {
            "threshold": DATA_WAIT_THRESHOLD,
            "data_wait_share": share,
            "data_wait_exceeds_threshold": (share >= DATA_WAIT_THRESHOLD) if share is not None else None,
            "note": ("Necessary, not sufficient: the rule also needs the micro-benchmark criterion (single-process decode "
                     "throughput below half the model's compute throughput, scripts/profile_loading.py) and must be read "
                     "together with device_info and the gpu samples: on a CPU-only run the data wait is not the diagnosis."),
        },
    }
