"""
Micro-benchmark: is image loading the bottleneck? No federated learning involved.

For a random sample of N images it measures
  1. the time of every stage of the current pipeline, one image at a time in this process: read the file, decode the PNG,
     convert to RGB, Resize(256), CenterCrop(224), ToTensor, Normalize (the stages of `get_eval_transform`);
  2. DataLoader throughput for each worker count (images per second, first batch latency);
  3. the compute ceiling: images per second of ResNet-18 on SYNTHETIC tensors, forward only and forward + backward + AdamW,
     on the device this process would use;
  4. the pre-registered criterion (docs/06): loading is a candidate bottleneck if the single-process decode rate is below
     half of the training compute rate. The other half of the rule (data wait >= 50% of the loaded phases) comes from the
     profile block of a real run's results JSON.

Order matters on purpose: the DataLoader worker measurements run BEFORE anything touches CUDA, because forking workers
after CUDA initialisation is the deadlock described in CLAUDE.md (environment gotchas).

    python scripts/profile_loading.py --data-root /kaggle/input/datasets/nih-chest-xrays/data \\
        --n-images 300 --workers 0,2,4 --output /kaggle/working/profile_loading.json

The first pass over the files may be cold (the page cache is empty); the script reports the stage times from a first and a
second pass over the same images so the file-system effect is visible.
"""

from __future__ import annotations

import argparse
import io
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from torchvision import transforms  # noqa: E402

from evefl import environment  # noqa: E402
from evefl.fl.dataset import ChestXray14Dataset, load_metadata  # noqa: E402
from evefl.fl.model import IMAGENET_MEAN, IMAGENET_STD, build_resnet18, get_eval_transform  # noqa: E402
from evefl.profiling import GpuSampler, device_info  # noqa: E402

STAGES = ["read_bytes", "decode", "convert_rgb", "resize", "center_crop", "to_tensor", "normalize"]


def _stage_times(path: Path) -> Dict[str, float]:
    """Seconds spent in each stage for one image, using the same operations as `get_eval_transform`."""
    t = {}
    t0 = time.perf_counter()
    data = path.read_bytes()
    t["read_bytes"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    image = Image.open(io.BytesIO(data))
    image.load()
    t["decode"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    image = image.convert("RGB")
    t["convert_rgb"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    image = transforms.Resize(256)(image)
    t["resize"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    image = transforms.CenterCrop(224)(image)
    t["center_crop"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    tensor = transforms.ToTensor()(image)
    t["to_tensor"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)(tensor)
    t["normalize"] = time.perf_counter() - t0
    return t


def _mean_ms(rows: List[Dict[str, float]]) -> Dict[str, float]:
    out = {k: 1000.0 * statistics.fmean(r[k] for r in rows) for k in STAGES}
    out["total"] = sum(out[k] for k in STAGES)
    return out


def _dataloader_rows(data_root: Path, indices: np.ndarray, batch_size: int, workers: List[int]) -> List[Dict[str, Any]]:
    rows = []
    for w in workers:
        dataset = ChestXray14Dataset(data_root, indices=indices, transform=get_eval_transform())
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=w)
        started = time.perf_counter()
        first_batch_s: Optional[float] = None
        n = 0
        for images, _targets in loader:
            if first_batch_s is None:
                first_batch_s = time.perf_counter() - started
            n += int(images.shape[0])
        total = time.perf_counter() - started
        rows.append({"num_workers": w, "n_images": n, "seconds": total, "images_per_s": n / total,
                     "first_batch_s": first_batch_s})
    return rows


def _compute_ceiling(batch_size: int, warmup: int, iters: int) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_resnet18(pretrained=False).to(device)
    x = torch.randn(batch_size, 3, 224, 224, device=device)
    y = (torch.rand(batch_size, 14, device=device) > 0.7).float()

    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize()

    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        sync()
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        sync()
        fwd = time.perf_counter() - t0

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()

    def step() -> None:
        optimizer.zero_grad()
        criterion(model(x), y).backward()
        optimizer.step()

    for _ in range(warmup):
        step()
    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    sync()
    train = time.perf_counter() - t0
    return {
        "device": device.type, "batch_size": batch_size, "iters": iters,
        "forward_images_per_s": batch_size * iters / fwd, "train_images_per_s": batch_size * iters / train,
        "note": "synthetic 224x224 tensors on one device, fp32, no data loading",
    }


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    p = argparse.ArgumentParser(description="Micro-benchmark of image loading and the compute ceiling.")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--n-images", type=int, default=300)
    p.add_argument("--workers", type=str, default="0,2,4", help="comma-separated DataLoader worker counts")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--compute-batch-size", type=int, default=32)
    p.add_argument("--compute-iters", type=int, default=10)
    p.add_argument("--compute-warmup", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, default=Path("profile_loading.json"))
    args = p.parse_args(argv)

    started = time.perf_counter()
    meta = load_metadata(args.data_root)
    rng = np.random.default_rng(args.seed)
    indices = np.sort(rng.choice(len(meta), size=min(args.n_images, len(meta)), replace=False))

    # dataset construction and the one-time path index (the first lookup miss builds it)
    t0 = time.perf_counter()
    probe = ChestXray14Dataset(args.data_root, indices=indices, transform=None)
    construct_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    paths = [probe._resolve_image_path(name) for name in probe._image_names]
    resolve_s = time.perf_counter() - t0

    sampler = GpuSampler(interval_s=1.0)
    sampler.start()
    first = [_stage_times(path) for path in paths]
    second = [_stage_times(path) for path in paths]

    workers = [int(w) for w in args.workers.split(",") if w.strip() != ""]
    loader_rows = _dataloader_rows(args.data_root, indices, args.batch_size, workers)

    # CUDA is touched only here, after every DataLoader worker has been forked/spawned and finished.
    compute = _compute_ceiling(args.compute_batch_size, args.compute_warmup, args.compute_iters)
    sampler.stop()

    stage_ms = _mean_ms(second)
    single_rate = 1000.0 / stage_ms["total"] if stage_ms["total"] > 0 else 0.0
    summary: Dict[str, Any] = {
        "label": "measured on the machine that ran this script; see device and environment",
        "n_images": int(len(indices)),
        "seed": args.seed,
        "dataset": {"data_root": str(args.data_root), "n_rows": int(len(meta)), "construct_seconds": construct_s,
                    "path_resolution_seconds_for_sample": resolve_s},
        "device": device_info(),
        "environment": environment.runtime_environment(),
        "stage_ms_per_image": stage_ms,
        "stage_ms_per_image_first_pass": _mean_ms(first),
        "single_process_images_per_s": single_rate,
        "dataloader": loader_rows,
        "compute": compute,
        "gpu_samples": sampler.to_dict(max_samples=200),
        "criterion": {
            "threshold_fraction": 0.5,
            "decode_below_half_of_compute": bool(single_rate < 0.5 * compute["train_images_per_s"]),
            "single_process_images_per_s": single_rate,
            "train_images_per_s": compute["train_images_per_s"],
            "note": ("One half of the pre-registered rule (docs/06). The other half is the data-wait share of a real run "
                     "(experiment.profile.summary). Read both with the device above: on a CPU-only run the compute "
                     "ceiling is the CPU and this comparison says nothing about a GPU."),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"device: {compute['device']} | images sampled: {len(indices)} | python {sys.version.split()[0]}")
    print("per-image stage times, second pass (ms): " + ", ".join(f"{k} {v:.2f}" for k, v in stage_ms.items()))
    print(f"single-process pipeline: {single_rate:.1f} images/s")
    for r in loader_rows:
        print(f"DataLoader num_workers={r['num_workers']}: {r['images_per_s']:.1f} images/s "
              f"(first batch {r['first_batch_s']:.2f} s)")
    print(f"compute ceiling (synthetic, batch {compute['batch_size']}): forward {compute['forward_images_per_s']:.1f}, "
          f"train {compute['train_images_per_s']:.1f} images/s on {compute['device']}")
    print(f"criterion: decode below half of training compute: {summary['criterion']['decode_below_half_of_compute']}")
    print(f"written: {args.output}")
    return summary


if __name__ == "__main__":
    main()
