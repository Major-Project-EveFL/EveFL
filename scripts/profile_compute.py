"""
Compute-only micro-benchmark of ResNet-18 on synthetic tensors: fp32 vs channels_last vs AMP vs both, at 224 and 128 px.

No data loading and no federated learning are involved, so the numbers are the model's own throughput on the device this
process runs on. They answer one question: how much do AMP and channels_last buy before they are wired into training?
Whether either changes what is learned is a different question; this script only reports, for each variant, the largest
difference of the output logits from the fp32 model on the same weights and inputs (inference, one batch).

    python scripts/profile_compute.py --batch-size 32 --image-sizes 224,128 --iters 30 --output /kaggle/working/profile_compute.json

Variants
    fp32               the current training path
    channels_last      model and input in NHWC memory format, fp32
    amp                torch.autocast (float16 on CUDA with a GradScaler for training, bfloat16 on CPU), NCHW
    amp_channels_last  both

Every rate is images per second, measured after `--warmup` iterations with a device synchronise around the timed loop.
Peak memory is `torch.cuda.max_memory_allocated` (MiB) for the variant's training loop, or null on CPU.
These are measurements of the machine that ran the script; read them with `device` in the output.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from evefl import environment  # noqa: E402
from evefl.fl.model import NUM_CLASSES, build_resnet18  # noqa: E402
from evefl.profiling import device_info  # noqa: E402

VARIANTS: Dict[str, Dict[str, bool]] = {
    "fp32": {"amp": False, "channels_last": False},
    "channels_last": {"amp": False, "channels_last": True},
    "amp": {"amp": True, "channels_last": False},
    "amp_channels_last": {"amp": True, "channels_last": True},
}


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _autocast_dtype(device: torch.device) -> torch.dtype:
    return torch.float16 if device.type == "cuda" else torch.bfloat16


def _prepare(model: torch.nn.Module, x: torch.Tensor, channels_last: bool):
    if channels_last:
        return model.to(memory_format=torch.channels_last), x.contiguous(memory_format=torch.channels_last)
    return model, x


def measure_variant(name: str, *, batch_size: int, image_size: int, warmup: int, iters: int, device: torch.device,
                    seed: int = 0) -> Dict[str, Any]:
    """Forward-only and training images/s of one variant on synthetic inputs (fresh model, fixed seed)."""
    cfg = VARIANTS[name]
    torch.manual_seed(seed)
    model = build_resnet18(pretrained=False).to(device)
    x = torch.randn(batch_size, 3, image_size, image_size, device=device)
    y = (torch.rand(batch_size, NUM_CLASSES, device=device) > 0.7).float()
    model, x = _prepare(model, x, cfg["channels_last"])
    amp = cfg["amp"]
    autocast = lambda: torch.autocast(device_type=device.type, dtype=_autocast_dtype(device), enabled=amp)  # noqa: E731

    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            with autocast():
                model(x)
        _sync(device)
        started = time.perf_counter()
        for _ in range(iters):
            with autocast():
                model(x)
        _sync(device)
        forward_s = time.perf_counter() - started

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()
    use_scaler = amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    def step() -> None:
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            loss = criterion(model(x).float(), y)
        if use_scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        step()
    _sync(device)
    started = time.perf_counter()
    for _ in range(iters):
        step()
    _sync(device)
    train_s = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated() / 2 ** 20 if device.type == "cuda" else None
    return {
        "variant": name, **cfg, "autocast_dtype": str(_autocast_dtype(device)).replace("torch.", "") if amp else None,
        "image_size": image_size, "batch_size": batch_size, "iters": iters,
        "forward_images_per_s": batch_size * iters / forward_s,
        "train_images_per_s": batch_size * iters / train_s,
        "peak_memory_mib_train": peak,
    }


def logit_difference_vs_fp32(name: str, *, batch_size: int, image_size: int, device: torch.device, seed: int = 0) -> Dict[str, float]:
    """Largest and mean absolute logit difference of a variant from the fp32 model (same weights, same inputs, eval mode)."""
    torch.manual_seed(seed)
    model = build_resnet18(pretrained=False).to(device).eval()
    x = torch.randn(batch_size, 3, image_size, image_size, device=device)
    with torch.no_grad():
        reference = model(x).float()
        cfg = VARIANTS[name]
        m, xi = _prepare(model, x, cfg["channels_last"])
        with torch.autocast(device_type=device.type, dtype=_autocast_dtype(device), enabled=cfg["amp"]):
            out = m(xi).float()
        model.to(memory_format=torch.contiguous_format)
    diff = (out - reference).abs()
    return {"max_abs": float(diff.max()), "mean_abs": float(diff.mean()), "reference_abs_mean": float(reference.abs().mean())}


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    p = argparse.ArgumentParser(description="Compute-only throughput of fp32, channels_last and AMP on synthetic tensors.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--image-sizes", type=str, default="224,128", help="comma-separated square sizes in pixels")
    p.add_argument("--variants", type=str, default=",".join(VARIANTS), help="comma-separated subset of: " + ", ".join(VARIANTS))
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, default=Path("profile_compute.json"))
    args = p.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in names if v not in VARIANTS]
    if unknown:
        p.error(f"unknown variant(s) {unknown}; choose from {list(VARIANTS)}")
    sizes = [int(s) for s in args.image_sizes.split(",") if s.strip()]

    started = time.perf_counter()
    rows: List[Dict[str, Any]] = []
    for size in sizes:
        for name in names:
            row = measure_variant(name, batch_size=args.batch_size, image_size=size, warmup=args.warmup, iters=args.iters,
                                  device=device, seed=args.seed)
            row["logit_difference_vs_fp32"] = logit_difference_vs_fp32(name, batch_size=args.batch_size, image_size=size,
                                                                       device=device, seed=args.seed)
            rows.append(row)
    for size in sizes:
        base = next((r for r in rows if r["image_size"] == size and r["variant"] == "fp32"), None)
        for r in rows:
            if r["image_size"] == size and base is not None:
                r["train_speedup_vs_fp32"] = r["train_images_per_s"] / base["train_images_per_s"]
                r["forward_speedup_vs_fp32"] = r["forward_images_per_s"] / base["forward_images_per_s"]

    summary = {
        "label": "measured on the machine that ran this script, on synthetic tensors; compute only, no data loading",
        "device": device_info(), "environment": environment.runtime_environment(), "seed": args.seed,
        "batch_size": args.batch_size, "image_sizes": sizes, "results": rows,
        "elapsed_seconds": time.perf_counter() - started,
        "note": ("AMP uses float16 with a GradScaler on CUDA and bfloat16 on CPU (a CPU run only checks that the script works). "
                 "A speed-up here says nothing about accuracy: wiring AMP into training would change the arithmetic and "
                 "is read against the run-to-run variation recorded in docs/validation/image_cache_lite_run_2026-10-15.json."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"device: {device.type} | batch {args.batch_size} | python {sys.version.split()[0]} | torch {torch.__version__}")
    print(f"{'px':>4} {'variant':<18} {'fwd img/s':>10} {'train img/s':>12} {'train x':>8} {'peak MiB':>9} {'max |dlogit|':>13}")
    for r in rows:
        peak = f"{r['peak_memory_mib_train']:.0f}" if r["peak_memory_mib_train"] is not None else "n/a"
        print(f"{r['image_size']:>4} {r['variant']:<18} {r['forward_images_per_s']:>10.1f} {r['train_images_per_s']:>12.1f} "
              f"{r.get('train_speedup_vs_fp32', float('nan')):>8.2f} {peak:>9} {r['logit_difference_vs_fp32']['max_abs']:>13.4f}")
    print(f"written: {args.output}")
    return summary


if __name__ == "__main__":
    main()
