"""
Copy a few public ChestX-ray14 TEST-split images, with the labels the dataset records for them, to a folder for the demonstration.

    python scripts/export_demo_samples.py --data-root /kaggle/input/datasets/nih-chest-xrays/data \\
        --partition-root /kaggle/working/partitions_full --output /kaggle/working/demo_samples --n 12 --seed 0

The images are public (NIH ChestX-ray14) and stay out of git (`demo_samples/` is ignored). The labels are the dataset's text-mined
labels, not diagnoses. The sample is seeded, drawn from the test split only (so none was used to train the demonstration model), and
about two thirds are images with at least one recorded finding. `samples.json` lists each file with its labels.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from evefl.fl.dataset import load_metadata  # noqa: E402
from evefl.fl.imagecache import resolve_entries  # noqa: E402

NOTE = ("Public NIH ChestX-ray14 test-split images copied for a demonstration and kept out of git. The labels are the dataset's "
        "recorded text-mined labels, not diagnoses. The model that scores them is a research prototype and is not diagnostic.")


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    p = argparse.ArgumentParser(description="Export a few test-split images with their recorded labels.")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--partition-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    df = load_metadata(args.data_root)
    test_rows = np.load(args.partition_root / "test" / "indices.npy")
    if not 1 <= args.n <= len(test_rows):
        p.error(f"--n must be between 1 and the test split size ({len(test_rows)})")
    labels = df["Finding Labels"].astype(str).to_numpy()
    names = df["Image Index"].astype(str).to_numpy()
    with_finding = [int(r) for r in test_rows if labels[r].strip() != "No Finding"]
    without = [int(r) for r in test_rows if labels[r].strip() == "No Finding"]

    rng = np.random.default_rng(np.random.SeedSequence([int(args.seed), 4242]))
    n_without = min(len(without), args.n // 3)
    n_with = min(len(with_finding), args.n - n_without)
    n_without = min(len(without), args.n - n_with)   # top up from the other group if one is short
    chosen = (list(rng.choice(with_finding, size=n_with, replace=False)) if n_with else []) + \
             (list(rng.choice(without, size=n_without, replace=False)) if n_without else [])
    chosen = sorted(int(r) for r in chosen)

    entries, _ = resolve_entries(args.data_root, [names[r] for r in chosen])
    args.output.mkdir(parents=True, exist_ok=True)
    samples = []
    for row, (name, path) in zip(chosen, entries):
        shutil.copy2(path, args.output / name)
        samples.append({"file": name, "labels": [x.strip() for x in labels[row].split("|") if x.strip()], "split": "test"})
    summary = {"note": NOTE, "seed": args.seed, "samples": samples}
    (args.output / "samples.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {len(samples)} images and samples.json to {args.output}")
    return summary


if __name__ == "__main__":
    main()
