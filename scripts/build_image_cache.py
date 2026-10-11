"""
Build or verify the uint8 image cache (evefl/fl/imagecache.py; acceptance rule in docs/06).

Build (resumable: a re-run only does the shards that are not complete):

    python scripts/build_image_cache.py --data-root /kaggle/input/datasets/nih-chest-xrays/data \\
        --output /kaggle/working/evefl_cache_u8_256 --workers 4

Trial on the first 2000 rows, with a pixel check on 200 images, to measure the rate before the full build:

    python scripts/build_image_cache.py --data-root ... --output /kaggle/working/cache_trial --workers 4 \\
        --limit 2000 --verify-sample 200

Verify an existing cache (re-hash every shard, compare a sample with the PNG pipeline):

    python scripts/build_image_cache.py --data-root ... --verify /kaggle/working/evefl_cache_u8_256 --verify-sample 1000

Exit code 0: built (or verified) and nothing failed. Exit code 1: at least one image failed (see failures.json in the cache
folder; nothing is excluded silently and no index.json is written), or a verification check failed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evefl.fl.dataset import load_metadata  # noqa: E402
from evefl.fl.imagecache import (  # noqa: E402
    FAILURES_NAME,
    META_NAME,
    CacheReader,
    build_cache,
    pick_verification_rows,
    resolve_entries,
    verify_against_png,
)


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             timeout=10)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _csv_sha256(data_root: Path) -> str:
    digest = hashlib.sha256()
    with open(data_root / "Data_Entry_2017.csv", "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.iterdir() if p.is_file())


def _run_verification(cache_dir: Path, data_root: Path, sample: int, seed: int, entries=None) -> Dict[str, Any]:
    reader = CacheReader(cache_dir)
    df = load_metadata(data_root)
    names = [str(x) for x in df["Image Index"].to_numpy()]
    reader.check_names(names[: reader.n_images])  # a trial cache covers the first rows only
    bad_hashes = reader.verify_shard_hashes()
    meta = reader.meta()
    non_l = meta.get("non_l_rows_listed", [])
    result: Dict[str, Any] = {"shard_hash_mismatches": bad_hashes, "cache_sha256": reader.cache_sha256}
    if sample > 0:
        if entries is None:
            entries, _ = resolve_entries(data_root, names[: reader.n_images],
                                         index_cache=cache_dir / "path_index.json")
        rows = pick_verification_rows(reader.n_images, non_l, sample=sample, seed=seed)
        check = verify_against_png(reader, entries, rows)
        result["pixel_check"] = check
        result["non_l_rows_in_sample"] = sum(1 for r in rows if r in set(non_l))
    result["ok"] = not bad_hashes and result.get("pixel_check", {"all_equal": True})["all_equal"]
    return result


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Build or verify the uint8 image cache.")
    p.add_argument("--data-root", type=Path, required=True, help="Folder with Data_Entry_2017.csv and the PNG images")
    p.add_argument("--output", type=Path, default=None, help="Cache folder to build into")
    p.add_argument("--verify", type=Path, default=None, help="Verify this existing cache folder instead of building")
    p.add_argument("--workers", type=int, default=4, help="Build processes (default 4; 1 runs in-process)")
    p.add_argument("--shard-size", type=int, default=4096)
    p.add_argument("--limit", type=int, default=None, help="Only the first N metadata rows (a trial build)")
    p.add_argument("--verify-sample", type=int, default=0,
                   help="Compare this many random rows (plus every non-L row, capped) with the PNG pipeline")
    p.add_argument("--seed", type=int, default=0, help="Seed of the verification sample")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    if args.verify is not None:
        result = _run_verification(args.verify, args.data_root, args.verify_sample, args.seed)
        print(json.dumps({k: v for k, v in result.items() if k != "pixel_check"}, indent=2))
        if "pixel_check" in result:
            chk = result["pixel_check"]
            print(f"pixel check: {chk['n_checked']} rows compared with the PNG pipeline | stored-array mismatches "
                  f"{len(chk['stored_array_mismatches'])} | eval-pipeline mismatches {len(chk['eval_pipeline_mismatches'])}")
        print("VERIFY:", "OK" if result["ok"] else "FAILED")
        return 0 if result["ok"] else 1

    if args.output is None:
        p.error("--output is required to build (or use --verify DIR)")

    df = load_metadata(args.data_root)
    names = [str(x) for x in df["Image Index"].to_numpy()]
    if args.limit is not None:
        names = names[: args.limit]
    print(f"metadata rows: {len(df)} | caching {len(names)} | workers {args.workers} | shard size {args.shard_size}",
          flush=True)

    entries, scan_seconds = resolve_entries(args.data_root, names, index_cache=args.output / "path_index.json")
    print(f"path resolution: {scan_seconds:.1f} s of folder scanning (0 = direct paths or a reused path_index.json)",
          flush=True)

    command = "python " + " ".join(shlex.quote(a) for a in sys.argv)
    report = build_cache(
        entries, args.output, workers=args.workers, shard_size=args.shard_size, csv_sha256=_csv_sha256(args.data_root),
        command=command, git_commit=_git_commit(), progress=lambda m: print(m, flush=True),
    )

    size = _dir_bytes(args.output)
    print("-" * 78)
    print(f"images {report.n_images} | shards {report.n_shards} (built {report.shards_built}, "
          f"already complete {report.shards_skipped})")
    print(f"build time {report.build_seconds:.0f} s | {report.images_per_s:.1f} images/s in this run | "
          f"folder size {size / 1e9:.2f} GB")
    print(f"source PNG modes: {report.mode_counts} | rows with a non-L mode: {len(report.non_l_rows)}")
    if args.limit is not None and report.images_per_s > 0:
        full = 112120 / report.images_per_s
        print(f"projection (not a measurement of the full build): 112,120 images at this rate = {full / 60:.0f} min, "
              f"plus the one-time path scan above")
    if report.failures:
        print(f"FAILED: {len(report.failures)} image(s) could not be cached; first 10:")
        for f in report.failures[:10]:
            print(f"  row {f['row']} {f['name']}: {f['reason']} {f['detail']}")
        print(f"full list: {args.output / FAILURES_NAME} ; no index.json was written. Nothing was excluded or replaced.")
        return 1
    print(f"cache_sha256: {report.cache_sha256}")
    print(f"meta: {args.output / META_NAME}")

    if args.verify_sample > 0:
        result = _run_verification(args.output, args.data_root, args.verify_sample, args.seed, entries=entries)
        chk = result["pixel_check"]
        print(f"pixel check: {chk['n_checked']} rows ({result['non_l_rows_in_sample']} with a non-L source mode) | "
              f"stored-array mismatches {len(chk['stored_array_mismatches'])} | "
              f"eval-pipeline mismatches {len(chk['eval_pipeline_mismatches'])}")
        print("VERIFY:", "OK" if result["ok"] else "FAILED")
        return 0 if result["ok"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
