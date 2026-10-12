"""
uint8 image cache for ChestX-ray14: build, read and verify.

Why: the PNG pipeline decodes a 1024 x 1024 PNG and resizes it for every image of every epoch. On the lite Kaggle run that
kept the GPU waiting for the DataLoader 92% of the time (docs/validation/kaggle_profile_2026-10-11.json). The cache does that
work once.

What is stored (pre-registered in docs/06, "acceptance of the uint8 image cache")
    for every row of Data_Entry_2017.csv, in row order:
        torchvision.transforms.Resize(256) applied to PIL `Image.open(path).convert("RGB")`  (the PNG pipeline's own call),
        then ONE uint8 channel, kept only after checking that the three channels are equal.
    Shards of `shard_size` images: `shard_00000.npy` (uint8 [count, 256, 256]) plus a sidecar `shard_00000.json`.
    `index.json`  : format, shapes, ordered-names hash, every shard's SHA-256. `cache_sha256` = SHA-256 of its bytes.
    `meta.json`   : library versions, CSV hash, source PNG modes, command, timings (provenance, not part of the cache hash).
    `failures.json`: every image that could not be cached; if it is not empty no `index.json` is written.

Nothing here touches CUDA, so the build can run in forked or spawned worker processes.

Units: pixels are uint8 grey levels 0 to 255; sizes are in pixels; times in seconds.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import logging
import multiprocessing
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import PIL
import torch
from PIL import Image
from torchvision import transforms

log = logging.getLogger(__name__)

CACHE_FORMAT = "evefl-uint8-cache-v1"
IMAGE_SIZE = 256          # side of the stored square, pixels (Resize(256) of a square image)
CROP_SIZE = 224           # side of the model input, pixels
SHARD_SIZE = 4096         # images per shard
INDEX_NAME = "index.json"
META_NAME = "meta.json"
FAILURES_NAME = "failures.json"
MAX_LISTED_NON_L_ROWS = 5000   # cap on the per-build list of rows whose source PNG mode is not "L"
RESIZE_DESCRIPTION = ("torchvision.transforms.Resize(256) applied to PIL Image.open(path).convert('RGB'); "
                      "channel 0 stored after checking that the three channels are equal")


class CacheError(RuntimeError):
    """The cache on disk is missing, incomplete or inconsistent with the metadata it is used with."""


class CacheImageError(Exception):
    """One image cannot be cached. `reason` is a short stable code, `detail` is free text for the failures file."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


# --------------------------------------------------------------------------
# One image
# --------------------------------------------------------------------------

def convert_to_cache_array(image: Image.Image) -> np.ndarray:
    """
    The cached array of one PIL image: `Resize(256)(image.convert("RGB"))`, one channel, uint8 [256, 256].

    Raises CacheImageError("not_256x256") if the resized image is not square 256 (a non-square or unusual source), and
    CacheImageError("channels_differ") if the three channels are not equal, because one stored channel could not then
    reproduce the RGB image the PNG pipeline sees.
    """
    resized = transforms.Resize(IMAGE_SIZE)(image.convert("RGB"))
    arr = np.asarray(resized)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise CacheImageError("unexpected_array", f"shape {arr.shape}")
    if arr.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
        raise CacheImageError("not_256x256", f"resized to {arr.shape[1]}x{arr.shape[0]}")
    if not (np.array_equal(arr[..., 0], arr[..., 1]) and np.array_equal(arr[..., 0], arr[..., 2])):
        raise CacheImageError("channels_differ", "R, G and B are not equal after convert('RGB')")
    return np.ascontiguousarray(arr[..., 0])


def center_crop_view(arr: np.ndarray) -> np.ndarray:
    """The 224 x 224 centre of a cached [256, 256] array: the same offset as `transforms.CenterCrop(224)` (16 pixels)."""
    top = int(round((arr.shape[0] - CROP_SIZE) / 2.0))
    left = int(round((arr.shape[1] - CROP_SIZE) / 2.0))
    return arr[top:top + CROP_SIZE, left:left + CROP_SIZE]


def random_crop_flip_view(arr: np.ndarray) -> np.ndarray:
    """
    A random 224 x 224 crop and a random horizontal flip of a cached array.

    Consumes the torch RNG exactly as `RandomCrop(224)` followed by `RandomHorizontalFlip()` do (two `torch.randint` calls for
    the offsets, then one `torch.rand(1)` against 0.5), so with the same torch seed the cache backend and the PNG backend
    draw the same crops. tests/test_image_cache.py checks that equality.
    """
    span = arr.shape[0] - CROP_SIZE + 1
    i = int(torch.randint(0, span, size=(1,)).item())
    j = int(torch.randint(0, arr.shape[1] - CROP_SIZE + 1, size=(1,)).item())
    crop = arr[i:i + CROP_SIZE, j:j + CROP_SIZE]
    if torch.rand(1) < 0.5:
        crop = crop[:, ::-1]
    return crop


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _names_sha256(names: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def _shard_stem(shard: int) -> str:
    return f"shard_{shard:05d}"


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _build_shard(task: Dict[str, Any]) -> Dict[str, Any]:
    """Build one shard (runs in a worker process or inline); returns its sidecar record."""
    try:
        torch.set_num_threads(1)  # one image at a time per process; the parallelism is across processes
    except Exception:  # noqa: BLE001
        pass
    started = time.perf_counter()
    out_dir = Path(task["out_dir"])
    shard = int(task["shard"])
    start = int(task["start"])
    entries: List[Tuple[str, str]] = task["entries"]
    arrays = np.zeros((len(entries), IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    failures: List[Dict[str, Any]] = []
    modes: Dict[str, int] = {}
    non_l_rows: List[int] = []
    for k, (name, path) in enumerate(entries):
        row = start + k
        try:
            with Image.open(path) as image:
                mode = image.mode
                arrays[k] = convert_to_cache_array(image)
            modes[mode] = modes.get(mode, 0) + 1
            if mode != "L":
                non_l_rows.append(row)
        except CacheImageError as exc:
            failures.append({"row": row, "name": name, "reason": exc.reason, "detail": exc.detail})
        except Exception as exc:  # noqa: BLE001 - any unreadable file is a failure to report, not a crash of the build
            failures.append({"row": row, "name": name, "reason": "unreadable", "detail": repr(exc)})

    npy_path = out_dir / f"{_shard_stem(shard)}.npy"
    tmp_path = npy_path.with_name(npy_path.name + ".tmp")
    with open(tmp_path, "wb") as f:
        np.save(f, arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, npy_path)
    record = {
        "shard": shard, "start": start, "count": len(entries), "file": npy_path.name,
        "sha256": _sha256_file(npy_path), "bytes": npy_path.stat().st_size,
        "names_sha256": _names_sha256([name for name, _ in entries]),
        "failures": failures, "modes": modes, "non_l_rows": non_l_rows, "seconds": time.perf_counter() - started,
    }
    _atomic_write_bytes(out_dir / f"{_shard_stem(shard)}.json", json.dumps(record, sort_keys=True).encode("utf-8"))
    return record


def _load_finished_sidecar(out_dir: Path, shard: int, start: int, names: Sequence[str]) -> Optional[Dict[str, Any]]:
    """The sidecar of a shard that is complete and matches the plan, or None (the shard must be (re)built)."""
    sidecar = out_dir / f"{_shard_stem(shard)}.json"
    npy = out_dir / f"{_shard_stem(shard)}.npy"
    if not sidecar.exists() or not npy.exists():
        return None
    try:
        record = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    ok = (
        record.get("start") == start and record.get("count") == len(names)
        and record.get("names_sha256") == _names_sha256(names)
        and record.get("bytes") == npy.stat().st_size
        and not record.get("failures")   # a shard with failures is attempted again on the next run
    )
    return record if ok else None


@dataclass
class BuildReport:
    n_images: int
    n_shards: int
    shards_built: int
    shards_skipped: int
    failures: List[Dict[str, Any]]
    build_seconds: float
    images_per_s: float                  # over the shards built in this call (0 if every shard was skipped)
    cache_sha256: Optional[str]          # None when there are failures (no index.json is written)
    mode_counts: Dict[str, int] = field(default_factory=dict)
    non_l_rows: List[int] = field(default_factory=list)
    output_dir: str = ""

    @property
    def ok(self) -> bool:
        return not self.failures and self.cache_sha256 is not None


def build_cache(
    entries: Sequence[Tuple[str, str]],
    output_dir: str | Path,
    *,
    workers: int = 4,
    shard_size: int = SHARD_SIZE,
    csv_sha256: Optional[str] = None,
    command: str = "",
    git_commit: Optional[str] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> BuildReport:
    """
    Build (or resume) the cache for `entries`, a list of (image name, path) in metadata row order.

    Every shard is attempted even if some images fail. Failed images are written to `failures.json` and `index.json` is
    written only when there are none. A shard already complete on disk (sidecar and file agree with the plan and have no
    failures) is skipped, so a re-run after an interrupted build only does the missing work. `workers <= 1` runs inline.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.tmp"):
        stale.unlink()
    say = progress or (lambda message: None)
    started = time.perf_counter()
    names = [name for name, _ in entries]
    n = len(entries)
    plan = [(s, s * shard_size) for s in range((n + shard_size - 1) // shard_size)]

    records: Dict[int, Dict[str, Any]] = {}
    todo: List[Dict[str, Any]] = []
    for shard, start in plan:
        chunk = list(entries[start:start + shard_size])
        finished = _load_finished_sidecar(out_dir, shard, start, [name for name, _ in chunk])
        if finished is not None:
            records[shard] = finished
        else:
            todo.append({"out_dir": str(out_dir), "shard": shard, "start": start,
                         "entries": [(name, str(path)) for name, path in chunk]})
    skipped = len(records)
    say(f"{len(plan)} shard(s) of up to {shard_size} images: {skipped} already complete, {len(todo)} to build")

    built_images = 0
    work_started = time.perf_counter()

    def _done(record: Dict[str, Any]) -> None:
        nonlocal built_images
        records[record["shard"]] = record
        built_images += record["count"]
        elapsed = time.perf_counter() - work_started
        say(f"shard {record['shard']:>3} done ({record['count']} images, {len(record['failures'])} failures) "
            f"| {built_images} images in {elapsed:.0f} s = {built_images / max(elapsed, 1e-9):.1f} images/s")

    if todo:
        if workers <= 1:
            for task in todo:
                _done(_build_shard(task))
        else:
            ctx = multiprocessing.get_context("spawn")
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
                for record in pool.map(_build_shard, todo):
                    _done(record)
    build_wall = time.perf_counter() - work_started

    ordered = [records[s] for s, _ in plan]
    failures = [f for record in ordered for f in record["failures"]]
    mode_counts: Dict[str, int] = {}
    non_l: List[int] = []
    for record in ordered:
        for mode, count in record["modes"].items():
            mode_counts[mode] = mode_counts.get(mode, 0) + count
        non_l.extend(record["non_l_rows"])

    _atomic_write_bytes(out_dir / FAILURES_NAME,
                        json.dumps({"n_failures": len(failures), "failures": failures}, indent=2).encode("utf-8"))

    cache_sha256: Optional[str] = None
    if not failures:
        index = {
            "format": CACHE_FORMAT, "image_size": IMAGE_SIZE, "channels_stored": 1, "pixel_dtype": "uint8",
            "shard_size": shard_size, "n_images": n, "names_sha256": _names_sha256(names),
            "resize": RESIZE_DESCRIPTION,
            "shards": [{"file": r["file"], "start": r["start"], "count": r["count"], "bytes": r["bytes"],
                        "sha256": r["sha256"]} for r in ordered],
        }
        index_bytes = json.dumps(index, indent=2, sort_keys=True).encode("utf-8")
        _atomic_write_bytes(out_dir / INDEX_NAME, index_bytes)
        cache_sha256 = hashlib.sha256(index_bytes).hexdigest()
    elif (out_dir / INDEX_NAME).exists():
        (out_dir / INDEX_NAME).unlink()  # an index left from an earlier complete build would describe a different cache

    total_seconds = time.perf_counter() - started
    rate = (built_images / build_wall) if todo and build_wall > 0 else 0.0
    meta = {
        "format": CACHE_FORMAT, "cache_sha256": cache_sha256, "n_images": n, "n_shards": len(plan),
        "shards_built_in_this_run": len(todo), "shards_skipped_in_this_run": skipped, "n_failures": len(failures),
        "source_png_modes": mode_counts, "non_l_rows_listed": non_l[:MAX_LISTED_NON_L_ROWS],
        "non_l_rows_total": len(non_l),
        "metadata_csv_sha256": csv_sha256, "command": command, "git_commit": git_commit,
        "workers": workers, "build_seconds_this_run": total_seconds,
        "images_per_s_this_run": rate,
        "python": sys.version.split()[0], "platform": platform.platform(), "cpu_count": os.cpu_count(),
        "libraries": _library_versions(),
        "note": "provenance only; the cache identity is cache_sha256 (the hash of index.json)",
    }
    _atomic_write_bytes(out_dir / META_NAME, json.dumps(meta, indent=2).encode("utf-8"))
    return BuildReport(
        n_images=n, n_shards=len(plan), shards_built=len(todo), shards_skipped=skipped, failures=failures,
        build_seconds=total_seconds, images_per_s=rate, cache_sha256=cache_sha256,
        mode_counts=mode_counts, non_l_rows=non_l, output_dir=str(out_dir),
    )


def _library_versions() -> Dict[str, str]:
    import torchvision

    return {"numpy": np.__version__, "Pillow": PIL.__version__,
            "torch": torch.__version__, "torchvision": torchvision.__version__}


# --------------------------------------------------------------------------
# Read and verify
# --------------------------------------------------------------------------

class CacheReader:
    """
    Read-only access to a built cache. The index is read and hashed on construction (cheap); shard files are checked for
    existence and size, and memory-mapped lazily. `verify_shard_hashes()` re-reads every shard (slow, run on demand).
    """

    def __init__(self, cache_root: str | Path):
        self.root = Path(cache_root)
        index_path = self.root / INDEX_NAME
        if not index_path.exists():
            raise CacheError(f"No {INDEX_NAME} in {self.root}: the cache is missing or its build did not finish "
                             f"(see {FAILURES_NAME}).")
        index_bytes = index_path.read_bytes()
        self.cache_sha256 = hashlib.sha256(index_bytes).hexdigest()
        self.index: Dict[str, Any] = json.loads(index_bytes.decode("utf-8"))
        if self.index.get("format") != CACHE_FORMAT:
            raise CacheError(f"Unsupported cache format {self.index.get('format')!r}; expected {CACHE_FORMAT!r}.")
        if self.index.get("image_size") != IMAGE_SIZE or self.index.get("channels_stored") != 1:
            raise CacheError("Cache image geometry differs from this code (256 x 256, one channel).")
        self.n_images = int(self.index["n_images"])
        self.shard_size = int(self.index["shard_size"])
        self._shards: List[Dict[str, Any]] = list(self.index["shards"])
        for shard in self._shards:
            path = self.root / shard["file"]
            if not path.exists():
                raise CacheError(f"Cache shard missing: {path}")
            if path.stat().st_size != shard["bytes"]:
                raise CacheError(f"Cache shard {path.name} has {path.stat().st_size} bytes, expected {shard['bytes']}.")
        self._maps: Dict[int, np.ndarray] = {}

    def __getstate__(self) -> Dict[str, Any]:
        state = dict(self.__dict__)
        state["_maps"] = {}  # memory maps are re-opened in each process, never pickled
        return state

    def check_names(self, names: Sequence[str]) -> None:
        """Refuse a cache that was built for a different metadata file (different names or order)."""
        if len(names) != self.n_images or _names_sha256([str(x) for x in names]) != self.index["names_sha256"]:
            raise CacheError(
                f"The cache was built for {self.n_images} images with another name list/order than the metadata in use "
                f"({len(names)} rows); rebuild it or use the matching Data_Entry_2017.csv.")

    def _map(self, shard: int) -> np.ndarray:
        arr = self._maps.get(shard)
        if arr is None:
            arr = np.load(self.root / self._shards[shard]["file"], mmap_mode="r")
            self._maps[shard] = arr
        return arr

    def image(self, row: int) -> np.ndarray:
        """The cached [256, 256] uint8 array of metadata row `row` (a read-only view of the memory map)."""
        if not 0 <= row < self.n_images:
            raise IndexError(f"row {row} outside the cache (0..{self.n_images - 1})")
        shard, offset = divmod(row, self.shard_size)
        return self._map(shard)[offset]

    def verify_shard_hashes(self) -> List[Dict[str, str]]:
        """Re-hash every shard file; returns the mismatches ([] means every shard reproduces the index)."""
        bad = []
        for shard in self._shards:
            actual = _sha256_file(self.root / shard["file"])
            if actual != shard["sha256"]:
                bad.append({"file": shard["file"], "expected": shard["sha256"], "actual": actual})
        return bad

    def meta(self) -> Dict[str, Any]:
        path = self.root / META_NAME
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def pick_verification_rows(n_images: int, non_l_rows: Sequence[int], *, sample: int, seed: int = 0) -> List[int]:
    """Random rows (seed) plus every non-L row (a seed-sample of `sample` of them if there are more), sorted, unique."""
    rng = np.random.default_rng(seed)
    random_rows = rng.choice(n_images, size=min(sample, n_images), replace=False).tolist()
    non_l = list(non_l_rows)
    if len(non_l) > sample:
        non_l = rng.choice(np.asarray(non_l), size=sample, replace=False).tolist()
    return sorted(set(int(r) for r in random_rows) | set(int(r) for r in non_l))


def verify_against_png(
    reader: CacheReader, entries: Sequence[Tuple[str, str]], rows: Sequence[int]
) -> Dict[str, Any]:
    """
    Compare the cache with the PNG pipeline on `rows`: (a) the stored array equals channel 0 of
    `Resize(256)(img.convert("RGB"))`; (b) the cache evaluation pipeline output equals `get_eval_transform()`'s, bit for bit.
    """
    from evefl.fl.model import get_eval_transform, prepare_batch

    eval_transform = get_eval_transform()
    cpu = torch.device("cpu")
    stored_mismatch: List[int] = []
    pipeline_mismatch: List[int] = []
    for row in rows:
        name, path = entries[row]
        with Image.open(path) as image:
            rgb = image.convert("RGB")
        expected_stored = np.asarray(transforms.Resize(IMAGE_SIZE)(rgb))[..., 0]
        cached = reader.image(row)
        if not np.array_equal(cached, expected_stored):
            stored_mismatch.append(row)
        png_tensor = eval_transform(rgb)
        cache_tensor = prepare_batch(torch.from_numpy(np.array(center_crop_view(cached))).view(1, 1, CROP_SIZE, CROP_SIZE),
                                     cpu)[0]
        if not torch.equal(png_tensor, cache_tensor):
            pipeline_mismatch.append(row)
    return {
        "n_checked": len(list(rows)), "stored_array_mismatches": stored_mismatch,
        "eval_pipeline_mismatches": pipeline_mismatch,
        "all_equal": not stored_mismatch and not pipeline_mismatch,
    }


# --------------------------------------------------------------------------
# Locating the source PNGs
# --------------------------------------------------------------------------

def resolve_entries(
    data_root: str | Path, names: Sequence[str], *, index_cache: Optional[str | Path] = None
) -> Tuple[List[Tuple[str, str]], float]:
    """
    (name, path) for every image name, in order, and the seconds spent scanning the folder tree (0.0 if no scan was needed).

    A name found in `<data_root>/images/` is used directly. If any is missing, the tree is scanned once with `rglob("*.png")`
    (the layout of NIH's Kaggle mirror: about 5 minutes for 112,120 files there, measured by scripts/profile_loading.py)
    and, when `index_cache` is given, the scan result is stored there and reused on the next call.
    A name that cannot be found anywhere raises FileNotFoundError listing the first few.
    """
    root = Path(data_root)
    direct = root / "images"
    found: Dict[str, str] = {}
    missing: List[str] = []
    for name in names:
        candidate = direct / name
        if candidate.exists():
            found[name] = str(candidate)
        else:
            missing.append(name)
    seconds = 0.0
    if missing:
        scanned: Dict[str, str] = {}
        cache_path = Path(index_cache) if index_cache is not None else None
        if cache_path is not None and cache_path.exists():
            scanned = json.loads(cache_path.read_text(encoding="utf-8"))
            scanned = {k: v for k, v in scanned.items() if Path(v).exists()}
        if any(name not in scanned for name in missing):
            started = time.perf_counter()
            scanned = {p.name: str(p) for p in root.rglob("*.png")}
            seconds = time.perf_counter() - started
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                _atomic_write_bytes(cache_path, json.dumps(scanned).encode("utf-8"))
        still_missing = [name for name in missing if name not in scanned]
        if still_missing:
            raise FileNotFoundError(f"{len(still_missing)} image(s) not found under {root}; first: {still_missing[:5]}")
        for name in missing:
            found[name] = scanned[name]
    return [(name, found[name]) for name in names], seconds
