"""
The uint8 image cache (evefl/fl/imagecache.py, the `uint8_cache` backend, scripts/build_image_cache.py).

The pre-registered acceptance rule is in docs/06 ("acceptance of the uint8 image cache"). These tests are its CI form on
synthetic PNGs: pixel equality of the model input with the PNG backend (evaluation and, under one torch seed, training),
failure handling, resume, hashes, no CUDA in the data path, and the config-hash and results-JSON rules.
"""

import gc
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from evefl.fl import imagecache
from evefl.fl.dataset import (
    DEFAULT_IMAGE_BACKEND,
    PngBackend,
    Uint8CacheBackend,
    create_image_backend,
    get_hospital_dataloader,
    image_backend_registry,
    load_metadata,
)
from evefl.fl.imagecache import (
    CacheError,
    CacheImageError,
    CacheReader,
    build_cache,
    convert_to_cache_array,
    pick_verification_rows,
    resolve_entries,
    verify_against_png,
)
from evefl.fl.model import CHESTXRAY_LABELS, IMAGENET_MEAN, IMAGENET_STD, prepare_batch
from evefl.fl.server import _build_arg_parser, build_experiment_log, run_experiment
from evefl.profiling import summarize_rounds

SIZE = 80  # source images are 80 x 80, so Resize(256) really resamples
CPU = torch.device("cpu")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _save(path: Path, grey: np.ndarray, mode: str) -> None:
    if mode == "L":
        Image.fromarray(grey, mode="L").save(path)
    elif mode == "RGB":
        Image.fromarray(np.stack([grey] * 3, axis=-1), mode="RGB").save(path)
    elif mode == "RGBA":
        alpha = np.full_like(grey, 255)
        Image.fromarray(np.stack([grey, grey, grey, alpha], axis=-1), mode="RGBA").save(path)
    else:
        raise ValueError(mode)


def make_dataset(root: Path, modes, *, extra=None, seed=0):
    """A data_root with images/ and Data_Entry_2017.csv; `modes` lists the PNG mode of each image."""
    rng = np.random.default_rng(seed)
    (root / "images").mkdir(parents=True)
    rows = []
    for i, mode in enumerate(modes):
        name = f"img_{i:03d}.png"
        _save(root / "images" / name, rng.integers(0, 256, (SIZE, SIZE), dtype=np.uint8), mode)
        rows.append({"Image Index": name, "Finding Labels": CHESTXRAY_LABELS[i % 14], "Patient ID": i})
    for name, writer in (extra or {}).items():
        writer(root / "images" / name)
        rows.append({"Image Index": name, "Finding Labels": "No Finding", "Patient ID": len(rows)})
    pd.DataFrame(rows).to_csv(root / "Data_Entry_2017.csv", index=False)
    names = [r["Image Index"] for r in rows]
    entries = [(n, str(root / "images" / n)) for n in names]
    return root, names, entries


@pytest.fixture
def mixed(tmp_path):
    return make_dataset(tmp_path / "data", ["L", "L", "RGB", "RGBA", "L", "RGB", "L"])


def _built(tmp_path, entries, *, shard_size=3, workers=1):
    out = tmp_path / "cache"
    report = build_cache(entries, out, workers=workers, shard_size=shard_size)
    assert report.ok, report.failures
    return out, report


# --------------------------------------------------------------------------
# one image: same Resize call, one channel, and what is refused
# --------------------------------------------------------------------------

def test_cached_array_is_channel_zero_of_resize_of_the_rgb_image_and_expansion_is_exact(tmp_path):
    root, _, entries = make_dataset(tmp_path / "d", ["L"])
    with Image.open(entries[0][1]) as image:
        stored = convert_to_cache_array(image)
        rgb_first = np.asarray(transforms.Resize(256)(image.convert("RGB")))   # the PNG pipeline: RGB, then resize
        grey_first = np.asarray(transforms.Resize(256)(image))                   # one channel resized, expanded after
    assert stored.dtype == np.uint8 and stored.shape == (256, 256)
    assert np.array_equal(stored, rgb_first[..., 0])
    expanded = np.stack([stored] * 3, axis=-1)
    assert np.array_equal(expanded, rgb_first)                                  # one channel stored == RGB before resize
    assert np.array_equal(grey_first, stored)                                   # and == resizing the grey image itself


def test_non_square_unequal_channel_and_unreadable_images_are_refused_with_a_reason(tmp_path):
    rng = np.random.default_rng(1)
    with pytest.raises(CacheImageError) as err:
        convert_to_cache_array(Image.fromarray(rng.integers(0, 256, (80, 120), dtype=np.uint8), mode="L"))
    assert err.value.reason == "not_256x256"
    with pytest.raises(CacheImageError) as err:
        convert_to_cache_array(Image.fromarray(rng.integers(0, 256, (80, 80, 3), dtype=np.uint8), mode="RGB"))
    assert err.value.reason == "channels_differ"


# --------------------------------------------------------------------------
# the model input is bit-identical to the PNG backend's
# --------------------------------------------------------------------------

def test_evaluation_input_is_bit_identical_to_the_png_backend_for_l_rgb_and_rgba_sources(tmp_path, mixed):
    root, _, entries = mixed
    cache, _ = _built(tmp_path, entries)
    png = PngBackend().make_dataset(root, None, train=False)
    cached = Uint8CacheBackend(cache).make_dataset(root, None, train=False)
    assert len(png) == len(cached) == 7
    for i in range(len(png)):
        png_image, png_target = png[i]
        cache_item, cache_target = cached[i]
        assert cache_item.dtype == torch.uint8 and tuple(cache_item.shape) == (1, 224, 224)
        assert torch.equal(png_image, prepare_batch(cache_item.unsqueeze(0), CPU)[0]), i
        assert torch.equal(png_target, cache_target)


def test_training_input_is_bit_identical_under_the_same_torch_seed(tmp_path, mixed):
    root, _, entries = mixed
    cache, _ = _built(tmp_path, entries)
    png = PngBackend().make_dataset(root, None, train=True)
    cached = Uint8CacheBackend(cache).make_dataset(root, None, train=True)
    shifted = 0
    for i in range(len(png)):
        for seed in (0, 1, 2):
            torch.manual_seed(seed * 100 + i)
            png_image, _ = png[i]
            torch.manual_seed(seed * 100 + i)
            cache_item, _ = cached[i]
            assert torch.equal(png_image, prepare_batch(cache_item.unsqueeze(0), CPU)[0]), (i, seed)
            centre, _ = Uint8CacheBackend(cache).make_dataset(root, None, train=False)[i]
            shifted += int(not torch.equal(cache_item, centre))
    assert shifted > 0  # the random crop/flip really varied the crop (the check is not vacuous)


def test_prepare_batch_matches_totensor_plus_normalize_and_leaves_float_batches_alone():
    rng = np.random.default_rng(3)
    grey = rng.integers(0, 256, (2, 1, 8, 8), dtype=np.uint8)
    out = prepare_batch(torch.from_numpy(grey), CPU)
    assert out.dtype == torch.float32 and tuple(out.shape) == (2, 3, 8, 8)
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    for b in range(2):
        rgb = torch.from_numpy(np.stack([grey[b, 0]] * 3, axis=-1))
        expected = normalize(rgb.permute(2, 0, 1).to(torch.float32).div(255.0))
        assert torch.equal(out[b], expected)
    floats = torch.randn(2, 3, 8, 8)
    assert prepare_batch(floats, CPU) is floats  # the PNG backend's batches are only moved


def test_pixel_verification_passes_on_a_good_cache_and_detects_a_corrupted_shard(tmp_path, mixed):
    root, _, entries = mixed
    cache, report = _built(tmp_path, entries)
    reader = CacheReader(cache)
    non_l = report.non_l_rows
    assert non_l == [2, 3, 5]  # the RGB and RGBA sources
    rows = pick_verification_rows(7, non_l, sample=4, seed=0)
    assert set(non_l) <= set(rows) and rows == pick_verification_rows(7, non_l, sample=4, seed=0)
    assert verify_against_png(reader, entries, rows)["all_equal"] is True
    assert reader.verify_shard_hashes() == []
    reader._maps.clear()  # release the memory maps: Windows will not overwrite a mapped file
    gc.collect()
    shard = cache / "shard_00000.npy"
    data = bytearray(shard.read_bytes())
    data[-1] ^= 0xFF
    shard.write_bytes(bytes(data))
    assert [b["file"] for b in CacheReader(cache).verify_shard_hashes()] == ["shard_00000.npy"]
    bad = verify_against_png(CacheReader(cache), entries, [2])
    assert bad["all_equal"] is False and bad["stored_array_mismatches"] == [2]


# --------------------------------------------------------------------------
# build: failures, resume, determinism, processes
# --------------------------------------------------------------------------

def test_failures_are_all_listed_every_shard_is_built_and_no_index_is_written(tmp_path):
    rng = np.random.default_rng(2)

    def non_square(path):
        Image.fromarray(rng.integers(0, 256, (80, 120), dtype=np.uint8), mode="L").save(path)

    def colour(path):
        Image.fromarray(rng.integers(0, 256, (SIZE, SIZE, 3), dtype=np.uint8), mode="RGB").save(path)

    def unreadable(path):
        path.write_bytes(b"this is not a png")

    _, names, entries = make_dataset(
        tmp_path / "d", ["L", "L", "L"],
        extra={"bad_nonsquare.png": non_square, "good_after.png": lambda p: _save(p, np.zeros((SIZE, SIZE), np.uint8), "L"),
               "bad_colour.png": colour, "bad_unreadable.png": unreadable})
    out = tmp_path / "cache"
    report = build_cache(entries, out, workers=1, shard_size=2)
    assert not report.ok and report.cache_sha256 is None
    assert sorted(f["reason"] for f in report.failures) == ["channels_differ", "not_256x256", "unreadable"]
    assert {f["name"] for f in report.failures} == {"bad_nonsquare.png", "bad_colour.png", "bad_unreadable.png"}
    assert len(list(out.glob("shard_*.npy"))) == 4          # every shard was attempted, none skipped at the first failure
    assert not (out / "index.json").exists()
    listed = json.loads((out / "failures.json").read_text())
    assert listed["n_failures"] == 3 and len(listed["failures"]) == 3
    with pytest.raises(CacheError):
        CacheReader(out)                                    # an incomplete build cannot be used


def test_resume_rebuilds_only_the_missing_shard_and_reproduces_the_same_cache(tmp_path, mixed):
    _, _, entries = mixed
    cache, first = _built(tmp_path, entries, shard_size=3)   # 7 images -> 3 shards
    stamp = {p.name: p.stat().st_mtime_ns for p in cache.glob("shard_*.npy")}
    (cache / "shard_00001.npy").unlink()
    (cache / "shard_00001.json").unlink()
    again = build_cache(entries, cache, workers=1, shard_size=3)
    assert again.ok and (again.shards_built, again.shards_skipped) == (1, 2)
    assert again.cache_sha256 == first.cache_sha256
    for name in ("shard_00000.npy", "shard_00002.npy"):
        assert (cache / name).stat().st_mtime_ns == stamp[name]
    third = build_cache(entries, cache, workers=1, shard_size=3)
    assert (third.shards_built, third.shards_skipped) == (0, 3)


def test_two_builds_give_the_same_hash_and_a_changed_image_changes_it(tmp_path, mixed):
    root, _, entries = mixed
    a = build_cache(entries, tmp_path / "a", workers=1, shard_size=4)
    b = build_cache(entries, tmp_path / "b", workers=1, shard_size=4)
    assert a.cache_sha256 == b.cache_sha256 and a.cache_sha256 is not None
    _save(root / "images" / "img_001.png", np.full((SIZE, SIZE), 7, dtype=np.uint8), "L")
    c = build_cache(entries, tmp_path / "c", workers=1, shard_size=4)
    assert c.cache_sha256 != a.cache_sha256


def test_meta_records_versions_modes_and_the_command(tmp_path, mixed):
    _, _, entries = mixed
    out = tmp_path / "cache"
    build_cache(entries, out, workers=1, shard_size=4, csv_sha256="ab" * 32, command="python build", git_commit="abc1234")
    meta = json.loads((out / "meta.json").read_text())
    assert meta["source_png_modes"] == {"L": 4, "RGB": 2, "RGBA": 1}
    assert meta["metadata_csv_sha256"] == "ab" * 32 and meta["command"] == "python build" and meta["git_commit"] == "abc1234"
    assert set(meta["libraries"]) == {"numpy", "Pillow", "torch", "torchvision"} and meta["python"]
    index = json.loads((out / "index.json").read_text())
    assert meta["cache_sha256"] == hashlib.sha256((out / "index.json").read_bytes()).hexdigest()
    assert index["n_images"] == 7 and sum(s["count"] for s in index["shards"]) == 7
    assert index["pixel_dtype"] == "uint8" and index["channels_stored"] == 1 and index["image_size"] == 256


def test_a_multi_process_build_equals_the_single_process_build(tmp_path, mixed):
    _, _, entries = mixed
    single = build_cache(entries, tmp_path / "one", workers=1, shard_size=2)
    multi = build_cache(entries, tmp_path / "two", workers=2, shard_size=2)
    assert single.ok and multi.ok and single.cache_sha256 == multi.cache_sha256


# --------------------------------------------------------------------------
# reader and backend
# --------------------------------------------------------------------------

def test_reader_refuses_a_cache_built_for_other_metadata_or_with_a_damaged_shard(tmp_path, mixed):
    root, names, entries = mixed
    cache, _ = _built(tmp_path, entries)
    reader = CacheReader(cache)
    reader.check_names(names)
    with pytest.raises(CacheError):
        reader.check_names(list(reversed(names)))
    with pytest.raises(CacheError):
        reader.check_names(names[:-1])
    (cache / "shard_00002.npy").write_bytes(b"short")
    with pytest.raises(CacheError):
        CacheReader(cache)
    with pytest.raises(IndexError):
        reader.image(99)


def test_reader_pickles_without_its_memory_maps(tmp_path, mixed):
    import pickle

    _, _, entries = mixed
    cache, _ = _built(tmp_path, entries)
    reader = CacheReader(cache)
    reader.image(0)
    assert reader._maps
    clone = pickle.loads(pickle.dumps(reader))
    assert clone._maps == {} and np.array_equal(clone.image(0), reader.image(0))


def test_backends_are_registered_default_to_png_and_validate_their_arguments(tmp_path, mixed):
    assert image_backend_registry.list_keys() == ["png", "uint8_cache"]
    assert DEFAULT_IMAGE_BACKEND == "png" and create_image_backend().name == "png"
    with pytest.raises(ValueError):
        Uint8CacheBackend(None)
    with pytest.raises(ValueError):
        PngBackend(tmp_path)
    with pytest.raises(CacheError):
        Uint8CacheBackend(tmp_path / "nowhere")
    _, _, entries = mixed
    cache, report = _built(tmp_path, entries)
    backend = create_image_backend("uint8_cache", cache)
    assert backend.describe()["cache_sha256"] == report.cache_sha256
    assert backend.config_identity() == {"name": "uint8_cache", "cache_sha256": report.cache_sha256}
    assert PngBackend().describe() == {"name": "png", "cache_sha256": None} and PngBackend().config_identity() is None


def test_settings_yaml_and_the_cli_agree_on_the_default_backend():
    path = Path(imagecache.__file__).resolve().parents[1] / "config" / "settings.yaml"
    section = yaml.safe_load(path.read_text(encoding="utf-8"))["data"]
    assert section["image_backend"] == DEFAULT_IMAGE_BACKEND and section["image_cache_root"] is None
    assert section["image_backend"] in image_backend_registry.list_keys()
    args = _build_arg_parser().parse_args(["--data-root", "d", "--partition-root", "p"])
    assert args.image_backend == section["image_backend"] and args.image_cache_root is None
    with pytest.raises(SystemExit):
        _build_arg_parser().parse_args(["--data-root", "d", "--partition-root", "p", "--image-backend", "nope"])


def test_the_data_path_never_initialises_cuda(tmp_path, mixed, monkeypatch):
    source = Path(imagecache.__file__).read_text(encoding="utf-8")
    assert "torch.cuda" not in source and ".cuda(" not in source
    root, _, entries = mixed
    cache, _ = _built(tmp_path, entries)

    def boom(*args, **kwargs):
        raise AssertionError("CUDA was initialised from the data path")

    monkeypatch.setattr(torch.cuda, "_lazy_init", boom)
    dataset = Uint8CacheBackend(cache).make_dataset(root, None, train=True)
    for images, _targets in DataLoader(dataset, batch_size=4, shuffle=True):
        prepare_batch(images, CPU)


# --------------------------------------------------------------------------
# results JSON and config hash
# --------------------------------------------------------------------------

LEGACY_ARGS = dict(experiment_name="x", seed=1, num_clients=3, num_rounds=2, local_epochs=1, batch_size=4, n_qubits=1024,
                   intercept_probability=0.0, elapsed_seconds=1.0, round_logs=[], n_failures=0,
                   init_weights={"pretrained": False, "source": "random_init", "weights_file": None, "sha256": None})


def test_config_hash_is_unchanged_for_png_and_changes_for_the_cache():
    legacy_config = {
        "name": "x", "seed": 1, "num_clients": 3, "num_rounds": 2, "local_epochs": 1, "batch_size": 4, "n_qubits": 1024,
        "intercept_probability": 0.0, "pretrained": False, "init_weights_sha256": None, "optimizer": None, "qkd": None,
        "policy": None, "channel_plan": None}
    legacy_hash = hashlib.sha256(json.dumps(legacy_config, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    png = build_experiment_log(**LEGACY_ARGS)["experiment"]
    assert png["config_hash"] == legacy_hash and "image_backend" not in png
    assert png["data_backend"] == {"name": "png", "cache_sha256": None}
    identity = {"name": "uint8_cache", "cache_sha256": "ab" * 32}
    cached = build_experiment_log(**LEGACY_ARGS, data_backend={"name": "uint8_cache", "cache_sha256": "ab" * 32},
                                  data_backend_identity=identity)["experiment"]
    assert cached["config_hash"] != legacy_hash and cached["image_backend"] == identity
    assert cached["data_backend"]["cache_sha256"] == "ab" * 32


def test_round_two_onward_data_wait_share_excludes_round_one():
    def phase(wait, wall):
        return {"fit": {"phase_s": wall, "data_wait_s": wait, "compute_s": wall - wait, "n_images": 10}}

    logs = [{"round": 1, "timing": {"phases": phase(90.0, 100.0)}},
            {"round": 2, "timing": {"phases": phase(10.0, 100.0)}},
            {"round": 3, "timing": {"phases": phase(30.0, 100.0)}}]
    summary = summarize_rounds(logs)
    assert summary["data_wait_share_loaded"] == pytest.approx(130.0 / 300.0)
    assert summary["data_wait_share_loaded_from_round_2"] == pytest.approx(40.0 / 200.0)
    assert summarize_rounds(logs[:1])["data_wait_share_loaded_from_round_2"] is None


def test_summarize_run_prints_the_backend_and_the_round_two_share():
    path = Path(__file__).resolve().parents[1] / "scripts" / "summarize_run.py"
    spec = importlib.util.spec_from_file_location("summarize_run_for_cache_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = {"experiment": {"name": "t", "config_hash": "abc", "seed": 0, "num_rounds": 1, "num_clients": 3,
                             "local_epochs": 1, "batch_size": 4, "pretrained": False,
                             "elapsed_seconds": 1.0,
                             "data_backend": {"name": "uint8_cache", "cache_sha256": "0123456789abcdef" * 4},
                             "profile": {"summary": {"data_wait_share_loaded": 0.4,
                                                     "data_wait_share_loaded_from_round_2": 0.1,
                                                     "rule_50_percent": {"threshold": 0.5,
                                                                         "data_wait_exceeds_threshold": False}}}},
              "rounds": []}
    text = "\n".join(module.summarize(report))
    assert "data backend    : uint8_cache  cache_sha256 0123456789abcdef..." in text
    assert "from round 2: 10%" in text


def test_run_experiment_on_the_cache_reproduces_the_png_run_and_records_the_cache(synthetic_data, tmp_path):
    data_root, partition_root = synthetic_data
    names = [str(x) for x in load_metadata(data_root)["Image Index"]]
    entries, _ = resolve_entries(data_root, names)
    cache = tmp_path / "cache"
    report = build_cache(entries, cache, workers=1, shard_size=16)
    assert report.ok and report.n_images == 36

    def run(name, backend):
        out = tmp_path / f"{name}.json"
        run_experiment(data_root=data_root, partition_root=partition_root, num_clients=3, num_rounds=1, local_epochs=1,
                       batch_size=4, n_qubits=1024, intercept_probability=0.0, seed=0, experiment_name="cache_e2e",
                       output_path=out, pretrained=False, image_backend=backend)
        return json.loads(out.read_text())

    png = run("png", None)
    cached = run("cache", create_image_backend("uint8_cache", cache))
    assert png["experiment"]["data_backend"] == {"name": "png", "cache_sha256": None}
    assert cached["experiment"]["data_backend"]["cache_sha256"] == report.cache_sha256
    assert cached["experiment"]["config_hash"] != png["experiment"]["config_hash"]
    assert "image_backend" in cached["experiment"] and "image_backend" not in png["experiment"]
    # identical model input and identical RNG use: on the CPU the validation loss agrees (reported in docs/06 as report-only
    # for GPU runs; here it is a regression check of the wiring)
    assert cached["rounds"][0]["val_eval"]["loss"] == pytest.approx(png["rounds"][0]["val_eval"]["loss"], abs=1e-5)
    assert cached["rounds"][0]["timing"]["phases"]["fit"]["n_images"] == png["rounds"][0]["timing"]["phases"]["fit"]["n_images"]


def test_hospital_loader_default_is_png_and_accepts_a_cache_backend(synthetic_data, tmp_path):
    data_root, partition_root = synthetic_data
    names = [str(x) for x in load_metadata(data_root)["Image Index"]]
    entries, _ = resolve_entries(data_root, names)
    cache = tmp_path / "cache"
    assert build_cache(entries, cache, workers=1, shard_size=16).ok
    default_images, _ = next(iter(get_hospital_dataloader(data_root, partition_root, 0, batch_size=4, train=False)))
    cache_images, _ = next(iter(get_hospital_dataloader(
        data_root, partition_root, 0, batch_size=4, train=False, image_backend=Uint8CacheBackend(cache))))
    assert default_images.dtype == torch.float32 and tuple(default_images.shape[1:]) == (3, 224, 224)
    assert cache_images.dtype == torch.uint8 and tuple(cache_images.shape[1:]) == (1, 224, 224)
    assert torch.equal(default_images, prepare_batch(cache_images, CPU))


# --------------------------------------------------------------------------
# folder scan and the script
# --------------------------------------------------------------------------

def test_resolve_entries_scans_nested_layouts_once_and_reuses_the_saved_index(tmp_path, monkeypatch):
    root = tmp_path / "data"
    nested = root / "images_001" / "images"
    nested.mkdir(parents=True)
    rng = np.random.default_rng(5)
    for name in ("a.png", "b.png"):
        Image.fromarray(rng.integers(0, 256, (SIZE, SIZE), dtype=np.uint8), mode="L").save(nested / name)
    cache = tmp_path / "path_index.json"
    entries, seconds = resolve_entries(root, ["b.png", "a.png"], index_cache=cache)
    assert [n for n, _ in entries] == ["b.png", "a.png"] and seconds > 0 and cache.exists()

    def no_scan(self, pattern):
        raise AssertionError("the folder tree was scanned again")

    monkeypatch.setattr(Path, "rglob", no_scan)
    again, seconds_again = resolve_entries(root, ["a.png"], index_cache=cache)
    assert again == [("a.png", str(nested / "a.png"))] and seconds_again == 0.0
    monkeypatch.undo()
    with pytest.raises(FileNotFoundError):
        resolve_entries(root, ["nope.png"], index_cache=cache)


def _script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_image_cache.py"
    spec = importlib.util.spec_from_file_location("build_image_cache_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_builds_verifies_and_exits_non_zero_on_a_failure(tmp_path, capsys, mixed):
    root, _, _ = mixed
    script = _script()
    out = tmp_path / "cache"
    code = script.main(["--data-root", str(root), "--output", str(out), "--workers", "1", "--shard-size", "3",
                        "--verify-sample", "4"])
    text = capsys.readouterr().out
    assert code == 0 and "cache_sha256:" in text and "VERIFY: OK" in text and "source PNG modes" in text
    assert script.main(["--data-root", str(root), "--verify", str(out), "--verify-sample", "7"]) == 0
    assert "VERIFY: OK" in capsys.readouterr().out
    (root / "images" / "img_000.png").write_bytes(b"corrupt")
    broken = tmp_path / "broken"
    assert script.main(["--data-root", str(root), "--output", str(broken), "--workers", "1", "--shard-size", "3"]) == 1
    text = capsys.readouterr().out
    assert "FAILED: 1 image(s)" in text and "img_000.png" in text and "Nothing was excluded" in text
    assert not (broken / "index.json").exists()


def test_script_trial_prints_a_labelled_projection(tmp_path, capsys, mixed):
    root, _, _ = mixed
    code = _script().main(["--data-root", str(root), "--output", str(tmp_path / "trial"), "--workers", "1",
                           "--limit", "4", "--shard-size", "4"])
    text = capsys.readouterr().out
    assert code == 0 and "projection (not a measurement of the full build)" in text
