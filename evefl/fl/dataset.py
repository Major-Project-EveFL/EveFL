"""
NIH ChestX-ray14 dataset loading and patient-level Dirichlet non-IID
partitioning across simulated hospital clients (see partition.py).

Expected layout on disk:
    <data_root>/
        images/                # all .png files (NIH's flat layout, or
                                # any nested layout — see __getitem__)
        Data_Entry_2017.csv    # official NIH metadata file

Run `partition_and_save()` ONCE before training. It writes:
    <partition_root>/
        hospital_0/indices.npy
        hospital_1/indices.npy
        hospital_2/indices.npy
        test/indices.npy           # held-out IID test split (disjoint patients): final report ONLY
        val/indices.npy            # held-out validation split (disjoint patients): model selection
        partition_meta.json

Nothing quantum here — this is standard PyTorch data loading.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from evefl.fl.model import CHESTXRAY_LABELS, NUM_CLASSES, get_eval_transform, get_train_transform
from evefl.fl.partition import audit_partition, split_by_patient

log = logging.getLogger(__name__)

TEST_FRACTION = 0.10
VAL_FRACTION = 0.10  # patients held out for model selection (the test set is for the final report only)
N_HOSPITALS = 3
DEFAULT_BATCH_SIZE = 32


def parse_label_matrix(finding_labels) -> np.ndarray:
    """Multi-hot [N, 14] float32 matrix from the CSV "Finding Labels" column
    ("No Finding" and unknown strings map to an all-zero row)."""
    label_matrix = np.zeros((len(finding_labels), NUM_CLASSES), dtype=np.float32)
    for row_i, finding_str in enumerate(finding_labels):
        for label in str(finding_str).split("|"):
            label = label.strip()
            if label in CHESTXRAY_LABELS:
                label_matrix[row_i, CHESTXRAY_LABELS.index(label)] = 1.0
    return label_matrix


def load_metadata(data_root: str | Path) -> pd.DataFrame:
    csv_path = Path(data_root) / "Data_Entry_2017.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Metadata CSV not found at {csv_path}. Download ChestX-ray14 "
            "from https://nihcc.app.box.com/v/ChestXray-NIHCC (or use the "
            "Kaggle-hosted copy)."
        )
    return pd.read_csv(csv_path)



# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ChestXray14Dataset(Dataset):
    """
    PyTorch Dataset for NIH ChestX-ray14.

    Labels are parsed once into an in-memory float32 tensor of shape
    [N, 14]; images are loaded from disk on demand in __getitem__.

    Args:
        data_root: folder containing images/ and Data_Entry_2017.csv
        indices:   optional row-index subset (used for partitioned splits)
        transform: torchvision transform; defaults to the training transform
    """

    # Shared across ALL instances pointed at the same data_root, so the
    # (expensive, ~112k-file) rglob scan for NIH's nested images_00X/
    # layout runs ONCE per process, not once per Dataset instantiation.
    # Without this, 3 hospital clients + N test-set evaluations each
    # trigger their own full rescan -- on Kaggle's mounted input, that
    # can turn a several-minute run into the better part of an hour.
    _shared_image_index_cache: dict[str, dict[str, Path]] = {}

    def __init__(
        self,
        data_root: str | Path,
        indices: Optional[np.ndarray] = None,
        transform=None,
    ):
        self.data_root = Path(data_root)
        self.transform = transform or get_train_transform()

        df = load_metadata(self.data_root)
        label_matrix = parse_label_matrix(df["Finding Labels"])

        self._image_names: np.ndarray = df["Image Index"].to_numpy()
        self._labels: torch.Tensor = torch.from_numpy(label_matrix)

        if indices is not None:
            self._image_names = self._image_names[indices]
            self._labels = self._labels[indices]

    def __len__(self) -> int:
        return len(self._image_names)

    def _resolve_image_path(self, image_name: str) -> Path:
        direct = self.data_root / "images" / image_name
        if direct.exists():
            return direct

        # NIH's Kaggle mirror sometimes ships as images_001/images, ...,
        # images_012/images rather than one flat images/ folder. Build a
        # name->path index once PER data_root (shared across every
        # Dataset instance, not rebuilt per instance -- see the class
        # docstring) instead of rglob-ing per image or per instance.
        root_key = str(self.data_root)
        if root_key not in ChestXray14Dataset._shared_image_index_cache:
            log.info("Building image path index under %s (first lookup miss, shared across this session)...",
                      self.data_root)
            ChestXray14Dataset._shared_image_index_cache[root_key] = {
                p.name: p for p in self.data_root.rglob("*.png")
            }

        index = ChestXray14Dataset._shared_image_index_cache[root_key]
        if image_name in index:
            return index[image_name]

        raise FileNotFoundError(f"Image not found anywhere under {self.data_root}: {image_name}")

    def __getitem__(self, idx: int):
        image_name = self._image_names[idx]
        img_path = self._resolve_image_path(image_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self._labels[idx]

    @property
    def labels(self) -> torch.Tensor:
        """Full label matrix — used by the partitioner for class statistics."""
        return self._labels


# ---------------------------------------------------------------------------
# Patient-level Dirichlet non-IID partitioning (scheme: see partition.py)
# ---------------------------------------------------------------------------

def partition_and_save(
    data_root: str | Path,
    partition_root: str | Path,
    *,
    n_clients: int = N_HOSPITALS,
    alpha: float = 0.5,
    test_fraction: float = TEST_FRACTION,
    seed: int = 42,
    subset_fraction: float = 1.0,
    val_fraction: float = VAL_FRACTION,
) -> None:
    """
    Split ChestX-ray14 BY PATIENT into n_clients disjoint non-IID hospital
    splits plus a disjoint IID test split, and write row indices to disk as
    .npy files (run once per experiment config, not once per round).

    Requires the "Patient ID" column of Data_Entry_2017.csv. The scheme
    (rarest-positive-label groups, Dirichlet over hospitals) is documented in
    evefl/fl/partition.py. The result is audited for overlap before writing;
    any overlap raises instead of saving.

    Args:
        subset_fraction: use only this fraction of PATIENTS, e.g. 0.02-0.05 for
            a quick end-to-end run on Kaggle, 1.0 for a full run.
    """
    partition_root = Path(partition_root)
    partition_root.mkdir(parents=True, exist_ok=True)

    log.info("Loading ChestX-ray14 metadata from %s ...", data_root)
    df = load_metadata(data_root)
    if "Patient ID" not in df.columns:
        raise KeyError(
            "Data_Entry_2017.csv has no 'Patient ID' column; a patient-level split needs it."
        )
    patient_ids = df["Patient ID"].to_numpy()
    labels = parse_label_matrix(df["Finding Labels"])

    split = split_by_patient(
        patient_ids, labels,
        n_clients=n_clients, alpha=alpha, test_fraction=test_fraction,
        seed=seed, subset_fraction=subset_fraction, val_fraction=val_fraction,
    )
    audit = audit_partition(
        patient_ids, labels, split.client_indices, split.test_indices,
        split.val_indices if val_fraction > 0 else None,
    )
    if not audit["ok"]:
        raise RuntimeError(f"Patient-level split is not disjoint: {audit['overlaps']}")

    for client_id, idx_array in enumerate(split.client_indices):
        out_dir = partition_root / f"hospital_{client_id}"
        out_dir.mkdir(exist_ok=True)
        np.save(out_dir / "indices.npy", idx_array)
        log.info("Hospital %d: %d images, %d patients -> %s",
                 client_id, len(idx_array), split.n_patients_per_client[client_id], out_dir)

    test_dir = partition_root / "test"
    test_dir.mkdir(exist_ok=True)
    np.save(test_dir / "indices.npy", split.test_indices)
    log.info("Test set: %d images, %d patients -> %s",
             len(split.test_indices), split.n_patients_test, test_dir)

    if val_fraction > 0:
        val_dir = partition_root / "val"
        val_dir.mkdir(exist_ok=True)
        np.save(val_dir / "indices.npy", split.val_indices)
        log.info("Validation set: %d images, %d patients -> %s",
                 len(split.val_indices), split.n_patients_val, val_dir)

    meta = {
        "scheme": "patient_level_dirichlet_rarest_label",
        "n_clients": n_clients,
        "alpha": alpha,
        "seed": seed,
        "test_fraction": test_fraction,
        "val_fraction": val_fraction,
        "n_patients_val": split.n_patients_val,
        "n_val": int(len(split.val_indices)),
        "subset_fraction": subset_fraction,
        "n_patients_total": split.n_patients_total,
        "n_patients_used": split.n_patients_used,
        "n_patients_test": split.n_patients_test,
        "n_patients_per_client": split.n_patients_per_client,
        "n_test": int(len(split.test_indices)),
        "hospital_sizes": [int(len(idx)) for idx in split.client_indices],
        "dominant_group_counts": {str(k): v for k, v in split.dominant_group_counts.items()},
        # The split draws from numpy's Generator (permutation, dirichlet), whose streams NumPy does not guarantee to be
        # identical across versions: a different version can give a different split and so different index_sha256.
        "numpy_version": np.__version__,
        "index_sha256": _index_hashes(partition_root),
        "audit": audit,
    }
    with open(partition_root / "partition_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    log.info("Partitioning complete (disjoint: %s). Metadata: %s/partition_meta.json",
             audit["ok"], partition_root)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _index_hashes(partition_root: Path) -> dict:
    """SHA-256 (hex) of every `<split>/indices.npy` under partition_root, keyed by split name."""
    return {
        p.parent.name: _sha256_file(p)
        for p in sorted(partition_root.glob("*/indices.npy"))
    }


def verify_partition_hashes(partition_root: str | Path) -> dict:
    """
    Compare the index files on disk with the SHA-256 values stored in partition_meta.json.

    Returns {split_name: {"expected": ..., "actual": ...}} for every split whose file is missing,
    changed or unexpected; an empty dict means the partition is byte-identical to what was saved.
    Raises KeyError if the metadata predates the hashes (no `index_sha256`).
    """
    partition_root = Path(partition_root)
    meta = json.loads((partition_root / "partition_meta.json").read_text())
    expected = meta["index_sha256"]
    actual = _index_hashes(partition_root)
    return {
        name: {"expected": expected.get(name), "actual": actual.get(name)}
        for name in sorted(set(expected) | set(actual))
        if expected.get(name) != actual.get(name)
    }


def audit_saved_partition(data_root: str | Path, partition_root: str | Path) -> dict:
    """Re-load a saved partition and recompute overlap counts and label stats."""
    partition_root = Path(partition_root)
    df = load_metadata(data_root)
    patient_ids = df["Patient ID"].to_numpy()
    labels = parse_label_matrix(df["Finding Labels"])
    client_dirs = sorted(partition_root.glob("hospital_*"), key=lambda p: int(p.name.split("_")[1]))
    client_indices = [np.load(d / "indices.npy") for d in client_dirs]
    test_indices = np.load(partition_root / "test" / "indices.npy")
    val_path = partition_root / "val" / "indices.npy"
    val_indices = np.load(val_path) if val_path.exists() else None
    return audit_partition(patient_ids, labels, client_indices, test_indices, val_indices)


# ---------------------------------------------------------------------------
# DataLoader factories
# ---------------------------------------------------------------------------

def get_hospital_dataloader(
    data_root: str | Path,
    partition_root: str | Path,
    hospital_id: int,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    train: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    idx_path = Path(partition_root) / f"hospital_{hospital_id}" / "indices.npy"
    if not idx_path.exists():
        raise FileNotFoundError(
            f"Partition not found at {idx_path}. Run dataset.partition_and_save() first."
        )

    indices = np.load(idx_path)
    transform = get_train_transform() if train else get_eval_transform()
    dataset = ChestXray14Dataset(data_root, indices=indices, transform=transform)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=train and len(dataset) > batch_size,
    )


def get_test_dataloader(
    data_root: str | Path,
    partition_root: str | Path,
    *,
    batch_size: int = 64,
    num_workers: int = 0,
    split: str = "test",
) -> DataLoader:
    """Held-out evaluation loader. `split` is "test" (final report only) or "val" (model selection)."""
    if split not in ("test", "val"):
        raise ValueError(f"split must be 'test' or 'val', got {split!r}")
    idx_path = Path(partition_root) / split / "indices.npy"
    if not idx_path.exists():
        raise FileNotFoundError(
            f"{split} partition not found at {idx_path}. Run dataset.partition_and_save() first."
        )

    indices = np.load(idx_path)
    dataset = ChestXray14Dataset(data_root, indices=indices, transform=get_eval_transform())

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
