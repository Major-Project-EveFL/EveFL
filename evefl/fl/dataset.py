"""
NIH ChestX-ray14 dataset loading and Dirichlet non-IID partitioning
across simulated hospital clients.

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
        test/indices.npy           # held-out IID test split
        partition_meta.json

Nothing quantum here — this is standard PyTorch data loading.
"""

from __future__ import annotations

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

log = logging.getLogger(__name__)

TEST_FRACTION = 0.10
N_HOSPITALS = 3
DEFAULT_BATCH_SIZE = 32


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

    def __init__(
        self,
        data_root: str | Path,
        indices: Optional[np.ndarray] = None,
        transform=None,
    ):
        self.data_root = Path(data_root)
        self.transform = transform or get_train_transform()

        csv_path = self.data_root / "Data_Entry_2017.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Metadata CSV not found at {csv_path}. Download ChestX-ray14 "
                "from https://nihcc.app.box.com/v/ChestXray-NIHCC (or use the "
                "Kaggle-hosted copy)."
            )

        df = pd.read_csv(csv_path)

        label_matrix = np.zeros((len(df), NUM_CLASSES), dtype=np.float32)
        for row_i, finding_str in enumerate(df["Finding Labels"]):
            for label in str(finding_str).split("|"):
                label = label.strip()
                if label in CHESTXRAY_LABELS:
                    label_matrix[row_i, CHESTXRAY_LABELS.index(label)] = 1.0

        self._image_names: np.ndarray = df["Image Index"].values
        self._labels: torch.Tensor = torch.from_numpy(label_matrix)

        if indices is not None:
            self._image_names = self._image_names[indices]
            self._labels = self._labels[indices]

        self._image_index_cache: dict[str, Path] | None = None

    def __len__(self) -> int:
        return len(self._image_names)

    def _resolve_image_path(self, image_name: str) -> Path:
        direct = self.data_root / "images" / image_name
        if direct.exists():
            return direct

        # NIH's Kaggle mirror sometimes ships as images_001/images, ...,
        # images_012/images rather than one flat images/ folder. Build a
        # name->path index once (lazily) instead of rglob-ing per image.
        if self._image_index_cache is None:
            log.info("Building image path index under %s (first lookup miss)...", self.data_root)
            self._image_index_cache = {p.name: p for p in self.data_root.rglob("*.png")}

        if image_name in self._image_index_cache:
            return self._image_index_cache[image_name]

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
# Dirichlet non-IID partitioning
# ---------------------------------------------------------------------------

def _dirichlet_partition(
    labels: np.ndarray,
    n_clients: int,
    alpha: float,
    rng: np.random.Generator,
) -> list[list[int]]:
    """
    Partition sample indices across n_clients using a class-wise Dir(alpha).

    For each of the 14 pathology classes, positive samples are split
    across clients with proportions drawn from Dir(alpha) — lower alpha
    means more skewed (one hospital sees almost all Cardiomegaly cases,
    say), higher alpha approaches IID. "No Finding" samples (no positive
    label at all) are distributed uniformly since there's no class
    signal to skew them by.
    """
    n_samples = labels.shape[0]
    n_classes = labels.shape[1]
    client_idx: list[set] = [set() for _ in range(n_clients)]

    for c in range(n_classes):
        positive_indices = np.where(labels[:, c] == 1)[0]
        if len(positive_indices) == 0:
            continue

        rng.shuffle(positive_indices)
        proportions = rng.dirichlet(alpha * np.ones(n_clients))

        counts = (proportions * len(positive_indices)).astype(int)
        counts[-1] = len(positive_indices) - counts[:-1].sum()  # remainder to last client

        start = 0
        for client_id, count in enumerate(counts):
            end = start + count
            for idx in positive_indices[start:end]:
                client_idx[client_id].add(int(idx))
            start = end

    all_assigned = set().union(*client_idx) if client_idx else set()
    unassigned = [i for i in range(n_samples) if i not in all_assigned]
    rng.shuffle(unassigned)
    for i, idx in enumerate(unassigned):
        client_idx[i % n_clients].add(idx)

    return [sorted(s) for s in client_idx]


def partition_and_save(
    data_root: str | Path,
    partition_root: str | Path,
    *,
    n_clients: int = N_HOSPITALS,
    alpha: float = 0.5,
    test_fraction: float = TEST_FRACTION,
    seed: int = 42,
    subset_fraction: float = 1.0,
) -> None:
    """
    Partition ChestX-ray14 into n_clients non-IID hospital splits + a
    shared IID test split, and write everything to disk as .npy index
    files (fast, reproducible re-loading; run this once per experiment
    config, not once per round).

    Args:
        subset_fraction: use only this fraction of the full dataset —
            e.g. 0.02-0.05 for a quick end-to-end smoke test on Kaggle,
            1.0 for a full run.
    """
    data_root = Path(data_root)
    partition_root = Path(partition_root)
    partition_root.mkdir(parents=True, exist_ok=True)

    log.info("Loading ChestX-ray14 metadata from %s ...", data_root)
    labels_only = ChestXray14Dataset(data_root, transform=get_train_transform())
    labels_np = labels_only.labels.numpy()
    n_total = len(labels_only)

    rng = np.random.default_rng(seed)
    all_indices = np.arange(n_total)
    rng.shuffle(all_indices)

    if subset_fraction < 1.0:
        n_keep = max(1, int(n_total * subset_fraction))
        all_indices = all_indices[:n_keep]
        log.info("Subset mode: using %.1f%% of data (%d/%d samples).",
                  subset_fraction * 100, n_keep, n_total)

    n_test = int(len(all_indices) * test_fraction)
    test_indices = all_indices[:n_test]
    trainval_idx = all_indices[n_test:]

    log.info("Test set: %d samples. Partitioning %d samples across %d hospitals (alpha=%.2f)...",
              n_test, len(trainval_idx), n_clients, alpha)

    trainval_labels = labels_np[trainval_idx]
    client_local_idx = _dirichlet_partition(trainval_labels, n_clients, alpha, rng)
    client_global_idx = [trainval_idx[local] for local in client_local_idx]

    for client_id, idx_array in enumerate(client_global_idx):
        out_dir = partition_root / f"hospital_{client_id}"
        out_dir.mkdir(exist_ok=True)
        np.save(out_dir / "indices.npy", np.array(idx_array))
        log.info("Hospital %d: %d samples -> %s", client_id, len(idx_array), out_dir)

    test_dir = partition_root / "test"
    test_dir.mkdir(exist_ok=True)
    np.save(test_dir / "indices.npy", test_indices)
    log.info("Test set: %d samples -> %s", len(test_indices), test_dir)

    meta = {
        "n_clients": n_clients,
        "alpha": alpha,
        "seed": seed,
        "test_fraction": test_fraction,
        "subset_fraction": subset_fraction,
        "n_total_used": int(len(all_indices)),
        "n_test": int(n_test),
        "hospital_sizes": [len(idx) for idx in client_global_idx],
    }
    with open(partition_root / "partition_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    log.info("Partitioning complete. Metadata: %s/partition_meta.json", partition_root)


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
) -> DataLoader:
    idx_path = Path(partition_root) / "test" / "indices.npy"
    if not idx_path.exists():
        raise FileNotFoundError(
            f"Test partition not found at {idx_path}. Run dataset.partition_and_save() first."
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
