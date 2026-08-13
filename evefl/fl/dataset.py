"""
NIH ChestX-ray14 dataset loader and Dirichlet partitioning.

Run partition_and_save() ONCE before starting FL training to create the
three non-IID hospital splits on disk. The FL training loop then loads
pre-partitioned data, which is faster and keeps experiments reproducible.

Dataset structure expected on disk:
    <data_root>/
        images/               # all 112,120 .png files
        Data_Entry_2017.csv   # official NIH metadata file

After partition_and_save():
    <partition_root>/
        hospital_0/indices.npy
        hospital_1/indices.npy
        hospital_2/indices.npy
        test/indices.npy       # held-out IID test split (10% of total)

Nothing quantum here — this is standard PyTorch data loading.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from evefl.fl.model import CHESTXRAY_LABELS, NUM_CLASSES, get_train_transform, get_eval_transform

log = logging.getLogger(__name__)

# Fraction of the full dataset held out as a shared IID test set
TEST_FRACTION = 0.10
N_HOSPITALS   = 3


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class ChestXray14Dataset(Dataset):
    """
    PyTorch Dataset for NIH ChestX-ray14.

    Loads images on demand from disk; labels are pre-loaded into memory
    as a float32 tensor (shape: [N, 14]).

    Args:
        data_root:  Path to the folder containing images/ and Data_Entry_2017.csv
        indices:    Optional numpy array of row indices to use (for partitioned subsets).
                    If None, the full dataset is used.
        transform:  torchvision transform applied to each image.
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
                f"Metadata CSV not found at {csv_path}. "
                "Download ChestX-ray14 from https://nihcc.app.box.com/v/ChestXray-NIHCC"
            )

        df = pd.read_csv(csv_path)

        # Build binary label matrix — one column per pathology
        label_matrix = np.zeros((len(df), NUM_CLASSES), dtype=np.float32)
        for i, finding_str in enumerate(df["Finding Labels"]):
            for label in finding_str.split("|"):
                label = label.strip()
                if label in CHESTXRAY_LABELS:
                    label_matrix[i, CHESTXRAY_LABELS.index(label)] = 1.0

        self._image_names: np.ndarray = df["Image Index"].values
        self._labels: torch.Tensor    = torch.from_numpy(label_matrix)

        # Apply index subset if provided
        if indices is not None:
            self._image_names = self._image_names[indices]
            self._labels      = self._labels[indices]

    def __len__(self) -> int:
        return len(self._image_names)

    def __getitem__(self, idx: int):
        img_path = self.data_root / "images" / self._image_names[idx]
        image    = Image.open(img_path).convert("RGB")   # X-rays are greyscale but ResNet needs 3ch
        if self.transform:
            image = self.transform(image)
        return image, self._labels[idx]

    @property
    def labels(self) -> torch.Tensor:
        """Full label matrix, useful for computing class statistics."""
        return self._labels


# ---------------------------------------------------------------------------
# Dirichlet partitioning
# ---------------------------------------------------------------------------

def _dirichlet_partition(
    labels: np.ndarray,
    n_clients: int,
    alpha: float,
    rng: np.random.Generator,
) -> list[list[int]]:
    """
    Partition dataset sample indices across n_clients using Dir(alpha).

    Strategy: for each of the 14 pathology classes, distribute the positive
    samples across clients using Dir(alpha) proportions. Non-positive samples
    (No Finding) are distributed uniformly. This creates heterogeneous
    class distributions — lower alpha = more skewed, higher alpha = more IID.

    alpha=0.5 gives the moderate non-IID setting used in the EveFL paper.

    Args:
        labels:    Binary label matrix, shape [N, num_classes].
        n_clients: Number of client partitions.
        alpha:     Dirichlet concentration parameter.
        rng:       Numpy random generator (pass a seeded one for reproducibility).

    Returns:
        List of n_clients lists, each containing sample indices.
    """
    n_samples  = labels.shape[0]
    n_classes  = labels.shape[1]
    client_idx: list[set] = [set() for _ in range(n_clients)]

    for c in range(n_classes):
        positive_indices = np.where(labels[:, c] == 1)[0]
        if len(positive_indices) == 0:
            continue

        rng.shuffle(positive_indices)
        proportions = rng.dirichlet(alpha * np.ones(n_clients))

        # Convert proportions to integer counts, ensuring all samples assigned
        counts = (proportions * len(positive_indices)).astype(int)
        counts[-1] = len(positive_indices) - counts[:-1].sum()  # remainder to last client

        start = 0
        for client_id, count in enumerate(counts):
            end = start + count
            for idx in positive_indices[start:end]:
                client_idx[client_id].add(int(idx))
            start = end

    # Any samples not yet assigned (no positive label = "No Finding") go uniformly
    all_assigned = set().union(*client_idx)
    unassigned   = [i for i in range(n_samples) if i not in all_assigned]
    rng.shuffle(unassigned)
    for i, idx in enumerate(unassigned):
        client_idx[i % n_clients].add(idx)

    return [sorted(s) for s in client_idx]


# ---------------------------------------------------------------------------
# One-time setup: partition and save to disk
# ---------------------------------------------------------------------------

def partition_and_save(
    data_root: str | Path,
    partition_root: str | Path,
    *,
    n_clients: int = N_HOSPITALS,
    alpha: float = 0.5,
    test_fraction: float = TEST_FRACTION,
    seed: int = 42,
    subset_fraction: float = 1.0,   # set to 0.15 for fast debug runs
) -> None:
    """
    Partition ChestX-ray14 into n_clients non-IID hospital splits + test set.

    Run this ONCE before starting FL training. Results are saved as
    .npy index files so training is reproducible and fast.

    Args:
        data_root:       Root folder containing images/ and Data_Entry_2017.csv
        partition_root:  Where to write hospital_0/, hospital_1/, etc.
        n_clients:       Number of hospital partitions (default 3).
        alpha:           Dirichlet concentration (0.5 = moderate non-IID).
        test_fraction:   Fraction held out as shared IID test set.
        seed:            RNG seed for full reproducibility.
        subset_fraction: Use only this fraction of the data (for debug runs).
                         Set to 1.0 for full evaluation.
    """
    data_root      = Path(data_root)
    partition_root = Path(partition_root)
    partition_root.mkdir(parents=True, exist_ok=True)

    log.info("Loading ChestX-ray14 metadata from %s …", data_root)
    # Load labels only (no images needed for partitioning)
    dummy = ChestXray14Dataset(data_root, transform=get_train_transform())
    labels_np = dummy.labels.numpy()   # shape [N, 14]
    n_total   = len(dummy)

    rng = np.random.default_rng(seed)
    all_indices = np.arange(n_total)
    rng.shuffle(all_indices)

    # Optional subset for debug runs
    if subset_fraction < 1.0:
        n_keep      = int(n_total * subset_fraction)
        all_indices = all_indices[:n_keep]
        log.info("Debug mode: using %.0f%% of data (%d/%d samples).",
                 subset_fraction * 100, n_keep, n_total)

    # Carve out IID test set first
    n_test          = int(len(all_indices) * test_fraction)
    test_indices    = all_indices[:n_test]
    trainval_idx    = all_indices[n_test:]

    log.info("Test set: %d samples. Partitioning %d samples across %d hospitals …",
             n_test, len(trainval_idx), n_clients)

    # Dirichlet partition of the training pool
    trainval_labels = labels_np[trainval_idx]
    client_local_idx = _dirichlet_partition(trainval_labels, n_clients, alpha, rng)

    # Map local positions back to global dataset indices
    client_global_idx = [trainval_idx[local] for local in client_local_idx]

    # Save
    for client_id, idx_array in enumerate(client_global_idx):
        out_dir = partition_root / f"hospital_{client_id}"
        out_dir.mkdir(exist_ok=True)
        np.save(out_dir / "indices.npy", np.array(idx_array))
        log.info("Hospital %d: %d samples saved to %s", client_id, len(idx_array), out_dir)

    test_dir = partition_root / "test"
    test_dir.mkdir(exist_ok=True)
    np.save(test_dir / "indices.npy", test_indices)
    log.info("Test set: %d samples saved to %s", len(test_indices), test_dir)

    # Save partition metadata for reference
    meta = {
        "n_clients":       n_clients,
        "alpha":           alpha,
        "seed":            seed,
        "test_fraction":   test_fraction,
        "subset_fraction": subset_fraction,
        "n_total_used":    len(all_indices),
        "n_test":          int(n_test),
        "hospital_sizes":  [len(idx) for idx in client_global_idx],
    }
    with open(partition_root / "partition_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    log.info("Partitioning complete. Metadata saved to %s/partition_meta.json", partition_root)


# ---------------------------------------------------------------------------
# DataLoader factory (used by client.py)
# ---------------------------------------------------------------------------

def get_hospital_dataloader(
    data_root: str | Path,
    partition_root: str | Path,
    hospital_id: int,
    *,
    batch_size: int = 32,
    train: bool = True,
) -> DataLoader:
    """
    Return a DataLoader for a specific hospital's partition.

    Args:
        data_root:      Root folder with images/ and metadata CSV.
        partition_root: Where the .npy index files live.
        hospital_id:    Integer 0, 1, or 2.
        batch_size:     Batch size for training (32 matches the paper).
        train:          If True, uses augmentation transforms; else eval transforms.
    """
    idx_path = Path(partition_root) / f"hospital_{hospital_id}" / "indices.npy"
    if not idx_path.exists():
        raise FileNotFoundError(
            f"Partition not found at {idx_path}. "
            "Run dataset.partition_and_save() first."
        )

    indices   = np.load(idx_path)
    transform = get_train_transform() if train else get_eval_transform()
    dataset   = ChestXray14Dataset(data_root, indices=indices, transform=transform)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        drop_last=train,   # keeps batch sizes consistent during training
    )


def get_test_dataloader(
    data_root: str | Path,
    partition_root: str | Path,
    *,
    batch_size: int = 64,
) -> DataLoader:
    """Return a DataLoader for the shared IID test set."""
    idx_path = Path(partition_root) / "test" / "indices.npy"
    indices  = np.load(idx_path)
    dataset  = ChestXray14Dataset(data_root, indices=indices, transform=get_eval_transform())

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )