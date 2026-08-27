"""
<<<<<<< HEAD
EveFL ChestX-ray14 dataset and federated partitioning.

Responsibilities
----------------

This module handles ONLY:

    1. Reading ChestX-ray14 metadata
    2. Loading X-ray images
    3. Converting pathology labels into 14-dimensional multi-label targets
    4. Creating hospital/client partitions
    5. Creating train/validation/test DataLoaders
    6. Keeping the client data isolated

This module does NOT handle:

    - Flower strategy
    - QBER
    - BB84
    - cryptography
    - security-state classification
    - model aggregation

Federated setup
---------------

EveFL uses:

    3 hospital clients

with a non-IID distribution.

The intended research configuration uses:

    Dirichlet alpha = 0.5

The dataset has 14 pathology labels.

The implementation also supports a small subset for smoke testing.
=======
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
>>>>>>> c8d000e56cdc54490d8e8750eb80577f1165bdf3
"""

from __future__ import annotations

import json
import logging
<<<<<<< HEAD
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


log = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

NUM_CLASSES = 14

DEFAULT_IMAGE_SIZE = 224

DEFAULT_DIRICHLET_ALPHA = 0.5

DEFAULT_NUM_CLIENTS = 3

DEFAULT_BATCH_SIZE = 32

DEFAULT_SEED = 42


PATHOLOGIES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]


# ============================================================================
# Dataset row
# ============================================================================

@dataclass
class ChestXrayRecord:
    """
    One image + multi-label target.
    """

    image_path: str

    labels: List[float]

    patient_id: Optional[str] = None


# ============================================================================
# Label parsing
# ============================================================================

def parse_labels(
    finding_labels: str,
) -> List[float]:
    """
    Convert the ChestX-ray14 Finding Labels field into a 14-dimensional
    multi-hot vector.

    Example:

        "Atelectasis|Effusion"

    becomes:

        [1, 0, 1, 0, ...]

    "No Finding" produces an all-zero vector.
    """

    target = np.zeros(
        NUM_CLASSES,
        dtype=np.float32,
    )

    if not isinstance(
        finding_labels,
        str,
    ):
        return target.tolist()

    labels = [
        label.strip()
        for label in finding_labels.split("|")
    ]

    for label in labels:

        if label == "No Finding":
            continue

        if label not in PATHOLOGIES:
            log.warning(
                "Unknown pathology label encountered: %s",
                label,
            )
            continue

        index = PATHOLOGIES.index(
            label
        )

        target[index] = 1.0

    return target.tolist()


# ============================================================================
# Metadata loading
# ============================================================================

def find_metadata_file(
    data_root: Path,
) -> Path:
    """
    Find the ChestX-ray14 metadata CSV.

    Common names include:

        Data_Entry_2017.csv
        data_entry_2017.csv

    """

    candidates = [
        data_root / "Data_Entry_2017.csv",
        data_root / "data_entry_2017.csv",
        data_root / "metadata.csv",
        data_root / "data.csv",
    ]

    for candidate in candidates:

        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not find ChestX-ray14 metadata CSV under:\n"
        f"{data_root}\n\n"
        "Expected one of:\n"
        + "\n".join(
            str(path)
            for path in candidates
        )
    )


def resolve_image_path(
    image_name: str,
    data_root: Path,
) -> Optional[Path]:
    """
    Locate an image anywhere below data_root.

    ChestX-ray14 commonly stores images inside image subdirectories.

    We first check common locations, then recursively search.
    """

    direct_candidates = [
        data_root / image_name,
        data_root / "images" / image_name,
        data_root / "images_001" / image_name,
        data_root / "images_002" / image_name,
        data_root / "images_003" / image_name,
        data_root / "images_004" / image_name,
        data_root / "images_005" / image_name,
        data_root / "images_006" / image_name,
        data_root / "images_007" / image_name,
        data_root / "images_008" / image_name,
        data_root / "images_009" / image_name,
        data_root / "images_010" / image_name,
        data_root / "images_011" / image_name,
        data_root / "images_012" / image_name,
    ]

    for candidate in direct_candidates:

        if candidate.exists():
            return candidate

    # Recursive fallback.
    matches = list(
        data_root.rglob(image_name)
    )

    if matches:
        return matches[0]

    return None


def load_records(
    data_root: Path,
    *,
    max_samples: Optional[int] = None,
    seed: int = DEFAULT_SEED,
) -> List[ChestXrayRecord]:
    """
    Read ChestX-ray14 metadata and resolve image paths.

    max_samples is useful for local smoke testing.

    Example:

        max_samples=300

    gives approximately 100 images per client when using 3 clients.
    """

    metadata_path = find_metadata_file(
        data_root
    )

    log.info(
        "Loading metadata from %s",
        metadata_path,
    )

    dataframe = pd.read_csv(
        metadata_path
    )

    required_columns = {
        "Image Index",
        "Finding Labels",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "ChestX-ray14 metadata is missing "
            f"columns: {sorted(missing_columns)}"
        )

    # --------------------------------------------------------------
    # Reproducible sampling
    # --------------------------------------------------------------

    if max_samples is not None:

        if max_samples <= 0:
            raise ValueError(
                "max_samples must be positive."
            )

        if max_samples < len(dataframe):

            dataframe = dataframe.sample(
                n=max_samples,
                random_state=seed,
            )

    records: List[
        ChestXrayRecord
    ] = []

    missing_images = 0

    for _, row in dataframe.iterrows():

        image_name = str(
            row["Image Index"]
        )

        image_path = resolve_image_path(
            image_name,
            data_root,
        )

        if image_path is None:

            missing_images += 1

            continue

        patient_id = None

        if "Patient ID" in dataframe.columns:

            patient_id = str(
                row["Patient ID"]
            )

        labels = parse_labels(
            row["Finding Labels"]
        )

        records.append(
            ChestXrayRecord(
                image_path=str(
                    image_path
                ),
                labels=labels,
                patient_id=patient_id,
            )
        )

    if not records:
        raise RuntimeError(
            "No usable ChestX-ray14 records were found."
        )

    log.info(
        "Loaded %d usable images.",
        len(records),
    )

    if missing_images:
        log.warning(
            "%d metadata rows had missing images.",
            missing_images,
        )

    return records


# ============================================================================
# Patient-level splitting
# ============================================================================

def split_records(
    records: Sequence[ChestXrayRecord],
    *,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = DEFAULT_SEED,
) -> Tuple[
    List[ChestXrayRecord],
    List[ChestXrayRecord],
    List[ChestXrayRecord],
]:
    """
    Split records into train/validation/test.

    When Patient ID is available, splitting is performed at patient level
    to avoid placing images from the same patient into different splits.

    This is important for medical imaging evaluation.
    """

    total = (
        train_fraction
        + validation_fraction
        + test_fraction
    )

    if not math.isclose(
        total,
        1.0,
        rel_tol=1e-6,
    ):
        raise ValueError(
            "train_fraction + validation_fraction + "
            "test_fraction must equal 1."
        )

    rng = random.Random(
        seed
    )

    # --------------------------------------------------------------
    # Patient-level split
    # --------------------------------------------------------------

    patient_groups: Dict[
        str,
        List[ChestXrayRecord],
    ] = {}

    records_without_patient: List[
        ChestXrayRecord
    ] = []

    for record in records:

        if record.patient_id is None:

            records_without_patient.append(
                record
            )

            continue

        patient_groups.setdefault(
            record.patient_id,
            [],
        ).append(record)

    # If patient IDs are available for most/all records, use them.
    if patient_groups:

        patient_ids = list(
            patient_groups.keys()
        )

        rng.shuffle(
            patient_ids
        )

        total_patients = len(
            patient_ids
        )

        train_patients = int(
            total_patients
            * train_fraction
        )

        validation_patients = int(
            total_patients
            * validation_fraction
        )

        train_ids = set(
            patient_ids[
                :train_patients
            ]
        )

        validation_ids = set(
            patient_ids[
                train_patients:
                train_patients
                + validation_patients
            ]
        )

        test_ids = set(
            patient_ids[
                train_patients
                + validation_patients:
            ]
        )

        train_records = []

        validation_records = []

        test_records = []

        for patient_id, patient_records in (
            patient_groups.items()
        ):

            if patient_id in train_ids:

                train_records.extend(
                    patient_records
                )

            elif patient_id in validation_ids:

                validation_records.extend(
                    patient_records
                )

            else:

                test_records.extend(
                    patient_records
                )

        # Records without patient IDs are assigned randomly.
        rng.shuffle(
            records_without_patient
        )

        for index, record in enumerate(
            records_without_patient
        ):

            ratio = (
                index
                / max(
                    len(records_without_patient),
                    1,
                )
            )

            if ratio < train_fraction:
                train_records.append(
                    record
                )

            elif ratio < (
                train_fraction
                + validation_fraction
            ):
                validation_records.append(
                    record
                )

            else:
                test_records.append(
                    record
                )

    else:

        # ----------------------------------------------------------
        # Image-level fallback
        # ----------------------------------------------------------

        shuffled = list(
            records
        )

        rng.shuffle(
            shuffled
        )

        total_records = len(
            shuffled
        )

        train_end = int(
            total_records
            * train_fraction
        )

        validation_end = (
            train_end
            + int(
                total_records
                * validation_fraction
            )
        )

        train_records = shuffled[
            :train_end
        ]

        validation_records = shuffled[
            train_end:validation_end
        ]

        test_records = shuffled[
            validation_end:
        ]

    log.info(
        "Split sizes: train=%d validation=%d test=%d",
        len(train_records),
        len(validation_records),
        len(test_records),
    )

    return (
        train_records,
        validation_records,
        test_records,
    )


# ============================================================================
# Federated partitioning
# ============================================================================

def _primary_label(
    record: ChestXrayRecord,
) -> int:
    """
    Return one primary pathology index for Dirichlet partitioning.

    Multi-label images can contain multiple pathologies.

    For partition assignment we need a single categorical signal.

    We therefore use the first positive pathology.

    Images with no positive pathology receive a special class.
    """

    positive_indices = [
        index
        for index, value
        in enumerate(record.labels)
        if value > 0.5
    ]

    if not positive_indices:

        return NUM_CLASSES

    return positive_indices[0]


def dirichlet_partition(
    records: Sequence[ChestXrayRecord],
    *,
    num_clients: int = DEFAULT_NUM_CLIENTS,
    alpha: float = DEFAULT_DIRICHLET_ALPHA,
    seed: int = DEFAULT_SEED,
    min_samples_per_client: int = 1,
) -> Dict[int, List[ChestXrayRecord]]:
    """
    Partition records between clients using a Dirichlet distribution.

    Smaller alpha:
        more heterogeneous / non-IID

    Larger alpha:
        more homogeneous

    EveFL's intended configuration:

        alpha = 0.5
    """

    if num_clients < 1:
        raise ValueError(
            "num_clients must be >= 1."
        )

    if alpha <= 0:
        raise ValueError(
            "Dirichlet alpha must be > 0."
        )

    rng = np.random.default_rng(
        seed
    )

    # --------------------------------------------------------------
    # Group by primary pathology
    # --------------------------------------------------------------

    class_records: Dict[
        int,
        List[ChestXrayRecord],
    ] = {}

    for record in records:

        label = _primary_label(
            record
        )

        class_records.setdefault(
            label,
            [],
        ).append(record)

    # --------------------------------------------------------------
    # Allocate each pathology independently.
    # --------------------------------------------------------------

    partitions: Dict[
        int,
        List[ChestXrayRecord],
    ] = {
        client_id: []
        for client_id
        in range(num_clients)
    }

    for label_records in (
        class_records.values()
    ):

        shuffled = list(
            label_records
        )

        rng.shuffle(
            shuffled
        )

        proportions = rng.dirichlet(
            np.full(
                num_clients,
                alpha,
                dtype=np.float64,
            )
        )

        counts = (
            proportions
            * len(shuffled)
        ).astype(int)

        # Floating-point truncation can leave some records unassigned.
        remainder = (
            len(shuffled)
            - int(counts.sum())
        )

        if remainder > 0:

            order = np.argsort(
                -proportions
            )

            for index in range(
                remainder
            ):

                counts[
                    order[
                        index
                        % num_clients
                    ]
                ] += 1

        cursor = 0

        for client_id in range(
            num_clients
        ):

            count = int(
                counts[client_id]
            )

            partitions[
                client_id
            ].extend(
                shuffled[
                    cursor:
                    cursor + count
                ]
            )

            cursor += count

    # --------------------------------------------------------------
    # Guarantee minimum client size.
    # --------------------------------------------------------------

    for client_id in range(
        num_clients
    ):

        if len(
            partitions[client_id]
        ) >= min_samples_per_client:
            continue

        # Find the largest donor.
        donor = max(
            partitions.keys(),
            key=lambda cid: len(
                partitions[cid]
            ),
        )

        while (
            len(
                partitions[client_id]
            )
            < min_samples_per_client
        ):

            if not partitions[donor]:
                raise RuntimeError(
                    "Unable to satisfy minimum samples "
                    "per federated client."
                )

            record = partitions[
                donor
            ].pop()

            partitions[
                client_id
            ].append(
                record
            )

    # --------------------------------------------------------------
    # Shuffle each client's records.
    # --------------------------------------------------------------

    for client_id in partitions:

        rng.shuffle(
            partitions[client_id]
        )

        log.info(
            "Client %d receives %d samples.",
            client_id,
            len(
                partitions[
                    client_id
                ]
            ),
        )

    return partitions


# ============================================================================
# PyTorch dataset
# ============================================================================

class ChestXrayDataset(
    Dataset
):
    """
    PyTorch Dataset for ChestX-ray14.
=======
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
>>>>>>> c8d000e56cdc54490d8e8750eb80577f1165bdf3
    """

    def __init__(
        self,
<<<<<<< HEAD
        records: Sequence[ChestXrayRecord],
        *,
        transform=None,
    ):
        self.records = list(
            records
        )

        self.transform = transform

    def __len__(
        self,
    ) -> int:
        return len(
            self.records
        )

    def __getitem__(
        self,
        index: int,
    ):
        record = self.records[
            index
        ]

        image = Image.open(
            record.image_path
        ).convert(
            "RGB"
        )

        if self.transform is not None:

            image = self.transform(
                image
            )

        target = torch.tensor(
            record.labels,
            dtype=torch.float32,
        )

        return (
            image,
            target,
        )


# ============================================================================
# Transforms
# ============================================================================

def get_train_transform(
    image_size: int = DEFAULT_IMAGE_SIZE,
):
    """
    Training augmentation.
    """

    return transforms.Compose(
        [
            transforms.Resize(
                (
                    image_size,
                    image_size,
                )
            ),

            transforms.RandomHorizontalFlip(
                p=0.5
            ),

            transforms.RandomRotation(
                degrees=7
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406,
                ],
                std=[
                    0.229,
                    0.224,
                    0.225,
                ],
            ),
        ]
    )


def get_eval_transform(
    image_size: int = DEFAULT_IMAGE_SIZE,
):
    """
    Validation/test transform.
    """

    return transforms.Compose(
        [
            transforms.Resize(
                (
                    image_size,
                    image_size,
                )
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406,
                ],
                std=[
                    0.229,
                    0.224,
                    0.225,
                ],
            ),
        ]
    )


# ============================================================================
# DataLoaders
# ============================================================================

def make_dataloader(
    records: Sequence[ChestXrayRecord],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    shuffle: bool = False,
    train: bool = False,
    num_workers: int = 0,
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> DataLoader:
    """
    Create a DataLoader.
    """

    if train:

        transform = get_train_transform(
            image_size
        )

    else:

        transform = get_eval_transform(
            image_size
        )

    dataset = ChestXrayDataset(
        records,
        transform=transform,
    )
=======
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
>>>>>>> c8d000e56cdc54490d8e8750eb80577f1165bdf3

    return DataLoader(
        dataset,
        batch_size=batch_size,
<<<<<<< HEAD
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


# ============================================================================
# Federated client data
# ============================================================================

@dataclass
class ClientData:
    """
    All data belonging to one federated hospital/client.
    """

    client_id: int

    train_records: List[
        ChestXrayRecord
    ]

    validation_records: List[
        ChestXrayRecord
    ]

    train_loader: DataLoader

    validation_loader: DataLoader


def prepare_federated_data(
    data_root: Path,
    partition_root: Path,
    *,
    num_clients: int = DEFAULT_NUM_CLIENTS,
    dirichlet_alpha: float = DEFAULT_DIRICHLET_ALPHA,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_samples: Optional[int] = None,
    seed: int = DEFAULT_SEED,
    num_workers: int = 0,
) -> Dict[int, ClientData]:
    """
    Prepare the complete federated dataset.

    Workflow:

        metadata
            ↓
        records
            ↓
        train/validation/test split
            ↓
        Dirichlet partition
            ↓
        client-specific DataLoaders

    The test set remains global and is NOT assigned to individual clients.
    """

    data_root = Path(
        data_root
    )

    partition_root = Path(
        partition_root
    )

    partition_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------------
    # Load records
    # --------------------------------------------------------------

    records = load_records(
        data_root=data_root,
        max_samples=max_samples,
        seed=seed,
    )

    # --------------------------------------------------------------
    # Split
    # --------------------------------------------------------------

    (
        train_records,
        validation_records,
        test_records,
    ) = split_records(
        records,
        seed=seed,
    )

    # --------------------------------------------------------------
    # Federated partition
    # --------------------------------------------------------------

    client_partitions = (
        dirichlet_partition(
            train_records,
            num_clients=num_clients,
            alpha=dirichlet_alpha,
            seed=seed,
        )
    )

    # --------------------------------------------------------------
    # Save partition metadata
    # --------------------------------------------------------------

    partition_manifest = {
        "num_clients": num_clients,
        "dirichlet_alpha": dirichlet_alpha,
        "seed": seed,
        "total_records": len(records),
        "train_records": len(
            train_records
        ),
        "validation_records": len(
            validation_records
        ),
        "test_records": len(
            test_records
        ),
        "clients": {
            str(client_id): len(
                client_records
            )
            for client_id, client_records
            in client_partitions.items()
        },
    }

    manifest_path = (
        partition_root
        / "partition_manifest.json"
    )

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            partition_manifest,
            file,
            indent=2,
        )

    # --------------------------------------------------------------
    # Build client loaders
    # --------------------------------------------------------------

    clients: Dict[
        int,
        ClientData,
    ] = {}

    # Validation data is currently shared for the local smoke/evaluation
    # interface. The final experiment can replace this with client-specific
    # validation if required by the evaluation protocol.
    #
    # Importantly, test_records are never sent to clients.
    for client_id in range(
        num_clients
    ):

        client_train = client_partitions[
            client_id
        ]

        client_validation = list(
            validation_records
        )

        train_loader = make_dataloader(
            client_train,
            batch_size=batch_size,
            shuffle=True,
            train=True,
            num_workers=num_workers,
        )

        validation_loader = (
            make_dataloader(
                client_validation,
                batch_size=batch_size,
                shuffle=False,
                train=False,
                num_workers=num_workers,
            )
        )

        clients[
            client_id
        ] = ClientData(
            client_id=client_id,
            train_records=list(
                client_train
            ),
            validation_records=client_validation,
            train_loader=train_loader,
            validation_loader=validation_loader,
        )

    # --------------------------------------------------------------
    # Save test set separately.
    # --------------------------------------------------------------

    test_manifest = [
        {
            "image_path": record.image_path,
            "labels": record.labels,
            "patient_id": record.patient_id,
        }
        for record in test_records
    ]

    test_manifest_path = (
        partition_root
        / "global_test_manifest.json"
    )

    with test_manifest_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            test_manifest,
            file,
            indent=2,
        )

    log.info(
        "Federated data preparation complete."
    )

    return clients


# ============================================================================
# Saved partition loading
# ============================================================================

def save_client_partition(
    client_id: int,
    records: Sequence[ChestXrayRecord],
    output_dir: Path,
) -> Path:
    """
    Save one client's partition as JSON.

    This is useful on Kaggle because partitioning can be performed once and
    reused across experiments.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / f"client_{client_id}.json"
    )

    payload = [
        {
            "image_path": record.image_path,
            "labels": record.labels,
            "patient_id": record.patient_id,
        }
        for record in records
    ]

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            payload,
            file,
            indent=2,
        )

    return output_path


def load_client_partition(
    client_id: int,
    partition_root: Path,
) -> List[ChestXrayRecord]:
    """
    Load a previously saved client partition.
    """

    path = (
        partition_root
        / f"client_{client_id}.json"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Client partition does not exist: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        payload = json.load(
            file
        )

    return [
        ChestXrayRecord(
            image_path=item[
                "image_path"
            ],
            labels=item[
                "labels"
            ],
            patient_id=item.get(
                "patient_id"
            ),
        )
        for item in payload
    ]


# ============================================================================
# Dataset statistics
# ============================================================================

def calculate_label_distribution(
    records: Sequence[ChestXrayRecord],
) -> Dict[str, int]:
    """
    Calculate pathology frequencies.
    """

    counts = {
        pathology: 0
        for pathology
        in PATHOLOGIES
    }

    for record in records:

        for index, value in enumerate(
            record.labels
        ):

            if value > 0.5:

                counts[
                    PATHOLOGIES[index]
                ] += 1

    return counts


def print_partition_statistics(
    partitions: Dict[
        int,
        Sequence[ChestXrayRecord],
    ],
) -> None:
    """
    Print useful non-IID partition diagnostics.

    This should be run before the full FL experiment.
    """

    print()
    print(
        "EveFL federated partition statistics"
    )
    print(
        "===================================="
    )

    for client_id, records in (
        sorted(
            partitions.items()
        )
    ):

        distribution = (
            calculate_label_distribution(
                records
            )
        )

        print(
            f"\nClient {client_id}"
        )

        print(
            f"Samples: {len(records)}"
        )

        for pathology, count in (
            distribution.items()
        ):

            if count > 0:

                print(
                    f"  {pathology}: {count}"
                )


# ============================================================================
# Smoke-test helper
# ============================================================================

def prepare_smoke_test_data(
    data_root: Path,
    partition_root: Path,
    *,
    num_clients: int = 3,
    samples: int = 300,
    seed: int = 42,
) -> Dict[int, ClientData]:
    """
    Prepare a deliberately small dataset for Phase-4 smoke testing.

    Recommended first run:

        300 images
        3 clients
        1 local epoch
        1-2 FL rounds

    Do NOT use the full ChestX-ray14 dataset until this passes.
    """

    log.info(
        "Preparing EveFL smoke-test dataset: %d samples.",
        samples,
    )

    return prepare_federated_data(
        data_root=data_root,
        partition_root=partition_root,
        num_clients=num_clients,
        dirichlet_alpha=DEFAULT_DIRICHLET_ALPHA,
        batch_size=4,
        max_samples=samples,
        seed=seed,
        num_workers=0,
=======
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
>>>>>>> c8d000e56cdc54490d8e8750eb80577f1165bdf3
    )