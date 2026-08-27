"""
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
"""

from __future__ import annotations

import json
import logging
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
    """

    def __init__(
        self,
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

    return DataLoader(
        dataset,
        batch_size=batch_size,
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
    )