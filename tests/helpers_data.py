"""Synthetic ChestX-ray14-shaped metadata (with `Patient ID`) for local tests.

The real dataset is only available on Kaggle; these helpers produce data with the
same structure: several images per patient, multi-label findings, a large
"No Finding" majority and a long tail of rare labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from evefl.fl.model import CHESTXRAY_LABELS, NUM_CLASSES

# Rough per-image label prevalences: common -> rare (14 labels).
_PREVALENCE = np.array([0.10, 0.03, 0.12, 0.18, 0.05, 0.05, 0.013, 0.05, 0.04, 0.02, 0.02, 0.015, 0.03, 0.002])
_NO_FINDING_SHARE = 0.53


def make_metadata(n_patients: int, *, max_images: int = 6, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return (patient_ids [n_images], labels [n_images, 14]) with 1..max_images images per patient.

    A patient has a persistent "condition set"; each of their images shows a subset of
    it (or nothing), so a patient's images share labels like the real data.
    """
    rng = np.random.default_rng(seed)
    patient_ids, rows = [], []
    for pid in range(n_patients):
        if rng.random() < _NO_FINDING_SHARE:
            conditions = np.zeros(NUM_CLASSES, dtype=np.float32)
        else:
            conditions = (rng.random(NUM_CLASSES) < _PREVALENCE * 4).astype(np.float32)
            if conditions.sum() == 0:
                conditions[rng.choice(NUM_CLASSES, p=_PREVALENCE / _PREVALENCE.sum())] = 1.0
        for _ in range(int(rng.integers(1, max_images + 1))):
            keep = rng.random(NUM_CLASSES) < 0.8
            rows.append(conditions * keep)
            patient_ids.append(10_000 + pid)
    return np.array(patient_ids), np.array(rows, dtype=np.float32)


def labels_to_finding_strings(labels: np.ndarray) -> list[str]:
    out = []
    for row in labels:
        names = [CHESTXRAY_LABELS[i] for i in np.flatnonzero(row)]
        out.append("|".join(names) if names else "No Finding")
    return out


def make_metadata_frame(n_patients: int, *, max_images: int = 6, seed: int = 0) -> pd.DataFrame:
    """Data_Entry_2017.csv-shaped DataFrame ("Image Index", "Finding Labels", "Patient ID")."""
    patient_ids, labels = make_metadata(n_patients, max_images=max_images, seed=seed)
    return pd.DataFrame({
        "Image Index": [f"{pid:08d}_{i:03d}.png" for i, pid in enumerate(patient_ids)],
        "Finding Labels": labels_to_finding_strings(labels),
        "Patient ID": patient_ids,
    })
