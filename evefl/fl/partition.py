"""
Patient-level, disjoint, non-IID partitioning of ChestX-ray14 (P0-4).

Why this exists
---------------
ChestX-ray14 has ~4 images per patient (112,120 images, 30,805 patients). The
previous partitioner split *images*, so the same patient's images could land in
train, test and several hospitals (leakage that inflates AUC), and a multi-label
image was added to every client that received any of its positive classes, so
hospitals overlapped. This module instead assigns WHOLE PATIENTS.

Scheme (this is what the paper must describe)
---------------------------------------------
1. Group images by `Patient ID`. A patient's label vector is the union of the
   labels over all their images.
2. Held-out test set: a random `test_fraction` of *patients* (IID w.r.t. label),
   disjoint from every hospital. A validation set (`val_fraction` of patients, also disjoint from the
   test set and every hospital) is held out the same way: it is for model selection (choosing rounds,
   learning rates, thresholds). The test set is for the final report ONLY and must never be used to pick
   anything.
3. Each remaining patient gets ONE dominant label:
     * the patient's RAREST positive label, where rarity is the number of
       patients (in the pool being partitioned) positive for that label;
       ties go to the lower label index;
     * "No Finding" (group id `NO_FINDING_GROUP`) only if the patient has no
       positive label at all.
   Using the rarest label keeps "No Finding" and the very common labels from
   swamping the group structure, so the Dirichlet skew stays visible.
4. For each dominant-label group g, draw proportions p_g ~ Dir(alpha * 1_K) over
   the K hospitals and assign that group's patients to hospitals accordingly.
   Small alpha => a hospital sees most patients of a given group (strong skew);
   large alpha => near IID. Redrawn (up to `max_attempts`) until every hospital
   has at least `min_patients_per_client` patients.
5. Every image follows its patient. Hospitals and test are pairwise disjoint at
   both patient and image level (`audit_partition` verifies this).

Only numpy here: no torch, no images, no quantum code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

NO_FINDING_GROUP = -1  # dominant-label id for patients with no positive label


@dataclass
class PatientSplit:
    """Row indices (into the metadata CSV) for each hospital and the test set."""

    client_indices: List[np.ndarray]
    test_indices: np.ndarray
    n_patients_total: int
    n_patients_used: int
    n_patients_test: int
    n_patients_per_client: List[int]
    dominant_group_counts: Dict[int, int]  # over the patients partitioned into hospitals
    val_indices: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    n_patients_val: int = 0


def patient_label_matrix(patient_ids: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse image-level labels to patient level (union over a patient's images).

    Returns (unique_patient_ids, patient_labels [P, C], inverse [n_images]) where
    `inverse[i]` is the row of `unique_patient_ids` that image i belongs to.
    """
    unique_ids, inverse = np.unique(patient_ids, return_inverse=True)
    inverse = inverse.reshape(-1)
    patient_labels = np.zeros((len(unique_ids), labels.shape[1]), dtype=labels.dtype)
    np.maximum.at(patient_labels, inverse, labels)
    return unique_ids, patient_labels, inverse


def dominant_labels(patient_labels: np.ndarray) -> np.ndarray:
    """Dominant label per patient: rarest positive label, else `NO_FINDING_GROUP`.

    Rarity = number of patients in `patient_labels` positive for the label; ties
    resolve to the lowest label index.
    """
    prevalence = patient_labels.sum(axis=0).astype(np.float64)
    # Labels a patient does not have get +inf, so argmin only considers their own positives.
    cost = np.where(patient_labels > 0, prevalence[None, :], np.inf)
    dominant = np.argmin(cost, axis=1)
    has_positive = (patient_labels > 0).any(axis=1)
    return np.where(has_positive, dominant, NO_FINDING_GROUP)


def dirichlet_patient_partition(
    dominant: np.ndarray,
    n_clients: int,
    alpha: float,
    rng: np.random.Generator,
    *,
    min_patients_per_client: int = 1,
    max_attempts: int = 100,
) -> List[np.ndarray]:
    """Assign patients (by position in `dominant`) to hospitals, group-wise Dir(alpha).

    Returns one array of patient positions per hospital; together they cover every
    patient exactly once.
    """
    if n_clients < 1:
        raise ValueError("n_clients must be >= 1")
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    if len(dominant) < n_clients * min_patients_per_client:
        raise ValueError(
            f"Only {len(dominant)} patients available for {n_clients} hospitals "
            f"(min {min_patients_per_client} each)."
        )

    groups = [np.flatnonzero(dominant == g) for g in np.unique(dominant)]

    for _ in range(max_attempts):
        buckets: List[List[int]] = [[] for _ in range(n_clients)]
        for members in groups:
            members = rng.permutation(members)
            proportions = rng.dirichlet(alpha * np.ones(n_clients))
            cuts = np.round(np.cumsum(proportions) * len(members)).astype(int)[:-1]
            for client_id, chunk in enumerate(np.split(members, cuts)):
                buckets[client_id].extend(int(p) for p in chunk)
        if all(len(b) >= min_patients_per_client for b in buckets):
            return [np.array(sorted(b), dtype=np.int64) for b in buckets]

    raise RuntimeError(
        f"Could not give every hospital >= {min_patients_per_client} patients in {max_attempts} "
        f"attempts (alpha={alpha}). Increase alpha, use more data, or lower min_patients_per_client."
    )


def split_by_patient(
    patient_ids: np.ndarray,
    labels: np.ndarray,
    *,
    n_clients: int,
    alpha: float,
    test_fraction: float,
    seed: int,
    subset_fraction: float = 1.0,
    min_patients_per_client: int = 1,
    val_fraction: float = 0.0,
) -> PatientSplit:
    """Full pipeline: subset -> test split -> validation split -> Dirichlet hospital split, all by patient."""
    if not 0.0 <= test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in [0, 1), got {test_fraction}")
    if not 0.0 <= val_fraction < 1.0 or test_fraction + val_fraction >= 1.0:
        raise ValueError(
            f"val_fraction must be in [0, 1) and test_fraction + val_fraction < 1, got "
            f"val_fraction={val_fraction}, test_fraction={test_fraction}"
        )
    if not 0.0 < subset_fraction <= 1.0:
        raise ValueError(f"subset_fraction must be in (0, 1], got {subset_fraction}")
    if len(patient_ids) != len(labels):
        raise ValueError("patient_ids and labels must have one row per image.")

    rng = np.random.default_rng(seed)
    unique_ids, patient_labels, inverse = patient_label_matrix(patient_ids, labels)
    n_total = len(unique_ids)

    order = rng.permutation(n_total)
    n_used = max(1, int(round(n_total * subset_fraction)))
    order = order[:n_used]

    n_test = int(round(n_used * test_fraction))
    if test_fraction > 0:
        n_test = max(1, n_test)
    test_pos = order[:n_test]
    n_val = int(round(n_used * val_fraction))
    if val_fraction > 0:
        n_val = max(1, n_val)
    val_pos = order[n_test:n_test + n_val]
    train_pos = order[n_test + n_val:]

    dominant = dominant_labels(patient_labels[train_pos])
    local_partition = dirichlet_patient_partition(
        dominant, n_clients, alpha, rng, min_patients_per_client=min_patients_per_client
    )
    client_patient_pos = [np.sort(train_pos[local]) for local in local_partition]

    def rows_of(patient_positions: np.ndarray) -> np.ndarray:
        return np.flatnonzero(np.isin(inverse, patient_positions))

    groups, counts = np.unique(dominant, return_counts=True)
    return PatientSplit(
        client_indices=[rows_of(pos) for pos in client_patient_pos],
        test_indices=rows_of(test_pos),
        val_indices=rows_of(val_pos),
        n_patients_val=int(n_val),
        n_patients_total=n_total,
        n_patients_used=n_used,
        n_patients_test=int(n_test),
        n_patients_per_client=[int(len(p)) for p in client_patient_pos],
        dominant_group_counts={int(g): int(c) for g, c in zip(groups, counts)},
    )


def audit_partition(
    patient_ids: np.ndarray,
    labels: np.ndarray,
    client_indices: Sequence[np.ndarray],
    test_indices: np.ndarray,
    val_indices: Optional[np.ndarray] = None,
) -> dict:
    """Overlap counts and label statistics for a saved partition.

    Every `overlaps` count must be 0 for a valid patient-level split; `ok` is True
    only then. Also reports per-hospital size, share of "No Finding" images and
    per-class positive counts, so the Dirichlet skew can be eyeballed.
    """
    names = [f"hospital_{i}" for i in range(len(client_indices))] + ["test"]
    held_out = [test_indices]
    if val_indices is not None:
        names.append("val")
        held_out.append(val_indices)
    sets = [np.asarray(idx, dtype=np.int64) for idx in list(client_indices) + held_out]
    patient_sets = [set(np.unique(patient_ids[idx]).tolist()) for idx in sets]
    image_sets = [set(idx.tolist()) for idx in sets]

    overlaps = {}
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            overlaps[f"{names[a]}~{names[b]}"] = {
                "patients": len(patient_sets[a] & patient_sets[b]),
                "images": len(image_sets[a] & image_sets[b]),
            }

    stats = {}
    for name, idx in zip(names, sets):
        lab = labels[idx]
        stats[name] = {
            "images": int(len(idx)),
            "patients": int(len(np.unique(patient_ids[idx]))),
            "no_finding_share": float((lab.sum(axis=1) == 0).mean()) if len(idx) else float("nan"),
            "positives_per_class": lab.sum(axis=0).astype(int).tolist(),
        }

    return {
        "ok": all(v["patients"] == 0 and v["images"] == 0 for v in overlaps.values()),
        "overlaps": overlaps,
        "stats": stats,
    }
