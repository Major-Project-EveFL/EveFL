"""
P0-4 tests: patient-level, disjoint, non-IID partitioning (evefl/fl/partition.py).

Run on synthetic ChestX-ray14-shaped metadata (tests/helpers_data.py); the real
dataset is only on Kaggle, where scripts/audit_partition.py checks the real split.
"""

import json

import numpy as np
import pytest

from evefl.fl.dataset import audit_saved_partition, partition_and_save
from evefl.fl.model import NUM_CLASSES
from evefl.fl.partition import (
    NO_FINDING_GROUP,
    audit_partition,
    dirichlet_patient_partition,
    dominant_labels,
    patient_label_matrix,
    split_by_patient,
)
from tests.helpers_data import make_metadata, make_metadata_frame


def _onehot(*label_sets):
    m = np.zeros((len(label_sets), NUM_CLASSES), dtype=np.float32)
    for i, labels in enumerate(label_sets):
        for j in labels:
            m[i, j] = 1.0
    return m


# --------------------------------------------------------------------------
# Patient labels and the dominant-label definition
# --------------------------------------------------------------------------

def test_patient_labels_are_the_union_over_their_images():
    ids = np.array([7, 7, 9, 7])
    labels = _onehot({0}, {3}, {5}, set())
    unique, patient_labels, inverse = patient_label_matrix(ids, labels)
    assert unique.tolist() == [7, 9]
    assert set(np.flatnonzero(patient_labels[0])) == {0, 3}
    assert set(np.flatnonzero(patient_labels[1])) == {5}
    assert inverse.tolist() == [0, 0, 1, 0]


def test_dominant_label_is_the_rarest_positive_label():
    # label 0 is common (4 patients), label 5 is rare (1 patient)
    patient_labels = _onehot({0}, {0}, {0}, {0, 5})
    dominant = dominant_labels(patient_labels)
    assert dominant.tolist() == [0, 0, 0, 5]  # the patient with both takes the rare one


def test_no_finding_only_when_patient_has_no_positive_label():
    patient_labels = _onehot(set(), {2}, {2, 7})
    dominant = dominant_labels(patient_labels)
    assert dominant[0] == NO_FINDING_GROUP
    assert dominant[1] != NO_FINDING_GROUP and dominant[2] != NO_FINDING_GROUP


def test_dominant_label_ties_go_to_lowest_index():
    patient_labels = _onehot({4, 9}, {4}, {9})  # labels 4 and 9 are equally prevalent (2 each)
    assert dominant_labels(patient_labels)[0] == 4


# --------------------------------------------------------------------------
# Dirichlet assignment
# --------------------------------------------------------------------------

def test_every_patient_assigned_to_exactly_one_hospital():
    dominant = np.random.default_rng(0).integers(-1, 14, size=500)
    parts = dirichlet_patient_partition(dominant, 3, 0.5, np.random.default_rng(1))
    all_assigned = np.concatenate(parts)
    assert sorted(all_assigned.tolist()) == list(range(500))


def _mean_max_group_share(dominant, parts):
    """Average over groups of the largest fraction of the group held by one hospital."""
    shares = []
    for g in np.unique(dominant):
        members = set(np.flatnonzero(dominant == g).tolist())
        shares.append(max(len(members & set(p.tolist())) for p in parts) / len(members))
    return float(np.mean(shares))


def test_small_alpha_is_more_skewed_than_large_alpha():
    _, patient_labels, _ = patient_label_matrix(*make_metadata(3000, seed=3))
    dominant = dominant_labels(patient_labels)
    skewed = dirichlet_patient_partition(dominant, 3, 0.1, np.random.default_rng(0))
    near_iid = dirichlet_patient_partition(dominant, 3, 1000.0, np.random.default_rng(0))
    assert _mean_max_group_share(dominant, skewed) > 0.6
    assert _mean_max_group_share(dominant, near_iid) < 0.4


def test_skew_survives_a_dominant_no_finding_class():
    """The reason for 'rarest positive label': with ~50% No Finding patients, the
    remaining patients must still be spread over many rare-label groups."""
    ids, labels = make_metadata(3000, seed=5)
    _, patient_labels, _ = patient_label_matrix(ids, labels)
    dominant = dominant_labels(patient_labels)
    groups, counts = np.unique(dominant, return_counts=True)
    finding = counts[groups != NO_FINDING_GROUP]
    assert (groups != NO_FINDING_GROUP).sum() >= 10  # many distinct dominant labels
    assert finding.sum() / counts.sum() > 0.3  # substantial share of non-"No Finding" patients


def test_too_few_patients_raises():
    with pytest.raises(ValueError):
        dirichlet_patient_partition(np.array([0, 1]), 3, 0.5, np.random.default_rng(0))


def test_impossible_min_patients_raises_runtime_error():
    dominant = np.zeros(6, dtype=int)  # one group, alpha tiny -> a hospital is always empty
    with pytest.raises(RuntimeError):
        dirichlet_patient_partition(dominant, 3, 0.001, np.random.default_rng(0),
                                    min_patients_per_client=2, max_attempts=5)


@pytest.mark.parametrize("bad_alpha", [0.0, -1.0])
def test_non_positive_alpha_rejected(bad_alpha):
    with pytest.raises(ValueError):
        dirichlet_patient_partition(np.zeros(10, dtype=int), 3, bad_alpha, np.random.default_rng(0))


# --------------------------------------------------------------------------
# End-to-end split: disjointness (the P0-4 bug) and bookkeeping
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dataset():
    return make_metadata(1500, max_images=6, seed=11)


@pytest.fixture(scope="module")
def split(dataset):
    ids, labels = dataset
    return split_by_patient(ids, labels, n_clients=3, alpha=0.5, test_fraction=0.1, seed=42)


def test_hospitals_and_test_share_no_patients_and_no_images(dataset, split):
    ids, labels = dataset
    audit = audit_partition(ids, labels, split.client_indices, split.test_indices)
    assert audit["ok"], audit["overlaps"]
    for pair, counts in audit["overlaps"].items():
        assert counts == {"patients": 0, "images": 0}, pair


def test_a_patients_images_all_land_in_the_same_place(dataset, split):
    ids, _ = dataset
    groups = [split.test_indices] + list(split.client_indices)
    homes = {}
    for g, idx in enumerate(groups):
        for pid in np.unique(ids[idx]):
            assert pid not in homes, f"patient {pid} appears in groups {homes[pid]} and {g}"
            homes[pid] = g
    for pid, g in homes.items():  # and none of their images are elsewhere
        assert set(np.flatnonzero(ids == pid).tolist()) <= set(groups[g].tolist())


def test_full_coverage_when_using_all_patients(dataset, split):
    ids, _ = dataset
    everything = np.concatenate([split.test_indices] + list(split.client_indices))
    assert sorted(everything.tolist()) == list(range(len(ids)))


def test_test_split_holds_about_test_fraction_of_patients(dataset, split):
    assert split.n_patients_test == round(split.n_patients_used * 0.1)


def test_every_hospital_gets_patients(split):
    assert all(n > 0 for n in split.n_patients_per_client)
    assert sum(split.n_patients_per_client) + split.n_patients_test == split.n_patients_used


def test_split_is_deterministic_and_seed_dependent(dataset):
    ids, labels = dataset
    a = split_by_patient(ids, labels, n_clients=3, alpha=0.5, test_fraction=0.1, seed=1)
    b = split_by_patient(ids, labels, n_clients=3, alpha=0.5, test_fraction=0.1, seed=1)
    c = split_by_patient(ids, labels, n_clients=3, alpha=0.5, test_fraction=0.1, seed=2)
    assert all(np.array_equal(x, y) for x, y in zip(a.client_indices, b.client_indices))
    assert not all(np.array_equal(x, y) for x, y in zip(a.client_indices, c.client_indices))


def test_subset_fraction_is_applied_to_patients_and_stays_disjoint(dataset):
    ids, labels = dataset
    s = split_by_patient(ids, labels, n_clients=3, alpha=0.5, test_fraction=0.1, seed=0, subset_fraction=0.2)
    assert s.n_patients_used == round(s.n_patients_total * 0.2)
    used_images = np.concatenate([s.test_indices] + list(s.client_indices))
    assert len(np.unique(ids[used_images])) == s.n_patients_used  # whole patients only
    assert audit_partition(ids, labels, s.client_indices, s.test_indices)["ok"]


def test_multilabel_images_are_not_duplicated_across_hospitals(dataset, split):
    """The old class-wise scheme added a multi-label image to every hospital that got any of its
    positive classes. Each image must now appear in exactly one hospital."""
    all_client_images = np.concatenate(split.client_indices)
    assert len(all_client_images) == len(np.unique(all_client_images))


@pytest.mark.parametrize("kwargs", [{"test_fraction": 1.0}, {"test_fraction": -0.1}, {"subset_fraction": 0.0}])
def test_invalid_fractions_rejected(dataset, kwargs):
    ids, labels = dataset
    params = {"test_fraction": 0.1, "subset_fraction": 1.0, **kwargs}
    with pytest.raises(ValueError):
        split_by_patient(ids, labels, n_clients=3, alpha=0.5, seed=0, **params)


# --------------------------------------------------------------------------
# audit_partition catches leakage
# --------------------------------------------------------------------------

def test_audit_detects_patient_leakage_between_train_and_test(dataset, split):
    ids, labels = dataset
    stolen_patient = ids[split.client_indices[0][0]]
    leaked_image = np.flatnonzero(ids == stolen_patient)[0]
    test_with_leak = np.append(split.test_indices, leaked_image)
    audit = audit_partition(ids, labels, split.client_indices, test_with_leak)
    assert audit["ok"] is False
    assert audit["overlaps"]["hospital_0~test"]["patients"] >= 1


def test_audit_detects_image_overlap_between_hospitals(dataset, split):
    ids, labels = dataset
    dup = np.append(split.client_indices[1], split.client_indices[0][0])
    audit = audit_partition(ids, labels, [split.client_indices[0], dup, split.client_indices[2]], split.test_indices)
    assert audit["ok"] is False
    assert audit["overlaps"]["hospital_0~hospital_1"]["images"] == 1


def test_audit_reports_label_statistics(dataset, split):
    ids, labels = dataset
    stats = audit_partition(ids, labels, split.client_indices, split.test_indices)["stats"]
    assert set(stats) == {"hospital_0", "hospital_1", "hospital_2", "test"}
    assert len(stats["hospital_0"]["positives_per_class"]) == NUM_CLASSES
    assert 0.0 <= stats["hospital_0"]["no_finding_share"] <= 1.0


# --------------------------------------------------------------------------
# partition_and_save round trip (CSV on disk, no images needed)
# --------------------------------------------------------------------------

def test_partition_and_save_round_trip(tmp_path):
    make_metadata_frame(400, seed=2).to_csv(tmp_path / "Data_Entry_2017.csv", index=False)
    out = tmp_path / "partitions"
    partition_and_save(tmp_path, out, n_clients=3, alpha=0.5, test_fraction=0.1, seed=7)

    for name in ("hospital_0", "hospital_1", "hospital_2", "test"):
        assert (out / name / "indices.npy").exists()
    meta = json.loads((out / "partition_meta.json").read_text())
    assert meta["scheme"] == "patient_level_dirichlet_rarest_label"
    assert meta["audit"]["ok"] is True

    audit = audit_saved_partition(tmp_path, out)
    assert audit["ok"] is True


def test_partition_and_save_requires_patient_id_column(tmp_path):
    frame = make_metadata_frame(50, seed=0).drop(columns=["Patient ID"])
    frame.to_csv(tmp_path / "Data_Entry_2017.csv", index=False)
    with pytest.raises(KeyError, match="Patient ID"):
        partition_and_save(tmp_path, tmp_path / "partitions")


# --------------------------------------------------------------------------
# Determinism and index-file hashes (a rerun with the same seed must be provably identical)
# --------------------------------------------------------------------------

def _index_files(root):
    return sorted(p.relative_to(root).as_posix() for p in root.glob("*/indices.npy"))


def test_partition_and_save_is_deterministic_for_the_same_seed(tmp_path):
    make_metadata_frame(400, seed=2).to_csv(tmp_path / "Data_Entry_2017.csv", index=False)
    runs = []
    for name in ("run_a", "run_b"):
        out = tmp_path / name
        partition_and_save(tmp_path, out, n_clients=3, alpha=0.5, test_fraction=0.1, seed=7)
        runs.append(out)
    files = _index_files(runs[0])
    assert files == _index_files(runs[1]) and len(files) == 5  # 3 hospitals + test + val
    for rel in files:
        assert np.array_equal(np.load(runs[0] / rel), np.load(runs[1] / rel)), rel
    meta_a = json.loads((runs[0] / "partition_meta.json").read_text())
    meta_b = json.loads((runs[1] / "partition_meta.json").read_text())
    assert meta_a["index_sha256"] == meta_b["index_sha256"]


def test_different_seed_changes_the_split_and_the_hashes(tmp_path):
    make_metadata_frame(400, seed=2).to_csv(tmp_path / "Data_Entry_2017.csv", index=False)
    partition_and_save(tmp_path, tmp_path / "s7", n_clients=3, seed=7)
    partition_and_save(tmp_path, tmp_path / "s8", n_clients=3, seed=8)
    h7 = json.loads((tmp_path / "s7" / "partition_meta.json").read_text())["index_sha256"]
    h8 = json.loads((tmp_path / "s8" / "partition_meta.json").read_text())["index_sha256"]
    assert h7 != h8


def test_meta_stores_sha256_of_every_index_file_and_verify_detects_tampering(tmp_path):
    from evefl.fl.dataset import verify_partition_hashes

    make_metadata_frame(400, seed=2).to_csv(tmp_path / "Data_Entry_2017.csv", index=False)
    out = tmp_path / "partitions"
    partition_and_save(tmp_path, out, n_clients=3, seed=7)
    meta = json.loads((out / "partition_meta.json").read_text())
    assert set(meta["index_sha256"]) == {"hospital_0", "hospital_1", "hospital_2", "test", "val"}
    assert all(len(h) == 64 for h in meta["index_sha256"].values())
    assert verify_partition_hashes(out) == {}  # nothing mismatched

    idx = np.load(out / "test" / "indices.npy")
    np.save(out / "test" / "indices.npy", idx[::-1].copy())  # same members, different order: still a different file
    assert list(verify_partition_hashes(out)) == ["test"]
