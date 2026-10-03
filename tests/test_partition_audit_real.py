"""Internal consistency of the committed real-data partition audit (docs/validation/partition_audit_real.json).

This does not re-run the audit (the dataset is only on Kaggle); it guards the saved counts against typos.
"""

import json
from pathlib import Path

AUDIT = Path(__file__).resolve().parents[1] / "docs" / "validation" / "partition_audit_real.json"


def _audit():
    return json.loads(AUDIT.read_text(encoding="utf-8"))


def test_all_overlaps_zero_and_ten_pairs():
    overlaps = _audit()["overlaps"]
    assert len(overlaps) == 10  # C(5, 2): three hospitals, test, val
    assert all(c == {"patients": 0, "images": 0} for c in overlaps.values())


def test_patients_and_images_add_up_to_the_dataset():
    a = _audit()
    stats = a["stats"]
    assert sum(s["patients"] for s in stats.values()) == a["provenance"]["dataset_patients"]
    assert sum(s["images"] for s in stats.values()) == a["provenance"]["dataset_rows"]
    assert [stats[f"hospital_{i}"]["images"] for i in range(3)] == a["hospital_sizes"]
    assert [stats[f"hospital_{i}"]["patients"] for i in range(3)] == a["n_patients_per_client"]
    assert stats["test"]["images"] == a["n_test"] and stats["val"]["images"] == a["n_val"]
    # dominant-label groups cover only the patients assigned to hospitals (test and val are held out first)
    hospital_patients = a["provenance"]["dataset_patients"] - a["n_patients_test"] - a["n_patients_val"]
    assert sum(a["dominant_group_counts"].values()) == hospital_patients == sum(a["n_patients_per_client"])


def test_class_vectors_match_class_order_and_the_thin_class_claims():
    a = _audit()
    n_classes = len(a["class_order"])
    assert n_classes == 14
    assert all(len(s["positives_per_class"]) == n_classes for s in a["stats"].values())
    hernia = a["class_order"].index("Hernia")
    assert a["stats"]["val"]["positives_per_class"][hernia] == 10
    assert a["stats"]["hospital_0"]["positives_per_class"][hernia] == 5
    assert a["stats"]["test"]["positives_per_class"][hernia] == 53


def test_no_patient_identifiers_are_stored():
    assert _audit()["provenance"]["contains_patient_ids"] is False
