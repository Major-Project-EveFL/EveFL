"""
Numbers for the dashboard's home page, read from the recorded validation files (never typed into a page).

Every value carries a label (measured, simulated or real code) and the repository file it comes from. AUC values are the macro
AUC over the classes that are not thin in the split (docs/06); QBER thresholds are fractions in [0, 1].
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
SEED0 = "docs/validation/pilot_b0_full_e1_seed0_2026-10-17.json"
SEED1 = "docs/validation/pilot_b0_full_e1_seed1_2026-10-24.json"
AUDIT = "docs/validation/partition_audit_real.json"


def _load(root: Path, rel: str) -> Dict[str, Any]:
    path = root / rel
    if not path.is_file():
        raise FileNotFoundError(f"recorded result not found: {rel} (looked in {root})")
    return json.loads(path.read_text(encoding="utf-8"))


def headline_numbers(root: Optional[Path] = None) -> Dict[str, Any]:
    from evefl.orchestration.state_machine import StateThresholds

    base = Path(root) if root is not None else REPO_ROOT
    seed1 = _load(base, SEED1)   # read first so a missing record names the newest file
    seed0 = _load(base, SEED0)
    audit = _load(base, AUDIT)
    served = seed1["demonstration_model"]
    runs = [{"seed": rec["run"]["seed"], "test_auc_round_50": rec["results"]["primary"]["test_auc_excl_thin"],
             "peak_validation_round": rec["observations"]["best_validation_auc_round"],
             "elapsed_seconds": rec["timing"]["elapsed_seconds"]} for rec in (seed0, seed1)]
    return {
        "served_model_test_auc": served["test_evaluation"]["test_auc_excl_thin"],
        "served_model_round": served["round"],
        "served_model_validation_auc": served["val_auc_excl_thin"],
        "runs": runs,
        "n_hospitals": seed1["run"]["num_clients"],
        "local_epochs": seed1["run"]["local_epochs"],
        "num_rounds": seed1["run"]["num_rounds"],
        "lockdown_bound": StateThresholds.MIN_LOCKDOWN_THRESHOLD,
        "partition_overlaps_all_zero": bool(audit["overlaps_all_zero"]),
        "labels": {"served_model_test_auc": "measured", "runs": "measured", "lockdown_bound": "real code",
                   "partition_overlaps_all_zero": "measured"},
        "sources": {"served_model": SEED1, "runs_seed0": SEED0, "audit": AUDIT,
                    "lockdown_bound": "evefl/orchestration/state_machine.py"},
    }
