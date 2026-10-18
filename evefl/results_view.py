"""
Read an experiment result into one plain structure for display (dashboard, reports).

Two file kinds exist and are understood:
- a results JSON written by `evefl.fl.server.run_experiment` (`{"experiment": {...}, "rounds": [...]}`), and
- a committed pilot record under `docs/validation/` (`{"run": {...}, "rounds": [...], "results": {...}}`).
Anything else is refused with a clear message instead of being half-shown: an older page silently displayed nothing for files
it could not read.

The loader only reads. It never recomputes a metric, never chooses a model, and carries the file's own label and caveats through
so a page can show them next to the numbers. Units: AUC in [0, 1], loss in nats, QBER as a fraction in [0, 1], times in seconds.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import pandas as pd

AUC_KEY = "macro_auc_roc_excl_thin"
ROUND_COLUMNS = ["round", "state", "system_qber", "learning_rate", "model_updated", "n_excluded",
                 "val_loss", "val_auc_star", "test_loss", "test_auc_star"]


@dataclass
class RunView:
    kind: str                                    # "results_json" or "validation_record"
    title: str
    label: str
    caveats: List[str]
    settings: Dict[str, Any]
    rounds: pd.DataFrame
    headline: List[Tuple[str, str]]
    per_class_auc: Dict[str, float] = field(default_factory=dict)
    thin_classes: List[str] = field(default_factory=list)
    timing: Optional[Dict[str, float]] = None
    checkpoints: Optional[Dict[str, Any]] = None
    warnings: List[str] = field(default_factory=list)


def _num(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else math.nan


def _fmt(value: float) -> str:
    return "n/a" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.4f}"


def classify(data: Mapping[str, Any]) -> str:
    rounds = data.get("rounds")
    if isinstance(rounds, list) and rounds and all(isinstance(r, dict) and "round" in r for r in rounds):
        if "experiment" in data and isinstance(data["experiment"], dict):
            return "results_json"
        if "run" in data and "results" in data:
            return "validation_record"
    raise ValueError("not a recognised EveFL results file (expected a results JSON with 'experiment' and 'rounds', or a "
                     "validation record with 'run', 'rounds' and 'results')")


def _frame(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in ROUND_COLUMNS:
        if column not in frame:
            frame[column] = math.nan if column not in ("state", "model_updated") else None
    return frame[ROUND_COLUMNS]


# --------------------------------------------------------------------------
# results JSON
# --------------------------------------------------------------------------

def _from_results_json(data: Mapping[str, Any]) -> RunView:
    exp: Mapping[str, Any] = data["experiment"]
    rows: List[Dict[str, Any]] = []
    last_test: Optional[Mapping[str, Any]] = None
    last_val: Optional[Mapping[str, Any]] = None
    for r in data["rounds"]:
        val, test = r.get("val_eval"), r.get("server_eval")
        if isinstance(test, dict):
            last_test = test
        if isinstance(val, dict):
            last_val = val
        rows.append({
            "round": int(r["round"]), "state": r.get("state"), "system_qber": _num(r.get("system_qber")),
            "learning_rate": _num(r.get("learning_rate")), "model_updated": r.get("model_updated"),
            "n_excluded": len(r.get("excluded_clients") or []),
            "val_loss": _num(val.get("loss")) if isinstance(val, dict) else math.nan,
            "val_auc_star": _num(val.get("metrics", {}).get(AUC_KEY)) if isinstance(val, dict) else math.nan,
            "test_loss": _num(test.get("loss")) if isinstance(test, dict) else math.nan,
            "test_auc_star": _num(test.get("metrics", {}).get(AUC_KEY)) if isinstance(test, dict) else math.nan,
        })
    frame = _frame(rows)
    warnings: List[str] = []
    if frame["val_auc_star"].isna().all():
        warnings.append("This file has no validation results, so nothing here was or can be selected on a validation split.")

    profile = exp.get("profile") or {}
    device = (profile.get("device") or {})
    devices = [d.get("name") for d in (device.get("devices") or []) if isinstance(d, dict)]
    backend = exp.get("data_backend") or {}
    evaluation = exp.get("evaluation") or {}
    settings: Dict[str, Any] = {
        "name": exp.get("name"), "config hash": exp.get("config_hash"), "seed": exp.get("seed"),
        "rounds": exp.get("num_rounds"), "clients": exp.get("num_clients"), "local epochs": exp.get("local_epochs"),
        "batch size": exp.get("batch_size"), "pretrained": exp.get("pretrained"),
        "policy mode": (exp.get("policy") or {}).get("mode"),
        "image backend": backend.get("name", "png (older file)"), "device": ", ".join(str(d) for d in devices) or device.get("default_device"),
        "elapsed seconds": exp.get("elapsed_seconds"), "state counts": exp.get("state_counts"),
        "test evaluated every": evaluation.get("test_eval_every"),
    }

    headline: List[Tuple[str, str]] = []
    tested = frame.dropna(subset=["test_auc_star"])
    if not tested.empty:
        row = tested.iloc[-1]
        headline.append((f"Last test evaluation (round {int(row['round'])})", _fmt(row["test_auc_star"])))
    valid = frame.dropna(subset=["val_auc_star"])
    if not valid.empty:
        values = [float(v) for v in valid["val_auc_star"].tolist()]
        numbers = [int(v) for v in valid["round"].tolist()]
        at = max(range(len(values)), key=lambda i: (values[i], numbers[i]))   # ties go to the later round
        headline.append((f"Best validation AUC (round {numbers[at]})", _fmt(values[at])))

    per_class = {}
    if last_test:
        per_class = {k: float(v) for k, v in (last_test.get("metrics", {}).get("per_class_auc") or {}).items()}
    thin = list((last_val or {}).get("metrics", {}).get("thin_classes") or []) if last_val else []

    timing = (profile.get("summary") or {}).get("phase_seconds")
    return RunView(
        kind="results_json", title=f"Run: {exp.get('name', 'unnamed')}",
        label=("measured; this is the file's own run. Whether it is a baseline depends on its settings (seeds, local epochs) shown below."),
        caveats=["One results file is one run: it has no confidence interval.",
                 "GPU runs with the same seed are not bit-identical; see docs/validation/image_cache_lite_run_2026-10-15.json."],
        settings=settings, rounds=frame, headline=headline, per_class_auc=per_class, thin_classes=thin,
        timing={k: float(v) for k, v in timing.items()} if isinstance(timing, dict) else None,
        checkpoints=exp.get("checkpoints"), warnings=warnings,
    )


# --------------------------------------------------------------------------
# validation record (pilot)
# --------------------------------------------------------------------------

def _from_validation_record(data: Mapping[str, Any]) -> RunView:
    run: Mapping[str, Any] = data["run"]
    results: Mapping[str, Any] = data["results"]
    rows = []
    for r in data["rounds"]:
        rows.append({
            "round": int(r["round"]), "state": "SECURE" if run.get("state_counts") == {"SECURE": run.get("num_rounds")} else None,
            "val_loss": _num(r.get("val_loss")), "val_auc_star": _num(r.get("val_auc_excl_thin")),
            "test_loss": _num(r.get("test_loss")), "test_auc_star": _num(r.get("test_auc_excl_thin")),
        })
    primary, secondary = results.get("primary", {}), results.get("secondary", {})
    headline = [
        (f"Primary: test AUC at round {primary.get('round')} (no selection)", _fmt(_num(primary.get("test_auc_excl_thin")))),
        (f"Secondary: test AUC at the validation-selected round ({secondary.get('round')})",
         _fmt(_num(secondary.get("test_auc_excl_thin")))),
    ]
    obs = data.get("observations") or {}
    if obs:
        headline.append((f"Validation AUC peak (round {obs.get('best_validation_auc_round')})", _fmt(_num(obs.get("best_validation_auc")))))
    settings = {
        "name": run.get("name"), "config hash": run.get("config_hash"), "seed": run.get("seed"), "rounds": run.get("num_rounds"),
        "clients": run.get("num_clients"), "local epochs": run.get("local_epochs"), "batch size": run.get("batch_size"),
        "image size": run.get("image_size"), "arithmetic": run.get("arithmetic"), "image backend": run.get("image_backend"),
        "elapsed seconds": (data.get("timing") or {}).get("elapsed_seconds"), "state counts": run.get("state_counts"),
        "device": (data.get("environment") or {}).get("device"),
    }
    timing = data.get("timing") or {}
    return RunView(
        kind="validation_record", title=f"Record: {run.get('name', 'unnamed')}", label=str(data.get("label", "")),
        caveats=list(data.get("caveats") or []), settings=settings, rounds=_frame(rows), headline=headline,
        per_class_auc={k: float(v) for k, v in (results.get("per_class_test_auc_round_50") or {}).items()},
        thin_classes=list(results.get("thin_classes_in_test_at_round_50") or []),
        timing={"elapsed (measured)": float(timing["elapsed_seconds"]), "projected": float(timing["projected_seconds"])}
        if "elapsed_seconds" in timing and "projected_seconds" in timing else None,
        checkpoints=None, warnings=[],
    )


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------

def load_run(source: Mapping[str, Any] | str | Path) -> RunView:
    """Load a results JSON or a pilot record from a path or an already-parsed dict; ValueError if it is neither."""
    data = source if isinstance(source, Mapping) else json.loads(Path(source).read_text(encoding="utf-8"))
    kind = classify(data)
    return _from_results_json(data) if kind == "results_json" else _from_validation_record(data)


def find_runs(repo_root: str | Path) -> List[Path]:
    """Loadable result files under `docs/validation/*.json` and `results/**/*.json`, sorted by relative path."""
    root = Path(repo_root)
    candidates = list((root / "docs" / "validation").glob("*.json")) + list((root / "results").glob("**/*.json"))
    found: List[Path] = []
    for path in candidates:
        try:
            load_run(path)
        except (ValueError, OSError, json.JSONDecodeError, KeyError, TypeError):
            continue
        found.append(path)
    return sorted(found, key=lambda p: p.relative_to(root).as_posix())
