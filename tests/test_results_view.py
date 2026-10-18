"""
The results explorer (`evefl/results_view.py` and `dashboard/pages/4_fl_results.py`).

The old page read a file format that no run produces any more, so it showed nothing. These tests pin the loader to the formats that
exist: the results JSON written by `run_experiment` (checked on a real tiny run, so the schema cannot drift unnoticed) and the
committed pilot record under docs/validation.
"""

import json
import math
from pathlib import Path

import pytest

from evefl import results_view as rv

REPO = Path(__file__).resolve().parents[1]
PILOT = REPO / "docs" / "validation" / "pilot_b0_full_e1_seed0_2026-10-17.json"
AUC = "macro_auc_roc_excl_thin"


def _metrics(auc, per_class=None, thin=None):
    return {AUC: auc, "per_class_auc": per_class or {}, "thin_classes": thin or []}


def _synthetic_results():
    rounds = []
    for n, (state, qber, val_auc, test_auc) in enumerate(
            [("SECURE", 0.01, 0.60, None), ("CAUTION", 0.07, 0.70, 0.72), ("LOCKDOWN", 0.20, 0.70, None)], start=1):
        entry = {"round": n, "state": state, "system_qber": qber, "learning_rate": 1e-3 / n, "model_updated": state != "LOCKDOWN",
                 "excluded_clients": ["1"] if state == "LOCKDOWN" else [],
                 "val_eval": {"loss": 0.2 - 0.01 * n, "metrics": _metrics(val_auc, thin=["Hernia"])},
                 "timing": {"round_s": 5.0, "phases": {"fit": {"phase_s": 4.0, "data_wait_s": 0.2, "compute_s": 3.8}}}}
        if test_auc is not None:
            entry["server_eval"] = {"loss": 0.19, "metrics": _metrics(test_auc, per_class={"Atelectasis": 0.8, "Hernia": 0.7})}
        rounds.append(entry)
    return {
        "experiment": {
            "name": "synthetic", "config_hash": "abc123", "seed": 3, "num_rounds": 3, "num_clients": 3, "local_epochs": 1,
            "batch_size": 4, "pretrained": False, "elapsed_seconds": 15.0, "state_counts": {"SECURE": 1, "CAUTION": 1, "LOCKDOWN": 1},
            "policy": {"mode": "global"}, "data_backend": {"name": "png", "cache_sha256": None},
            "evaluation": {"federated_evaluate": False, "test_eval_every": 2, "val_eval_every": 1},
            "profile": {"device": {"default_device": "cpu", "devices": []},
                        "summary": {"phase_seconds": {"fit": 12.0}, "data_wait_share_loaded": 0.05}},
            "checkpoints": {"final": {"file": "x_final.pt", "round": 3}, "best_validation": {"file": "x_best_val.pt", "round": 2}},
        },
        "rounds": rounds,
    }


# --------------------------------------------------------------------------
# the committed pilot record
# --------------------------------------------------------------------------

def test_the_pilot_record_loads_with_its_label_caveats_and_headline_numbers():
    view = rv.load_run(PILOT)
    assert view.kind == "validation_record"
    assert "single-seed pilot" in view.label
    assert any("not the paper baseline" in c for c in view.caveats)
    assert len(view.rounds) == 50 and list(view.rounds["round"])[:3] == [1, 2, 3]
    assert view.rounds["test_auc_star"].notna().sum() == 10
    headline = dict(view.headline)
    assert headline["Primary: test AUC at round 50 (no selection)"] == "0.8086"
    assert headline["Secondary: test AUC at the validation-selected round (25)"] == "0.8316"
    assert headline["Validation AUC peak (round 26)"] == "0.8230"
    assert len(view.per_class_auc) == 14 and view.thin_classes == []
    assert view.settings["local epochs"] == 1 and view.settings["seed"] == 0


# --------------------------------------------------------------------------
# a results JSON
# --------------------------------------------------------------------------

def test_a_results_json_gives_states_qber_exclusions_and_validation_and_test_columns():
    view = rv.load_run(_synthetic_results())
    assert view.kind == "results_json" and list(view.rounds["state"]) == ["SECURE", "CAUTION", "LOCKDOWN"]
    assert list(view.rounds["n_excluded"]) == [0, 0, 1]
    assert view.rounds["system_qber"].tolist() == [0.01, 0.07, 0.20]
    assert view.rounds["val_auc_star"].tolist() == [0.60, 0.70, 0.70]
    assert math.isnan(view.rounds["test_auc_star"].iloc[0]) and view.rounds["test_auc_star"].iloc[1] == 0.72
    assert view.settings["config hash"] == "abc123" and view.settings["image backend"] == "png"
    assert view.timing == {"fit": 12.0} and view.checkpoints["best_validation"]["round"] == 2


def test_headline_uses_the_last_test_evaluation_and_the_best_validation_round_with_ties_to_the_later():
    headline = dict(rv.load_run(_synthetic_results()).headline)
    assert headline["Last test evaluation (round 2)"] == "0.7200"
    assert headline["Best validation AUC (round 3)"] == "0.7000"
    assert rv.load_run(_synthetic_results()).per_class_auc == {"Atelectasis": 0.8, "Hernia": 0.7}


def test_a_file_without_validation_results_still_loads_and_says_so():
    data = _synthetic_results()
    for entry in data["rounds"]:
        entry.pop("val_eval")
    view = rv.load_run(data)
    assert view.rounds["val_auc_star"].isna().all() and any("validation" in w.lower() for w in view.warnings)


def test_an_unrecognised_file_is_refused_with_a_clear_message():
    with pytest.raises(ValueError, match="not a recognised"):
        rv.load_run({"security": {"rounds": []}})
    with pytest.raises(ValueError):
        rv.load_run({"experiment": {}, "rounds": [{"no_round_key": 1}]})


def test_find_runs_lists_only_loadable_files_in_a_stable_order(tmp_path):
    (tmp_path / "results" / "kaggle").mkdir(parents=True)
    (tmp_path / "docs" / "validation").mkdir(parents=True)
    (tmp_path / "results" / "kaggle" / "good.json").write_text(json.dumps(_synthetic_results()), encoding="utf-8")
    (tmp_path / "results" / "kaggle" / "bad.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "results" / "other.json").write_text(json.dumps({"hello": 1}), encoding="utf-8")
    (tmp_path / "docs" / "validation" / PILOT.name).write_text(PILOT.read_text(encoding="utf-8"), encoding="utf-8")
    found = rv.find_runs(tmp_path)
    assert [p.relative_to(tmp_path).as_posix() for p in found] == [
        f"docs/validation/{PILOT.name}", "results/kaggle/good.json"]


# --------------------------------------------------------------------------
# the real schema, from a real tiny run
# --------------------------------------------------------------------------

def test_a_real_results_file_from_run_experiment_loads(synthetic_data, tmp_path):
    from evefl.fl.server import run_experiment

    data_root, partition_root = synthetic_data
    out = tmp_path / "real.json"
    run_experiment(data_root=data_root, partition_root=partition_root, num_clients=3, num_rounds=2, local_epochs=1, batch_size=4,
                   n_qubits=1024, intercept_probability=0.0, seed=0, experiment_name="real_tiny", output_path=out,
                   pretrained=False, test_eval_every=2)
    view = rv.load_run(out)
    assert view.kind == "results_json" and len(view.rounds) == 2
    assert list(view.rounds["state"]) == ["SECURE", "SECURE"]
    assert view.rounds["val_loss"].notna().all() and view.rounds["test_loss"].notna().sum() == 1
    assert view.settings["name"] == "real_tiny" and view.timing and "fit" in view.timing


# --------------------------------------------------------------------------
# the page renders
# --------------------------------------------------------------------------

def test_the_page_renders_the_pilot_by_default_with_its_warning_and_no_exception():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(REPO / "dashboard" / "pages" / "4_fl_results.py"), default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    texts = " ".join(str(w.value) for w in list(at.warning) + list(at.info) + list(at.caption))
    assert "single-seed pilot" in texts
    assert any("0.8086" in str(m.value) for m in at.metric)
