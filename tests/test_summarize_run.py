import importlib.util
from pathlib import Path


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "summarize_run.py"
    spec = importlib.util.spec_from_file_location("summarize_run_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _report():
    metrics = {"macro_auc_roc": 0.6123, "per_class_auc": {}, "skipped_classes": ["Hernia"], "n_scored_classes": 13,
               "n_test_examples": 10}
    return {
        "experiment": {"name": "t", "config_hash": "abc", "seed": 0, "num_rounds": 2, "num_clients": 3,
                       "local_epochs": 1, "batch_size": 4, "pretrained": False, "init_weights": {"source": "random_init"},
                       "qkd": {"n_qubits": 1024}, "optimizer": {"name": "adamw"}, "state_counts": {"SECURE": 2},
                       "n_failures": 0, "elapsed_seconds": 12.0},
        "rounds": [
            {"round": 1, "state": "SECURE", "system_qber": 0.0, "learning_rate": 1e-3, "model_updated": True,
             "qkd_per_client": {"0": {"qber_sample_size": 126}, "1": {"qber_sample_size": 131}},
             "server_eval": {"loss": 0.7, "metrics": metrics}, "val_eval": {"loss": 0.7, "metrics": metrics}},
            {"round": 2, "state": "LOCKDOWN", "system_qber": 0.2, "learning_rate": None, "model_updated": False},
        ],
    }


def test_summary_has_one_line_per_round_and_key_fields():
    lines = _load().summarize(_report())
    text = "\n".join(lines)
    assert "config_hash abc" in text and "126-131" in text and "0.6123" in text
    assert "LOCKDOWN" in text and "['Hernia']" in text
    assert sum(1 for line in lines if line.strip().startswith(("1 ", "2 "))) == 2


def test_missing_metrics_print_na_not_crash():
    report = _report()
    report["rounds"][0].pop("val_eval")
    assert "n/a" in "\n".join(_load().summarize(report))
