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
               "thin_classes": ["Hernia"], "macro_auc_roc_excl_thin": 0.6555,
               "n_test_examples": 10}
    return {
        "experiment": {"name": "t", "config_hash": "abc", "seed": 0, "num_rounds": 2, "num_clients": 3,
                       "local_epochs": 1, "batch_size": 4, "pretrained": False, "init_weights": {"source": "random_init"},
                       "qkd": {"n_qubits": 1024}, "optimizer": {"name": "adamw"}, "state_counts": {"SECURE": 2},
                       "n_failures": 0, "elapsed_seconds": 12.0,
                       "policy": {"mode": "per_client", "min_clients": 1, "hysteresis_margin": 0.0, "hysteresis_dwell": 1},
                       "channel_plan": {"default_noise": 0.01, "link_noise": {},
                                        "attacks": {"0": {"alpha": 0.6, "kind": "constant"}}},
                       "rounds_discarded": 1, "exclusion_counts": {"lockdown": 2, "no_key": 1},
                       "participation_actual": {"0": 1, "1": 2, "2": 2},
                       "environment": {"python": "3.13.15", "system": "Linux", "machine": "x86_64",
                                       "libraries": {"torch": "2.11.0+cu128", "numpy": "2.5.3", "flwr": "1.13.0",
                                                     "qiskit": None}}},
        "rounds": [
            {"round": 1, "state": "SECURE", "system_qber": 0.0, "learning_rate": 1e-3, "model_updated": True,
             "excluded_clients": ["0"], "qkd_per_client": {"0": {"qber_sample_size": 126}, "1": {"qber_sample_size": 131}},
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


def test_summary_shows_policy_channel_plan_exclusions_and_participation():
    lines = _load().summarize(_report())
    text = "\n".join(lines)
    assert "per_client" in text and "min_clients=1" in text
    assert "attacks on links: {'0': 0.6}" in text
    assert "rounds discarded: 1" in text
    assert "no_key=1" in text and "lockdown=2" in text
    assert "participation   : {'0': 1, '1': 2, '2': 2}" in text
    round1 = next(line for line in lines if line.strip().startswith("1 "))
    assert "[0]" in round1  # excluded column lists the excluded client ids


def test_old_results_without_policy_fields_still_summarise():
    report = _report()
    for key in ("policy", "channel_plan", "rounds_discarded", "exclusion_counts", "participation_actual", "environment"):
        report["experiment"].pop(key)
    report["rounds"][0].pop("excluded_clients")
    text = "\n".join(_load().summarize(report))
    assert "policy          : n/a" in text and "config_hash abc" in text
    assert "environment     : n/a" in text


def test_summary_shows_selection_auc_and_thin_classes():
    text = "\n".join(_load().summarize(_report()))
    assert "val_AUC*" in text and "0.6555" in text
    assert "thin classes in the validation AUC (last evaluated round): ['Hernia']" in text


def test_summary_shows_the_recorded_environment_and_omits_missing_packages():
    text = "\n".join(_load().summarize(_report()))
    assert "environment     : python 3.13.15 Linux/x86_64" in text
    assert "torch 2.11.0+cu128" in text and "numpy 2.5.3" in text and "flwr 1.13.0" in text
    assert "qiskit" not in text


def _profiled_report():
    report = _report()
    report["experiment"]["evaluation"] = {"federated_evaluate": False, "test_eval_every": 5, "val_eval_every": 1}
    report["experiment"]["profile"] = {
        "device": {"cuda_available": True, "default_device": "cuda", "device_count": 2, "cpu_count": 4,
                   "devices": [{"index": 0, "name": "Tesla T4", "total_memory_mib": 15360},
                               {"index": 1, "name": "Tesla T4", "total_memory_mib": 15360}]},
        "dataloader": {"num_workers": 0, "batch_size": 32, "pin_memory": True},
        "gpu": {"available": True, "n_samples": 2, "samples": [
            [1.0, [{"index": 0, "util_pct": 10.0, "mem_mib": 100.0}, {"index": 1, "util_pct": 0.0, "mem_mib": 0.0}]],
            [3.0, [{"index": 0, "util_pct": 30.0, "mem_mib": 100.0}, {"index": 1, "util_pct": 0.0, "mem_mib": 0.0}]]]},
        "summary": {"data_wait_share_loaded": 0.71, "phase_seconds": {"fit": 99.0, "server_eval_test": 15.0},
                    "images_per_s": {"fit": 43.2, "server_eval_test": 44.7},
                    "rule_50_percent": {"threshold": 0.5, "data_wait_share": 0.71, "data_wait_exceeds_threshold": True}},
    }
    return report


def test_summary_shows_device_evaluation_settings_and_the_data_wait_share():
    text = "\n".join(_load().summarize(_profiled_report()))
    assert "evaluation      : federated=False test_every=5 val_every=1" in text
    assert "device          : cuda: 2 x Tesla T4 (15360 MiB), cpu_count=4" in text
    assert "data wait 71% of the loaded phases" in text and "exceeds 50%: True" in text
    assert "fit 43.2" in text and "server_eval_test 44.7" in text
    assert "gpu util mean (all samples) GPU0 20%, GPU1 0%" in text


def test_summary_of_a_cpu_run_and_of_an_old_result_without_profile():
    report = _profiled_report()
    report["experiment"]["profile"]["device"] = {"cuda_available": False, "default_device": "cpu", "device_count": 0,
                                                 "cpu_count": 4, "devices": []}
    report["experiment"]["profile"]["gpu"] = {"available": False, "n_samples": 0, "samples": []}
    text = "\n".join(_load().summarize(report))
    assert "device          : cpu only (cpu_count=4)" in text and "no gpu samples" in text
    old = "\n".join(_load().summarize(_report()))
    assert "device          : n/a" in old and "profile         : n/a" in old and "evaluation      : n/a" in old
