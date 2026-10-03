"""
Tests for the QKD config guard (meaningless sample sizes) and the strategy's use of the
exact numpy backend: experiment-seed dependence, baseline noise, sim-only ground truth.
"""

import logging
from pathlib import Path

import numpy as np
import pytest
from flwr.common import ndarrays_to_parameters

from evefl.fl.server import build_experiment_log, run_experiment
from evefl.fl.strategy import EveFLStrategy
from evefl.quantum.config import PRESETS, QKDConfig
from tests.test_screening import _Manager, _Proxy

# --------------------------------------------------------------------------
# QKDConfig guard
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n_qubits", [16, 64, 128, 256])
def test_tiny_samples_are_refused(n_qubits):
    """n=16 (evefl.ipynb) gives a ~2-bit QBER sample, n=64 (the other notebook) ~8 bits."""
    with pytest.raises(ValueError, match="mostly be noise"):
        QKDConfig("bb84_numpy", n_qubits, 0.25).validate()


def test_lite_preset_is_accepted_and_expected_sample_is_at_least_100():
    cfg = QKDConfig("bb84_numpy", **PRESETS["lite"])
    cfg.validate()
    assert cfg.n_qubits >= 1024 and cfg.expected_sample_size >= 100


def test_override_allows_a_small_sample_but_warns(caplog):
    with caplog.at_level(logging.WARNING):
        QKDConfig("bb84_numpy", 16, 0.25).validate(allow_small_sample=True)
    assert "override active" in caplog.text


@pytest.mark.parametrize("kwargs", [{"n_qubits": 0}, {"sample_fraction": 0.0}, {"sample_fraction": 1.5},
                                    {"bit_flip_probability": 0.7}])
def test_impossible_settings_are_rejected_even_with_override(kwargs):
    base = {"backend": "bb84_numpy", "n_qubits": 1024, "sample_fraction": 0.25}
    with pytest.raises(ValueError):
        QKDConfig(**{**base, **kwargs}).validate(allow_small_sample=True)


def test_run_experiment_refuses_to_start_on_a_tiny_sample(tmp_path):
    with pytest.raises(ValueError, match="mostly be noise"):
        run_experiment(
            data_root=tmp_path, partition_root=tmp_path, num_clients=3, num_rounds=1, local_epochs=1,
            batch_size=4, n_qubits=16, intercept_probability=0.0, seed=0, experiment_name="t",
            output_path=Path(tmp_path) / "out.json", pretrained=False,
        )


def test_results_json_records_the_qkd_config_and_it_changes_the_hash():
    def experiment(**qkd):
        return build_experiment_log(
            experiment_name="x", seed=1, num_clients=3, num_rounds=2, local_epochs=1, batch_size=4, n_qubits=1024,
            intercept_probability=0.0, elapsed_seconds=0.0, round_logs=[], n_failures=0,
            init_weights={"pretrained": False, "source": "random_init", "weights_file": None, "sha256": None},
            qkd=QKDConfig("bb84_numpy", 1024, 0.25, **qkd).to_dict(),
        )["experiment"]

    clean, noisy = experiment(), experiment(bit_flip_probability=0.02)
    assert clean["qkd"]["expected_sample_size"] == 128 and clean["qkd"]["backend"] == "bb84_numpy"
    assert clean["config_hash"] != noisy["config_hash"]


# --------------------------------------------------------------------------
# Strategy with the real numpy backend
# --------------------------------------------------------------------------

def _qbers(experiment_seed, alpha, rounds=8, bit_flip=0.0, n_qubits=1024):
    init = ndarrays_to_parameters([np.zeros(2, np.float32)])
    strategy = EveFLStrategy(initial_parameters=init, n_qubits=n_qubits, intercept_probability=alpha,
                             bit_flip_probability=bit_flip, experiment_seed=experiment_seed)
    manager = _Manager([_Proxy(str(i)) for i in range(3)])
    out = []
    for r in range(1, rounds + 1):
        strategy.configure_fit(r, init, manager)
        out.append(dict(strategy._round_qber_per_client))
    return out, strategy


def test_experiment_seeds_give_different_qber_trajectories_through_the_strategy():
    trajectories = [tuple(map(str, _qbers(seed, alpha=0.5)[0])) for seed in range(5)]
    assert len(set(trajectories)) == 5


def test_same_experiment_seed_reproduces_the_trajectory():
    assert _qbers(3, 0.5)[0] == _qbers(3, 0.5)[0]


def test_no_eve_no_noise_stays_at_zero_qber_and_secure():
    per_round, strategy = _qbers(0, alpha=0.0)
    assert all(q == 0.0 for rnd in per_round for q in rnd.values())
    assert strategy._round_state.value == "SECURE"


def test_baseline_noise_raises_the_observed_qber():
    clean = np.mean([q for rnd in _qbers(1, 0.0, bit_flip=0.0)[0] for q in rnd.values()])
    noisy = np.mean([q for rnd in _qbers(1, 0.0, bit_flip=0.03)[0] for q in rnd.values()])
    assert clean == 0.0 and 0.01 < noisy < 0.06


def test_round_details_log_sample_size_and_simulation_ground_truth():
    _, strategy = _qbers(0, alpha=0.5, rounds=1)
    details = strategy._round_qkd_details
    assert set(details) == {"0", "1", "2"}
    for d in details.values():
        assert d["qber_sample_size"] > 0 and 0 <= d["qber_sample_errors"] <= d["qber_sample_size"]
        assert 0.0 <= d["sim_only_true_qber"] <= 1.0


def test_small_sample_strategy_warns(caplog):
    with caplog.at_level(logging.WARNING):
        EveFLStrategy(initial_parameters=ndarrays_to_parameters([np.zeros(1, np.float32)]), n_qubits=16)
    assert "mostly be noise" in caplog.text
