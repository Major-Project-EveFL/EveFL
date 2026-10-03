"""Library versions are recorded in every results JSON (and the NumPy version in partition_meta.json).

They are provenance, not configuration: they must never enter the config hash, otherwise the same experiment run on
two machines would get two hashes.
"""

import json
import sys

import numpy as np

from evefl import environment
from evefl.fl.dataset import partition_and_save
from evefl.fl.server import build_experiment_log
from tests.helpers_data import make_metadata_frame

INIT = {"pretrained": False, "source": "random_init", "weights_file": None, "sha256": None}


def _log():
    return build_experiment_log(
        experiment_name="t", seed=0, num_clients=3, num_rounds=1, local_epochs=1, batch_size=4, n_qubits=8,
        intercept_probability=0.0, elapsed_seconds=0.0, round_logs=[], n_failures=0, init_weights=INIT,
    )


def test_library_versions_report_installed_packages_and_none_for_missing_ones():
    versions = environment.library_versions(("numpy", "definitely-not-installed-pkg"))
    assert versions["numpy"] == np.__version__
    assert versions["definitely-not-installed-pkg"] is None


def test_default_package_list_covers_the_stack_that_affects_results():
    for name in ("torch", "torchvision", "numpy", "scipy", "pandas", "scikit-learn", "flwr", "cryptography"):
        assert name in environment.TRACKED_PACKAGES


def test_runtime_environment_has_python_platform_and_libraries():
    env = environment.runtime_environment()
    assert env["python"] == sys.version.split()[0]
    assert {"implementation", "system", "machine", "libraries"} <= set(env)
    assert env["libraries"]["numpy"] == np.__version__
    json.dumps(env)  # serialisable


def test_every_results_json_records_the_environment():
    experiment = _log()["experiment"]
    assert experiment["environment"]["libraries"]["numpy"] == np.__version__
    assert experiment["environment"]["python"] == sys.version.split()[0]


def test_environment_is_not_part_of_the_config_hash(monkeypatch):
    first = _log()["experiment"]
    monkeypatch.setattr(environment, "runtime_environment", lambda: {"python": "9.9.9", "libraries": {"numpy": "0.0"}})
    second = _log()["experiment"]
    assert second["environment"]["python"] == "9.9.9" and first["environment"]["python"] != "9.9.9"
    assert first["config_hash"] == second["config_hash"]
    assert "environment" not in first["config_hash"]


def test_partition_meta_records_the_numpy_version(tmp_path):
    make_metadata_frame(300, seed=2).to_csv(tmp_path / "Data_Entry_2017.csv", index=False)
    partition_and_save(tmp_path, tmp_path / "p", n_clients=3, seed=7)
    meta = json.loads((tmp_path / "p" / "partition_meta.json").read_text())
    assert meta["numpy_version"] == np.__version__
