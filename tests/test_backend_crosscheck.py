"""
Cross-validation of the Qiskit backend against the exact numpy backend and the analytic QBER.

Marked `slow` (Qiskit runs one Aer job per measurement): excluded from default runs and CI,
executed by the nightly / manual workflow (.github/workflows/nightly-validation.yml) and by
`pytest -m slow`. The larger, archived version lives in scripts/validate_backends.py.
"""

import pytest

from evefl.quantum.base import ChannelModel
from evefl.quantum.bb84 import BB84Protocol
from evefl.quantum.bb84_numpy import BB84NumpyProtocol
from evefl.quantum.validation import analytic_check, compare_backends, run_trials

pytestmark = pytest.mark.slow

ALPHA = 1e-3
N_QUBITS = 128


@pytest.mark.parametrize("alpha,e", [(0.0, 0.0), (1.0, 0.0), (0.6, 0.03), (0.0, 0.05)])
def test_qiskit_and_numpy_backends_agree_and_match_theory(alpha, e):
    channel = ChannelModel(alpha, e)
    qiskit = run_trials(lambda t: BB84Protocol(seed=1000 + t), channel, N_QUBITS, n_trials=40)
    numpy_ = run_trials(lambda t: BB84NumpyProtocol.for_round(7, t, "0"), channel, N_QUBITS, n_trials=400)

    for name, counts in (("qiskit", qiskit), ("numpy", numpy_)):
        check = analytic_check(counts, channel, confidence=0.999)
        assert check["qber_ci_low"] <= check["expected_qber"] <= check["qber_ci_high"], (name, check)
        assert check["sifted_pvalue"] > ALPHA, (name, check)

    comparison = compare_backends(qiskit, numpy_)
    assert comparison["pvalue"] > ALPHA, comparison
