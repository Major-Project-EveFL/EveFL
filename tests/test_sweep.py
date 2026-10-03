"""
Tests for the sweep engine (evefl/quantum/sweep.py) and the pre-registered selection rule
(scripts/qber_sweep.py). The engine's exact analytic values are checked against brute-force
enumeration, the paired Monte Carlo, and the full bb84_numpy protocol.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from evefl.orchestration.state_machine import StateThresholds
from evefl.quantum.base import ChannelModel
from evefl.quantum.bb84_numpy import BB84NumpyProtocol
from evefl.quantum.sweep import (
    PairedDraws,
    analytic_single_link,
    check_against_protocol,
    error_probability,
    min_errors_for,
    sample_size,
    simulate_system_rates,
    system_probability,
    wilson_interval,
)

REAL = StateThresholds()  # 0.05 / 0.11


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------

@pytest.mark.parametrize("threshold", [0.05, 0.11, 0.2, 0.3333])
def test_min_errors_matches_the_controller_float_comparison(threshold):
    m = np.arange(1, 3000)
    k = min_errors_for(m, threshold)
    assert np.all(k / m >= threshold)
    assert np.all((k - 1) / m < threshold)


def test_error_probability_is_the_channel_formula():
    assert error_probability(0.5, 0.02) == pytest.approx(ChannelModel(0.5, 0.02).expected_qber)


def test_sample_size_mirrors_the_protocol():
    for seed in range(20):
        result = BB84NumpyProtocol(sample_fraction=0.25, seed=seed).run_exchange(1000, intercept_probability=0.3)
        assert result.metadata["qber_sample_size"] == int(sample_size(np.array([result.n_sifted]), 0.25)[0])


def test_sample_size_of_an_empty_sifted_key_is_zero():
    assert sample_size(np.array([0, 1, 2, 10]), 0.25).tolist() == [0, 1, 1, 2]


def test_system_probability():
    assert system_probability(0.0, 3) == 0.0 and system_probability(1.0, 3) == 1.0
    assert system_probability(0.1, 3) == pytest.approx(1 - 0.9 ** 3)


def test_wilson_interval_known_values():
    lo, hi = wilson_interval(0, 100)
    assert lo == 0.0 and hi == pytest.approx(0.0370, abs=1e-3)
    lo, hi = wilson_interval(50, 100)
    assert (lo, hi) == pytest.approx((0.4038, 0.5962), abs=1e-3)


# --------------------------------------------------------------------------
# Exact analytic values vs brute-force enumeration
# --------------------------------------------------------------------------

def _brute_force(n, f, p, thresholds):
    ns = ld = 0.0
    for s in range(n + 1):
        ws = stats.binom.pmf(s, n, 0.5)
        m = int(sample_size(np.array([s]), f)[0])
        if m == 0:
            continue
        for k in range(m + 1):
            q = k / m
            pk = ws * stats.binom.pmf(k, m, p)
            ns += pk * (not (q < thresholds.secure_max))
            ld += pk * (not (q < thresholds.caution_max))
    return ns, ld


@pytest.mark.parametrize("n,f,p", [(12, 0.5, 0.15), (40, 0.25, 0.07), (64, 0.5, 0.02), (30, 1.0, 0.3)])
def test_analytic_single_link_equals_brute_force(n, f, p):
    thresholds = StateThresholds(secure_max=0.2, caution_max=0.4)
    expected_ns, expected_ld = _brute_force(n, f, p, thresholds)
    got = analytic_single_link(n, f, p, thresholds)
    assert got["p_not_secure"] == pytest.approx(expected_ns, abs=1e-9)
    assert got["p_lockdown"] == pytest.approx(expected_ld, abs=1e-9)


def test_analytic_with_real_thresholds_equals_brute_force():
    n, f, p = 60, 0.5, 0.06
    expected_ns, expected_ld = _brute_force(n, f, p, REAL)
    got = analytic_single_link(n, f, p, REAL)
    assert got["p_not_secure"] == pytest.approx(expected_ns, abs=1e-9)
    assert got["p_lockdown"] == pytest.approx(expected_ld, abs=1e-9)


def test_no_eve_no_noise_never_alarms_and_full_eve_at_full_sample_always_alarms():
    assert analytic_single_link(1024, 0.25, 0.0, REAL)["p_not_secure"] == 0.0
    assert analytic_single_link(16384, 0.5, error_probability(1.0, 0.0), REAL)["p_not_secure"] > 1 - 1e-12


def test_expected_sample_and_leftover_are_consistent():
    r = analytic_single_link(1024, 0.25, 0.0, REAL)
    assert r["expected_sample"] == pytest.approx(1024 * 0.5 * 0.25, rel=0.02)
    assert r["expected_leftover_per_qubit"] == pytest.approx(0.5 * 0.75, rel=0.02)


def test_more_qubits_help_detection_and_false_alarms():
    """Sanity of the physics the sweep is meant to expose."""
    small = analytic_single_link(256, 0.25, error_probability(0.0, 0.03), REAL)["p_not_secure"]
    large = analytic_single_link(8192, 0.25, error_probability(0.0, 0.03), REAL)["p_not_secure"]
    assert large < small  # false CAUTION at 3% noise falls with block size
    weak_small = analytic_single_link(256, 0.25, error_probability(0.3, 0.0), REAL)["p_not_secure"]
    weak_large = analytic_single_link(8192, 0.25, error_probability(0.3, 0.0), REAL)["p_not_secure"]
    assert weak_large > weak_small  # detection of a 7.5% QBER improves


# --------------------------------------------------------------------------
# Monte Carlo agrees with the analytic values; pairing
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n,f,e,alpha", [(1024, 0.25, 0.0, 0.3), (1024, 0.25, 0.02, 0.0), (512, 0.5, 0.01, 0.5),
                                         (4096, 0.1, 0.03, 0.0), (256, 0.25, 0.0, 1.0)])
def test_simulated_rates_match_analytic_within_confidence_interval(n, f, e, alpha):
    p = error_probability(alpha, e)
    draws = PairedDraws.draw(n, rounds=20_000, links=3, seed=0)
    sim = simulate_system_rates(draws, f, p, REAL)
    exact = system_probability(analytic_single_link(n, f, p, REAL)["p_not_secure"], 3)
    # 99.99% interval so the deterministic test does not sit on a knife edge
    lo, hi = wilson_interval(round(sim["p_not_secure"] * sim["rounds"]), sim["rounds"], z=3.89)
    assert lo <= exact <= hi, (exact, sim)


def test_paired_draws_make_detection_monotone_in_alpha_and_noise():
    draws = PairedDraws.draw(1024, rounds=5000, links=3, seed=1)
    by_alpha = [simulate_system_rates(draws, 0.25, error_probability(a, 0.0), REAL)["p_not_secure"]
                for a in np.linspace(0, 1, 11)]
    assert by_alpha == sorted(by_alpha)
    by_noise = [simulate_system_rates(draws, 0.25, error_probability(0.3, e), REAL)["p_not_secure"]
                for e in (0.0, 0.01, 0.02, 0.03)]
    assert by_noise == sorted(by_noise)


def test_draws_are_reproducible_and_depend_on_block_size():
    a, b = PairedDraws.draw(512, 100, 3, 0), PairedDraws.draw(512, 100, 3, 0)
    c = PairedDraws.draw(1024, 100, 3, 0)
    assert np.array_equal(a.sifted, b.sifted) and not np.array_equal(a.uniforms, c.uniforms)


def test_full_protocol_agrees_with_the_analytic_engine():
    result = check_against_protocol(n=512, f=0.25, e=0.01, alpha=0.3, rounds=600, links=3, seed=0, thresholds=REAL)
    assert result["pvalue_not_secure"] > 1e-3, result
    assert result["pvalue_lockdown"] > 1e-3, result


# --------------------------------------------------------------------------
# The pre-registered selection rule (scripts/qber_sweep.py)
# --------------------------------------------------------------------------

def _load_sweep_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "qber_sweep.py"
    spec = importlib.util.spec_from_file_location("qber_sweep_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cells(module, grid_n, grid_f):
    cells = []
    for n in grid_n:
        for f in grid_f:
            for e in module.NOISES:
                for alpha in module.ALPHAS:
                    single = analytic_single_link(n, f, error_probability(alpha, e), module.THRESHOLDS)
                    cells.append({
                        "n_qubits": n, "sample_fraction": f, "bit_flip": e, "alpha": alpha,
                        "expected_sample": single["expected_sample"],
                        "expected_leftover_per_qubit": single["expected_leftover_per_qubit"],
                        "analytic": {"system_p_not_secure": system_probability(single["p_not_secure"], 3)},
                    })
    return cells


def test_selection_rule_picks_the_smallest_sufficient_block_and_respects_all_criteria(monkeypatch):
    module = _load_sweep_script()
    grid_n, grid_f = (256, 512, 1024, 2048), (0.10, 0.25, 0.50)
    monkeypatch.setattr(module, "N_QUBITS", grid_n)
    monkeypatch.setattr(module, "SAMPLE_FRACTIONS", grid_f)
    selection = module.select_headline(_cells(module, grid_n, grid_f))

    assert selection["candidates"], "expected at least one candidate on this grid"
    head = selection["headline"]
    assert head["n_qubits"] == min(c["n_qubits"] for c in selection["candidates"])
    for c in selection["candidates"]:
        assert c["false_caution_e0.01"] <= 0.01
        assert c["detection_a0.3_e0"] >= 0.95 and c["detection_a0.3_e0.01"] >= 0.95
        assert c["expected_sample"] >= 100
    # tie-break: among the smallest n, the larger leftover key
    same_n = [c for c in selection["candidates"] if c["n_qubits"] == head["n_qubits"]]
    assert head["expected_leftover_per_qubit"] == max(c["expected_leftover_per_qubit"] for c in same_n)


def test_selection_rule_reports_no_headline_when_nothing_qualifies(monkeypatch):
    module = _load_sweep_script()
    grid_n, grid_f = (256,), (0.10,)  # expected sample ~13 bits: fails criterion 3 and detection
    monkeypatch.setattr(module, "N_QUBITS", grid_n)
    monkeypatch.setattr(module, "SAMPLE_FRACTIONS", grid_f)
    selection = module.select_headline(_cells(module, grid_n, grid_f))
    assert selection["candidates"] == [] and selection["headline"] is None
