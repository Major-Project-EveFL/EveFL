"""
Tests for the classical post-processing pipeline (evefl/quantum/postprocess.py): finite-key bound,
Cascade, Toeplitz hashing, authentication, and the whole pipeline including the simulation-only check
against what an intercept-resend Eve actually knows.
"""

import math

import numpy as np
import pytest

from evefl.quantum import postprocess as pp
from evefl.quantum.base import ChannelModel, QKDResult
from evefl.quantum.bb84_numpy import BB84NumpyProtocol
from evefl.quantum.postprocess import (
    RoundKey,
    authenticate,
    binary_entropy,
    cascade_reconcile,
    finite_key_length,
    postprocess,
    random_bits,
    statistical_deviation,
    toeplitz_hash,
    verify_authentication,
)

AUTH_KEY = b"\x07" * 32


def _exchange(n_qubits=2**17, alpha=0.0, e=0.01, seed=3, fraction=0.1, eve=False):
    return BB84NumpyProtocol(sample_fraction=fraction, seed=seed, record_eve_view=eve).run_exchange(
        n_qubits, channel=ChannelModel(alpha, e))


# --------------------------------------------------------------------------
# Binary entropy and the finite-key bound (formula of Tomamichel et al. 2012, Supp. Thm 2, Eq. S3-S4)
# --------------------------------------------------------------------------

def test_binary_entropy_values_and_truncation():
    assert binary_entropy(0.0) == 0.0 and binary_entropy(0.5) == pytest.approx(1.0)
    assert binary_entropy(0.11) == pytest.approx(0.4999, abs=1e-3)
    assert binary_entropy(0.2) == pytest.approx(binary_entropy(0.2))
    assert binary_entropy(0.7) == 1.0 and binary_entropy(1.0) == 1.0  # truncated at 1 above 1/2
    assert binary_entropy(-0.1) == 0.0


def test_statistical_deviation_matches_the_published_formula():
    n, k, eps = 1000, 100, 1e-10
    expected = math.sqrt((n + k) / (n * k) * (k + 1) / k * math.log(1 / eps))
    assert statistical_deviation(n, k, eps) == pytest.approx(expected)
    assert statistical_deviation(n, k, eps) == pytest.approx(0.5058, abs=1e-3)  # hand: sqrt(0.011 * 1.01 * ln(1e10))


def test_key_length_equals_the_formula_at_the_chosen_epsilons():
    n, k, q, leak = 60_000, 6_000, 0.01, 4_000.0
    b = finite_key_length(n, k, q, leak, eps_sec=1e-10, eps_cor=1e-15)
    assert 2 * b.eps + b.eps_bar == pytest.approx(1e-10)
    expected = (n * (1 - binary_entropy(q + statistical_deviation(n, k, b.eps)))
                - 2 * math.log2(1 / (2 * b.eps_bar)) - leak - math.log2(2 / 1e-15))
    assert b.raw_length == pytest.approx(expected) and b.key_length == math.floor(expected)
    assert b.qber_upper == pytest.approx(q + b.mu)


def test_optimised_split_is_at_least_as_good_as_any_fixed_split():
    n, k, q, leak = 60_000, 6_000, 0.01, 4_000.0
    best = finite_key_length(n, k, q, leak).raw_length
    for fraction in (0.1, 0.5, 0.9):
        eps = fraction * 1e-10 / 2
        eps_bar = 1e-10 - 2 * eps
        raw = (n * (1 - binary_entropy(q + statistical_deviation(n, k, eps)))
               - 2 * math.log2(1 / (2 * eps_bar)) - leak - math.log2(2 / 1e-15))
        assert best >= raw - 1e-9


def test_key_length_is_monotone_in_each_input():
    base = dict(n_key=50_000, k_sample=5_000, observed_qber=0.02, leak_ec=3_000.0)
    l0 = finite_key_length(**base).raw_length
    assert finite_key_length(**{**base, "n_key": 100_000}).raw_length > l0
    assert finite_key_length(**{**base, "observed_qber": 0.04}).raw_length < l0
    assert finite_key_length(**{**base, "leak_ec": 6_000.0}).raw_length < l0
    assert finite_key_length(**{**base, "k_sample": 10_000}).raw_length != l0
    assert finite_key_length(**base, eps_sec=1e-14).raw_length < l0  # tighter security costs key


def test_asymptotic_rate_approaches_one_minus_two_h():
    n = 10**7
    q = 0.01
    b = finite_key_length(n, n // 10, q, leak_ec=n * binary_entropy(q))
    # mu does not vanish at k = n/10 (it shrinks like 1/sqrt(k)), so allow a few percent of slack
    assert b.key_length / n == pytest.approx(1 - 2 * binary_entropy(q), abs=0.04)
    bigger = finite_key_length(n, n // 2, q, leak_ec=n * binary_entropy(q))
    assert abs(bigger.key_length / n - (1 - 2 * binary_entropy(q))) < abs(b.key_length / n - (1 - 2 * binary_entropy(q)))


def test_headline_size_blocks_yield_no_key_at_the_default_epsilons():
    """n = 1024 qubits (about 384 key bits + 128 sample bits): the finite-key penalty exceeds the key."""
    for q in (0.0, 0.0125, 0.05):
        b = finite_key_length(384, 128, q, leak_ec=1.1 * 384 * binary_entropy(q))
        assert b.key_length == 0 and b.raw_length < 0


def test_a_few_thousand_qubits_are_enough_for_a_positive_key_at_one_percent_error():
    s = 2200  # sifted bits (about 4400 qubits), 25% sample
    k = s // 4
    n = s - k
    assert finite_key_length(n, k, 0.01, 1.1 * n * binary_entropy(0.01)).key_length > 0


@pytest.mark.parametrize("kwargs", [{"n_key": 0}, {"k_sample": 0}, {"observed_qber": 1.5}, {"leak_ec": -1.0},
                                    {"eps_sec": 0.0}, {"eps_cor": 1.0}])
def test_finite_key_length_validates_inputs(kwargs):
    params = dict(n_key=1000, k_sample=100, observed_qber=0.01, leak_ec=10.0)
    with pytest.raises(ValueError):
        finite_key_length(**{**params, **kwargs})


# --------------------------------------------------------------------------
# Cascade
# --------------------------------------------------------------------------

def _noisy_pair(n, q, seed):
    rng = np.random.default_rng(seed)
    alice = rng.integers(0, 2, n, dtype=np.uint8)
    flips = (rng.random(n) < q).astype(np.uint8)
    return alice, alice ^ flips


def test_cascade_with_no_errors_leaks_only_block_parities_and_changes_nothing():
    alice = np.random.default_rng(0).integers(0, 2, 1000, dtype=np.uint8)
    r = cascade_reconcile(alice, alice.copy(), 0.01, np.random.default_rng(1))
    assert np.array_equal(r.corrected, alice) and r.errors_corrected == 0
    assert r.leaked_bits == len(r.parity_transcript) > 0


@pytest.mark.parametrize("q", [0.005, 0.02, 0.05, 0.08])
def test_cascade_corrects_the_errors_with_realistic_efficiency(q):
    n = 30_000
    alice, bob = _noisy_pair(n, q, seed=int(q * 1000))
    r = cascade_reconcile(alice, bob, q, np.random.default_rng(2))
    residual = int(np.count_nonzero(r.corrected != alice))
    assert residual <= 2  # Cascade can leave a few; the verification hash catches them
    efficiency = r.leaked_bits / (n * binary_entropy(q))
    assert 1.0 < efficiency < 1.6
    assert r.leaked_bits == len(r.parity_transcript)  # every leaked bit is in the transcript


def test_cascade_is_deterministic_given_the_seed_and_validates_lengths():
    alice, bob = _noisy_pair(5000, 0.03, seed=1)
    a = cascade_reconcile(alice, bob, 0.03, np.random.default_rng(5))
    b = cascade_reconcile(alice, bob, 0.03, np.random.default_rng(5))
    assert a.leaked_bits == b.leaked_bits and np.array_equal(a.corrected, b.corrected)
    with pytest.raises(ValueError):
        cascade_reconcile(alice, bob[:-1], 0.03, np.random.default_rng(0))
    empty = cascade_reconcile(np.zeros(0, np.uint8), np.zeros(0, np.uint8), 0.01, np.random.default_rng(0))
    assert empty.leaked_bits == 0


def test_cascade_handles_zero_estimated_error_rate_via_the_blocksize_floor():
    alice, bob = _noisy_pair(4000, 0.004, seed=9)
    r = cascade_reconcile(alice, bob, 0.0, np.random.default_rng(3))  # estimate 0 but errors exist
    assert int(np.count_nonzero(r.corrected != alice)) <= 2


# --------------------------------------------------------------------------
# Toeplitz hashing, randomness, authentication, RoundKey
# --------------------------------------------------------------------------

def _explicit_toeplitz(bits, out_bits, seed):
    n = len(bits)
    matrix = np.array([[seed[i - j + n - 1] for j in range(n)] for i in range(out_bits)], dtype=np.int64)
    return (matrix @ bits.astype(np.int64)) % 2


def test_toeplitz_hash_equals_the_explicit_matrix_product():
    rng = np.random.default_rng(0)
    for n, out in ((10, 4), (37, 20), (64, 64)):
        bits = rng.integers(0, 2, n, dtype=np.uint8)
        seed = rng.integers(0, 2, n + out - 1, dtype=np.uint8)
        assert np.array_equal(toeplitz_hash(bits, out, seed), _explicit_toeplitz(bits, out, seed))


def test_toeplitz_hash_is_linear_and_validates_seed_length():
    rng = np.random.default_rng(1)
    x, y = rng.integers(0, 2, 200, dtype=np.uint8), rng.integers(0, 2, 200, dtype=np.uint8)
    seed = rng.integers(0, 2, 200 + 50 - 1, dtype=np.uint8)
    assert np.array_equal(toeplitz_hash(x ^ y, 50, seed), toeplitz_hash(x, 50, seed) ^ toeplitz_hash(y, 50, seed))
    with pytest.raises(ValueError):
        toeplitz_hash(x, 50, seed[:-1])
    assert len(toeplitz_hash(x, 0, rng.integers(0, 2, 199, dtype=np.uint8))) == 0


def test_toeplitz_hash_is_close_to_universal_on_collisions():
    rng = np.random.default_rng(2)
    x, y = rng.integers(0, 2, 64, dtype=np.uint8), rng.integers(0, 2, 64, dtype=np.uint8)
    out, trials, collisions = 6, 4000, 0
    for _ in range(trials):
        seed = rng.integers(0, 2, 64 + out - 1, dtype=np.uint8)
        collisions += np.array_equal(toeplitz_hash(x, out, seed), toeplitz_hash(y, out, seed))
    assert abs(collisions / trials - 2 ** -out) < 0.01  # a 2-universal family collides with prob <= 2^-out


def test_hash_seeds_come_from_secrets_not_random(monkeypatch):
    calls = []
    real = pp.secrets.token_bytes
    monkeypatch.setattr(pp.secrets, "token_bytes", lambda n: calls.append(n) or real(n))
    bits = random_bits(1000)
    assert calls == [125] and set(np.unique(bits)) <= {0, 1} and len(bits) == 1000
    assert not np.array_equal(random_bits(256), random_bits(256))
    source = open(pp.__file__, encoding="utf-8").read()
    assert "import random" not in source and "from random" not in source


def test_authentication_accepts_the_genuine_transcript_only():
    tag = authenticate(b"transcript", AUTH_KEY)
    assert verify_authentication(b"transcript", tag, AUTH_KEY)
    assert not verify_authentication(b"transcripT", tag, AUTH_KEY)
    assert not verify_authentication(b"transcript", tag, b"\x08" * 32)


def test_round_key_zeroizes_and_never_prints_key_material():
    key = RoundKey("0", 1, bytearray(b"\xde\xad\xbe\xef"), 32, 0.01, 0.07, 100, 1e-10, 1e-15)
    assert "deadbeef" not in repr(key).lower() and "\\xde" not in repr(key)
    key.zeroize()
    assert bytes(key.key) == b"\x00\x00\x00\x00" and key.zeroized and key.simulated


# --------------------------------------------------------------------------
# The whole pipeline
# --------------------------------------------------------------------------

def test_pipeline_produces_a_key_and_discards_the_sample():
    result = _exchange(alpha=0.0, e=0.01)
    out = postprocess(result, client_id="3", round_id=7, auth_key=AUTH_KEY, ec_seed=1)
    assert out.ok and out.round_key is not None
    key = out.round_key
    k = len(result.sample_indices)
    assert out.details["n_key"] == result.n_sifted - k  # compared bits removed from the key
    assert out.details["qber"] == pytest.approx(result.qber)
    assert key.key_bits == 8 * len(key.key) > 0 and key.key_bits <= out.details["key_length"]
    assert key.client_id == "3" and key.round_id == 7 and key.provenance == "bb84-sim"
    assert key.leaked_bits > out.details["leak_ec"]  # includes the verification hash bits
    assert key.qber_upper == pytest.approx(result.qber + out.details["mu"])


def test_two_runs_on_the_same_exchange_give_different_keys_because_hash_seeds_are_random():
    result = _exchange()
    a = postprocess(result, client_id="0", round_id=1, auth_key=AUTH_KEY, ec_seed=1).round_key
    b = postprocess(result, client_id="0", round_id=1, auth_key=AUTH_KEY, ec_seed=1).round_key
    assert bytes(a.key) != bytes(b.key)


def test_headline_size_exchange_yields_no_key():
    result = BB84NumpyProtocol(sample_fraction=0.25, seed=1).run_exchange(1024, channel=ChannelModel(0.0, 0.01))
    out = postprocess(result, client_id="0", round_id=1, auth_key=AUTH_KEY)
    assert out.status == pp.STATUS_NO_KEY and out.round_key is None and out.details["raw_length"] < 0


def test_qber_at_or_above_the_lockdown_threshold_aborts_with_ge():
    s, k = 2000, 200
    rng = np.random.default_rng(0)
    alice = rng.integers(0, 2, s, dtype=np.uint8)
    bob = alice.copy()
    sample = np.arange(k)
    bob[sample[:22]] ^= 1  # exactly 22 / 200 = 0.11 errors in the sample
    result = QKDResult(alice.tolist(), 0.11, 4000, s, 0.5, True, {}, bob.tolist(), sample.tolist())
    out = postprocess(result, client_id="0", round_id=1, auth_key=AUTH_KEY)
    assert out.status == pp.STATUS_ABORT_QBER and out.round_key is None
    bob2 = alice.copy()
    bob2[sample[:21]] ^= 1  # 0.105 < 0.11: proceeds past the QBER check
    result2 = QKDResult(alice.tolist(), 0.105, 4000, s, 0.5, True, {}, bob2.tolist(), sample.tolist())
    assert postprocess(result2, client_id="0", round_id=1, auth_key=AUTH_KEY).status != pp.STATUS_ABORT_QBER


def test_failed_error_correction_is_caught_by_the_verification_hash(monkeypatch):
    real = pp.cascade_reconcile

    def broken(alice, bob, q, rng, passes=pp.CASCADE_PASSES):
        r = real(alice, bob, q, rng, passes)
        r.corrected = r.corrected.copy()
        r.corrected[0] ^= 1  # a residual error Cascade missed
        return r

    monkeypatch.setattr(pp, "cascade_reconcile", broken)
    out = postprocess(_exchange(), client_id="0", round_id=1, auth_key=AUTH_KEY, ec_seed=1)
    assert out.status == pp.STATUS_ABORT_EC and out.round_key is None


def test_tampered_transcript_fails_authentication():
    out = postprocess(_exchange(), client_id="0", round_id=1, auth_key=AUTH_KEY, ec_seed=1, tamper_transcript=True)
    assert out.status == pp.STATUS_ABORT_AUTH and out.round_key is None


def test_pipeline_needs_bobs_key_and_the_sample_positions():
    result = _exchange(n_qubits=4096)
    result.bob_sifted_key = None
    with pytest.raises(ValueError, match="bob_sifted_key"):
        postprocess(result, client_id="0", round_id=1, auth_key=AUTH_KEY)


# --------------------------------------------------------------------------
# Simulation-only check: the key length stays below what Eve does NOT know (weak Eve included)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.15, 0.2])
def test_key_length_never_exceeds_the_entropy_left_after_intercept_resend_eve(alpha):
    """Eve learns a sifted bit exactly when she intercepted it in Alice's basis; her other results are
    independent of Alice's bits. So H_min(X|E) = n_key - (bits she knows), minus what error correction and the
    verification hash disclose. The finite-key length l must stay below that, in EVERY trial, including weak Eve
    whose QBER estimate can come out low by chance (this is what the mu term protects against)."""
    violations, produced = 0, 0
    for trial in range(8):
        result = _exchange(alpha=alpha, e=0.0, seed=100 + trial, eve=True)
        out = postprocess(result, client_id="0", round_id=trial, auth_key=AUTH_KEY, ec_seed=trial)
        if not out.ok:
            continue
        produced += 1
        known = np.asarray(result.metadata["sim_only_eve_known_positions"], dtype=np.int64)
        in_sample = np.zeros(result.n_sifted, dtype=bool)
        in_sample[result.sample_indices] = True
        known_in_key = int(np.count_nonzero(~in_sample[known]))
        disclosed = out.details["leak_ec"] + math.ceil(math.log2(1 / pp.DEFAULT_EPS_COR))
        residual_entropy = out.details["n_key"] - known_in_key - disclosed
        violations += out.round_key.key_bits > residual_entropy
    assert produced >= 1, "expected at least one key at this block size"
    assert violations == 0


def test_strong_eve_means_no_key_or_abort_at_this_block_size():
    """alpha = 0.3 (QBER 7.5%): the finite-size penalty leaves no key at 2^17 qubits; alpha = 0.5 (12.5%) is above
    the 0.11 abort threshold. Either way no key reaches the FL layer."""
    weak_block = postprocess(_exchange(alpha=0.3, e=0.0), client_id="0", round_id=1, auth_key=AUTH_KEY, ec_seed=1)
    assert weak_block.status == pp.STATUS_NO_KEY
    aborted = postprocess(_exchange(alpha=0.5, e=0.0), client_id="0", round_id=1, auth_key=AUTH_KEY, ec_seed=1)
    assert aborted.status == pp.STATUS_ABORT_QBER and aborted.round_key is None
