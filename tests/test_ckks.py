import struct

import pytest

from evefl.crypto.ckks import CKKSAggregator


@pytest.fixture
def aggregator():
    return CKKSAggregator()


@pytest.fixture
def context(aggregator):
    return aggregator.create_context()


def test_encrypt_decrypt_round_trip(aggregator, context):
    """Sanity check: a single encrypted vector decrypts back close to
    the original (CKKS is approximate arithmetic, not exact)."""
    original = [1.5, -2.3, 0.0, 42.0]
    ciphertext = aggregator.encrypt_vector(original, context)
    decrypted = aggregator.decrypt_vector(ciphertext, context)

    for orig, dec in zip(original, decrypted):
        assert abs(orig - dec) < 1e-3


def test_homomorphic_weighted_aggregation_matches_plaintext(aggregator, context):
    """The real point of CKKS: sum ciphertexts, decrypt only the sum,
    and confirm it matches the plaintext weighted average — without
    ever decrypting the individual client vectors."""
    client_updates = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [-1.0, 0.0, 1.0],
    ]
    weights = [0.5, 0.3, 0.2]

    ciphertexts = [aggregator.encrypt_vector(v, context) for v in client_updates]

    # Server-side: only public_material needed, no secret key required
    server_context = type(context)(public_material=context.public_material, secret_material=None)
    agg_ciphertext = aggregator.aggregate(ciphertexts, weights, server_context)

    decrypted_agg = aggregator.decrypt_vector(agg_ciphertext, context)

    expected = [
        sum(w * v[i] for w, v in zip(weights, client_updates))
        for i in range(len(client_updates[0]))
    ]

    for exp, dec in zip(expected, decrypted_agg):
        assert abs(exp - dec) < 1e-2


def test_server_cannot_decrypt_without_secret_key(aggregator, context):
    """Confirm the server-blind property: a context without
    secret_material cannot decrypt anything."""
    server_context = type(context)(public_material=context.public_material, secret_material=None)
    ciphertext = aggregator.encrypt_vector([1.0, 2.0], context)

    with pytest.raises(ValueError):
        aggregator.decrypt_vector(ciphertext, server_context)


def test_mismatched_lengths_raise(aggregator, context):
    ciphertexts = [aggregator.encrypt_vector([1.0], context)]
    with pytest.raises(ValueError):
        aggregator.aggregate(ciphertexts, weights=[0.5, 0.5], context=context)


def test_empty_aggregate_raises(aggregator, context):
    with pytest.raises(ValueError):
        aggregator.aggregate([], [], context)


def test_real_ciphertext_inflation_measurement(aggregator, context):
    """Honest, measured ciphertext inflation for a gradient-sized
    vector — replaces the fabricated '17.3x' figure. Run this and use
    the printed ratio in the paper instead of an unverified citation."""
    n = 1000  # stand-in for a gradient chunk (not full ResNet-18 yet)
    plaintext_vector = [float(i) * 0.001 for i in range(n)]
    plaintext_bytes = n * 8  # 8 bytes per float64

    ciphertext = aggregator.encrypt_vector(plaintext_vector, context)
    ciphertext_bytes = len(ciphertext)

    inflation_ratio = ciphertext_bytes / plaintext_bytes
    print(f"\nCKKS inflation for n={n}: plaintext={plaintext_bytes}B, "
          f"ciphertext={ciphertext_bytes}B, ratio={inflation_ratio:.1f}x")

    assert ciphertext_bytes > plaintext_bytes  # inflation is real, direction sanity check