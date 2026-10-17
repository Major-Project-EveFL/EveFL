"""
Secure aggregation, step 2a: the mask PRG (ADR 0003, amendment 4 "PRG layout" and acceptance test 17).

For each unordered pair {i, j}, round r and floating-point tensor index t the stream key is
HKDF-SHA256(seed_ij, info = lp("evefl/v1/mask-stream") || u64(r) || lp(min) || lp(max) || u32(t)); the stream is AES-256-CTR
under that key from an all-zero counter block, element k being the little-endian uint64 in keystream bytes [8k, 8k+8).
"""

import itertools
import re
from pathlib import Path

import numpy as np
import pytest
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from evefl.crypto.secagg import fixedpoint as fp
from evefl.crypto.secagg import prg
from evefl.crypto.secagg.encoding import lp, u32, u64

SEED = bytes(range(32))
OTHER_SEED = bytes(range(1, 33))


def _key(seed=SEED, round_id=1, a="0", b="1", t=0):
    return prg.stream_key(seed, round_id=round_id, client_a=a, client_b=b, tensor_index=t)


# --------------------------------------------------------------------------
# encodings
# --------------------------------------------------------------------------

def test_length_prefixed_fields_are_injective():
    assert lp(b"a") + lp(b"bc") != lp(b"ab") + lp(b"c")
    assert lp(b"") + lp(b"x") != lp(b"x") + lp(b"")
    assert u32(1) != u32(2) and len(u32(7)) == 4 and len(u64(7)) == 8
    for bad in (-1, 2 ** 32):
        with pytest.raises(ValueError):
            u32(bad)
    with pytest.raises(ValueError):
        u64(2 ** 64)


def test_info_string_follows_the_adr_layout():
    expected = lp(b"evefl/v1/mask-stream") + u64(5) + lp(b"alice") + lp(b"bob") + u32(9)
    assert prg.stream_info(round_id=5, client_a="bob", client_b="alice", tensor_index=9) == expected


# --------------------------------------------------------------------------
# stream keys: distinct, symmetric in the pair
# --------------------------------------------------------------------------

def test_key_is_32_bytes_deterministic_and_symmetric_in_the_pair():
    k = _key()
    assert len(k) == 32 and k == _key()
    assert _key(a="0", b="1") == _key(a="1", b="0")


def test_every_change_of_seed_round_pair_or_tensor_gives_a_different_key():
    base = _key()
    assert _key(seed=OTHER_SEED) != base
    assert _key(round_id=2) != base
    assert _key(a="0", b="2") != base
    assert _key(a="2", b="1") != base
    assert _key(t=1) != base


def test_over_a_domain_of_pairs_rounds_and_tensors_every_key_is_distinct():
    clients = [str(i) for i in range(8)]
    keys = set()
    count = 0
    for a, b in itertools.combinations(clients, 2):
        for r in range(1, 5):
            for t in range(6):
                keys.add(_key(a=a, b=b, round_id=r, t=t))
                count += 1
    assert len(keys) == count == 28 * 4 * 6


def test_client_ids_that_would_collide_under_naive_concatenation_give_different_keys():
    assert _key(a="1", b="23") != _key(a="12", b="3")
    assert _key(a="", b="ab") != _key(a="a", b="b")


def test_argument_ranges_are_checked():
    with pytest.raises(ValueError):
        prg.stream_key(b"short", round_id=1, client_a="0", client_b="1", tensor_index=0)
    with pytest.raises(ValueError):
        _key(a="3", b="3")   # a pair needs two different clients
    with pytest.raises(ValueError):
        _key(round_id=-1)
    with pytest.raises(ValueError):
        _key(t=2 ** 32)


# --------------------------------------------------------------------------
# expansion: AES-256-CTR from counter zero, little-endian uint64
# --------------------------------------------------------------------------

def test_stream_starts_at_counter_zero_and_is_little_endian_uint64():
    key = _key()
    first_block = Cipher(algorithms.AES(key), modes.ECB()).encryptor().update(bytes(16))   # CTR block 0 = AES_K(0^128)
    out = prg.expand(key, 4)
    assert out.dtype == np.uint64 and out.shape == (4,)
    assert out[0].item() == int.from_bytes(first_block[:8], "little")
    assert out[1].item() == int.from_bytes(first_block[8:], "little")
    second_block = Cipher(algorithms.AES(key), modes.ECB()).encryptor().update((1).to_bytes(16, "big"))
    assert out[2].item() == int.from_bytes(second_block[:8], "little")


def test_expansion_is_deterministic_prefix_consistent_and_key_dependent():
    key = _key()
    assert np.array_equal(prg.expand(key, 100), prg.expand(key, 100))
    assert np.array_equal(prg.expand(key, 40), prg.expand(key, 100)[:40])
    assert not np.array_equal(prg.expand(key, 100), prg.expand(_key(t=1), 100))
    assert prg.expand(key, 0).shape == (0,)
    with pytest.raises(ValueError):
        prg.expand(b"short", 4)
    with pytest.raises(ValueError):
        prg.expand(key, -1)


def test_mask_words_look_uniform_over_the_full_64_bit_range():
    words = prg.expand(_key(), 50_000)
    top_bit = np.mean((words >> np.uint64(63)) == 1)
    low_bit = np.mean((words & np.uint64(1)) == 1)
    assert 0.48 < top_bit < 0.52 and 0.48 < low_bit < 0.52   # a sanity check of the layout, not a randomness proof


# --------------------------------------------------------------------------
# partners derive the same mask; opposite signs cancel
# --------------------------------------------------------------------------

def test_the_two_partners_derive_identical_masks_and_different_ones_for_any_change():
    mine = prg.MaskStreams(SEED, round_id=3, me="0", other="1")
    theirs = prg.MaskStreams(SEED, round_id=3, me="1", other="0")
    a, b = mine.take(0, 1_000), theirs.take(0, 1_000)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, mine.take(1, 1_000))                                       # another tensor
    assert not np.array_equal(a, prg.MaskStreams(SEED, round_id=4, me="0", other="1").take(0, 1_000))   # another round
    assert not np.array_equal(a, prg.MaskStreams(SEED, round_id=3, me="0", other="2").take(0, 1_000))   # another pair
    assert not np.array_equal(a, prg.MaskStreams(OTHER_SEED, round_id=3, me="0", other="1").take(0, 1_000))


def test_signed_masks_cancel_exactly_in_the_sum_over_all_participants():
    ids = ["a", "b", "c", "d", "e"]
    seeds = {frozenset(p): bytes([i + 1]) * 32 for i, p in enumerate(itertools.combinations(ids, 2))}
    total = {t: np.zeros(64, dtype=np.uint64) for t in range(3)}
    for me in ids:
        streams = {other: prg.MaskStreams(seeds[frozenset((me, other))], round_id=7, me=me, other=other)
                   for other in ids if other != me}
        for t in range(3):
            for s in streams.values():
                total[t] = fp.add(total[t], s.signed_mask(t, 64))
    assert all(not total[t].any() for t in total)


def test_the_sign_follows_the_byte_order_of_the_ids():
    low = prg.MaskStreams(SEED, round_id=1, me="a", other="b")
    high = prg.MaskStreams(SEED, round_id=1, me="b", other="a")
    plus, minus = low.signed_mask(0, 8), high.signed_mask(0, 8)
    assert np.array_equal(fp.add(plus, minus), np.zeros(8, dtype=np.uint64))
    raw = prg.MaskStreams(SEED, round_id=1, me="a", other="b").take(5, 8)
    assert np.array_equal(low.signed_mask(5, 8), raw)                 # the smaller id adds
    assert np.array_equal(high.signed_mask(5, 8), fp.sub(np.zeros(8, dtype=np.uint64), raw))   # the larger subtracts


# --------------------------------------------------------------------------
# no reuse, enforced in code
# --------------------------------------------------------------------------

def test_a_second_request_for_the_same_pair_round_tensor_raises():
    s = prg.MaskStreams(SEED, round_id=1, me="0", other="1")
    s.take(0, 10)
    with pytest.raises(prg.StreamReuseError):
        s.take(0, 10)
    with pytest.raises(prg.StreamReuseError):
        s.take(0, 5)          # a shorter prefix is still a reuse of the stream
    with pytest.raises(prg.StreamReuseError):
        s.signed_mask(0, 10)  # and so is the signed form
    s.take(1, 10)             # a different tensor is a different stream


def test_each_object_owns_its_own_consumption_record_but_never_the_same_key_twice():
    first = prg.MaskStreams(SEED, round_id=1, me="0", other="1")
    first.take(0, 4)
    clone = prg.MaskStreams(SEED, round_id=1, me="0", other="1")
    assert clone.take(0, 4) is not None   # the partner side legitimately expands the same stream once, from its own object


def test_a_round_id_is_used_for_at_most_one_attempt_and_an_aborted_attempt_is_never_resumed():
    guard = prg.RoundGuard()
    guard.begin(5)
    with pytest.raises(prg.RoundReuseError):
        guard.begin(5)
    guard.abort(5)
    with pytest.raises(prg.RoundReuseError):
        guard.begin(5)          # a retry is a new round with new keys
    guard.begin(6)
    assert guard.state(5) == "aborted" and guard.state(6) == "begun"
    with pytest.raises(prg.RoundReuseError):
        guard.abort(7)          # cannot abort a round that never began


def test_round_ids_are_checked():
    with pytest.raises(ValueError):
        prg.RoundGuard().begin(-1)


# --------------------------------------------------------------------------
# hygiene
# --------------------------------------------------------------------------

def test_the_secagg_package_uses_no_non_cryptographic_randomness():
    package = Path(prg.__file__).parent
    offenders = []
    for path in package.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        if re.search(r"^\s*(import random|from random import)|np\.random|numpy\.random|default_rng", text, flags=re.MULTILINE):
            offenders.append(path.name)
    assert offenders == []
