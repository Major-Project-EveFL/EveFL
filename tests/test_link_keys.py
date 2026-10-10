"""
Link protection (ADR 0003, layer 1): key combiner, HKDF bound to round and client, AEAD with bound associated data,
single-use keys and best-effort zeroization. Acceptance tests 4, 5 and 6 of the ADR.

The keys in these tests are random bytes made up for the test. Keys from the simulator are seed-determined and
NOT secret (docs/adr/0001); nothing here claims to protect real data.
"""

import itertools
import os

import numpy as np
import pytest

from evefl.crypto import linkkeys
from evefl.crypto.classical import ClassicalCipherSuite
from evefl.crypto.linkkeys import (
    Direction,
    HeaderMismatchError,
    KeyComponent,
    KeyReuseError,
    KeyZeroizedError,
    LinkAuthError,
    SealedMessage,
    combine_key_material,
    derive_link_key,
    link_aad,
    link_info,
    model_digest,
    open_sealed,
    provenance_tag,
    seal,
    shapes_digest,
)
from evefl.quantum.postprocess import RoundKey

SUITE = ClassicalCipherSuite()
MODEL = b"\x11" * 32
SHAPES = b"\x22" * 32
C2S, S2C = Direction.CLIENT_TO_SERVER, Direction.SERVER_TO_CLIENT


def make_round_key(client_id="0", round_id=1, material=None, transcript=None) -> RoundKey:
    return RoundKey(
        client_id=client_id, round_id=round_id, key=bytearray(material if material is not None else os.urandom(64)),
        key_bits=512, qber=0.01, qber_upper=0.02, leaked_bits=0, eps_sec=1e-10, eps_cor=1e-15,
        transcript_sha256=transcript if transcript is not None else os.urandom(32),
    )


def seal_kwargs(**overrides):
    base = dict(state="SECURE", model_hash=MODEL, shapes_digest=SHAPES)
    base.update(overrides)
    return base


def key_pair(client_id="0", round_id=1, direction=C2S, round_key=None):
    """Client end and server end of the same link key (two independent objects, derived separately)."""
    rk = round_key or make_round_key(client_id, round_id)
    return (derive_link_key(rk, direction=direction, role="client", suite=SUITE),
            derive_link_key(rk, direction=direction, role="server", suite=SUITE))


# --------------------------------------------------------------------------
# Encodings are injective (ADR acceptance test 5, direct test of the encoding)
# --------------------------------------------------------------------------

def test_link_info_is_injective_over_a_tricky_domain():
    clients = ["", "0", "1", "10", "01", "a", "ab", "a|b", "a\x00b", "client-0"]
    rounds = [0, 1, 2, 255, 256, 2 ** 32, 2 ** 63]
    encodings = {link_info(d, r, c) for d, r, c in itertools.product([C2S, S2C], rounds, clients)}
    assert len(encodings) == 2 * len(rounds) * len(clients)


def test_link_info_binds_label_direction_round_and_client_visibly():
    base = link_info(C2S, 7, "3")
    assert base.startswith(b"\x00\x00\x00\x10evefl/v1/aes-gcm")
    assert base != link_info(S2C, 7, "3") and base != link_info(C2S, 8, "3") and base != link_info(C2S, 7, "4")


def test_link_aad_is_injective_and_changes_with_every_field():
    base = dict(round_id=5, client_id="2", direction=C2S, state="SECURE", model_hash=MODEL, shapes_digest=SHAPES)
    reference = link_aad(**base)
    changes = dict(round_id=6, client_id="20", direction=S2C, state="CAUTION", model_hash=b"\x12" * 32,
                   shapes_digest=b"\x23" * 32)
    for field, value in changes.items():
        assert link_aad(**{**base, field: value}) != reference, field
    # field boundaries cannot be shifted: moving a byte between adjacent variable-length fields changes the encoding
    assert link_aad(**{**base, "client_id": "2S", "state": "ECURE"}) != reference


def test_digests_depend_on_values_shapes_and_dtypes():
    a = [np.arange(6, dtype=np.float32).reshape(2, 3), np.ones(4, dtype=np.float32)]
    assert model_digest(a) == model_digest([x.copy() for x in a])
    changed = [a[0].copy(), a[1].copy()]
    changed[1][0] = 2.0
    assert model_digest(changed) != model_digest(a) and shapes_digest(changed) == shapes_digest(a)
    assert shapes_digest([a[0].reshape(3, 2), a[1]]) != shapes_digest(a)
    assert shapes_digest([a[0].astype(np.float64), a[1]]) != shapes_digest(a)
    assert len(model_digest(a)) == len(shapes_digest(a)) == 32


# --------------------------------------------------------------------------
# Key combiner: KEM-ready, labelled, length-prefixed, zeroizable
# --------------------------------------------------------------------------

def test_combiner_is_deterministic_labelled_and_returns_a_bytearray():
    ikm = combine_key_material([KeyComponent("qkd", bytearray(b"secret-a"))])
    assert isinstance(ikm, bytearray)
    assert bytes(ikm) == bytes(combine_key_material([KeyComponent("qkd", b"secret-a")]))
    assert b"qkd" in bytes(ikm) and b"secret-a" in bytes(ikm)


def test_combiner_resists_boundary_shifting_and_reordering():
    a = combine_key_material([KeyComponent("qkd", b"ab"), KeyComponent("kem", b"c")])
    b = combine_key_material([KeyComponent("qkd", b"a"), KeyComponent("kem", b"bc")])
    c = combine_key_material([KeyComponent("kem", b"c"), KeyComponent("qkd", b"ab")])
    assert len({bytes(a), bytes(b), bytes(c)}) == 3


def test_combiner_accepts_a_second_component_without_changing_callers_and_validates_input():
    one = combine_key_material([KeyComponent("qkd", b"k")])
    two = combine_key_material([KeyComponent("qkd", b"k"), KeyComponent("kem", b"s")])
    assert bytes(one) != bytes(two)
    with pytest.raises(ValueError, match="at least one"):
        combine_key_material([])
    with pytest.raises(ValueError, match="duplicate"):
        combine_key_material([KeyComponent("qkd", b"k"), KeyComponent("qkd", b"j")])
    with pytest.raises(ValueError, match="empty"):
        combine_key_material([KeyComponent("qkd", b"")])


def test_provenance_tag_says_qkd_only_for_the_single_component_case_and_makes_no_hybrid_claim():
    assert provenance_tag([KeyComponent("qkd", b"k")]) == "qkd-only"
    tag = provenance_tag([KeyComponent("qkd", b"k"), KeyComponent("kem", b"s")])
    assert tag == "qkd+kem" and "post-quantum" not in tag and "hybrid" not in tag


# --------------------------------------------------------------------------
# Derivation: one fresh key per client, round and direction (ADR acceptance test 5)
# --------------------------------------------------------------------------

def test_same_key_material_gives_different_keys_for_different_client_round_or_direction():
    """Isolates the binding: identical key bytes and transcript, only the labels differ."""
    material, transcript = os.urandom(64), os.urandom(32)

    def derived(client, round_id, direction):
        rk = make_round_key(client, round_id, material=material, transcript=transcript)
        return bytes(derive_link_key(rk, direction=direction, role="client", suite=SUITE)._key)

    keys = {derived(c, r, d) for c in ("0", "1", "2") for r in (1, 2, 3) for d in (C2S, S2C)}
    assert len(keys) == 3 * 3 * 2
    assert derived("0", 1, C2S) == derived("0", 1, C2S)  # deterministic given the same inputs


def test_every_key_over_clients_times_rounds_is_distinct():
    keys = []
    for client in range(6):
        for rnd in range(1, 9):
            rk = make_round_key(str(client), rnd)
            keys.append(bytes(derive_link_key(rk, direction=C2S, role="client", suite=SUITE)._key))
    assert len(set(keys)) == 6 * 8 and all(len(k) == 32 for k in keys)


def test_the_transcript_salt_changes_the_key_and_a_missing_transcript_is_allowed():
    material = os.urandom(64)
    a = derive_link_key(make_round_key(material=material, transcript=b"\x01" * 32), direction=C2S, role="client", suite=SUITE)
    b = derive_link_key(make_round_key(material=material, transcript=b"\x02" * 32), direction=C2S, role="client", suite=SUITE)
    c = derive_link_key(make_round_key(material=material, transcript=b""), direction=C2S, role="client", suite=SUITE)
    assert len({bytes(a._key), bytes(b._key), bytes(c._key)}) == 3


def test_client_and_server_ends_derive_the_same_key_independently():
    client_end, server_end = key_pair()
    assert client_end._key is not server_end._key and bytes(client_end._key) == bytes(server_end._key)
    assert client_end.role == "client" and server_end.role == "server"


def test_derivation_rejects_unknown_roles_and_zeroized_round_keys():
    rk = make_round_key()
    with pytest.raises(ValueError, match="role"):
        derive_link_key(rk, direction=C2S, role="eavesdropper", suite=SUITE)
    rk.zeroize()
    with pytest.raises(KeyZeroizedError):
        derive_link_key(rk, direction=C2S, role="client", suite=SUITE)


# --------------------------------------------------------------------------
# AEAD with bound associated data (ADR acceptance test 4)
# --------------------------------------------------------------------------

def test_roundtrip_and_ciphertext_does_not_contain_the_plaintext():
    client_end, server_end = key_pair()
    plaintext = b"model update bytes " * 20
    message = seal(client_end, plaintext, **seal_kwargs())
    assert plaintext not in message.ciphertext and len(message.nonce) == 12
    assert open_sealed(server_end, message, **seal_kwargs()) == plaintext


def test_header_fields_are_visible_and_describe_the_message():
    client_end, _ = key_pair("4", 9)
    message = seal(client_end, b"x", **seal_kwargs())
    assert (message.round_id, message.client_id, message.direction) == (9, "4", C2S)


@pytest.mark.parametrize("position", [0, 5, -1])
def test_flipping_any_ciphertext_bit_is_rejected(position):
    client_end, server_end = key_pair()
    message = seal(client_end, b"payload-" * 8, **seal_kwargs())
    tampered = bytearray(message.ciphertext)
    tampered[position] ^= 0x01
    with pytest.raises(LinkAuthError):
        open_sealed(server_end, SealedMessage(message.round_id, message.client_id, message.direction,
                                              message.nonce, bytes(tampered)), **seal_kwargs())


def test_tampered_nonce_is_rejected():
    client_end, server_end = key_pair()
    message = seal(client_end, b"payload", **seal_kwargs())
    nonce = bytearray(message.nonce)
    nonce[0] ^= 0x01
    with pytest.raises(LinkAuthError):
        open_sealed(server_end, SealedMessage(message.round_id, message.client_id, message.direction,
                                              bytes(nonce), message.ciphertext), **seal_kwargs())


@pytest.mark.parametrize("field, wrong", [
    ("state", "CAUTION"), ("model_hash", b"\x99" * 32), ("shapes_digest", b"\x98" * 32),
])
def test_wrong_associated_data_at_open_is_rejected(field, wrong):
    client_end, server_end = key_pair()
    message = seal(client_end, b"payload", **seal_kwargs())
    with pytest.raises(LinkAuthError):
        open_sealed(server_end, message, **seal_kwargs(**{field: wrong}))


def test_a_header_that_disagrees_with_the_key_is_rejected_before_decryption():
    client_end, server_end = key_pair("0", 1)
    message = seal(client_end, b"payload", **seal_kwargs())
    for forged in (SealedMessage(2, "0", C2S, message.nonce, message.ciphertext),
                   SealedMessage(1, "9", C2S, message.nonce, message.ciphertext),
                   SealedMessage(1, "0", S2C, message.nonce, message.ciphertext)):
        with pytest.raises(HeaderMismatchError):
            open_sealed(server_end, forged, **seal_kwargs())


def test_another_clients_key_does_not_decrypt_even_if_the_header_is_rewritten():
    c0_client, _ = key_pair("0", 1)
    _, c1_server = key_pair("1", 1)
    message = seal(c0_client, b"payload", **seal_kwargs())
    with pytest.raises(HeaderMismatchError):
        open_sealed(c1_server, message, **seal_kwargs())
    rewritten = SealedMessage(1, "1", C2S, message.nonce, message.ciphertext)
    with pytest.raises(LinkAuthError):
        open_sealed(c1_server, rewritten, **seal_kwargs())


def test_other_rounds_key_does_not_decrypt_even_if_the_header_is_rewritten():
    material, transcript = os.urandom(64), os.urandom(32)
    r1_client = derive_link_key(make_round_key("0", 1, material, transcript), direction=C2S, role="client", suite=SUITE)
    r2_server = derive_link_key(make_round_key("0", 2, material, transcript), direction=C2S, role="server", suite=SUITE)
    message = seal(r1_client, b"payload", **seal_kwargs())
    with pytest.raises(LinkAuthError):
        open_sealed(r2_server, SealedMessage(2, "0", C2S, message.nonce, message.ciphertext), **seal_kwargs())


def test_the_two_directions_use_different_keys():
    rk = make_round_key()
    c2s = derive_link_key(rk, direction=C2S, role="client", suite=SUITE)
    s2c_server = derive_link_key(rk, direction=S2C, role="server", suite=SUITE)
    message = seal(c2s, b"payload", **seal_kwargs())
    with pytest.raises(HeaderMismatchError):
        open_sealed(s2c_server, message, **seal_kwargs())
    assert bytes(c2s._key) != bytes(s2c_server._key)


def test_nonces_are_random_and_do_not_repeat_across_keys():
    nonces = {seal(key_pair("0", r)[0], b"p", **seal_kwargs()).nonce for r in range(1, 40)}
    assert len(nonces) == 39


# --------------------------------------------------------------------------
# Single use: a second key or message for the same (client, round, direction) is refused
# --------------------------------------------------------------------------

def test_a_key_seals_at_most_one_message_and_opens_at_most_one():
    client_end, server_end = key_pair()
    message = seal(client_end, b"payload", **seal_kwargs())
    with pytest.raises(KeyReuseError):
        seal(client_end, b"second", **seal_kwargs())
    assert open_sealed(server_end, message, **seal_kwargs()) == b"payload"
    with pytest.raises(KeyReuseError):  # a replay of the same message
        open_sealed(server_end, message, **seal_kwargs())


def test_a_failed_open_does_not_burn_the_key_but_a_successful_one_does():
    client_end, server_end = key_pair()
    message = seal(client_end, b"payload", **seal_kwargs())
    with pytest.raises(LinkAuthError):
        open_sealed(server_end, message, **seal_kwargs(state="CAUTION"))
    assert open_sealed(server_end, message, **seal_kwargs()) == b"payload"


# --------------------------------------------------------------------------
# Zeroization (best effort, ADR acceptance test 6): our own buffers only
# --------------------------------------------------------------------------

def test_zeroize_overwrites_the_link_key_buffer_and_blocks_further_use():
    client_end, _ = key_pair()
    assert not client_end.is_zero()
    buffer = client_end._key
    client_end.zeroize()
    assert client_end.is_zero() and all(b == 0 for b in buffer) and client_end.zeroized
    with pytest.raises(KeyZeroizedError):
        seal(client_end, b"payload", **seal_kwargs())


def test_the_combined_key_material_buffer_is_zeroed_after_derivation(monkeypatch):
    captured = []
    real = linkkeys.combine_key_material

    def spy(components):
        buffer = real(components)
        captured.append(buffer)
        return buffer

    monkeypatch.setattr(linkkeys, "combine_key_material", spy)
    derive_link_key(make_round_key(), direction=C2S, role="client", suite=SUITE)
    assert len(captured) == 1 and len(captured[0]) > 0 and all(b == 0 for b in captured[0])


def test_repr_never_prints_key_material():
    client_end, _ = key_pair()
    text = repr(client_end)
    assert bytes(client_end._key).hex() not in text and "client" in text and "qkd-only" in text


# --------------------------------------------------------------------------
# Wire format round trip
# --------------------------------------------------------------------------

def test_sealed_message_serialises_and_parses_back_exactly():
    client_end, server_end = key_pair("12", 77)
    message = seal(client_end, b"payload" * 100, **seal_kwargs())
    parsed = SealedMessage.from_bytes(message.to_bytes())
    assert parsed == message
    assert open_sealed(server_end, parsed, **seal_kwargs()) == b"payload" * 100


@pytest.mark.parametrize("cut", [0, 3, 10])
def test_truncated_wire_data_is_rejected_not_misparsed(cut):
    client_end, _ = key_pair()
    wire = seal(client_end, b"payload", **seal_kwargs()).to_bytes()
    with pytest.raises(ValueError):
        SealedMessage.from_bytes(wire[:cut] if cut else b"")
    with pytest.raises(ValueError):
        SealedMessage.from_bytes(wire + b"\x00")
