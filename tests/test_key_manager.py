"""
KeyManager (ADR 0003, step 1): one key per client per round, status for the policy, release with zeroization, and no
unprotected fallback. Acceptance tests 4 to 7 of the ADR at the manager level.

Most tests use a fake provider with random keys. The tests marked "real" run the simulated QKD pipeline (about a second
each); those keys are seed-determined and NOT secret.
"""

import os
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import pytest
import yaml

from evefl.crypto import keymanager
from evefl.crypto.classical import ClassicalCipherSuite
from evefl.crypto.keymanager import (
    KeyManager,
    KeyProvider,
    NoKeyError,
    ProtectedReceiver,
    ProtectedSender,
    SimulatedQKDKeyProvider,
    create_key_manager,
    key_provider_registry,
)
from evefl.crypto.linkkeys import Direction, KeyReuseError, LinkAuthError, SealedMessage, open_sealed, seal
from evefl.orchestration.policy import KEY_OK
from evefl.quantum import postprocess as pp
from evefl.quantum.base import ChannelModel
from evefl.quantum.postprocess import PostprocessResult, RoundKey

C2S, S2C = Direction.CLIENT_TO_SERVER, Direction.SERVER_TO_CLIENT
MODEL, SHAPES = b"\x11" * 32, b"\x22" * 32
AD = dict(state="SECURE", model_hash=MODEL, shapes_digest=SHAPES)


class FakeProvider(KeyProvider):
    """Random 64-byte keys for healthy clients; a fixed failing status for the clients in `failing`."""

    def __init__(self, failing: Optional[Mapping[str, str]] = None):
        self.failing = dict(failing or {})
        self.calls: List[tuple] = []
        self.issued: List[RoundKey] = []

    @property
    def name(self) -> str:
        return "fake"

    def generate(self, client_id: str, round_id: int) -> PostprocessResult:
        self.calls.append((client_id, round_id))
        status = self.failing.get(client_id)
        if status is not None:
            return PostprocessResult(status, f"fake {status}", None, {})
        round_key = RoundKey(client_id=client_id, round_id=round_id, key=bytearray(os.urandom(64)), key_bits=512,
                             qber=0.01, qber_upper=0.02, leaked_bits=0, eps_sec=1e-10, eps_cor=1e-15,
                             transcript_sha256=os.urandom(32))
        self.issued.append(round_key)
        return PostprocessResult(pp.STATUS_OK, "ok", round_key, {})


class SpyTransport:
    def __init__(self):
        self.sent: List[bytes] = []

    def send(self, data: bytes) -> None:
        self.sent.append(data)


# --------------------------------------------------------------------------
# Issue: one key per (client, round); statuses the policy understands
# --------------------------------------------------------------------------

def test_issue_returns_an_ok_status_and_a_handle_for_a_healthy_exchange():
    manager = KeyManager(FakeProvider())
    issue = manager.issue("0", 1)
    assert issue.ok and issue.status == KEY_OK == "ok" and issue.handle is not None
    assert (issue.client_id, issue.round_id) == ("0", 1)
    assert manager.handle("0", 1) is issue.handle


def test_a_second_key_for_the_same_client_and_round_is_refused_even_after_release():
    provider = FakeProvider()
    manager = KeyManager(provider)
    manager.issue("0", 1).handle.release()
    with pytest.raises(KeyReuseError):
        manager.issue("0", 1)
    assert provider.calls == [("0", 1)]  # the provider was not asked again
    assert manager.issue("0", 2).ok and manager.issue("1", 1).ok


def test_failures_are_passed_to_the_policy_unchanged_and_carry_no_handle():
    manager = KeyManager(FakeProvider({"1": pp.STATUS_NO_KEY, "2": pp.STATUS_ABORT_QBER}))
    assert manager.issue("0", 1).ok
    for cid, status in (("1", "no_key"), ("2", "abort_qber_high")):
        issue = manager.issue(cid, 1)
        assert not issue.ok and issue.status == status and issue.handle is None
        with pytest.raises(NoKeyError):
            manager.handle(cid, 1)


def test_a_failed_exchange_still_uses_up_the_client_round_pair():
    manager = KeyManager(FakeProvider({"1": pp.STATUS_NO_KEY}))
    manager.issue("1", 1)
    with pytest.raises(KeyReuseError):
        manager.issue("1", 1)


def test_status_fn_matches_the_strategy_hook_and_issues_each_key_once():
    provider = FakeProvider({"2": pp.STATUS_NO_KEY})
    manager = KeyManager(provider)
    assert manager.key_status_fn(1, "0") == "ok" and manager.key_status_fn(1, "2") == "no_key"
    assert provider.calls == [("0", 1), ("2", 1)]
    assert manager.handle("0", 1) is not None  # the key made for the status is the one used for the round


def test_handle_for_a_client_that_was_never_issued_is_refused():
    with pytest.raises(NoKeyError):
        KeyManager(FakeProvider()).handle("0", 1)


# --------------------------------------------------------------------------
# Handles: ends, directions, uniqueness over clients x rounds
# --------------------------------------------------------------------------

def test_end_to_end_both_directions_between_the_two_ends():
    handle = KeyManager(FakeProvider()).issue("0", 1).handle
    sender = handle.link_key(C2S, "client")
    receiver = handle.link_key(C2S, "server")
    assert open_sealed(receiver, seal(sender, b"update", **AD), **AD) == b"update"
    down_sender, down_receiver = handle.link_key(S2C, "server"), handle.link_key(S2C, "client")
    assert open_sealed(down_receiver, seal(down_sender, b"model", **AD), **AD) == b"model"


def test_each_direction_and_role_key_can_be_taken_only_once():
    handle = KeyManager(FakeProvider()).issue("0", 1).handle
    handle.link_key(C2S, "client")
    with pytest.raises(KeyReuseError):
        handle.link_key(C2S, "client")
    handle.link_key(C2S, "server")  # the other end of the same direction is a different key object
    with pytest.raises(ValueError, match="role"):
        handle.link_key(C2S, "observer")


def test_all_keys_over_clients_times_rounds_times_directions_are_distinct():
    manager = KeyManager(FakeProvider())
    keys = []
    for client in range(5):
        for rnd in range(1, 9):
            handle = manager.issue(str(client), rnd).handle
            for direction in (C2S, S2C):
                keys.append(bytes(handle.link_key(direction, "client")._key))
    assert len(keys) == 5 * 8 * 2 and len(set(keys)) == len(keys)


def test_handle_metadata_is_labelled_simulated_and_qkd_only():
    handle = KeyManager(FakeProvider()).issue("0", 1).handle
    assert handle.provenance == "qkd-only" and handle.simulated is True
    assert "qkd-only" in repr(handle) and "simulated" in repr(handle)


# --------------------------------------------------------------------------
# Release: zeroization of everything the manager holds (best effort)
# --------------------------------------------------------------------------

def test_release_zeroizes_the_round_key_and_every_derived_link_key():
    provider = FakeProvider()
    manager = KeyManager(provider)
    handle = manager.issue("0", 1).handle
    keys = [handle.link_key(C2S, "client"), handle.link_key(C2S, "server"), handle.link_key(S2C, "client")]
    round_key = provider.issued[0]
    assert not any(k.is_zero() for k in keys) and not round_key.zeroized
    handle.release()
    assert all(k.is_zero() and k.zeroized for k in keys)
    assert round_key.zeroized and all(b == 0 for b in round_key.key)
    assert handle.released
    with pytest.raises(NoKeyError):
        manager.handle("0", 1)
    with pytest.raises(NoKeyError):
        handle.link_key(S2C, "server")


def test_release_is_idempotent_and_release_all_covers_every_open_handle():
    provider = FakeProvider()
    manager = KeyManager(provider)
    handles = [manager.issue(str(c), 1).handle for c in range(3)]
    handles[0].release()
    handles[0].release()
    manager.release_round(1)
    assert all(h.released for h in handles) and all(rk.zeroized for rk in provider.issued)


# --------------------------------------------------------------------------
# No fallback: without a key nothing leaves in the clear (ADR acceptance test 7)
# --------------------------------------------------------------------------

def test_sender_without_a_key_raises_and_sends_nothing():
    manager = KeyManager(FakeProvider({"1": pp.STATUS_NO_KEY}))
    transport = SpyTransport()
    sender = ProtectedSender(manager, transport)
    manager.issue("1", 1)
    with pytest.raises(NoKeyError):
        sender.send("1", 1, b"model update", direction=C2S, **AD)
    with pytest.raises(NoKeyError):  # never issued at all
        sender.send("7", 1, b"model update", direction=C2S, **AD)
    assert transport.sent == []


def test_sender_after_release_raises_and_sends_nothing():
    manager = KeyManager(FakeProvider())
    transport = SpyTransport()
    manager.issue("0", 1).handle.release()
    with pytest.raises(NoKeyError):
        ProtectedSender(manager, transport).send("0", 1, b"secret-update", direction=C2S, **AD)
    assert transport.sent == []


def test_sender_with_a_key_sends_exactly_one_protected_message_and_the_receiver_opens_it():
    manager = KeyManager(FakeProvider())
    manager.issue("0", 1)
    transport = SpyTransport()
    plaintext = b"secret-model-update " * 30
    ProtectedSender(manager, transport).send("0", 1, plaintext, direction=C2S, **AD)
    assert len(transport.sent) == 1 and plaintext not in transport.sent[0] and b"secret-model" not in transport.sent[0]
    assert ProtectedReceiver(manager).receive(transport.sent[0], **AD) == plaintext


def test_a_second_send_for_the_same_client_round_and_direction_is_refused():
    manager = KeyManager(FakeProvider())
    manager.issue("0", 1)
    transport = SpyTransport()
    sender = ProtectedSender(manager, transport)
    sender.send("0", 1, b"first", direction=C2S, **AD)
    with pytest.raises(KeyReuseError):
        sender.send("0", 1, b"second", direction=C2S, **AD)
    assert len(transport.sent) == 1


def test_receiver_rejects_a_tampered_message_and_a_replay():
    manager = KeyManager(FakeProvider())
    manager.issue("0", 1)
    transport = SpyTransport()
    ProtectedSender(manager, transport).send("0", 1, b"payload", direction=C2S, **AD)
    wire = transport.sent[0]
    receiver = ProtectedReceiver(manager)
    flipped = bytearray(wire)
    flipped[-1] ^= 0x01
    with pytest.raises(LinkAuthError):
        receiver.receive(bytes(flipped), **AD)
    assert receiver.receive(wire, **AD) == b"payload"
    with pytest.raises(KeyReuseError):
        receiver.receive(wire, **AD)


def test_receiver_cannot_be_fooled_by_a_header_naming_another_clients_round():
    manager = KeyManager(FakeProvider())
    manager.issue("0", 1)
    manager.issue("1", 1)
    transport = SpyTransport()
    ProtectedSender(manager, transport).send("0", 1, b"from-zero", direction=C2S, **AD)
    message = SealedMessage.from_bytes(transport.sent[0])
    forged = SealedMessage(1, "1", C2S, message.nonce, message.ciphertext).to_bytes()
    with pytest.raises(LinkAuthError):
        ProtectedReceiver(manager).receive(forged, **AD)


# --------------------------------------------------------------------------
# Provider registry and configuration
# --------------------------------------------------------------------------

def test_simulated_qkd_provider_is_registered_and_selected_by_name():
    assert "simulated_qkd" in key_provider_registry.list_keys()
    assert key_provider_registry.get("simulated_qkd") is SimulatedQKDKeyProvider
    with pytest.raises(KeyError):
        key_provider_registry.get("no-such-provider")


def test_create_key_manager_reads_the_crypto_section_and_rejects_unknown_providers():
    settings = {"crypto": {"key_provider": "simulated_qkd", "key_block": {"n_qubits": 2 ** 14, "sample_fraction": 0.1}}}
    manager = create_key_manager(settings, experiment_seed=3, channel_for=lambda cid, rnd: ChannelModel(0.0, 0.01))
    assert isinstance(manager.provider, SimulatedQKDKeyProvider)
    assert manager.provider.config.n_qubits == 2 ** 14
    with pytest.raises(KeyError):
        create_key_manager({"crypto": {"key_provider": "nope"}}, experiment_seed=3,
                           channel_for=lambda cid, rnd: ChannelModel())


def test_the_default_settings_file_names_the_simulated_provider():
    path = Path(keymanager.__file__).resolve().parents[1] / "config" / "settings.yaml"
    section = yaml.safe_load(path.read_text(encoding="utf-8"))["crypto"]
    assert section["key_provider"] == "simulated_qkd" and section["key_block"]["n_qubits"] == 2 ** 17


# --------------------------------------------------------------------------
# Real (simulated QKD pipeline): about a second each, keys are seed-determined and not secret
# --------------------------------------------------------------------------

def _real_manager(alpha_by_client: Optional[Dict[str, float]] = None) -> KeyManager:
    alphas = alpha_by_client or {}
    provider = SimulatedQKDKeyProvider(
        experiment_seed=11, channel_for=lambda cid, rnd: ChannelModel(alphas.get(cid, 0.0), 0.01),
        auth_key=os.urandom(32), ec_seed=1)
    return KeyManager(provider, ClassicalCipherSuite())


def test_real_healthy_link_gives_a_usable_key_and_an_attacked_link_gives_none():
    manager = _real_manager({"1": 0.5})
    healthy, attacked = manager.issue("0", 1), manager.issue("1", 1)
    assert healthy.ok and healthy.handle.simulated and healthy.handle.provenance == "qkd-only"
    assert attacked.status == pp.STATUS_ABORT_QBER and attacked.handle is None
    sender, receiver = healthy.handle.link_key(C2S, "client"), healthy.handle.link_key(C2S, "server")
    assert open_sealed(receiver, seal(sender, b"update", **AD), **AD) == b"update"


def test_real_keys_differ_between_clients_and_rounds():
    manager = _real_manager()
    keys = [bytes(manager.issue(c, r).handle.link_key(C2S, "client")._key) for c, r in (("0", 1), ("1", 1), ("0", 2))]
    assert len(set(keys)) == 3
