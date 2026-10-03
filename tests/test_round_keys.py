"""
The controller's QBER exchange and the key block are different exchanges (docs/adr/0001): the controller can
say SECURE while the key exchange aborts or yields no key, and the policy then discards the round with a
separate reason.
"""

import numpy as np
import pytest

from evefl.orchestration.policy import (
    REASON_KEY_ABORT,
    REASON_NO_KEY,
    REASON_NONE,
    apply_key_policy,
)
from evefl.orchestration.state_machine import SecurityState, StateController
from evefl.quantum import postprocess as pp
from evefl.quantum.base import ChannelModel
from evefl.quantum.bb84_numpy import BB84NumpyProtocol
from evefl.quantum.round_keys import KeyBlockConfig, generate_round_key
from evefl.quantum.seeding import qkd_seed_sequence

AUTH_KEY = b"\x05" * 32
SEED = 11


def _controller_state(channel: ChannelModel, round_id: int = 1, cid: str = "0") -> SecurityState:
    """The controller's own exchange: headline block (1024 qubits, 25% sample), controller seed stream."""
    protocol = BB84NumpyProtocol(sample_fraction=0.25, seed=qkd_seed_sequence(SEED, round_id, cid))
    return StateController().classify(protocol.run_exchange(1024, channel=channel).qber)


# --------------------------------------------------------------------------
# Independence of the two exchanges
# --------------------------------------------------------------------------

def test_controller_stream_is_unchanged_so_validated_results_are_reproduced():
    expected = np.random.SeedSequence(entropy=[SEED, 3, 2])
    assert qkd_seed_sequence(SEED, 3, "2").entropy == expected.entropy
    assert qkd_seed_sequence(SEED, 3, "2", purpose="controller").entropy == expected.entropy


def test_key_stream_differs_from_the_controller_stream_and_bad_purpose_is_rejected():
    assert qkd_seed_sequence(SEED, 3, "2", purpose="key").entropy != qkd_seed_sequence(SEED, 3, "2").entropy
    with pytest.raises(ValueError, match="purpose"):
        qkd_seed_sequence(SEED, 3, "2", purpose="other")


def test_controller_and_key_exchanges_are_different_measurements():
    channel = ChannelModel(0.2, 0.0)
    controller = BB84NumpyProtocol(sample_fraction=0.25, seed=qkd_seed_sequence(SEED, 1, "0")).run_exchange(
        1024, channel=channel)
    key_block = BB84NumpyProtocol(sample_fraction=0.1, seed=qkd_seed_sequence(SEED, 1, "0", purpose="key")).run_exchange(
        1024, channel=channel)
    assert controller.sifted_key != key_block.sifted_key
    assert controller.sample_indices != key_block.sample_indices


def test_key_block_config_defaults_and_validation():
    assert KeyBlockConfig().n_qubits == 2 ** 17
    for bad in ({"n_qubits": 0}, {"sample_fraction": 0.0}, {"sample_fraction": 1.0}):
        with pytest.raises(ValueError):
            KeyBlockConfig(**bad)


# --------------------------------------------------------------------------
# Controller SECURE, key exchange fails -> policy discards the round
# --------------------------------------------------------------------------

def test_controller_says_secure_while_the_key_exchange_aborts_for_high_qber():
    """Eve attacks only the key block: the controller's own exchange is clean."""
    state = _controller_state(ChannelModel(0.0, 0.0))
    assert state == SecurityState.SECURE
    key = generate_round_key(experiment_seed=SEED, server_round=1, client_id="0", channel=ChannelModel(0.5, 0.0),
                             auth_key=AUTH_KEY, ec_seed=1)
    assert key.status == pp.STATUS_ABORT_QBER and key.round_key is None

    decision = apply_key_policy(state, {"0": key.status})
    assert decision.state == SecurityState.LOCKDOWN and decision.discard_round
    assert decision.reason == REASON_KEY_ABORT and decision.failed_clients == ["0"]


def test_controller_says_secure_while_the_key_block_yields_no_key():
    """A QBER the controller cannot see (it only samples 128 bits) but the finite-key bound punishes."""
    state = _controller_state(ChannelModel(0.0, 0.0))
    assert state == SecurityState.SECURE
    key = generate_round_key(experiment_seed=SEED, server_round=2, client_id="1", channel=ChannelModel(0.3, 0.0),
                             auth_key=AUTH_KEY, ec_seed=1)
    assert key.status == pp.STATUS_NO_KEY and key.round_key is None

    decision = apply_key_policy(state, {"0": "ok", "1": key.status, "2": "ok"})
    assert decision.state == SecurityState.LOCKDOWN and decision.reason == REASON_NO_KEY
    assert decision.failed_clients == ["1"]


def test_healthy_key_exchange_leaves_the_controller_state_alone():
    state = _controller_state(ChannelModel(0.0, 0.01))
    key = generate_round_key(experiment_seed=SEED, server_round=3, client_id="0", channel=ChannelModel(0.0, 0.01),
                             auth_key=AUTH_KEY, ec_seed=1)
    assert key.ok and key.round_key is not None and key.round_key.key_bits >= 256
    decision = apply_key_policy(state, {"0": key.status})
    assert decision.state == state == SecurityState.SECURE and decision.reason == REASON_NONE


def test_key_exchange_is_reproducible_in_its_raw_exchange_but_keys_stay_random():
    kwargs = dict(experiment_seed=SEED, server_round=4, client_id="0", channel=ChannelModel(0.0, 0.01),
                  auth_key=AUTH_KEY, ec_seed=1)
    a, b = generate_round_key(**kwargs), generate_round_key(**kwargs)
    assert a.details["n_sifted"] == b.details["n_sifted"] and a.details["qber"] == b.details["qber"]
    assert bytes(a.round_key.key) != bytes(b.round_key.key)  # privacy-amplification seeds come from secrets
