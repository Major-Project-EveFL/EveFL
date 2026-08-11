import os

import pytest

from evefl.crypto.classical import ClassicalCipherSuite


def test_derive_key_length():
    suite = ClassicalCipherSuite()
    key = suite.derive_key(os.urandom(32), length=32)
    assert len(key) == 32


def test_derive_key_deterministic_for_same_input():
    suite = ClassicalCipherSuite()
    material = b"fixed-test-material-not-random"
    key1 = suite.derive_key(material, info=b"round-1")
    key2 = suite.derive_key(material, info=b"round-1")
    assert key1 == key2


def test_derive_key_differs_with_info():
    suite = ClassicalCipherSuite()
    material = os.urandom(32)
    key1 = suite.derive_key(material, info=b"round-1")
    key2 = suite.derive_key(material, info=b"round-2")
    assert key1 != key2


def test_encrypt_decrypt_round_trip():
    suite = ClassicalCipherSuite()
    key = suite.derive_key(os.urandom(32))
    plaintext = b"fake model weights payload"

    payload = suite.encrypt(plaintext, key)
    recovered = suite.decrypt(payload, key)

    assert recovered == plaintext


def test_decrypt_fails_with_wrong_key():
    suite = ClassicalCipherSuite()
    key = suite.derive_key(os.urandom(32))
    wrong_key = suite.derive_key(os.urandom(32))
    payload = suite.encrypt(b"secret", key)

    with pytest.raises(Exception):
        suite.decrypt(payload, wrong_key)
