"""Injective byte encodings shared by the secure-aggregation modules: big-endian fixed-width integers and 4-byte length prefixes."""

from __future__ import annotations


def lp(data: bytes) -> bytes:
    """Length-prefixed field: a 4-byte big-endian length followed by the bytes (unambiguous when concatenated)."""
    if len(data) >= 2 ** 32:
        raise ValueError("field too long to encode")
    return len(data).to_bytes(4, "big") + data


def u32(value: int) -> bytes:
    """Unsigned 32-bit big-endian integer."""
    if not 0 <= value < 2 ** 32:
        raise ValueError(f"integer out of range for a 32-bit field: {value}")
    return value.to_bytes(4, "big")


def u64(value: int) -> bytes:
    """Unsigned 64-bit big-endian integer."""
    if not 0 <= value < 2 ** 64:
        raise ValueError(f"integer out of range for a 64-bit field: {value}")
    return value.to_bytes(8, "big")
