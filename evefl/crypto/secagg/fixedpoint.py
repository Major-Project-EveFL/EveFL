"""
64-bit fixed-point arithmetic for masked aggregation (ADR 0003, "Fixed-point arithmetic").

A real number x is stored as the two's-complement uint64 of round(x * 2^32): 32 fractional bits. Sums are taken modulo 2^64, which
is what lets pairwise masks cancel exactly. The representable SUM range is [-2^31, 2^31); a value outside it wraps silently into a
plausible-looking wrong number, so range is asserted, not assumed:

- before masking, every client calls `assert_encodable(x, n_participants)`: every element finite and |x| * n < 2^31, so that the sum
  of n such values cannot wrap;
- after unmasking, the aggregator calls `check_aggregate`, which aborts if any element of the decoded sum has magnitude of at
  least `AGGREGATE_GUARD` = 2^30. Honest weighted deltas are many orders of magnitude smaller. This catches a wrap that lands
  far from zero; a wrap that happens to land near zero cannot be detected from the sum alone, which is why the client-side
  assertion (and, later, the aggregate-norm rule) is the primary defence.

Units: values are dimensionless model-parameter deltas (float32/float64 in, uint64 out). Rounding error of one encoded value is at
most half a grid unit, 2^-33; the sum of n values differs from the real sum by at most n * 2^-33. Exactness of the float64
intermediate: a value is exact on the grid for |x| < 2^21; above that, low bits are rounded by float64 itself.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

FRACTIONAL_BITS = 32
MODULUS_BITS = 64
SCALE = 2 ** FRACTIONAL_BITS
LIMIT = 2 ** (MODULUS_BITS - FRACTIONAL_BITS - 1)   # 2^31: the sum range is [-LIMIT, LIMIT)
AGGREGATE_GUARD = 2.0 ** 30                          # magnitude at which a decoded aggregate is treated as a suspected wrap


class RangeError(ValueError):
    """A value or an aggregate is outside the safe fixed-point range (abort reason `secagg_range`)."""


def _as_float64(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def assert_encodable(x: np.ndarray, n_participants: int) -> None:
    """Raise RangeError unless every element is finite and |x| * n_participants < 2^31 (so the sum of n values cannot wrap)."""
    if n_participants < 1:
        raise ValueError("n_participants must be at least 1")
    values = _as_float64(x)
    if not np.all(np.isfinite(values)):
        raise RangeError("non-finite value (NaN or infinity) in an update")
    if values.size and np.max(np.abs(values)) * n_participants >= LIMIT:
        raise RangeError(f"|element| * {n_participants} reaches 2^31; the sum would wrap in 64-bit fixed point")


def encode(x: np.ndarray) -> np.ndarray:
    """Encode floats as two's-complement uint64 with 32 fractional bits (same shape). Single values must lie in [-2^31, 2^31)."""
    values = _as_float64(x)
    if not np.all(np.isfinite(values)):
        raise RangeError("non-finite value cannot be encoded")
    if values.size and (np.max(values) >= LIMIT or np.min(values) < -LIMIT):
        raise RangeError("value outside [-2^31, 2^31) cannot be encoded")
    return np.rint(values * SCALE).astype(np.int64).view(np.uint64)


def decode(u: np.ndarray) -> np.ndarray:
    """Decode uint64 fixed-point values to float64 (the inverse of `encode` up to half a grid unit)."""
    return np.asarray(u, dtype=np.uint64).view(np.int64).astype(np.float64) / SCALE


def add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise sum modulo 2^64."""
    with np.errstate(over="ignore"):
        return np.add(np.asarray(a, dtype=np.uint64), np.asarray(b, dtype=np.uint64), dtype=np.uint64)


def sub(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise difference modulo 2^64."""
    with np.errstate(over="ignore"):
        return np.subtract(np.asarray(a, dtype=np.uint64), np.asarray(b, dtype=np.uint64), dtype=np.uint64)


def sum_encoded(parts: Sequence[np.ndarray]) -> np.ndarray:
    """Sum of encoded arrays modulo 2^64."""
    if not parts:
        raise ValueError("nothing to sum")
    total = np.asarray(parts[0], dtype=np.uint64).copy()
    for part in parts[1:]:
        total = add(total, part)
    return total


def check_aggregate(u: np.ndarray) -> None:
    """Abort (RangeError) if any element of the decoded aggregate has magnitude of at least AGGREGATE_GUARD (2^30)."""
    decoded = decode(u)
    if decoded.size and np.max(np.abs(decoded)) >= AGGREGATE_GUARD:
        raise RangeError("aggregate element close to the fixed-point boundary: suspected wrap-around, round aborted")
