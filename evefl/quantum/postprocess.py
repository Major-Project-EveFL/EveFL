"""
Classical post-processing of a BB84 exchange: parameter estimation, error correction, privacy
amplification and authentication, with a finite-key bound.

SIMULATION CAVEAT. The raw key comes from the simulator's seeded RNG, so a `RoundKey` produced here is a
deterministic function of the simulation seed: it is NOT secret and must never be used to protect real data.
Only the post-processing randomness that a real system would draw (the hash seeds, the pre-shared
authentication key) uses `secrets` / `os.urandom`. This module implements the PIPELINE and its key-length
accounting so the rest of EveFL can be built against the real interfaces.

Pipeline (docs/03, "Required key pipeline")
1. Sift (done by the QKD backend) -> n_key + k bits.
2. Parameter estimation: the k publicly compared bits give lambda = errors / k. They are DISCARDED from the
   key (earlier code left them in). Abort if lambda >= qber_abort (default 0.11, compared with >=).
3. Error correction: Cascade on the remaining n_key bits; every disclosed parity bit is counted (leak_EC).
   Then a verification hash of ceil(log2(1/eps_cor)) bits (random Toeplitz, a universal2 family) must match.
4. Privacy amplification: random Toeplitz hashing to l bits, where l comes from the finite-key bound below.
   If l <= 0 the round yields NO KEY (status "no_key").
5. Authentication of the classical transcript with HMAC-SHA256 under a pre-shared key. This is computational,
   not the information-theoretic Wegman-Carter authentication a real QKD system needs.

Finite-key bound (Tomamichel, Lim, Gisin, Renner, "Tight finite-key analysis for quantum cryptography",
Nat. Commun. 3, 634 (2012), arXiv:1103.4130; formulas checked against the paper, Supplementary Theorem 2, Eq. S3-S4):

    l <= floor( n * (q - h(Q + mu(eps))) - 2 log2(1 / (2 eps_bar)) - leak_EC - log2(2 / eps_cor) ),
    mu(eps) = sqrt( (n + k) / (n k) * (k + 1) / k * ln(1 / eps) ),   maximised over 2 eps + eps_bar <= eps_sec,

with q = 1 for BB84 with ideal single-photon sources, h the binary entropy truncated at 1 for x > 1/2, all logs
base 2. n is the number of key bits, k the number of sampled bits. The paper's statement uses the abort
threshold Q_tol; its Lemma 3 gives Q_key < lambda + mu for the OBSERVED lambda (except with probability eps),
which is what is used here. ASSUMPTIONS to state wherever this is quoted: the paper's parameter estimation
samples the complementary basis, while here the sample is a uniformly random subset of the sifted key
(Serfling-type deviation), which is justified only because the simulated channels (intercept-resend with random
basis, bit flips) are basis-symmetric. It is therefore a faithful FORM of the bound for these channels, not a
general-attack security proof.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import secrets
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy.signal import fftconvolve

from evefl.quantum.base import QKDResult

DEFAULT_EPS_SEC = 1e-10
DEFAULT_EPS_COR = 1e-15
DEFAULT_QBER_ABORT = 0.11  # LOCKDOWN threshold; compared with >=
CASCADE_PASSES = 4
CASCADE_MIN_QBER = 0.005  # block sizing floor when the estimate is (near) zero

STATUS_OK = "ok"
STATUS_NO_KEY = "no_key"
STATUS_ABORT_QBER = "abort_qber_high"
STATUS_ABORT_EC = "abort_ec_verification"
STATUS_ABORT_AUTH = "abort_auth"


# ---------------------------------------------------------------------------
# Finite-key bound
# ---------------------------------------------------------------------------

def binary_entropy(x: float) -> float:
    """Binary entropy in bits, truncated: 1 for x > 1/2 (TLGR footnote 1), 0 at x = 0."""
    if x <= 0.0:
        return 0.0
    if x > 0.5:
        return 1.0
    return -x * math.log2(x) - (1.0 - x) * math.log2(1.0 - x)


@dataclass(frozen=True)
class FiniteKeyBound:
    key_length: int            # floor of the bound, never negative
    raw_length: float          # the bound before flooring / clamping (negative means no key)
    mu: float
    qber_upper: float          # min(1, observed + mu): upper confidence bound on the key-bit error rate
    eps: float                 # smoothing parameter chosen by the optimisation
    eps_bar: float
    eps_sec: float
    eps_cor: float


def statistical_deviation(n_key: int, k_sample: int, eps: float) -> float:
    """mu(eps) = sqrt((n + k) / (n k) * (k + 1) / k * ln(1/eps))."""
    return math.sqrt((n_key + k_sample) / (n_key * k_sample) * (k_sample + 1) / k_sample * math.log(1.0 / eps))


def finite_key_length(
    n_key: int,
    k_sample: int,
    observed_qber: float,
    leak_ec: float,
    eps_sec: float = DEFAULT_EPS_SEC,
    eps_cor: float = DEFAULT_EPS_COR,
) -> FiniteKeyBound:
    """Secret key length l (bits) for n_key raw bits, a k_sample-bit estimate and leak_EC leaked bits."""
    if n_key < 1 or k_sample < 1:
        raise ValueError(f"n_key and k_sample must be >= 1, got {n_key}, {k_sample}")
    if not 0.0 <= observed_qber <= 1.0:
        raise ValueError(f"observed_qber must be in [0, 1], got {observed_qber}")
    if not 0.0 < eps_sec < 1.0 or not 0.0 < eps_cor < 1.0:
        raise ValueError("eps_sec and eps_cor must be in (0, 1)")
    if leak_ec < 0:
        raise ValueError("leak_ec must be >= 0")

    best: Optional[FiniteKeyBound] = None
    for fraction in np.linspace(0.02, 0.98, 97):  # 2 eps = fraction * eps_sec, eps_bar = the rest
        eps = float(fraction) * eps_sec / 2.0
        eps_bar = eps_sec - 2.0 * eps
        mu = statistical_deviation(n_key, k_sample, eps)
        raw = (
            n_key * (1.0 - binary_entropy(observed_qber + mu))
            - 2.0 * math.log2(1.0 / (2.0 * eps_bar))
            - leak_ec
            - math.log2(2.0 / eps_cor)
        )
        if best is None or raw > best.raw_length:
            best = FiniteKeyBound(
                key_length=max(0, math.floor(raw)), raw_length=raw, mu=mu,
                qber_upper=min(1.0, observed_qber + mu), eps=eps, eps_bar=eps_bar,
                eps_sec=eps_sec, eps_cor=eps_cor,
            )
    assert best is not None
    return best


# ---------------------------------------------------------------------------
# Error correction: Cascade
# ---------------------------------------------------------------------------

@dataclass
class Reconciliation:
    corrected: np.ndarray       # Bob's key after correction
    leaked_bits: int            # every parity bit disclosed (leak_EC)
    errors_corrected: int
    parity_transcript: bytes    # the disclosed parities, for the authenticated transcript


def cascade_reconcile(
    alice: np.ndarray,
    bob: np.ndarray,
    qber_estimate: float,
    rng: np.random.Generator,
    passes: int = CASCADE_PASSES,
) -> Reconciliation:
    """Cascade (Brassard-Salvail 1993): `passes` passes with block sizes k1, 2 k1, 4 k1, ..., k1 = 0.73 / Q.

    Each block parity Alice discloses counts as one leaked bit, as does each step of a binary search; fixing
    an error re-checks the blocks containing it in all earlier passes. The permutations are public, so a
    plain seeded numpy generator is fine here. Cascade does not guarantee zero residual errors: the
    verification hash after it catches the rest.
    """
    a = np.asarray(alice, dtype=np.uint8)
    b = np.asarray(bob, dtype=np.uint8).copy()
    n = len(a)
    if len(b) != n:
        raise ValueError("alice and bob keys must have the same length")
    if n == 0:
        return Reconciliation(b, 0, 0, b"")

    q = max(float(qber_estimate), CASCADE_MIN_QBER)
    k1 = min(n, max(2, math.ceil(0.73 / q)))

    leaked = 0
    corrected = 0
    transcript = bytearray()

    def mismatch(positions: np.ndarray) -> int:
        nonlocal leaked
        leaked += 1  # Alice discloses the parity of these positions
        parity_a = int(a[positions].sum()) & 1
        transcript.append(parity_a)
        return parity_a ^ (int(b[positions].sum()) & 1)

    perms: List[np.ndarray] = []
    sizes: List[int] = []
    block_of: List[np.ndarray] = []

    def members(pass_index: int, block: int) -> np.ndarray:
        size = sizes[pass_index]
        return perms[pass_index][block * size:(block + 1) * size]

    def fix(start_pass: int, start_block: np.ndarray, current_pass: int) -> None:
        """Correct one error in `start_block` (known to have odd mismatch), then cascade into earlier passes."""
        nonlocal corrected
        stack = [(start_pass, start_block, True)]
        while stack:
            pass_index, block, known_odd = stack.pop()
            # A block pushed by the cascade must be re-checked (this discloses one more parity bit).
            if not known_odd and mismatch(block) == 0:
                continue
            candidates = block
            while len(candidates) > 1:  # binary search for one error in an odd-mismatch block
                half = len(candidates) // 2
                first = candidates[:half]
                candidates = first if mismatch(first) else candidates[half:]
            position = int(candidates[0])
            b[position] ^= 1
            corrected += 1
            for j in range(current_pass + 1):
                if j == pass_index:
                    continue
                stack.append((j, members(j, int(block_of[j][position])), False))

    for i in range(max(1, passes)):
        size = min(n, k1 * (2 ** i))
        perm = np.arange(n) if i == 0 else rng.permutation(n)
        perms.append(perm)
        sizes.append(size)
        owner = np.empty(n, dtype=np.int64)
        owner[perm] = np.arange(n) // size
        block_of.append(owner)
        for block_index in range(math.ceil(n / size)):
            block = members(i, block_index)
            if mismatch(block):
                fix(i, block, i)

    return Reconciliation(b, leaked, corrected, bytes(transcript))


# ---------------------------------------------------------------------------
# Toeplitz hashing (universal2): verification hash and privacy amplification
# ---------------------------------------------------------------------------

def random_bits(count: int) -> np.ndarray:
    """`count` uniformly random bits from the OS CSPRNG (`secrets`), never from `random`."""
    raw = np.frombuffer(secrets.token_bytes(math.ceil(count / 8)), dtype=np.uint8)
    return np.unpackbits(raw)[:count]


def toeplitz_hash(bits: np.ndarray, out_bits: int, seed_bits: np.ndarray) -> np.ndarray:
    """Multiply `bits` by the out_bits x n Toeplitz matrix defined by `seed_bits` (n + out_bits - 1 of them), mod 2."""
    x = np.asarray(bits, dtype=np.float64)
    n = len(x)
    if out_bits < 0 or len(seed_bits) != n + out_bits - 1:
        raise ValueError(f"seed must have n + out_bits - 1 = {n + out_bits - 1} bits, got {len(seed_bits)}")
    if out_bits == 0:
        return np.zeros(0, dtype=np.uint8)
    conv = fftconvolve(np.asarray(seed_bits, dtype=np.float64), x)
    return (np.rint(conv[n - 1:n - 1 + out_bits]).astype(np.int64) & 1).astype(np.uint8)


# ---------------------------------------------------------------------------
# Authentication (HMAC in simulation)
# ---------------------------------------------------------------------------

def authenticate(transcript: bytes, auth_key: bytes) -> bytes:
    """HMAC-SHA256 tag over the classical transcript (computational; see the module docstring)."""
    return hmac.new(auth_key, hashlib.sha256(transcript).digest(), hashlib.sha256).digest()


def verify_authentication(transcript: bytes, tag: bytes, auth_key: bytes) -> bool:
    return hmac.compare_digest(authenticate(transcript, auth_key), tag)


# ---------------------------------------------------------------------------
# Result types and the pipeline
# ---------------------------------------------------------------------------

@dataclass
class RoundKey:
    """Per-client, per-round key material. Single use; call `zeroize()` after use.

    `key` is a bytearray so it can be overwritten (Python cannot guarantee no copies exist; this is best
    effort and is stated as such). In this simulator the key is a deterministic function of the seed:
    `simulated` is always True and it must not protect real data.
    """

    client_id: str
    round_id: int
    key: bytearray
    key_bits: int
    qber: float
    qber_upper: float
    leaked_bits: int
    eps_sec: float
    eps_cor: float
    provenance: str = "bb84-sim"
    simulated: bool = True
    zeroized: bool = False

    def zeroize(self) -> None:
        for i in range(len(self.key)):
            self.key[i] = 0
        self.zeroized = True

    def __repr__(self) -> str:  # never print key material
        return (f"RoundKey(client={self.client_id!r}, round={self.round_id}, bits={self.key_bits}, "
                f"qber={self.qber:.4f}, provenance={self.provenance!r}, zeroized={self.zeroized})")


@dataclass
class PostprocessResult:
    status: str
    reason: str
    round_key: Optional[RoundKey]
    details: Dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK


def postprocess(
    result: QKDResult,
    *,
    client_id: str,
    round_id: int,
    auth_key: bytes,
    eps_sec: float = DEFAULT_EPS_SEC,
    eps_cor: float = DEFAULT_EPS_COR,
    qber_abort: float = DEFAULT_QBER_ABORT,
    ec_seed: Optional[int] = None,
    tamper_transcript: bool = False,
) -> PostprocessResult:
    """Run the full classical pipeline on one QKD exchange. See the module docstring for the steps.

    `tamper_transcript` is a TEST hook: it flips a bit in the transcript Bob authenticates, to exercise the
    authentication failure path.
    """
    if result.bob_sifted_key is None or result.sample_indices is None:
        raise ValueError("QKDResult lacks bob_sifted_key / sample_indices; use a backend that records them.")

    alice_all = np.asarray(result.sifted_key, dtype=np.uint8)
    bob_all = np.asarray(result.bob_sifted_key, dtype=np.uint8)
    sample_idx = np.asarray(result.sample_indices, dtype=np.int64)
    s, k = len(alice_all), len(sample_idx)
    details: Dict[str, float] = {"n_sifted": s, "k_sample": k}

    # --- 1. parameter estimation; the compared bits are discarded from the key -------------------------------
    if k == 0 or s - k == 0:
        return PostprocessResult(STATUS_NO_KEY, "empty_sample_or_key", None, details)
    sample_errors = int(np.count_nonzero(alice_all[sample_idx] != bob_all[sample_idx]))
    observed = sample_errors / k
    keep = np.ones(s, dtype=bool)
    keep[sample_idx] = False
    key_a, key_b = alice_all[keep], bob_all[keep]
    n_key = len(key_a)
    details.update({"n_key": n_key, "qber": observed})

    if not (observed < qber_abort):  # >= the LOCKDOWN threshold
        return PostprocessResult(STATUS_ABORT_QBER, f"qber {observed:.4f} >= {qber_abort}", None, details)

    # --- 2. error correction + verification hash ------------------------------------------------------------
    rng = np.random.default_rng(ec_seed if ec_seed is not None else secrets.randbits(63))
    reconciliation = cascade_reconcile(key_a, key_b, observed, rng)
    leak_ec = reconciliation.leaked_bits
    details.update({"leak_ec": leak_ec, "ec_errors_corrected": reconciliation.errors_corrected})
    if n_key > 0 and observed > 0:
        details["ec_efficiency"] = leak_ec / (n_key * binary_entropy(observed))

    hash_bits = math.ceil(math.log2(1.0 / eps_cor))
    hash_seed = random_bits(n_key + hash_bits - 1)
    tag_a = toeplitz_hash(key_a, hash_bits, hash_seed)
    tag_b = toeplitz_hash(reconciliation.corrected, hash_bits, hash_seed)
    if not np.array_equal(tag_a, tag_b):
        return PostprocessResult(STATUS_ABORT_EC, "verification hash mismatch after error correction", None, details)

    # --- 3. finite-key length -------------------------------------------------------------------------------
    bound = finite_key_length(n_key, k, observed, leak_ec, eps_sec, eps_cor)
    details.update({"mu": bound.mu, "qber_upper": bound.qber_upper, "raw_length": bound.raw_length,
                    "key_length": bound.key_length})
    key_bytes = bound.key_length // 8
    if key_bytes == 0:
        return PostprocessResult(STATUS_NO_KEY, "finite-key bound leaves no key", None, details)

    # --- 4. privacy amplification ---------------------------------------------------------------------------
    out_bits = key_bytes * 8
    pa_seed = random_bits(n_key + out_bits - 1)
    secret_a = toeplitz_hash(key_a, out_bits, pa_seed)
    secret_b = toeplitz_hash(reconciliation.corrected, out_bits, pa_seed)
    if not np.array_equal(secret_a, secret_b):  # cannot happen after a passed verification hash
        return PostprocessResult(STATUS_ABORT_EC, "privacy amplification outputs differ", None, details)

    # --- 5. authentication of the classical transcript -------------------------------------------------------
    transcript = b"".join([
        sample_idx.astype("<i4").tobytes(), alice_all[sample_idx].tobytes(), bob_all[sample_idx].tobytes(),
        reconciliation.parity_transcript, np.packbits(hash_seed).tobytes(), np.packbits(tag_a).tobytes(),
        np.packbits(pa_seed).tobytes(), client_id.encode("utf-8"), int(round_id).to_bytes(8, "big"),
    ])
    tag = authenticate(transcript, auth_key)
    received = bytearray(transcript)
    if tamper_transcript:
        received[0] ^= 1
    if not verify_authentication(bytes(received), tag, auth_key):
        return PostprocessResult(STATUS_ABORT_AUTH, "classical transcript failed authentication", None, details)

    round_key = RoundKey(
        client_id=client_id, round_id=round_id, key=bytearray(np.packbits(secret_a).tobytes()),
        key_bits=out_bits, qber=observed, qber_upper=bound.qber_upper, leaked_bits=leak_ec + hash_bits,
        eps_sec=eps_sec, eps_cor=eps_cor,
    )
    return PostprocessResult(STATUS_OK, "ok", round_key, details)
