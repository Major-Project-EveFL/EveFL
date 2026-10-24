"""
A demonstration of pairwise-masked secure aggregation, built ONLY from the step-2a primitives (ADR 0003): the 64-bit fixed-point codec
(`evefl.crypto.secagg.fixedpoint`) and the mask PRG (`evefl.crypto.secagg.prg`).

What this is: a simulated library run in one process on synthetic updates, to show what the masking does and does not hide.
What it is not: it is not wired into the federated-learning path; the pairwise seeds are fresh bytes from the operating system that
STAND IN for the key agreement of step 2b (ephemeral X25519, signed advertisements, set checks), which is not implemented; there is
no aggregator class yet (step 2c); and the keys of the rest of the system are simulated and not secret. Nothing here is a statement
about the security of a deployment.

The weights are the registry weights of ADR 0003: the first three are the audited hospital sizes (46,975, 18,018 and 25,136 images),
the fourth and fifth are illustrative. Updates are synthetic Gaussian numbers; their randomness is not cryptographic and is unrelated
to the masks, whose seeds come from `os.urandom`.
"""

from __future__ import annotations

import itertools
import os
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Tuple

import numpy as np

from evefl.crypto.secagg import fixedpoint as fp
from evefl.crypto.secagg import prg

HOSPITAL_SIZES = (46975, 18018, 25136, 21000, 15000)
LABEL = ("SIMULATED: this runs the secure-aggregation primitives of ADR 0003 step 2a (the fixed-point codec and the mask generator) "
         "in one process on synthetic updates. It is a library demonstration that is not wired into the FL path. The pairwise seeds "
         "are fresh random bytes standing in for the key agreement of step 2b, which is not implemented; there is no aggregator "
         "class yet (step 2c); the keys elsewhere in this project are simulated and not secret.")


@dataclass(frozen=True)
class DemoSettings:
    n_clients: int = 3
    n_params: int = 16
    update_scale: float = 0.01               # standard deviation of the synthetic updates
    seed: int = 0                            # for the synthetic updates only, never for masks
    round_id: int = 1
    drop_client: Optional[str] = None        # a hospital whose masked update never arrives
    colluding: Tuple[str, ...] = ()          # hospitals that share their pairwise seeds with the server
    pair_seeds: Optional[Mapping[FrozenSet[str], bytes]] = None   # tests only; None = fresh os.urandom seeds


@dataclass
class DemoResult:
    label: str
    client_ids: List[str]
    registry_weights: np.ndarray
    weights_normalised: np.ndarray
    updates: np.ndarray                      # x_i: what each hospital trained (synthetic)
    weighted: np.ndarray                     # y_i = (w_i / sum w) * x_i: what is masked
    masked: np.ndarray                       # uint64: what the server receives
    aggregate_plain: np.ndarray              # sum of y_i computed in the clear
    aggregate_secure: np.ndarray             # decode(sum of masked): what the server learns
    max_abs_error: float
    tolerance: float                         # n_clients * 2^-33
    dropout: Optional[Dict[str, Any]]
    collusion: Optional[Dict[str, Any]]


def _fresh_pair_seeds(ids: List[str]) -> Dict[FrozenSet[str], bytes]:
    """One fresh 32-byte seed per unordered pair, straight from the operating system's random source."""
    return {frozenset(pair): os.urandom(prg.SEED_BYTES) for pair in itertools.combinations(ids, 2)}


def new_pair_seeds(n_clients: int) -> Dict[FrozenSet[str], bytes]:
    """Fresh operating-system random seeds for the pairs of `n_clients` hospitals (a stand-in for the step-2b key agreement)."""
    return _fresh_pair_seeds([str(i) for i in range(n_clients)])


def _validate(s: DemoSettings, ids: List[str]) -> None:
    if not 3 <= s.n_clients <= 5:
        raise ValueError("the demo supports 3 to 5 hospitals (the privacy floor of ADR 0003 is 3)")
    if not 1 <= s.n_params <= 100_000:
        raise ValueError("n_params must be between 1 and 100000")
    if not s.update_scale > 0.0:
        raise ValueError("update_scale must be positive")
    if s.drop_client is not None and s.drop_client not in ids:
        raise ValueError(f"drop_client {s.drop_client!r} is not one of {ids}")
    if s.colluding:
        if any(c not in ids for c in s.colluding):
            raise ValueError(f"colluding hospitals must be among {ids}")
        if len(set(s.colluding)) >= len(ids):
            raise ValueError("if every hospital colludes there is nothing left to protect or to learn")
    if s.pair_seeds is not None:
        for pair in itertools.combinations(ids, 2):
            seed = s.pair_seeds.get(frozenset(pair))
            if seed is None or len(seed) != prg.SEED_BYTES:
                raise ValueError(f"pair_seeds must hold a {prg.SEED_BYTES}-byte seed for every pair; missing {pair}")


def _mask_toward(seeds: Mapping[FrozenSet[str], bytes], round_id: int, me: str, other: str, n_params: int) -> np.ndarray:
    """The signed mask hospital `me` adds because of its pair with `other` (the same stream the partner expands)."""
    return prg.MaskStreams(seeds[frozenset((me, other))], round_id=round_id, me=me, other=other).signed_mask(0, n_params)


def _collusion(s: DemoSettings, ids: List[str], seeds: Mapping[FrozenSet[str], bytes], masked: np.ndarray,
               weighted: np.ndarray) -> Dict[str, Any]:
    colluders = [c for c in ids if c in set(s.colluding)]
    others = [i for i in ids if i not in set(colluders)]
    index = {cid: k for k, cid in enumerate(ids)}
    # The server sees the masked updates of the non-colluders; each colluder hands over its own pairwise seeds.
    total = fp.sum_encoded([masked[index[i]] for i in others])
    for i in others:
        for c in colluders:
            total = fp.sub(total, _mask_toward(seeds, s.round_id, i, c, s.n_params))
    recovered = fp.decode(total)
    truth = weighted[[index[i] for i in others]].sum(axis=0)
    out: Dict[str, Any] = {
        "colluding": colluders, "learns_sum_of": others,
        "individual_revealed": others[0] if len(others) == 1 else None,
        "sum_error": float(np.max(np.abs(recovered - truth))),
    }
    if len(others) > 1:
        first = others[0]
        guess = masked[index[first]]
        for c in colluders:
            guess = fp.sub(guess, _mask_toward(seeds, s.round_id, first, c, s.n_params))
        out[f"single_guess_error_hospital_{first}"] = float(np.mean(np.abs(fp.decode(guess) - weighted[index[first]])))
    return out


def run_demo(settings: DemoSettings) -> DemoResult:
    """One masked-aggregation round on synthetic updates; also the dropout and collusion what-ifs if the settings ask for them."""
    ids = [str(i) for i in range(settings.n_clients)]
    _validate(settings, ids)
    n, p = settings.n_clients, settings.n_params
    registry = np.array(HOSPITAL_SIZES[:n], dtype=float)
    w_hat = registry / registry.sum()

    rng = np.random.default_rng(np.random.SeedSequence([int(settings.seed), n, p]))   # synthetic data only
    updates = rng.normal(0.0, settings.update_scale, size=(n, p))
    weighted = w_hat[:, None] * updates
    for row in weighted:
        fp.assert_encodable(row, n)               # refused before any masking if the sum could wrap

    seeds = dict(settings.pair_seeds) if settings.pair_seeds is not None else _fresh_pair_seeds(ids)
    encoded = fp.encode(weighted)
    masked = np.empty_like(encoded)
    for i, cid in enumerate(ids):
        row = encoded[i].copy()
        for other in ids:
            if other != cid:
                row = fp.add(row, _mask_toward(seeds, settings.round_id, cid, other, p))
        masked[i] = row

    total = fp.sum_encoded(list(masked))
    fp.check_aggregate(total)
    secure = fp.decode(total)
    plain = weighted.sum(axis=0)

    dropout = None
    if settings.drop_client is not None:
        remaining = [masked[k] for k, cid in enumerate(ids) if cid != settings.drop_client]
        garbage = fp.decode(fp.sum_encoded(remaining))
        dropout = {
            "dropped": settings.drop_client, "round_aborts": True, "reason": "secagg_dropout",
            "garbage_mean_abs": float(np.mean(np.abs(garbage))),
            "note": ("The masks that pair the missing hospital with the others never cancel, so what is left decodes to numbers of "
                     "the size of the fixed-point range. The protocol does not try to repair this: the server knows who is "
                     "missing, the whole round is aborted and the model is not updated (ADR 0003, Dropout)."),
        }

    return DemoResult(
        label=LABEL, client_ids=ids, registry_weights=registry, weights_normalised=w_hat, updates=updates, weighted=weighted,
        masked=masked, aggregate_plain=plain, aggregate_secure=secure, max_abs_error=float(np.max(np.abs(secure - plain))),
        tolerance=n * 2.0 ** -33, dropout=dropout,
        collusion=_collusion(settings, ids, seeds, masked, weighted) if settings.colluding else None)
