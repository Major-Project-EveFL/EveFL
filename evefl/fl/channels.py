"""
Declarative per-link attack plans (Phase 2).

A `ChannelPlan` says, for every client link and every round, how much Eve intercepts and how noisy the link is.
It is built only from static / step / window / intermittent specs with their own seeds, never from arbitrary
functions, so the FULL plan can be written into the results JSON (`to_dict`) and reproduced (`from_dict`).

    static        Eve intercepts with probability alpha in every round
    step          from `start_round` on
    window        from `start_round` to `end_round` (inclusive), then she stops
    intermittent  in each round she is active with probability `probability` (decided from `seed`, round and
                  client, so it is reproducible and independent of every other random stream)

CLI forms (see `parse_link_attack` / `parse_link_noise`):
    --link-attack 0:0.6                  Eve on link 0, alpha 0.6, every round
    --link-attack 1:0.6:step@20          from round 20
    --link-attack 2:0.6:window@21-30     rounds 21 to 30
    --link-attack 0:0.6:intermittent@0.3 active in 30% of rounds
    --link-noise 1:0.02                  2% bit-flip noise on link 1 only
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from evefl.quantum.base import ChannelModel
from evefl.quantum.seeding import client_id_to_int

ATTACK_KINDS = ("static", "step", "window", "intermittent")


@dataclass(frozen=True)
class LinkAttack:
    alpha: float
    kind: str = "static"
    start_round: int = 1
    end_round: Optional[int] = None
    probability: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")
        if self.kind not in ATTACK_KINDS:
            raise ValueError(f"kind must be one of {ATTACK_KINDS}, got {self.kind!r}")
        if self.start_round < 1:
            raise ValueError(f"start_round must be >= 1, got {self.start_round}")
        if self.kind == "window":
            if self.end_round is None or self.end_round < self.start_round:
                raise ValueError("a window attack needs end_round >= start_round")
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(f"probability must be in [0, 1], got {self.probability}")

    def active(self, server_round: int, cid: str) -> bool:
        if self.kind == "static":
            return True
        if self.kind == "step":
            return server_round >= self.start_round
        if self.kind == "window":
            assert self.end_round is not None
            return self.start_round <= server_round <= self.end_round
        draw = np.random.default_rng(
            np.random.SeedSequence([int(self.seed), int(server_round), client_id_to_int(cid), 2])
        ).random()
        return bool(draw < self.probability)


@dataclass(frozen=True)
class ChannelPlan:
    attacks: Mapping[str, LinkAttack] = field(default_factory=dict)   # client id -> attack
    link_noise: Mapping[str, float] = field(default_factory=dict)     # client id -> bit-flip probability
    default_noise: float = 0.0                                         # every other link

    def __post_init__(self) -> None:
        for value in [self.default_noise, *self.link_noise.values()]:
            if not 0.0 <= value <= 0.5:
                raise ValueError(f"bit-flip noise must be in [0, 0.5], got {value}")

    def channel_for(self, cid: str, server_round: int) -> ChannelModel:
        attack = self.attacks.get(cid)
        alpha = attack.alpha if attack is not None and attack.active(server_round, cid) else 0.0
        return ChannelModel(intercept_probability=alpha, bit_flip_probability=self.link_noise.get(cid, self.default_noise))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "default_noise": self.default_noise,
            "link_noise": dict(self.link_noise),
            "attacks": {cid: asdict(a) for cid, a in self.attacks.items()},
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ChannelPlan":
        return cls(
            attacks={cid: LinkAttack(**a) for cid, a in dict(raw.get("attacks") or {}).items()},
            link_noise={cid: float(v) for cid, v in dict(raw.get("link_noise") or {}).items()},
            default_noise=float(raw.get("default_noise", 0.0)),
        )


def parse_link_attack(spec: str, *, seed: int = 0) -> Tuple[str, LinkAttack]:
    """Parse '<cid>:<alpha>[:<kind>@<arg>]' (see the module docstring)."""
    parts = spec.split(":")
    if len(parts) not in (2, 3) or not parts[0]:
        raise ValueError(f"link attack must look like 'cid:alpha[:kind@arg]', got {spec!r}")
    cid, alpha = parts[0], float(parts[1])
    if len(parts) == 2:
        return cid, LinkAttack(alpha=alpha, seed=seed)
    kind, _, arg = parts[2].partition("@")
    if kind == "step":
        return cid, LinkAttack(alpha=alpha, kind="step", start_round=int(arg), seed=seed)
    if kind == "window":
        start, _, end = arg.partition("-")
        return cid, LinkAttack(alpha=alpha, kind="window", start_round=int(start), end_round=int(end), seed=seed)
    if kind == "intermittent":
        return cid, LinkAttack(alpha=alpha, kind="intermittent", probability=float(arg), seed=seed)
    raise ValueError(f"unknown attack kind {kind!r} in {spec!r}; expected step@N, window@A-B or intermittent@P")


def parse_link_noise(spec: str) -> Tuple[str, float]:
    parts = spec.split(":")
    if len(parts) != 2 or not parts[0]:
        raise ValueError(f"link noise must look like 'cid:probability', got {spec!r}")
    return parts[0], float(parts[1])


def build_plan(
    attack_specs: Sequence[str] = (),
    noise_specs: Sequence[str] = (),
    *,
    default_noise: float = 0.0,
    seed: int = 0,
) -> ChannelPlan:
    attacks: Dict[str, LinkAttack] = {}
    for spec in attack_specs:
        cid, attack = parse_link_attack(spec, seed=seed)
        if cid in attacks:
            raise ValueError(f"link {cid} has more than one attack spec")
        attacks[cid] = attack
    noise: Dict[str, float] = {}
    for spec in noise_specs:
        cid, value = parse_link_noise(spec)
        if cid in noise:
            raise ValueError(f"link {cid} has more than one noise spec")
        noise[cid] = value
    return ChannelPlan(attacks=attacks, link_noise=noise, default_noise=default_noise)


def attacked_links(plan: ChannelPlan) -> List[str]:
    return sorted(plan.attacks)
