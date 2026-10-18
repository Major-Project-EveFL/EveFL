"""
Attack lab: Eve on a chosen link, and what the three round policies do about it.

Everything here is a SIMULATED control-plane statistic. It samples per-link QBER trajectories with the exact count-level model
validated in Phase 1 (`evefl.fl.control_sim.sample_qber_trajectories`) and feeds ONE trajectory set to the REAL policy classes
(`global_binary` = B2, `global` = B3, `per_client` = B4 in `evefl.orchestration.policy`), so the three policies are compared on
identical channels and what is measured is the code that ships. It counts rounds and client-rounds; it trains no model, so it
says nothing about accuracy: excluding a hospital under non-IID data has a model-quality cost that only training runs can measure.

The LOCKDOWN boundary (QBER >= 0.11) and the 5% CAUTION boundary are the controller defaults and are deliberately not settings
here. Units: QBER is a fraction in [0, 1]; "percent" fields are 0 to 100 and say so in their names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from evefl.fl.channels import ATTACK_KINDS, ChannelPlan, LinkAttack
from evefl.fl.control_sim import (
    PolicyRun,
    client_rounds_trained,
    link_state_changes_per_round,
    rounds_discarded,
    sample_qber_trajectories,
    share_of_link_rounds,
    simulate_policy,
)
from evefl.orchestration.policy import PolicyConfig, policy_registry
from evefl.orchestration.state_machine import HysteresisConfig, SecurityState

LABEL = ("SIMULATED control-plane statistics: rounds and client-rounds counted by running the real policy code on sampled QBER "
         "trajectories. They are not accuracy: no model is trained, and the model-quality cost of excluding a hospital is not "
         "measured here.")
DEFAULT_MODES = ("global_binary", "global", "per_client")


@dataclass(frozen=True)
class LabSettings:
    attacked_link: Optional[str] = "0"      # None: no attack
    alpha: float = 0.6                      # Eve's intercept probability, a fraction in [0, 1]
    attack_kind: str = "static"             # static | step | window | intermittent
    attack_start: int = 21                  # first attacked round (step, window)
    attack_end: int = 30                    # last attacked round (window)
    attack_probability: float = 0.3         # per-round activation probability (intermittent)
    noise: float = 0.01                     # baseline bit-flip probability on every link
    rounds: int = 50
    reps: int = 300                         # simulated runs behind the summary table
    seed: int = 0
    n_qubits: int = 1024                    # the controller's QBER block (ADR 0001)
    sample_fraction: float = 0.25
    hysteresis_margin: float = 0.0          # slows only the way OUT of a state (ADR 0002)
    hysteresis_dwell: int = 1
    min_clients: int = 2
    links: Tuple[str, ...] = ("0", "1", "2")
    modes: Tuple[str, ...] = DEFAULT_MODES

    def validate(self) -> None:
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")
        if not 0.0 <= self.noise <= 0.5:
            raise ValueError(f"noise must be in [0, 0.5], got {self.noise}")
        if self.rounds < 1 or self.reps < 1:
            raise ValueError("rounds and reps must be at least 1")
        if self.attack_kind not in ATTACK_KINDS:
            raise ValueError(f"attack_kind must be one of {ATTACK_KINDS}, got {self.attack_kind!r}")
        if self.attacked_link is not None and self.attacked_link not in self.links:
            raise ValueError(f"attacked_link {self.attacked_link!r} is not one of the links {self.links}")
        if self.min_clients < 1:
            raise ValueError("min_clients must be at least 1")
        known = policy_registry.list_keys()
        unknown = [m for m in self.modes if m not in known]
        if unknown:
            raise ValueError(f"unknown policy mode(s) {unknown}; available: {known}")


@dataclass
class LabResult:
    label: str
    settings: LabSettings
    summary: pd.DataFrame                   # one row per policy mode
    trace: Dict[str, pd.DataFrame]          # mode -> rows (round, link, qber, state, included, discarded) for the first run
    reps: int = 0
    notes: List[str] = field(default_factory=list)


def plan_for(settings: LabSettings, rep: int = 0) -> ChannelPlan:
    attacks = {}
    if settings.attacked_link is not None:
        attacks[settings.attacked_link] = LinkAttack(
            alpha=settings.alpha, kind=settings.attack_kind, start_round=settings.attack_start,
            end_round=settings.attack_end if settings.attack_kind == "window" else None,
            probability=settings.attack_probability, seed=settings.seed + rep)
    return ChannelPlan(attacks=attacks, default_noise=settings.noise)


def _policy_config(settings: LabSettings, mode: str) -> PolicyConfig:
    return PolicyConfig(mode=mode, min_clients=settings.min_clients,
                        hysteresis=HysteresisConfig(margin=settings.hysteresis_margin, dwell=settings.hysteresis_dwell))


def _summary_row(runs: List[PolicyRun], settings: LabSettings) -> Dict[str, float]:
    links = list(settings.links)
    attacked = settings.attacked_link
    attacked_rounds = float("nan")
    if attacked is not None:
        attacked_rounds = float(np.mean([sum(attacked in inc for inc in run.included) for run in runs]))
    return {
        "rounds_discarded_pct": 100.0 * rounds_discarded(runs) / settings.rounds,
        "client_rounds_trained_pct": 100.0 * client_rounds_trained(runs) / (settings.rounds * len(links)),
        "attacked_client_rounds_trained": attacked_rounds,
        "caution_share_pct": 100.0 * share_of_link_rounds(runs, links, SecurityState.CAUTION),
        "lockdown_share_pct": 100.0 * share_of_link_rounds(runs, links, SecurityState.LOCKDOWN),
        "state_changes_per_link_round": link_state_changes_per_round(runs, links),
    }


def _trace_frame(run: PolicyRun, qbers: np.ndarray, links: Tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for t in range(qbers.shape[0]):
        for j, cid in enumerate(links):
            rows.append({"round": t + 1, "link": cid, "qber": float(qbers[t, j]), "state": run.link_states[t][cid].value,
                         "included": cid in run.included[t], "discarded": bool(run.discarded[t])})
    return pd.DataFrame(rows)


def run_lab(settings: LabSettings) -> LabResult:
    """Sample `reps` QBER trajectories once, run every selected policy on the same trajectories, and summarise."""
    settings.validate()
    links = list(settings.links)
    qbers = sample_qber_trajectories(
        lambda rep: plan_for(settings, rep), links, settings.rounds, settings.reps,
        n_qubits=settings.n_qubits, sample_fraction=settings.sample_fraction, seed=settings.seed)
    summary_rows: Dict[str, Dict[str, float]] = {}
    traces: Dict[str, pd.DataFrame] = {}
    for mode in settings.modes:
        runs = simulate_policy(qbers, links, _policy_config(settings, mode))
        summary_rows[mode] = _summary_row(runs, settings)
        traces[mode] = _trace_frame(runs[0], qbers[0], settings.links)
    summary = pd.DataFrame.from_dict(summary_rows, orient="index")
    summary.index.name = "policy mode"
    notes = ["global_binary and global discard exactly the same rounds (CAUTION never discards); they differ only in how they "
             "train, which this simulation does not measure."]
    if settings.attacked_link is not None and settings.alpha > 0:
        notes.append("Excluding the attacked hospital keeps the others training, but its data is missing from training exactly "
                     "when Eve chooses (selective exclusion is an attack lever, docs/03).")
    return LabResult(label=LABEL, settings=settings, summary=summary, trace=traces, reps=settings.reps, notes=notes)
