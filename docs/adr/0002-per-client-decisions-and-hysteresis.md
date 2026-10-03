# ADR 0002: Per-client decisions, hysteresis and the three policy modes

Status: accepted (2026-10). Results in `docs/validation/hysteresis_sweep.json` (pre-registered in `docs/06`).
Everything quantitative below is a SIMULATED control-plane statistic (rounds and client-rounds); none of it measures model
accuracy, and excluding a hospital under non-IID data has a model-quality cost that only the Phase 5 training runs can measure.

## Context
With one global state, a static Eve at alpha >= 0.44 on a single link stops training in nearly every round, which is what a
binary pause (QKDFL-style) does. The paper's claim that a graduated response is better can only be tested if Eve can sit on one
link while the other hospitals keep training, and if the controller does not flap on sampling noise.

## Decisions
1. **Hysteresis slows only the way out.** Entering CAUTION at 0.05 and LOCKDOWN at 0.11 is immediate (compared with `>=`, the
   0.11 constant is never softened); a state is left only after `dwell` consecutive readings below the boundary by `margin`.
   Off by default (`margin = 0, dwell = 1` is the memoryless classifier). Property-tested.
2. **Three policy modes** behind an ABC and a registry, selected from YAML: `global_binary` (B2, pause at the worst link's
   QBER >= 0.11), `global` (B3), `per_client` (B4: a LOCKDOWN link is excluded, the rest aggregate with renormalised weights,
   fewer than `min_clients` (default 2) discards the round).
3. **Rules for an excluded client**: it is not sent the model (training or evaluation), its update is never aggregated even if
   it arrives, its link is still measured every round (otherwise it could never rejoin), and there is no fallback to unprotected
   transmission. A client without a key is excluded the same way in `per_client` mode (reason `no_key` / `key_abort`, logged apart
   from `qber_exclusion`); in the global modes a key failure still discards the round.
4. **Screening scope**: median/MAD statistics over ALL included updates, clipping/down-weighting only for clients whose own state
   is CAUTION; SECURE clients stay plain FedAvg. Screening is inactive below 3 updates, a direct cost of excluding a hospital.
5. **Declarative `ChannelPlan`** (static / step / window / intermittent attacks with their own seeds, per-link noise); the full
   plan is written into the results JSON.

## Results (simulated; 3 links, 1024-qubit controller block, 50 rounds, 1000 trajectories per cell)
Policy modes under Eve on ONE link, 1% noise, hysteresis off:

| Eve | global_binary / global: rounds discarded | per_client: rounds discarded | client-rounds trained (global) | client-rounds trained (per_client) | attacked client's rounds |
|---|---|---|---|---|---|
| static alpha 0.3 | 14.2% | 0% | 85.8% | 95.3% | 42.9 of 50 |
| static alpha 0.44 | 59.4% | 0% | 40.6% | 80.2% | 20.3 of 50 |
| static alpha 0.6 | 93.1% | 0% | 6.9% | 69.0% | 3.5 of 50 |
| static alpha 1.0 | 100% | 0% | 0% | 66.7% | 0 of 50 |
| window alpha 0.6, rounds 21-30 | 18.6% | 0% | 81.4% | 93.8% | 40.7 of 50 |
| intermittent alpha 0.6, p = 0.3 | 27.9% | 0% | 72.1% | 90.7% | 36.1 of 50 |

`global_binary` and `global` discard exactly the same rounds (CAUTION never discards); they differ only in how they train, which
this simulation does not measure. The per-client advantage in client-rounds is real, but it is a count of rounds, not of accuracy:
the attacked hospital's data is missing from training exactly when Eve chooses.

Hysteresis, anchored at 2% noise (pre-registered rule): 11 of 20 (dwell, margin) cells qualify; the rule recommends
**dwell 3, margin 0.01**. Honest reading of the numbers:
- Detection latency is identical in every cell (escalation is immediate); the 0.11 entry is never delayed (asserted on every trajectory).
- Recovery takes `dwell` rounds (3.6 rounds to SECURE at 2% noise), and 138.1 vs 140.5 client-rounds are trained under the
  10-round attack (1.7% fewer).
- At 2% noise the benefit for quiet-channel flapping is SMALL: 0.0304 -> 0.0286 state changes per link-round (6% fewer), while
  the share of link-rounds spent outside SECURE rises from 1.5% to 4.9%, because each isolated false CAUTION is held for longer.
  At 3% noise flapping falls 27% (0.167 -> 0.123) at the price of 31% of link-rounds outside SECURE.
- Where it clearly helps is the borderline case (Eve at alpha = 0.44, mean QBER 11%): flaps per link-round fall from 0.163 to 0.025
  (85% fewer) at 2% noise.
Hysteresis buys stability, not accuracy, and costs rounds in the higher state. It stays OFF by default; enable dwell 3, margin 0.01
for experiments where borderline attackers matter, and report the extra time in CAUTION.

## Consequences and open items
- The earlier "4.6% / 25.6% false CAUTION at 2% / 3% noise" figures are SYSTEM-level (max over 3 links); per link they are 1.6% / 9.4%.
  Per-link numbers are what flapping depends on in `per_client` mode.
- **Selective exclusion is an attack lever** (docs/03): an adversary who disturbs one link chooses which hospital drops out of
  training. Participation counts are logged every round; Phase 5 must report them with per-client performance.
- **Non-IID bias** from excluding a hospital is not measured here and must not be assumed to be zero.
- **Screening is inactive** when exclusion leaves fewer than 3 updates.
- The clipping change for CAUTION (instead of the 0.5 down-weight on `main`) is still held for the guide; this ADR's screening
  rule applies to whichever response is current.
