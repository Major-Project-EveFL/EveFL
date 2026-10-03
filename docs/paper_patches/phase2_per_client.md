# Paper patch: per-client decisions, hysteresis and the policy-mode comparison (Phase 2)

Labels: **simulated** = control-plane simulation (`evefl/fl/control_sim.py`, `docs/validation/hysteresis_sweep.json`, rule
pre-registered in docs/06); these count rounds and client-rounds, NOT accuracy. Details and caveats: docs/adr/0002.

## What to say (and not say) about the central claim
1. **Say:** with Eve on one link of three, a global pause (QKDFL-style `global_binary`, and also the global three-state
   controller, which discards the same rounds) stops training in 59% of rounds at alpha = 0.44 and 93% at alpha = 0.6 (simulated,
   1% noise, 50 rounds); per-client exclusion discards none and keeps the two clean hospitals training in every round
   (80% and 69% of client-rounds). For alpha = 0.3 the difference is small (14% vs 0% of rounds discarded).
2. **Do not say** this improves accuracy or is free. The attacked hospital's data is missing from training exactly when the
   adversary chooses (3.5 of 50 rounds at alpha = 0.6); under non-IID data this biases the model. Measure it in Phase 5 with
   per-client performance and participation counts, and report both.
3. **Selective exclusion is an attack lever** (add to the threat model, docs/03): an adversary who disturbs one link chooses which
   hospital drops out. Participation counts are logged every round.
4. Per-client exclusion also reduces the number of updates available for screening: with two updates left the median/MAD
   screening is inactive (needs 3). State it as a direct cost.

## Hysteresis (state plainly: it buys stability, not accuracy, and costs rounds in the higher state)
- Entering CAUTION at 0.05 and LOCKDOWN at 0.11 is immediate; only leaving a state is delayed (dwell, margin). The 0.11 threshold is
  never softened. Detection latency is identical with and without hysteresis (asserted on every simulated trajectory).
- Pre-registered rule at 2% noise recommends dwell 3, margin 0.01. Reality check to report: on a quiet 2%-noise channel it reduces
  state changes only 6% (0.0304 to 0.0286 per link-round) while tripling the time outside SECURE (1.5% to 4.9% of link-rounds).
  It helps clearly for a borderline attacker (alpha = 0.44): 85% fewer state changes (0.163 to 0.025).
- The earlier false-CAUTION figures for 2% and 3% noise (4.6%, 25.6%) are system-level (max over 3 links); per link they are
  1.6% and 9.4%.
- Do not claim hysteresis solves the 3% noise limitation (at 3%, 31% of link-rounds are outside SECURE with the recommended
  setting); baseline-relative thresholds remain future work.

## Mechanics to describe
Per-link channel plans (static, step, window, intermittent), per-client state and action (TRAIN, TRAIN_PROX, EXCLUDE), min_clients
= 2 of 3, renormalised aggregation weights, excluded clients not sent the model and never aggregated, links measured every round so
excluded clients can rejoin, key failures excluding only that client (`no_key` / `key_abort`) with logs separate from
`qber_exclusion`, FedProx and the reduced learning rate only for clients in CAUTION.
