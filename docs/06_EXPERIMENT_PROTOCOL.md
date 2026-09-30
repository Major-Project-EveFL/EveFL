# 06 Experiment Protocol

Rule. Every number in the paper comes from a saved result file with a config hash and seeds.
Anything else is marked projected or removed.

## Setup
ResNet-18 (14 outputs, BCEWithLogits), NIH ChestX-ray14, 3 clients, Dirichlet(0.5), patient-level
disjoint splits, 10% IID test split, E local epochs (default 5, reduce for budget), B=32,
AdamW with cosine annealing, 50 rounds. Seeds 0 to 4 (minimum 3). Report mean and 95% CI.

## Compute budget (rough estimate, measure it)
Full data at 224px, 5 epochs, 50 rounds, 4+ scenarios is far beyond Kaggle session and weekly GPU
quotas. Provide a lite profile (data subset, 128 to 160px, 1 to 2 epochs) for sweeps and run the
full profile only for the headline scenarios. LOCKDOWN rounds are nearly free because clients
skip training.

## Baselines
B0 unsecured FedAvg. B1 FedAvg with AEAD only (same accuracy as B0, measure overhead).
B2 QKDFL-style global binary pause at 11%, reimplemented. B3 EveFL global states.
B4 EveFL per-client exclusion. B5 secure aggregation baseline (pairwise-mask) against CKKS.

## Attack settings
- Static Eve, alpha in {0, 0.1, ..., 1.0}, on all links.
- Single compromised link (Eve on 1 of 3), the setting where graduated response can win.
- Intermittent Eve (per-round activation probability) and step attack at round 20.
- Baseline noise sweep, depolarizing in {0, 1, 2, 3}%, to measure false CAUTION and false LOCKDOWN.
- Byzantine client (scaled or sign-flipped update) to test screening and the norm proof.

## Metrics
Macro AUC-ROC (and per-class), rounds to AUC 0.80 (if reached), detection latency in rounds,
false alarm rate, effective training rounds, MIA success measured with a shadow-model or
loss-threshold attack on our own trained models, bytes and seconds per round per layer, memory.

## Outputs
JSON per run (config, seeds, per-round log, metrics), aggregated tables, and three core figures.
FSM diagram, QBER validation curve (simulated vs alpha/4 with noise), accuracy-security Pareto
curve. `evefl report` regenerates all figures and LaTeX tables from results. Placeholders in the
paper stay commented until a result file replaces them.

## Pre-registered: QBER sample-size sweep (Phase 1)

Committed BEFORE the sweep is run, so the headline `n_qubits` / `sample_fraction` cannot be tuned
to the results. Nothing in the paper is measured yet, so no number needs protecting; the point is
that the choice follows a stated rule.

### What is swept
- `n_qubits` in {256, 512, 1024, 2048, 4096, 8192, 16384}
- `sample_fraction` in {0.10, 0.25, 0.50}
- baseline bit-flip noise e in {0, 0.01, 0.02, 0.03}
- Eve intercept probability alpha in {0.0, 0.1, ..., 1.0}
- 3 independent links with the same alpha; system QBER = max over the 3 links (as in the strategy)
- thresholds fixed at secure_max = 0.05, caution_max = 0.11 (never tuned); the state is whatever
  `StateController.classify` returns; no hysteresis (this measures the per-round classifier)
- backend `bb84_numpy`, 20,000 simulated rounds per cell, experiment seed 0 (streams from
  SeedSequence(0, round, link)), so all alpha rows are paired. Rates are reported with Wilson 95%
  intervals and next to the exact analytic value (binomial mixture over the random sifted length),
  which must agree with the simulation.

### Quantities reported for every cell
- false CAUTION rate: P(system state != SECURE) at alpha = 0
- false LOCKDOWN rate: P(system state = LOCKDOWN) at alpha = 0
- detection rate: P(system state != SECURE) at each alpha > 0 (also P(LOCKDOWN) as a secondary column)
- expected sample size m and expected leftover sifted key per qubit sent, n_sifted * (1 - sample_fraction) / n
  (a bigger sample leaves less key; this is a cost, reported not optimised)

### Selection rule for the headline setting
A setting (n_qubits, sample_fraction) is a CANDIDATE iff ALL hold:
1. false CAUTION rate <= 1% at e = 0.01 (system level, max over 3 links). Anchored at 1% noise only.
2. detection rate >= 95% at alpha = 0.30 for e in {0, 0.01} (system level, max over 3 links).
   alpha = 0.30 is the CAUTION demo value; it is the reliability claim the paper may make.
3. expected sample size m >= 100 bits (the config guard).

Headline = the candidate with the SMALLEST n_qubits; ties broken by the larger leftover key (smaller
sample_fraction). Reason: the criteria only get easier with larger blocks, so this is the smallest
block for which the claims hold, and any larger block does at least as well. If no candidate exists,
that itself is the finding and the claims must be weakened; the grid is not extended after seeing results.

Not part of the rule, reported as limitations:
- e = 0.02 and e = 0.03 are reported but not required. A 3% baseline sits close to the 5% boundary, so
  persistent false CAUTION there is expected. Dynamic thresholds (Zhang et al., cited in the paper;
  VERIFY the reference) are the proper fix and are out of scope.
- Larger blocks are more realistic (real QKD systems process far more than 1024 qubits per block);
  the headline is the minimum sufficient block, not a claim about any real system.
- QBER is a link-disturbance signal, not an attack classifier (docs/03).

Outputs: `docs/validation/qber_sweep.json` (all cells, config, seeds) and the table/figure generated from it.

### Clarifications (added before the sweep was run; no criterion changed)
- The selection rule is evaluated on the EXACT analytic probabilities (binomial mixture over the random
  sifted length, independent links), not on Monte Carlo estimates, so the choice has no simulation noise.
  Comparisons with the boundaries use the same float comparison as `StateController.classify`
  (QBER = k/m, SECURE iff k/m < 0.05). The simulation is reported next to it as a check.
- The sweep engine samples (sifted length, sample errors) directly from their exact distributions
  (sifted ~ Binomial(n, 1/2); given the sample size m, errors ~ Binomial(m, p) with
  p = e + (1 - 2e) * alpha / 4), using common uniform draws across alpha and e so the rows are paired.
  This is exact, not an approximation, and is verified against the full `bb84_numpy` protocol on a
  subset of cells (results in the same JSON). It replaces per-qubit simulation only for speed.

### Result of the sweep (2026-09-30, `docs/validation/qber_sweep.json`)
Produced by `scripts/qber_sweep.py` exactly as pre-registered (grid, thresholds, seed, rule unchanged).
The exact analytic rates agree with the paired Monte Carlo (analytic value inside the simulated 95% interval
in 907 of 924 cells) and with the full `bb84_numpy` protocol on four cells (p-values 0.49 to 1.0).
- 15 of the 21 (n_qubits, sample_fraction) settings qualify. By the rule, the headline is
  **n_qubits = 512, sample_fraction = 0.5** (expected sample 128 bits; false CAUTION 0.13% at e = 1%;
  detection at alpha = 0.3 of 99.7% at e = 0 and 99.9% at e = 1%; leftover sifted key 0.25 per qubit).
- Tension worth deciding explicitly: that setting spends half of the sifted key on the sample.
  n_qubits = 1024, sample_fraction = 0.25 also qualifies (false CAUTION 0.11%, detection 99.7% / 99.9%,
  leftover 0.375 per qubit) and is closer to realistic block sizes. The rule was fixed before the run and
  is not changed here; any other choice must be argued explicitly in the paper as a deviation.
- False CAUTION at alpha = 0 (system level, f = 0.25) grows quickly with baseline noise: at n = 1024 it is
  0.11% (e = 1%), 4.6% (e = 2%), 25.6% (e = 3%); n = 4096 brings e = 3% down to 2.1% and n = 8192 to 0.08%.
  With 3% noise the 5% boundary is simply close to the noise floor. Reported as a limitation (dynamic
  thresholds are the proper fix; verify the Zhang et al. citation).
- Detection is essentially complete from alpha = 0.3 (n = 1024, e = 1%: 99.9%); at alpha = 0.2 it is 95.2% and at
  alpha = 0.1 only 40%, so weak intermittent interception is not reliably caught at these block sizes.
