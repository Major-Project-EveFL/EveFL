# Paper patches checklist

The paper is out of step with the code in several places. Tick each item only when the paper
text (LaTeX) has actually been changed, not when the code was. Every number in the paper must
come from a saved results file with a config hash (see docs/06).

Status: `[ ]` open, `[~]` code done / paper text pending, `[x]` paper updated.

## Code changes the paper must describe
- [~] **Partition scheme (P0-4).** Patient-level, disjoint split; dominant label = rarest positive
  label, "No Finding" only if none; Dirichlet(0.5) over groups; test = random 10% of patients.
  Replaces the per-sample p ~ Dir(alpha) description. Patch text: `partition_scheme.md`
  (on branch `fix/p0-4-patient-level-split` until merged). Report audited overlap counts (all 0)
  and hospital sizes from the Kaggle audit.
- [~] **Anomaly rule (P0-2).** Old: `mean + 2*std` on full-parameter norms. New: update-delta
  norms, flag if norm > median + k * max(1.4826*MAD, rel_floor*median), k=3, rel_floor=0.25, needs
  >= 3 clients. Say why (mean+2std cannot fire with 3 clients: max z = 1.155; MAD floor for
  near-identical honest updates). Source: `evefl/fl/screening.py`.
- [ ] **CAUTION response (clipping).** Flagged updates are clipped to the acceptance bound, not
  down-weighted by 0.5. Branch `fix/caution-clip-flagged-updates` is HELD until the guide has been
  told and agrees; then merge it and add its paragraph here.
- [~] **Initialisation (P0-5).** State whether runs are ImageNet-pretrained (default) and where the
  weights came from (results JSON records `pretrained`, source and SHA-256). Never mix pretrained
  and from-scratch results in one table.
- [~] **Evaluation protocol (P0-3).** Macro AUC-ROC on the held-out test patients after every round
  (LOCKDOWN rounds included, so the curve has flat stretches); classes with no positives in the
  test set are skipped and listed. Per-class AUC in the results JSON.
- [~] **Optimiser and schedule (P1-8).** Code now matches the paper: AdamW (weight decay 0.01, not
  specified in the paper: state it), base lr 1e-3, cosine annealing over the global round
  (`lr_t = 0.5*base*(1+cos(pi*(t-1)/T))`), lr x0.5 in CAUTION (5e-4). Fresh optimiser each round
  (Adam moments are not carried across rounds): say so. The old code actually ran Adam at 1e-4, so
  any earlier Kaggle numbers used a different setting.

- [~] **Channel and noise model (Phase 1).** Intercept-resend Eve with probability alpha plus a
  bit-flip channel with probability e on Bob's result (applied after Eve), so the expected QBER is
  `e + (1 - 2e) * alpha / 4` (not alpha/4). Say it is a bit-flip model, not depolarizing. The
  simulator is exact (no approximation) and was cross-validated against gate-level Qiskit
  (`docs/validation/backend_crosscheck.json`).
- [~] **Sample size (Phase 1).** Sweep done (`docs/validation/qber_sweep.json`, results in docs/06); headline choice pending the team's decision between the rule's output (512, 0.5) and (1024, 0.25). n_qubits and sample_fraction are explicit experiment parameters,
  recorded per run. The headline setting comes from the sample-size sweep with a pre-registered
  selection rule (docs/06); state the rule, the sweep, and the limitation that a 3% noise baseline
  sits near the 5% boundary (persistent false CAUTION there is expected; dynamic thresholds are the
  proper fix, verify the Zhang et al. citation). Notebooks that used 16 or 64 qubits (QBER sample
  of 2 to 8 bits) said nothing about the controller.

## Claims that are not measured yet
- [ ] **Measured vs projected labels.** Every Section VIII number (AUC, MIA, overhead, Table IV)
  is projected. Label each "projected" or replace it with a measured value from a results file.
  "17.3x" CKKS inflation and the QKDFL comparison numbers must go or be re-measured.
- [~] **Table III / Theorem 1.** Done in code and data: `theorem1_table3.md` (restated theorem with exact numbers,
  regenerated simulated Table III as `docs/validation/table3.tex`). The paper text still has to be replaced.
- [ ] **"Graduated beats binary".** Needs per-client-link or intermittent-Eve experiments (P1-5),
  otherwise remove the claim.
- [ ] **Encryption claims.** AES-GCM, CKKS and Groth16 are not in the FL path yet; do not describe
  them as active (P1-1).

## Hygiene
- [ ] Remove COVID-19 claims (ChestX-ray14 has no such label).
- [ ] Verify novelty citations (QKDFL, FedSec) by DOI; fix citation inconsistencies (docs/04 P1-10).
- [ ] Optional, discuss with guide: use the official ChestX-ray14 `test_list.txt` patient-level test
  set instead of a random 10% of patients, so AUC is comparable with published numbers.
