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

## Claims that are not measured yet
- [ ] **Measured vs projected labels.** Every Section VIII number (AUC, MIA, overhead, Table IV)
  is projected. Label each "projected" or replace it with a measured value from a results file.
  "17.3x" CKKS inflation and the QKDFL comparison numbers must go or be re-measured.
- [ ] **Table III / Theorem 1.** Regenerate from real runs; restate Theorem 1 for the actual QBER
  sample size (see docs/04 P1-4).
- [ ] **"Graduated beats binary".** Needs per-client-link or intermittent-Eve experiments (P1-5),
  otherwise remove the claim.
- [ ] **Encryption claims.** AES-GCM, CKKS and Groth16 are not in the FL path yet; do not describe
  them as active (P1-1).

## Hygiene
- [ ] Remove COVID-19 claims (ChestX-ray14 has no such label).
- [ ] Verify novelty citations (QKDFL, FedSec) by DOI; fix citation inconsistencies (docs/04 P1-10).
- [ ] Optional, discuss with guide: use the official ChestX-ray14 `test_list.txt` patient-level test
  set instead of a random 10% of patients, so AUC is comparable with published numbers.
