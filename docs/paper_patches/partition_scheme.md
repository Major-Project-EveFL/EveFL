# Paper patch: data partitioning (P0-4)

Replace the current description of the non-IID split (the paper describes per-sample
p ~ Dir(alpha); the old code did a class-wise split of images) with the text below. It
matches `evefl/fl/partition.py` exactly. Also fix any statement that the test split is
"IID over images": it is IID over *patients*.

## Suggested text (LaTeX-ready)

```latex
\subsection{Data Partitioning}
ChestX-ray14 contains 112{,}120 frontal radiographs from 30{,}805 patients. Because a
patient contributes several images, all splits are made at the \emph{patient} level so
that no patient appears in more than one hospital, the validation set or the test set. We first hold out a
random 10\% of patients as a common test set and a further random 10\% as a validation set used for all
model selection (choice of round, learning rate, thresholds); the test set is used for the final report only. Each remaining patient is assigned a
\emph{dominant label}: their rarest positive pathology (rarity being the number of
patients positive for that label), or ``No Finding'' only if they have no positive label.
For each dominant-label group $g$ we draw proportions $p_g \sim \mathrm{Dir}(\alpha\mathbf{1}_K)$
over the $K{=}3$ simulated hospitals and assign that group's patients accordingly, with
$\alpha{=}0.5$; all images of a patient follow the patient. Redrawing is repeated until every
hospital holds at least one patient. Using the rarest positive label prevents the large
``No Finding'' class and the common pathologies from washing out the label skew.
Hospital and test partitions are pairwise disjoint at both the patient and image level,
which we verify programmatically (overlap counts are reported in the released partition
metadata).
```

## Audited counts to cite (measured on the real data)
Source: `docs/validation/partition_audit_real.json` (counts only, no patient identifiers). Run on Kaggle with seed 42,
alpha = 0.5, test and validation fractions 0.1, at commit 9873d10 of branch `fix/p0-4-patient-level-split`.

- All ten pair overlaps (three hospitals, test, validation) are 0 patients and 0 images.
- Totals match the dataset: 12,869 + 6,470 + 5,306 + 3,080 + 3,080 = 30,805 patients; 46,975 + 18,018 + 25,136 + 10,889 +
  11,102 = 112,120 images. The hospitals hold 42%, 16% and 22% of the images; test 9.7%, validation 9.9%.
- The skew is non-IID by design (Dirichlet alpha = 0.5), e.g. Cardiomegaly positives: 1,868 in hospital 0 against 94 in
  hospital 1. Hospital 0 sees only 5 Hernia positives (hospital 1: 86, hospital 2: 73, test: 53, validation: 10). Say so in
  the paper so the low count is not mistaken for a bug.
- Validation Hernia (10 positives) is thin: model selection uses the macro AUC over non-thin classes, rule pre-registered in
  `docs/06_EXPERIMENT_PROTOCOL.md` ("Model-selection rule and thin classes").
- Limitation: the index files of that Kaggle run were not hashed (the hashes were added afterwards). Reproducibility is
  therefore evidenced by identical counts on a rerun, and from then on by `index_sha256` in `partition_meta.json`
  (`scripts/audit_partition.py` verifies it). Do not state "byte-identical" for the audited run until a rerun has been compared.
- Reproducibility holds for a fixed NumPy version only: the split uses numpy's `Generator`, whose streams are not guaranteed
  identical across versions. `partition_meta.json` records `numpy_version` (the audited run did not), and every results JSON
  records the library versions. Quote the NumPy version next to the seed in the paper.

## Notes for the authors
- Replace "per-sample p ~ Dir(alpha)" wherever it appears (system model / experimental setup).
- Report the audited overlap counts (all zero) and per-hospital patient/image counts (see the section above; the Kaggle
  audit is done).
- The validation split is by patient too, disjoint from the hospitals and the test set. State that the test
  set was never used to choose anything. Results JSON: `val_eval` (selection) and `server_eval` (test, report).
- Limitation to state: a multi-label patient is represented by one dominant label for the
  purpose of assignment only; all their labels are kept for training and evaluation.
