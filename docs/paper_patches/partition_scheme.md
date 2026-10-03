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

## Notes for the authors
- Replace "per-sample p ~ Dir(alpha)" wherever it appears (system model / experimental setup).
- Report the audited overlap counts (all zero) and per-hospital patient/image counts from
  `partition_meta.json` after the Kaggle run (`scripts/audit_partition.py`).
- The validation split is by patient too, disjoint from the hospitals and the test set. State that the test
  set was never used to choose anything. Results JSON: `val_eval` (selection) and `server_eval` (test, report).
- Limitation to state: a multi-label patient is represented by one dominant label for the
  purpose of assignment only; all their labels are kept for training and evaluation.
