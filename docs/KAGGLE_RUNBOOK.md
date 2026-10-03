# Kaggle runbook: partition audit and one lite alpha = 0 baseline

Purpose: check the patient-level split on the REAL ChestX-ray14 metadata and produce one small, honest baseline run,
then paste both outputs back. Nothing in the paper may be quoted from these runs without the saved JSON.

Replace `<DATA>` with your dataset folder (it must contain `Data_Entry_2017.csv` and the images, flat or nested)
and `<WEIGHTS>` with the Kaggle dataset holding `resnet18_imagenet1k_v1.pth`
(create it with `python scripts/cache_pretrained_weights.py` on a machine with internet), or leave
pretrained off for a first check (see the note in cell 4).

## Cell 1. Get this branch (internet on) and install
```bash
!git clone -b fix/p0-4-patient-level-split https://github.com/Parineeta-2307/EveFL_Federated_Learning.git /kaggle/working/EveFL
%cd /kaggle/working/EveFL
!pip install -q -e . --no-deps
!pip install -q qiskit==1.2.4 qiskit-aer==0.15.1 flwr==1.13.0 scikit-learn pandas
```
(Kaggle already ships torch/torchvision/numpy; `--no-deps` avoids replacing them.)

## Cell 2. Partition the FULL dataset and audit it (reads only the CSV, takes seconds)
```bash
!python -c "from evefl.fl.dataset import partition_and_save; partition_and_save('<DATA>', '/kaggle/working/partitions_full', seed=42)"
!python scripts/audit_partition.py --data-root <DATA> --partition-root /kaggle/working/partitions_full
```
Paste the whole output. Expected: every overlap count 0, `RESULT: OK`, hospital sizes plausibly skewed (not
equal, none nearly empty), and a validation split present. Also paste `partition_meta.json`:
```bash
!cat /kaggle/working/partitions_full/partition_meta.json | head -40
```

## Cell 3. Partition a small subset for the lite run
```bash
!python -c "from evefl.fl.dataset import partition_and_save; partition_and_save('<DATA>', '/kaggle/working/partitions_lite', seed=42, subset_fraction=0.05)"
!python scripts/audit_partition.py --data-root <DATA> --partition-root /kaggle/working/partitions_lite | tail -25
```

## Cell 4. One lite baseline: no attack, no extra noise (alpha = 0)
```bash
!python -m evefl.fl.server \
    --data-root <DATA> --partition-root /kaggle/working/partitions_lite \
    --num-rounds 5 --local-epochs 1 --batch-size 32 --seed 0 \
    --intercept-probability 0.0 --qubits 1024 --sample-fraction 0.25 \
    --pretrained --pretrained-weights <WEIGHTS>/resnet18_imagenet1k_v1.pth \
    --experiment-name lite_alpha0_seed0 --output /kaggle/working/results/lite_alpha0_seed0.json
```
No weights dataset yet? Replace the two `--pretrained*` lines with `--no-pretrained` (a from-scratch run; the JSON records
this, so it cannot be confused with a pretrained one).

## Cell 5. Summarise (paste this back)
```bash
!python scripts/summarize_run.py /kaggle/working/results/lite_alpha0_seed0.json
```
Expected for alpha = 0 with no noise: every round SECURE, system QBER exactly 0.0000, a ~128-bit QBER sample, `model_updated`
True, and validation and test macro AUC columns filled (the AUC after 5 lite rounds will be low; that is fine, it is only
a pipeline check). Also paste any line mentioning skipped classes or a failure count above 0.

## What I will check when you paste
- all overlap counts are 0 and `val` is disjoint from the hospitals and the test set,
- hospital sizes and the No-Finding share per hospital show real skew without an empty hospital,
- the results JSON has `pretrained`, `config_hash`, `qkd`, `optimizer`, per-round `val_eval` and `server_eval`,
- alpha = 0 stayed SECURE every round (a false CAUTION here would be a bug).
