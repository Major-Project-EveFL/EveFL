# Kaggle runbook: install, partition re-audit and one lite alpha = 0 baseline

Purpose: get EveFL running on the Kaggle image, prove the real-data split is reproducible, and produce one small, honest
baseline run, then paste the outputs back. Nothing in the paper may be quoted from these runs without the saved JSON.

The Kaggle image is **Python 3.13 with torch 2.11 and torchvision 0.26**, not the Python 3.12 / torch 2.5.1 environment
that `requirements.txt` pins and CI tests. `pip install -r requirements.txt` fails there (`torchvision==0.20.1` and
`qiskit-aer==0.15.1` have no Python 3.13 build, and `grpcio<=1.64.3`, which flwr 1.13.0 requires, has none either).
Use `requirements-kaggle.txt` with `--no-deps`, as below. A run on this image is therefore a different software
environment from CI; say so when quoting its numbers.

Dataset folder on Kaggle (must contain `Data_Entry_2017.csv` and the images, flat or nested):
`/kaggle/input/datasets/nih-chest-xrays/data`. Weights for `--pretrained` come from a Kaggle dataset holding
`resnet18_imagenet1k_v1.pth` (create it with `python scripts/cache_pretrained_weights.py` on a machine with internet);
without one, use `--no-pretrained` (see cell 5).

## Cell 1. Get the code (internet on) and install
```python
!git clone https://github.com/Parineeta-2307/EveFL_Federated_Learning.git /kaggle/working/EveFL
%cd /kaggle/working/EveFL
!pip install -q --no-deps -r requirements-kaggle.txt
!pip install -q --no-deps -e .
```
If the folder already exists from an earlier attempt, run `%cd /kaggle/working/EveFL` and `!git pull` instead of cloning.
No kernel restart is needed: nothing that the image already ships is replaced. If you ever do restart
(`os._exit(0)`), the working directory resets, so run `%cd /kaggle/working/EveFL` before any `pip` or `python` command.

## Cell 2. Check the install (paste the output, also if it fails)
```python
%cd /kaggle/working/EveFL
import sys
sys.path.insert(0, "/kaggle/working/EveFL")  # a running kernel does not see the new editable install
import torch, torchvision, numpy, flwr
print("python", sys.version.split()[0], "| torch", torch.__version__, "| torchvision", torchvision.__version__,
      "| numpy", numpy.__version__, "| flwr", flwr.__version__)
import evefl.fl.server
print("import evefl.fl.server: OK")
```
Expected: `flwr 1.13.0` and `import evefl.fl.server: OK`. If an import fails with `ModuleNotFoundError: No module named 'X'`,
paste the whole traceback; do not guess a fix (the usual cause is a package the image lacks, which is then added to
`requirements-kaggle.txt` with a pinned version).

## Cell 3. Re-partition the FULL dataset and audit it (reads only the CSV, takes seconds)
```python
DATA = "/kaggle/input/datasets/nih-chest-xrays/data"
!python -c "from evefl.fl.dataset import partition_and_save; partition_and_save('{DATA}', '/kaggle/working/partitions_full', seed=42)"
!python scripts/audit_partition.py --data-root {DATA} --partition-root /kaggle/working/partitions_full
```
Paste the whole output. Expected: every overlap count 0, `Index-file hashes: OK`, `RESULT: OK`. These are the same parameters as
the audited run (seed 42, alpha 0.5, test and validation fractions 0.1, all patients), so the counts must reproduce the
committed audit exactly. Check that, and print the new index-file hashes:
```python
import json
audit = json.load(open("docs/validation/partition_audit_real.json"))
meta = json.load(open("/kaggle/working/partitions_full/partition_meta.json"))
same = all(meta["audit"]["stats"][k] == audit["stats"][k] for k in audit["stats"])
print("counts identical to docs/validation/partition_audit_real.json:", same)
print(json.dumps(meta["index_sha256"], indent=2))
```
Paste both lines of output. The hashes are the proof of reproducibility from now on; a rerun that prints other hashes for the
same seed means the split is NOT deterministic and must be reported.

## Cell 4. Partition a small subset for the lite run
```python
!python -c "from evefl.fl.dataset import partition_and_save; partition_and_save('{DATA}', '/kaggle/working/partitions_lite', seed=42, subset_fraction=0.05)"
!python scripts/audit_partition.py --data-root {DATA} --partition-root /kaggle/working/partitions_lite | tail -25
```

## Cell 5. One lite baseline: no attack, no extra noise (alpha = 0)
```python
WEIGHTS = "/kaggle/input/<your-weights-dataset>"  # only for the pretrained variant
!python -m evefl.fl.server \
    --data-root {DATA} --partition-root /kaggle/working/partitions_lite \
    --num-rounds 5 --local-epochs 1 --batch-size 32 --seed 0 \
    --intercept-probability 0.0 --qubits 1024 --sample-fraction 0.25 \
    --pretrained --pretrained-weights {WEIGHTS}/resnet18_imagenet1k_v1.pth \
    --experiment-name lite_alpha0_seed0 --output /kaggle/working/results/lite_alpha0_seed0.json
```
No weights dataset yet? Replace the two `--pretrained*` lines with `--no-pretrained` (a from-scratch run; the JSON records this,
so it cannot be confused with a pretrained one). The default policy mode (`global`) is used; the controller block is 1024 qubits with a
0.25 sample fraction (about 128 bits), which is the documented headline setting. That block is too small to yield a secret
key (ADR 0001), and this run does not use keys: it only exercises the training and evaluation pipeline.

## Cell 6. Summarise (paste this back)
```python
!python scripts/summarize_run.py /kaggle/working/results/lite_alpha0_seed0.json
```
Expected for alpha = 0 with no noise: every round SECURE, system QBER exactly 0.0000, a QBER sample of about 128 bits,
`model_updated` True, an empty `excluded` list each round, `rounds discarded: 0`, and the `val_AUC`, `val_AUC*` and
`test_AUC` columns filled (the AUC after 5 lite rounds will be low; that is fine, it is only a pipeline check).
`val_AUC*` is the model-selection metric (macro over classes with at least 20 positives, rule in
`docs/06_EXPERIMENT_PROTOCOL.md`). With the 5% subset several classes may be thin or skipped; that is expected here.
Also paste any line mentioning skipped classes or a failure count above 0.

## What I will check when you paste
- the install cell printed `flwr 1.13.0` and the import line, with no traceback,
- all overlap counts are 0, the hashes are present and `counts identical ... : True`,
- the results JSON has `pretrained`, `config_hash`, `qkd`, `optimizer`, `policy`, per-round `val_eval` and `server_eval`,
- alpha = 0 stayed SECURE every round with nothing excluded (a false CAUTION here would be a bug).
