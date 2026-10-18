# Kaggle runbook: install, partition re-audit and one lite alpha = 0 baseline

Purpose: get EveFL running on the Kaggle image, prove the real-data split is reproducible, and produce one small, honest
baseline run, then paste the outputs back. Nothing in the paper may be quoted from these runs without the saved JSON.
Run the six cells ONE BY ONE, in order (not "Run all"): cell 2 must be able to stop the sequence.

The Kaggle image is **Python 3.13 with torch 2.11 and torchvision 0.26**, not the Python 3.12 / torch 2.5.1 environment
that `requirements.txt` pins and CI tests. `pip install -r requirements.txt` fails there (`torchvision==0.20.1` and
`qiskit-aer==0.15.1` have no Python 3.13 build, and `grpcio<=1.64.3`, which flwr 1.13.0 requires, has none either).
Use `requirements-kaggle.txt` with `--no-deps`, as below. A run on this image is therefore a different software
environment from CI; say so when quoting its numbers.

## One-time setup in Kaggle
- **Dataset folder:** `/kaggle/input/datasets/nih-chest-xrays/data` (contains `Data_Entry_2017.csv` and the images).
- **Pretrained weights:** the private Kaggle dataset `parineetarana1/evefl-resnet18-weights`, attached to the notebook; the
  file is `/kaggle/input/datasets/parineetarana1/evefl-resnet18-weights/resnet18_imagenet.pth`, SHA-256
  `9803814b9183a8bdd46cd0c9cd178ec98b51de241084613e975f4fa54d9a57f2` (cell 2 checks it).
- **The repository is private**, so cell 1 needs a GitHub token in Kaggle Secrets:
  1. On GitHub create a *fine-grained personal access token*: only this repository, permission **Contents: read-only**, and a
     short expiry. Nothing else.
  2. In the notebook open *Add-ons > Secrets*, add a secret named exactly `GITHUB_TOKEN`, and switch it on for this notebook.
  3. The token is read inside cell 1, passed to git through an environment variable (it is not in the clone URL, so it is not
     stored in `.git/config`), masked in any error text, and never printed or written to a file. Keep the notebook private and
     revoke the token when the run is done.

## Cell 1. Get the code, install, check the imports (internet on)
Output to paste: the printed commit line, the versions line and `import evefl.fl.server: OK` (or the whole traceback).
```python
import base64, os, subprocess
from kaggle_secrets import UserSecretsClient

REPO = "https://github.com/Parineeta-2307/EveFL_Federated_Learning.git"
DEST = "/kaggle/working/EveFL"

token = UserSecretsClient().get_secret("GITHUB_TOKEN")  # never printed
basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_CONFIG_COUNT": "1",
       "GIT_CONFIG_KEY_0": "http.extraheader", "GIT_CONFIG_VALUE_0": f"AUTHORIZATION: basic {basic}"}

def git(*args):
    r = subprocess.run(["git", *args], env=env, capture_output=True, text=True)
    text = (r.stdout + r.stderr).replace(token, "***").replace(basic, "***")
    if r.returncode:
        raise RuntimeError(f"git {args[0]} failed:\n{text}")
    return text.strip()

if os.path.isdir(f"{DEST}/.git"):
    git("-C", DEST, "pull", "--ff-only")
else:
    git("clone", REPO, DEST)
print(git("-C", DEST, "log", "--oneline", "-1"))
del token, basic, env  # the credential does not outlive this cell

%cd /kaggle/working/EveFL
!pip install -q --no-deps -r requirements-kaggle.txt
!pip install -q --no-deps -e .

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
`requirements-kaggle.txt` with a pinned version). If you ever restart the kernel (`os._exit(0)`), the working directory
resets: run cell 1 again, or at least `%cd /kaggle/working/EveFL`, before any other command.

## Cell 2. Check the pretrained weights (stops the run if the hash differs)
Output to paste: the two printed lines (`sha256:` and `OK`), or the error.
```python
import hashlib, os

WEIGHTS = "/kaggle/input/datasets/parineetarana1/evefl-resnet18-weights/resnet18_imagenet.pth"
EXPECTED_SHA256 = "9803814b9183a8bdd46cd0c9cd178ec98b51de241084613e975f4fa54d9a57f2"

if not os.path.exists(WEIGHTS):
    raise RuntimeError(f"Weights file not found: {WEIGHTS}. Attach the private dataset parineetarana1/evefl-resnet18-weights.")
digest = hashlib.sha256()
with open(WEIGHTS, "rb") as f:
    for block in iter(lambda: f.read(1 << 20), b""):
        digest.update(block)
print("sha256:", digest.hexdigest())
if digest.hexdigest() != EXPECTED_SHA256:
    raise RuntimeError(f"STOP: weights SHA-256 differs from the expected value {EXPECTED_SHA256}. Do not run the baseline.")
print("OK: weights match the expected SHA-256")
```

## Cell 3. Re-partition the FULL dataset, audit it, compare with the committed audit (seconds; reads only the CSV)
Output to paste: everything this cell prints (overlap table, `Index-file hashes`, `RESULT`, the `counts identical` line, the
NumPy version and the hashes).
```python
%cd /kaggle/working/EveFL
DATA = "/kaggle/input/datasets/nih-chest-xrays/data"
!python -c "from evefl.fl.dataset import partition_and_save; partition_and_save('{DATA}', '/kaggle/working/partitions_full', seed=42)"
!python scripts/audit_partition.py --data-root {DATA} --partition-root /kaggle/working/partitions_full

import json
audit = json.load(open("docs/validation/partition_audit_real.json"))
meta = json.load(open("/kaggle/working/partitions_full/partition_meta.json"))
same = all(meta["audit"]["stats"][k] == audit["stats"][k] for k in audit["stats"])
print("counts identical to docs/validation/partition_audit_real.json:", same)
print("numpy", meta["numpy_version"])
print(json.dumps(meta["index_sha256"], indent=2))
```
Expected: every overlap count 0, `Index-file hashes: OK`, `RESULT: OK`. These are the same parameters as the audited run
(seed 42, alpha 0.5, test and validation fractions 0.1, all patients).

The hashes prove reproducibility, but only between runs with the SAME NumPy version: the split draws from numpy's
`Generator` (`permutation`, `dirichlet`), whose streams NumPy does not guarantee to be identical across versions. So a hash
(or even count) mismatch between Kaggle and the laptop (NumPy 1.26.4) is possible without any bug in our code. How to read it:
- same seed, same NumPy version, different hashes: the split is NOT deterministic, report it;
- different NumPy version, different hashes: an expected possibility, not a defect; the partition must be cited together with
  `numpy_version`, and the audited split cannot be recreated on a machine with another NumPy;
- counts identical to the committed audit: the strongest evidence that this image reproduces the audited split (the audited
  run did not record its NumPy version; from now on `partition_meta.json` does).

Every results JSON also records the interpreter and library versions (`experiment.environment`, shown by
`summarize_run.py`), kept out of the config hash.

## Cell 4. Partition a small subset for the lite run
Output to paste: the last lines of the audit (overlaps 0, `RESULT: OK`).
```python
%cd /kaggle/working/EveFL
DATA = "/kaggle/input/datasets/nih-chest-xrays/data"
!python -c "from evefl.fl.dataset import partition_and_save; partition_and_save('{DATA}', '/kaggle/working/partitions_lite', seed=42, subset_fraction=0.05)"
!python scripts/audit_partition.py --data-root {DATA} --partition-root /kaggle/working/partitions_lite | tail -25
```

## Cell 5. One lite baseline: no attack, no extra noise (alpha = 0), pretrained
Output to paste: the last 15 or so log lines, or the error. The run takes a while.
```python
%cd /kaggle/working/EveFL
DATA = "/kaggle/input/datasets/nih-chest-xrays/data"
WEIGHTS = "/kaggle/input/datasets/parineetarana1/evefl-resnet18-weights/resnet18_imagenet.pth"
!python -m evefl.fl.server \
    --data-root {DATA} --partition-root /kaggle/working/partitions_lite \
    --num-rounds 5 --local-epochs 1 --batch-size 32 --seed 0 \
    --intercept-probability 0.0 --qubits 1024 --sample-fraction 0.25 \
    --pretrained --pretrained-weights {WEIGHTS} \
    --experiment-name lite_alpha0_seed0 --output /kaggle/working/results/lite_alpha0_seed0.json
```
The default policy mode (`global`) is used; the controller block is 1024 qubits with a 0.25 sample fraction (about 128 bits),
the documented headline setting. That block is too small to yield a secret key (ADR 0001), and this run does not use keys: it
only exercises the training and evaluation pipeline.

## Cell 6. Summarise and confirm the weights recorded in the results (paste this back)
Output to paste: everything this cell prints.
```python
%cd /kaggle/working/EveFL
!python scripts/summarize_run.py /kaggle/working/results/lite_alpha0_seed0.json

import json
exp = json.load(open("/kaggle/working/results/lite_alpha0_seed0.json"))["experiment"]
print("weights SHA-256 recorded in the results JSON:", exp["init_weights_sha256"])
print("matches the expected value:", exp["init_weights_sha256"] == "9803814b9183a8bdd46cd0c9cd178ec98b51de241084613e975f4fa54d9a57f2")
```
Expected for alpha = 0 with no noise: every round SECURE, system QBER exactly 0.0000, a QBER sample of about 128 bits,
`model_updated` True, an empty `excluded` list each round, `rounds discarded: 0`, and the `val_AUC`, `val_AUC*` and `test_AUC`
columns filled (the AUC after 5 lite rounds will be low; that is fine, it is only a pipeline check). `val_AUC*` is the
model-selection metric (macro over classes with at least 20 positives, rule in `docs/06_EXPERIMENT_PROTOCOL.md`); with the 5%
subset several classes may be thin or skipped, which is expected here. Also paste any line mentioning skipped classes or a
failure count above 0.

## Cell 7. Micro-benchmark: is image loading the bottleneck? (no FL, a few minutes)
Run this BEFORE cell 5 if you want the numbers to be read together, and paste the whole printed output. It samples 300 images
and times each stage of the current pipeline in one process, DataLoader throughput at 0, 2 and 4 workers, and the model's
compute ceiling on synthetic tensors. It also writes `/kaggle/working/profile_loading.json`.
```python
%cd /kaggle/working/EveFL
DATA = "/kaggle/input/datasets/nih-chest-xrays/data"
!nvidia-smi
!python scripts/profile_loading.py --data-root {DATA} --n-images 300 --workers 0,2,4 \
    --output /kaggle/working/profile_loading.json
```
The DataLoader measurements run before anything touches CUDA, on purpose (the fork-after-CUDA deadlock in CLAUDE.md).
Do not change `--workers` to values above the CPU count that `nvidia-smi` and the output report.

## Cell 8. Compute-only benchmark: what do channels_last and AMP buy? (GPU notebook, about 2 minutes, no data needed)
Synthetic tensors, ResNet-18, fp32 against channels_last, AMP (float16 with a GradScaler) and both, at 224 and 128 px. It
measures throughput and, for inference, the largest logit difference from fp32; it does not train anything and says nothing
about accuracy. Run after the code cell (G1) of the GPU notebook.
```python
%cd /kaggle/temp/EveFL
!python scripts/profile_compute.py --batch-size 32 --image-sizes 224,128 --warmup 5 --iters 30 \
    --output /kaggle/working/profile_compute.json
```

## What changed in cells 5 and 6 (profiling)
- Federated evaluate is now OFF by default (it was a second full pass over each client's training data and fed nothing).
  Add `--federated-evaluate` to cell 5 only if you want that pass measured; the value is recorded in the results JSON.
- `--test-eval-every N` runs the TEST split every N rounds plus the final round; validation runs every round and alone drives
  selection. Default 1 (every round), as before.
- The results JSON now has `experiment.profile` (device, nvidia-smi samples, time per phase, loader wait) and
  `experiment.evaluation`. `summarize_run.py` prints a `device`, `evaluation` and `profile` line; paste them.

## Image cache: a minimal CPU notebook that builds and verifies it
Why: image loading was the bottleneck (docs/validation/kaggle_profile_2026-10-11.json). The cache is built once, saved as a
private dataset, and attached to every later run. Acceptance is pre-registered in docs/06.

Make a NEW notebook for this. Settings: **Accelerator: None** (the build is CPU-only; this saves GPU quota), Internet **on**,
attach the `nih-chest-xrays` dataset, enable the `GITHUB_TOKEN` secret for this notebook (secrets are per notebook).
Run it with **Save & Run All**; it does not need your attention while it runs. Session limit: Kaggle documents 12 hours for
a CPU or GPU notebook session; the build is projected at about 25 to 40 minutes in total (projected, not measured: see the
trial cell). If the session dies the work in `/kaggle/working` is lost unless the version was saved, and a rebuild costs the
full build time again.

**Cell 1. Clone and install** (the clone goes to `/kaggle/temp`, so it is not saved into the dataset):
```python
import base64, os, subprocess
from kaggle_secrets import UserSecretsClient

REPO = "https://github.com/Parineeta-2307/EveFL_Federated_Learning.git"
os.makedirs("/kaggle/temp", exist_ok=True)
DEST = "/kaggle/temp/EveFL"

token = UserSecretsClient().get_secret("GITHUB_TOKEN")  # never printed
basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_CONFIG_COUNT": "1",
       "GIT_CONFIG_KEY_0": "http.extraheader", "GIT_CONFIG_VALUE_0": f"AUTHORIZATION: basic {basic}"}

def git(*args):
    r = subprocess.run(["git", *args], env=env, capture_output=True, text=True)
    text = (r.stdout + r.stderr).replace(token, "***").replace(basic, "***")
    if r.returncode:
        raise RuntimeError(f"git {args[0]} failed:\n{text}")
    return text.strip()

if os.path.isdir(f"{DEST}/.git"):
    git("-C", DEST, "pull", "--ff-only")
else:
    git("clone", REPO, DEST)
print(git("-C", DEST, "log", "--oneline", "-1"))
del token, basic, env

%cd /kaggle/temp/EveFL
!pip install -q --no-deps -r requirements-kaggle.txt
!pip install -q --no-deps -e .

import sys
sys.path.insert(0, "/kaggle/temp/EveFL")
import torch, torchvision, numpy, PIL
print("python", sys.version.split()[0], "| torch", torch.__version__, "| torchvision", torchvision.__version__,
      "| numpy", numpy.__version__, "| Pillow", PIL.__version__, "| cpu_count", os.cpu_count())
import evefl.fl.imagecache
print("import evefl.fl.imagecache: OK")
```
The printed commit must include the merge of `feat/image-cache`. `cpu_count` should be 4; if it is not, the 4-process
estimate does not hold.

**Optional trial cell** (about 6 minutes; run it in your normal notebook, before the full build): it builds 2000 images and
checks 200 against the PNG pipeline, and prints a labelled projection of the full build from the measured rate.
```python
%cd /kaggle/temp/EveFL
DATA = "/kaggle/input/datasets/nih-chest-xrays/data"
!python scripts/build_image_cache.py --data-root {DATA} --output /kaggle/temp/cache_trial --workers 4 \
    --limit 2000 --verify-sample 200
```

**Cell B. Build** (resumable: running it again only does missing shards; it finishes every shard even if an image fails,
then exits non-zero and lists the failures in `failures.json`; it never skips or replaces an image silently):
```python
%cd /kaggle/temp/EveFL
DATA = "/kaggle/input/datasets/nih-chest-xrays/data"
OUT = "/kaggle/working/evefl_cache_u8_256"
!python scripts/build_image_cache.py --data-root {DATA} --output {OUT} --workers 4
```

**Cell C. Verify** (re-hashes every shard and compares 1000 random images plus every non-grey-mode image with the PNG
pipeline; this is acceptance items 2 and 3):
```python
%cd /kaggle/temp/EveFL
DATA = "/kaggle/input/datasets/nih-chest-xrays/data"
OUT = "/kaggle/working/evefl_cache_u8_256"
!python scripts/build_image_cache.py --data-root {DATA} --verify {OUT} --verify-sample 1000
```

**Cell D. Report** (paste this back, with the output of B and C):
```python
import json, pathlib
p = pathlib.Path("/kaggle/working/evefl_cache_u8_256")
meta = json.load(open(p / "meta.json"))
print(json.dumps({k: v for k, v in meta.items() if k != "non_l_rows_listed"}, indent=2))
print("failures:", json.load(open(p / "failures.json"))["n_failures"])
!du -sh {p}
```

**Save it as a private dataset** (in the Kaggle UI, no token needed): when the run has finished, open the saved version, go to
the **Output** tab, choose **New Dataset**, name it `evefl-cache-u8-256`, set visibility **Private**. It is about 7.35 GB.

### Using the cache in a GPU notebook (a fresh session has no clone, no partition and no paths)
Settings: **Accelerator GPU**, Internet on, the `GITHUB_TOKEN` secret enabled, and three datasets attached: `nih-chest-xrays`,
the weights dataset `parineetarana1/evefl-resnet18-weights` and the cache dataset `evefl-cache-u8-256`. Run the six cells below
in order; each needs the one before it. Do NOT run the cache build cell in this notebook. They all use the clone in
`/kaggle/temp/EveFL`, and the input paths are found by code (dataset paths differ between notebooks), never typed.

**G1. Code (clone or update `main`, install, import check).**
```python
import base64, os, subprocess
from kaggle_secrets import UserSecretsClient

REPO = "https://github.com/Parineeta-2307/EveFL_Federated_Learning.git"
os.makedirs("/kaggle/temp", exist_ok=True)
DEST = "/kaggle/temp/EveFL"

token = UserSecretsClient().get_secret("GITHUB_TOKEN")  # never printed
basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_CONFIG_COUNT": "1",
       "GIT_CONFIG_KEY_0": "http.extraheader", "GIT_CONFIG_VALUE_0": f"AUTHORIZATION: basic {basic}"}

def git(*args):
    r = subprocess.run(["git", *args], env=env, capture_output=True, text=True)
    text = (r.stdout + r.stderr).replace(token, "***").replace(basic, "***")
    if r.returncode:
        raise RuntimeError(f"git {args[0]} failed:\n{text}")
    return text.strip()

if os.path.isdir(f"{DEST}/.git"):
    git("-C", DEST, "fetch", "origin")
    git("-C", DEST, "checkout", "main")
    git("-C", DEST, "pull", "--ff-only", "origin", "main")
else:
    git("clone", REPO, DEST)
print(git("-C", DEST, "log", "--oneline", "-1"))
del token, basic, env

%cd /kaggle/temp/EveFL
!pip install -q --no-deps -r requirements-kaggle.txt
!pip install -q --no-deps -e .

import sys
sys.path.insert(0, "/kaggle/temp/EveFL")
import torch, torchvision, numpy, PIL
print("python", sys.version.split()[0], "| torch", torch.__version__, "| torchvision", torchvision.__version__,
      "| numpy", numpy.__version__, "| Pillow", PIL.__version__, "| cpu_count", os.cpu_count(),
      "| cuda", torch.cuda.is_available(), torch.cuda.device_count())
import evefl.fl.imagecache
print("import evefl.fl.imagecache: OK")
```

**G2. Find the inputs and check their hashes** (the walk does not enter the `images*` folders, so it takes seconds, not minutes).
It stops with a clear message if a dataset is not attached or a hash differs.
```python
import hashlib, os

def find_files(root, names, max_depth=6, skip_prefix=("images",)):
    """Walk /kaggle/input without descending into the 112k PNG folders (names starting with 'images')."""
    found = {n: [] for n in names}
    base = root.rstrip("/").count("/")
    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath.count("/") - base
        dirnames[:] = [d for d in dirnames if not d.startswith(skip_prefix)] if depth < max_depth else []
        for n in names:
            if n in filenames:
                found[n].append(os.path.join(dirpath, n))
    return found

def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

EXPECTED_WEIGHTS = "9803814b9183a8bdd46cd0c9cd178ec98b51de241084613e975f4fa54d9a57f2"
EXPECTED_CACHE = "db497ea60fdc0f336ac94c3ff8d935bf755ddcb61ea567fa38fb3a01806a456f"

hits = find_files("/kaggle/input", ["Data_Entry_2017.csv", "resnet18_imagenet.pth", "index.json"])
print(hits)

csv = hits["Data_Entry_2017.csv"]
assert len(csv) == 1, f"expected one Data_Entry_2017.csv, found {csv}: attach the nih-chest-xrays dataset"
DATA = os.path.dirname(csv[0])

w = hits["resnet18_imagenet.pth"]
assert len(w) == 1, f"expected one resnet18_imagenet.pth, found {w}: attach parineetarana1/evefl-resnet18-weights"
WEIGHTS = w[0]
assert sha256_of(WEIGHTS) == EXPECTED_WEIGHTS, "STOP: weights SHA-256 differs from the expected value"

caches = [os.path.dirname(p) for p in hits["index.json"]
          if os.path.exists(os.path.join(os.path.dirname(p), "shard_00000.npy"))]
assert len(caches) == 1, f"expected one attached cache dataset (index.json + shard_00000.npy), found {caches}"
CACHE = caches[0]
assert sha256_of(os.path.join(CACHE, "index.json")) == EXPECTED_CACHE, "STOP: cache_sha256 differs from the recorded value"

print("DATA   =", DATA)
print("WEIGHTS=", WEIGHTS, "(sha256 OK)")
print("CACHE  =", CACHE, "(index.json sha256 = the recorded cache_sha256)")
```

**G3. The lite partition** (5% of patients, seed 42; a fresh session has none, and the server would otherwise partition the
FULL dataset). Expect hospital images 1095, 1232, 1975, test 664, val 512, `RESULT: OK`.
```python
%cd /kaggle/temp/EveFL
!python -c "from evefl.fl.dataset import partition_and_save; partition_and_save('{DATA}', '/kaggle/working/partitions_lite', seed=42, subset_fraction=0.05)"
!python scripts/audit_partition.py --data-root {DATA} --partition-root /kaggle/working/partitions_lite | tail -25
```

**G4. Verify the attached cache dataset** (re-hashes every shard, compares 200 rows with the PNG pipeline; a few minutes).
Expect `shard_hash_mismatches: []`, both mismatch counts 0, `VERIFY: OK`.
```python
%cd /kaggle/temp/EveFL
!python scripts/build_image_cache.py --data-root {DATA} --verify {CACHE} --verify-sample 200
```

**G5. The lite alpha = 0 run on the cache backend** (the same settings as the profile run, plus the backend).
```python
%cd /kaggle/temp/EveFL
!python -m evefl.fl.server \
    --data-root {DATA} --partition-root /kaggle/working/partitions_lite \
    --num-rounds 5 --local-epochs 1 --batch-size 32 --seed 0 \
    --intercept-probability 0.0 --qubits 1024 --sample-fraction 0.25 \
    --pretrained --pretrained-weights {WEIGHTS} \
    --image-backend uint8_cache --image-cache-root {CACHE} \
    --experiment-name lite_alpha0_seed0_cache --output /kaggle/working/results/lite_alpha0_seed0_cache.json
```

**G6. Summary** (paste back everything; it prints the numbers the acceptance rule reads).
```python
%cd /kaggle/temp/EveFL
import json
OUT = "/kaggle/working/results/lite_alpha0_seed0_cache.json"
!python scripts/summarize_run.py {OUT}
r = json.load(open(OUT))
e = r["experiment"]
p = e["profile"]
s = p["summary"]
print("data_backend:", json.dumps(e["data_backend"]))
print("config_hash :", e["config_hash"])
print("device      :", p["device"]["default_device"], [d["name"] for d in p["device"]["devices"]])
print("data wait share, all rounds  :", s["data_wait_share_loaded"])
print("data wait share, from round 2:", s["data_wait_share_loaded_from_round_2"])
print("images/s per phase           :", {k: round(v, 1) for k, v in s["images_per_s"].items()})
for rd in r["rounds"]:
    for name, ph in rd["timing"]["phases"].items():
        print("round", rd["round"], name, {k: round(ph[k], 1) for k in ("phase_s", "data_wait_s", "compute_s", "n_images") if k in ph})
print("gpu util mean per phase, round 2:", {k: v["mean_util_pct"] for k, v in p["gpu_by_phase"].items() if k.startswith("round2:")})
```
Reading it: the first pass over a freshly attached dataset reads the shards from disk (cold page cache), and G4 already reads
every shard once, so round 1 is the least representative; the acceptance rule reads the data-wait share from round 2 onward and
the round 1 figure is reported next to it. Say in your paste that G4 ran before G5.

## First full-data baseline pilot (B0, one seed): one GPU notebook, run with Save & Run All
Pre-registered in docs/06 ("first full-data baseline pilot"), which must be merged before this runs. Projected about 3.8 h on
one T4 (the 12 h session limit leaves a wide margin). A committed run does not depend on the browser, but a version that fails
keeps no output, and the runner writes its results JSON only at the end, so the per-round lines are also tee'd to a log file.
Settings: Accelerator GPU, Internet on, `GITHUB_TOKEN` secret enabled, datasets attached: `nih-chest-xrays`, the weights dataset
and `evefl-cache-u8-256`. Cells, in this order:
1. **G1** and 2. **G2** of "Using the cache in a GPU notebook" above, unchanged (clone `main`, install; find the inputs and
   check the weights and cache hashes).
3. **The FULL partition** (partition seed 42; the server would otherwise partition with the run seed), its audit, and the check
   against the recorded hashes:
```python
%cd /kaggle/temp/EveFL
import json
!python -c "from evefl.fl.dataset import partition_and_save; partition_and_save('{DATA}', '/kaggle/working/partitions_full', seed=42)"
!python scripts/audit_partition.py --data-root {DATA} --partition-root /kaggle/working/partitions_full | tail -12
meta = json.load(open("/kaggle/working/partitions_full/partition_meta.json"))
audit = json.load(open("docs/validation/partition_audit_real.json"))
recorded = json.load(open("docs/validation/kaggle_rerun_2026-10-04.json"))["partition_full_rerun"]["index_sha256"]
print("counts identical to the audited split:", all(meta["audit"]["stats"][k] == audit["stats"][k] for k in audit["stats"]))
print("numpy", meta["numpy_version"], "| index hashes equal the 2026-10-04 record:", meta["index_sha256"] == recorded)
assert meta["audit"]["ok"], "STOP: partition overlap"
```
4. **Verify the attached cache** (a few minutes; it also warms the page cache):
```python
%cd /kaggle/temp/EveFL
!python scripts/build_image_cache.py --data-root {DATA} --verify {CACHE} --verify-sample 200
```
5. **The pilot** (about 4 h; the log is also saved):
```python
%cd /kaggle/temp/EveFL
!mkdir -p /kaggle/working/results
!python -m evefl.fl.server \
    --data-root {DATA} --partition-root /kaggle/working/partitions_full \
    --num-rounds 50 --local-epochs 1 --batch-size 32 --seed 0 \
    --intercept-probability 0.0 --qubits 1024 --sample-fraction 0.25 \
    --pretrained --pretrained-weights {WEIGHTS} \
    --image-backend uint8_cache --image-cache-root {CACHE} --test-eval-every 5 \
    --experiment-name pilot_b0_full_e1_seed0 --output /kaggle/working/results/pilot_b0_full_e1_seed0.json \
    2>&1 | tee /kaggle/working/results/pilot_b0_full_e1_seed0.log
```
6. **Checks and the numbers to paste back:**
```python
%cd /kaggle/temp/EveFL
import json
OUT = "/kaggle/working/results/pilot_b0_full_e1_seed0.json"
!python scripts/summarize_run.py {OUT}
r = json.load(open(OUT))
e = r["experiment"]
print("state_counts:", e["state_counts"], "| failures:", e["n_failures"], "| elapsed s:", round(e["elapsed_seconds"]), "| projected 3.8 h = 13680 s")
print("weights sha256:", e["init_weights"]["sha256"])
print("data_backend:", json.dumps({k: e["data_backend"][k] for k in ("name", "cache_sha256")}))
print("data wait share from round 2:", e["profile"]["summary"]["data_wait_share_loaded_from_round_2"])
print(f"{'round':>5} {'val_loss':>9} {'val_AUC*':>9} {'test_loss':>10} {'test_AUC*':>10}")
for rd in r["rounds"]:
    t = rd.get("server_eval")
    print(f"{rd['round']:>5} {rd['val_eval']['loss']:>9.4f} {rd['val_eval']['metrics']['macro_auc_roc_excl_thin']:>9.4f} "
          + (f"{t['loss']:>10.4f} {t['metrics']['macro_auc_roc_excl_thin']:>10.4f}" if t else ""))
last = r["rounds"][-1]["server_eval"]["metrics"]
print("final-round per-class test AUC:", {k: round(v, 3) for k, v in last["per_class_auc"].items()})
print("thin classes:", last["thin_classes"])
v1, v50 = (r["rounds"][i]["val_eval"]["metrics"]["macro_auc_roc_excl_thin"] for i in (0, -1))
print("sanity: val_AUC* round 50 > round 1:", v50 > v1)
```

## What I will check when you paste
- cell 1 printed a commit line, `flwr 1.13.0` and the import line, with no traceback and no token anywhere in the output,
- cell 2 printed `OK`,
- all overlap counts are 0, the hashes are present and `counts identical ... : True`,
- the results JSON has `pretrained`, `config_hash`, `qkd`, `optimizer`, `policy`, `environment`, per-round `val_eval` and
  `server_eval`, and the recorded weights hash matches,
- alpha = 0 stayed SECURE every round with nothing excluded (a false CAUTION here would be a bug).
