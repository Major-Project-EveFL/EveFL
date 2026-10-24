"""
Inference with a saved checkpoint, for the upload page and the checkpoint evaluation script.

A checkpoint is read only from a local folder with `torch.load(..., weights_only=True)` (tensors and plain values; no arbitrary
objects), and must be one of ours (`evefl-checkpoint-v1`, written by `--save-model`). An image goes through exactly the evaluation
preprocessing the model was scored with: convert to RGB, resize to 256, centre-crop 224, ImageNet normalisation
(`evefl.fl.model.get_eval_transform`); a grey, RGB or RGBA copy of the same picture therefore gives the same prediction.

The output is 14 independent probabilities, one per pathology label of ChestX-ray14 (multi-label, sigmoid). They are model scores
from a research prototype trained on a public dataset: they are not diagnoses and no clinical claim is made.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
from PIL import Image

from evefl.fl.checkpoint import CHECKPOINT_FORMAT
from evefl.fl.model import CHESTXRAY_LABELS, build_resnet18, get_eval_transform

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
BEST_SUFFIX = "_best_val.pt"
FINAL_SUFFIX = "_final.pt"


@dataclass
class LoadedModel:
    model: torch.nn.Module
    file: str
    kind: str                       # "best_val", "final" or "unknown"
    round: int
    val_auc_excl_thin: Any          # float for a best-validation checkpoint, None for a final one
    sha256: str
    metadata: Dict[str, Any]


def find_checkpoints(directory: str | Path) -> List[Path]:
    """Checkpoint files in `directory`: best-validation files first (newest first), then final-round files."""
    folder = Path(directory)
    if not folder.is_dir():
        return []
    newest_first = lambda paths: sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)  # noqa: E731
    return newest_first(folder.glob(f"*{BEST_SUFFIX}")) + newest_first(folder.glob(f"*{FINAL_SUFFIX}"))


def _kind(path: Path) -> str:
    if path.name.endswith(BEST_SUFFIX):
        return "best_val"
    return "final" if path.name.endswith(FINAL_SUFFIX) else "unknown"


def load_checkpoint(path: str | Path) -> LoadedModel:
    """Load one of our checkpoints into an eval-mode ResNet-18; ValueError if it is not one or does not fit the model."""
    path = Path(path)
    raw = path.read_bytes()
    try:
        payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 - any unreadable file is reported as one clear error
        raise ValueError(f"could not read {path.name} as a checkpoint: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT or "state_dict" not in payload:
        raise ValueError(f"{path.name} is not an EveFL checkpoint (expected format {CHECKPOINT_FORMAT!r})")
    model = build_resnet18(pretrained=False)
    try:
        model.load_state_dict(payload["state_dict"], strict=True)
    except RuntimeError as exc:
        raise ValueError(f"{path.name} does not match the ResNet-18 used here: {exc}") from exc
    model.eval()
    return LoadedModel(model=model, file=path.name, kind=_kind(path), round=int(payload.get("round", -1)),
                       val_auc_excl_thin=payload.get("val_auc_excl_thin"), sha256=hashlib.sha256(raw).hexdigest(),
                       metadata=dict(payload.get("metadata") or {}))


def open_upload(data: bytes) -> Image.Image:
    """Decode uploaded bytes into a PIL image, refusing very large or unreadable files."""
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError(f"the file is too large (limit {MAX_UPLOAD_BYTES // (1024 * 1024)} MB)")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"could not read the file as an image: {exc}") from exc
    return image


def predict(loaded: LoadedModel, image: Image.Image) -> pd.DataFrame:
    """Probability for each of the 14 findings, sorted from highest to lowest. Independent sigmoids, not a distribution."""
    tensor = get_eval_transform()(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probabilities = torch.sigmoid(loaded.model(tensor))[0].tolist()
    table = pd.DataFrame({"finding": CHESTXRAY_LABELS, "probability": probabilities})
    return table.sort_values("probability", ascending=False, kind="stable").reset_index(drop=True)
