"""
Server-side centralized evaluation against the shared held-out
ChestX-ray14 test set.

Plugs into `EveFLStrategy(evaluate_fn=...)`, matching the
`evaluate_fn(server_round, parameters_ndarrays, config) -> Optional[(loss, metrics)]`
contract that `EveFLStrategy.evaluate()` calls (see strategy.py).

Requires scikit-learn (added to requirements.txt) for roc_auc_score.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from evefl.fl.client import set_model_parameters
from evefl.fl.dataset import get_test_dataloader
from evefl.fl.model import CHESTXRAY_LABELS, build_resnet18, get_device

log = logging.getLogger(__name__)


def _per_class_auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    """
    Per-pathology AUC-ROC.

    A class is SKIPPED (not scored as 0, not scored as 0.5) if the
    held-out split happens to contain only one label value for it —
    AUC is mathematically undefined there, and on a small demo subset
    this is common. Better to omit a number than fabricate one.
    """
    from sklearn.metrics import roc_auc_score

    per_class: Dict[str, float] = {}
    for i, label in enumerate(CHESTXRAY_LABELS):
        col = y_true[:, i]
        if len(np.unique(col)) < 2:
            continue
        per_class[label] = float(roc_auc_score(col, y_score[:, i]))
    return per_class


@torch.no_grad()
def evaluate_global_model(
    ndarrays,
    *,
    data_root: Path,
    partition_root: Path,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Load `ndarrays` into a fresh ResNet-18 and evaluate against the
    shared IID test split written by `dataset.partition_and_save()`.

    Returns (mean_bce_loss, metrics) where metrics includes
    `mean_auc_roc` (averaged over classes with a defined AUC on this
    split) and one `auc_<pathology>` entry per scorable class.
    """
    device = device or get_device()
    model = build_resnet18(pretrained=False)
    set_model_parameters(model, ndarrays)
    model.to(device)
    model.eval()

    criterion = nn.BCEWithLogitsLoss()
    loader = get_test_dataloader(data_root, partition_root, batch_size=batch_size)

    total_loss, total_examples = 0.0, 0
    all_targets, all_probs = [], []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)

        n = int(images.shape[0])
        total_loss += float(loss.item()) * n
        total_examples += n
        all_targets.append(targets.cpu().numpy())
        all_probs.append(torch.sigmoid(logits).cpu().numpy())

    if total_examples == 0:
        log.warning("Global eval: test set was empty — check partition_root.")
        return float("nan"), {"n_test_examples": 0}

    mean_loss = total_loss / total_examples
    y_true = np.concatenate(all_targets, axis=0)
    y_score = np.concatenate(all_probs, axis=0)

    per_class_auc = _per_class_auc_roc(y_true, y_score)
    mean_auc = float(np.mean(list(per_class_auc.values()))) if per_class_auc else float("nan")

    metrics: Dict[str, float] = {"mean_auc_roc": mean_auc, "n_test_examples": total_examples}
    metrics.update({f"auc_{label}": v for label, v in per_class_auc.items()})

    log.info(
        "Global eval | loss=%.4f mean_auc_roc=%s (n=%d, %d/%d classes scorable)",
        mean_loss,
        f"{mean_auc:.4f}" if per_class_auc else "n/a",
        total_examples, len(per_class_auc), len(CHESTXRAY_LABELS),
    )
    return mean_loss, metrics


def make_evaluate_fn(
    *,
    data_root: Path,
    partition_root: Path,
    batch_size: int = 64,
    every_n_rounds: int = 1,
    num_rounds: Optional[int] = None,
):
    """
    Build the `evaluate_fn` EveFLStrategy expects.

    Set `every_n_rounds > 1` to skip most rounds during a fast demo run
    (a full forward pass over the test set on every single round adds
    up quickly on CPU/limited GPU time) while still always evaluating
    the final round if `num_rounds` is given.
    """
    device = get_device()

    def evaluate_fn(server_round: int, ndarrays, config) -> Optional[Tuple[float, Dict[str, float]]]:
        is_final_round = num_rounds is not None and server_round == num_rounds
        if every_n_rounds > 1 and server_round % every_n_rounds != 0 and not is_final_round:
            return None
        return evaluate_global_model(
            ndarrays,
            data_root=data_root,
            partition_root=partition_root,
            batch_size=batch_size,
            device=device,
        )

    return evaluate_fn
