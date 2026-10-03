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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from evefl.fl.client import set_model_parameters
from evefl.fl.dataset import get_test_dataloader
from evefl.fl.model import CHESTXRAY_LABELS, build_resnet18, get_device

log = logging.getLogger(__name__)

# Pre-registered in docs/06 (before any tuning): a class with fewer than this many positives (or negatives) in the
# evaluated split is "thin"; its AUC is too noisy to steer model selection. A convention, not a power calculation.
MIN_POSITIVES_FOR_SELECTION = 20


def _per_class_auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[Dict[str, float], List[str]]:
    """
    Per-pathology AUC-ROC. Returns (per_class_auc, skipped_classes).

    A class is SKIPPED (not scored as 0, not scored as 0.5) if the
    held-out split contains only one label value for it (no positives,
    or no negatives) — AUC is mathematically undefined there, and on a
    small demo subset this is common. Better to omit a number than
    fabricate one; the skipped names are returned so they get logged.
    """
    from sklearn.metrics import roc_auc_score

    per_class: Dict[str, float] = {}
    skipped: List[str] = []
    for i, label in enumerate(CHESTXRAY_LABELS):
        col = y_true[:, i]
        if len(np.unique(col)) < 2:
            skipped.append(label)
            continue
        per_class[label] = float(roc_auc_score(col, y_score[:, i]))
    return per_class, skipped


def _support_summary(
    y_true: np.ndarray, per_class_auc: Dict[str, float], min_positives: int = MIN_POSITIVES_FOR_SELECTION
) -> Dict[str, Any]:
    """
    Per-class support and the thin-class flag.

    A SCORED class is thin if it has fewer than `min_positives` positives or fewer than `min_positives`
    negatives in `y_true`. Returns n_positives_per_class (all classes), thin_classes (names, label order),
    and macro_auc_roc_excl_thin: the mean AUC over scored, non-thin classes (NaN if there are none). Thin
    classes stay in `per_class_auc` and in the all-class `macro_auc_roc`; only the selection macro drops them.
    """
    n_rows = int(y_true.shape[0])
    positives = {label: int(y_true[:, i].sum()) for i, label in enumerate(CHESTXRAY_LABELS)}
    thin = [
        label for label in CHESTXRAY_LABELS
        if label in per_class_auc and min(positives[label], n_rows - positives[label]) < min_positives
    ]
    kept = [auc for label, auc in per_class_auc.items() if label not in thin]
    return {
        "n_positives_per_class": positives,
        "thin_classes": thin,
        "min_positives": min_positives,
        "macro_auc_roc_excl_thin": float(np.mean(kept)) if kept else float("nan"),
    }


@torch.no_grad()
def evaluate_global_model(
    ndarrays,
    *,
    data_root: Path,
    partition_root: Path,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
    split: str = "test",
) -> Tuple[float, Dict[str, Any]]:
    """
    Load `ndarrays` into a fresh ResNet-18 and evaluate against the
    shared held-out test split written by `dataset.partition_and_save()`.

    Returns (mean_bce_loss, metrics). `metrics` is JSON-ready:
        macro_auc_roc     mean AUC over the scorable classes (NaN if none)
        per_class_auc     {pathology: AUC} for scorable classes only
        skipped_classes   pathologies with no positives (or no negatives)
                          in the test split, which are left out of the macro
        n_scored_classes, n_test_examples
        n_positives_per_class, thin_classes, min_positives, macro_auc_roc_excl_thin
                          support per class; scored classes with < min_positives positives (or
                          negatives) are "thin" and left out of macro_auc_roc_excl_thin, which is the
                          model-selection metric on the validation split (docs/06)
    Written verbatim into each round's `server_eval` entry of the results JSON.
    """
    device = device or get_device()
    model = build_resnet18(pretrained=False)
    set_model_parameters(model, ndarrays)
    model.to(device)
    model.eval()

    criterion = nn.BCEWithLogitsLoss()
    loader = get_test_dataloader(data_root, partition_root, batch_size=batch_size, split=split)

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
        return float("nan"), {"macro_auc_roc": float("nan"), "per_class_auc": {}, "skipped_classes": list(CHESTXRAY_LABELS),
                              "n_scored_classes": 0, "n_test_examples": 0,
                              "n_positives_per_class": {label: 0 for label in CHESTXRAY_LABELS}, "thin_classes": [],
                              "min_positives": MIN_POSITIVES_FOR_SELECTION, "macro_auc_roc_excl_thin": float("nan")}

    mean_loss = total_loss / total_examples
    y_true = np.concatenate(all_targets, axis=0)
    y_score = np.concatenate(all_probs, axis=0)

    per_class_auc, skipped = _per_class_auc_roc(y_true, y_score)
    macro_auc = float(np.mean(list(per_class_auc.values()))) if per_class_auc else float("nan")

    metrics: Dict[str, Any] = {
        "macro_auc_roc": macro_auc,
        "per_class_auc": per_class_auc,
        "skipped_classes": skipped,
        "n_scored_classes": len(per_class_auc),
        "n_test_examples": total_examples,
        **_support_summary(y_true, per_class_auc),
    }

    log.info(
        "Global eval | loss=%.4f macro_auc_roc=%s (n=%d, %d/%d classes scored)",
        mean_loss,
        f"{macro_auc:.4f}" if per_class_auc else "n/a",
        total_examples, len(per_class_auc), len(CHESTXRAY_LABELS),
    )
    if skipped:
        log.warning("Global eval | classes skipped (single-valued in test split): %s", ", ".join(skipped))
    if metrics["thin_classes"]:
        log.info("Global eval | thin classes (< %d positives or negatives; excluded from macro_auc_roc_excl_thin): %s",
                 MIN_POSITIVES_FOR_SELECTION, ", ".join(metrics["thin_classes"]))
    return mean_loss, metrics


def make_evaluate_fn(
    *,
    data_root: Path,
    partition_root: Path,
    batch_size: int = 64,
    every_n_rounds: int = 1,
    num_rounds: Optional[int] = None,
    split: str = "test",
):
    """
    Build the `evaluate_fn` EveFLStrategy expects.

    The runner calls this after EVERY round, LOCKDOWN rounds included (the
    model is then the unchanged last good one), so the AUC curve shows the
    flat stretches. Only raise `every_n_rounds` above 1 to save time.

    Set `every_n_rounds > 1` to skip most rounds during a fast demo run
    (a full forward pass over the test set on every single round adds
    up quickly on CPU/limited GPU time) while still always evaluating
    the final round if `num_rounds` is given.
    """
    device = get_device()

    def evaluate_fn(server_round: int, ndarrays, config) -> Optional[Tuple[float, Dict[str, Any]]]:
        is_final_round = num_rounds is not None and server_round == num_rounds
        if every_n_rounds > 1 and server_round % every_n_rounds != 0 and not is_final_round:
            return None
        return evaluate_global_model(
            ndarrays,
            data_root=data_root,
            partition_root=partition_root,
            batch_size=batch_size,
            device=device,
            split=split,
        )

    return evaluate_fn
