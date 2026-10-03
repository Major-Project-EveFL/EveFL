"""
ResNet-18 for NIH ChestX-ray14 multi-label classification.

Nothing quantum happens in this file. The quantum layer (BB84 / QBER,
see evefl/quantum/) monitors the communication channel that gradients
travel over between clients and the server — it never touches image
data. X-ray images never leave their hospital; only model parameters
are exchanged, and QBER tells the orchestration layer (strategy.py)
whether that exchange looks like it's being intercepted.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights

# NIH ChestX-ray14 pathology labels, in the fixed order used for the
# one-hot / multi-hot label matrix everywhere in evefl.fl.
CHESTXRAY_LABELS: list[str] = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
]
NUM_CLASSES = len(CHESTXRAY_LABELS)  # 14

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_resnet18(pretrained: bool = True, weights_path: Optional[str | Path] = None) -> nn.Module:
    """
    ResNet-18 with its final FC layer replaced by a 14-way linear head.

    Outputs are raw logits (no sigmoid applied in the model) so training
    can use BCEWithLogitsLoss, which fuses sigmoid + log for numerical
    stability. Apply torch.sigmoid(logits) at inference time.

    Args:
        pretrained: start from ImageNet weights (the paper's setting).
        weights_path: with `pretrained=True`, load the torchvision ResNet-18
            ImageNet state dict from this local file instead of downloading it
            (Kaggle notebooks often have no internet; upload the file as a
            Kaggle dataset, see scripts/cache_pretrained_weights.py). Ignored,
            and rejected, when `pretrained=False`.
    """
    if weights_path is not None and not pretrained:
        raise ValueError("weights_path was given but pretrained=False; refusing to guess which one is meant.")

    if pretrained and weights_path is not None:
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"Pretrained weights file not found: {weights_path}")
        model = models.resnet18(weights=None)
        model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    elif pretrained:
        try:
            model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        except Exception as exc:  # network failure surfaces as URLError / RuntimeError
            raise RuntimeError(
                "Could not download the ImageNet ResNet-18 weights (no internet?). Provide them as a "
                "local file via weights_path / --pretrained-weights (see scripts/cache_pretrained_weights.py), "
                "or set pretrained=False explicitly."
            ) from exc
    else:
        model = models.resnet18(weights=None)

    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def describe_initialisation(pretrained: bool, weights_path: Optional[str | Path] = None) -> dict:
    """What the global model was initialised from, for the results JSON."""
    if not pretrained:
        return {"pretrained": False, "source": "random_init", "weights_file": None, "sha256": None}
    if weights_path is None:
        return {"pretrained": True, "source": "torchvision IMAGENET1K_V1 (download/cache)",
                "weights_file": None, "sha256": None}
    return {"pretrained": True, "source": "local file", "weights_file": Path(weights_path).name,
            "sha256": file_sha256(weights_path)}


def get_criterion() -> nn.Module:
    """Multi-label BCE-with-logits — one independent binary decision per pathology."""
    return nn.BCEWithLogitsLoss()


def get_train_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # Chest X-rays are greyscale on disk but loaded as 3-channel RGB
        # (PIL .convert("RGB")) since ResNet-18 expects 3 input channels.
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
