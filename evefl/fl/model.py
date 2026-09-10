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


def build_resnet18(pretrained: bool = True) -> nn.Module:
    """
    ResNet-18 with its final FC layer replaced by a 14-way linear head.

    Outputs are raw logits (no sigmoid applied in the model) so training
    can use BCEWithLogitsLoss, which fuses sigmoid + log for numerical
    stability. Apply torch.sigmoid(logits) at inference time.
    """
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


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
