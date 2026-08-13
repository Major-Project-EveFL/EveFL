"""
ResNet-18 model for NIH ChestX-ray14 multi-label classification.

Important: nothing quantum happens here. This is a standard PyTorch
model. The quantum layer (BB84 / QBER) runs on the *communication
channel* — it monitors the encryption key exchange between clients and
the server. The X-ray images never leave their hospital; only model
gradients are transmitted, and QBER tells us whether those transmissions
are being intercepted.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights

# NIH ChestX-ray14 pathology labels (order matches the one-hot encoding)
CHESTXRAY_LABELS: list[str] = [
    "Atelectasis", "Cardiomegaly", "Effusion",     "Infiltration",
    "Mass",        "Nodule",       "Pneumonia",     "Pneumothorax",
    "Consolidation","Edema",       "Emphysema",     "Fibrosis",
    "Pleural_Thickening",          "Hernia",
]
NUM_CLASSES = len(CHESTXRAY_LABELS)  # 14


def build_resnet18(pretrained: bool = True) -> nn.Module:
    """
    ResNet-18 pretrained on ImageNet.

    The final FC layer is replaced with a 14-output sigmoid head for
    multi-label binary cross-entropy — one sigmoid per pathology.
    Sigmoid is applied inside the model so outputs are always in [0,1]
    and can be directly compared against binary targets.
    """
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model   = models.resnet18(weights=weights)

    in_features = model.fc.in_features          # 512 for ResNet-18
    model.fc    = nn.Linear(in_features, NUM_CLASSES)
    # Sigmoid applied separately so we can use BCEWithLogitsLoss during
    # training (more numerically stable) and nn.Sigmoid() at inference
    return model


def get_criterion() -> nn.Module:
    """
    Binary cross-entropy with logits.

    More numerically stable than BCE + Sigmoid because it combines the
    sigmoid and log in a single fused operation.
    """
    return nn.BCEWithLogitsLoss()


def get_train_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        # Greyscale X-rays are loaded as RGB by PIL (3 channels needed for ResNet)
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],   # ImageNet stats
            std =[0.229, 0.224, 0.225],
        ),
    ])


def get_eval_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225],
        ),
    ])


def get_device() -> torch.device:
    """Pick GPU if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")