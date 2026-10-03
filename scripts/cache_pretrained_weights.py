"""
Save the torchvision ImageNet ResNet-18 weights to a plain file, so they can be
uploaded as a Kaggle dataset and loaded offline (Kaggle notebooks may have no
internet). Run this once on a machine WITH internet:

    python scripts/cache_pretrained_weights.py --out resnet18_imagenet1k_v1.pth

Then create a Kaggle dataset containing that file and pass it to the experiment:

    python -m evefl.fl.server ... --pretrained --pretrained-weights /kaggle/input/<dataset>/resnet18_imagenet1k_v1.pth

The SHA-256 of the file is recorded in every results JSON.
"""

import argparse
import sys
from pathlib import Path

import torch
from torchvision import models
from torchvision.models import ResNet18_Weights

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evefl.fl.model import file_sha256  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Cache torchvision ResNet-18 ImageNet weights to a file")
    parser.add_argument("--out", type=Path, default=Path("resnet18_imagenet1k_v1.pth"))
    args = parser.parse_args()

    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"Saved {args.out} ({args.out.stat().st_size / 1e6:.1f} MB) sha256={file_sha256(args.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
