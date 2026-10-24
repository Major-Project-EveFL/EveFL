"""
Evaluate a saved checkpoint ONCE on the test split.

    python scripts/eval_checkpoint.py --checkpoint checkpoints/pilot_b0_full_e1_seed1_best_val.pt \\
        --data-root D --partition-root P --image-backend uint8_cache --image-cache-root C --output results/eval_best_val.json

Used for the checkpoint the demonstration serves: it is chosen by validation AUC while the run is going (see `--save-model`), and its
test AUC is obtained here afterwards. The number selects nothing. Only the test split is accepted; the validation split already drove
the selection. The output records which checkpoint (file, round, SHA-256) was scored.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evefl import environment, inference  # noqa: E402
from evefl.fl.dataset import create_image_backend  # noqa: E402
from evefl.fl.evaluation import evaluate_global_model  # noqa: E402
from evefl.fl.model import get_device  # noqa: E402

LABEL = ("Checkpoint evaluated once on the test split after selection by validation. The test result selects nothing and is "
         "reported with the checkpoint's file name, round and SHA-256. Single model, single evaluation: no confidence interval.")


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    p = argparse.ArgumentParser(description="Evaluate one saved checkpoint once on the test split.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--partition-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", choices=["test"], default="test", help="only the test split is accepted")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--image-backend", type=str, default="png")
    p.add_argument("--image-cache-root", type=Path, default=None)
    args = p.parse_args(argv)
    if not args.checkpoint.is_file():
        p.error(f"checkpoint not found: {args.checkpoint}")

    loaded = inference.load_checkpoint(args.checkpoint)
    ndarrays = [tensor.detach().cpu().numpy() for tensor in loaded.model.state_dict().values()]
    backend = create_image_backend(args.image_backend, args.image_cache_root)
    loss, metrics = evaluate_global_model(
        ndarrays, data_root=args.data_root, partition_root=args.partition_root, batch_size=args.batch_size,
        device=get_device(), split=args.split, image_backend=backend)
    result = {
        "label": LABEL, "split": args.split, "loss": loss, "metrics": metrics,
        "checkpoint": {"file": loaded.file, "kind": loaded.kind, "round": loaded.round,
                       "val_auc_excl_thin": loaded.val_auc_excl_thin, "sha256": loaded.sha256, "metadata": loaded.metadata},
        "data_backend": backend.describe(), "environment": environment.runtime_environment(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    print(f"test macro AUC {metrics.get('macro_auc_roc'):.4f} (non-thin {metrics.get('macro_auc_roc_excl_thin'):.4f}) "
          f"for {loaded.file}, round {loaded.round}; written {args.output}")
    return result


if __name__ == "__main__":
    main()
