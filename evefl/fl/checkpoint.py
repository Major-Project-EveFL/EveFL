"""
Model checkpoints for a federated run (`--save-model DIR`).

Two files are written at the end of a run, both in the ResNet-18 `state_dict` order that `get_model_parameters` uses:
- `<name>_final.pt`: the global model after the last round;
- `<name>_best_val.pt`: the global model of the round with the highest VALIDATION macro AUC over non-thin classes
  (`macro_auc_roc_excl_thin`); ties go to the later round, which is the rule used for the pilot's secondary result.
The test split is never read here: the choice is made from validation results only, so it can be fixed before any test number of
the run is seen. Checkpoints are not part of the results JSON and not part of git (`*.pt` is ignored); the manifest returned by
`finalize` (file names, rounds, SHA-256, sizes, metadata) is recorded in the results JSON under `experiment.checkpoints` and in
`<name>_checkpoints.json` next to the files. Saving changes nothing that is trained, so it is not part of the config hash.

The best parameters are copied when they are observed, so the caller may reuse its buffers.
File format: `torch.save` of a dict with only tensors and plain Python values, loadable with `torch.load(..., weights_only=True)`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from flwr.common import Parameters, parameters_to_ndarrays

from evefl.fl.model import build_resnet18

log = logging.getLogger(__name__)

CHECKPOINT_FORMAT = "evefl-checkpoint-v1"
SELECTION_METRIC = "macro_auc_roc_excl_thin"
SELECTION_RULE = ("highest validation macro AUC over non-thin classes (ties to the later round); the test split is never used "
                  "for this choice")


def _state_keys() -> List[str]:
    return list(build_resnet18(pretrained=False).state_dict().keys())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class CheckpointSaver:
    """Observes the global model after each round, remembers the best by validation AUC, and writes both files at the end."""

    def __init__(self, directory: str | Path, *, name: str, metadata: Mapping[str, Any]):
        self.directory = Path(directory)
        self.name = name
        self.metadata: Dict[str, Any] = dict(metadata)
        self._keys = _state_keys()
        self._best: Optional[Tuple[int, float, List[np.ndarray]]] = None

    def _ndarrays(self, parameters: Parameters) -> List[np.ndarray]:
        arrays = parameters_to_ndarrays(parameters)
        if len(arrays) != len(self._keys):
            raise ValueError(f"expected {len(self._keys)} tensors for ResNet-18, got {len(arrays)}")
        return arrays

    def observe(self, server_round: int, parameters: Parameters, val_eval: Optional[Tuple[float, Mapping[str, Any]]]) -> None:
        """Called after every round with the global parameters and that round's validation result (or None)."""
        arrays = self._ndarrays(parameters)   # validates the length even when nothing is kept
        if val_eval is None:
            return
        value = val_eval[1].get(SELECTION_METRIC)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            return
        if self._best is None or float(value) >= self._best[1]:
            self._best = (server_round, float(value), [np.array(a, copy=True) for a in arrays])

    def _write(self, label: str, server_round: int, arrays: Sequence[np.ndarray], val_auc: Optional[float]) -> Dict[str, Any]:
        state = OrderedDict((key, torch.from_numpy(np.array(array, copy=True))) for key, array in zip(self._keys, arrays))
        payload = {"format": CHECKPOINT_FORMAT, "round": int(server_round), "val_auc_excl_thin": val_auc,
                   "state_dict": state, "metadata": self.metadata}
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{self.name}_{label}.pt"
        torch.save(payload, path)
        return {"file": path.name, "round": int(server_round), "val_auc_excl_thin": val_auc,
                "sha256": _sha256(path), "bytes": path.stat().st_size}

    def finalize(self, final_parameters: Parameters, *, final_round: int) -> Dict[str, Any]:
        """Write `<name>_final.pt` and, if any round had a usable validation AUC, `<name>_best_val.pt`; return the manifest."""
        manifest: Dict[str, Any] = {
            "format": CHECKPOINT_FORMAT, "selection": SELECTION_RULE, "metadata": self.metadata,
            "final": self._write("final", final_round, self._ndarrays(final_parameters), None),
            "best_validation": None,
        }
        if self._best is not None:
            best_round, best_value, best_arrays = self._best
            manifest["best_validation"] = self._write("best_val", best_round, best_arrays, best_value)
        else:
            manifest["note"] = "no validation AUC was available in any round, so no best-validation checkpoint was written"
        (self.directory / f"{self.name}_checkpoints.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        log.info("Checkpoints written to %s: final round %d, best validation round %s", self.directory, final_round,
                 manifest["best_validation"]["round"] if manifest["best_validation"] else "n/a")
        return manifest
