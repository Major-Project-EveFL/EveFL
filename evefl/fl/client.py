"""
Flower client for one simulated hospital.

The client is deliberately "dumb" about security: it does local SGD
training and, if told to (via `config["fedprox_mu"] > 0`), adds a
FedProx proximal term. It does NOT decide whether an attack is
happening — that's the server-side strategy's job (see strategy.py),
based on that round's BB84 QBER. The client just obeys whatever
`config["state"]` it's handed:

    SECURE   -> plain local SGD, `fedprox_mu` is 0
    CAUTION  -> FedProx local SGD, `fedprox_mu` > 0 (keeps the local
                model from drifting far from the last trusted global
                model, since we're less sure this round's aggregate
                will be trustworthy)
    LOCKDOWN -> skip local training entirely, return the model
                unchanged with num_examples=0, so this round can never
                count toward the aggregate even if something upstream
                forgot to check the state.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from flwr.client import Client, NumPyClient
from flwr.common import Scalar

from evefl.fl.dataset import get_hospital_dataloader
from evefl.fl.model import build_resnet18, get_criterion, get_device
from evefl.orchestration.state_machine import SecurityState

log = logging.getLogger(__name__)

DEFAULT_LR = 1e-4
DEFAULT_LOCAL_EPOCHS = 1


# ---------------------------------------------------------------------------
# Parameter <-> ndarray helpers
# ---------------------------------------------------------------------------

def get_model_parameters(model: nn.Module) -> List[np.ndarray]:
    return [val.detach().cpu().numpy() for val in model.state_dict().values()]


def set_model_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    state_dict_keys = list(model.state_dict().keys())
    if len(parameters) != len(state_dict_keys):
        raise ValueError(
            f"Parameter count mismatch: model has {len(state_dict_keys)} tensors, "
            f"received {len(parameters)}."
        )
    new_state = {}
    for key, array, existing in zip(state_dict_keys, parameters, model.state_dict().values()):
        tensor = torch.from_numpy(array)
        if tensor.shape != existing.shape:
            raise ValueError(
                f"Shape mismatch for '{key}': model expects {tuple(existing.shape)}, "
                f"received {tuple(tensor.shape)}."
            )
        new_state[key] = tensor
    model.load_state_dict(new_state, strict=True)


def _fedprox_term(model: nn.Module, global_params: List[torch.Tensor], mu: float) -> torch.Tensor:
    """(mu / 2) * ||local_params - global_params||^2 — pulls local
    training back toward the last trusted global model."""
    proximal = torch.tensor(0.0, device=next(model.parameters()).device)
    for local_p, global_p in zip(model.parameters(), global_params):
        proximal = proximal + torch.sum((local_p - global_p) ** 2)
    return (mu / 2.0) * proximal


# ---------------------------------------------------------------------------
# Local training / evaluation loops
# ---------------------------------------------------------------------------

def _train_one_client(
    model: nn.Module,
    dataloader,
    *,
    device: torch.device,
    epochs: int,
    lr: float,
    fedprox_mu: float,
) -> Dict[str, float]:
    model.to(device)
    model.train()
    criterion = get_criterion()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    global_params = None
    if fedprox_mu > 0.0:
        global_params = [p.detach().clone() for p in model.parameters()]

    total_loss, n_batches = 0.0, 0
    for _epoch in range(epochs):
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            if global_params is not None:
                loss = loss + _fedprox_term(model, global_params, fedprox_mu)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1

    mean_loss = total_loss / n_batches if n_batches > 0 else float("nan")
    return {"train_loss": mean_loss, "n_batches": n_batches}


@torch.no_grad()
def _evaluate_one_client(model: nn.Module, dataloader, *, device: torch.device) -> Tuple[float, int]:
    model.to(device)
    model.eval()
    criterion = get_criterion()

    total_loss, total_examples = 0.0, 0
    for images, targets in dataloader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        loss = criterion(logits, targets)
        n = images.shape[0]
        total_loss += float(loss.item()) * n
        total_examples += n

    if total_examples == 0:
        return float("nan"), 0
    return total_loss / total_examples, total_examples


# ---------------------------------------------------------------------------
# The Flower client
# ---------------------------------------------------------------------------

class EveFLClient(NumPyClient):
    """One simulated hospital. Holds only its own local ResNet-18 copy
    and its own local (non-IID) data partition — never sees other
    hospitals' data or the aggregated data distribution."""

    def __init__(
        self,
        cid: str,
        data_root: Path,
        partition_root: Path,
        *,
        batch_size: int = 32,
        local_epochs: int = DEFAULT_LOCAL_EPOCHS,
        lr: float = DEFAULT_LR,
        device: torch.device | None = None,
    ):
        self.cid = cid
        self.hospital_id = int(cid)
        self.data_root = data_root
        self.partition_root = partition_root
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.lr = lr
        self.device = device or get_device()

        self.model = build_resnet18(pretrained=True)
        self._train_loader = None  # lazy: built on first fit(), dataset load is not free

    def _get_train_loader(self):
        if self._train_loader is None:
            self._train_loader = get_hospital_dataloader(
                self.data_root, self.partition_root, self.hospital_id,
                batch_size=self.batch_size, train=True,
            )
        return self._train_loader

    def get_parameters(self, config: Dict[str, Scalar]) -> List[np.ndarray]:
        return get_model_parameters(self.model)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, Scalar]
    ) -> Tuple[List[np.ndarray], int, Dict[str, Scalar]]:
        set_model_parameters(self.model, parameters)

        state = str(config.get("state", SecurityState.SECURE.value))
        qber = float(config.get("qber", 0.0))

        if state == SecurityState.LOCKDOWN.value:
            # Defense in depth: even if the caller forgot to honor
            # aggregate_fit's LOCKDOWN short-circuit, this client
            # refuses to train or report a usable update.
            log.info("[hospital %s] LOCKDOWN this round (qber=%.4f) — skipping local training.",
                      self.cid, qber)
            return parameters, 0, {"state": state, "qber": qber, "train_loss": float("nan")}

        fedprox_mu = float(config.get("fedprox_mu", 0.0))
        metrics = _train_one_client(
            self.model, self._get_train_loader(),
            device=self.device, epochs=self.local_epochs, lr=self.lr,
            fedprox_mu=fedprox_mu,
        )
        metrics.update({"state": state, "qber": qber, "fedprox_mu": fedprox_mu})

        n_examples = len(self._get_train_loader().dataset)
        return get_model_parameters(self.model), n_examples, metrics

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        set_model_parameters(self.model, parameters)
        # Cheap proxy: evaluate on this hospital's own training split
        # (a proper held-out per-client split is a straightforward
        # extension of partition_and_save if you want it later).
        # The dashboard's real accuracy number should come from the
        # server-side evaluate_fn against the shared test set — see
        # evaluation.py — since this is a non-IID, small local sample.
        loader = self._get_train_loader()
        loss, n_examples = _evaluate_one_client(self.model, loader, device=self.device)
        return loss, n_examples, {}


def create_client_fn(
    data_root: str | Path,
    partition_root: str | Path,
    *,
    batch_size: int = 32,
    local_epochs: int = DEFAULT_LOCAL_EPOCHS,
    lr: float = DEFAULT_LR,
) -> Callable[[str], Client]:
    """
    Factory matching what `flwr.simulation.start_simulation(client_fn=...)`
    would have expected — and what `runner.build_local_client_proxies()`
    now uses to build in-process proxies instead.
    """
    data_root = Path(data_root)
    partition_root = Path(partition_root)

    def client_fn(cid: str) -> Client:
        return EveFLClient(
            cid, data_root, partition_root,
            batch_size=batch_size, local_epochs=local_epochs, lr=lr,
        ).to_client()

    return client_fn
