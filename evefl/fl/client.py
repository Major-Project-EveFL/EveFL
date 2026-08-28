"""
EveFL Flower client.

Each federated client:
    1. Loads its own hospital partition via get_hospital_dataloader()
    2. Receives the current global model parameters from the server
    3. Receives security configuration (state, proximal_mu) from the strategy
    4. Performs local training with BCEWithLogitsLoss
    5. Applies FedProx proximal term when server state is CAUTION
    6. Returns updated parameters to Flower

The client does NOT decide the security state.
The server/strategy decides it.
The client only follows the policy received via FitIns config.

Flower version target: flwr==1.11.1
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from evefl.fl.dataset import get_hospital_dataloader
from evefl.fl.model import build_resnet18, get_device, NUM_CLASSES

log = logging.getLogger(__name__)

DEFAULT_LOCAL_EPOCHS = 5
DEFAULT_BATCH_SIZE = 32


# ============================================================================
# Parameter conversion
# ============================================================================

def get_model_parameters(model: nn.Module) -> List[np.ndarray]:
    """
    Extract model parameters as NumPy arrays.

    Flower communicates model parameters rather than serializing the entire
    PyTorch model. Order matches model.state_dict().
    """
    return [
        value.detach().cpu().numpy().copy()
        for _, value in model.state_dict().items()
    ]


def set_model_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    """
    Load Flower parameters into a PyTorch model.
    """
    state_dict = model.state_dict()

    if len(parameters) != len(state_dict):
        raise ValueError(
            f"Parameter count mismatch. Model expects {len(state_dict)} tensors, "
            f"but Flower supplied {len(parameters)}."
        )

    converted = {}
    for (name, reference), parameter in zip(state_dict.items(), parameters):
        array = np.asarray(parameter)
        if tuple(array.shape) != tuple(reference.shape):
            raise ValueError(
                f"Shape mismatch for '{name}': received {array.shape}, "
                f"expected {tuple(reference.shape)}."
            )
        converted[name] = torch.tensor(array, dtype=reference.dtype)

    model.load_state_dict(converted, strict=True)


# ============================================================================
# FedProx
# ============================================================================

def calculate_fedprox_term(model: nn.Module, global_parameters: List[np.ndarray]) -> torch.Tensor:
    """
    Calculate the FedProx proximal term: ||w - w_global||^2.

    The mu multiplication is performed by the caller in the training loop.
    """
    device = next(model.parameters()).device
    total = torch.zeros(1, device=device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if len(trainable_params) != len(global_parameters):
        raise ValueError(
            f"Trainable param count ({len(trainable_params)}) does not match "
            f"global parameter list ({len(global_parameters)})."
        )

    for param, global_array in zip(trainable_params, global_parameters):
        global_tensor = torch.tensor(global_array, dtype=param.dtype, device=device)
        total = total + torch.sum((param - global_tensor) ** 2)

    return total


# ============================================================================
# Flower client
# ============================================================================

class EveFLClient(fl.client.NumPyClient):
    """
    Flower NumPyClient for EveFL.

    Same class used for all hospital clients.
    Client-specific data is provided through the train_loader.
    """

    def __init__(
        self,
        *,
        client_id: int,
        train_loader,
        device: Optional[torch.device] = None,
    ):
        self.client_id = int(client_id)
        self.train_loader = train_loader
        self.device = device if device is not None else get_device()

        self.model = build_resnet18(pretrained=False)
        self.model.to(self.device)

        # Multi-label classification: 14 pathology labels
        self.criterion = nn.BCEWithLogitsLoss()

        log.info("Created EveFL client %d on %s", self.client_id, self.device)

    def get_parameters(self, config: Dict[str, fl.common.Scalar]) -> List[np.ndarray]:
        """Return current local model parameters."""
        return get_model_parameters(self.model)

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, fl.common.Scalar],
    ) -> Tuple[List[np.ndarray], int, Dict[str, fl.common.Scalar]]:
        """
        Perform local training.

        Expected config keys from strategy:
            state          : "SECURE" | "CAUTION" | "LOCKDOWN"
            proximal_mu    : float (0.0 for SECURE, >0 for CAUTION)
            local_epochs   : int
            server_round   : int
            client_id      : str
            qber           : float
            q_max          : float
        """
        # -------------------------------------------------------------
        # Load global model
        # -------------------------------------------------------------
        set_model_parameters(self.model, parameters)

        # -------------------------------------------------------------
        # Read server policy
        # -------------------------------------------------------------
        state = str(config.get("state", "SECURE")).upper()
        local_epochs = int(config.get("local_epochs", DEFAULT_LOCAL_EPOCHS))
        proximal_mu = float(config.get("proximal_mu", 0.0))
        qber = float(config.get("qber", 0.0))
        q_max = float(config.get("q_max", qber))
        server_round = int(config.get("server_round", config.get("round_id", 0)))

        if state not in {"SECURE", "CAUTION", "LOCKDOWN"}:
            raise ValueError(f"Unknown security state from server: {state}")

        # -------------------------------------------------------------
        # LOCKDOWN guard
        # -------------------------------------------------------------
        if state == "LOCKDOWN":
            log.warning(
                "[Client %d | Round %d] LOCKDOWN received. Skipping local training.",
                self.client_id, server_round,
            )
            return (
                get_model_parameters(self.model),
                0,
                {
                    "state": "LOCKDOWN",
                    "qber": qber,
                    "q_max": q_max,
                    "proximal_mu": 0.0,
                    "local_epochs": 0,
                    "training_skipped": 1,
                },
            )

        # -------------------------------------------------------------
        # FedProx activation
        # -------------------------------------------------------------
        use_fedprox = (state == "CAUTION" and proximal_mu > 0.0)
        global_parameters = None

        if use_fedprox:
            global_parameters = [np.array(p, copy=True) for p in parameters]
            log.info(
                "[Client %d | Round %d] CAUTION: FedProx enabled, mu=%.6f",
                self.client_id, server_round, proximal_mu,
            )
        else:
            log.info(
                "[Client %d | Round %d] %s: standard local objective",
                self.client_id, server_round, state,
            )

        # -------------------------------------------------------------
        # Optimizer (smoke-test: Adam; paper may use AdamW + cosine)
        # -------------------------------------------------------------
        optimizer = Adam(self.model.parameters(), lr=1e-3)
        self.model.train()

        total_loss = 0.0
        total_examples = 0
        total_batches = 0

        # -------------------------------------------------------------
        # Local epochs
        # -------------------------------------------------------------
        for epoch in range(local_epochs):
            epoch_loss = 0.0
            epoch_examples = 0

            for images, targets in self.train_loader:
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                logits = self.model(images)

                if logits.ndim != 2 or logits.shape[1] != NUM_CLASSES:
                    raise RuntimeError(
                        f"Model output must be [batch_size, {NUM_CLASSES}]. "
                        f"Received {tuple(logits.shape)}."
                    )

                classification_loss = self.criterion(logits, targets)
                loss = classification_loss

                proximal_loss_value = 0.0
                if use_fedprox:
                    if global_parameters is None:
                        raise RuntimeError("FedProx enabled but global parameters missing.")

                    proximal_distance = calculate_fedprox_term(self.model, global_parameters)
                    proximal_loss = (proximal_mu / 2.0) * proximal_distance
                    loss = classification_loss + proximal_loss
                    proximal_loss_value = float(proximal_loss.detach().cpu().item())

                loss.backward()
                optimizer.step()

                batch_size = int(images.shape[0])
                batch_loss = float(loss.detach().cpu().item())
                epoch_loss += batch_loss * batch_size
                epoch_examples += batch_size
                total_batches += 1

                log.debug(
                    "[Client %d | Round %d | Epoch %d] batch_loss=%.6f cls_loss=%.6f prox_loss=%.6f",
                    self.client_id, server_round, epoch + 1,
                    batch_loss,
                    float(classification_loss.detach().cpu().item()),
                    proximal_loss_value,
                )

            total_loss += epoch_loss
            total_examples += epoch_examples

            mean_epoch_loss = epoch_loss / max(epoch_examples, 1)
            log.info(
                "[Client %d | Round %d] Epoch %d/%d loss=%.6f",
                self.client_id, server_round, epoch + 1, local_epochs, mean_epoch_loss,
            )

        updated_parameters = get_model_parameters(self.model)
        mean_loss = total_loss / max(total_examples, 1)

        log.info(
            "[Client %d | Round %d] Training complete: examples=%d batches=%d loss=%.6f",
            self.client_id, server_round, total_examples, total_batches, mean_loss,
        )

        metrics: Dict[str, fl.common.Scalar] = {
            "client_id": self.client_id,
            "round": server_round,
            "state": state,
            "qber": qber,
            "q_max": q_max,
            "loss": mean_loss,
            "local_epochs": local_epochs,
            "proximal_mu": proximal_mu if use_fedprox else 0.0,
            "fedprox_active": 1 if use_fedprox else 0,
            "num_batches": total_batches,
        }

        return updated_parameters, total_examples, metrics

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, fl.common.Scalar],
    ) -> Tuple[float, int, Dict[str, fl.common.Scalar]]:
        """
        Local evaluation placeholder.

        The intended final evaluation uses a global held-out test set,
        not per-client evaluation. Returning NaN keeps the interface valid.
        """
        set_model_parameters(self.model, parameters)
        return float("nan"), 0, {"evaluation_available": 0}


# ============================================================================
# Flower client factory
# ============================================================================

def create_client_fn(
    *,
    data_root: Path,
    partition_root: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
):
    """
    Create the Flower client factory used by server.py / simulation.

    Flower supplies cid (str) and this factory maps cid -> hospital partition.
    """
    def client_fn(cid: str) -> fl.client.Client:
        try:
            client_id = int(cid)
        except ValueError as exc:
            raise ValueError(
                f"Flower client IDs must be integer-compatible. Received cid={cid!r}"
            ) from exc

        log.info("Initializing Flower client %d", client_id)

        train_loader = get_hospital_dataloader(
            data_root=data_root,
            partition_root=partition_root,
            hospital_id=client_id,
            batch_size=batch_size,
            train=True,
        )

        client = EveFLClient(
            client_id=client_id,
            train_loader=train_loader,
        )

        return client.to_client()

    return client_fn


# ============================================================================
# Standalone client builder for unit tests
# ============================================================================

def build_test_client(
    *,
    client_id: int,
    train_loader,
) -> EveFLClient:
    """
    Build a client directly for unit tests without Flower simulation.
    """
    return EveFLClient(
        client_id=client_id,
        train_loader=train_loader,
    )