"""
EveFL Flower client.

Responsibilities
----------------

Each federated client:

    1. Loads its own hospital partition
    2. Receives the current global model
    3. Receives security configuration from the server
    4. Performs local training
    5. Applies FedProx when the server enters CAUTION
    6. Returns the locally trained parameters to Flower

Security state is communicated as a string:

    SECURE
    CAUTION
    LOCKDOWN

The client does NOT decide the security state.

The server/strategy decides it.

The client only follows the policy received from the server.
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

from evefl.fl.dataset import (
    DEFAULT_BATCH_SIZE,
    ClientData,
    load_client_partition,
    make_dataloader,
)
from evefl.fl.model import (
    build_resnet18,
    get_device,
)


log = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

DEFAULT_LOCAL_EPOCHS = 5

DEFAULT_PROXIMAL_MU = 0.01

NUM_CLASSES = 14


# ============================================================================
# Parameter conversion
# ============================================================================

def get_model_parameters(
    model: nn.Module,
) -> List[np.ndarray]:
    """
    Extract model parameters as NumPy arrays.

    Flower communicates model parameters rather than serializing the entire
    PyTorch model.
    """

    return [
        value.detach()
        .cpu()
        .numpy()
        .copy()
        for _, value
        in model.state_dict().items()
    ]


def set_model_parameters(
    model: nn.Module,
    parameters: List[np.ndarray],
) -> None:
    """
    Load Flower parameters into a PyTorch model.
    """

    state_dict = model.state_dict()

    if len(parameters) != len(state_dict):
        raise ValueError(
            "Parameter count mismatch.\n"
            f"Model expects {len(state_dict)} tensors, "
            f"but Flower supplied {len(parameters)}."
        )

    converted = {}

    for (
        (name, reference),
        parameter,
    ) in zip(
        state_dict.items(),
        parameters,
    ):

        array = np.asarray(
            parameter
        )

        if tuple(
            array.shape
        ) != tuple(
            reference.shape
        ):
            raise ValueError(
                f"Shape mismatch for '{name}': "
                f"received {array.shape}, "
                f"expected {tuple(reference.shape)}."
            )

        converted[name] = torch.tensor(
            array,
            dtype=reference.dtype,
        )

    model.load_state_dict(
        converted,
        strict=True,
    )


# ============================================================================
# FedProx
# ============================================================================

def calculate_fedprox_term(
    model: nn.Module,
    global_parameters: List[np.ndarray],
) -> torch.Tensor:
    """
    Calculate the FedProx proximal term.

        μ / 2 * ||w - w_global||²

    The μ multiplication is intentionally performed by the caller.

    This function only calculates:

        ||w - w_global||²

    so that the training loop remains explicit.
    """

    device = next(
        model.parameters()
    ).device

    total = torch.zeros(
        1,
        device=device,
    )

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    if len(trainable_parameters) != len(
        global_parameters
    ):
        raise ValueError(
            "Number of trainable model parameters does not match "
            "the global parameter list."
        )

    for parameter, global_array in zip(
        trainable_parameters,
        global_parameters,
    ):

        global_tensor = torch.tensor(
            global_array,
            dtype=parameter.dtype,
            device=device,
        )

        total = total + torch.sum(
            (parameter - global_tensor) ** 2
        )

    return total


# ============================================================================
# Client
# ============================================================================

class EveFLClient(
    fl.client.NumPyClient
):
    """
    Flower client for EveFL.

    The same class is used for all hospital clients.

    Client-specific information is provided through client_id and its
    corresponding partition.
    """

    def __init__(
        self,
        *,
        client_id: int,
        train_loader,
        validation_loader=None,
        device: Optional[torch.device] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        self.client_id = int(
            client_id
        )

        self.train_loader = (
            train_loader
        )

        self.validation_loader = (
            validation_loader
        )

        self.device = (
            device
            if device is not None
            else get_device()
        )

        self.batch_size = int(
            batch_size
        )

        self.model = build_resnet18(
            pretrained=False
        )

        self.model.to(
            self.device
        )

        # Multi-label ChestX-ray14 classification.
        #
        # Each image can have multiple simultaneous pathology labels.
        self.criterion = (
            nn.BCEWithLogitsLoss()
        )

        log.info(
            "Created EveFL client %d on %s",
            self.client_id,
            self.device,
        )

    # ------------------------------------------------------------------
    # Flower get_parameters
    # ------------------------------------------------------------------

    def get_parameters(
        self,
        config: Dict[str, fl.common.Scalar],
    ) -> List[np.ndarray]:
        """
        Return current local model parameters.
        """

        del config

        return get_model_parameters(
            self.model
        )

    # ------------------------------------------------------------------
    # Flower fit
    # ------------------------------------------------------------------

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, fl.common.Scalar],
    ) -> Tuple[
        List[np.ndarray],
        int,
        Dict[str, fl.common.Scalar],
    ]:
        """
        Perform local training.

        Expected server configuration:

            state
            qber
            q_max
            local_epochs
            proximal_mu
        """

        # --------------------------------------------------------------
        # Load global model
        # --------------------------------------------------------------

        set_model_parameters(
            self.model,
            parameters,
        )

        # --------------------------------------------------------------
        # Read server policy
        # --------------------------------------------------------------

        state = str(
            config.get(
                "state",
                "SECURE",
            )
        ).upper()

        local_epochs = int(
            config.get(
                "local_epochs",
                DEFAULT_LOCAL_EPOCHS,
            )
        )

        proximal_mu = float(
            config.get(
                "proximal_mu",
                0.0,
            )
        )

        qber = float(
            config.get(
                "qber",
                0.0,
            )
        )

        q_max = float(
            config.get(
                "q_max",
                qber,
            )
        )

        server_round = int(
            config.get(
                "server_round",
                config.get(
                    "round_id",
                    0,
                ),
            )
        )

        # --------------------------------------------------------------
        # Validate state
        # --------------------------------------------------------------

        if state not in {
            "SECURE",
            "CAUTION",
            "LOCKDOWN",
        }:
            raise ValueError(
                f"Unknown security state received from server: {state}"
            )

        # --------------------------------------------------------------
        # LOCKDOWN safety check
        # --------------------------------------------------------------

        # Normally the server should never call fit() for a lockdown round.
        #
        # But this guard prevents accidental local training if a future
        # Flower implementation sends a FitIns anyway.
        if state == "LOCKDOWN":

            log.warning(
                "[Client %d | Round %d] "
                "LOCKDOWN received. "
                "Skipping local training.",
                self.client_id,
                server_round,
            )

            return (
                get_model_parameters(
                    self.model
                ),
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

        # --------------------------------------------------------------
        # FedProx activation
        # --------------------------------------------------------------

        use_fedprox = (
            state == "CAUTION"
            and proximal_mu > 0.0
        )

        global_parameters = None

        if use_fedprox:

            global_parameters = [
                np.array(
                    parameter,
                    copy=True,
                )
                for parameter in parameters
            ]

            log.info(
                "[Client %d | Round %d] "
                "CAUTION: FedProx enabled, mu=%.6f",
                self.client_id,
                server_round,
                proximal_mu,
            )

        else:

            log.info(
                "[Client %d | Round %d] "
                "%s: standard local objective",
                self.client_id,
                server_round,
                state,
            )

        # --------------------------------------------------------------
        # Optimizer
        # --------------------------------------------------------------

        optimizer = Adam(
            self.model.parameters(),
            lr=1e-3,
        )

        self.model.train()

        total_loss = 0.0

        total_examples = 0

        total_batches = 0

        # --------------------------------------------------------------
        # Local epochs
        # --------------------------------------------------------------

        for epoch in range(
            local_epochs
        ):

            epoch_loss = 0.0

            epoch_examples = 0

            for (
                images,
                targets,
            ) in self.train_loader:

                images = images.to(
                    self.device,
                    non_blocking=True,
                )

                targets = targets.to(
                    self.device,
                    non_blocking=True,
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                # --------------------------------------------------
                # Forward pass
                # --------------------------------------------------

                logits = self.model(
                    images
                )

                if logits.ndim != 2:

                    raise RuntimeError(
                        "Model output must have shape "
                        "[batch_size, num_classes]. "
                        f"Received {tuple(logits.shape)}."
                    )

                if logits.shape[1] != NUM_CLASSES:

                    raise RuntimeError(
                        "Model must output 14 pathology logits. "
                        f"Received {logits.shape[1]}."
                    )

                # --------------------------------------------------
                # Standard multi-label loss
                # --------------------------------------------------

                classification_loss = (
                    self.criterion(
                        logits,
                        targets,
                    )
                )

                loss = classification_loss

                # --------------------------------------------------
                # FedProx
                # --------------------------------------------------

                proximal_loss_value = 0.0

                if use_fedprox:

                    if global_parameters is None:
                        raise RuntimeError(
                            "FedProx was enabled but global parameters "
                            "were not stored."
                        )

                    proximal_distance = (
                        calculate_fedprox_term(
                            self.model,
                            global_parameters,
                        )
                    )

                    proximal_loss = (
                        proximal_mu
                        / 2.0
                    ) * proximal_distance

                    loss = (
                        classification_loss
                        + proximal_loss
                    )

                    proximal_loss_value = (
                        float(
                            proximal_loss.detach()
                            .cpu()
                            .item()
                        )
                    )

                # --------------------------------------------------
                # Backpropagation
                # --------------------------------------------------

                loss.backward()

                optimizer.step()

                batch_size = int(
                    images.shape[0]
                )

                batch_loss = float(
                    loss.detach()
                    .cpu()
                    .item()
                )

                epoch_loss += (
                    batch_loss
                    * batch_size
                )

                epoch_examples += (
                    batch_size
                )

                total_batches += 1

                log.debug(
                    "[Client %d | Round %d | Epoch %d] "
                    "batch_loss=%.6f "
                    "classification_loss=%.6f "
                    "proximal_loss=%.6f",
                    self.client_id,
                    server_round,
                    epoch + 1,
                    batch_loss,
                    float(
                        classification_loss
                        .detach()
                        .cpu()
                        .item()
                    ),
                    proximal_loss_value,
                )

            total_loss += epoch_loss

            total_examples += epoch_examples

            mean_epoch_loss = (
                epoch_loss
                / max(
                    epoch_examples,
                    1,
                )
            )

            log.info(
                "[Client %d | Round %d] "
                "Epoch %d/%d loss=%.6f",
                self.client_id,
                server_round,
                epoch + 1,
                local_epochs,
                mean_epoch_loss,
            )

        # --------------------------------------------------------------
        # Final parameters
        # --------------------------------------------------------------

        updated_parameters = (
            get_model_parameters(
                self.model
            )
        )

        mean_loss = (
            total_loss
            / max(
                total_examples,
                1,
            )
        )

        log.info(
            "[Client %d | Round %d] "
            "Training complete: examples=%d "
            "batches=%d loss=%.6f",
            self.client_id,
            server_round,
            total_examples,
            total_batches,
            mean_loss,
        )

        # --------------------------------------------------------------
        # Return FitRes-compatible values
        # --------------------------------------------------------------

        metrics: Dict[
            str,
            fl.common.Scalar,
        ] = {
            "client_id": self.client_id,
            "round": server_round,
            "state": state,
            "qber": qber,
            "q_max": q_max,
            "loss": mean_loss,
            "local_epochs": local_epochs,
            "proximal_mu": (
                proximal_mu
                if use_fedprox
                else 0.0
            ),
            "fedprox_active": (
                1
                if use_fedprox
                else 0
            ),
            "num_batches": total_batches,
        }

        return (
            updated_parameters,
            total_examples,
            metrics,
        )

    # ------------------------------------------------------------------
    # Flower evaluate
    # ------------------------------------------------------------------

    def evaluate(
        self,
        parameters: List[np.ndarray],
    ):
        """
        Local evaluation.

        The first Phase-4 implementation does not use client-side evaluation
        because the intended final evaluation is a global held-out test set.

        Returning NaN keeps the method valid without pretending that local
        evaluation is the paper's final metric.
        """

        set_model_parameters(
            self.model,
            parameters,
        )

        return (
            float("nan"),
            0,
            {
                "evaluation_available": 0,
            },
        )


# ============================================================================
# Client data loading
# ============================================================================

def load_client_data(
    client_id: int,
    *,
    data_root: Path,
    partition_root: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = 0,
) -> ClientData:
    """
    Load one client's partition.

    If client partition files already exist, use them.

    Expected:

        partition_root/
            client_0.json
            client_1.json
            client_2.json
    """

    client_id = int(
        client_id
    )

    partition_root = Path(
        partition_root
    )

    data_root = Path(
        data_root
    )

    partition_file = (
        partition_root
        / f"client_{client_id}.json"
    )

    if not partition_file.exists():

        raise FileNotFoundError(
            f"Client {client_id} partition was not found:\n"
            f"{partition_file}\n\n"
            "Run the dataset partitioning step before starting "
            "the Flower simulation."
        )

    train_records = (
        load_client_partition(
            client_id,
            partition_root,
        )
    )

    if not train_records:

        raise RuntimeError(
            f"Client {client_id} has no training records."
        )

    # --------------------------------------------------------------
    # Validation
    # --------------------------------------------------------------

    # The first Phase-4 smoke test uses the client's training records for
    # constructing a minimal validation loader only when a separate
    # validation manifest has not yet been established.
    #
    # This is NOT the final paper evaluation protocol.
    validation_records = list(
        train_records
    )

    train_loader = make_dataloader(
        train_records,
        batch_size=batch_size,
        shuffle=True,
        train=True,
        num_workers=num_workers,
    )

    validation_loader = make_dataloader(
        validation_records,
        batch_size=batch_size,
        shuffle=False,
        train=False,
        num_workers=num_workers,
    )

    # data_root is intentionally accepted here because the function is the
    # boundary where dataset configuration enters the client. The partition
    # JSON already contains resolved image paths.
    del data_root

    return ClientData(
        client_id=client_id,
        train_records=train_records,
        validation_records=validation_records,
        train_loader=train_loader,
        validation_loader=validation_loader,
    )


# ============================================================================
# Flower client factory
# ============================================================================

def create_client_fn(
    *,
    data_root: Path,
    partition_root: Path,
    pretrained: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = 0,
):
    """
    Create the Flower client factory used by server.py.

    Flower supplies:

        cid

    and this factory maps cid -> hospital/client partition.
    """

    # pretrained is accepted for API compatibility with the model creation
    # layer. Phase 4 should begin from the same deterministic architecture
    # across all clients.
    del pretrained

    def client_fn(
        cid: str,
    ):
        """
        Construct one EveFLClient.
        """

        try:
            client_id = int(
                cid
            )

        except ValueError as exc:

            raise ValueError(
                "Flower client IDs must be integer-compatible. "
                f"Received cid={cid!r}"
            ) from exc

        log.info(
            "Initializing Flower client %d",
            client_id,
        )

        client_data = load_client_data(
            client_id=client_id,
            data_root=Path(
                data_root
            ),
            partition_root=Path(
                partition_root
            ),
            batch_size=batch_size,
            num_workers=num_workers,
        )

        client = EveFLClient(
            client_id=client_id,
            train_loader=(
                client_data.train_loader
            ),
            validation_loader=(
                client_data.validation_loader
            ),
        )

        return client.to_client()

    return client_fn


# ============================================================================
# Standalone client factory for tests
# ============================================================================

def build_test_client(
    *,
    client_id: int,
    train_loader,
    validation_loader=None,
) -> EveFLClient:
    """
    Build a client directly for unit tests.

    This avoids requiring Flower's simulation runtime.
    """

    return EveFLClient(
        client_id=client_id,
        train_loader=train_loader,
        validation_loader=validation_loader,
    )