"""
<<<<<<< HEAD
EveFL QBER-aware Flower strategy.

Phase 4 server-side orchestration:

    per-client BB84
            |
            v
        q_i(t)
            |
            v
       q_max(t)
            |
            v
    StateController
            |
     +------+------+ 
     |      |      |
  SECURE CAUTION LOCKDOWN
     |      |      |
   FedAvg FedProx  reject
          + anomaly
          weighting

This module intentionally does NOT perform local PyTorch training.

Local training belongs in:
    evefl/fl/client.py

QBER generation belongs in:
    evefl/quantum/bb84.py

State classification belongs in:
    evefl/orchestration/state_machine.py

This is the server-side brain of Phase 4. It will:
run BB84 for each client each round
calculate q_max
call your existing StateController
send the resulting policy to clients
perform normal FedAvg in SECURE
perform anomaly-aware downweighting in CAUTION
reject updates in LOCKDOWN
preserve the previous global model during lockdown
record round-level security information for the later JSON log
=======
EveFL Flower Strategy — Phase 4 core.

Wires together:
  - BB84 QKD engine  (evefl.quantum.bb84)
  - Three-state controller  (evefl.orchestration.state_machine)
  - Classical cipher suite  (evefl.crypto.classical)

Per round:
  1. Run BB84 for every participating client → per-client QBER
  2. Take max QBER → StateController → SECURE / CAUTION / LOCKDOWN
  3. configure_fit() injects state config into FitIns (mu, round_id, state)
  4. aggregate_fit() dispatches to the correct aggregation rule
     SECURE   → weighted FedAvg
     CAUTION  → weighted FedAvg with gradient-norm anomaly scoring
                (clients received mu > 0, so they already trained with proximal term)
     LOCKDOWN → discard all updates, return current parameters unchanged
>>>>>>> c8d000e56cdc54490d8e8750eb80577f1165bdf3
"""

from __future__ import annotations

<<<<<<< HEAD
import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import flwr as fl
import numpy as np

from evefl.quantum.bb84 import BB84Protocol
from evefl.orchestration.state_machine import StateController
=======
import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import flwr as fl
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from evefl.orchestration.state_machine import (
    SecurityState,
    StateController,
    StateThresholds,
)
from evefl.quantum.bb84 import BB84Protocol
from evefl.quantum.base import QKDResult
>>>>>>> c8d000e56cdc54490d8e8750eb80577f1165bdf3

log = logging.getLogger(__name__)


<<<<<<< HEAD
# ---------------------------------------------------------------------------
# EveFL configuration
# ---------------------------------------------------------------------------

DEFAULT_N_QUBITS = 1024

DEFAULT_INTERCEPT_PROBABILITY = 0.0

DEFAULT_PROXIMAL_MU = 0.01

DEFAULT_ANOMALY_THRESHOLD = 2.0

DEFAULT_CLIENTS_PER_ROUND = 3


# ---------------------------------------------------------------------------
# Round record
# ---------------------------------------------------------------------------

@dataclass
class RoundSecurityRecord:
    """
    Stores all security information produced during one FL round.

    This becomes the source of data for results/fl_training_log.json.
    """

    round_id: int

    client_qber: Dict[str, float] = field(default_factory=dict)

    q_max: float = 0.0

    state: str = "SECURE"

    reason: str = ""

    policy: str = ""

    accepted_clients: List[str] = field(default_factory=list)

    rejected_clients: List[str] = field(default_factory=list)

    anomaly_scores: Dict[str, float] = field(default_factory=dict)

    aggregation_weights: Dict[str, float] = field(default_factory=dict)

    lockdown: bool = False

    rekey_required: bool = False


# ---------------------------------------------------------------------------
# Parameter helpers
# ---------------------------------------------------------------------------

def _parameters_to_numpy(
    parameters: fl.common.Parameters,
) -> List[np.ndarray]:
    """
    Convert Flower Parameters into NumPy arrays.
    """
    return fl.common.parameters_to_ndarrays(parameters)


def _numpy_to_parameters(
    parameters: List[np.ndarray],
) -> fl.common.Parameters:
    """
    Convert NumPy arrays into Flower Parameters.
    """
    return fl.common.ndarrays_to_parameters(parameters)


def _copy_parameters(
    parameters: fl.common.Parameters,
) -> fl.common.Parameters:
    """
    Deep-copy a Flower Parameters object.

    This is important for LOCKDOWN because the previous global parameters
    must remain unchanged.
    """
    arrays = _parameters_to_numpy(parameters)

    copied = [
        np.array(array, copy=True)
        for array in arrays
    ]

    return _numpy_to_parameters(copied)


# ---------------------------------------------------------------------------
# Update distance / anomaly scoring
# ---------------------------------------------------------------------------

def _flatten_parameters(
    parameters: List[np.ndarray],
) -> np.ndarray:
    """
    Flatten a model parameter list into one vector.

    Used only for update-level anomaly scoring.
    """
    if not parameters:
        return np.array([], dtype=np.float64)

    flattened = [
        np.asarray(parameter, dtype=np.float64).reshape(-1)
        for parameter in parameters
    ]

    return np.concatenate(flattened)


def _update_distance(
    client_parameters: List[np.ndarray],
    global_parameters: List[np.ndarray],
) -> float:
    """
    Calculate normalized L2 distance between client and global parameters.

        ||w_client - w_global|| / (||w_global|| + epsilon)

    This is used as an update anomaly signal.

    It is deliberately independent of QBER.

    QBER tells us about the quantum communication channel.

    Update distance tells us about the submitted ML update.

    EveFL combines both signals through the security policy.
    """
    client_vector = _flatten_parameters(client_parameters)
    global_vector = _flatten_parameters(global_parameters)

    if client_vector.shape != global_vector.shape:
        raise ValueError(
            "Client/global parameter shape mismatch during anomaly scoring."
        )

    if client_vector.size == 0:
        return 0.0

    difference = client_vector - global_vector

    numerator = float(np.linalg.norm(difference))
    denominator = float(np.linalg.norm(global_vector))

    return numerator / (denominator + 1e-12)


def _compute_anomaly_scores(
    results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
    previous_parameters: fl.common.Parameters,
) -> Dict[str, float]:
    """
    Calculate update anomaly scores for all successful clients.

    The score is a normalized parameter-update distance.

    Higher score = larger deviation from the current global model.
    """
    global_arrays = _parameters_to_numpy(previous_parameters)

    scores: Dict[str, float] = {}

    for client_proxy, fit_res in results:
        client_id = str(client_proxy.cid)

        client_arrays = fl.common.parameters_to_ndarrays(
            fit_res.parameters
        )

        score = _update_distance(
            client_arrays,
            global_arrays,
        )

        scores[client_id] = float(score)

    return scores


# ---------------------------------------------------------------------------
# Anomaly-aware aggregation
# ---------------------------------------------------------------------------

def _safe_weight(
    anomaly_score: float,
    threshold: float,
) -> float:
    """
    Convert an anomaly score into an aggregation weight.

    score <= threshold:
        full weight

    score > threshold:
        smoothly reduced weight

    The weight never reaches zero here.

    Complete rejection is reserved for LOCKDOWN or an explicit rejection
    policy.
    """
    if anomaly_score <= threshold:
        return 1.0

    # Smooth inverse penalty.
    return float(
        threshold / max(anomaly_score, 1e-12)
    )


def _weighted_average(
    results: List[
        Tuple[
            fl.server.client_proxy.ClientProxy,
            fl.common.FitRes,
        ]
    ],
    weights: Dict[str, float],
) -> fl.common.Parameters:
    """
    Perform weighted parameter aggregation.

    Base FedAvg weight:
        number of examples

    EveFL CAUTION modifies it using the anomaly weight:

        effective_weight =
            num_examples × anomaly_weight
    """
    if not results:
        raise ValueError(
            "Cannot aggregate an empty result set."
        )

    first_parameters = fl.common.parameters_to_ndarrays(
        results[0][1].parameters
    )

    weighted_parameters = [
        np.zeros_like(array, dtype=np.float64)
        for array in first_parameters
    ]

    total_weight = 0.0

    for client_proxy, fit_res in results:
        client_id = str(client_proxy.cid)

        anomaly_weight = float(
            weights.get(client_id, 1.0)
        )

        example_weight = float(
            fit_res.num_examples
        )

        effective_weight = (
            example_weight * anomaly_weight
        )

        if effective_weight <= 0.0:
            continue

        client_parameters = fl.common.parameters_to_ndarrays(
            fit_res.parameters
        )

        if len(client_parameters) != len(weighted_parameters):
            raise ValueError(
                "Client parameter count does not match "
                "the first client parameter count."
            )

        for index, parameter in enumerate(client_parameters):
            weighted_parameters[index] += (
                parameter.astype(np.float64)
                * effective_weight
            )

        total_weight += effective_weight

    if total_weight <= 0.0:
        raise RuntimeError(
            "All client aggregation weights became zero."
        )

    aggregated = [
        parameter / total_weight
        for parameter in weighted_parameters
    ]

    # Restore the original parameter dtypes.
    reference_parameters = fl.common.parameters_to_ndarrays(
        results[0][1].parameters
    )

    aggregated = [
        parameter.astype(reference.dtype)
        for parameter, reference in zip(
            aggregated,
            reference_parameters,
        )
    ]

    return fl.common.ndarrays_to_parameters(
        aggregated
    )


# ---------------------------------------------------------------------------
# EveFL strategy
# ---------------------------------------------------------------------------

class EveFLStrategy(fl.server.strategy.FedAvg):
    """
    EveFL security-aware FedAvg strategy.

    Security states:

        SECURE
            Normal FedAvg.

        CAUTION
            Clients receive FedProx configuration.
            Submitted updates are additionally anomaly-weighted.

        LOCKDOWN
            No client updates are accepted.
            The previous global parameters are retained.

    QBER:

        Each selected client gets an independent BB84 simulation.

        q_i(t) = QBER observed for client i in round t

        q_max(t) = max_i q_i(t)

    The q_max value is passed into StateController.
=======
# Weight floor for flagged suspicious clients in CAUTION state
_SUSPICIOUS_WEIGHT_FLOOR = 0.05
# Anomaly threshold: flag if L2 norm > mean + k * std
_ANOMALY_K = 2.0
# FedProx proximal coefficient sent to clients in CAUTION state
_MU_CAUTION = 0.01
_MU_SECURE  = 0.0


class EveFlStrategy(fl.server.strategy.Strategy):
    """
    QBER-aware Flower strategy.

    Args:
        initial_parameters:  Starting model parameters (required by Flower).
        eve_intercept_rate:  Alpha ∈ [0,1] — Eve's per-photon intercept probability.
                             0.0 = no Eve, 1.0 = full intercept-resend.
        n_qubits:            Number of BB84 photons per client per round.
        thresholds:          QBER thresholds for state transitions (default: 5%/11%).
        fraction_fit:        Fraction of clients sampled per round (1.0 = all).
        min_fit_clients:     Minimum clients required to start a round.
        min_available_clients: Minimum clients that must be connected.
        evaluate_fn:         Optional server-side evaluation function.
>>>>>>> c8d000e56cdc54490d8e8750eb80577f1165bdf3
    """

    def __init__(
        self,
<<<<<<< HEAD
        *,
        num_clients: int = DEFAULT_CLIENTS_PER_ROUND,
        n_qubits: int = DEFAULT_N_QUBITS,
        intercept_probability: float = DEFAULT_INTERCEPT_PROBABILITY,
        proximal_mu: float = DEFAULT_PROXIMAL_MU,
        anomaly_threshold: float = DEFAULT_ANOMALY_THRESHOLD,
        local_epochs: int = 5,
        seed: int = 42,
        **kwargs: Any,
    ):
        """
        Initialize the EveFL strategy.

        Parameters:
            num_clients:
                Number of clients expected in the experiment.

            n_qubits:
                Number of BB84 qubits simulated per client per round.

            intercept_probability:
                Eve intercept-resend probability.

                0.0 = no attack
                1.0 = full intercept-resend attack

                This can later be varied for experiments.

            proximal_mu:
                FedProx coefficient used in CAUTION.

            anomaly_threshold:
                Threshold for update-distance anomaly weighting.

            local_epochs:
                Number of local training epochs.

            seed:
                Base reproducibility seed.
        """

        self.num_clients = int(num_clients)

        self.n_qubits = int(n_qubits)

        self.intercept_probability = float(
            intercept_probability
        )

        self.proximal_mu = float(proximal_mu)

        self.anomaly_threshold = float(
            anomaly_threshold
        )

        self.local_epochs = int(local_epochs)

        self.seed = int(seed)

        # The StateController is deliberately kept separate from Flower.
        self.controller = StateController()

        # Previous global parameters.
        #
        # Required for LOCKDOWN because the strategy needs to return the
        # previous model instead of accepting the compromised round.
        self.previous_parameters: Optional[
            fl.common.Parameters
        ] = None

        # Current security state.
        self.current_state = "SECURE"

        # Current round record.
        self.round_records: Dict[
            int,
            RoundSecurityRecord,
        ] = {}

        # Signals whether the previous round caused lockdown.
        self.lockdown_active = False

        # Signals that a fresh QKD exchange should be performed.
        self.rekey_required = False

        super().__init__(
            **kwargs,
        )

    # ------------------------------------------------------------------
    # QBER
    # ------------------------------------------------------------------

    def _simulate_client_qber(
        self,
        client_id: int,
        server_round: int,
    ) -> float:
        """
        Run an independent BB84 simulation for one client.

        A fresh deterministic seed is generated from:

            base seed
            + client ID
            + server round

        This prevents every client from receiving the exact same random
        QBER while keeping the experiment reproducible.
        """
        simulation_seed = (
            self.seed
            + server_round * 10_000
            + client_id
        )

        protocol = BB84Protocol(
            sample_fraction=1.0,
            seed=simulation_seed,
        )

        result = protocol.run_exchange(
            n_qubits=self.n_qubits,
            intercept_probability=self.intercept_probability,
        )

        # The Phase-1 implementation may return a float or a structured
        # result. The normal EveFL contract expects the QBER value.
        if isinstance(result, (float, int, np.floating)):
            qber = float(result)

        elif isinstance(result, dict):
            if "qber" not in result:
                raise ValueError(
                    "BB84 run_exchange() returned a dictionary without "
                    "a 'qber' field."
                )

            qber = float(result["qber"])

        else:
            raise TypeError(
                "Unsupported BB84 run_exchange() return type: "
                f"{type(result).__name__}"
            )

        if not 0.0 <= qber <= 1.0:
            raise ValueError(
                f"BB84 returned invalid QBER={qber}."
            )

        return qber

    def _get_round_qber(
        self,
        server_round: int,
        client_ids: List[str],
    ) -> Dict[str, float]:
        """
        Generate QBER for every selected client.
        """
        qber_values: Dict[str, float] = {}

        for cid in client_ids:
            client_id = int(cid)

            qber = self._simulate_client_qber(
                client_id=client_id,
                server_round=server_round,
            )

            qber_values[cid] = qber

            log.info(
                "[Round %d] Client %s QBER = %.4f",
                server_round,
                cid,
                qber,
            )

        return qber_values

    # ------------------------------------------------------------------
    # State controller
    # ------------------------------------------------------------------

    def _classify_security_state(
        self,
        q_max: float,
    ) -> Tuple[str, str]:
        """
        Pass q_max through the existing Phase-3 StateController.

        The method handles the two common controller styles:

            classify(qber)
            update(qber)

        without modifying Phase 3 itself.

        Returns:
            state string,
            reason string
        """
        controller = self.controller

        # Prefer update() when available because it is the state-transition
        # API used by the EveFL controller architecture.
        if hasattr(controller, "update"):
            result = controller.update(q_max)

        elif hasattr(controller, "classify"):
            result = controller.classify(q_max)

        else:
            raise AttributeError(
                "StateController must provide either update() or classify()."
            )

        # Handle an object with .state / .reason.
        if hasattr(result, "state"):
            state = str(result.state)

            reason = str(
                getattr(
                    result,
                    "reason",
                    "",
                )
            )

            return self._normalize_state(state), reason

        # Handle a tuple:
        #
        #     (state, reason)
        #
        if isinstance(result, tuple):
            if len(result) == 0:
                raise ValueError(
                    "StateController returned an empty tuple."
                )

            state = str(result[0])

            reason = (
                str(result[1])
                if len(result) > 1
                else ""
            )

            return self._normalize_state(state), reason

        # Handle a direct enum/string state.
        state = str(result)

        return self._normalize_state(state), ""

    @staticmethod
    def _normalize_state(state: str) -> str:
        """
        Normalize enum/string representations into:

            SECURE
            CAUTION
            LOCKDOWN
        """
        normalized = state.upper()

        if "." in normalized:
            normalized = normalized.split(".")[-1]

        if normalized not in {
            "SECURE",
            "CAUTION",
            "LOCKDOWN",
        }:
            raise ValueError(
                f"Unknown EveFL security state: {state!r}"
            )

        return normalized

    # ------------------------------------------------------------------
    # configure_fit
    # ------------------------------------------------------------------
=======
        initial_parameters: Parameters,
        *,
        eve_intercept_rate: float = 0.0,
        n_qubits: int = 1024,
        thresholds: StateThresholds | None = None,
        fraction_fit: float = 1.0,
        min_fit_clients: int = 3,
        min_available_clients: int = 3,
        evaluate_fn=None,
    ):
        super().__init__()
        self._initial_parameters = initial_parameters
        self._eve_intercept_rate = eve_intercept_rate
        self._n_qubits = n_qubits
        self._fraction_fit = fraction_fit
        self._min_fit_clients = min_fit_clients
        self._min_available_clients = min_available_clients
        self._evaluate_fn = evaluate_fn

        # Phase 3: state controller
        self._controller = StateController(thresholds or StateThresholds())

        # Phase 1: one BB84 engine per logical client slot
        # Re-created each round to ensure independent key material
        self._bb84 = BB84Protocol()

        # Track current round state for aggregate_fit()
        self._round_state: SecurityState = SecurityState.SECURE
        self._round_qber_per_client: Dict[str, float] = {}

        # Preserve the last good parameters for LOCKDOWN rounds
        self._last_good_parameters: Parameters = initial_parameters

        # Round log for dashboard / W&B
        self.round_logs: List[dict] = []

    # ------------------------------------------------------------------
    # Flower required interface
    # ------------------------------------------------------------------

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        return self._initial_parameters
>>>>>>> c8d000e56cdc54490d8e8750eb80577f1165bdf3

    def configure_fit(
        self,
        server_round: int,
<<<<<<< HEAD
        parameters: fl.common.Parameters,
    ):
        """
        Configure clients for the current FL round.

        Sequence:

            1. Select clients
            2. Generate per-client QBER
            3. Calculate q_max
            4. Run StateController
            5. Build client configuration
            6. Return FitIns to selected clients
        """
        if server_round == 1:
            # Store the initial global parameters.
            self.previous_parameters = _copy_parameters(
                parameters
            )

        # --------------------------------------------------------------
        # Client selection
        # --------------------------------------------------------------

        client_manager = self._client_manager

        if client_manager is None:
            raise RuntimeError(
                "Flower client manager is not initialized."
            )

        clients = client_manager.sample(
            num_clients=self.num_clients,
            min_num_clients=self.num_clients,
        )

        client_ids = [
            str(client.cid)
            for client in clients
        ]

        # --------------------------------------------------------------
        # QBER
        # --------------------------------------------------------------

        client_qber = self._get_round_qber(
            server_round=server_round,
            client_ids=client_ids,
        )

        q_max = max(
            client_qber.values(),
            default=0.0,
        )

        # --------------------------------------------------------------
        # State transition
        # --------------------------------------------------------------

        state, reason = self._classify_security_state(
            q_max
        )

        self.current_state = state

        log.info(
            "[Round %d] q_max=%.4f state=%s reason=%s",
            server_round,
            q_max,
            state,
            reason,
        )

        # --------------------------------------------------------------
        # Determine policy
        # --------------------------------------------------------------

        if state == "SECURE":
            policy = "FEDAVG"
            proximal_mu = 0.0

        elif state == "CAUTION":
            policy = "FEDPROX_ANOMALY"
            proximal_mu = self.proximal_mu

        elif state == "LOCKDOWN":
            policy = "REJECT_AND_REKEY"
            proximal_mu = 0.0

            self.lockdown_active = True
            self.rekey_required = True

        else:
            raise AssertionError(
                f"Unhandled security state {state}"
            )

        # --------------------------------------------------------------
        # Save round record
        # --------------------------------------------------------------

        record = RoundSecurityRecord(
            round_id=server_round,
            client_qber=client_qber,
            q_max=float(q_max),
            state=state,
            reason=reason,
            policy=policy,
            lockdown=(state == "LOCKDOWN"),
            rekey_required=(state == "LOCKDOWN"),
        )

        self.round_records[server_round] = record

        # --------------------------------------------------------------
        # LOCKDOWN
        # --------------------------------------------------------------

        if state == "LOCKDOWN":
            log.warning(
                "[Round %d] LOCKDOWN — clients will not be trained.",
                server_round,
            )

            # Returning an empty instruction list means the strategy does
            # not dispatch normal FitIns for this round.
            return []

        # --------------------------------------------------------------
        # Normal/Caution client configuration
        # --------------------------------------------------------------

        fit_instructions = []

        for client in clients:
            cid = str(client.cid)

            config: Dict[str, fl.common.Scalar] = {
                "round_id": server_round,
                "server_round": server_round,
                "client_id": cid,
                "state": state,
                "qber": float(
                    client_qber[cid]
                ),
                "q_max": float(q_max),
                "local_epochs": self.local_epochs,
                "proximal_mu": float(proximal_mu),
            }

            fit_ins = fl.common.FitIns(
                parameters=parameters,
                config=config,
            )

            fit_instructions.append(
                (
                    client,
                    fit_ins,
                )
            )

        return fit_instructions

    # ------------------------------------------------------------------
    # aggregate_fit
    # ------------------------------------------------------------------
=======
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """
        Called before every FL round.

        Runs BB84 for each client, determines system state, and injects
        the state configuration into FitIns so clients know which
        training mode to use (mu=0 for SECURE, mu=0.01 for CAUTION).
        """
        clients = self._sample_clients(client_manager)

        # 1 — Run BB84 per client, collect QBER readings
        qber_readings: Dict[str, float] = {}
        sifted_keys:   Dict[str, list]  = {}

        for client in clients:
            result: QKDResult = self._bb84.run_exchange(
                n_qubits=self._n_qubits,
                eve_intercept_rate=self._eve_intercept_rate,
            )
            cid = client.cid
            qber_readings[cid] = result.qber
            sifted_keys[cid]   = result.sifted_key
            log.debug("[Round %d] Client %s QBER=%.4f", server_round, cid, result.qber)

        # 2 — System-level QBER = max across all clients
        #     Any single compromised channel triggers system-wide response
        system_qber = max(qber_readings.values()) if qber_readings else 0.0
        transition  = self._controller.update(system_qber)
        self._round_state           = transition.new_state
        self._round_qber_per_client = qber_readings

        log.info(
            "[Round %d] system_qber=%.4f  state=%s%s",
            server_round,
            system_qber,
            self._round_state.value,
            "  *** STATE CHANGE ***" if transition.changed else "",
        )

        # 3 — Build per-client FitIns with state config
        mu = _MU_CAUTION if self._round_state == SecurityState.CAUTION else _MU_SECURE

        fit_configurations = []
        for client in clients:
            config: Dict[str, Scalar] = {
                "server_round":   server_round,
                "state":          self._round_state.value,
                "proximal_mu":    mu,
                "local_epochs":   5,
                # Pass round+client_id so clients can derive the same AES key
                "round_id":       server_round,
                "client_id":      client.cid,
                "qber":           qber_readings[client.cid],
            }
            fit_configurations.append((client, FitIns(parameters, config)))

        return fit_configurations
>>>>>>> c8d000e56cdc54490d8e8750eb80577f1165bdf3

    def aggregate_fit(
        self,
        server_round: int,
<<<<<<< HEAD
        results,
        failures,
    ):
        """
        Aggregate client updates according to the current security state.

        SECURE:
            Standard FedAvg.

        CAUTION:
            Update anomaly scoring + weighted aggregation.

        LOCKDOWN:
            Keep the previous global model unchanged.

        IMPORTANT:

        The strategy never mutates the previous parameter object directly.
        This makes the lockdown behavior deterministic and testable.
        """
        del failures

        # --------------------------------------------------------------
        # LOCKDOWN
        # --------------------------------------------------------------

        if self.current_state == "LOCKDOWN":
            log.warning(
                "[Round %d] LOCKDOWN — discarding all client updates.",
                server_round,
            )

            record = self.round_records.get(
                server_round
            )

            if record is not None:
                record.accepted_clients = []
                record.rejected_clients = [
                    str(client.cid)
                    for client, _ in results
                ]

            if self.previous_parameters is None:
                raise RuntimeError(
                    "LOCKDOWN occurred before previous_parameters "
                    "was initialized."
                )

            # No client update can modify the global model.
            return (
                _copy_parameters(
                    self.previous_parameters
                ),
                {
                    "state": "LOCKDOWN",
                    "policy": "REJECT_AND_REKEY",
                    "q_max": float(
                        self.round_records[
                            server_round
                        ].q_max
                    ),
                    "accepted_clients": 0,
                    "rejected_clients": len(results),
                },
            )

        # --------------------------------------------------------------
        # No results
        # --------------------------------------------------------------

        if not results:
            log.warning(
                "[Round %d] No successful client updates.",
                server_round,
            )

            if self.previous_parameters is None:
                return None

            return (
                _copy_parameters(
                    self.previous_parameters
                ),
                {
                    "state": self.current_state,
                    "policy": "NO_UPDATE",
                    "accepted_clients": 0,
                },
            )

        # --------------------------------------------------------------
        # SECURE
        # --------------------------------------------------------------

        if self.current_state == "SECURE":
            aggregated_parameters, metrics = (
                super().aggregate_fit(
                    server_round,
                    results,
                    [],
                )
            )

            if aggregated_parameters is None:
                return None

            self.previous_parameters = _copy_parameters(
                aggregated_parameters
            )

            record = self.round_records.get(
                server_round
            )

            if record is not None:
                record.accepted_clients = [
                    str(client.cid)
                    for client, _ in results
                ]

                record.aggregation_weights = {
                    str(client.cid): 1.0
                    for client, _ in results
                }

            result_metrics = dict(
                metrics or {}
            )

            result_metrics.update(
                {
                    "state": "SECURE",
                    "policy": "FEDAVG",
                    "accepted_clients": len(results),
                }
            )

            return (
                aggregated_parameters,
                result_metrics,
            )

        # --------------------------------------------------------------
        # CAUTION
        # --------------------------------------------------------------

        if self.current_state == "CAUTION":
            anomaly_scores = _compute_anomaly_scores(
                results=results,
                previous_parameters=self.previous_parameters,
            )

            aggregation_weights = {
                client_id: _safe_weight(
                    score,
                    self.anomaly_threshold,
                )
                for client_id, score
                in anomaly_scores.items()
            }

            aggregated_parameters = _weighted_average(
                results=results,
                weights=aggregation_weights,
            )

            self.previous_parameters = _copy_parameters(
                aggregated_parameters
            )

            record = self.round_records.get(
                server_round
            )

            if record is not None:
                record.accepted_clients = [
                    str(client.cid)
                    for client, _ in results
                ]

                record.anomaly_scores = dict(
                    anomaly_scores
                )

                record.aggregation_weights = dict(
                    aggregation_weights
                )

            log.info(
                "[Round %d] CAUTION aggregation complete.",
                server_round,
            )

            for client_id, score in anomaly_scores.items():
                log.info(
                    "[Round %d] Client %s anomaly=%.6f weight=%.6f",
                    server_round,
                    client_id,
                    score,
                    aggregation_weights[client_id],
                )

            return (
                aggregated_parameters,
                {
                    "state": "CAUTION",
                    "policy": "FEDPROX_ANOMALY",
                    "accepted_clients": len(results),
                    "mean_anomaly_score": float(
                        np.mean(
                            list(
                                anomaly_scores.values()
                            )
                        )
                    ),
                    "mean_aggregation_weight": float(
                        np.mean(
                            list(
                                aggregation_weights.values()
                            )
                        )
                    ),
                },
            )

        raise RuntimeError(
            f"Unsupported security state: "
            f"{self.current_state}"
        )

    # ------------------------------------------------------------------
    # Re-keying
    # ------------------------------------------------------------------

    def perform_rekey(self, server_round: int) -> None:
        """
        Perform the conceptual re-keying trigger after LOCKDOWN.

        Phase 2 already provides the cryptographic abstraction.

        For the first Phase-4 implementation, the important orchestration
        event is that a fresh QKD exchange is performed before training
        resumes.

        This method therefore performs a fresh BB84 exchange and records the
        resulting QBER.

        The actual AES-GCM key establishment should be connected here once
        the project defines how server/client keys are distributed.

        We deliberately do not fabricate a key exchange protocol here.
        """
        if not self.rekey_required:
            return

        log.warning(
            "[Round %d] Performing fresh BB84 re-key trigger.",
            server_round,
        )

        # Use server_round + 1 so this exchange is independent from the
        # QBER exchange that caused the lockdown.
        rekey_seed = (
            self.seed
            + (server_round + 1) * 100_000
        )

        protocol = BB84Protocol(
            sample_fraction=1.0,
            seed=rekey_seed,
        )

        rekey_result = protocol.run_exchange(
            n_qubits=self.n_qubits,
            intercept_probability=0.0,
        )

        if isinstance(
            rekey_result,
            (float, int, np.floating),
        ):
            rekey_qber = float(
                rekey_result
            )

        elif isinstance(rekey_result, dict):
            rekey_qber = float(
                rekey_result["qber"]
            )

        else:
            raise TypeError(
                "Unsupported BB84 re-key result type."
            )

        log.info(
            "[Round %d] Re-key BB84 QBER = %.4f",
            server_round,
            rekey_qber,
        )

        self.rekey_required = False
        self.lockdown_active = False

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_security_log(self) -> List[Dict[str, Any]]:
        """
        Return round security records as JSON-serializable dictionaries.

        The server runner can write this directly into:
            results/fl_training_log.json
        """
        output: List[Dict[str, Any]] = []

        for round_id in sorted(
            self.round_records.keys()
        ):
            record = self.round_records[
                round_id
            ]

            output.append(
                {
                    "round": record.round_id,
                    "client_qber": dict(
                        record.client_qber
                    ),
                    "q_max": record.q_max,
                    "state": record.state,
                    "reason": record.reason,
                    "policy": record.policy,
                    "accepted_clients": list(
                        record.accepted_clients
                    ),
                    "rejected_clients": list(
                        record.rejected_clients
                    ),
                    "anomaly_scores": dict(
                        record.anomaly_scores
                    ),
                    "aggregation_weights": dict(
                        record.aggregation_weights
                    ),
                    "lockdown": record.lockdown,
                    "rekey_required": record.rekey_required,
                }
            )

        return output
=======
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate client updates based on the state set in configure_fit().

        SECURE   → weighted FedAvg (standard)
        CAUTION  → weighted FedAvg with gradient-norm anomaly scoring
        LOCKDOWN → discard all updates, return last good parameters
        """
        state = self._round_state

        # Log round outcome
        system_qber = max(self._round_qber_per_client.values(), default=0.0)
        round_log = {
            "round":       server_round,
            "state":       state.value,
            "system_qber": system_qber,
            "n_results":   len(results),
            "n_failures":  len(failures),
        }

        if state == SecurityState.LOCKDOWN:
            log.warning("[Round %d] LOCKDOWN — discarding all %d updates.", server_round, len(results))
            round_log["aggregation"] = "lockdown_skipped"
            self.round_logs.append(round_log)
            metrics = {"state": state.value, "qber": system_qber, "skipped": 1}
            # Return last good model; round counter still advances in Flower
            return self._last_good_parameters, metrics

        if not results:
            log.warning("[Round %d] No results to aggregate.", server_round)
            return None, {}

        if state == SecurityState.CAUTION:
            aggregated, metrics = self._caution_aggregate(results, server_round)
            round_log["aggregation"] = "fedprox+anomaly"
        else:
            aggregated, metrics = self._secure_aggregate(results, server_round)
            round_log["aggregation"] = "fedavg"

        metrics["state"]  = state.value
        metrics["qber"]   = system_qber

        if aggregated is not None:
            self._last_good_parameters = aggregated

        round_log["metrics"] = metrics
        self.round_logs.append(round_log)
        return aggregated, metrics

    # ------------------------------------------------------------------
    # Aggregation rules
    # ------------------------------------------------------------------

    def _secure_aggregate(
        self,
        results: List[Tuple[ClientProxy, FitRes]],
        server_round: int,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Standard FedAvg — weighted average by number of training examples."""
        total_examples = sum(fit_res.num_examples for _, fit_res in results)
        if total_examples == 0:
            return None, {}

        weighted_arrays: Optional[NDArrays] = None
        for _, fit_res in results:
            weight     = fit_res.num_examples / total_examples
            client_nda = parameters_to_ndarrays(fit_res.parameters)
            if weighted_arrays is None:
                weighted_arrays = [arr * weight for arr in client_nda]
            else:
                for j, arr in enumerate(client_nda):
                    weighted_arrays[j] += arr * weight

        log.info("[Round %d] SECURE FedAvg over %d clients (%d examples).",
                 server_round, len(results), total_examples)
        return ndarrays_to_parameters(weighted_arrays), {"n_clients": len(results)}

    def _caution_aggregate(
        self,
        results: List[Tuple[ClientProxy, FitRes]],
        server_round: int,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        FedAvg with gradient-norm anomaly scoring.

        Clients flagged as suspicious (||∇W|| > mean + 2σ) receive a
        reduced aggregation weight of 0.05 regardless of dataset size.
        Non-flagged clients share the remaining weight proportionally.

        Note: clients already trained with proximal_mu=0.01 (FedProx)
        because configure_fit() injected that into their FitIns config.
        """
        # Compute L2 norm of each client's update (first parameter layer proxy)
        client_data: List[Tuple[ClientProxy, FitRes, NDArrays, float]] = []
        for proxy, fit_res in results:
            nda  = parameters_to_ndarrays(fit_res.parameters)
            norm = float(np.sqrt(sum(np.sum(a ** 2) for a in nda)))
            client_data.append((proxy, fit_res, nda, norm))

        norms = np.array([norm for *_, norm in client_data])
        mean_norm = float(norms.mean())
        std_norm  = float(norms.std()) if len(norms) > 1 else 0.0

        threshold = mean_norm + _ANOMALY_K * std_norm
        suspicious_cids = []

        # Assign weights
        weights: Dict[str, float] = {}
        total_examples = sum(fit_res.num_examples for _, fit_res, _, _ in client_data)

        for proxy, fit_res, _, norm in client_data:
            cid = proxy.cid
            if norm > threshold:
                weights[cid] = _SUSPICIOUS_WEIGHT_FLOOR
                suspicious_cids.append(cid)
            else:
                weights[cid] = fit_res.num_examples / total_examples if total_examples > 0 else 0.0

        # Renormalize: non-suspicious clients share 1 - sum(floor weights)
        total_suspicious_weight = len(suspicious_cids) * _SUSPICIOUS_WEIGHT_FLOOR
        remaining_weight        = 1.0 - total_suspicious_weight
        total_clean_examples = sum(
            fit_res.num_examples
            for proxy, fit_res, _, _ in client_data
            if proxy.cid not in suspicious_cids
        )
        for proxy, fit_res, _, _ in client_data:
            cid = proxy.cid
            if cid not in suspicious_cids and total_clean_examples > 0:
                weights[cid] = (fit_res.num_examples / total_clean_examples) * remaining_weight

        if suspicious_cids:
            log.warning(
                "[Round %d] CAUTION: %d suspicious client(s): %s  (threshold norm=%.4f)",
                server_round, len(suspicious_cids), suspicious_cids, threshold,
            )

        # Weighted aggregation
        weighted_arrays: Optional[NDArrays] = None
        for proxy, _, nda, _ in client_data:
            w = weights[proxy.cid]
            if weighted_arrays is None:
                weighted_arrays = [arr * w for arr in nda]
            else:
                for j, arr in enumerate(nda):
                    weighted_arrays[j] += arr * w

        metrics = {
            "n_clients":         len(results),
            "n_suspicious":      len(suspicious_cids),
            "mean_grad_norm":    mean_norm,
            "anomaly_threshold": threshold,
        }
        log.info("[Round %d] CAUTION aggregation: %d/%d clients clean.",
                 server_round, len(results) - len(suspicious_cids), len(results))
        return ndarrays_to_parameters(weighted_arrays), metrics

    # ------------------------------------------------------------------
    # Evaluate (pass-through; server-side eval optional)
    # ------------------------------------------------------------------

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        clients = self._sample_clients(client_manager)
        config  = {"server_round": server_round}
        return [(c, EvaluateIns(parameters, config)) for c in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if not results:
            return None, {}
        # Weighted average of loss
        total_examples = sum(r.num_examples for _, r in results)
        loss = sum(r.loss * r.num_examples for _, r in results) / total_examples
        metrics = {
            "auc_roc": float(
                np.mean([r.metrics.get("auc_roc", 0.0) for _, r in results])
            )
        }
        return loss, metrics

    def evaluate(
        self,
        server_round: int,
        parameters: Parameters,
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        if self._evaluate_fn is None:
            return None
        return self._evaluate_fn(server_round, parameters_to_ndarrays(parameters), {})

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sample_clients(self, client_manager: ClientManager) -> List[ClientProxy]:
        """Sample clients for this round."""
        sample_size = max(
            self._min_fit_clients,
            int(client_manager.num_available() * self._fraction_fit),
        )
        return client_manager.sample(
            num_clients=sample_size,
            min_num_clients=self._min_available_clients,
        )
>>>>>>> c8d000e56cdc54490d8e8750eb80577f1165bdf3
