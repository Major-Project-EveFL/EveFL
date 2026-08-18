"""
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
"""

from __future__ import annotations

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

log = logging.getLogger(__name__)


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
    """

    def __init__(
        self,
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

    def configure_fit(
        self,
        server_round: int,
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

    def aggregate_fit(
        self,
        server_round: int,
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