"""
EveFLStrategy: the piece that actually connects the quantum layer to
federated aggregation.

Every round, before any client trains anything:

    1. Run one independent BB84 exchange per participating client
       (evefl.quantum.factory.create_protocol) — this simulates the QKD
       handshake on that hospital's link to the server for this round.
    2. Take system_qber = max(per-client QBER). A single compromised
       link is the thing we care about catching, so we use the
       worst-case reading rather than an average that a healthy
       majority could dilute.
    3. Classify system_qber via StateController -> SECURE / CAUTION / LOCKDOWN
       (evefl.orchestration.state_machine).
    4. Every client gets told the state and (if CAUTION) a FedProx mu,
       via FitIns.config — the client itself doesn't decide any of this
       (see client.py).
    5. Aggregate according to that state:
         SECURE   -> plain FedAvg
         CAUTION  -> FedAvg with anomaly-downweighting (an update whose
                     L2 norm is an outlier gets its weight halved, not
                     zeroed — one hospital legitimately having more
                     signal shouldn't be punished the same as a
                     genuinely poisoned update)
         LOCKDOWN -> discard the round entirely, keep the last known
                     good parameters, flag `rekey_recommended`.

This class implements the real `flwr.server.strategy.Strategy` ABC.
It is executed by the Ray-free sequential driver in runner.py — nothing
in this file assumes or depends on Ray, gRPC, or Flower's simulation
engine; it only needs a `ClientManager`-shaped object and
`ClientProxy`-shaped objects, which is exactly what runner.py's
`SequentialClientManager` / `LocalClientProxy` provide.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy

from evefl.fl.schedule import cosine_lr
from evefl.fl.screening import flag_anomalous_updates, update_delta_norm
from evefl.orchestration.state_machine import SecurityState, StateController, StateThresholds
from evefl.quantum.base import ChannelModel, QKDResult
from evefl.quantum.config import QKDConfig
from evefl.quantum.factory import DEFAULT_BACKEND, create_protocol
from evefl.quantum.seeding import qkd_seed_sequence

log = logging.getLogger(__name__)

DEFAULT_CAUTION_FEDPROX_MU = 0.01
DEFAULT_ANOMALY_K = 3.0  # flag an update if its delta norm > median + k*max(1.4826*MAD, rel_floor*median)
DEFAULT_CAUTION_LR_MULTIPLIER = 0.5  # paper: reduced learning rate in CAUTION (1e-3 -> 5e-4)
DEFAULT_ANOMALY_REL_FLOOR = 0.25  # spread floor, so near-identical honest clients don't make MAD ~ 0 (see screening.py)


class EveFLStrategy(Strategy):
    def __init__(
        self,
        initial_parameters: Parameters,
        *,
        intercept_probability: float = 0.0,
        intercept_probability_schedule: Optional[Callable[[int], float]] = None,
        n_qubits: int = 1024,
        thresholds: Optional[StateThresholds] = None,
        fraction_fit: float = 1.0,
        min_fit_clients: int = 3,
        min_available_clients: int = 3,
        caution_fedprox_mu: float = DEFAULT_CAUTION_FEDPROX_MU,
        anomaly_k: float = DEFAULT_ANOMALY_K,
        anomaly_rel_floor: float = DEFAULT_ANOMALY_REL_FLOOR,
        base_lr: Optional[float] = None,
        num_rounds: Optional[int] = None,
        caution_lr_multiplier: float = DEFAULT_CAUTION_LR_MULTIPLIER,
        qkd_backend: str = DEFAULT_BACKEND,
        sample_fraction: float = 0.25,
        bit_flip_probability: float = 0.0,
        experiment_seed: int = 0,
        evaluate_fn: Optional[Callable[[int, List[np.ndarray], dict],
                                        Optional[Tuple[float, Dict[str, Scalar]]]]] = None,
    ):
        """
        intercept_probability: fixed Eve intercept-resend probability
            used every round, unless `intercept_probability_schedule`
            is given.
        intercept_probability_schedule: optional callable
            `server_round -> alpha in [0, 1]`, for experiments/demos
            where Eve's activity changes over the run (quiet, then an
            attack, then quiet again). Overrides `intercept_probability`
            when provided.
        evaluate_fn: optional centralized evaluation callback matching
            `(server_round, ndarrays, config) -> Optional[(loss, metrics)]`
            — see evaluation.py's `make_evaluate_fn()`.
        """
        super().__init__()
        self._initial_parameters = initial_parameters
        self._intercept_probability = float(intercept_probability)
        self._intercept_probability_schedule = intercept_probability_schedule
        self._n_qubits = int(n_qubits)
        self._qkd_backend = qkd_backend
        self._sample_fraction = sample_fraction
        self._bit_flip_probability = bit_flip_probability
        # Streams are derived from (experiment_seed, round, client): different experiment seeds give
        # different QBER trajectories, and scenarios that differ only in alpha are paired.
        self._experiment_seed = int(experiment_seed)
        _qkd = QKDConfig(qkd_backend, self._n_qubits, sample_fraction, bit_flip_probability)
        if _qkd.expected_sample_size < _qkd.MIN_EXPECTED_SAMPLE:
            log.warning(
                "QKD config expects only ~%.0f bits in the QBER sample (n_qubits=%d, sample_fraction=%.2f): "
                "the state decisions will mostly be noise. Use at least %d.",
                _qkd.expected_sample_size, self._n_qubits, sample_fraction, _qkd.MIN_EXPECTED_SAMPLE,
            )
        self._state_controller = StateController(thresholds or StateThresholds())
        self._fraction_fit = fraction_fit
        self._min_fit_clients = min_fit_clients
        self._min_available_clients = min_available_clients
        self._caution_fedprox_mu = caution_fedprox_mu
        self._anomaly_k = anomaly_k
        self._anomaly_rel_floor = anomaly_rel_floor
        # If base_lr is given, the server sends each client a per-round learning rate that follows
        # a cosine over the GLOBAL round (schedule.py). Without num_rounds the base rate is constant.
        self._base_lr = base_lr
        self._num_rounds = num_rounds
        self._caution_lr_multiplier = caution_lr_multiplier
        self._evaluate_fn = evaluate_fn

        # Last known good parameters — what LOCKDOWN rounds fall back to.
        self._last_good_parameters = initial_parameters

        # Per-round bookkeeping, exported by server.py to results JSON.
        # This is the single source of truth for what happened each
        # round (also read/patched by runner.py to attach eval results).
        self.round_logs: List[dict] = []

        # Set during configure_fit(), read during aggregate_fit() for the
        # SAME round — configure_fit always runs immediately before
        # aggregate_fit for a given server_round in both the real Flower
        # server and the sequential runner, so this is safe.
        self._round_state: Optional[SecurityState] = None
        self._round_system_qber: float = 0.0
        self._round_intercept_probability: float = 0.0
        self._round_qber_per_client: Dict[str, float] = {}
        self._round_learning_rate: Optional[float] = None
        self._round_qkd_details: Dict[str, Dict[str, float]] = {}
        # Global weights sent to clients this round: the reference for update-delta norms.
        self._round_global_ndarrays: List[np.ndarray] = []

    # ------------------------------------------------------------------
    # Strategy interface
    # ------------------------------------------------------------------

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        return self._initial_parameters

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        clients = self._sample_clients(client_manager)
        self._round_global_ndarrays = parameters_to_ndarrays(parameters)

        # -- 1. Resolve this round's Eve intercept probability --------
        if self._intercept_probability_schedule is not None:
            alpha = float(self._intercept_probability_schedule(server_round))
            if not 0.0 <= alpha <= 1.0:
                raise ValueError(
                    f"intercept_probability_schedule({server_round}) returned "
                    f"{alpha}, outside [0, 1]."
                )
        else:
            alpha = self._intercept_probability
        self._round_intercept_probability = alpha

        # -- 2. One BB84 exchange per client, this round ---------------
        qber_per_client: Dict[str, float] = {}
        qkd_details: Dict[str, Dict[str, float]] = {}
        channel = ChannelModel(intercept_probability=alpha, bit_flip_probability=self._bit_flip_probability)
        for client in clients:
            protocol = create_protocol(
                self._qkd_backend,
                sample_fraction=self._sample_fraction,
                seed_sequence=qkd_seed_sequence(self._experiment_seed, server_round, client.cid),
            )
            result: QKDResult = protocol.run_exchange(n_qubits=self._n_qubits, channel=channel)
            qber_per_client[client.cid] = result.qber
            # Only `qber` (the sampled estimate) drives the controller. The sample size/errors are
            # what a real system observes; `sim_only_true_qber` is simulation ground truth, logged
            # to measure estimator error.
            qkd_details[client.cid] = {
                "qber_sample_size": result.metadata.get("qber_sample_size"),
                "qber_sample_errors": result.metadata.get("qber_sample_errors"),
                "n_sifted": result.n_sifted,
                "sim_only_true_qber": result.metadata.get("sim_only_true_qber"),
            }

        system_qber = max(qber_per_client.values()) if qber_per_client else 0.0

        # -- 3. Classify -------------------------------------------------
        transition = self._state_controller.update(system_qber)
        state = transition.new_state

        self._round_state = state
        self._round_system_qber = system_qber
        self._round_qber_per_client = qber_per_client
        self._round_qkd_details = qkd_details

        if transition.changed:
            log.warning(
                "[Round %d] Security state change: %s -> %s (system QBER=%.4f)",
                server_round, transition.previous_state, state.value, system_qber,
            )
        else:
            log.info("[Round %d] State=%s system_qber=%.4f", server_round, state.value, system_qber)

        # -- 4. Build per-client FitIns -----------------------------------
        fedprox_mu = self._caution_fedprox_mu if state == SecurityState.CAUTION else 0.0
        config: Dict[str, Scalar] = {
            "state": state.value,
            "qber": system_qber,
            "fedprox_mu": fedprox_mu,
            "server_round": server_round,
        }
        if self._base_lr is not None:
            learning_rate = (
                cosine_lr(self._base_lr, server_round, self._num_rounds)
                if self._num_rounds is not None else self._base_lr
            )
            if state == SecurityState.CAUTION:
                learning_rate *= self._caution_lr_multiplier
            config["learning_rate"] = learning_rate
            self._round_learning_rate = learning_rate
        fit_ins = FitIns(parameters, config)
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        state = self._round_state or SecurityState.SECURE
        system_qber = self._round_system_qber

        round_log = {
            "round": server_round,
            "state": state.value,
            "system_qber": system_qber,
            "intercept_probability": self._round_intercept_probability,
            "qber_per_client": dict(self._round_qber_per_client),
            "qkd_per_client": {cid: dict(d) for cid, d in self._round_qkd_details.items()},
            "learning_rate": self._round_learning_rate,
            "n_results": len(results),
            "n_failures": len(failures),
        }

        # ---- LOCKDOWN: never aggregate. Keep last good params. --------
        if state == SecurityState.LOCKDOWN:
            round_log.update({
                "aggregation": "lockdown_discarded",
                "model_updated": False,
                "rekey_recommended": True,
            })
            self.round_logs.append(round_log)
            log.warning("[Round %d] LOCKDOWN — round discarded, keeping last known good parameters.",
                        server_round)
            return self._last_good_parameters, round_log

        if not results:
            round_log.update({"aggregation": "no_results", "model_updated": False, "rekey_recommended": False})
            self.round_logs.append(round_log)
            return self._last_good_parameters, round_log

        # ---- Extract client updates -------------------------------------
        client_ndarrays = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
        num_examples = [fit_res.num_examples for _, fit_res in results]
        cids = [proxy.cid for proxy, _ in results]

        if sum(num_examples) == 0:
            # Every client returned num_examples=0 (e.g. all skipped
            # training for some reason) — nothing usable to aggregate.
            round_log.update({"aggregation": "no_examples", "model_updated": False, "rekey_recommended": False})
            self.round_logs.append(round_log)
            return self._last_good_parameters, round_log

        weights = np.array(num_examples, dtype=np.float64)

        anomalous_cids: List[str] = []
        if state == SecurityState.CAUTION:
            norms = [update_delta_norm(nd, self._round_global_ndarrays) for nd in client_ndarrays]
            round_log["update_delta_norms"] = dict(zip(cids, norms))
            flags = flag_anomalous_updates(norms, k=self._anomaly_k, rel_floor=self._anomaly_rel_floor)
            for i, flagged in enumerate(flags):
                if flagged:
                    weights[i] *= 0.5  # downweight, don't zero — could be legitimate signal
                    anomalous_cids.append(cids[i])
            if anomalous_cids:
                log.warning("[Round %d] CAUTION anomaly downweighted client(s): %s",
                            server_round, anomalous_cids)

        weights = weights / weights.sum()

        n_tensors = len(client_ndarrays[0])
        aggregated: List[np.ndarray] = []
        for tensor_i in range(n_tensors):
            stacked = np.stack([client_ndarrays[c][tensor_i] for c in range(len(client_ndarrays))])
            weighted = np.tensordot(weights, stacked, axes=1).astype(stacked.dtype)
            aggregated.append(weighted)

        aggregated_parameters = ndarrays_to_parameters(aggregated)
        self._last_good_parameters = aggregated_parameters

        round_log.update({
            "aggregation": "fedavg" if state == SecurityState.SECURE else "fedprox_anomaly_weighted",
            "model_updated": True,
            "anomalous_clients": anomalous_cids,
            "fedprox_active_clients": sum(
                1 for _, fit_res in results if float(fit_res.metrics.get("fedprox_mu", 0.0)) > 0.0
            ),
            "rekey_recommended": False,
        })
        self.round_logs.append(round_log)

        return aggregated_parameters, round_log

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        if self._round_state == SecurityState.LOCKDOWN:
            return []  # don't bother round-tripping to clients on a discarded round
        clients = self._sample_clients(client_manager)
        eval_ins = EvaluateIns(parameters, {"server_round": server_round})
        return [(client, eval_ins) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if not results:
            return None, {}
        total_examples = sum(r.num_examples for _, r in results)
        if total_examples == 0:
            return None, {}
        weighted_loss = sum(r.loss * r.num_examples for _, r in results) / total_examples
        return weighted_loss, {"n_clients_evaluated": len(results)}

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        if self._evaluate_fn is None:
            return None
        ndarrays = parameters_to_ndarrays(parameters)
        return self._evaluate_fn(server_round, ndarrays, {})

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sample_clients(self, client_manager: ClientManager) -> List[ClientProxy]:
        sample_size = max(self._min_fit_clients, int(client_manager.num_available() * self._fraction_fit))
        return client_manager.sample(num_clients=sample_size, min_num_clients=self._min_available_clients)
