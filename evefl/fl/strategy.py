"""
EveFLStrategy: the piece that actually connects the quantum layer to
federated aggregation.

Every round, before any client trains anything:

    1. Run one independent BB84 exchange per client link
       (evefl.quantum.factory.create_protocol) — this simulates the QKD
       handshake on that hospital's link to the server for this round.
       EVERY link is measured every round, including links whose client is
       currently excluded (otherwise an excluded client could never rejoin).
       Each link has its own channel (Eve and noise) from a declarative
       `ChannelPlan` (evefl.fl.channels), so Eve can sit on one link of three.
    2. The round policy (evefl.orchestration.policy, mode `global_binary`,
       `global` or `per_client`) turns the per-link QBERs (and optional key
       outcomes) into a decision: each client is TRAIN, TRAIN_PROX (CAUTION)
       or EXCLUDE, or the whole round is discarded.
    3. Only included clients are sent the model and a FitIns with their OWN state,
       a FedProx mu if they are in CAUTION and a learning rate (reduced in CAUTION).
       An excluded client is not sent the model at all.
    4. Aggregate the included updates with renormalised weights:
         SECURE   -> plain FedAvg
         CAUTION  -> median/MAD screening over all included updates (needs >= 3),
                     applied only to clients whose OWN state is CAUTION
         discard  -> keep the last known good parameters, flag `rekey_recommended`.
       An update from an excluded client is never aggregated, even if it arrives.

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

from evefl.fl.channels import ChannelPlan
from evefl.fl.schedule import cosine_lr
from evefl.fl.screening import flag_anomalous_updates, update_delta_norm
from evefl.orchestration.policy import (
    ClientAction,
    PolicyConfig,
    RoundDecision,
    create_policy,
)
from evefl.orchestration.state_machine import SecurityState, StateThresholds
from evefl.quantum.base import ChannelModel, QKDResult
from evefl.quantum.config import QKDConfig
from evefl.quantum.factory import DEFAULT_BACKEND, create_protocol
from evefl.quantum.seeding import qkd_seed_sequence

log = logging.getLogger(__name__)

DEFAULT_CAUTION_FEDPROX_MU = 0.01
DEFAULT_ANOMALY_K = 3.0  # flag an update if its delta norm > median + k*max(1.4826*MAD, rel_floor*median)
DEFAULT_CAUTION_LR_MULTIPLIER = 0.5  # paper: reduced learning rate in CAUTION (1e-3 -> 5e-4)
DEFAULT_ANOMALY_REL_FLOOR = 0.25  # spread floor, so near-identical honest clients don't make MAD ~ 0 (see screening.py)

KeyStatusFn = Callable[[int, str], str]  # (server_round, client id) -> "ok" | "no_key" | "abort_*"


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
        channels: Optional[ChannelPlan] = None,
        policy_config: Optional[PolicyConfig] = None,
        key_status_fn: Optional[KeyStatusFn] = None,
        evaluate_fn: Optional[Callable[[int, List[np.ndarray], dict],
                                        Optional[Tuple[float, Dict[str, Scalar]]]]] = None,
        val_evaluate_fn: Optional[Callable[[int, List[np.ndarray], dict],
                                            Optional[Tuple[float, Dict[str, Scalar]]]]] = None,
    ):
        """
        intercept_probability / intercept_probability_schedule: legacy "same Eve on every link" setting
            (a fixed alpha, or a callable `server_round -> alpha`). Mutually exclusive with `channels`.
        channels: declarative per-link plan (evefl.fl.channels.ChannelPlan); replaces the legacy
            alpha/noise arguments and can put Eve on one link only.
        policy_config: round policy mode (global_binary | global | per_client), min_clients and hysteresis.
            Default: the `global` three-state policy on `thresholds`.
        key_status_fn: optional `(server_round, cid) -> status` for key generation outcomes; None means keys
            are not required this round.
        evaluate_fn: optional centralized evaluation callback matching
            `(server_round, ndarrays, config) -> Optional[(loss, metrics)]`
            — see evaluation.py's `make_evaluate_fn()`.
        """
        super().__init__()
        if channels is not None and (intercept_probability != 0.0 or intercept_probability_schedule is not None
                                     or bit_flip_probability != 0.0):
            raise ValueError("Pass either `channels` or the legacy intercept/bit-flip arguments, not both.")
        if policy_config is not None and thresholds is not None:
            raise ValueError("Pass thresholds inside `policy_config`, not both.")
        self._initial_parameters = initial_parameters
        self._intercept_probability = float(intercept_probability)
        self._intercept_probability_schedule = intercept_probability_schedule
        self._channels = channels
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
        self._policy_config = policy_config or PolicyConfig(mode="global", thresholds=thresholds or StateThresholds())
        self._policy = create_policy(self._policy_config)
        self._key_status_fn = key_status_fn
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
        self._val_evaluate_fn = val_evaluate_fn  # validation split: for model selection, never the test split

        # Last known good parameters — what discarded rounds fall back to.
        self._last_good_parameters = initial_parameters

        # Per-round bookkeeping, exported by server.py to results JSON.
        # This is the single source of truth for what happened each
        # round (also read/patched by runner.py to attach eval results).
        self.round_logs: List[dict] = []
        # Rounds in which each client's update was actually aggregated (the policy's own count is the intent).
        self.participation_actual: Dict[str, int] = {}

        # Set during configure_fit(), read during aggregate_fit() for the
        # SAME round — configure_fit always runs immediately before
        # aggregate_fit for a given server_round in both the real Flower
        # server and the sequential runner, so this is safe.
        self._round_decision: Optional[RoundDecision] = None
        self._round_state: Optional[SecurityState] = None
        self._round_system_qber: float = 0.0
        self._round_intercept_probability: float = 0.0
        self._round_alpha_per_client: Dict[str, float] = {}
        self._round_qber_per_client: Dict[str, float] = {}
        self._round_learning_rate: Optional[float] = None
        self._round_lr_per_client: Dict[str, float] = {}
        self._round_qkd_details: Dict[str, Dict[str, float]] = {}
        self._round_key_status: Dict[str, str] = {}
        # Global weights sent to clients this round: the reference for update-delta norms.
        self._round_global_ndarrays: List[np.ndarray] = []

    # ------------------------------------------------------------------
    # Strategy interface
    # ------------------------------------------------------------------

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        return self._initial_parameters

    def _channel_for(self, cid: str, server_round: int) -> ChannelModel:
        if self._channels is not None:
            return self._channels.channel_for(cid, server_round)
        if self._intercept_probability_schedule is not None:
            alpha = float(self._intercept_probability_schedule(server_round))
            if not 0.0 <= alpha <= 1.0:
                raise ValueError(
                    f"intercept_probability_schedule({server_round}) returned {alpha}, outside [0, 1]."
                )
        else:
            alpha = self._intercept_probability
        return ChannelModel(intercept_probability=alpha, bit_flip_probability=self._bit_flip_probability)

    def _learning_rate(self, server_round: int, action: ClientAction) -> Optional[float]:
        if self._base_lr is None:
            return None
        learning_rate = (
            cosine_lr(self._base_lr, server_round, self._num_rounds)
            if self._num_rounds is not None else self._base_lr
        )
        if action == ClientAction.TRAIN_PROX:
            learning_rate *= self._caution_lr_multiplier
        return learning_rate

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        clients = self._sample_clients(client_manager)
        self._round_global_ndarrays = parameters_to_ndarrays(parameters)

        # -- 1. One BB84 exchange per client link, this round (excluded links too) ----------------------
        qber_per_client: Dict[str, float] = {}
        qkd_details: Dict[str, Dict[str, float]] = {}
        alpha_per_client: Dict[str, float] = {}
        for client in clients:
            channel = self._channel_for(client.cid, server_round)
            alpha_per_client[client.cid] = channel.intercept_probability
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
        self._round_alpha_per_client = alpha_per_client
        self._round_intercept_probability = max(alpha_per_client.values()) if alpha_per_client else 0.0

        # -- 2. Decide ----------------------------------------------------------------------------------
        key_status: Optional[Dict[str, str]] = None
        if self._key_status_fn is not None:
            key_status = {client.cid: self._key_status_fn(server_round, client.cid) for client in clients}
        self._round_key_status = dict(key_status or {})
        decision = self._policy.decide(qber_per_client, key_status)

        self._round_decision = decision
        self._round_state = decision.global_state
        self._round_system_qber = system_qber
        self._round_qber_per_client = qber_per_client
        self._round_qkd_details = qkd_details

        if decision.discard_round:
            log.warning("[Round %d] mode=%s round DISCARDED (%s: %s), keeping last known good parameters.",
                        server_round, decision.mode, decision.reason, decision.detail)
        elif decision.excluded:
            log.warning("[Round %d] mode=%s excluded %s (%s); continuing with %s.", server_round, decision.mode,
                        decision.excluded, decision.exclusion_reasons, decision.included)
        else:
            log.info("[Round %d] mode=%s state=%s system_qber=%.4f", server_round, decision.mode,
                     decision.global_state.value, system_qber)

        # -- 3. FitIns only for included clients (excluded clients are not sent the model) ------------------
        self._round_lr_per_client = {}
        instructions: List[Tuple[ClientProxy, FitIns]] = []
        if decision.discard_round:
            return instructions
        for client in clients:
            client_decision = decision.clients[client.cid]
            if client_decision.action == ClientAction.EXCLUDE:
                continue
            fedprox_mu = self._caution_fedprox_mu if client_decision.action == ClientAction.TRAIN_PROX else 0.0
            config: Dict[str, Scalar] = {
                "state": client_decision.state.value,
                "qber": qber_per_client[client.cid],
                "fedprox_mu": fedprox_mu,
                "server_round": server_round,
            }
            learning_rate = self._learning_rate(server_round, client_decision.action)
            if learning_rate is not None:
                config["learning_rate"] = learning_rate
                self._round_lr_per_client[client.cid] = learning_rate
            instructions.append((client, FitIns(parameters, config)))
        self._round_learning_rate = next(iter(self._round_lr_per_client.values()), None)
        return instructions

    def _base_log(self, server_round: int, n_results: int, n_failures: int) -> dict:
        decision = self._round_decision
        round_log: dict = {
            "round": server_round,
            "state": (self._round_state or SecurityState.SECURE).value,
            "system_qber": self._round_system_qber,
            "intercept_probability": self._round_intercept_probability,
            "intercept_probability_per_client": dict(self._round_alpha_per_client),
            "qber_per_client": dict(self._round_qber_per_client),
            "qkd_per_client": {cid: dict(d) for cid, d in self._round_qkd_details.items()},
            "learning_rate": self._round_learning_rate,
            "learning_rate_per_client": dict(self._round_lr_per_client),
            "n_results": n_results,
            "n_failures": n_failures,
        }
        if decision is not None:
            round_log.update({
                "policy_mode": decision.mode,
                "round_discarded": decision.discard_round,
                "discard_reason": decision.reason,
                "client_states": {cid: d.state.value for cid, d in decision.clients.items()},
                "client_actions": {cid: d.action.value for cid, d in decision.clients.items()},
                "included_clients": decision.included,
                "excluded_clients": decision.excluded,
                "exclusion_reasons": decision.exclusion_reasons,
                "key_status": dict(self._round_key_status),
                "participation_intended": dict(decision.participation),
            })
        return round_log

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        decision = self._round_decision
        round_log = self._base_log(server_round, len(results), len(failures))

        # ---- Discarded round (LOCKDOWN / too few clients / key failure): keep last good params. --------
        if decision is not None and decision.discard_round:
            round_log.update({
                "aggregation": "lockdown_discarded",
                "model_updated": False,
                "rekey_recommended": True,
                "participation_actual": dict(self.participation_actual),
            })
            self.round_logs.append(round_log)
            log.warning("[Round %d] round discarded (%s) — keeping last known good parameters.",
                        server_round, decision.reason)
            return self._last_good_parameters, round_log

        # ---- An excluded client's update is NEVER aggregated, even if it arrives. ----------------------
        if decision is not None:
            included = set(decision.included)
            ignored = [proxy.cid for proxy, _ in results if proxy.cid not in included]
            if ignored:
                log.warning("[Round %d] ignoring update(s) from excluded client(s) %s", server_round, ignored)
            results = [(proxy, fit_res) for proxy, fit_res in results if proxy.cid in included]
            round_log["ignored_updates_from"] = ignored

        if not results:
            round_log.update({"aggregation": "no_results", "model_updated": False, "rekey_recommended": False,
                              "participation_actual": dict(self.participation_actual)})
            self.round_logs.append(round_log)
            return self._last_good_parameters, round_log

        # ---- Extract client updates -------------------------------------
        client_ndarrays = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
        num_examples = [fit_res.num_examples for _, fit_res in results]
        cids = [proxy.cid for proxy, _ in results]

        if sum(num_examples) == 0:
            # Every client returned num_examples=0 (e.g. all skipped
            # training for some reason) — nothing usable to aggregate.
            round_log.update({"aggregation": "no_examples", "model_updated": False, "rekey_recommended": False,
                              "participation_actual": dict(self.participation_actual)})
            self.round_logs.append(round_log)
            return self._last_good_parameters, round_log

        weights = np.array(num_examples, dtype=np.float64)

        # ---- Screening: statistics over ALL included updates, applied only to CAUTION clients ----------
        anomalous_cids: List[str] = []
        clip_clients = set(decision.clip_clients) if decision is not None else set()
        any_caution = decision is not None and any(
            decision.clients[c].state == SecurityState.CAUTION for c in cids if c in decision.clients)
        screening_active = bool(clip_clients)
        round_log["screening_active"] = screening_active
        if any_caution and not screening_active:
            # Fewer than 3 updates remain (or none of them is in CAUTION): the median/MAD statistics are undefined.
            round_log["screening_inactive_reason"] = "fewer than 3 included updates"
        if screening_active:
            norms = [update_delta_norm(nd, self._round_global_ndarrays) for nd in client_ndarrays]
            round_log["update_delta_norms"] = dict(zip(cids, norms))
            flags = flag_anomalous_updates(norms, k=self._anomaly_k, rel_floor=self._anomaly_rel_floor)
            for i, flagged in enumerate(flags):
                if flagged and cids[i] in clip_clients:  # clean (SECURE) clients stay plain FedAvg
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
        for cid in cids:
            self.participation_actual[cid] = self.participation_actual.get(cid, 0) + 1

        round_log.update({
            "aggregation": "fedavg" if not any_caution else "fedprox_anomaly_weighted",
            "model_updated": True,
            "anomalous_clients": anomalous_cids,
            "aggregated_clients": cids,
            "aggregation_weights": dict(zip(cids, (float(w) for w in weights))),
            "fedprox_active_clients": sum(
                1 for _, fit_res in results if float(fit_res.metrics.get("fedprox_mu", 0.0)) > 0.0
            ),
            "rekey_recommended": False,
            "participation_actual": dict(self.participation_actual),
        })
        self.round_logs.append(round_log)

        return aggregated_parameters, round_log

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        decision = self._round_decision
        if decision is not None and decision.discard_round:
            return []  # don't bother round-tripping to clients on a discarded round
        clients = self._sample_clients(client_manager)
        if decision is not None:  # never send the model to an excluded client
            included = set(decision.included)
            clients = [c for c in clients if c.cid in included]
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

    def evaluate_validation(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate the global model on the VALIDATION patients (used to choose rounds / hyperparameters)."""
        if self._val_evaluate_fn is None:
            return None
        return self._val_evaluate_fn(server_round, parameters_to_ndarrays(parameters), {})

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sample_clients(self, client_manager: ClientManager) -> List[ClientProxy]:
        sample_size = max(self._min_fit_clients, int(client_manager.num_available() * self._fraction_fit))
        return client_manager.sample(num_clients=sample_size, min_num_clients=self._min_available_clients)

