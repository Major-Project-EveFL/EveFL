"""
Ray-free sequential FL driver.

Why this file exists
---------------------
`fl.simulation.start_simulation` drives Flower clients through Ray actors.
On Kaggle, Ray's GPU-actor scheduler repeatedly deadlocked against an
already-initialized CUDA context (each client got its own Ray actor
trying to claim a GPU fraction while the main process already held a
CUDA context for the parent notebook kernel).

The fix: skip Ray (and skip multiprocessing) entirely. Flower's
`Strategy` interface only needs a `ClientManager` (to sample clients)
and `ClientProxy` objects (to call `.fit()` / `.evaluate()`). Both are
just plain Python interfaces — nothing about them requires the network
transport or the simulation engine. So we:

    1. Wrap each real `flwr.client.NumPyClient` in a `LocalClientProxy`
       that implements `ClientProxy` by calling the client's methods
       directly, in-process, in the current Python thread.
    2. Implement a trivial `SequentialClientManager` that always returns
       the same fixed set of `LocalClientProxy` objects.
    3. Drive `configure_fit -> fit -> aggregate_fit -> evaluate` in a
       plain `for` loop over rounds.

This uses the *real* `flwr.client.NumPyClient` and `flwr.server.strategy`
classes end to end — nothing here is a mock of Flower's semantics, it's
just a different (single-process, single-thread) execution engine for
the same Strategy/Client contracts. `EveFLStrategy` (strategy.py) and
`EveFLClient` (client.py) never call into Ray or gRPC, so nothing about
them needs to change to run this way — this is the engine from day one,
not a later patch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from flwr.client.client import Client
from flwr.common import (
    Code,
    DisconnectRes,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    GetPropertiesIns,
    GetPropertiesRes,
    Parameters,
    ReconnectIns,
    Scalar,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy

log = logging.getLogger(__name__)


# ============================================================================
# In-process ClientProxy: calls the real Client directly, no network/Ray
# ============================================================================

class LocalClientProxy(ClientProxy):
    """
    A `ClientProxy` that calls a local, in-process `flwr.client.Client`
    directly instead of going over gRPC (real server) or a Ray actor
    (simulation). Everything downstream of this class — `EveFLStrategy`,
    `client_manager.sample()`, `aggregate_fit()` — sees a completely
    ordinary Flower `ClientProxy` and cannot tell the difference.
    """

    def __init__(self, cid: str, client: Client):
        super().__init__(cid)
        self._client = client

    def get_properties(
        self, ins: GetPropertiesIns, timeout: Optional[float], group_id: Optional[int]
    ) -> GetPropertiesRes:
        return self._client.get_properties(ins)

    def get_parameters(
        self, ins: GetParametersIns, timeout: Optional[float], group_id: Optional[int]
    ) -> GetParametersRes:
        return self._client.get_parameters(ins)

    def fit(self, ins: FitIns, timeout: Optional[float], group_id: Optional[int]) -> FitRes:
        return self._client.fit(ins)

    def evaluate(
        self, ins: EvaluateIns, timeout: Optional[float], group_id: Optional[int]
    ) -> EvaluateRes:
        return self._client.evaluate(ins)

    def reconnect(
        self, ins: ReconnectIns, timeout: Optional[float], group_id: Optional[int]
    ) -> DisconnectRes:
        # Nothing to actually disconnect — there is no network connection.
        return DisconnectRes(reason="")


# ============================================================================
# Fixed-membership ClientManager: same 3 hospitals every round
# ============================================================================

class SequentialClientManager:
    """
    Minimal stand-in for `flwr.server.client_manager.ClientManager`.

    EveFLStrategy only calls `.num_available()` and `.sample(...)` on
    whatever client_manager it's given (see `_sample_clients()` in
    strategy.py) — it never registers/unregisters clients dynamically,
    so a fixed-membership manager is all a sequential run needs.
    """

    def __init__(self, proxies: List[ClientProxy]):
        self._proxies: Dict[str, ClientProxy] = {p.cid: p for p in proxies}

    def num_available(self) -> int:
        return len(self._proxies)

    def sample(
        self,
        num_clients: int,
        min_num_clients: Optional[int] = None,
        criterion=None,
    ) -> List[ClientProxy]:
        proxies = list(self._proxies.values())
        if criterion is not None:
            proxies = [p for p in proxies if criterion.select(p)]
        if min_num_clients is not None and len(proxies) < min_num_clients:
            raise RuntimeError(
                f"Only {len(proxies)} client(s) available, "
                f"but min_num_clients={min_num_clients} was required."
            )
        return proxies[: min(num_clients, len(proxies))]

    def all(self) -> Dict[str, ClientProxy]:
        return dict(self._proxies)


def build_local_client_proxies(
    client_fn: Callable[[str], Client],
    num_clients: int,
) -> List[LocalClientProxy]:
    """
    Build `num_clients` in-process client proxies using the *same*
    `client_fn` factory that `fl.simulation.start_simulation` would have
    used (see `create_client_fn` in client.py) — cid "0", "1", "2", ...
    """
    proxies = []
    for i in range(num_clients):
        cid = str(i)
        client = client_fn(cid)
        proxies.append(LocalClientProxy(cid=cid, client=client))
    return proxies


# ============================================================================
# Round bookkeeping
# ============================================================================

@dataclass
class RoundFailure:
    server_round: int
    phase: str  # "fit" | "evaluate"
    cid: str
    error: str


@dataclass
class SequentialHistory:
    """Lightweight run history. `EveFLStrategy.round_logs` remains the
    single source of truth for per-round security state / aggregation
    details; this just tracks failures and gives a quick summary."""

    failures: List[RoundFailure] = field(default_factory=list)

    def add_failure(self, server_round: int, phase: str, cid: str, error: str) -> None:
        self.failures.append(RoundFailure(server_round, phase, cid, error))


# ============================================================================
# The sequential driver
# ============================================================================

def run_sequential_fl(
    *,
    client_proxies: List[ClientProxy],
    strategy: Strategy,
    num_rounds: int,
    initial_parameters: Parameters,
    timeout: Optional[float] = None,
    run_federated_evaluate: bool = True,
) -> Tuple[Parameters, SequentialHistory]:
    """
    Drive `num_rounds` of federated learning with no Ray, no
    multiprocessing, and no gRPC — a single Python process, single
    thread, one round strictly after another.

    Mirrors exactly what `flwr.server.Server.fit()` does internally,
    minus the client-manager registration/networking machinery that
    only matters for real distributed deployments or Ray simulation.

    Returns:
        (final_parameters, history)
    """
    if not client_proxies:
        raise ValueError("run_sequential_fl() requires at least one client proxy.")

    client_manager = SequentialClientManager(client_proxies)
    current_parameters = initial_parameters
    history = SequentialHistory()

    log.info(
        "Starting sequential (Ray-free) FL run: %d round(s), %d client(s).",
        num_rounds, len(client_proxies),
    )

    for server_round in range(1, num_rounds + 1):
        # ------------------------------------------------------------
        # FIT
        # ------------------------------------------------------------
        fit_instructions = strategy.configure_fit(server_round, current_parameters, client_manager)

        fit_results: List[Tuple[ClientProxy, FitRes]] = []
        fit_failures: List[BaseException] = []

        for proxy, fit_ins in fit_instructions:
            try:
                fit_res = proxy.fit(fit_ins, timeout=timeout, group_id=server_round)
                if fit_res.status.code != Code.OK:
                    raise RuntimeError(f"Client {proxy.cid} returned status {fit_res.status}")
                fit_results.append((proxy, fit_res))
            except Exception as exc:  # noqa: BLE001 - deliberately broad, mirrors flwr's own failure handling
                log.exception("[Round %d] Client %s failed during fit().", server_round, proxy.cid)
                fit_failures.append(exc)
                history.add_failure(server_round, "fit", proxy.cid, str(exc))

        aggregated_parameters, _fit_metrics = strategy.aggregate_fit(
            server_round, fit_results, fit_failures
        )
        if aggregated_parameters is not None:
            current_parameters = aggregated_parameters

        # ------------------------------------------------------------
        # SERVER-SIDE EVAL (centralized held-out set, if strategy has one)
        # ------------------------------------------------------------
        server_eval = strategy.evaluate(server_round, current_parameters)
        if server_eval is not None:
            loss, metrics = server_eval
            log.info("[Round %d] server-side eval: loss=%.4f metrics=%s", server_round, loss, metrics)
            _attach_eval_to_round_log(strategy, server_round, "server_eval", loss, metrics)

        # ------------------------------------------------------------
        # FEDERATED EVAL (optional, per-client evaluate())
        # ------------------------------------------------------------
        if run_federated_evaluate:
            eval_instructions = strategy.configure_evaluate(server_round, current_parameters, client_manager)
            if eval_instructions:
                eval_results: List[Tuple[ClientProxy, EvaluateRes]] = []
                eval_failures: List[BaseException] = []

                for proxy, eval_ins in eval_instructions:
                    try:
                        eval_res = proxy.evaluate(eval_ins, timeout=timeout, group_id=server_round)
                        if eval_res.status.code != Code.OK:
                            raise RuntimeError(f"Client {proxy.cid} returned status {eval_res.status}")
                        eval_results.append((proxy, eval_res))
                    except Exception as exc:  # noqa: BLE001
                        log.exception("[Round %d] Client %s failed during evaluate().", server_round, proxy.cid)
                        eval_failures.append(exc)
                        history.add_failure(server_round, "evaluate", proxy.cid, str(exc))

                loss, metrics = strategy.aggregate_evaluate(server_round, eval_results, eval_failures)
                if loss is not None:
                    _attach_eval_to_round_log(strategy, server_round, "federated_eval", loss, metrics)

    log.info(
        "Sequential FL run complete: %d round(s), %d failure(s).",
        num_rounds, len(history.failures),
    )
    return current_parameters, history


def _attach_eval_to_round_log(
    strategy: Strategy,
    server_round: int,
    key: str,
    loss: float,
    metrics: Dict[str, Scalar],
) -> None:
    """
    Merge eval results into `strategy.round_logs[-1]` so
    `EveFLStrategy.round_logs` stays the single source of truth for
    the JSON export in server.py, instead of introducing a second,
    parallel history structure.
    """
    round_logs = getattr(strategy, "round_logs", None)
    if not round_logs:
        return
    last = round_logs[-1]
    if last.get("round") != server_round:
        return
    last[key] = {"loss": loss, "metrics": dict(metrics)}
