"""SequentialClientManager is a real flwr ClientManager with fixed membership (it used to be duck-typed)."""

import pytest
from flwr.server.client_manager import ClientManager

from evefl.fl.runner import LocalClientProxy, SequentialClientManager


class _NoClient:
    """Never called: these tests only exercise the manager's bookkeeping."""


def _proxies(n):
    return [LocalClientProxy(str(i), _NoClient()) for i in range(n)]  # type: ignore[arg-type]


def test_is_a_flwr_client_manager():
    assert isinstance(SequentialClientManager(_proxies(2)), ClientManager)


def test_membership_is_fixed_and_all_clients_are_always_available():
    proxies = _proxies(3)
    manager = SequentialClientManager(proxies)
    assert manager.num_available() == 3
    assert list(manager.all()) == ["0", "1", "2"]
    assert manager.wait_for(num_clients=3, timeout=0) is True
    assert manager.wait_for(num_clients=4, timeout=0) is False


def test_register_and_unregister_are_refused_because_membership_is_fixed():
    manager = SequentialClientManager(_proxies(2))
    with pytest.raises(RuntimeError, match="fixed"):
        manager.register(LocalClientProxy("9", _NoClient()))  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="fixed"):
        manager.unregister(manager.all()["0"])
    assert manager.num_available() == 2


def test_sample_still_honours_min_clients_and_criterion():
    manager = SequentialClientManager(_proxies(3))
    assert [p.cid for p in manager.sample(2)] == ["0", "1"]
    with pytest.raises(RuntimeError, match="min_num_clients=4"):
        manager.sample(4, min_num_clients=4)
