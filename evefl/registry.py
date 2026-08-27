"""
Generic registry for pluggable components.

Every extensible subsystem in EveFL (QKD protocols, cipher suites,
state classifiers, attack detectors) uses one of these registries so
new implementations can be added without touching existing code.

Usage:
    quantum_registry = Registry("quantum_protocol")

    @quantum_registry.register("bb84")
    class BB84Protocol(QKDProtocol):
        ...

    protocol_cls = quantum_registry.get("bb84")
"""

from __future__ import annotations

from typing import Callable, Dict, Generic, Type, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, name: str):
        self.name = name
        self._items: Dict[str, Type[T]] = {}

    def register(self, key: str) -> Callable[[Type[T]], Type[T]]:
        def decorator(cls: Type[T]) -> Type[T]:
            if key in self._items:
                raise ValueError(
                    f"'{key}' is already registered in registry '{self.name}'. "
                    f"Choose a different key or remove the existing registration."
                )
            self._items[key] = cls
            return cls

        return decorator

    def get(self, key: str) -> Type[T]:
        if key not in self._items:
            raise KeyError(
                f"'{key}' not found in registry '{self.name}'. "
                f"Available: {list(self._items.keys())}"
            )
        return self._items[key]

    def list_keys(self) -> list[str]:
        return list(self._items.keys())
