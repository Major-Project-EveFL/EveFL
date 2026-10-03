"""
Software environment recorded next to every result.

Results and partitions depend on library versions (NumPy's Generator streams are not guaranteed identical across
versions; PyTorch kernels differ across builds), so each results JSON carries this block. It is provenance, not
configuration: it is kept OUT of the experiment config hash so the same experiment hashes the same on every machine.
Versions come from package metadata; nothing is imported, so a missing package is reported as None instead of failing.
"""

from __future__ import annotations

import platform
from importlib import metadata
from typing import Any, Dict, Iterable, Optional

TRACKED_PACKAGES = (
    "torch", "torchvision", "numpy", "scipy", "pandas", "scikit-learn", "flwr", "cryptography",
    "pycryptodome", "PyYAML", "qiskit", "qiskit-aer", "tenseal",
)


def library_versions(packages: Iterable[str] = TRACKED_PACKAGES) -> Dict[str, Optional[str]]:
    """Installed version of each package (e.g. '2.5.1+cpu'), or None if it is not installed."""
    versions: Dict[str, Optional[str]] = {}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def runtime_environment() -> Dict[str, Any]:
    """Interpreter, operating system family, CPU architecture and tracked library versions (JSON-ready)."""
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "system": platform.system(),
        "machine": platform.machine(),
        "libraries": library_versions(),
    }
