"""
QKD experiment parameters, with a guard against meaningless configurations.

The QBER the controller sees is an estimate from a public sample of the sifted key. With
`n_qubits` sent, about n/2 bits survive sifting, so the sample holds about
n * sample_fraction / 2 bits. Below ~100 the state decisions are mostly noise (n_qubits=16
gives a 2-bit sample; n_qubits=64 gives 8), so experiments refuse to start on such a config
unless the caller explicitly opts in.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import ClassVar, Dict

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class QKDConfig:
    backend: str
    n_qubits: int
    sample_fraction: float
    bit_flip_probability: float = 0.0

    MIN_EXPECTED_SAMPLE: ClassVar[int] = 100

    @property
    def expected_sample_size(self) -> float:
        return self.n_qubits * 0.5 * self.sample_fraction

    def validate(self, *, allow_small_sample: bool = False) -> None:
        """Raise ValueError for impossible or meaningless settings (warn instead if allowed)."""
        if self.n_qubits < 1:
            raise ValueError(f"n_qubits must be >= 1, got {self.n_qubits}")
        if not 0.0 < self.sample_fraction <= 1.0:
            raise ValueError(f"sample_fraction must be in (0, 1], got {self.sample_fraction}")
        if not 0.0 <= self.bit_flip_probability <= 0.5:
            raise ValueError(f"bit_flip_probability must be in [0, 0.5], got {self.bit_flip_probability}")
        if self.expected_sample_size < self.MIN_EXPECTED_SAMPLE:
            message = (
                f"n_qubits={self.n_qubits} with sample_fraction={self.sample_fraction} gives an expected QBER "
                f"sample of only ~{self.expected_sample_size:.0f} bits (< {self.MIN_EXPECTED_SAMPLE}); the "
                f"controller's SECURE/CAUTION/LOCKDOWN decisions would mostly be noise. Use a preset such as "
                f"'lite' (n_qubits=1024), or pass allow_small_sample=True / --allow-small-sample to override."
            )
            if not allow_small_sample:
                raise ValueError(message)
            log.warning("%s (override active)", message)

    def to_dict(self) -> Dict[str, object]:
        return {
            "backend": self.backend,
            "n_qubits": self.n_qubits,
            "sample_fraction": self.sample_fraction,
            "bit_flip_probability": self.bit_flip_probability,
            "expected_sample_size": self.expected_sample_size,
        }


# Named presets. "lite" is the minimum sensible setting for demos and quick runs. The headline
# preset for reported results is chosen from the sample-size sweep (docs/06) and added later.
PRESETS: Dict[str, Dict[str, object]] = {
    "lite": {"n_qubits": 1024, "sample_fraction": 0.25},
    # Headline for reported results: a documented deviation from the pre-registered sweep rule (docs/06).
    "headline": {"n_qubits": 1024, "sample_fraction": 0.25},
}
