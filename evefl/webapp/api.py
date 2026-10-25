"""
The data behind the demonstration web app, as plain functions that return JSON-ready dictionaries.

Nothing here touches HTTP, so each function is tested directly. Numbers come from the recorded files or from running the library; a
value that is not a finite number becomes `None` (JSON has no NaN). QBER values are fractions in [0, 1].
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

import numpy as np

from evefl import selfcheck, showcase
from evefl.orchestration.state_machine import StateThresholds

TAGLINE = ("Hospitals train a chest X-ray AI together without sharing a single image. A simulated quantum link watches each "
           "hospital's connection, and a controller decides, round by round, whose update can be trusted.")


def to_jsonable(value: Any) -> Any:
    """Make `value` safe for `json.dumps(..., allow_nan=False)`: NaN and infinity become None, NumPy values become Python values."""
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(dataclasses.asdict(value))
    return value


def summary(root: Optional[Path] = None) -> Dict[str, Any]:
    """The home page: the headline numbers (each with a label and a source file) and the controller's fixed thresholds."""
    thresholds = StateThresholds()
    return to_jsonable({
        "title": "EveFL",
        "tagline": TAGLINE,
        "headline": showcase.headline_numbers(root),
        "thresholds": {"secure_max": thresholds.secure_max, "caution_max": thresholds.caution_max},
    })


def check_plan(checkpoint_dir: Optional[Path] = None) -> Dict[str, Any]:
    """The checks that a live run will perform, so the page can show the queue before the first result arrives."""
    return {"checks": [{"name": c.name, "title": c.title, "what": c.what, "kind": c.kind}
                       for c in selfcheck.selected_checks(checkpoint_dir=checkpoint_dir)]}


def check_events(checkpoint_dir: Optional[Path] = None) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Run the live checks one at a time: ("result", {...}) for each, then ("done", summary)."""
    results = []
    for result in selfcheck.iter_checks(checkpoint_dir=checkpoint_dir):
        results.append(result)
        yield "result", to_jsonable(result)
    yield "done", to_jsonable(selfcheck.summarise(results))
