"""
Live system check for the dashboard's home page.

Each check calls real library code and reports pass, fail or skip with a one-line detail and its duration. A check that raises or
asserts is a failure (reported, never hidden, never fatal to the others); a check that cannot run on this machine, such as the
model check without a saved checkpoint, is a skip with its reason. Nothing here is a security certification: the key material in
the link-protection check is random bytes made up for the check, and the simulator's own keys are seed-determined and not secret.

QBER values are fractions in [0, 1].
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATION = REPO_ROOT / "docs" / "validation"


class SkipCheck(Exception):
    """Raised by a check that cannot run on this machine; the message is the reason shown to the viewer."""


@dataclass(frozen=True)
class Check:
    name: str
    title: str
    what: str            # plain-language statement of what is exercised
    kind: str            # "real code", "simulated" or "recorded"
    run: Callable[[], str]


@dataclass(frozen=True)
class CheckResult:
    name: str
    title: str
    what: str
    kind: str
    status: str          # "pass", "fail" or "skip"
    detail: str
    seconds: float


def default_checkpoint_dir() -> Path:
    return Path(os.environ.get("EVEFL_CHECKPOINT_DIR", REPO_ROOT / "checkpoints"))


# ----------------------------------------------------------------------------------------------------------------------
# the checks
# ----------------------------------------------------------------------------------------------------------------------

def _controller() -> str:
    from evefl.orchestration.state_machine import SecurityState, StateController, StateThresholds

    controller = StateController()
    cases = [(0.0, SecurityState.SECURE), (0.0499, SecurityState.SECURE), (0.05, SecurityState.CAUTION),
             (0.1099, SecurityState.CAUTION), (0.11, SecurityState.LOCKDOWN), (1.0, SecurityState.LOCKDOWN)]
    for qber, expected in cases:
        got = controller.classify(qber)
        assert got == expected, f"QBER {qber} gave {got.value}, expected {expected.value}"
    try:
        StateThresholds(secure_max=0.05, caution_max=0.10)
    except ValueError:
        refused = True
    else:
        refused = False
    assert refused, "a LOCKDOWN threshold below 0.11 was accepted"
    return (f"{len(cases)} boundary cases classified as designed (SECURE below 0.05, CAUTION up to 0.11, LOCKDOWN from 0.11); "
            "a LOCKDOWN threshold of 0.10 was refused, it can never be set below 0.11")


def _bb84() -> str:
    from evefl.quantum.base import ChannelModel
    from evefl.quantum.bb84_numpy import BB84NumpyProtocol

    clean = BB84NumpyProtocol(seed=7).run_exchange(n_qubits=1024, channel=ChannelModel(intercept_probability=0.0, bit_flip_probability=0.0))
    attacked = BB84NumpyProtocol(seed=7).run_exchange(n_qubits=1024, channel=ChannelModel(intercept_probability=1.0, bit_flip_probability=0.0))
    assert clean.qber == 0.0, f"no Eve but QBER {clean.qber}"
    assert 0.10 <= attacked.qber <= 0.40, f"Eve on every photon gave QBER {attacked.qber}, theory says 0.25"
    return (f"no Eve: QBER {clean.qber:.3f}; Eve on every photon: QBER {attacked.qber:.3f} (theory 0.25; a 128-bit sample, so the "
            "estimate is noisy); fixed seed 7")


def _round_key(round_id: int):
    from evefl.quantum.postprocess import RoundKey

    return RoundKey(client_id="0", round_id=round_id, key=bytearray(os.urandom(64)), key_bits=512, qber=0.01, qber_upper=0.02,
                    leaked_bits=0, eps_sec=1e-10, eps_cor=1e-15, transcript_sha256=os.urandom(32))


def _link_protection() -> str:
    from evefl.crypto.classical import ClassicalCipherSuite
    from evefl.crypto.linkkeys import (
        Direction,
        KeyReuseError,
        LinkAuthError,
        LinkKey,
        SealedMessage,
        derive_link_key,
        open_sealed,
        seal,
    )

    suite = ClassicalCipherSuite()
    model_hash, shapes = bytes([0x11]) * 32, bytes([0x22]) * 32

    def pair(round_id: int) -> tuple[LinkKey, LinkKey]:
        rk = _round_key(round_id)
        d = Direction.CLIENT_TO_SERVER
        return (derive_link_key(rk, direction=d, role="client", suite=suite), derive_link_key(rk, direction=d, role="server", suite=suite))

    def seal_update(key: LinkKey, data: bytes) -> SealedMessage:
        return seal(key, data, state="SECURE", model_hash=model_hash, shapes_digest=shapes)

    def open_update(key: LinkKey, message: SealedMessage, state: str = "SECURE") -> bytes:
        return open_sealed(key, message, state=state, model_hash=model_hash, shapes_digest=shapes)

    update = b"model update bytes " * 64
    client, server = pair(1)
    sealed = seal_update(client, update)
    assert open_update(server, sealed) == update, "the opened message differs from the sealed one"
    try:
        open_update(server, sealed)
    except KeyReuseError:
        replay_refused = True
    else:
        replay_refused = False
    assert replay_refused, "a replayed message was accepted"

    client2, server2 = pair(2)
    good = seal_update(client2, update)
    flipped = bytearray(good.ciphertext)
    flipped[0] ^= 1
    tampered = SealedMessage(good.round_id, good.client_id, good.direction, good.nonce, bytes(flipped))
    try:
        open_update(server2, tampered)
    except LinkAuthError:
        tamper_refused = True
    else:
        tamper_refused = False
    assert tamper_refused, "a message with one flipped bit was accepted"

    client3, server3 = pair(3)
    sealed3 = seal_update(client3, update)
    try:
        open_update(server3, sealed3, state="CAUTION")
    except LinkAuthError:
        aad_refused = True
    else:
        aad_refused = False
    assert aad_refused, "a message opened under the wrong associated data"
    return ("an update was sealed with AES-256-GCM, bound to round, client and state, and opened intact; a replay, a flipped bit and "
            "wrong associated data were each refused. The keys here are random test bytes, not simulator keys")


def _secure_aggregation() -> str:
    from evefl.secagg_demo import DemoSettings, run_demo

    result = run_demo(DemoSettings(n_clients=3, seed=0))
    assert result.max_abs_error <= result.tolerance, f"sum error {result.max_abs_error} above tolerance {result.tolerance}"
    assert not (result.masked.astype("float64") == result.weighted).any(), "a masked value equals an unmasked one"
    dropped = run_demo(DemoSettings(n_clients=3, seed=0, drop_client="1"))
    assert dropped.dropout is not None and dropped.dropout["round_aborts"], "a dropout did not abort the round"
    return (f"3 hospitals' masked updates summed to the plain weighted average within fixed-point rounding (error "
            f"{result.max_abs_error:.1e}, allowed {result.tolerance:.1e}); with one hospital missing the masks do not cancel and "
            "the round aborts. A library demonstration, not wired into training")


def _partition_audit() -> str:
    record = json.loads((VALIDATION / "partition_audit_real.json").read_text(encoding="utf-8"))
    assert record["overlaps_all_zero"] is True, "the recorded audit reports an overlap"
    pairs = record["overlaps"]
    assert pairs and all(v["patients"] == 0 and v["images"] == 0 for v in pairs.values()), "a recorded overlap is not zero"
    return (f"recorded audit of the real ChestX-ray14 metadata: {len(pairs)} pairs of hospital, validation and test sets share no "
            "patient and no image")


def _recorded_results() -> str:
    from evefl import results_view as rv

    parts = []
    for name in ("pilot_b0_full_e1_seed0_2026-10-17.json", "pilot_b0_full_e1_seed1_2026-10-24.json"):
        path = VALIDATION / name
        view = rv.load_run(path)
        assert len(view.rounds) == 50, f"{name} has {len(view.rounds)} rounds"
        record = json.loads(path.read_text(encoding="utf-8"))
        primary = record["results"]["primary"]
        parts.append(f"seed {record['run']['seed']}: test AUC {primary['test_auc_excl_thin']:.4f} at round {primary['round']}")
    return "2 recorded single-seed runs load, 50 rounds each (" + "; ".join(parts) + "); measured on Kaggle, no confidence interval"


def _model_check(checkpoint_dir: Path) -> Callable[[], str]:
    def run() -> str:
        from PIL import Image

        from evefl import inference as inf

        found = inf.find_checkpoints(checkpoint_dir)
        if not found:
            raise SkipCheck(f"no checkpoint found in {checkpoint_dir}: copy the .pt files there to enable this check")
        loaded = inf.load_checkpoint(found[0])
        table = inf.predict(loaded, Image.new("L", (300, 300), color=128))
        assert len(table) == 14, f"{len(table)} findings instead of 14"
        assert table["probability"].between(0.0, 1.0).all(), "a score is outside [0, 1]"
        return (f"loaded {loaded.file} ({loaded.kind.replace('_', ' ')}, round {loaded.round}) with weights_only=True and scored a "
                "picture: 14 findings, every score in [0, 1]; a research prototype, not diagnostic")
    return run


def builtin_checks(checkpoint_dir: Optional[Path] = None) -> List[Check]:
    directory = Path(checkpoint_dir) if checkpoint_dir is not None else default_checkpoint_dir()
    return [
        Check("controller", "QBER controller", "Classifies QBER into SECURE, CAUTION or LOCKDOWN and refuses a LOCKDOWN threshold below 0.11.",
              "real code", _controller),
        Check("bb84", "BB84 simulator", "Runs a simulated key exchange with and without an eavesdropper and reads the QBER.",
              "simulated", _bb84),
        Check("link_protection", "Link protection", "Encrypts and decrypts an update with bound associated data, then tries a replay, tampering and wrong data.",
              "real code", _link_protection),
        Check("secure_aggregation", "Secure aggregation", "Masks three hospitals' updates, sums the masked values, and aborts when one hospital is missing.",
              "simulated", _secure_aggregation),
        Check("partition_audit", "Patient-level data split", "Reads the audit of the real metadata: no patient or image appears in two sets.",
              "recorded", _partition_audit),
        Check("recorded_results", "Recorded training runs", "Loads the two single-seed runs on the real data through the same reader the results page uses.",
              "recorded", _recorded_results),
        Check("model", "Trained model", "Loads the saved checkpoint safely and scores a picture.", "real code", _model_check(directory)),
    ]


# ----------------------------------------------------------------------------------------------------------------------
# running
# ----------------------------------------------------------------------------------------------------------------------

def run_check(check: Check) -> CheckResult:
    start = time.perf_counter()
    try:
        detail, status = check.run(), "pass"
    except SkipCheck as reason:
        detail, status = str(reason), "skip"
    except AssertionError as exc:
        detail, status = f"check failed: {exc}", "fail"
    except Exception as exc:  # noqa: BLE001 - a broken check must be reported, not hide the others
        detail, status = f"check raised {type(exc).__name__}: {exc}", "fail"
    return CheckResult(check.name, check.title, check.what, check.kind, status, detail, time.perf_counter() - start)


def selected_checks(only: Optional[Sequence[str]] = None, checkpoint_dir: Optional[Path] = None,
                    extra: Iterable[Check] = ()) -> List[Check]:
    """The built-in checks (all, or those named in `only`) followed by every `extra` check."""
    return [c for c in builtin_checks(checkpoint_dir) if only is None or c.name in only] + list(extra)


def iter_checks(only: Optional[Sequence[str]] = None, checkpoint_dir: Optional[Path] = None,
                extra: Iterable[Check] = ()) -> Iterator[CheckResult]:
    """Run the checks one at a time and yield each result as it finishes (the page shows them appearing)."""
    for check in selected_checks(only, checkpoint_dir, extra):
        yield run_check(check)


def run_checks(only: Optional[Sequence[str]] = None, checkpoint_dir: Optional[Path] = None,
               extra: Iterable[Check] = ()) -> List[CheckResult]:
    """Run the built-in checks (all, or those named in `only`) and then every `extra` check."""
    return list(iter_checks(only, checkpoint_dir, extra))


def warm_up() -> None:
    """Import the heavy libraries once (PyTorch takes about 15 s on the first use) so a later click on the check is quick.

    Safe to call from a background thread; failures are ignored because the checks report their own errors.
    """
    try:
        import qiskit  # noqa: F401  (the QBER explorer page uses the Qiskit reference backend)
        import qiskit_aer  # noqa: F401
        import torch  # noqa: F401
        import torchvision  # noqa: F401

        from evefl import inference, results_view  # noqa: F401
        from evefl.crypto import linkkeys  # noqa: F401
    except Exception:  # noqa: BLE001
        pass


def summarise(results: Sequence[CheckResult]) -> Dict[str, object]:
    failed = sum(r.status == "fail" for r in results)
    return {"total": len(results), "passed": sum(r.status == "pass" for r in results), "failed": failed,
            "skipped": sum(r.status == "skip" for r in results), "healthy": failed == 0,
            "seconds": sum(r.seconds for r in results)}
