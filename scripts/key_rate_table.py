"""
Key yield of the post-processing pipeline versus block size, on SIMULATED exchanges.

For each block size and channel it runs the full pipeline (parameter estimation, Cascade, verification hash,
finite-key privacy amplification, authentication) at the default eps_sec = 1e-10, eps_cor = 1e-15 and reports the
secret key length, key bits per qubit sent, the Cascade efficiency (leaked bits / (n h(Q))) and the run time.
It shows the block size a key actually needs (the headline controller setting of 1024 qubits yields no key).

    python scripts/key_rate_table.py

Writes docs/validation/key_rates.json. All numbers are simulated; the keys are not secret (simulation seed).
"""

from __future__ import annotations

import datetime
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evefl.quantum.base import ChannelModel  # noqa: E402
from evefl.quantum.bb84_numpy import BB84NumpyProtocol  # noqa: E402
from evefl.quantum.postprocess import DEFAULT_EPS_COR, DEFAULT_EPS_SEC, postprocess  # noqa: E402

LOG2_SIZES = (10, 12, 13, 14, 15, 16, 17, 18, 19, 20)
CHANNELS = {"alpha=0,e=1%": ChannelModel(0.0, 0.01), "alpha=0.1,e=0": ChannelModel(0.1, 0.0),
            "alpha=0.2,e=1%": ChannelModel(0.2, 0.01)}
SAMPLE_FRACTION = 0.1
TRIALS = 3
AUTH_KEY = b"\x01" * 32  # simulation-only pre-shared key


def main() -> int:
    rows = []
    for name, channel in CHANNELS.items():
        for log2n in LOG2_SIZES:
            n = 2 ** log2n
            outcomes = []
            for trial in range(TRIALS):
                result = BB84NumpyProtocol(sample_fraction=SAMPLE_FRACTION, seed=trial).run_exchange(n, channel=channel)
                started = time.time()
                out = postprocess(result, client_id="0", round_id=trial, auth_key=AUTH_KEY, ec_seed=trial)
                outcomes.append((out, time.time() - started))
            ok = [o for o, _ in outcomes if o.ok]
            row = {
                "channel": name, "n_qubits": n, "sample_fraction": SAMPLE_FRACTION, "trials": TRIALS,
                "keys_produced": len(ok),
                "statuses": sorted({o.status for o, _ in outcomes}),
                "key_bits_mean": sum(o.round_key.key_bits for o in ok) / len(ok) if ok else 0,
                "key_bits_per_qubit": (sum(o.round_key.key_bits for o in ok) / len(ok) / n) if ok else 0.0,
                "cascade_efficiency_mean": (sum(o.details["ec_efficiency"] for o in ok if "ec_efficiency" in o.details)
                                            / max(1, sum("ec_efficiency" in o.details for o in ok))) if ok else None,
                "seconds_mean": sum(t for _, t in outcomes) / TRIALS,
            }
            rows.append(row)
            print(f"{name:<16} n=2^{log2n:<2} keys={row['keys_produced']}/{TRIALS} "
                  f"bits={row['key_bits_mean']:>10.0f} per_qubit={row['key_bits_per_qubit']:.3f} "
                  f"status={','.join(row['statuses'])} ({row['seconds_mean']:.1f}s)", flush=True)

    report = {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "label": "simulated; keys are not secret (simulation seed)",
        "eps_sec": DEFAULT_EPS_SEC, "eps_cor": DEFAULT_EPS_COR, "rows": rows,
    }
    target = REPO_ROOT / "docs" / "validation" / "key_rates.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"Wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
