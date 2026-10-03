"""
Print a compact, paste-friendly summary of one results JSON (from `python -m evefl.fl.server`).

Shows the experiment config (incl. config hash, pretrained, QKD setting, optimiser) and one line per
round: state, system QBER, QBER sample size, learning rate, whether the model changed, and the validation /
test macro AUC. Use it to paste the outcome of a Kaggle run back into the chat or into the paper notes.

    python scripts/summarize_run.py results/lite_alpha0.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def _auc(entry: Dict[str, Any] | None) -> str:
    if not entry:
        return "   n/a"
    value = entry["metrics"].get("macro_auc_roc")
    return "   n/a" if value is None or value != value else f"{value:6.4f}"


def summarize(report: Dict[str, Any]) -> List[str]:
    exp = report["experiment"]
    lines = [
        f"experiment      : {exp['name']}  (config_hash {exp.get('config_hash')})",
        f"seed / rounds   : {exp['seed']} / {exp['num_rounds']}   clients={exp['num_clients']}  "
        f"local_epochs={exp['local_epochs']}  batch={exp['batch_size']}",
        f"pretrained      : {exp.get('pretrained')}  weights={exp.get('init_weights', {}).get('source')}",
        f"qkd             : {exp.get('qkd')}",
        f"optimizer       : {exp.get('optimizer')}",
        f"state counts    : {exp.get('state_counts')}   failures={exp.get('n_failures')}",
        f"elapsed seconds : {exp.get('elapsed_seconds'):.0f}",
        "",
        "round state     sysQBER  sample  lr        updated  val_AUC  test_AUC",
    ]
    skipped = None
    for r in report["rounds"]:
        sizes = [d.get("qber_sample_size") for d in (r.get("qkd_per_client") or {}).values()]
        sample = f"{min(sizes)}-{max(sizes)}" if sizes and None not in sizes else "n/a"
        lr = r.get("learning_rate")
        lines.append(
            f"{r['round']:>5} {r['state']:<9} {r['system_qber']:7.4f}  {sample:>6}  "
            f"{('%.2e' % lr) if lr else 'n/a':<9} {str(r.get('model_updated')):<7} "
            f"{_auc(r.get('val_eval'))}  {_auc(r.get('server_eval'))}"
        )
        if r.get("server_eval"):
            skipped = r["server_eval"]["metrics"].get("skipped_classes")
    lines.append("")
    lines.append(f"classes skipped in the test AUC (last evaluated round): {skipped}")
    return lines


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    print("\n".join(summarize(report)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
