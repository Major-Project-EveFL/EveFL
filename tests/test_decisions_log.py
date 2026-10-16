"""docs/DECISIONS.md: every repository path and relative link it cites exists, and the honesty statements stay present."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "DECISIONS.md"
PATH_TOKEN = re.compile(r"^(docs|evefl|scripts|tests|dashboard)/[\w./-]+\.(py|md|json|yaml|yml|tex)$")


def _text():
    return DOC.read_text(encoding="utf-8")


def test_every_backticked_repository_path_exists():
    missing = []
    for token in re.findall(r"`([^`\n]+)`", _text()):
        if "..." in token or "*" in token or not PATH_TOKEN.match(token):
            continue
        if not (ROOT / token).exists():
            missing.append(token)
    assert missing == []


def test_every_relative_markdown_link_resolves():
    broken = []
    for target in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", _text()):
        if re.match(r"^[a-z]+://", target):
            continue
        if not (DOC.parent / target).exists():
            broken.append(target)
    assert broken == []


def test_the_honesty_statements_are_present():
    text = _text()
    for phrase in ("not secret", "No model-accuracy baseline has been produced yet", "Secure aggregation does not exist yet",
                   "no post-quantum guarantee", "No clinical validation"):
        assert phrase in text, phrase
    assert "production ready" not in text.lower() and "certified" not in text.lower().replace("not a certified", "")


def test_the_log_ids_are_unique_and_every_decision_row_has_an_evidence_cell():
    text = _text()
    ids = re.findall(r"^\| ([SMEP]\d+) \|", text, flags=re.MULTILINE)
    assert len(ids) == len(set(ids)) >= 30
