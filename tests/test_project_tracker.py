"""docs/PROJECT_TRACKER.md: statuses are from a fixed vocabulary, the summary counts match the rows, and every cited path exists."""

import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "PROJECT_TRACKER.md"
STATUSES = ("DONE", "PARTIAL", "TODO", "WAITING", "STRETCH")
ROW = re.compile(r"^\| (\d+\.\d+) \| (.+?) \| (DONE|PARTIAL|TODO|WAITING|STRETCH) \|", flags=re.MULTILINE)
PATH_TOKEN = re.compile(r"^(docs|evefl|scripts|tests|dashboard|\.github)/[\w./-]+\.(py|md|json|yaml|yml|tex)$")


def _text():
    return DOC.read_text(encoding="utf-8")


def test_every_row_has_a_status_from_the_vocabulary_and_ids_are_unique():
    text = _text()
    rows = ROW.findall(text)
    all_ids = re.findall(r"^\| (\d+\.\d+) \|", text, flags=re.MULTILINE)
    assert len(rows) == len(all_ids) >= 50          # no row with a malformed status
    assert len({r[0] for r in rows}) == len(rows)


def test_summary_counts_equal_the_rows():
    text = _text()
    counts = Counter(r[2] for r in ROW.findall(text))
    line = re.search(r"^Counts: (.+)$", text, flags=re.MULTILINE).group(1)
    expected = ", ".join(f"{s} {counts.get(s, 0)}" for s in STATUSES) + f" ({sum(counts.values())} items)"
    assert line == expected


def test_every_backticked_repository_path_exists():
    missing = [t for t in re.findall(r"`([^`\n]+)`", _text()) if PATH_TOKEN.match(t) and not (ROOT / t).exists()]
    assert missing == []


def test_every_relative_link_resolves_and_nothing_overclaims():
    text = _text()
    broken = [t for t in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", text)
              if not re.match(r"^[a-z]+://", t) and not (DOC.parent / t).exists()]
    assert broken == []
    assert "production ready" not in text.lower()
    assert "nothing with their own images" in text   # the statement about what a hospital can do today stays
