"""README.md and docs/DEMO.md: cited paths and links exist, the not-wired and not-diagnostic statements stay, no forbidden claims."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
DEMO = ROOT / "docs" / "DEMO.md"
PATH_TOKEN = re.compile(r"^(docs|evefl|scripts|tests|dashboard)/[\w./-]+\.(py|md|json|yaml|yml)$")
FORBIDDEN = ("production ready", "production-ready", "fips compliant", "information-theoretically secure end to end")


def _text(path):
    return path.read_text(encoding="utf-8")


def test_every_backticked_repository_path_exists():
    for doc in (README, DEMO):
        missing = [t for t in re.findall(r"`([^`\n]+)`", _text(doc))
                   if "..." not in t and "*" not in t and PATH_TOKEN.match(t) and not (ROOT / t).exists()]
        assert missing == [], (doc.name, missing)


def test_every_relative_markdown_link_resolves():
    for doc in (README, DEMO):
        broken = [t for t in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", _text(doc))
                  if not re.match(r"^[a-z]+://", t) and not (doc.parent / t).exists()]
        assert broken == [], (doc.name, broken)


def test_the_readme_keeps_its_honesty_statements_and_makes_no_forbidden_claim():
    text = _text(README)
    for phrase in ("Research prototype", "not diagnostic", "not wired into the FL path", "simulator", "not secret",
                   "no confidence interval", "never configurable below 0.11"):
        assert phrase in text, phrase
    assert not any(bad in text.lower() for bad in FORBIDDEN)
    assert "AES-256-GCM encrypted gradients" not in text            # the link encryption is not in the FL path


def test_the_demo_guide_lists_what_not_to_say_and_labels_the_numbers():
    text = _text(DEMO)
    section = text.split("## Never say", 1)[1].split("##", 1)[0].lower()
    for named in ("production ready", "certified", "fips compliant", "information-theoretically secure end to end", "diagnoses",
                  "projected", "secure aggregation", "statistically"):
        assert named in section, named
    before_section = text.split("## Never say", 1)[0].lower()
    assert not any(bad in before_section for bad in FORBIDDEN)
    assert "**measured**" in text and "**simulated**" in text and "not diagnostic" in text
