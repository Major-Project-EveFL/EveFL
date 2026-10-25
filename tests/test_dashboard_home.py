"""
The dashboard's entry point, theme and Mission Control page (`dashboard/app.py`, `dashboard/theme.py`,
`dashboard/pages/0_mission_control.py`).

The home page shows numbers that come from the recorded files, and its live system check really runs the library. Every page must
also tolerate an already-set page config, because `st.set_page_config` may be called once per run and the entry file calls it.
"""

import re
from pathlib import Path

from streamlit.testing.v1 import AppTest

from dashboard import theme
from evefl import showcase

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "dashboard" / "app.py"
HOME = ROOT / "dashboard" / "pages" / "0_mission_control.py"
PAGES = sorted((ROOT / "dashboard" / "pages").glob("[0-9]_*.py"))


def _text(at):
    return " ".join(m.value for m in at.markdown)


def test_the_entry_point_runs_without_an_exception_and_injects_the_theme():
    # AppTest (Streamlit 1.39) does not render the pages behind st.navigation, so the page content is tested on the page file below.
    at = AppTest.from_file(str(APP), default_timeout=180).run()
    assert not at.exception, [e.value for e in at.exception]
    text = _text(at)
    assert "ev-hero" in text and "EveFL" in text


def test_mission_control_shows_the_hero_the_badges_and_the_button():
    at = AppTest.from_file(str(HOME), default_timeout=180).run()
    assert not at.exception, [e.value for e in at.exception]
    text = _text(at)
    assert "RESEARCH PROTOTYPE" in text and "SIMULATED" in text and "MEASURED" in text and "REAL CODE" in text
    assert [b.label for b in at.button] == ["▶  Run live system check"]


def test_the_headline_numbers_on_the_page_are_the_recorded_ones():
    at = AppTest.from_file(str(HOME), default_timeout=180).run()
    h = showcase.headline_numbers()
    text = _text(at)
    assert f"{h['served_model_test_auc']:.3f}" in text
    assert all(f"{r['test_auc_round_50']:.3f}" in text for r in h["runs"])
    assert f"{h['lockdown_bound'] * 100:.0f}%" in text


def test_pressing_the_button_runs_the_real_checks_and_none_of_them_fails():
    at = AppTest.from_file(str(HOME), default_timeout=300).run()
    at.button[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    results = at.session_state["system_check"]
    assert results and len(results) >= 7
    assert all(r.status in ("pass", "skip") for r in results), [(r.name, r.detail) for r in results if r.status == "fail"]
    assert {r.name for r in results if r.status == "pass"} >= {"controller", "bb84", "link_protection", "secure_aggregation"}
    assert "checks passed" in _text(at)


def test_every_page_tolerates_the_page_config_already_being_set():
    assert len(PAGES) >= 8
    for page in PAGES:
        source = page.read_text(encoding="utf-8")
        if "st.set_page_config" in source:
            assert re.search(r"try:\s*\n\s+st\.set_page_config\(.*\)\s*\nexcept st\.errors\.StreamlitAPIException:", source), page.name


def test_the_navigation_lists_every_page_file():
    source = APP.read_text(encoding="utf-8")
    listed = set(re.findall(r'st\.Page\("pages/([^"]+)"', source))
    assert listed == {p.name for p in PAGES}


def test_theme_escapes_text_and_labels_every_kind_of_evidence():
    row = theme.check_row("<b>t</b>", "<script>alert(1)</script>", "fail", 0.01, "real code", what="<i>w</i>")
    assert "<script>" not in row and "&lt;script&gt;" in row and "&lt;b&gt;" in row
    for kind in ("measured", "simulated", "library", "real code", "recorded", "prototype", "stub"):
        assert theme.badge(kind).startswith('<span class="ev-chip"')
    assert set(theme.STATE_COLORS) == {"SECURE", "CAUTION", "LOCKDOWN"}


def test_the_theme_needs_no_network_resource():
    css = theme.CSS
    assert "http://" not in css and "https://" not in css and "@import" not in css and "url(" not in css
