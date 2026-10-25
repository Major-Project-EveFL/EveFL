"""
The demo web app's static files (`evefl/webapp/static/`): no network resource, no unsafe markup injection, every import resolves.

The app is shown on a laptop that may be offline, and its text comes from an API, so a CDN link or an `innerHTML` of API data would
each be a defect: one breaks the page without internet, the other lets data become markup.
"""

import re
from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "evefl" / "webapp" / "static"
FILES = [p for p in STATIC.rglob("*") if p.is_file() and p.suffix in (".js", ".css", ".html", ".svg")]
JS = [p for p in FILES if p.suffix == ".js"]


def test_the_static_folder_has_the_shell_and_every_page_module():
    names = {p.relative_to(STATIC).as_posix() for p in FILES}
    assert {"index.html", "css/app.css", "js/app.js", "js/lib/dom.js", "js/pages/home.js", "favicon.svg"} <= names


def test_nothing_is_loaded_from_the_network():
    allowed = ("http://www.w3.org/2000/svg",)           # an XML namespace name, not a request
    for path in FILES:
        text = path.read_text(encoding="utf-8")
        urls = [u for u in re.findall(r"https?://[^\s\"')]+", text) if not u.startswith(allowed)]
        assert urls == [], (path.name, urls)
        assert "@import" not in text and "//cdn" not in text, path.name


def test_no_script_injects_markup_or_evaluates_strings():
    banned = (r"\.innerHTML\s*[+]?=", r"\.outerHTML\s*=", r"insertAdjacentHTML", r"document\.write", r"\beval\(", r"new Function\(", r"setTimeout\(\s*[\"']")
    for path in JS:
        text = path.read_text(encoding="utf-8")
        for pattern in banned:
            assert not re.search(pattern, text), (path.name, pattern)


def test_the_page_has_no_inline_script_or_inline_event_handler():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert not re.search(r"<script(?![^>]*\bsrc=)", html)
    assert not re.search(r"\son[a-z]+\s*=", html)


def test_every_relative_import_resolves_to_a_file():
    for path in JS:
        text = path.read_text(encoding="utf-8")
        for target in re.findall(r"""(?:from\s+|import\(\s*)["'](\.{1,2}/[^"']+)["']""", text):
            assert (path.parent / target).resolve().is_file(), (path.name, target)


def test_every_page_registered_in_the_router_exists_and_exports_render():
    app = (STATIC / "js" / "app.js").read_text(encoding="utf-8")
    pages = re.findall(r'import\("\./pages/([a-z_]+\.js)"\)', app)
    assert pages, "the router registers no page"
    for name in pages:
        source = (STATIC / "js" / "pages" / name).read_text(encoding="utf-8")
        assert re.search(r"export\s+(async\s+)?function\s+render\s*\(", source), name


def test_the_honest_wording_is_on_the_home_page():
    text = (STATIC / "js" / "pages" / "home.js").read_text(encoding="utf-8")
    for phrase in ("not diagnostic", "simulator", "not secret", "no confidence interval"):
        assert phrase in text, phrase
    assert "production ready" not in text.lower() and "certified" not in text.lower()
