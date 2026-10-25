"""
One look for every dashboard page: dark theme, state colours, honesty badges.

Everything is inline CSS and HTML (no web fonts, no CDN), so the dashboard looks the same offline. Colours carry meaning:
green = SECURE, amber = CAUTION, red = LOCKDOWN, cyan = the quantum link. The badge on a number says what kind of evidence it is.
"""

from __future__ import annotations

import html
from typing import Dict, Iterable, Sequence

import streamlit as st

STATE_COLORS: Dict[str, str] = {"SECURE": "#34d399", "CAUTION": "#fbbf24", "LOCKDOWN": "#f87171"}

# kind -> (text, colour). The wording is part of the honesty rule: a viewer must see how each number was obtained.
BADGES: Dict[str, tuple] = {
    "measured": ("MEASURED", "#60a5fa"),
    "simulated": ("SIMULATED", "#c084fc"),
    "library": ("LIBRARY, NOT IN TRAINING", "#94a3b8"),
    "real code": ("REAL CODE", "#34d399"),
    "recorded": ("RECORDED", "#60a5fa"),
    "prototype": ("RESEARCH PROTOTYPE", "#fbbf24"),
    "stub": ("STUB", "#94a3b8"),
}

CSS = """
<style>
:root { --bg:#0b1020; --panel:#121a30; --panel2:#17213d; --line:#25314f; --text:#e6ecff; --muted:#93a1c4;
        --cyan:#22d3ee; --violet:#a78bfa; --green:#34d399; --amber:#fbbf24; --red:#f87171; }
.block-container { padding-top: 2.2rem; max-width: 1180px; }
h1, h2, h3 { letter-spacing: -0.01em; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0e1530 0%, #0b1020 100%); border-right: 1px solid var(--line); }
.ev-brand { display:flex; align-items:center; gap:.6rem; padding:.2rem 0 .8rem 0; }
.ev-brand .logo { width:34px; height:34px; border-radius:10px; display:grid; place-items:center; font-size:19px;
        background: linear-gradient(135deg, var(--cyan), var(--violet)); }
.ev-brand .name { font-weight:800; font-size:1.15rem; line-height:1.1; }
.ev-brand .sub { color:var(--muted); font-size:.72rem; }
.ev-hero { border:1px solid var(--line); border-radius:20px; padding:2.1rem 2.3rem; margin-bottom:1.2rem;
        background: radial-gradient(900px 300px at 85% -20%, rgba(34,211,238,.22), transparent 60%),
                    radial-gradient(700px 300px at 0% 120%, rgba(167,139,250,.20), transparent 60%), var(--panel); }
.ev-hero h1 { margin:0 0 .4rem 0; font-size:2.5rem; font-weight:800; }
.ev-hero .lead { color:#c8d3f2; font-size:1.12rem; max-width:46rem; margin:0 0 .9rem 0; }
.ev-chip { display:inline-block; font-size:.72rem; font-weight:700; letter-spacing:.04em; padding:.18rem .55rem; border-radius:999px;
        border:1px solid currentColor; margin:0 .35rem .3rem 0; white-space:nowrap; }
.ev-grid { display:grid; gap:.9rem; margin:.4rem 0 1.2rem 0; }
.ev-grid.c4 { grid-template-columns: repeat(4, minmax(0,1fr)); } .ev-grid.c3 { grid-template-columns: repeat(3, minmax(0,1fr)); }
.ev-grid.c2 { grid-template-columns: repeat(2, minmax(0,1fr)); }
@media (max-width: 900px) { .ev-grid.c4, .ev-grid.c3 { grid-template-columns: repeat(2, minmax(0,1fr)); } }
@media (max-width: 560px) { .ev-grid.c4, .ev-grid.c3, .ev-grid.c2 { grid-template-columns: 1fr; } }
.ev-card { background:var(--panel); border:1px solid var(--line); border-radius:16px; padding:1.05rem 1.15rem; }
.ev-card:hover { border-color:#34456e; }
.ev-kpi .label { color:var(--muted); font-size:.78rem; text-transform:uppercase; letter-spacing:.06em; margin-bottom:.25rem; }
.ev-kpi .value { font-size:2.2rem; font-weight:800; line-height:1.05; background: linear-gradient(90deg, var(--cyan), var(--violet));
        -webkit-background-clip:text; background-clip:text; color:transparent; }
.ev-kpi .note { color:#b4c0e0; font-size:.84rem; margin-top:.35rem; }
.ev-step .num { width:30px; height:30px; border-radius:50%; display:grid; place-items:center; font-weight:800; margin-bottom:.5rem;
        background: rgba(34,211,238,.14); color:var(--cyan); border:1px solid rgba(34,211,238,.45); }
.ev-step h4 { margin:.1rem 0 .3rem 0; font-size:1.02rem; } .ev-step p { margin:0; color:#b4c0e0; font-size:.88rem; }
.ev-section { margin:1.5rem 0 .3rem 0; } .ev-section h2 { margin:0; font-size:1.45rem; } .ev-section p { margin:.15rem 0 0 0; color:var(--muted); }
.ev-check { display:grid; grid-template-columns: 2.1rem 1fr auto; gap:.2rem .8rem; align-items:start; padding:.7rem .9rem;
        border:1px solid var(--line); border-radius:12px; background:var(--panel); margin-bottom:.5rem; }
.ev-check .icon { font-size:1.25rem; line-height:1.4; } .ev-check .title { font-weight:700; }
.ev-check .what { color:var(--muted); font-size:.8rem; margin-top:.05rem; }
.ev-check .detail { color:#b4c0e0; font-size:.85rem; margin-top:.1rem; } .ev-check .time { color:var(--muted); font-size:.78rem; white-space:nowrap; }
.ev-check.pass { border-left:4px solid var(--green); } .ev-check.fail { border-left:4px solid var(--red); } .ev-check.skip { border-left:4px solid var(--amber); }
.ev-banner { border-radius:14px; padding:.85rem 1.1rem; margin:.4rem 0 1rem 0; font-weight:600; border:1px solid; }
.ev-banner.ok { color:var(--green); background:rgba(52,211,153,.08); } .ev-banner.bad { color:var(--red); background:rgba(248,113,113,.08); }
.ev-banner.warn { color:var(--amber); background:rgba(251,191,36,.08); }
.ev-honest { color:var(--muted); font-size:.82rem; border-top:1px solid var(--line); padding-top:.7rem; margin-top:1.4rem; }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def badge(kind: str) -> str:
    text, colour = BADGES[kind]
    return f'<span class="ev-chip" style="color:{colour}">{html.escape(text)}</span>'


def sidebar_brand() -> None:
    st.sidebar.markdown(
        '<div class="ev-brand"><div class="logo">🛰️</div><div><div class="name">EveFL</div>'
        '<div class="sub">quantum channel-aware federated learning</div></div></div>', unsafe_allow_html=True)


def section(title: str, sub: str = "") -> None:
    st.markdown(f'<div class="ev-section"><h2>{html.escape(title)}</h2>' + (f"<p>{html.escape(sub)}</p>" if sub else "") + "</div>",
                unsafe_allow_html=True)


def hero(title: str, lead: str, chips: Sequence[str]) -> None:
    st.markdown(f'<div class="ev-hero"><h1>{html.escape(title)}</h1><p class="lead">{html.escape(lead)}</p>'
                + "".join(badge(c) for c in chips) + "</div>", unsafe_allow_html=True)


def kpi_row(items: Iterable[Dict[str, str]]) -> None:
    """Each item: label, value, note, kind (a BADGES key). Values and notes are text, escaped here."""
    cards = []
    for item in items:
        cards.append(f'<div class="ev-card ev-kpi"><div class="label">{html.escape(item["label"])}</div>'
                     f'<div class="value">{html.escape(item["value"])}</div><div class="note">{html.escape(item["note"])}</div>'
                     f'<div style="margin-top:.5rem">{badge(item["kind"])}</div></div>')
    st.markdown(f'<div class="ev-grid c4">{"".join(cards)}</div>', unsafe_allow_html=True)


def step_cards(steps: Sequence[Dict[str, str]]) -> None:
    cards = [f'<div class="ev-card ev-step"><div class="num">{i}</div><h4>{html.escape(s["title"])}</h4><p>{html.escape(s["text"])}</p></div>'
             for i, s in enumerate(steps, 1)]
    st.markdown(f'<div class="ev-grid c4">{"".join(cards)}</div>', unsafe_allow_html=True)


def check_row(title: str, detail: str, status: str, seconds: float, kind: str, what: str = "") -> str:
    icon = {"pass": "✅", "fail": "❌", "skip": "⚠️"}[status]
    what_html = f'<div class="what">{html.escape(what)}</div>' if what else ""
    return (f'<div class="ev-check {status}"><div class="icon">{icon}</div><div><div class="title">{html.escape(title)} {badge(kind)}</div>'
            f'{what_html}<div class="detail">{html.escape(detail)}</div></div><div class="time">{seconds * 1000:.0f} ms</div></div>')


def banner(text: str, level: str) -> None:
    st.markdown(f'<div class="ev-banner {level}">{html.escape(text)}</div>', unsafe_allow_html=True)


def honest_footer() -> None:
    st.markdown('<div class="ev-honest">Research prototype for a final-year engineering project. The quantum layer is a simulator '
                '(its keys are not secret), the classifier is not diagnostic, and every number on this page is labelled measured, '
                'simulated, recorded or real code. See docs/DECISIONS.md for the evidence behind each one.</div>', unsafe_allow_html=True)


def page_link(target: str, label: str, icon: str) -> None:
    """A link to another dashboard page; plain text if the page is not registered (a page opened on its own, or a renamed page)."""
    try:
        st.page_link(target, label=label, icon=icon)
    except st.errors.StreamlitPageNotFoundError:
        st.markdown(f"{icon} {html.escape(label)}")
