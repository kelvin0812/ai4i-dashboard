"""Visual styling: global CSS + reusable page header / callout.

Colours for the UI chrome live here; data colours live in src/config.py.
"""

import html

import streamlit as st

RED = "#e8173f"
CRIMSON = "#9f1239"
DEEP = "#4c0519"

_CSS = f"""
<style>
/* Sidebar: crimson gradient */
[data-testid="stSidebar"] > div:first-child {{
    background: linear-gradient(180deg, #be123c 0%, {CRIMSON} 45%, {DEEP} 100%);
}}
[data-testid="stSidebar"] [data-testid="stSidebarNavLink"][aria-current="page"] {{
    background: rgba(255, 255, 255, 0.18);
}}
[data-testid="stSidebar"] hr {{ border-color: rgba(255, 255, 255, 0.25); }}
/* Multiselect chips: white pill, crimson text (default would be white on white) */
[data-testid="stSidebar"] [data-tag] {{
    background: #ffffff !important;
}}
[data-testid="stSidebar"] [data-tag] span,
[data-testid="stSidebar"] [data-tag] button {{
    color: {CRIMSON} !important; font-weight: 700;
}}

/* Page header banner */
.pm-hero {{
    background: linear-gradient(120deg, {DEEP} 0%, {CRIMSON} 35%, {RED} 75%, #ff4d6d 100%);
    border-radius: 1rem;
    padding: 1.6rem 1.9rem 1.4rem;
    margin: 0 0 0.6rem 0;
    color: #fff;
    box-shadow: 0 10px 30px -12px rgba(159, 18, 57, 0.55);
    position: relative;
    overflow: hidden;
}}
.pm-hero::after {{
    content: "";
    position: absolute; right: -60px; top: -60px;
    width: 220px; height: 220px; border-radius: 50%;
    background: radial-gradient(circle, rgba(255,255,255,0.22) 0%, rgba(255,255,255,0) 70%);
}}
.pm-hero .pm-eyebrow {{
    font-size: 0.75rem; letter-spacing: 0.14em; text-transform: uppercase;
    opacity: 0.85; font-weight: 600; margin-bottom: 0.25rem;
}}
.pm-hero h1 {{
    color: #fff; font-size: 2.1rem; font-weight: 800; margin: 0; padding: 0;
    line-height: 1.15;
}}
.pm-hero p {{ color: rgba(255,255,255,0.9); margin: 0.45rem 0 0; font-size: 0.98rem; max-width: 52rem; }}

/* Section headings: red accent bar */
[data-testid="stMain"] [data-testid="stHeadingWithActionElements"] h3 {{
    border-left: 5px solid {RED};
    padding-left: 0.65rem;
}}

/* KPI cards */
[data-testid="stMain"] [data-testid="stMetric"] {{
    background: linear-gradient(180deg, #ffffff 0%, #fff5f7 100%);
    border: 1px solid #f5d0d7;
    border-top: 4px solid {RED};
    border-radius: 0.9rem;
    padding: 1rem 1.2rem;
    box-shadow: 0 6px 18px -10px rgba(159, 18, 57, 0.35);
}}
[data-testid="stMain"] [data-testid="stMetricLabel"] p {{
    text-transform: uppercase; letter-spacing: 0.06em; font-size: 0.78rem;
    font-weight: 600; color: {CRIMSON};
}}
[data-testid="stMain"] [data-testid="stMetricValue"] {{ font-weight: 800; color: #1f1215; }}

/* Callout */
.pm-callout {{
    border: 1px solid #f5d0d7; border-left: 5px solid {RED};
    background: #fff1f3; border-radius: 0.75rem; padding: 0.9rem 1.1rem; color: #1f1215;
}}
.pm-callout b {{ color: {CRIMSON}; }}
</style>
"""


def apply_theme() -> None:
    """Inject the global CSS (call once per run, from app.py)."""
    st.html(_CSS)


def page_header(title: str, subtitle: str = "", eyebrow: str = "AI4I 2020 · Predictive maintenance") -> None:
    sub = f"<p>{html.escape(subtitle)}</p>" if subtitle else ""
    st.html(
        f'<div class="pm-hero"><div class="pm-eyebrow">{html.escape(eyebrow)}</div>'
        f"<h1>{html.escape(title)}</h1>{sub}</div>"
    )


def callout(html_text: str) -> None:
    """Red-accented note box. `html_text` is trusted, app-authored HTML."""
    st.html(f'<div class="pm-callout">{html_text}</div>')
