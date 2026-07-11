"""Dashboard visual design: small reusable HTML component renderers.

Same pattern already used in app/progress.py (render_skeleton,
render_error_card): plain functions emitting markup via
st.markdown(unsafe_allow_html=True), styled by the CSS classes theme.py
defines (.vdx-ring-card, .vdx-pill, .vdx-datagrid, .vdx-stat-tile,
.vdx-recommendation-card). Each is parameterized rather than bespoke per
metric, so one ring/pill/grid/tile/card implementation covers every
section in main.py.
"""

import html

import streamlit as st

# Matches theme.py's --vdx-teal -- kept as its own local copy rather than a
# cross-module import, matching how this codebase already keeps each
# module's small brand-color constants local (see report/pdf.py,
# app/charts.py).
_DEFAULT_RING_COLOR = "#0E7C86"

# Semantic, not color-named (see theme.py's module docstring): "normal" (no
# finding / calm, teal), "attention" (a finding is present, copper), "info"
# (neutral/informational, not a status verdict, muted slate). Renamed from
# the old "emerald"/"amber"/"blue" when the underlying colors were re-hued
# for the glass redesign -- a color-named CSS class mapping to a different
# color than its name implies is a latent bug waiting to happen.
_PILL_VARIANTS = {"normal", "attention", "info"}


def render_ring(label: str, display_value: str, pct: float, color: str = _DEFAULT_RING_COLOR) -> None:
    """A small circular gauge (conic-gradient ring, instrument-bezel style
    -- see theme.py) with a label under it and `display_value` printed in
    the center. `pct` (0-100) drives the filled arc -- callers scale their
    own metric onto that range (e.g. a 0.0-1.0 ratio becomes
    `pct=value*100`), keeping this component dumb and reusable rather than
    tied to any one metric's units.
    """
    pct_clamped = max(0.0, min(100.0, pct))
    st.markdown(
        f"""<div class="vdx-ring-card">
    <div class="vdx-ring" style="--pct:{pct_clamped:.1f};--ring-color:{html.escape(color)}">
        <div class="vdx-ring-inner">{html.escape(display_value)}</div>
    </div>
    <div class="vdx-ring-label">{html.escape(label)}</div>
</div>""",
        unsafe_allow_html=True,
    )


def render_pill(text: str, variant: str = "info") -> None:
    """A small status/severity badge. `variant` picks the color: "normal"
    (no finding / calm), "attention" (a finding is present), "info"
    (neutral/informational).
    """
    if variant not in _PILL_VARIANTS:
        raise ValueError(f"Unknown pill variant: {variant!r}. Choose from {sorted(_PILL_VARIANTS)}")
    st.markdown(
        f'<span class="vdx-pill vdx-pill-{variant}">{html.escape(text)}</span>',
        unsafe_allow_html=True,
    )


def render_datagrid(rows: list) -> None:
    """A compact label/value table for secondary, detail-level numbers --
    headline metrics stay in st.metric tiles or render_ring()/
    render_stat_tile() cards; this is for the supporting rows (e.g. branch
    count, tortuosity, disc/cup diameters in px). `rows` is a list of
    (label, value) pairs, both coerced to str.
    """
    body = "".join(
        f"<tr><td>{html.escape(str(label))}</td><td>{html.escape(str(value))}</td></tr>" for label, value in rows
    )
    st.markdown(f'<table class="vdx-datagrid"><tbody>{body}</tbody></table>', unsafe_allow_html=True)


def render_stat_tile(
    title: str,
    pill_text: str,
    pill_variant: str,
    ring_value: str,
    ring_pct: float,
    ring_color: str = _DEFAULT_RING_COLOR,
) -> None:
    """One dense glass card combining a title, a status pill, and a ring
    gauge -- the direct fix for the old per-disease layout (a full
    subheader + a separate pill + a separate 2-column ring/datagrid block,
    repeated three times for DR/glaucoma/AMD with only the numbers
    changing). Three of these side by side (see app/main.py's Disease
    Screening panel) replace that with one compact, scannable row.
    """
    if pill_variant not in _PILL_VARIANTS:
        raise ValueError(f"Unknown pill variant: {pill_variant!r}. Choose from {sorted(_PILL_VARIANTS)}")
    pct_clamped = max(0.0, min(100.0, ring_pct))
    st.markdown(
        f"""<div class="vdx-stat-tile">
    <div class="vdx-stat-tile-header">
        <div class="vdx-stat-tile-title">{html.escape(title)}</div>
        <span class="vdx-pill vdx-pill-{pill_variant}">{html.escape(pill_text)}</span>
    </div>
    <div class="vdx-stat-tile-body">
        <div class="vdx-ring" style="--pct:{pct_clamped:.1f};--ring-color:{html.escape(ring_color)}">
            <div class="vdx-ring-inner">{html.escape(ring_value)}</div>
        </div>
    </div>
</div>""",
        unsafe_allow_html=True,
    )


def render_recommendation_card(text: str) -> None:
    """report/content.py's synthesized recommendation paragraph (severity
    phrasing across all three detectors + the disclaimer, see
    report/content.py's _build_recommendation()) is the one piece of the
    old page-length Report Preview walk that wasn't already shown
    elsewhere on the dashboard (see app/main.py's module docstring for the
    rest of that story) -- given its own card here rather than a plain
    st.markdown paragraph, since it's the closest thing this page has to
    "the actual conclusion" and deserves to read that way.
    """
    st.markdown(
        f"""<div class="vdx-recommendation-card">
    <div class="vdx-recommendation-title">Recommendation</div>
    <div class="vdx-recommendation-body">{html.escape(text)}</div>
</div>""",
        unsafe_allow_html=True,
    )
