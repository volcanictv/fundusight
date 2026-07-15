"""Dashboard visual design: small reusable HTML component renderers.

Same pattern already used in app/progress.py (render_skeleton,
render_error_card): plain functions emitting markup via
st.markdown(unsafe_allow_html=True), styled by the CSS classes theme.py
defines (.fdx-ring-card, .fdx-pill, .fdx-datagrid, .fdx-stat-tile,
.fdx-recommendation-card). Each is parameterized rather than bespoke per
metric, so one ring/pill/grid/tile/card implementation covers every
section in main.py.
"""

import html

import streamlit as st

from src.report.content import DISCLAIMER

# Matches theme.py's --fdx-primary -- kept as its own local copy rather than
# a cross-module import, matching how this codebase already keeps each
# module's small brand-color constants local (see report/pdf.py,
# app/charts.py).
_DEFAULT_RING_COLOR = "#3525CD"

# Semantic, not color-named (see theme.py's module docstring): "normal" (no
# finding / calm, indigo), "attention" (a finding is present, tertiary
# orange), "info" (neutral/informational, not a status verdict, sky blue).
# A color-named CSS class mapping to a different color than its name
# implies is a latent bug waiting to happen, so these stay role-based.
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
        f"""<div class="fdx-ring-card">
    <div class="fdx-ring" style="--pct:{pct_clamped:.1f};--ring-color:{html.escape(color)}">
        <div class="fdx-ring-inner">{html.escape(display_value)}</div>
    </div>
    <div class="fdx-ring-label">{html.escape(label)}</div>
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
        f'<span class="fdx-pill fdx-pill-{variant}">{html.escape(text)}</span>',
        unsafe_allow_html=True,
    )


def render_datagrid(rows: list) -> None:
    """A compact label/value table for secondary, detail-level numbers --
    headline metrics stay in st.metric tiles or render_ring()/
    render_stat_tile() cards; this is for the supporting rows (e.g. branch
    count, tortuosity, disc/cup diameters in px). `rows` is a list of
    (label, value) pairs, both coerced to str.

    Wrapped in the same glass card treatment as render_ring()'s
    .fdx-ring-card: its own CSS (.fdx-datagrid tr:last-child td {
    border-bottom: none }) deliberately drops the last row's bottom
    border on the assumption a container's own border supplies the
    table's visual "closing" edge, so the table needs a container to
    avoid trailing off into blank page background after the last row.
    """
    body = "".join(
        f"<tr><td>{html.escape(str(label))}</td><td>{html.escape(str(value))}</td></tr>" for label, value in rows
    )
    st.markdown(
        f'<div class="fdx-datagrid-card"><table class="fdx-datagrid"><tbody>{body}</tbody></table></div>',
        unsafe_allow_html=True,
    )


def render_stat_tile(
    title: str,
    pill_text: str,
    pill_variant: str,
    ring_value: str,
    ring_pct: float,
    ring_color: str = _DEFAULT_RING_COLOR,
    subtitle: str = "",
) -> None:
    """One dense glass card combining a title, a status pill, and a ring
    gauge. Three of these side by side (see app/main.py's Disease
    Screening panel) form one compact, scannable row for DR/glaucoma/AMD.

    `subtitle`, if given, renders as a small muted line under the ring -- used
    for the Monte-Carlo Dropout '± x%' uncertainty (see mc_dropout.py).
    """
    if pill_variant not in _PILL_VARIANTS:
        raise ValueError(f"Unknown pill variant: {pill_variant!r}. Choose from {sorted(_PILL_VARIANTS)}")
    pct_clamped = max(0.0, min(100.0, ring_pct))
    subtitle_html = (
        f'<div style="text-align:center;margin-top:0.4rem;font-size:0.72rem;'
        f'letter-spacing:0.02em;opacity:0.6;font-variant-numeric:tabular-nums;">'
        f"{html.escape(subtitle)}</div>"
        if subtitle
        else ""
    )
    st.markdown(
        f"""<div class="fdx-stat-tile">
    <div class="fdx-stat-tile-header">
        <div class="fdx-stat-tile-title">{html.escape(title)}</div>
        <span class="fdx-pill fdx-pill-{pill_variant}">{html.escape(pill_text)}</span>
    </div>
    <div class="fdx-stat-tile-body">
        <div class="fdx-ring" style="--pct:{pct_clamped:.1f};--ring-color:{html.escape(ring_color)}">
            <div class="fdx-ring-inner">{html.escape(ring_value)}</div>
        </div>
    </div>
    {subtitle_html}
</div>""",
        unsafe_allow_html=True,
    )


def render_recommendation_card(text: str) -> None:
    """report/content.py's synthesized recommendation paragraph (severity
    phrasing across all three detectors + the disclaimer, see
    report/content.py's _build_recommendation()), given its own card here
    rather than a plain st.markdown paragraph since it's the closest thing
    this page has to "the actual conclusion" and deserves to read that way.

    `_build_recommendation()` always appends the shared DISCLAIMER
    constant as its last sentence -- rendering it as one run-on paragraph
    left the disclaimer just trailing off the clinical summary with no
    visual break, undercutting exactly the thing it's meant to stand out
    as (a distinct legal/educational notice, not one more clause of
    findings). Splitting it onto its own muted line inside the same card
    fixes that without changing the underlying text at all.
    """
    summary = text
    disclaimer = ""
    if text.endswith(DISCLAIMER):
        summary = text[: -len(DISCLAIMER)].rstrip()
        disclaimer = DISCLAIMER

    disclaimer_html = f'<div class="fdx-recommendation-disclaimer">{html.escape(disclaimer)}</div>' if disclaimer else ""
    st.markdown(
        f"""<div class="fdx-recommendation-card">
    <div class="fdx-recommendation-title">Recommendation</div>
    <div class="fdx-recommendation-body">{html.escape(summary)}</div>
    {disclaimer_html}
</div>""",
        unsafe_allow_html=True,
    )
