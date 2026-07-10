"""Dashboard visual overhaul: small reusable HTML component renderers.

Same pattern already used in app/progress.py (render_skeleton,
render_error_card) and app/render_preview.py (the HTML table renderer):
plain functions emitting markup via st.markdown(unsafe_allow_html=True),
styled by the CSS classes theme.py defines (.vdx-ring-card, .vdx-pill,
.vdx-datagrid). Each is parameterized rather than bespoke per metric, so
one ring/pill/grid implementation covers every section in main.py.
"""

import html

import streamlit as st

# Matches theme.py's _ACCENT / report/pdf.py's _ACCENT_COLOR -- kept as its
# own local copy rather than a cross-module import, matching how this
# codebase already keeps each module's small brand-color constants local
# (see report/pdf.py, app/charts.py).
_DEFAULT_RING_COLOR = "#0071E3"

_PILL_VARIANTS = {"emerald", "amber", "blue"}


def render_ring(label: str, display_value: str, pct: float, color: str = _DEFAULT_RING_COLOR) -> None:
    """A small circular gauge (conic-gradient ring) with a label under it
    and `display_value` printed in the center. `pct` (0-100) drives the
    filled arc -- callers scale their own metric onto that range (e.g. a
    0.0-1.0 ratio becomes `pct=value*100`), keeping this component dumb
    and reusable rather than tied to any one metric's units.
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


def render_pill(text: str, variant: str = "blue") -> None:
    """A small status/severity badge. `variant` picks the color: emerald
    (good/normal), amber (elevated/attention), blue (neutral/informational).
    """
    if variant not in _PILL_VARIANTS:
        raise ValueError(f"Unknown pill variant: {variant!r}. Choose from {sorted(_PILL_VARIANTS)}")
    st.markdown(
        f'<span class="vdx-pill vdx-pill-{variant}">{html.escape(text)}</span>',
        unsafe_allow_html=True,
    )


def render_datagrid(rows: list) -> None:
    """A compact label/value table for secondary, detail-level numbers --
    headline metrics stay in st.metric tiles or render_ring() cards; this
    is for the supporting rows (e.g. branch count, tortuosity, disc/cup
    diameters in px). `rows` is a list of (label, value) pairs, both
    coerced to str.
    """
    body = "".join(
        f"<tr><td>{html.escape(str(label))}</td><td>{html.escape(str(value))}</td></tr>" for label, value in rows
    )
    st.markdown(f'<table class="vdx-datagrid"><tbody>{body}</tbody></table>', unsafe_allow_html=True)
