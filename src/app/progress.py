"""Dashboard v2: the loading/progress experience.

Fixes the "did the site crash?" problem the opaque single st.spinner() had:
a sticky banner that stays pinned to the viewport regardless of scroll
position (see theme.py's .vdx-progress-banner CSS), paired with skeleton
placeholders for every result section so the whole results area shows
loading shape immediately instead of staying blank until the entire ~10-30s
pipeline finishes. Also the one place a mid-pipeline exception gets turned
into a readable card instead of a raw traceback -- the same "looks broken"
problem, via a different path.

Renders raw HTML via st.markdown(unsafe_allow_html=True), mirroring the
established pattern in app/components.py's small renderers.
"""

import html
import traceback

import streamlit as st

from src.report.pipeline import STAGE_NAMES

_STAGE_LABELS = {
    "quality": "Checking image quality",
    "preprocessing": "Preprocessing",
    "detection": "Detecting diabetic retinopathy",
    "glaucoma": "Detecting glaucoma signs",
    "amd": "Detecting AMD signs",
    "vessels": "Analyzing vessels",
    "optic_disc": "Analyzing optic disc",
}


class ProgressBanner:
    """One st.empty() placeholder, repeatedly replaced in place -- the
    correct Streamlit primitive for "the same UI element updating over
    time" rather than a new element appended on each call.
    """

    def __init__(self) -> None:
        self._placeholder = st.empty()
        self._render(step=0, label="Starting analysis…")

    def _render(self, step: int, label: str) -> None:
        fraction = step / len(STAGE_NAMES)
        with self._placeholder.container():
            # The spacer reserves the gap in normal document flow; the
            # banner itself is `position: fixed` (see theme.py for why
            # `position: sticky` doesn't work through Streamlit's wrapper
            # divs) and floats over that reserved gap, staying pinned
            # regardless of scroll.
            st.markdown(
                f"""<div class="vdx-progress-banner-spacer"></div>
<div class="vdx-progress-banner">
    <div class="vdx-progress-label">{html.escape(label)}</div>
    <div class="vdx-progress-track">
        <div class="vdx-progress-fill" style="width:{fraction * 100:.1f}%"></div>
    </div>
</div>""",
                unsafe_allow_html=True,
            )

    def advance(self, stage_name: str) -> None:
        step = STAGE_NAMES.index(stage_name) + 1
        self._render(step, f"{_STAGE_LABELS[stage_name]}…")

    def finish(self) -> None:
        self._placeholder.empty()


def render_skeleton(stage_key: str) -> None:
    """A generic shimmering placeholder shape shared by all 5 result
    sections -- each is a metrics row plus one image, so one skeleton
    shape reused everywhere is enough; no need for 5 bespoke ones.
    """
    st.markdown(
        f"""<div class="vdx-skeleton" data-stage="{html.escape(stage_key)}">
    <div class="vdx-skeleton-box vdx-skeleton-title vdx-shimmer"></div>
    <div class="vdx-skeleton-metrics">
        <div class="vdx-skeleton-box vdx-skeleton-pill vdx-shimmer"></div>
        <div class="vdx-skeleton-box vdx-skeleton-pill vdx-shimmer"></div>
        <div class="vdx-skeleton-box vdx-skeleton-pill vdx-shimmer"></div>
    </div>
    <div class="vdx-skeleton-box vdx-skeleton-image vdx-shimmer"></div>
</div>""",
        unsafe_allow_html=True,
    )


def render_error_card(exc: Exception) -> None:
    """A restrained error state instead of a raw Streamlit traceback --
    the full traceback still goes to the server console (not swallowed),
    the user just doesn't see it dumped in the UI.
    """
    traceback.print_exc()
    st.markdown(
        f"""<div class="vdx-error-card">
    <div class="vdx-error-title">Something went wrong while analyzing this image.</div>
    <div class="vdx-error-detail">{html.escape(str(exc))}</div>
</div>""",
        unsafe_allow_html=True,
    )
