"""Phase 9: dashboard visual theme.

One CSS injection point for the "Apple-like" design ethos requested for
this app: generous whitespace, a restrained near-monochrome palette with a
single accent color, high-contrast typography, and motion limited to short
hover/expand transitions rather than anything decorative. Streamlit has no
first-class theming API expressive enough for this (its config.toml theme
only covers a handful of colors), so this injects scoped CSS instead --
targeting Streamlit's `data-testid` attributes rather than its generated
class names, since those are the one part of its DOM that's meant to be a
stable styling hook across versions.
"""

import streamlit as st

# Same accent as report/pdf.py's _ACCENT_COLOR -- keeps the in-app preview
# and the exported PDF visually consistent, not two different brands.
_ACCENT = "#0071E3"
_TEXT = "#1D1D1F"
_MUTED = "#6E6E73"
_RULE = "#D2D2D7"
_BACKGROUND = "#FBFBFD"

_CSS = f"""
<style>
:root {{
    --vdx-accent: {_ACCENT};
    --vdx-text: {_TEXT};
    --vdx-muted: {_MUTED};
    --vdx-rule: {_RULE};
}}

.stApp {{
    background-color: {_BACKGROUND};
}}

/* System font stack instead of Streamlit's default -- matches the native
   feel of Apple platform UI rather than looking like a generic web app. */
html, body, [class*="css"] {{
    font-family: -apple-system, "SF Pro Text", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    color: var(--vdx-text);
}}

h1, h2, h3 {{
    font-weight: 600;
    letter-spacing: -0.01em;
    color: var(--vdx-text);
}}

/* Generous whitespace: widen the default gutters Streamlit's block
   container ships with rather than letting content edge-to-edge. */
.block-container {{
    padding-top: 2.5rem;
    padding-bottom: 3rem;
    max-width: 880px;
}}

hr {{
    border: none;
    border-top: 1px solid var(--vdx-rule);
    margin: 1.75rem 0;
}}

div[data-testid="stMetric"] {{
    background-color: white;
    border: 1px solid var(--vdx-rule);
    border-radius: 12px;
    padding: 0.75rem 1rem;
}}

div[data-testid="stMetricValue"] {{
    color: var(--vdx-text);
}}

/* Buttons: restrained accent fill, subtle motion on hover only -- no
   animation on page load or idle state ("purposeful", not decorative). */
.stButton > button, .stDownloadButton > button {{
    background-color: var(--vdx-accent);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.5rem 1.25rem;
    font-weight: 500;
    transition: opacity 0.15s ease;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
    opacity: 0.85;
    color: white;
}}

div[data-testid="stExpander"] {{
    border: 1px solid var(--vdx-rule);
    border-radius: 12px;
    transition: border-color 0.15s ease;
}}

.vdx-caption {{
    color: var(--vdx-muted);
    font-size: 0.85rem;
    margin-top: 0.25rem;
}}

.vdx-disclaimer {{
    color: var(--vdx-muted);
    font-size: 0.8rem;
    border-top: 1px solid var(--vdx-rule);
    padding-top: 0.75rem;
    margin-top: 2rem;
}}

/* Print support: pressing Ctrl+P on the live preview should print just
   the report content, not Streamlit's own chrome. The PDF download is
   still the primary/reliable print path (see report/pdf.py) -- this is a
   convenience for the live-preview screen itself. */
@media print {{
    [data-testid="stSidebar"],
    [data-testid="stHeader"],
    [data-testid="stToolbar"],
    [data-testid="stFileUploaderDropzone"],
    .stButton,
    .stDownloadButton {{
        display: none !important;
    }}
    .block-container {{
        max-width: 100% !important;
    }}
}}
</style>
"""


def inject_css() -> None:
    """Call once near the top of the page, before any other widgets."""
    st.markdown(_CSS, unsafe_allow_html=True)
