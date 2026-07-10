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

/* --- v2: loading/progress experience ---------------------------------
   Fixes the "did the site crash?" problem an opaque, unpinned spinner
   had: this banner stays visible regardless of scroll position.

   `position: sticky` was tried first and does NOT work here -- verified
   live: it requires being a DIRECT child of the tall block providing its
   "roaming range" (confirmed empirically -- even one plain, unstyled
   wrapper div between a sticky element and its tall ancestor breaks it in
   this Chromium build). Streamlit always wraps `st.markdown()` output in
   several of its own layers (stMarkdownContainer > ... > stElementContainer
   > stVerticalBlock), so a sticky element rendered through a normal
   Streamlit call can never be a direct child of anything tall enough.

   `position: fixed` sidesteps that (anchored to the viewport, independent
   of any ancestor's height), but has its OWN headless-Chromium-specific
   gotcha, also verified live: a fixed element spanning the full viewport
   width (`left: 0; right: 0`, or any explicit width equal to the
   viewport) silently fails to paint its TEXT content in this environment
   (background/border still render -- only text disappears) -- reproduced
   with zero Streamlit involvement, isolated down to element width alone
   (a ~840px-wide fixed box renders text fine; a 1440px/100vw one doesn't,
   same content, same everything else). Centering it as a fixed-width
   floating card via `left: 50%; transform: translateX(-50%)` rather than
   `left/right: 0` avoids the bug entirely -- and reads as more
   "restrained/Apple-like" than an edge-to-edge bar anyway.

   `top: 76px` clears Streamlit's own header toolbar (measured live:
   [data-testid="stHeader"], 60px tall, z-index 999990, position: absolute
   -- stays visually fixed in place too) plus a small gap. A spacer
   (.vdx-progress-banner-spacer, rendered in normal flow right before this)
   reserves room so real content isn't heavily covered when the banner
   first appears -- some overlap is fine/expected for a floating card. */
.vdx-progress-banner-spacer {{
    height: 3rem;
}}

.vdx-progress-banner {{
    position: fixed;
    top: 76px;
    left: 50%;
    transform: translateX(-50%);
    width: min(880px, calc(100vw - 3rem));
    box-sizing: border-box;
    z-index: 999;
    background-color: rgba(251, 251, 253, 0.92);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--vdx-rule);
    border-radius: 14px;
    padding: 0.9rem 1.25rem;
    box-shadow: 0 8px 28px rgba(0, 0, 0, 0.1);
}}

.vdx-progress-label {{
    font-size: 0.9rem;
    font-weight: 500;
    color: var(--vdx-text);
    margin-bottom: 0.5rem;
    /* Short, subtle "something happened" tick each time the label swaps
       -- proportional feedback, not decoration (fires once per .advance()
       call since it's a CSS animation on a freshly-set element, not an
       infinite loop). */
    animation: vdxLabelPulse 0.3s ease-out;
}}

.vdx-progress-track {{
    height: 4px;
    border-radius: 2px;
    background-color: var(--vdx-rule);
    overflow: hidden;
}}

.vdx-progress-fill {{
    height: 100%;
    border-radius: 2px;
    background-color: var(--vdx-accent);
    transition: width 0.35s cubic-bezier(0.16, 1, 0.3, 1);
}}

@keyframes vdxLabelPulse {{
    0% {{ opacity: 0.4; transform: translateY(2px); }}
    100% {{ opacity: 1; transform: translateY(0); }}
}}

/* Skeleton placeholders: the entire results area shows loading shape
   immediately (not just the banner) -- a scrolled-down user sees "this is
   loading" everywhere on screen, not a blank gap below an off-screen
   spinner. */
.vdx-skeleton {{
    margin-bottom: 1.5rem;
}}

.vdx-skeleton-box {{
    border-radius: 10px;
    background: linear-gradient(100deg, var(--vdx-rule) 30%, #ECECEF 45%, var(--vdx-rule) 60%);
    background-size: 200% 100%;
    animation: vdxShimmer 1.6s ease-in-out infinite;
}}

.vdx-skeleton-title {{
    height: 1.1rem;
    width: 40%;
    margin-bottom: 0.75rem;
}}

.vdx-skeleton-metrics {{
    display: flex;
    gap: 0.75rem;
    margin-bottom: 0.75rem;
}}

.vdx-skeleton-pill {{
    height: 3.5rem;
    flex: 1;
}}

.vdx-skeleton-image {{
    height: 220px;
    width: 100%;
}}

@keyframes vdxShimmer {{
    0% {{ background-position: 200% 0; }}
    100% {{ background-position: -200% 0; }}
}}

/* Result sections fade/slide in gently as they're revealed -- scoped via
   st.container(key=...)'s stable "st-key-<key>" class (confirmed present
   in the installed Streamlit build), not a generic per-widget selector,
   so a whole section animates as one unit rather than each metric tile
   animating separately. Short and subtle enough that it reads fine
   replaying on every rerun (cache-hit redraws use the same keys), not
   just first mount. */
[class*="st-key-vdx-section-"] {{
    animation: vdxFadeInUp 0.4s cubic-bezier(0.16, 1, 0.3, 1) both;
}}

@keyframes vdxFadeInUp {{
    from {{ opacity: 0; transform: translateY(10px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}

.vdx-error-card {{
    background-color: #FBF6EC;
    border: 1px solid #EFE1BE;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 1rem;
}}

.vdx-error-title {{
    font-weight: 600;
    color: var(--vdx-text);
    margin-bottom: 0.25rem;
}}

.vdx-error-detail {{
    color: var(--vdx-muted);
    font-size: 0.85rem;
    font-family: ui-monospace, "SF Mono", Consolas, monospace;
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
    .stDownloadButton,
    .vdx-progress-banner,
    .vdx-progress-banner-spacer,
    .vdx-skeleton {{
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
