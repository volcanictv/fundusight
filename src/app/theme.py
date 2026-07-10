"""Phase 9: dashboard visual theme.

One CSS injection point for the dashboard's visual design: a dense,
structured "medical suite" look -- light neutral surfaces, bento-grid
metric cards, circular/bar micro-visualizations, badge pills, and compact
data grids in place of plain text rows. Streamlit has no first-class
theming API expressive enough for this (its config.toml theme only covers
a handful of colors), so this injects scoped CSS instead -- targeting
Streamlit's `data-testid` attributes rather than its generated class
names, since those are the one part of its DOM that's meant to be a
stable styling hook across versions.
"""

import streamlit as st

# Same accent as report/pdf.py's _ACCENT_COLOR -- keeps the in-app preview
# and the exported PDF visually consistent, not two different brands.
_ACCENT = "#0071E3"
_TEXT = "#14171C"
_MUTED = "#6B7280"
_BORDER = "#E4E6EA"
_BACKGROUND = "#F7F8FA"
_CARD = "#FFFFFF"
_SUCCESS = "#059669"
_WARNING = "#B45309"
_CHIP_BG = "#F0F1F3"

_CSS = f"""
<style>
/* Inter (sans, UI/prose) + JetBrains Mono (numeric metrics) -- both open
   and Google-Fonts-hosted. "SF Pro Display" was the original ask, but
   it's Apple-licensed and not legally self-hostable off Apple platforms;
   Inter is the standard open substitute, explicitly designed as an
   SF-Pro-adjacent UI face. Trade-off worth flagging: this adds a network
   dependency the app didn't have before (it was deliberately
   system-fonts-only for offline robustness) -- the font-family fallback
   chains below mean a blocked/slow request just silently degrades to
   system fonts, no broken layout, but the "premium" typography itself
   needs network access to actually show up.
   */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600&display=swap');

:root {{
    --vdx-accent: {_ACCENT};
    --vdx-text: {_TEXT};
    --vdx-muted: {_MUTED};
    --vdx-rule: {_BORDER};
    --vdx-background: {_BACKGROUND};
    --vdx-card: {_CARD};
    --vdx-success: {_SUCCESS};
    --vdx-warning: {_WARNING};
    --vdx-chip-bg: {_CHIP_BG};
    --vdx-font-sans: 'Inter', -apple-system, "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    --vdx-font-mono: 'JetBrains Mono', ui-monospace, "SF Mono", Consolas, monospace;
}}

.stApp {{
    background-color: var(--vdx-background);
}}

html, body, [class*="css"] {{
    font-family: var(--vdx-font-sans) !important;
    color: var(--vdx-text);
}}

/* !important on the font-family here specifically: verified live that
   Streamlit ships its own emotion-generated rule directly targeting
   headings (e.g. ".st-emotion-cache-<hash> h1"), which -- as a
   class+type selector -- beats a plain "h1, h2, h3" selector on
   specificity regardless of injection order, and any DIRECT rule on an
   element always beats inheriting the font-family from html/body above.
   This is a deliberate, targeted override of a third-party framework's
   own opinionated default (whose hashed class name isn't a stable
   selector to out-specificity against), not a general !important habit. */
h1, h2, h3 {{
    font-family: var(--vdx-font-sans) !important;
    font-weight: 600;
    letter-spacing: -0.01em;
    color: var(--vdx-text);
}}

/* Denser than a generic "Apple-like" restrained layout: the page already
   sets layout="wide" in main.py, so let content actually use that width
   rather than capping it back down, and cut the top padding -- a
   data-dense dashboard doesn't need as much breathing room above the
   fold as a marketing-style page does. */
.block-container {{
    padding-top: 1.25rem;
    padding-bottom: 3rem;
    max-width: 1280px;
}}

hr {{
    border: none;
    border-top: 1px solid var(--vdx-rule);
    margin: 1.1rem 0;
}}

/* Bento metric cards -- upgrades every existing st.metric() call with no
   Python changes: tighter padding, card background, monospaced numeric
   value (medical/technical readouts read as more precise in a mono
   face), uppercase/tracked label. */
div[data-testid="stMetric"] {{
    background-color: var(--vdx-card);
    border: 1px solid var(--vdx-rule);
    border-radius: 10px;
    padding: 0.55rem 0.8rem;
}}

div[data-testid="stMetricValue"] {{
    color: var(--vdx-text);
    font-family: var(--vdx-font-mono);
    font-size: 1.35rem;
}}

div[data-testid="stMetricLabel"] {{
    color: var(--vdx-muted);
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.045em;
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

/* --- Micro-visualizations: circular ring gauge ------------------------
   One reusable component (see app/components.py's render_ring()),
   parameterized entirely through inline CSS custom properties
   (--pct 0-100, --ring-color) rather than five bespoke pieces per metric.
   conic-gradient draws the filled arc; the inner disc masks the center
   to give the "ring" (not pie-chart) look, with the value printed in the
   monospace face for a precise, technical readout. */
.vdx-ring-card {{
    background: var(--vdx-card);
    border: 1px solid var(--vdx-rule);
    border-radius: 12px;
    padding: 0.9rem;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.55rem;
}}

.vdx-ring {{
    position: relative;
    width: 64px;
    height: 64px;
    border-radius: 50%;
    background: conic-gradient(var(--ring-color) calc(var(--pct) * 1%), var(--vdx-chip-bg) 0);
}}

.vdx-ring-inner {{
    position: absolute;
    inset: 7px;
    border-radius: 50%;
    background: var(--vdx-card);
    display: grid;
    place-items: center;
    font-family: var(--vdx-font-mono);
    font-weight: 600;
    font-size: 0.78rem;
    color: var(--vdx-text);
}}

.vdx-ring-label {{
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.045em;
    color: var(--vdx-muted);
    text-align: center;
}}

/* --- Badge pills -------------------------------------------------------
   Reusable status/severity indicator (see app/components.py's
   render_pill()) -- light tint background, saturated text, never color
   alone carrying meaning since the text itself states the label. */
.vdx-pill {{
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.25rem 0.7rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    font-family: var(--vdx-font-sans);
}}
.vdx-pill-emerald {{ background: #ECFDF5; color: var(--vdx-success); }}
.vdx-pill-amber {{ background: #FFFBEB; color: var(--vdx-warning); }}
.vdx-pill-blue {{ background: #EFF6FF; color: var(--vdx-accent); }}

/* --- Compact data grid ---------------------------------------------
   Secondary/detail numbers (e.g. branch count, tortuosity, disc/cup
   diameters) -- headline numbers stay in ring cards or st.metric tiles,
   this is for the supporting detail rows (see app/components.py's
   render_datagrid()). */
.vdx-datagrid {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
    margin-top: 0.25rem;
}}
.vdx-datagrid tr:nth-child(even) {{
    background: var(--vdx-background);
}}
.vdx-datagrid td {{
    padding: 0.45rem 0.65rem;
    border-bottom: 1px solid var(--vdx-rule);
}}
.vdx-datagrid tr:last-child td {{
    border-bottom: none;
}}
.vdx-datagrid td:first-child {{
    font-weight: 600;
    color: var(--vdx-text);
}}
.vdx-datagrid td:last-child {{
    font-family: var(--vdx-font-mono);
    text-align: right;
    color: var(--vdx-text);
}}

/* --- Image hover-zoom ---------------------------------------------
   Targets Streamlit's own stImage wrapper directly -- no custom wrapper
   needed. Scaling the <img> itself (not the wrapper) inside an
   overflow:hidden card keeps the zoom clipped to a fixed rounded frame
   instead of spilling over neighboring content, and keeps the caption
   (Streamlit renders it as a sibling under the image, not inside the
   scaled element) from zooming along with the image. */
div[data-testid="stImage"] {{
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid var(--vdx-rule);
}}
div[data-testid="stImage"] img {{
    display: block;
    transition: transform 0.2s cubic-bezier(0.16, 1, 0.3, 1);
}}
div[data-testid="stImage"]:hover img {{
    transform: scale(1.04);
}}

/* --- Pills navigation (st.pills / st.segmented_control) ---------------
   Both widgets share one underlying component -- confirmed via the
   installed Streamlit build's own frontend bundle -- data-testid
   "stButtonGroup", not a guessed "stPills". Styled as a rounded segmented
   track; the selected-pill accent fill comes from Streamlit's own
   aria-checked/data-selected state on the inner button, confirmed and
   tuned live rather than guessed (see app/main.py's image comparison
   viewer). */
div[data-testid="stButtonGroup"] {{
    background: var(--vdx-chip-bg);
    padding: 0.25rem;
    border-radius: 999px;
    gap: 0.15rem;
}}
div[data-testid="stButtonGroup"] button {{
    border-radius: 999px !important;
    font-size: 0.82rem;
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
   "restrained/Apple-like" than an edge-to-edge bar anyway. The disclaimer
   footer below follows this exact same proven pattern.

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
    font-family: var(--vdx-font-mono);
}}

/* --- Disclaimer footer -------------------------------------------------
   "Educational/portfolio demonstration only" no longer breaks the page's
   flow as an inline caption under the title -- it's a small floating
   footer, following the exact same fixed+centered+bounded-width pattern
   as .vdx-progress-banner above (never left:0;right:0 -- see that block's
   comment for why). Bottom-anchored deliberately: it then never shares
   vertical territory with Streamlit's own header (top 0-60px) or the
   progress banner (top 76px) during active loading, so there's no
   stacking/collision logic needed between the two floating elements. */
.vdx-footer-spacer {{
    height: 2.5rem;
}}

.vdx-disclaimer-footer {{
    position: fixed;
    bottom: 1rem;
    left: 50%;
    transform: translateX(-50%);
    width: min(680px, calc(100vw - 3rem));
    box-sizing: border-box;
    z-index: 900;
    background-color: rgba(255, 255, 255, 0.92);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--vdx-rule);
    border-radius: 12px;
    padding: 0.5rem 1rem;
    font-size: 0.78rem;
    color: var(--vdx-muted);
    text-align: center;
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
    .vdx-skeleton,
    .vdx-disclaimer-footer,
    .vdx-footer-spacer {{
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
