"""Phase 9 / redesign: dashboard visual theme.

A dense, glass-surfaced "instrument panel" look: frosted white cards
(backdrop-filter blur + saturation boost, the trick that makes Apple's own
frosted materials read as "alive" rather than washed out) float over a
softly two-toned light background, with a warm copper / deep teal accent
duo drawn from the subject itself -- copper for the optic disc's pale
warm tissue and "a finding is present," teal for fundus-imaging equipment's
characteristic blue-green and "normal/calm." Streamlit has no first-class
theming API expressive enough for this (its config.toml theme only covers
a handful of colors), so this injects scoped CSS instead -- targeting
Streamlit's `data-testid` attributes rather than its generated class
names, since those are the one part of its DOM that's meant to be a
stable styling hook across versions.

Deliberately NOT applied to report/pdf.py's ReportLab output -- that's a
separate, print-optimized renderer (ink-conscious, A4) where glass/blur/
gradient treatments would actively work against the stated goal of a clean
printed page. Both renderers still walk the same report/content.py
Section list, so they can never disagree on *content*, only presentation.
"""

import streamlit as st

# Same accent family as report/pdf.py's restrained palette in spirit, but
# re-hued for the glass theme -- see module docstring for why copper/teal,
# not the old flat blue/emerald/amber trio.
_TEAL = "#0E7C86"  # primary accent: buttons, links, progress fill, "normal" status
_COPPER = "#B3611A"  # "a finding is present" / attention status, used sparingly
_INFO = "#5B6B7A"  # neutral/informational pill, deliberately quiet -- not a 3rd loud accent
_TEXT = "#1A1D23"
_MUTED = "#5F6570"
_BORDER = "#DCE0E7"
_BACKGROUND = "#ECEEF3"
_GLASS = "rgba(255, 255, 255, 0.72)"
_GLASS_BORDER = "rgba(255, 255, 255, 0.85)"
_TRACK = "rgba(255, 255, 255, 0.5)"

_CSS = f"""
<style>
/* Inter (UI/prose) + JetBrains Mono (numeric data) stay -- both already
   doing their job well, no reason to churn a typeface that works. Fraunces
   (a variable serif with warm, slightly editorial terminals) is new: used
   only for the page title, section headers, and each disease tile's
   verdict line -- an "authored report" register contrasted deliberately
   against the cold, precise mono used for raw numbers. This is the one
   real typographic risk this redesign takes; everything else stays
   disciplined around it. All three are open, Google-Fonts-hosted (same
   network-dependency trade-off as before -- degrades to system fonts
   silently if blocked, no broken layout). */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=JetBrains+Mono:wght@500;600&display=swap');

:root {{
    --vdx-teal: {_TEAL};
    --vdx-copper: {_COPPER};
    --vdx-info: {_INFO};
    --vdx-text: {_TEXT};
    --vdx-muted: {_MUTED};
    --vdx-rule: {_BORDER};
    --vdx-background: {_BACKGROUND};
    --vdx-glass: {_GLASS};
    --vdx-glass-border: {_GLASS_BORDER};
    --vdx-track: {_TRACK};
    --vdx-font-sans: 'Inter', -apple-system, "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    --vdx-font-serif: 'Fraunces', Georgia, "Times New Roman", serif;
    --vdx-font-mono: 'JetBrains Mono', ui-monospace, "SF Mono", Consolas, monospace;
}}

/* Ambient cursor-following glow -- the one deliberate exception to an
   otherwise no-gradients brief. `--vdx-mouse-x`/`--vdx-mouse-y` are written
   by inject_ambient_cursor()'s JS (a CCv2 component, see below) onto
   `documentElement`, so this radial gradient's *position* tracks the
   cursor while its *color* stays strictly monochromatic (white fading to
   the near-white base, no hue) -- a faint light-following effect, not a
   visible color gradient. `background-attachment: fixed` ties it to the
   viewport (not scrolled content), matching the mouse coordinates' own
   viewport-relative frame. The `50% 40%` fallback (before the first
   mousemove fires, or with JS disabled) keeps it looking intentional, not
   broken, in that split second.

   Sized/opacity-tuned after checking real screenshots: a first pass (peak
   alpha 0.85, radius 1100px) measured as only a ~3-4% brightness delta
   between opposite cursor corners -- technically hue-neutral but too subtle
   to read as deliberate rather than a screenshot artifact. Wider radius +
   higher peak alpha (still fading to fully transparent, still pure white,
   so it can't become "a visible color gradient") makes the moving light
   actually legible without introducing any hue. */
.stApp {{
    background-color: var(--vdx-background);
    background-image: radial-gradient(
        circle 1500px at var(--vdx-mouse-x, 50%) var(--vdx-mouse-y, 40%),
        rgba(255, 255, 255, 1) 0%,
        rgba(255, 255, 255, 0) 75%
    );
    background-attachment: fixed;
}}

html, body, [class*="css"] {{
    font-family: var(--vdx-font-sans) !important;
    color: var(--vdx-text);
}}

/* !important here specifically: verified live that Streamlit ships its own
   emotion-generated rule directly targeting headings (e.g.
   ".st-emotion-cache-<hash> h1"), which -- as a class+type selector --
   beats a plain "h1, h2, h3" selector on specificity regardless of
   injection order. Deliberate, targeted override of a third-party
   framework's own opinionated default, not a general !important habit. */
h1, h2, h3 {{
    font-family: var(--vdx-font-serif) !important;
    font-weight: 600;
    letter-spacing: -0.01em;
    color: var(--vdx-text);
}}

/* Denser than before: less top padding, and the page already uses
   layout="wide" so content should use that width rather than capping it
   back down -- an information-dense dashboard doesn't want as much
   breathing room above the fold as a marketing page. */
.block-container {{
    padding-top: 0.75rem;
    padding-bottom: 3rem;
    max-width: 1360px;
}}

hr {{
    border: none;
    border-top: 1px solid var(--vdx-rule);
    margin: 0.85rem 0;
}}

/* --- Shared glass surface -----------------------------------------------
   One consistent "material" applied to every card-like element below
   (metric tiles, ring cards, stat tiles, expanders): translucent white +
   blur + a saturation boost (the trick that keeps Apple-style frosted
   materials from reading as merely faded) + a soft top/left highlight
   edge suggesting glass catching light, plus a diffuse shadow for lift. */
div[data-testid="stMetric"],
.vdx-ring-card,
.vdx-stat-tile,
div[data-testid="stExpander"] {{
    background: var(--vdx-glass);
    backdrop-filter: blur(20px) saturate(160%);
    -webkit-backdrop-filter: blur(20px) saturate(160%);
    border: 1px solid var(--vdx-glass-border);
    border-radius: 14px;
    box-shadow: 0 4px 24px rgba(20, 23, 30, 0.06);
}}

div[data-testid="stMetric"] {{
    padding: 0.5rem 0.75rem;
}}

div[data-testid="stMetricValue"] {{
    color: var(--vdx-text);
    font-family: var(--vdx-font-mono);
    font-size: 1.25rem;
}}

div[data-testid="stMetricLabel"] {{
    color: var(--vdx-muted);
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.045em;
}}

/* Buttons/links: teal is the single primary accent now (replacing the old
   flat blue) -- restrained motion on hover only, never on load/idle. */
.stButton > button, .stDownloadButton > button {{
    background-color: var(--vdx-teal);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.45rem 1.15rem;
    font-weight: 500;
    transition: opacity 0.15s ease;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
    opacity: 0.85;
    color: white;
}}

div[data-testid="stExpander"] {{
    transition: border-color 0.15s ease;
}}

.vdx-caption {{
    color: var(--vdx-muted);
    font-size: 0.82rem;
    margin-top: 0.2rem;
}}

.vdx-disclaimer {{
    color: var(--vdx-muted);
    font-size: 0.8rem;
    border-top: 1px solid var(--vdx-rule);
    padding-top: 0.75rem;
    margin-top: 1.5rem;
}}

/* --- Recommendation card --------------------------------------------------
   The one thing the old page-length "Report Preview" walk had that
   wasn't already shown elsewhere on the dashboard (see app/main.py's
   module docstring) -- given the same glass material as every other
   surface here, plus a teal left rule (this app's single primary accent)
   to read as "the conclusion", not just another paragraph of body text. */
.vdx-recommendation-card {{
    background: var(--vdx-glass);
    backdrop-filter: blur(20px) saturate(160%);
    -webkit-backdrop-filter: blur(20px) saturate(160%);
    border: 1px solid var(--vdx-glass-border);
    border-left: 3px solid var(--vdx-teal);
    border-radius: 14px;
    padding: 1rem 1.35rem;
    box-shadow: 0 4px 24px rgba(20, 23, 30, 0.06);
    margin: 0.25rem 0 1.25rem;
}}

.vdx-recommendation-title {{
    font-family: var(--vdx-font-serif);
    font-weight: 600;
    font-size: 1.05rem;
    color: var(--vdx-text);
    margin-bottom: 0.45rem;
}}

.vdx-recommendation-body {{
    font-size: 0.92rem;
    line-height: 1.55;
    color: var(--vdx-text);
}}

/* --- Micro-visualization: instrument-bezel ring gauge --------------------
   One reusable component (see app/components.py's render_ring()),
   parameterized entirely through inline CSS custom properties (--pct
   0-100, --ring-color). Heavier than the old thin conic-gradient ring --
   a thicker arc + an inset shadow on the inner disc suggests a lens/
   eyepiece bezel rather than a flat progress ring, the one deliberate
   "instrument" signature this redesign leans on. */
.vdx-ring-card {{
    padding: 0.75rem;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.4rem;
}}

.vdx-ring {{
    position: relative;
    width: 60px;
    height: 60px;
    border-radius: 50%;
    background: conic-gradient(var(--ring-color) calc(var(--pct) * 1%), var(--vdx-track) 0);
    box-shadow: inset 0 1px 3px rgba(20, 23, 30, 0.15);
}}

.vdx-ring-inner {{
    position: absolute;
    inset: 8px;
    border-radius: 50%;
    background: var(--vdx-glass);
    box-shadow: inset 0 1px 2px rgba(20, 23, 30, 0.12);
    display: grid;
    place-items: center;
    font-family: var(--vdx-font-mono);
    font-weight: 600;
    font-size: 0.72rem;
    color: var(--vdx-text);
}}

.vdx-ring-label {{
    font-size: 0.66rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.045em;
    color: var(--vdx-muted);
    text-align: center;
}}

/* --- Badge pills ----------------------------------------------------------
   Semantic variant names (see app/components.py's render_pill()): "normal"
   (teal -- no finding / calm), "attention" (copper -- a finding is
   present), "info" (neutral slate -- informational, not a status verdict).
   Renamed from the old color-named "emerald"/"amber"/"blue" since those
   names no longer describe the actual colors once re-hued -- a latent
   mismatch-bug class this avoids outright. */
.vdx-pill {{
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.22rem 0.65rem;
    border-radius: 999px;
    font-size: 0.74rem;
    font-weight: 600;
    font-family: var(--vdx-font-sans);
}}
.vdx-pill-normal {{ background: rgba(14, 124, 134, 0.12); color: var(--vdx-teal); }}
.vdx-pill-attention {{ background: rgba(179, 97, 26, 0.12); color: var(--vdx-copper); }}
.vdx-pill-info {{ background: rgba(91, 107, 122, 0.12); color: var(--vdx-info); }}

/* --- Compact stat tile -----------------------------------------------------
   One dense glass unit combining a label + pill + ring gauge (see
   app/components.py's render_stat_tile()) -- replaces the old two-column
   ring/datagrid layout for the three disease-detection tiles, the direct
   fix for their previous repetition (full subheader + pill + ring +
   datagrid + full-size image, three times over). */
.vdx-stat-tile {{
    padding: 0.85rem 0.9rem;
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
    height: 100%;
}}

.vdx-stat-tile-header {{
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 0.5rem;
}}

.vdx-stat-tile-title {{
    font-family: var(--vdx-font-serif);
    font-weight: 600;
    font-size: 0.98rem;
    color: var(--vdx-text);
    line-height: 1.25;
}}

.vdx-stat-tile-body {{
    display: flex;
    align-items: center;
    gap: 0.75rem;
}}

/* --- Compact data grid ---------------------------------------------------
   Secondary/detail numbers (see app/components.py's render_datagrid()) --
   headline numbers stay in ring cards or st.metric tiles. */
.vdx-datagrid {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
    margin-top: 0.2rem;
}}
.vdx-datagrid tr:nth-child(even) {{
    background: rgba(255, 255, 255, 0.35);
}}
.vdx-datagrid td {{
    padding: 0.4rem 0.6rem;
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

/* --- Image hover-zoom ------------------------------------------------------
   Targets Streamlit's own stImage wrapper directly. Scaling the <img>
   itself (not the wrapper) inside an overflow:hidden card keeps the zoom
   clipped to a fixed rounded frame, and keeps the caption (a sibling under
   the image, not inside the scaled element) from zooming along with it. */
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

/* --- Pills navigation (st.pills / st.segmented_control) --------------------
   Both widgets share one underlying component -- confirmed via the
   installed Streamlit build's own frontend bundle -- data-testid
   "stButtonGroup". Styled as a rounded segmented track on the glass
   material; the selected-pill accent fill comes from Streamlit's own
   aria-checked/data-selected state on the inner button. */
div[data-testid="stButtonGroup"] {{
    background: var(--vdx-track);
    padding: 0.22rem;
    border-radius: 999px;
    gap: 0.15rem;
}}
div[data-testid="stButtonGroup"] button {{
    border-radius: 999px !important;
    font-size: 0.8rem;
}}

/* --- v2: loading/progress experience ---------------------------------
   Fixes the "did the site crash?" problem an opaque, unpinned spinner
   had: this banner stays visible regardless of scroll position.

   `position: sticky` was tried first and does NOT work here -- verified
   live: it requires being a DIRECT child of the tall block providing its
   "roaming range" (confirmed empirically -- even one plain, unstyled
   wrapper div between a sticky element and its tall ancestor breaks it in
   this Chromium build). Streamlit always wraps `st.markdown()` output in
   several of its own layers, so a sticky element rendered through a
   normal Streamlit call can never be a direct child of anything tall
   enough.

   `position: fixed` sidesteps that, but has its OWN headless-Chromium-
   specific gotcha, also verified live: a fixed element spanning the full
   viewport width silently fails to paint its TEXT content in this
   environment (background/border still render -- only text disappears).
   Centering it as a fixed-width floating card via `left: 50%; transform:
   translateX(-50%)` avoids the bug entirely.

   `top: 76px` clears Streamlit's own header toolbar (measured live:
   [data-testid="stHeader"], 60px tall) plus a small gap. A spacer
   (.vdx-progress-banner-spacer, rendered in normal flow right before this)
   reserves room so real content isn't heavily covered when the banner
   first appears. */
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
    background: var(--vdx-glass);
    backdrop-filter: blur(20px) saturate(160%);
    -webkit-backdrop-filter: blur(20px) saturate(160%);
    border: 1px solid var(--vdx-glass-border);
    border-radius: 14px;
    padding: 0.9rem 1.25rem;
    box-shadow: 0 8px 28px rgba(20, 23, 30, 0.12);
}}

.vdx-progress-label {{
    font-size: 0.9rem;
    font-weight: 500;
    color: var(--vdx-text);
    margin-bottom: 0.5rem;
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
    background-color: var(--vdx-teal);
    transition: width 0.35s cubic-bezier(0.16, 1, 0.3, 1);
}}

@keyframes vdxLabelPulse {{
    0% {{ opacity: 0.4; transform: translateY(2px); }}
    100% {{ opacity: 1; transform: translateY(0); }}
}}

.vdx-skeleton {{
    margin-bottom: 1.25rem;
}}

.vdx-skeleton-box {{
    border-radius: 10px;
    background: linear-gradient(100deg, var(--vdx-rule) 30%, #F5F6F9 45%, var(--vdx-rule) 60%);
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
    height: 200px;
    width: 100%;
}}

@keyframes vdxShimmer {{
    0% {{ background-position: 200% 0; }}
    100% {{ background-position: -200% 0; }}
}}

/* Result sections fade/slide in gently as they're revealed -- scoped via
   st.container(key=...)'s stable "st-key-<key>" class, so a whole section
   animates as one unit rather than each metric tile animating separately.

   Deliberately NOT `animation-fill-mode: both` -- confirmed live that a
   fill-mode of "both"/"forwards" here breaks Streamlit's image fullscreen
   view: as long as a (possibly finished) CSS animation is still holding an
   element at a keyframe that touches `transform` -- even the identity
   transform, even spelled as `transform: none` in the keyframe itself --
   Chromium keeps treating the element as transformed and gives it a
   containing block for `position: fixed` descendants. Streamlit's
   fullscreen image overlay IS `position: fixed` and expects the viewport
   as its containing block; with fill-mode "both" it was instead getting
   trapped inside this small section card. Default fill-mode ("none")
   avoids the whole class of bug: once the 0.4s animation ends, the
   element fully reverts to its un-animated base style (no transform, full
   opacity already the default here), same visual end state, nothing left
   "holding" a transform. */
[class*="st-key-vdx-section-"] {{
    animation: vdxFadeInUp 0.4s cubic-bezier(0.16, 1, 0.3, 1);
}}

@keyframes vdxFadeInUp {{
    from {{ opacity: 0; transform: translateY(10px); }}
    to {{ opacity: 1; transform: none; }}
}}

.vdx-error-card {{
    background-color: rgba(179, 97, 26, 0.08);
    border: 1px solid rgba(179, 97, 26, 0.25);
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
   Small floating footer, same fixed+centered+bounded-width pattern as
   .vdx-progress-banner above. Bottom-anchored deliberately so it never
   shares vertical territory with Streamlit's own header or the progress
   banner during active loading. */
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
    background: var(--vdx-glass);
    backdrop-filter: blur(20px) saturate(160%);
    -webkit-backdrop-filter: blur(20px) saturate(160%);
    border: 1px solid var(--vdx-glass-border);
    border-radius: 12px;
    padding: 0.5rem 1rem;
    font-size: 0.78rem;
    color: var(--vdx-muted);
    text-align: center;
}}

/* Print support: pressing Ctrl+P on the live preview should print just the
   report content, not Streamlit's own chrome, and glass/blur/gradient
   treatments should flatten to plain white -- they don't print well and
   the PDF download (report/pdf.py, unaffected by this theme entirely) is
   the primary/reliable print path; this is just a convenience fallback
   for the live-preview screen itself. */
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
    .stApp {{
        background-image: none !important;
    }}
    div[data-testid="stMetric"],
    .vdx-ring-card,
    .vdx-stat-tile,
    .vdx-recommendation-card,
    div[data-testid="stExpander"] {{
        background: white !important;
        backdrop-filter: none !important;
        -webkit-backdrop-filter: none !important;
        box-shadow: none !important;
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


# Registered once at import time (not inside a function called per-render --
# see CCv2 docs: re-registering a component name mid-session logs a warning
# and can cause confusing behavior). This component renders no visible
# markup; its only job is the JS side effect below.
_AMBIENT_CURSOR = st.components.v2.component(
    "vdx_ambient_cursor",
    js="""
export default function (component) {
    // Lerp toward the cursor rather than snapping straight to it -- an
    // "ambient light drifting to follow you" read, not a background that
    // visibly jumps on every mouse tick. 50/40 matches the CSS fallback
    // in theme.py's .stApp rule (upper-middle glow before the first
    // mousemove event fires).
    let targetX = 50, targetY = 40;
    let curX = 50, curY = 40;
    let rafId = null;

    const onMove = (e) => {
        targetX = (e.clientX / window.innerWidth) * 100;
        targetY = (e.clientY / window.innerHeight) * 100;
    };
    window.addEventListener("mousemove", onMove);

    const tick = () => {
        curX += (targetX - curX) * 0.06;
        curY += (targetY - curY) * 0.06;
        document.documentElement.style.setProperty("--vdx-mouse-x", curX.toFixed(2) + "%");
        document.documentElement.style.setProperty("--vdx-mouse-y", curY.toFixed(2) + "%");
        rafId = requestAnimationFrame(tick);
    };
    tick();

    // Streamlit reruns can remount this component; tearing down the old
    // listener/loop here (the documented CCv2 cleanup pattern) keeps a
    // rerun-heavy session from stacking up duplicate mousemove listeners
    // and rAF loops racing each other.
    return () => {
        window.removeEventListener("mousemove", onMove);
        if (rafId) cancelAnimationFrame(rafId);
    };
}
""",
)


def inject_ambient_cursor() -> None:
    """Call once per page load, anywhere after inject_css(). Pure JS side
    effect (see _AMBIENT_CURSOR above) -- no visible output, no return
    value used.
    """
    _AMBIENT_CURSOR()
