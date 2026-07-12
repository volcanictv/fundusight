"""Dashboard visual theme: "Clinical Liquid Glass" -- adopted from a
Stitch-generated reference mockup (`Front-End Template/stitch_visiondx_
retinal_screening_dashboard/`: DESIGN.md + code.html + screen.png).

- Indigo (#3525CD, "primary") is the single primary accent -- buttons,
  links, the "normal/no finding" semantic, focus rings. Sky-blue secondary
  (#0EA5E9) covers neutral/informational accents. Burnt-orange tertiary
  (#A44100) is "a finding is present" / attention, used sparingly.
- Hanken Grotesk (headlines) + Inter (body) + Geist (small mono-ish labels/
  data) -- matches the reference's actual Google Fonts import exactly
  (its DESIGN.md prose says Manrope/JetBrains Mono, but its own code.html
  loads Hanken Grotesk/Inter/Geist; code.html is the literal rendered
  artifact behind screen.png, so it's ground truth here, not the prose doc).
- Frost-white background (#F7F9FB) with glass-card surfaces: translucent
  white + blur, a soft black hairline border (rgba(0,0,0,0.08), the
  reference's own `border-luminous` token) and a large, very soft ambient
  shadow ("Light Diffusion" in the reference's own DESIGN.md).
- Material Symbols Outlined icons (settings, cloud_upload, rocket_launch,
  etc.) throughout, matching the reference's header/dropzone/buttons.

Streamlit has no first-class theming API expressive enough for this (its
config.toml theme only covers a handful of colors), so this injects scoped
CSS instead -- targeting Streamlit's `data-testid` attributes rather than
its generated class names, since those are the one part of its DOM that's
meant to be a stable styling hook across versions.

Deliberately NOT applied to report/pdf.py's ReportLab output -- that's a
separate, print-optimized renderer (ink-conscious, A4) where glass/blur/
gradient treatments would work against a clean printed page. Both
renderers still walk the same report/content.py Section list, so they can
never disagree on *content*, only presentation.
"""

import streamlit as st

# "Clinical Liquid Glass" palette -- see module docstring for where these
# come from. Named by ROLE (primary/secondary/tertiary), not by hue, so a
# future re-hue doesn't leave a misleading name behind the way the old
# "_TEAL"/"_COPPER" constants would have here.
_PRIMARY = "#3525CD"  # indigo -- single primary accent: buttons, links, progress fill, "normal" status
_PRIMARY_CONTAINER = "#4F46E5"  # lighter indigo -- hover/active states, "Initialize" button base
_SECONDARY = "#0EA5E9"  # sky blue -- neutral/informational accent, not a status verdict
_TERTIARY = "#A44100"  # burnt orange -- "a finding is present" / attention status, used sparingly
_ERROR = "#BA1A1A"  # real app/pipeline errors -- distinct from the clinical "attention" tertiary
_TEXT = "#191C1E"
_MUTED = "#464555"
_OUTLINE = "#C7C4D8"  # hairline borders, dividers
_BACKGROUND = "#F7F9FB"
_GLASS = "rgba(255, 255, 255, 0.7)"
_GLASS_BORDER = "rgba(0, 0, 0, 0.08)"  # the reference's "border-luminous" token
_TRACK = "rgba(0, 0, 0, 0.06)"

_CSS = f"""
<style>
/* Hanken Grotesk (headlines) + Inter (body) + Geist (small/mono-ish
   labels and data) -- matching the Stitch reference's own Google Fonts
   import exactly (see module docstring for why code.html, not DESIGN.md's
   prose, is ground truth here). Degrades to system fonts silently if the
   network is blocked, no broken layout. */
@import url('https://fonts.googleapis.com/css2?family=Hanken+Grotesk:wght@600;700&family=Inter:wght@400;500;600;700&family=Geist:wght@400;500;600&display=swap');

/* Material Symbols Outlined needs all FOUR variable-font axes (opsz,
   wght, FILL, GRAD) requested in the @import -- three silently fails to
   apply at all, leaving every hand-inserted `material-symbols-outlined`
   span (e.g. the intake dropzone's upload icon) rendering as literal
   fallback text ("cloud_upload") instead of a glyph. Also `display=block`
   here, not `swap` like the text fonts above: this is a ligature-based
   icon font, so "swap" briefly (or, if the request fails, permanently)
   shows the raw icon NAME as text; "block" hides the fallback instead.
   Streamlit's own `icon=":material/...":` shorthand (used on st.button
   below) bundles its own copy of this font and is unaffected -- only the
   manually-inserted spans are. */
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=block');

:root {{
    --fdx-primary: {_PRIMARY};
    --fdx-primary-container: {_PRIMARY_CONTAINER};
    --fdx-secondary: {_SECONDARY};
    --fdx-tertiary: {_TERTIARY};
    --fdx-error: {_ERROR};
    --fdx-text: {_TEXT};
    --fdx-muted: {_MUTED};
    --fdx-rule: {_OUTLINE};
    --fdx-background: {_BACKGROUND};
    --fdx-glass: {_GLASS};
    --fdx-glass-border: {_GLASS_BORDER};
    --fdx-track: {_TRACK};
    --fdx-font-sans: 'Inter', -apple-system, "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    --fdx-font-display: 'Hanken Grotesk', 'Inter', -apple-system, "Segoe UI", sans-serif;
    --fdx-font-mono: 'Geist', 'Inter', ui-monospace, "SF Mono", Consolas, monospace;
}}

/* Material Symbols Outlined -- variable font, filled via the "FILL" axis
   rather than swapping to a separate "filled" font file. Un-set icons
   would otherwise render as literal ligature text (e.g. the word
   "settings") for a moment before the font loads; sizing/alignment is
   handled per-usage via inline style where needed. */
.material-symbols-outlined {{
    font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
    vertical-align: middle;
    line-height: 1;
}}

/* Ambient cursor-following glow -- the one deliberate exception to an
   otherwise no-gradients brief. `--fdx-mouse-x`/`--fdx-mouse-y` are written
   by inject_ambient_cursor()'s JS (a CCv2 component, see below) onto
   `documentElement`, so this radial gradient's *position* tracks the
   cursor while its *color* stays strictly monochromatic (white fading to
   the near-white base, no hue) -- a faint light-following effect, not a
   visible color gradient. The `50% 40%` fallback (before the first
   mousemove fires, or with JS disabled) keeps it looking intentional, not
   broken, in that split second.

   The gradient lives on `.stApp::before`, a separate fixed full-viewport
   layer BEHIND the real content, rather than directly on `.stApp`:
   `filter: blur()` affects an element's entire rendered output including
   its children, so applying it straight to `.stApp` would blur the whole
   dashboard, not just its background. A dedicated `::before` layer
   (z-index behind everything, pointer-events disabled so it never
   intercepts clicks) lets the blur apply to ONLY the glow itself. `.stApp`
   keeps the plain background-color as a fallback base underneath.

   CRITICAL: do NOT add `position: relative` (or any value besides the
   default `static`) to `.stApp`. It isn't needed -- a `position: fixed`
   child resolves against the viewport by default, it does not need its
   parent to be positioned, unlike `position: absolute` -- and it breaks
   the app: `position: relative` here makes `.stApp` the nearest
   positioned ancestor for Streamlit's OWN `stAppViewContainer` (which is
   `position: absolute` internally), silently changing its containing
   block. The dashboard still renders on first load (likely luck/caching)
   but goes completely blank after any rerun, with DOM/computed styles
   still reporting normal-looking values -- very hard to spot without a
   bisect. */
.stApp {{
    background-color: var(--fdx-background);
}}
.stApp::before {{
    content: "";
    position: fixed;
    inset: 0;
    z-index: -1;
    pointer-events: none;
    background-image: radial-gradient(
        circle 1500px at var(--fdx-mouse-x, 50%) var(--fdx-mouse-y, 40%),
        rgba(255, 255, 255, 1) 0%,
        rgba(255, 255, 255, 0) 75%
    );
    filter: blur(48px);
}}

html, body, [class*="css"] {{
    font-family: var(--fdx-font-sans) !important;
    color: var(--fdx-text);
}}

/* !important here specifically: verified live that Streamlit ships its own
   emotion-generated rule directly targeting headings (e.g.
   ".st-emotion-cache-<hash> h1"), which -- as a class+type selector --
   beats a plain "h1, h2, h3" selector on specificity regardless of
   injection order. Deliberate, targeted override of a third-party
   framework's own opinionated default, not a general !important habit. */
h1, h2, h3 {{
    font-family: var(--fdx-font-display) !important;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: var(--fdx-text);
}}

/* The page uses layout="wide", so content should use that width rather
   than capping it back down -- an information-dense dashboard doesn't
   want as much breathing room above the fold as a marketing page. */
.block-container {{
    padding-top: 0.75rem;
    padding-bottom: 3rem;
    max-width: 1360px;
}}

hr {{
    border: none;
    border-top: 1px solid var(--fdx-rule);
    margin: 0.85rem 0;
}}

/* --- Shared glass surface -----------------------------------------------
   One consistent "material" applied to every card-like element below
   (metric tiles, ring cards, stat tiles, expanders): translucent white +
   blur, a soft black hairline ("luminous edge", the reference's own
   `border-luminous` token), and a large, very soft ambient shadow -- the
   reference's "Light Diffusion" elevation model (DESIGN.md: "a very
   large, very soft shadow... provides a subtle lift without appearing
   heavy"). */
div[data-testid="stMetric"],
.fdx-ring-card,
.fdx-stat-tile,
.fdx-datagrid-card,
div[data-testid="stExpander"] {{
    background: var(--fdx-glass);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border: 1px solid var(--fdx-glass-border);
    border-radius: 16px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.04);
}}

div[data-testid="stMetric"] {{
    padding: 0.5rem 0.75rem;
}}

div[data-testid="stMetricValue"] {{
    color: var(--fdx-text);
    font-family: var(--fdx-font-mono);
    font-size: 1.25rem;
}}

div[data-testid="stMetricLabel"] {{
    color: var(--fdx-muted);
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.045em;
}}

/* Buttons/links: indigo is the single primary accent -- restrained motion
   on hover only, never on load/idle. A soft indigo-tinted glow shadow (the
   reference's `shadow-primary/20`) instead of a flat drop shadow reads as
   "the glass itself is lit from within", not just lifted.

   `color: white` on `.stButton > button *` (not just the button itself)
   matters: st.download_button (and st.button with no explicit type=)
   render as Streamlit's "secondary" kind, and Streamlit's own base CSS
   sets color directly on the inner text node (the <p> inside
   stMarkdownContainer) to its secondary-button text color -- this app's
   own primary indigo, same as the background here. A directly-matching
   rule on that inner node always wins over an inherited one regardless of
   specificity, so setting color only on the button produces invisible
   indigo-on-indigo text until :hover's opacity change reveals it.
   Targeting the inner node explicitly, at the same depth Streamlit's own
   rule does, fixes it at the actual point of conflict. */
.stButton > button, .stDownloadButton > button {{
    background-color: var(--fdx-primary);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.5rem 1.25rem;
    font-weight: 600;
    box-shadow: 0 8px 20px rgba(53, 37, 205, 0.2);
    transition: opacity 0.15s ease, transform 0.15s ease;
}}
.stButton > button *, .stDownloadButton > button * {{
    color: white;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
    opacity: 0.9;
    color: white;
    transform: translateY(-1px);
}}

div[data-testid="stExpander"] {{
    transition: border-color 0.15s ease;
}}

.fdx-caption {{
    color: var(--fdx-muted);
    font-size: 0.82rem;
    margin-top: 0.2rem;
}}

.fdx-disclaimer {{
    color: var(--fdx-muted);
    font-size: 0.8rem;
    border-top: 1px solid var(--fdx-rule);
    padding-top: 0.75rem;
    margin-top: 1.5rem;
}}

/* --- Recommendation card --------------------------------------------------
   Same glass material as every other surface here, plus an indigo left
   rule (this app's single primary accent) so it reads as "the
   conclusion", not just another paragraph of body text. */
.fdx-recommendation-card {{
    background: var(--fdx-glass);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border: 1px solid var(--fdx-glass-border);
    border-left: 3px solid var(--fdx-primary);
    border-radius: 16px;
    padding: 1rem 1.35rem;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.04);
    margin: 0.25rem 0 1.25rem;
}}

.fdx-recommendation-title {{
    font-family: var(--fdx-font-display);
    font-weight: 700;
    font-size: 1.05rem;
    color: var(--fdx-text);
    margin-bottom: 0.45rem;
}}

.fdx-recommendation-body {{
    font-size: 0.92rem;
    line-height: 1.55;
    color: var(--fdx-text);
}}

/* A visual break from the clinical summary above it -- see
   components.py's render_recommendation_card() for why this is split out
   of the main paragraph: a legal/educational disclaimer trailing off as
   just the last clause of a run-on paragraph reads as an afterthought,
   not the distinct notice it's meant to be. */
.fdx-recommendation-disclaimer {{
    font-size: 0.78rem;
    line-height: 1.5;
    color: var(--fdx-muted);
    border-top: 1px solid var(--fdx-rule);
    margin-top: 0.75rem;
    padding-top: 0.6rem;
}}

/* --- Micro-visualization: instrument-bezel ring gauge --------------------
   One reusable component (see app/components.py's render_ring()),
   parameterized entirely through inline CSS custom properties (--pct
   0-100, --ring-color). A thicker arc + an inset shadow on the inner disc
   suggests a lens/eyepiece bezel rather than a flat progress ring. Every
   ring on the page (Image Quality, the three Disease Screening tiles,
   Vessel Density, Vertical CDR) uses this one rule.

   `@property` registers --pct as a real animatable numeric value (browsers
   otherwise treat all custom properties as opaque strings, which can't be
   interpolated) -- this is what makes the fill-in animation below possible
   at all. */
@property --pct {{
    syntax: "<number>";
    inherits: true;
    initial-value: 0;
}}

/* The animation only has a "from" keyframe -- CSS animations fill in an
   implicit "to" keyframe from the element's own resolved --pct (the value
   the inline style already sets), so this animates 0 -> the real value on
   every mount without needing JS or a second, duplicated value anywhere. */
@keyframes fdx-ring-fill {{
    from {{ --pct: 0; }}
}}

.fdx-ring-card {{
    padding: 0.85rem;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.5rem;
}}

.fdx-ring {{
    position: relative;
    width: 84px;
    height: 84px;
    border-radius: 50%;
    background: conic-gradient(var(--ring-color) calc(var(--pct) * 1%), var(--fdx-track) 0);
    box-shadow: inset 0 1px 3px rgba(20, 23, 30, 0.15);
    animation: fdx-ring-fill 1s ease-out;
}}

.fdx-ring-inner {{
    position: absolute;
    inset: 11px;
    border-radius: 50%;
    background: var(--fdx-glass);
    box-shadow: inset 0 1px 2px rgba(20, 23, 30, 0.12);
    display: grid;
    place-items: center;
    font-family: var(--fdx-font-mono);
    font-weight: 600;
    font-size: 0.92rem;
    color: var(--fdx-text);
}}

.fdx-ring-label {{
    font-size: 0.66rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.045em;
    color: var(--fdx-muted);
    text-align: center;
}}

/* --- Badge pills ----------------------------------------------------------
   Semantic variant names (see app/components.py's render_pill()): "normal"
   (indigo -- no finding / calm), "attention" (tertiary orange -- a finding
   is present), "info" (secondary sky-blue -- informational, not a status
   verdict). */
.fdx-pill {{
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.22rem 0.65rem;
    border-radius: 999px;
    font-size: 0.74rem;
    font-weight: 600;
    font-family: var(--fdx-font-sans);
}}
.fdx-pill-normal {{ background: rgba(53, 37, 205, 0.1); color: var(--fdx-primary); }}
.fdx-pill-attention {{ background: rgba(164, 65, 0, 0.1); color: var(--fdx-tertiary); }}
.fdx-pill-info {{ background: rgba(14, 165, 233, 0.12); color: var(--fdx-secondary); }}

/* --- Compact stat tile -----------------------------------------------------
   One dense glass unit combining a label + pill + ring gauge (see
   app/components.py's render_stat_tile()), used for the three
   disease-detection tiles. */
.fdx-stat-tile {{
    padding: 0.85rem 0.9rem;
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
    height: 100%;
}}

.fdx-stat-tile-header {{
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 0.5rem;
}}

.fdx-stat-tile-title {{
    font-family: var(--fdx-font-display);
    font-weight: 700;
    font-size: 0.98rem;
    color: var(--fdx-text);
    line-height: 1.25;
}}

.fdx-stat-tile-body {{
    display: flex;
    align-items: center;
    gap: 0.75rem;
}}

/* Each disease tile's probability-breakdown chart, always visible (see
   app/main.py's render_detection_section() et al.), in its own glass card
   matching the tile above it. */
[class*="st-key-fdx-chart-"] {{
    background: var(--fdx-glass);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border: 1px solid var(--fdx-glass-border);
    border-radius: 16px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.04);
    padding: 0.6rem 0.75rem 0.75rem;
    margin-top: 0.6rem;
}}

/* The Preprocessing tile (Overview row) -- same glass card LOOK as its
   row neighbor, the Image Quality tile, but deliberately WITHOUT
   backdrop-filter, unlike every other glass card in this file. This is
   the one glass card that wraps real <img> content (the before/after
   preprocessing photos), and backdrop-filter -- like transform/filter/
   perspective -- makes an element a containing block for `position:
   fixed` descendants. Streamlit's native image-fullscreen overlay is
   `position: fixed` and expects the viewport as its containing block;
   with backdrop-filter here, clicking fullscreen on either photo instead
   traps the expanded view inside this card's own small bounds (the same
   class of bug as the animation fill-mode issue below, different root
   cause). A near-opaque plain background (no blur-through) keeps the same
   visual weight without creating that containing block. */
[class*="st-key-fdx-preprocessing-card"] {{
    background: rgba(255, 255, 255, 0.88);
    border: 1px solid var(--fdx-glass-border);
    border-radius: 16px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.04);
    padding: 0.85rem 0.9rem;
    height: 100%;
    box-sizing: border-box;
}}

/* --- Compact data grid ---------------------------------------------------
   Secondary/detail numbers (see app/components.py's render_datagrid()) --
   headline numbers stay in ring cards or st.metric tiles. The card
   wrapper (.fdx-datagrid-card, added to the shared glass surface list
   above) gives this table a resolved bottom edge instead of trailing off
   after the last row (see render_datagrid()'s docstring), and matches the
   visual weight of the ring card it usually sits beside. `height: 100%` +
   the flex parent's `align-items: stretch` (Streamlit's own column
   default) lets it match its neighboring ring-card's height rather than
   shrink-wrapping to just its own (usually shorter) row count. */
.fdx-datagrid-card {{
    padding: 0.6rem 0.75rem;
    height: 100%;
    box-sizing: border-box;
    display: flex;
    align-items: center;
}}

.fdx-datagrid {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
}}
.fdx-datagrid tr:nth-child(even) {{
    background: rgba(255, 255, 255, 0.35);
}}
.fdx-datagrid td {{
    padding: 0.4rem 0.6rem;
    border-bottom: 1px solid var(--fdx-rule);
}}
.fdx-datagrid tr:last-child td {{
    border-bottom: none;
}}
.fdx-datagrid td:first-child {{
    font-weight: 600;
    color: var(--fdx-text);
}}
.fdx-datagrid td:last-child {{
    font-family: var(--fdx-font-mono);
    text-align: right;
    color: var(--fdx-text);
}}

/* --- Image hover-zoom (Amazon product-page style) --------------------------
   Hover an image and the region under the cursor zooms in, following the
   cursor as it moves, no fullscreen transition needed. The zoom-in
   trigger/reset itself is plain CSS `:hover` (native, instant, no JS
   needed for that part); `inject_image_zoom()`'s JS (a CCv2 component,
   see below) only continuously updates `transform-origin` to the
   cursor's position WITHIN the hovered image via a delegated `mousemove`
   listener, so the scaled-up view tracks whatever the cursor is over
   rather than always zooming toward a fixed center point. Scaling the
   <img> itself (not the wrapper) inside an overflow:hidden card keeps the
   zoom clipped to the original frame, and keeps the caption (a sibling
   under the image, not inside the scaled element) from zooming along
   with it.

   Streamlit's native fullscreen click-to-expand is still technically
   reachable (its own button still renders), and the containing-block fix
   for it (see the fdxFadeInUp comment below) stays in place defensively --
   it's just no longer the primary way to inspect an image closely here. */
div[data-testid="stImage"] {{
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid var(--fdx-glass-border);
}}
div[data-testid="stImage"] img {{
    display: block;
    transform-origin: 50% 50%;
    transition: transform 0.15s ease-out;
    cursor: zoom-in;
}}
div[data-testid="stImage"]:hover img {{
    transform: scale(2.2);
}}

/* --- Pills navigation (st.pills / st.segmented_control) --------------------
   Both widgets share one underlying component, data-testid
   "stButtonGroup". Styled as a rounded segmented track on the glass
   material; the selected-pill accent fill comes from Streamlit's own
   aria-checked/data-selected state on the inner button. */
div[data-testid="stButtonGroup"] {{
    background: var(--fdx-track);
    padding: 0.22rem;
    border-radius: 999px;
    gap: 0.15rem;
}}
div[data-testid="stButtonGroup"] button {{
    border-radius: 999px !important;
    font-size: 0.8rem;
}}

/* --- Fixed header bar -----------------------------------------------------
   The reference mockup wraps the whole app in a fixed top nav (logo +
   settings + export affordance) and a fixed bottom bar, both frosted
   glass "surface-glass" panels. Streamlit already renders its own
   [data-testid="stHeader"] toolbar at the very top; this bar is rendered
   as ordinary content (via st.markdown) positioned fixed just below it
   rather than replacing Streamlit's own header, so the toolbar's own
   menu/deploy controls stay reachable. */
.fdx-header-spacer {{
    height: 3.25rem;
}}

/* `.fdx-header-spacer` only reserves room for the fixed header in NORMAL
   top-to-bottom document flow. It does nothing for scrollIntoView() /
   keyboard-focus scrolling / anchor jumps, which can still land a target
   element flush at the very top of the scroll container -- exactly where
   the fixed header sits, hiding whatever scrolled there underneath it.
   `scroll-padding-top` fixes this at the browser level for any scrolling
   method, not just the ones this app controls directly. Streamlit's real
   scrollable container is `section.stMain`, not `html`/`body`, so that's
   what needs the padding. */
section.stMain {{
    scroll-padding-top: 3.5rem;
}}

.fdx-header {{
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    z-index: 999998;
    height: 3.25rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 1.5rem;
    background: rgba(255, 255, 255, 0.85);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border-bottom: 1px solid var(--fdx-glass-border);
}}

.fdx-header-logo {{
    font-family: var(--fdx-font-display);
    font-weight: 700;
    font-size: 1.25rem;
    color: var(--fdx-primary);
    letter-spacing: -0.01em;
}}

.fdx-header-status {{
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-family: var(--fdx-font-mono);
    font-size: 0.7rem;
    font-weight: 500;
    color: var(--fdx-muted);
    text-transform: uppercase;
    letter-spacing: 0.04em;
}}

.fdx-status-dot {{
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #10B981;
    box-shadow: 0 0 8px rgba(16, 185, 129, 0.5);
}}

/* --- Intake panel -----------------------------------------------------
   The centered "Patient Intake & Signal Acquisition" glass panel shown
   before an image is available (see app/main.py's render_intake_screen())
   -- the reference mockup's one concrete screen, reproduced directly.
   `st.container(key=...)` produces a stable "st-key-<key>" class (same
   mechanism the result sections above rely on for their fade-in
   animation), which is what these two selectors target -- NOT a custom
   class of the same name, since raw HTML classes can't wrap real
   Streamlit widgets (the toggle inside fdx-intake-toggle-row is a real
   st.toggle, not decoration). */
[class*="st-key-fdx-intake-panel"] {{
    background: var(--fdx-glass);
    backdrop-filter: blur(32px);
    -webkit-backdrop-filter: blur(32px);
    border: 1px solid var(--fdx-glass-border);
    border-radius: 24px;
    padding: 2.5rem;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.06);
    max-width: 960px;
    margin: 3vh auto 0;
}}

.fdx-intake-eyebrow {{
    font-family: var(--fdx-font-mono);
    font-size: 0.7rem;
    font-weight: 600;
    color: var(--fdx-primary);
    letter-spacing: 0.2em;
    text-transform: uppercase;
}}

.fdx-intake-description {{
    font-size: 0.95rem;
    line-height: 1.6;
    color: var(--fdx-muted);
    max-width: 32rem;
}}

[class*="st-key-fdx-intake-toggle-row"] {{
    padding: 0.9rem 1rem;
    background: rgba(255, 255, 255, 0.4);
    border: 1px solid var(--fdx-glass-border);
    border-radius: 14px;
}}

.fdx-intake-toggle-label {{
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--fdx-text);
}}

.fdx-intake-toggle-sublabel {{
    font-size: 0.68rem;
    color: var(--fdx-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 500;
}}

/* Streamlit's native toggle already IS a real, working control -- this
   just recolors its track/thumb to the indigo accent instead of
   restyling/replacing the widget (which would risk losing its
   click/keyboard behavior). */
div[data-testid="stCheckbox"] label div[data-testid="stWidgetLabel"] {{
    display: none;
}}
button[aria-checked="true"][role="switch"] {{
    background-color: var(--fdx-primary) !important;
}}

/* Streamlit's file_uploader restyled as the reference's dashed dropzone --
   real drag-and-drop/click-to-browse behavior preserved (this only
   changes CSS, not the widget), just re-skinned so it reads as "Drop
   clinical images here" rather than Streamlit's default uploader chrome.
   This is the SIDEBAR/default look (own dashed border, own background,
   Streamlit's own instructional text kept since nothing else labels it
   there); the intake panel overrides this right below since it wraps the
   same widget in its own dashed box with a custom heading instead. */
[data-testid="stFileUploaderDropzone"] {{
    background: rgba(53, 37, 205, 0.03) !important;
    border: 2px dashed var(--fdx-rule) !important;
    border-radius: 16px !important;
    padding: 2rem !important;
    transition: background 0.2s ease, border-color 0.2s ease;
}}
[data-testid="stFileUploaderDropzone"]:hover {{
    background: rgba(53, 37, 205, 0.06) !important;
    border-color: var(--fdx-primary) !important;
}}
[data-testid="stFileUploaderDropzoneInstructions"] svg {{
    display: none;
}}
[data-testid="stBaseButton-secondary"] {{
    border-radius: 999px !important;
    border-color: rgba(53, 37, 205, 0.25) !important;
    color: var(--fdx-primary) !important;
}}

/* Intake panel's dropzone: ONE dashed box combining a centered icon +
   heading + caption (rendered by main.py's _resolve_image_source()) with
   the actual uploader, matching the reference's single-composition
   dropzone. Stripping the native widget's own border/background/padding
   and hiding its own instructional text lets this outer wrapper be the
   one visible box instead of a doubled-up heading-above-widget look. */
[class*="st-key-fdx-dropzone-wrapper"] {{
    background: rgba(53, 37, 205, 0.03);
    border: 2px dashed var(--fdx-rule);
    border-radius: 16px;
    padding: 1.5rem;
    transition: background 0.2s ease, border-color 0.2s ease;
}}
[class*="st-key-fdx-dropzone-wrapper"]:hover {{
    background: rgba(53, 37, 205, 0.06);
    border-color: var(--fdx-primary);
}}
[class*="st-key-fdx-dropzone-wrapper"] [data-testid="stFileUploaderDropzone"] {{
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}}
[class*="st-key-fdx-dropzone-wrapper"] [data-testid="stFileUploaderDropzoneInstructions"] {{
    display: none;
}}
[class*="st-key-fdx-dropzone-wrapper"] [data-testid="stFileUploaderDropzone"] section > button {{
    margin: 0 auto;
    display: flex;
}}

/* Patient ID input: a floating small uppercase label above the field
   (matching the reference) instead of Streamlit's default label-above
   spacing -- label_visibility="collapsed" is used in main.py and this
   supplies the visual label instead via a preceding markdown span, so the
   real <label> (kept for accessibility) can stay visually hidden without
   an awkward gap where it used to be. */
.fdx-field-label {{
    font-family: var(--fdx-font-mono);
    font-size: 0.65rem;
    font-weight: 600;
    color: var(--fdx-primary);
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 0.35rem;
    display: block;
}}
div[data-testid="stTextInput"] input {{
    background: rgba(255, 255, 255, 0.4) !important;
    border: 1px solid rgba(0, 0, 0, 0.1) !important;
    border-radius: 14px !important;
    height: 3rem;
}}
div[data-testid="stTextInput"] input:focus {{
    border-color: var(--fdx-primary) !important;
    box-shadow: 0 0 0 3px rgba(53, 37, 205, 0.15) !important;
}}

/* --- Loading/progress experience ---------------------------------
   Fixes the "did the site crash?" problem an opaque, unpinned spinner
   had: this banner stays visible regardless of scroll position.

   `position: sticky` does NOT work here: it requires being a DIRECT
   child of the tall block providing its "roaming range" -- even one
   plain, unstyled wrapper div between a sticky element and its tall
   ancestor breaks it. Streamlit always wraps `st.markdown()` output in
   several of its own layers, so a sticky element rendered through a
   normal Streamlit call can never be a direct child of anything tall
   enough.

   `position: fixed` sidesteps that, but has its own gotcha: a fixed
   element spanning the full viewport width silently fails to paint its
   TEXT content in this environment (background/border still render --
   only text disappears). Centering it as a fixed-width floating card via
   `left: 50%; transform: translateX(-50%)` avoids the bug entirely.

   `top: 132px` clears Streamlit's own header toolbar PLUS this theme's own
   fixed .fdx-header bar (60px + 52px + a small gap). A spacer
   (.fdx-progress-banner-spacer, rendered in normal flow right before this)
   reserves room so real content isn't heavily covered when the banner
   first appears. */
.fdx-progress-banner-spacer {{
    height: 3rem;
}}

.fdx-progress-banner {{
    position: fixed;
    top: 132px;
    left: 50%;
    transform: translateX(-50%);
    width: min(880px, calc(100vw - 3rem));
    box-sizing: border-box;
    z-index: 999;
    background: var(--fdx-glass);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border: 1px solid var(--fdx-glass-border);
    border-radius: 16px;
    padding: 0.9rem 1.25rem;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
}}

.fdx-progress-label {{
    font-size: 0.9rem;
    font-weight: 500;
    color: var(--fdx-text);
    margin-bottom: 0.5rem;
    animation: fdxLabelPulse 0.3s ease-out;
}}

.fdx-progress-track {{
    height: 4px;
    border-radius: 2px;
    background-color: var(--fdx-rule);
    overflow: hidden;
}}

.fdx-progress-fill {{
    height: 100%;
    border-radius: 2px;
    background-color: var(--fdx-primary);
    transition: width 0.35s cubic-bezier(0.16, 1, 0.3, 1);
}}

@keyframes fdxLabelPulse {{
    0% {{ opacity: 0.4; transform: translateY(2px); }}
    100% {{ opacity: 1; transform: translateY(0); }}
}}

.fdx-skeleton {{
    margin-bottom: 1.25rem;
}}

.fdx-skeleton-box {{
    border-radius: 10px;
    background: linear-gradient(100deg, var(--fdx-rule) 30%, #F5F6F9 45%, var(--fdx-rule) 60%);
    background-size: 200% 100%;
    animation: fdxShimmer 1.6s ease-in-out infinite;
}}

.fdx-skeleton-title {{
    height: 1.1rem;
    width: 40%;
    margin-bottom: 0.75rem;
}}

.fdx-skeleton-metrics {{
    display: flex;
    gap: 0.75rem;
    margin-bottom: 0.75rem;
}}

.fdx-skeleton-pill {{
    height: 3.5rem;
    flex: 1;
}}

.fdx-skeleton-image {{
    height: 200px;
    width: 100%;
}}

@keyframes fdxShimmer {{
    0% {{ background-position: 200% 0; }}
    100% {{ background-position: -200% 0; }}
}}

/* Result sections fade/slide in gently as they're revealed -- scoped via
   st.container(key=...)'s stable "st-key-<key>" class, so a whole section
   animates as one unit rather than each metric tile animating separately.

   Deliberately NOT `animation-fill-mode: both`: a fill-mode of
   "both"/"forwards" here breaks Streamlit's image fullscreen view. As
   long as a (possibly finished) CSS animation is still holding an element
   at a keyframe that touches `transform` -- even the identity transform,
   even spelled as `transform: none` in the keyframe itself -- Chromium
   keeps treating the element as transformed and gives it a containing
   block for `position: fixed` descendants. Streamlit's fullscreen image
   overlay IS `position: fixed` and expects the viewport as its containing
   block; with fill-mode "both" it gets trapped inside this small section
   card instead. Default fill-mode ("none") avoids the whole class of bug:
   once the 0.4s animation ends, the element fully reverts to its
   un-animated base style, nothing left "holding" a transform. */
[class*="st-key-fdx-section-"] {{
    animation: fdxFadeInUp 0.4s cubic-bezier(0.16, 1, 0.3, 1);
}}

@keyframes fdxFadeInUp {{
    from {{ opacity: 0; transform: translateY(10px); }}
    to {{ opacity: 1; transform: none; }}
}}

.fdx-error-card {{
    background-color: rgba(186, 26, 26, 0.06);
    border: 1px solid rgba(186, 26, 26, 0.25);
    border-radius: 14px;
    padding: 1rem 1.25rem;
    margin-bottom: 1rem;
}}

.fdx-error-title {{
    font-weight: 600;
    color: var(--fdx-text);
    margin-bottom: 0.25rem;
}}

.fdx-error-detail {{
    color: var(--fdx-muted);
    font-size: 0.85rem;
    font-family: var(--fdx-font-mono);
}}

/* --- Disclaimer footer -------------------------------------------------
   Small floating footer, same fixed+centered+bounded-width pattern as
   .fdx-progress-banner above. Bottom-anchored so it never shares vertical
   territory with Streamlit's own header or the progress banner during
   active loading. A generous spacer and a near-opaque background (0.94
   alpha) guard against a shorter browser viewport leaving real content
   (e.g. a detection tile's Grad-CAM cross-reference caption) sitting
   right underneath this footer. */
.fdx-footer-spacer {{
    height: 4.5rem;
}}

.fdx-disclaimer-footer {{
    position: fixed;
    bottom: 1rem;
    left: 50%;
    transform: translateX(-50%);
    width: min(680px, calc(100vw - 3rem));
    box-sizing: border-box;
    z-index: 900;
    background: rgba(255, 255, 255, 0.94);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border: 1px solid var(--fdx-glass-border);
    border-radius: 12px;
    padding: 0.5rem 1rem;
    font-size: 0.78rem;
    color: var(--fdx-muted);
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
    .fdx-header,
    .fdx-header-spacer,
    .fdx-progress-banner,
    .fdx-progress-banner-spacer,
    .fdx-skeleton,
    .fdx-disclaimer-footer,
    .fdx-footer-spacer {{
        display: none !important;
    }}
    .stApp {{
        background-image: none !important;
    }}
    div[data-testid="stMetric"],
    .fdx-ring-card,
    .fdx-stat-tile,
    .fdx-recommendation-card,
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
    "fdx_ambient_cursor",
    js="""
export default function (component) {
    // Lerp toward the cursor rather than snapping straight to it -- an
    // "ambient light drifting to follow you" feel, not a background that
    // visibly jumps on every mouse tick. 50/40 matches the CSS fallback
    // in theme.py's .stApp rule.
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
        document.documentElement.style.setProperty("--fdx-mouse-x", curX.toFixed(2) + "%");
        document.documentElement.style.setProperty("--fdx-mouse-y", curY.toFixed(2) + "%");
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


# Separate component from _AMBIENT_CURSOR above (same "register once at
# import time" pattern, same reasoning) -- distinct concern, kept modular
# rather than folded into one do-everything mousemove handler.
_IMAGE_ZOOM = st.components.v2.component(
    "fdx_image_zoom",
    js="""
export default function (component) {
    // The zoom-in/out TRIGGER is plain CSS `:hover` (see theme.py's
    // `div[data-testid="stImage"]:hover img { transform: scale(2.2) }`)
    // -- instant, no JS needed for that part. This only continuously
    // updates WHERE the zoom centers, via transform-origin, so hovering
    // near an image's top-left shows a magnified top-left rather than
    // always zooming toward a fixed center point (an Amazon-product-page
    // -style magnifier, not just a bigger image). Delegated off
    // `document` (one listener total, not one per image) since Streamlit
    // reruns can replace image elements between interactions.
    const onMove = (e) => {
        const img = e.target.closest('[data-testid="stImage"] img');
        if (!img) return;
        const rect = img.getBoundingClientRect();
        const x = ((e.clientX - rect.left) / rect.width) * 100;
        const y = ((e.clientY - rect.top) / rect.height) * 100;
        img.style.transformOrigin = `${x}% ${y}%`;
    };
    document.addEventListener("mousemove", onMove);

    return () => {
        document.removeEventListener("mousemove", onMove);
    };
}
""",
)


def inject_image_zoom() -> None:
    """Call once per page load, anywhere after inject_css(). Pure JS side
    effect (see _IMAGE_ZOOM above) -- no visible output, no return value
    used.
    """
    _IMAGE_ZOOM()


def render_header() -> None:
    """Fixed top header bar matching the reference mockup's nav. Rendered
    as ordinary content positioned fixed just below Streamlit's own header
    toolbar (see .fdx-header's CSS comment), so Streamlit's own menu/
    deploy controls stay reachable rather than being covered over.

    Logo only, no status pill -- the reference mockup's own status line
    lives in the intake card's footer row (app/main.py's
    render_intake_screen()), not its nav, so that's the one kept.
    """
    st.markdown(
        """<div class="fdx-header-spacer"></div>
<div class="fdx-header">
    <span class="fdx-header-logo">Fundusight</span>
</div>""",
        unsafe_allow_html=True,
    )
