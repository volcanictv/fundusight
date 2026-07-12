"""Phase 9 / redesign: dashboard visual theme.

"Clinical Liquid Glass" -- adopted from a Stitch-generated reference mockup
the user supplied directly (`Front-End Template/stitch_visiondx_retinal_
screening_dashboard/`: DESIGN.md + code.html + screen.png), which redoes
the app's whole visual language, not just a component or two. Replaces the
prior copper/teal "instrument panel" system wholesale:

- Indigo (#3525CD, "primary") replaces teal as the single primary accent --
  buttons, links, the "normal/no finding" semantic, focus rings. A sky-blue
  secondary (#0EA5E9) covers neutral/informational accents. A burnt-orange
  tertiary (#A44100) replaces copper for "a finding is present" -- similar
  warm-attention role, re-hued to the new family rather than dropped.
- Hanken Grotesk (headlines) + Inter (body) + Geist (small mono-ish labels/
  data) replace Fraunces/Inter/JetBrains Mono -- matching the reference's
  actual Google Fonts import exactly (its DESIGN.md prose says Manrope/
  JetBrains Mono, but its own code.html loads Hanken Grotesk/Inter/Geist;
  code.html is the literal rendered artifact behind screen.png, so it's
  ground truth here, not the prose doc).
- Frost-white background (#F7F9FB, was #ECEEF3) with the same glass-card
  mechanics as before (translucent white + blur), just re-tuned toward the
  reference's "luminous edge" look: a soft black hairline border
  (rgba(0,0,0,0.08), the reference's own `border-luminous` token) and a
  large, very soft ambient shadow, rather than the previous saturate-boost
  + white-border combination.
- Material Symbols Outlined icons (settings, cloud_upload, rocket_launch,
  etc.) replace the previous no-icon convention, matching the reference's
  icon usage in its header/dropzone/buttons.

Streamlit has no first-class theming API expressive enough for this (its
config.toml theme only covers a handful of colors), so this injects scoped
CSS instead -- targeting Streamlit's `data-testid` attributes rather than
its generated class names, since those are the one part of its DOM that's
meant to be a stable styling hook across versions.

Deliberately NOT applied to report/pdf.py's ReportLab output -- that's a
separate, print-optimized renderer (ink-conscious, A4) where glass/blur/
gradient treatments would actively work against the stated goal of a clean
printed page. Both renderers still walk the same report/content.py
Section list, so they can never disagree on *content*, only presentation.
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

/* Material Symbols Outlined, in its own @import: this variable font
   registers FOUR axes (opsz, wght, FILL, GRAD) -- an earlier version of
   this rule only requested three (opsz, wght, FILL), which silently
   failed to apply at all, leaving every hand-inserted
   `material-symbols-outlined` span (e.g. the intake dropzone's upload
   icon) rendering as literal fallback text ("cloud_upload") instead of a
   glyph -- confirmed live via screenshot. Also `display=block` here
   specifically, not `swap` like the text fonts above: this is a
   ligature-based icon font, so "swap" briefly (or, if the request ever
   fails, permanently) shows the raw icon NAME as text; "block" hides the
   fallback text instead of showing it. Streamlit's own `icon=":material/
   ...":` shorthand (used on st.button below) bundles its own copy of this
   font and was unaffected by this bug -- only the manually-inserted spans
   were. */
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=block');

:root {{
    --vdx-primary: {_PRIMARY};
    --vdx-primary-container: {_PRIMARY_CONTAINER};
    --vdx-secondary: {_SECONDARY};
    --vdx-tertiary: {_TERTIARY};
    --vdx-error: {_ERROR};
    --vdx-text: {_TEXT};
    --vdx-muted: {_MUTED};
    --vdx-rule: {_OUTLINE};
    --vdx-background: {_BACKGROUND};
    --vdx-glass: {_GLASS};
    --vdx-glass-border: {_GLASS_BORDER};
    --vdx-track: {_TRACK};
    --vdx-font-sans: 'Inter', -apple-system, "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    --vdx-font-display: 'Hanken Grotesk', 'Inter', -apple-system, "Segoe UI", sans-serif;
    --vdx-font-mono: 'Geist', 'Inter', ui-monospace, "SF Mono", Consolas, monospace;
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
   otherwise no-gradients brief. `--vdx-mouse-x`/`--vdx-mouse-y` are written
   by inject_ambient_cursor()'s JS (a CCv2 component, see below) onto
   `documentElement`, so this radial gradient's *position* tracks the
   cursor while its *color* stays strictly monochromatic (white fading to
   the near-white base, no hue) -- a faint light-following effect, not a
   visible color gradient. The `50% 40%` fallback (before the first
   mousemove fires, or with JS disabled) keeps it looking intentional, not
   broken, in that split second.

   The gradient itself lives on `.stApp::before`, a separate fixed
   full-viewport layer BEHIND the real content, rather than directly on
   `.stApp` -- a design-review pass asked for the glow to carry a frosted/
   blurred glass quality, not just a soft-edged gradient. `filter: blur()`
   affects an element's entire rendered output including its children, so
   applying it straight to `.stApp` would blur the whole dashboard, not
   just its background; a dedicated `::before` layer (z-index behind
   everything, pointer-events disabled so it never intercepts clicks) lets
   the blur apply to ONLY the glow itself. `.stApp` keeps the plain
   background-color as a fallback base underneath.

   CRITICAL: do NOT add `position: relative` (or any other value besides
   the default `static`) to `.stApp` to "properly" scope this pseudo-
   element -- it isn't needed (a `position: fixed` child resolves against
   the viewport by default; it does NOT need its parent to be positioned,
   unlike `position: absolute`), and adding it broke the entire app: a
   first version of this rule added `position: relative` here, which made
   `.stApp` the new nearest positioned ancestor for Streamlit's OWN
   `stAppViewContainer` (which is `position: absolute` internally). That
   silently changed stAppViewContainer's containing block, and the
   dashboard rendered fine on first load (likely luck/caching) but went
   completely blank after ANY rerun (confirmed live: toggling the demo-
   mode switch was enough) -- DOM and computed styles all still reported
   normal-looking values, making this very hard to spot; only a git-stash
   bisect back to the last known-good commit conclusively isolated it to
   this one declaration. Left undocumented, a future "cleanup" pass could
   easily re-add it by habit. */
.stApp {{
    background-color: var(--vdx-background);
}}
.stApp::before {{
    content: "";
    position: fixed;
    inset: 0;
    z-index: -1;
    pointer-events: none;
    background-image: radial-gradient(
        circle 1500px at var(--vdx-mouse-x, 50%) var(--vdx-mouse-y, 40%),
        rgba(255, 255, 255, 1) 0%,
        rgba(255, 255, 255, 0) 75%
    );
    filter: blur(48px);
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
    font-family: var(--vdx-font-display) !important;
    font-weight: 700;
    letter-spacing: -0.02em;
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
   blur, a soft black hairline ("luminous edge", the reference's own
   `border-luminous` token) instead of the previous white border, and a
   large, very soft ambient shadow -- the reference's "Light Diffusion"
   elevation model (DESIGN.md: "a very large, very soft shadow... provides
   a subtle lift without appearing heavy"), replacing the old tighter,
   darker shadow. */
div[data-testid="stMetric"],
.vdx-ring-card,
.vdx-stat-tile,
.vdx-datagrid-card,
div[data-testid="stExpander"] {{
    background: var(--vdx-glass);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border: 1px solid var(--vdx-glass-border);
    border-radius: 16px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.04);
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

/* Buttons/links: indigo is the single primary accent now -- restrained
   motion on hover only, never on load/idle. A soft indigo-tinted glow
   shadow (the reference's `shadow-primary/20`) instead of a flat drop
   shadow reads as "the glass itself is lit from within", not just lifted. */
.stButton > button, .stDownloadButton > button {{
    background-color: var(--vdx-primary);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.5rem 1.25rem;
    font-weight: 600;
    box-shadow: 0 8px 20px rgba(53, 37, 205, 0.2);
    transition: opacity 0.15s ease, transform 0.15s ease;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
    opacity: 0.9;
    color: white;
    transform: translateY(-1px);
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
   surface here, plus an indigo left rule (this app's single primary
   accent) to read as "the conclusion", not just another paragraph of
   body text. */
.vdx-recommendation-card {{
    background: var(--vdx-glass);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border: 1px solid var(--vdx-glass-border);
    border-left: 3px solid var(--vdx-primary);
    border-radius: 16px;
    padding: 1rem 1.35rem;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.04);
    margin: 0.25rem 0 1.25rem;
}}

.vdx-recommendation-title {{
    font-family: var(--vdx-font-display);
    font-weight: 700;
    font-size: 1.05rem;
    color: var(--vdx-text);
    margin-bottom: 0.45rem;
}}

.vdx-recommendation-body {{
    font-size: 0.92rem;
    line-height: 1.55;
    color: var(--vdx-text);
}}

/* A visual break from the clinical summary above it -- see
   components.py's render_recommendation_card() for why this is split out
   of the main paragraph: a legal/educational disclaimer trailing off as
   just the last clause of a run-on paragraph reads as an afterthought,
   not the distinct notice it's meant to be. */
.vdx-recommendation-disclaimer {{
    font-size: 0.78rem;
    line-height: 1.5;
    color: var(--vdx-muted);
    border-top: 1px solid var(--vdx-rule);
    margin-top: 0.75rem;
    padding-top: 0.6rem;
}}

/* --- Micro-visualization: instrument-bezel ring gauge --------------------
   One reusable component (see app/components.py's render_ring()),
   parameterized entirely through inline CSS custom properties (--pct
   0-100, --ring-color). A thicker arc + an inset shadow on the inner disc
   suggests a lens/eyepiece bezel rather than a flat progress ring. */
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
   (indigo -- no finding / calm), "attention" (tertiary orange -- a finding
   is present), "info" (secondary sky-blue -- informational, not a status
   verdict). */
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
.vdx-pill-normal {{ background: rgba(53, 37, 205, 0.1); color: var(--vdx-primary); }}
.vdx-pill-attention {{ background: rgba(164, 65, 0, 0.1); color: var(--vdx-tertiary); }}
.vdx-pill-info {{ background: rgba(14, 165, 233, 0.12); color: var(--vdx-secondary); }}

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
    font-family: var(--vdx-font-display);
    font-weight: 700;
    font-size: 0.98rem;
    color: var(--vdx-text);
    line-height: 1.25;
}}

.vdx-stat-tile-body {{
    display: flex;
    align-items: center;
    gap: 0.75rem;
}}

/* Each disease tile's probability-breakdown chart, always visible now
   (see app/main.py's render_detection_section() et al. -- previously
   tucked behind a collapsed st.expander that a design-review pass
   flagged as hiding the most informative view of each tile behind an
   unprompted extra click). Same glass card treatment as the tile above
   it, in its own card rather than the expander's own (now-removed) one. */
[class*="st-key-vdx-chart-"] {{
    background: var(--vdx-glass);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border: 1px solid var(--vdx-glass-border);
    border-radius: 16px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.04);
    padding: 0.6rem 0.75rem 0.75rem;
    margin-top: 0.6rem;
}}

/* The Preprocessing tile (Overview row) -- same glass card LOOK as its
   row neighbor, the Image Quality tile (previously rendered as bare
   images directly on the page background, the one Overview-row element
   still without a card, which a design-review pass flagged as breaking
   the row's visual harmony) -- but deliberately WITHOUT backdrop-filter,
   unlike every other glass card in this file. Confirmed live: this is
   the one glass card that wraps real <img> content (the before/after
   preprocessing photos), and backdrop-filter -- like transform/filter/
   perspective -- makes an element a containing block for `position:
   fixed` descendants. Streamlit's native image-fullscreen overlay is
   `position: fixed` and expects the viewport as its containing block;
   with backdrop-filter here, clicking fullscreen on either photo instead
   trapped the expanded view inside this card's own small bounds -- the
   exact bug an earlier pass fixed for a different root cause (an
   animation's residual transform), now reproduced by a different
   property on a card that didn't exist yet at the time. A near-opaque
   plain background (no blur-through) keeps the same visual weight
   without creating that containing block. */
[class*="st-key-vdx-preprocessing-card"] {{
    background: rgba(255, 255, 255, 0.88);
    border: 1px solid var(--vdx-glass-border);
    border-radius: 16px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.04);
    padding: 0.85rem 0.9rem;
    height: 100%;
    box-sizing: border-box;
}}

/* --- Compact data grid ---------------------------------------------------
   Secondary/detail numbers (see app/components.py's render_datagrid()) --
   headline numbers stay in ring cards or st.metric tiles. The card
   wrapper (.vdx-datagrid-card, added to the shared glass surface list
   above) is what gives this table a resolved bottom edge instead of
   trailing off after the last row (see render_datagrid()'s docstring for
   the full story) and matches the visual weight of the ring card it
   usually sits beside. `height: 100%` + the flex parent's
   `align-items: stretch` (Streamlit's own column default) lets it match
   its neighboring ring-card's height rather than shrink-wrapping to just
   its own (usually shorter) row count. */
.vdx-datagrid-card {{
    padding: 0.6rem 0.75rem;
    height: 100%;
    box-sizing: border-box;
    display: flex;
    align-items: center;
}}

.vdx-datagrid {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
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

/* --- Image hover-zoom (Amazon product-page style) --------------------------
   Replaces an earlier, much milder hover effect (a flat 1.04x scale, no
   cursor tracking) -- a design-review pass asked for a real magnifier:
   hover an image and the region under the cursor zooms in, following the
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
   reachable (its own button still renders) and the earlier containing-
   block fix for it (see the vdxFadeInUp comment below) is deliberately
   left in place defensively -- it's just no longer the primary way to
   inspect an image closely here. */
div[data-testid="stImage"] {{
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid var(--vdx-glass-border);
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

/* --- Fixed header bar -----------------------------------------------------
   New in this pass -- the reference mockup wraps the whole app in a fixed
   top nav (logo + settings + export affordance) and a fixed bottom bar,
   both frosted glass "surface-glass" panels. Streamlit already renders its
   own [data-testid="stHeader"] toolbar at the very top; this bar is
   rendered as ordinary content (via st.markdown) positioned fixed just
   below it rather than replacing Streamlit's own header, so the toolbar's
   own menu/deploy controls stay reachable. */
.vdx-header-spacer {{
    height: 3.25rem;
}}

/* `.vdx-header-spacer` only reserves room for the fixed header in NORMAL
   top-to-bottom document flow. It does nothing for scrollIntoView() /
   keyboard-focus scrolling / anchor jumps, which can still land a target
   element flush at the very top of the scroll container -- exactly where
   the fixed header sits, hiding whatever scrolled there underneath it
   (confirmed live: an automated scrollIntoView() on a button below the
   fold left it positioned under this header, intercepting clicks meant
   for it). `scroll-padding-top` fixes this at the browser level for any
   scrolling method, not just the ones this app controls directly.
   Streamlit's real scrollable container is `section.stMain`, not
   `html`/`body` (confirmed live during an earlier investigation into
   this app's fullscreen-image bug), so that's what needs the padding. */
section.stMain {{
    scroll-padding-top: 3.5rem;
}}

.vdx-header {{
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
    border-bottom: 1px solid var(--vdx-glass-border);
}}

.vdx-header-logo {{
    font-family: var(--vdx-font-display);
    font-weight: 700;
    font-size: 1.25rem;
    color: var(--vdx-primary);
    letter-spacing: -0.01em;
}}

.vdx-header-status {{
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-family: var(--vdx-font-mono);
    font-size: 0.7rem;
    font-weight: 500;
    color: var(--vdx-muted);
    text-transform: uppercase;
    letter-spacing: 0.04em;
}}

.vdx-status-dot {{
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #10B981;
    box-shadow: 0 0 8px rgba(16, 185, 129, 0.5);
}}

/* --- Sidebar ---------------------------------------------------------
   A design-review pass caught this: every other surface (the intake
   panel, result tiles, the recommendation card) got the glass treatment,
   but the persistent sidebar (explainability method once results exist,
   or the full compact input set) was left as bare default Streamlit --
   a flat gray panel with plain black-bordered widgets, no blur, no accent
   color on focus. Since the sidebar is visible on every screen, that gap
   undermined the "whole frontend" redesign more than any single
   component would. Same glass material as the rest of the app, and its
   own selectbox/text input/toggle re-themed to match what main.py's
   intake-panel versions of those same widgets already look like. */
[data-testid="stSidebar"] {{
    background: var(--vdx-glass);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border-right: 1px solid var(--vdx-glass-border);
}}
[data-testid="stSidebar"] [data-testid="stTextInput"] input,
[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] > div {{
    background: rgba(255, 255, 255, 0.6) !important;
    border: 1px solid rgba(0, 0, 0, 0.1) !important;
    border-radius: 12px !important;
}}
[data-testid="stSidebar"] [data-testid="stTextInput"] input:focus {{
    border-color: var(--vdx-primary) !important;
    box-shadow: 0 0 0 3px rgba(53, 37, 205, 0.15) !important;
}}
[data-testid="stSidebar"] button[aria-checked="true"][role="switch"] {{
    background-color: var(--vdx-primary) !important;
}}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {{
    background: rgba(53, 37, 205, 0.03) !important;
    border: 2px dashed var(--vdx-rule) !important;
    border-radius: 14px !important;
}}

/* --- Intake panel -----------------------------------------------------
   The centered "Patient Intake & Signal Acquisition" glass panel shown
   before an image is available (see app/main.py's render_intake_screen())
   -- this IS the reference mockup's one concrete screen, reproduced
   directly rather than reinterpreted. `st.container(key=...)` produces a
   stable "st-key-<key>" class (confirmed live, same mechanism the result
   sections above already rely on for their fade-in animation), which is
   what these two selectors target -- NOT a custom class of the same name,
   since raw HTML classes can't wrap real Streamlit widgets (the toggle
   inside vdx-intake-toggle-row is a real st.toggle, not decoration). */
[class*="st-key-vdx-intake-panel"] {{
    background: var(--vdx-glass);
    backdrop-filter: blur(32px);
    -webkit-backdrop-filter: blur(32px);
    border: 1px solid var(--vdx-glass-border);
    border-radius: 24px;
    padding: 2.5rem;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.06);
    max-width: 960px;
    margin: 3vh auto 0;
}}

.vdx-intake-eyebrow {{
    font-family: var(--vdx-font-mono);
    font-size: 0.7rem;
    font-weight: 600;
    color: var(--vdx-primary);
    letter-spacing: 0.2em;
    text-transform: uppercase;
}}

.vdx-intake-description {{
    font-size: 0.95rem;
    line-height: 1.6;
    color: var(--vdx-muted);
    max-width: 32rem;
}}

[class*="st-key-vdx-intake-toggle-row"] {{
    padding: 0.9rem 1rem;
    background: rgba(255, 255, 255, 0.4);
    border: 1px solid var(--vdx-glass-border);
    border-radius: 14px;
}}

.vdx-intake-toggle-label {{
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--vdx-text);
}}

.vdx-intake-toggle-sublabel {{
    font-size: 0.68rem;
    color: var(--vdx-muted);
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
    background-color: var(--vdx-primary) !important;
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
    border: 2px dashed var(--vdx-rule) !important;
    border-radius: 16px !important;
    padding: 2rem !important;
    transition: background 0.2s ease, border-color 0.2s ease;
}}
[data-testid="stFileUploaderDropzone"]:hover {{
    background: rgba(53, 37, 205, 0.06) !important;
    border-color: var(--vdx-primary) !important;
}}
[data-testid="stFileUploaderDropzoneInstructions"] svg {{
    display: none;
}}
[data-testid="stBaseButton-secondary"] {{
    border-radius: 999px !important;
    border-color: rgba(53, 37, 205, 0.25) !important;
    color: var(--vdx-primary) !important;
}}

/* Intake panel's dropzone: ONE dashed box combining a centered icon +
   heading + caption (rendered by main.py's _resolve_image_source()) with
   the actual uploader -- a design-review pass found the previous version
   (custom heading sitting ABOVE a separately-bordered native widget, with
   Streamlit's own "Drag and drop file here" text still showing inside
   it) read as an unfinished, doubled-up translation of the reference's
   single-composition dropzone. Stripping the native widget's own border/
   background/padding and hiding its own instructional text lets this
   outer wrapper be the one visible box instead. */
[class*="st-key-vdx-dropzone-wrapper"] {{
    background: rgba(53, 37, 205, 0.03);
    border: 2px dashed var(--vdx-rule);
    border-radius: 16px;
    padding: 1.5rem;
    transition: background 0.2s ease, border-color 0.2s ease;
}}
[class*="st-key-vdx-dropzone-wrapper"]:hover {{
    background: rgba(53, 37, 205, 0.06);
    border-color: var(--vdx-primary);
}}
[class*="st-key-vdx-dropzone-wrapper"] [data-testid="stFileUploaderDropzone"] {{
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}}
[class*="st-key-vdx-dropzone-wrapper"] [data-testid="stFileUploaderDropzoneInstructions"] {{
    display: none;
}}
[class*="st-key-vdx-dropzone-wrapper"] [data-testid="stFileUploaderDropzone"] section > button {{
    margin: 0 auto;
    display: flex;
}}

/* Patient ID input: a floating small uppercase label above the field
   (matching the reference) instead of Streamlit's default label-above
   spacing -- label_visibility="collapsed" is used in main.py and this
   supplies the visual label instead via a preceding markdown span, so the
   real <label> (kept for accessibility) can stay visually hidden without
   an awkward gap where it used to be. */
.vdx-field-label {{
    font-family: var(--vdx-font-mono);
    font-size: 0.65rem;
    font-weight: 600;
    color: var(--vdx-primary);
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
    border-color: var(--vdx-primary) !important;
    box-shadow: 0 0 0 3px rgba(53, 37, 205, 0.15) !important;
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

   `top: 132px` clears Streamlit's own header toolbar PLUS this theme's own
   fixed .vdx-header bar (60px + 52px + a small gap). A spacer
   (.vdx-progress-banner-spacer, rendered in normal flow right before this)
   reserves room so real content isn't heavily covered when the banner
   first appears. */
.vdx-progress-banner-spacer {{
    height: 3rem;
}}

.vdx-progress-banner {{
    position: fixed;
    top: 132px;
    left: 50%;
    transform: translateX(-50%);
    width: min(880px, calc(100vw - 3rem));
    box-sizing: border-box;
    z-index: 999;
    background: var(--vdx-glass);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border: 1px solid var(--vdx-glass-border);
    border-radius: 16px;
    padding: 0.9rem 1.25rem;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
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
    background-color: var(--vdx-primary);
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
    background-color: rgba(186, 26, 26, 0.06);
    border: 1px solid rgba(186, 26, 26, 0.25);
    border-radius: 14px;
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
   banner during active loading.

   Spacer grown from 2.5rem -> 4.5rem and the footer's own background
   pushed to near-opaque (0.72 -> 0.94 alpha) after a design-review pass
   found real content (a detection tile's Grad-CAM cross-reference
   caption) sitting only ~15px above this footer at the bottom of the
   page -- on a shorter browser viewport that's a real overlap risk, not
   just a tight-but-fine gap. Both changes independently reduce that risk:
   more reserved space below the last real content, and a background
   solid enough to fully mask anything that still ends up underneath it. */
.vdx-footer-spacer {{
    height: 4.5rem;
}}

.vdx-disclaimer-footer {{
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
    .vdx-header,
    .vdx-header-spacer,
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


# Separate component from _AMBIENT_CURSOR above (same "register once at
# import time" pattern, same reasoning) -- distinct concern, kept modular
# rather than folded into one do-everything mousemove handler.
_IMAGE_ZOOM = st.components.v2.component(
    "vdx_image_zoom",
    js="""
export default function (component) {
    // The zoom-in/out TRIGGER is plain CSS `:hover` (see theme.py's
    // `div[data-testid="stImage"]:hover img { transform: scale(2.2) }`)
    // -- instant, no JS needed for that part. This only continuously
    // updates WHERE the zoom centers, via transform-origin, so hovering
    // near an image's top-left shows a magnified top-left rather than
    // always zooming toward a fixed center point regardless of cursor
    // position (an Amazon-product-page-style magnifier, not just a
    // bigger image). Delegated off `document` (one listener total, not
    // one per image) since Streamlit reruns can replace image elements
    // between interactions.
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
    """Fixed top header bar -- new in this pass, matching the reference
    mockup's nav. Rendered as ordinary content positioned fixed just below
    Streamlit's own header toolbar (see .vdx-header's CSS comment), so
    Streamlit's own menu/deploy controls stay reachable rather than being
    covered over.

    Logo only -- no status pill here. A first pass duplicated "Engine
    online" in the header AND "Core engine ready" at the bottom of the
    intake panel (app/main.py's render_intake_screen()), two status
    indicators with different wording for what read as the same
    underlying state. The reference mockup's own status line lives in the
    intake card's footer row, not its nav, so that's the one kept; this
    header also has no gear/export/avatar to visually balance a status
    pill against (this app has no settings/profile screen those would
    open), so a bare wordmark reads as deliberate rather than unfinished.
    """
    st.markdown(
        """<div class="vdx-header-spacer"></div>
<div class="vdx-header">
    <span class="vdx-header-logo">VisionDx</span>
</div>""",
        unsafe_allow_html=True,
    )
