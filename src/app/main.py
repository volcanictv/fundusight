"""Phase 9 / redesign: Streamlit dashboard.

Ties every pipeline stage together: upload (or demo mode) -> quality ->
preprocessing preview -> DR/glaucoma/AMD detection + Grad-CAM each ->
vessel biomarkers -> optic disc/cup/CDR -> a recommendation summary -> PDF
download.

Redesign pass: the page is now a dense multi-column dashboard instead of
seven full-width stacked sections. Three ROW groups (Overview / Disease
Screening / Biomarkers) each hold their stages side by side in columns.
The Disease Screening row in particular replaces what used to be three
near-identical full sections (subheader + pill + ring + datagrid +
full-size Grad-CAM image, once each for DR/glaucoma/AMD) with three
compact tiles -- the Grad-CAM images moved to the existing Image Comparison
viewer at the bottom instead of repeating three times inline.

Second redesign pass: the page used to ALSO render a full second
"Report Preview" walk through report/content.py's Section list -- quality
metrics, preprocessing before/after, each detection's text+table+image,
vessel/optic-disc metrics+image -- immediately below everything above,
duplicating essentially all of it a second time (the whole point of that
walk was "verify what's in the PDF before downloading it", but by this
point every number and image it showed already exists on the page). That
walk is gone. What replaces it: Glaucoma/AMD tiles gained the same
probability-breakdown expander DR already had (previously that split was
only visible in the deleted walk-through's table), and the "Recommendation"
text -- the one genuinely unique thing that walk had -- now renders directly
via render_recommendation_card() (components.py) instead of being buried at
the bottom of a page-length duplicate. app/render_preview.py is deleted
along with it; nothing else imported its Section-walking helpers.

Third redesign pass ("Clinical Liquid Glass"): the whole visual language --
colors, fonts, glass treatment, icons -- was replaced wholesale to match a
Stitch-generated reference mockup the user supplied directly (`Front-End
Template/stitch_visiondx_retinal_screening_dashboard/`), not iterated on
from scratch. See theme.py's module docstring for the token-level
rationale. Structurally, this pass also replaces the old sidebar-only
intake controls (patient ID / demo toggle / uploader, permanently pinned
in `st.sidebar`) with a centered "Patient Intake & Signal Acquisition"
glass panel (render_intake_screen(), below) shown before an image is
available -- the reference mockup's one concrete screen, reproduced
directly. The sidebar still holds one setting (explainability method)
that only matters once results exist, so it stays there rather than
cluttering the intake panel with a control that isn't relevant yet.

v2 (kept): the pipeline runs progressively, not behind one opaque spinner.
Each stage still gets its own st.empty() placeholder, filled the moment it
finishes (via report/pipeline.run_pipeline()'s on_stage callback), behind a
sticky progress banner. See app/progress.py.

Run with (from the repo root, matching this project's Windows venv
convention -- see README):

    .venv\\Scripts\\python.exe -m streamlit run src/app/main.py

Everything here only reads the documented dict/dataclass keys the pipeline
modules expose (see report/pipeline.py, report/content.py) -- never a
specific accuracy number or checkpoint detail, so a future retrained
checkpoint (see ROADMAP.md's Phase 6 note) needs no changes on this page.
"""

import hashlib
import html
import re

import cv2
import numpy as np
import streamlit as st

from src.app.charts import binary_probability_chart, probability_bar_chart
from src.app.components import render_datagrid, render_recommendation_card, render_ring, render_stat_tile
from src.app.demo_data import list_demo_images, load_demo_image
from src.app.progress import ProgressBanner, render_error_card, render_skeleton
from src.app.theme import inject_ambient_cursor, inject_css, inject_image_zoom, render_header
from src.detection.amd_infer import AMD_LABELS
from src.detection.glaucoma_infer import GLAUCOMA_LABELS
from src.explainability.gradcam import CAM_METHODS
from src.report import overlays
from src.report.content import build_report_content
from src.report.pdf import generate_pdf
from src.report.pipeline import run_pipeline

st.set_page_config(page_title="VisionDx", page_icon="\U0001f441", layout="wide")
inject_css()
inject_ambient_cursor()
inject_image_zoom()
render_header()

# CSS custom-property references, not hex literals -- theme.py's :root is
# the single source of truth for these two accent colors; every caller here
# just points at it rather than duplicating hex values.
_PRIMARY = "var(--vdx-primary)"
_TERTIARY = "var(--vdx-tertiary)"


def _to_rgb(array: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(array, cv2.COLOR_BGR2RGB)


def _decode_upload(uploaded_file) -> np.ndarray:
    file_bytes = np.frombuffer(uploaded_file.getvalue(), np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


def _safe_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_") or "report"


def _tile_label(text: str) -> None:
    # Reuses the same serif micro-heading style render_stat_tile()'s own
    # title uses (see theme.py's .vdx-stat-tile-title) so every column in
    # the dense grid below -- disease tiles included -- reads as one
    # consistent family of "instrument readouts," not a mix of full
    # st.subheader blocks and small ad-hoc labels.
    st.markdown(f'<div class="vdx-stat-tile-title">{html.escape(text)}</div>', unsafe_allow_html=True)


def _unavailable_tile(title: str) -> None:
    st.markdown(
        f'<div class="vdx-stat-tile"><div class="vdx-stat-tile-title">{html.escape(title)}</div>'
        f'<div class="vdx-caption">Model not available in this build — no trained checkpoint was found.</div></div>',
        unsafe_allow_html=True,
    )


# --- Per-section renderers. Each is exactly what used to be inline code
# here before v2 -- extracted so both the progressive (cache-miss) and
# direct (cache-hit) render paths below can call the same functions. ---


def render_quality_section(quality: dict) -> None:
    _tile_label("Image Quality")
    color = _PRIMARY if quality["passed"] else _TERTIARY
    ring_col, grid_col = st.columns([1, 1])
    with ring_col:
        render_ring("Quality", f"{quality['score']:.0f}", quality["score"], color=color)
    with grid_col:
        render_datagrid(
            [
                ("Focus", f"{quality['checks']['focus']['score']:.0f}/100"),
                ("Exposure", f"{quality['checks']['exposure']['score']:.0f}/100"),
                ("Passed", "Yes" if quality["passed"] else "No"),
            ]
        )
    if not quality["passed"]:
        st.caption("Findings below may be less reliable — consider retaking the photo.")


def render_preprocessing_section(preview: dict) -> None:
    # Side-by-side before/after -- a single "after" image (this tile's
    # shape before the Report Preview merge) meant the only way to actually
    # compare original vs. preprocessed was flipping between two pills in
    # Image Comparison one at a time, with nothing to hold the earlier
    # frame against. Showing both here restores a simultaneous comparison
    # without bringing back a full duplicate walk -- Image Comparison still
    # covers the same two images individually at full size for closer
    # inspection, this is just the side-by-side "what changed" view.
    # Wrapped in the same glass card treatment as its Overview-row
    # neighbor (the Image Quality ring+datagrid) -- a design-review pass
    # flagged the Overview row's tiles as visually inconsistent, and this
    # tile was the one still rendering as bare images/text directly on
    # the page background instead of its own card.
    with st.container(key="vdx-preprocessing-card"):
        _tile_label("Preprocessing")
        before_col, after_col = st.columns(2)
        with before_col:
            st.image(_to_rgb(preview["before"]), caption="Original", width="stretch")
        with after_col:
            st.image(_to_rgb(preview["after"]), caption="Illumination + CLAHE + color norm.", width="stretch")


def render_detection_section(detection: dict | None, cam_overlay) -> None:
    if detection is None:
        _unavailable_tile("Diabetic Retinopathy")
        return
    variant = "normal" if detection["class_idx"] == 0 else "attention"
    color = _PRIMARY if variant == "normal" else _TERTIARY
    render_stat_tile(
        "Diabetic Retinopathy",
        detection["label"],
        variant,
        f"{detection['probability'] * 100:.0f}%",
        detection["probability"] * 100,
        ring_color=color,
    )
    # Severity breakdown is now always visible, not tucked behind a
    # collapsed expander -- a design-review pass flagged that the most
    # informative view of a disease tile (the actual probability
    # distribution) required an extra click nobody was prompted to make.
    # DR is 5-class (not binary like glaucoma/AMD) -- genuinely has more to
    # show, so it alone gets a 5-row chart rather than forcing every
    # disease tile into an identical shape.
    with st.container(key="vdx-chart-dr"):
        st.plotly_chart(probability_bar_chart(detection), width="stretch", config={"displayModeBar": False})
        if cam_overlay is not None:
            # The tile itself only ever showed the pill+ring+chart -- the
            # actual Grad-CAM heatmap lives in Image Comparison further
            # down the page (see render_image_comparison()), with nothing
            # on this tile pointing there. A one-line cross-reference beats
            # making the reader already know it moved.
            st.caption('Grad-CAM overlay: see "Grad-CAM (DR)" in Image Comparison below.')


def render_glaucoma_section(glaucoma: dict | None, cam_overlay) -> None:
    if glaucoma is None:
        _unavailable_tile("Glaucoma")
        return
    variant = "normal" if glaucoma["class_idx"] == 0 else "attention"
    color = _PRIMARY if variant == "normal" else _TERTIARY
    render_stat_tile(
        "Glaucoma",
        glaucoma["label"],
        variant,
        f"{glaucoma['probability'] * 100:.0f}%",
        glaucoma["probability"] * 100,
        ring_color=color,
    )
    # Parity with DR's breakdown chart -- this used to be the one place
    # glaucoma's absent/present probability split was visible at all
    # (buried in the old Report Preview walk, now merged away). Binary
    # classifiers don't need DR's ordinal 5-row chart, just this 2-row
    # one. Always visible, not behind an expander -- see render_detection_
    # section()'s comment for why.
    with st.container(key="vdx-chart-glaucoma"):
        st.plotly_chart(
            binary_probability_chart(glaucoma, GLAUCOMA_LABELS), width="stretch", config={"displayModeBar": False}
        )
        if cam_overlay is not None:
            st.caption('Grad-CAM overlay: see "Grad-CAM (Glaucoma)" in Image Comparison below.')


def render_amd_section(amd: dict | None, cam_overlay) -> None:
    if amd is None:
        _unavailable_tile("AMD")
        return
    variant = "normal" if amd["class_idx"] == 0 else "attention"
    color = _PRIMARY if variant == "normal" else _TERTIARY
    render_stat_tile(
        "AMD",
        amd["label"],
        variant,
        f"{amd['probability'] * 100:.0f}%",
        amd["probability"] * 100,
        ring_color=color,
    )
    with st.container(key="vdx-chart-amd"):
        st.plotly_chart(binary_probability_chart(amd, AMD_LABELS), width="stretch", config={"displayModeBar": False})
        if cam_overlay is not None:
            st.caption('Grad-CAM overlay: see "Grad-CAM (AMD)" in Image Comparison below.')


def render_vessel_section(vessel_result: dict, working_image: np.ndarray) -> None:
    # No inline overlay image here -- same density pattern the Disease
    # Screening tiles already use: the vessel mask overlay is one of the
    # views in the Image Comparison pills viewer below, so showing it again
    # here would be the exact repeated-full-size-image redundancy that
    # redesign already fixed for DR/glaucoma/AMD, just not carried through
    # to this section yet.
    _tile_label("Vessel Biomarkers")
    ring_col, grid_col = st.columns([1, 2])
    with ring_col:
        render_ring("Density", f"{vessel_result['vessel_density']:.1f}%", vessel_result["vessel_density"], color=_PRIMARY)
    with grid_col:
        render_datagrid(
            [
                ("Branch points", str(vessel_result["branch_count"])),
                ("Tortuosity", f"{vessel_result['tortuosity']:.3f}"),
                ("Avg. width", f"{vessel_result['average_width']:.2f} px"),
            ]
        )


def render_optic_disc_section(optic_disc_result: dict, working_image: np.ndarray) -> None:
    # Same redundancy fix as render_vessel_section above -- the disc/cup/
    # macula overlay is already one of the Image Comparison views below, so
    # it no longer repeats inline here.
    _tile_label("Optic Disc / Cup / Macula")
    cdr = optic_disc_result["vertical_cdr"]
    # Same 0.5 elevated-CDR threshold report/content.py's recommendation
    # text already uses -- an educational observation, not a diagnosis.
    ring_color = _TERTIARY if cdr >= 0.5 else _PRIMARY
    ring_col, grid_col = st.columns([1, 2])
    with ring_col:
        render_ring("Vertical CDR", f"{cdr:.2f}", cdr * 100, color=ring_color)
    with grid_col:
        render_datagrid(
            [
                ("Disc diameter", f"{optic_disc_result['disc_diameter_px']} px"),
                ("Cup diameter", f"{optic_disc_result['cup_diameter_px']} px"),
            ]
        )
    if not optic_disc_result["disc_found"] or optic_disc_result["disc_diameter_px"] == 0:
        # disc_found only reflects Stage 6.1's classical localization
        # succeeding -- Stage 6.2's segmentation can still independently
        # come back empty (a real, observed failure mode on out-of-domain
        # input with the current provisional checkpoint, see ROADMAP.md's
        # Phase 6 note), which disc_found alone wouldn't catch.
        st.warning("Optic disc could not be confidently segmented in this image — cup/disc measurements above are not meaningful.")


_MAX_COMPARISON_IMAGES_PER_ROW = 3


def render_image_comparison(result: dict) -> None:
    """A multi-select image viewer -- pick two or more views and see them
    side by side simultaneously, via st.pills in multi-selection mode
    (real Streamlit widget state/rerun handling -- not inert custom HTML
    buttons, which can't communicate back to Python without a full custom
    component). Reuses arrays already computed elsewhere in `result`/
    overlays.* rather than recomputing anything. The ONLY place Grad-CAM
    overlays are shown at full size -- the disease tiles above no longer
    repeat them.

    Replaces an earlier single-selection version of this viewer (one pill
    active at a time, one image shown) -- a design-review pass pointed out
    that comparing two views meant memorizing one while looking at the
    other, not an actual side-by-side comparison. Multi-selection with
    row-chunked columns (_MAX_COMPARISON_IMAGES_PER_ROW) handles any
    number of simultaneously selected views without any of them getting
    too narrow to read. Hover any image for a magnified view (theme.py's
    .vdx-zoom-* rules, Amazon-product-page style) instead of Streamlit's
    native fullscreen click-to-expand, which is no longer the primary way
    to inspect these closely (the earlier fullscreen containing-block fix
    stays in theme.py regardless, since fullscreen is still technically
    reachable).

    Also hosts the "Explainability method" selectbox -- it used to live in
    the (now-removed) sidebar; this is the one place its choice actually
    has a visible effect (which Grad-CAM overlays exist to pick from
    below), so it's more discoverable here than buried in a settings
    expander elsewhere on the page.

    Deliberately called only once the FULL pipeline result is available
    (see main flow below) rather than from inside on_stage -- some of
    these images (Grad-CAM, the two overlays) don't exist yet mid-pipeline,
    and threading "which images are ready so far" through a progressively
    updating pills widget isn't worth the complexity for a comparison view
    that's naturally a "look at everything together" step anyway, same
    reasoning as why the Report Preview section only appears once
    `result` is final.
    """
    header_col, method_col = st.columns([3, 1], vertical_alignment="bottom")
    with header_col:
        st.subheader("Image Comparison")
    with method_col:
        st.selectbox("Explainability method", options=list(CAM_METHODS), index=0, key="cam_method")

    images = {
        "Original": result["preprocessing_preview"]["before"],
        "Preprocessed": result["preprocessing_preview"]["after"],
    }
    if result["cam_overlay"] is not None:
        images["Grad-CAM (DR)"] = result["cam_overlay"]
    if result["glaucoma_cam_overlay"] is not None:
        images["Grad-CAM (Glaucoma)"] = result["glaucoma_cam_overlay"]
    if result["amd_cam_overlay"] is not None:
        images["Grad-CAM (AMD)"] = result["amd_cam_overlay"]
    images["Vessel mask"] = overlays.vessel_mask_overlay(result["working_image"], result["vessels"])
    images["Optic disc"] = overlays.optic_disc_overlay(result["working_image"], result["optic_disc"])

    options = list(images)
    default_selection = options[:2]
    st.caption("Select two or more views to compare them side by side. Hover an image to magnify.")
    selected = st.pills(
        "Compare views", options, selection_mode="multi", default=default_selection, key="image_compare"
    )
    if not selected:
        selected = default_selection

    for start in range(0, len(selected), _MAX_COMPARISON_IMAGES_PER_ROW):
        chunk = selected[start : start + _MAX_COMPARISON_IMAGES_PER_ROW]
        columns = st.columns(len(chunk))
        for column, name in zip(columns, chunk):
            with column:
                st.image(_to_rgb(images[name]), caption=name, width="stretch")


# stage_name -> (render function, extractor from the finished result dict).
# Single source of truth for both render paths below, and for on_stage's
# dispatch -- the on_stage callback value is already shaped to match each
# render function's positional args directly (see pipeline.run_pipeline's
# docstring), so both paths call `fn(*args)` against the same functions.
_SECTIONS = [
    ("quality", render_quality_section, lambda r: (r["quality"],)),
    ("preprocessing", render_preprocessing_section, lambda r: (r["preprocessing_preview"],)),
    ("detection", render_detection_section, lambda r: (r["detection"], r["cam_overlay"])),
    ("glaucoma", render_glaucoma_section, lambda r: (r["glaucoma"], r["glaucoma_cam_overlay"])),
    ("amd", render_amd_section, lambda r: (r["amd"], r["amd_cam_overlay"])),
    ("vessels", render_vessel_section, lambda r: (r["vessels"], r["working_image"])),
    ("optic_disc", render_optic_disc_section, lambda r: (r["optic_disc"], r["working_image"])),
]
_RENDER_BY_STAGE = {stage: fn for stage, fn, _ in _SECTIONS}
_EXTRACT_BY_STAGE = {stage: extract for stage, _, extract in _SECTIONS}

# Visual grouping only -- pipeline.py's STAGE_NAMES/on_stage order is
# unchanged, this just says which stages share one dashboard row and how
# many columns that row gets. The repetition fix lives here: Disease
# Screening puts what used to be three full-width sections into one row of
# three compact tiles.
_ROWS = [
    ("Overview", ["quality", "preprocessing"]),
    ("Disease Screening", ["detection", "glaucoma", "amd"]),
    ("Biomarkers", ["vessels", "optic_disc"]),
]


def _create_stage_placeholders() -> dict:
    """One st.empty() per stage, arranged into the _ROWS grid, filled
    immediately with a skeleton -- same "whole results area shows loading
    shape at once" goal as before, just laid out densely instead of
    stacked full-width.
    """
    placeholders = {}
    for title, stages in _ROWS:
        st.subheader(title)
        cols = st.columns(len(stages))
        for col, stage in zip(cols, stages):
            with col:
                placeholders[stage] = st.empty()
                with placeholders[stage].container(key=f"vdx-section-{stage}-skeleton"):
                    render_skeleton(stage)
    return placeholders


def _render_all_sections(result: dict) -> None:
    """Cache-hit path: redraw the cached result directly into the same
    _ROWS grid, no staged reveal or skeletons (there's no actual loading
    happening).
    """
    for title, stages in _ROWS:
        st.subheader(title)
        cols = st.columns(len(stages))
        for col, stage in zip(cols, stages):
            with col:
                with st.container(key=f"vdx-section-{stage}-content"):
                    _RENDER_BY_STAGE[stage](*_EXTRACT_BY_STAGE[stage](result))


# A floating footer instead of an inline caption -- doesn't interrupt the
# page's flow. Fixed + centered + bounded-width, same proven pattern as
# the progress banner (see theme.py's comment on .vdx-progress-banner for
# why -- an edge-to-edge fixed element silently fails to paint its text
# in this environment).
st.markdown(
    '<div class="vdx-disclaimer-footer">AI-assisted retinal disease analysis pipeline '
    "— educational/portfolio demonstration, not a diagnostic device.</div>",
    unsafe_allow_html=True,
)



def _resolve_image_source(*, container) -> tuple[str, np.ndarray | None]:
    """Demo-sample picker or file uploader, whichever `demo_mode` (already
    rendered by the caller) selects. `container` is `st` (main area) or
    `st.sidebar` -- same two widgets either way, just placed differently
    depending on render_intake_screen() vs. the compact sidebar path below.
    """
    patient_id_input = st.session_state.get("patient_id", "")
    effective_patient_id = patient_id_input
    image = None

    if st.session_state.get("demo_mode"):
        demo_images = list_demo_images()
        if not demo_images:
            container.info("No local demo images found — download APTOS 2019 per the README to use demo mode.")
        else:
            options = {f"{item['label']} — {item['id_code']}": item for item in demo_images}
            choice = container.selectbox("Sample image", list(options), key="demo_sample")
            selected = options[choice]
            image = load_demo_image(selected["path"])
            if not effective_patient_id:
                effective_patient_id = f"DEMO-{selected['id_code']}"
    elif container is st:
        # The intake panel composes its own icon + heading + caption INSIDE
        # the same dashed box as the uploader (see .st-key-vdx-dropzone-
        # wrapper in theme.py, which strips the native dropzone's own
        # border/background so this wrapper reads as the one dashed box,
        # matching the reference mockup's single-composition dropzone
        # instead of a custom heading sitting above a separately-bordered
        # native widget). Streamlit's own instructional text is hidden by
        # that same CSS since it would otherwise repeat this heading.
        with st.container(key="vdx-dropzone-wrapper"):
            st.markdown(
                '<div style="text-align: center;">'
                '<span class="material-symbols-outlined" style="font-size: 2.25rem; color: var(--vdx-primary);">cloud_upload</span>'
                '<p style="font-family: var(--vdx-font-display); font-weight: 700; '
                'font-size: 1.05rem; margin: 0.5rem 0 0.15rem;">Drop a fundus photo here</p>'
                '<p class="vdx-caption" style="margin-bottom: 0.75rem;">PNG, JPG, or JPEG</p>'
                "</div>",
                unsafe_allow_html=True,
            )
            uploaded = container.file_uploader(
                "Upload a fundus photo", type=["png", "jpg", "jpeg"], key="file_uploader", label_visibility="collapsed"
            )
        if uploaded is not None:
            image = _decode_upload(uploaded)
    else:
        # Compact sidebar path: no custom heading, so the native label
        # stays as the only one.
        uploaded = container.file_uploader("Upload a fundus photo", type=["png", "jpg", "jpeg"], key="file_uploader")
        if uploaded is not None:
            image = _decode_upload(uploaded)

    return effective_patient_id, image


def render_intake_screen() -> tuple[str, np.ndarray | None, bool]:
    """The centered "Patient Intake & Signal Acquisition" glass panel --
    the Clinical Liquid Glass reference mockup's one concrete screen,
    reproduced directly (see this module's docstring). Shown only before
    the user has both picked an image source AND clicked "Initialize
    analysis" for the first time this session; after that, the same
    session-state-keyed controls move to a compact sidebar instead (see
    the call site below) so they stay reachable without this panel taking
    over the page every time settings change.

    Returns (effective_patient_id, image, initialize_clicked).
    """
    with st.container(key="vdx-intake-panel"):
        left_col, right_col = st.columns([5, 7], gap="large")

        with left_col:
            st.markdown(
                '<span class="vdx-intake-eyebrow">System entry portal</span>'
                '<h2 style="margin-top: 0.4rem;">Patient Intake &amp; Signal Acquisition</h2>'
                '<p class="vdx-intake-description">Upload a fundus photo or enter patient '
                "details to run the automated screening pipeline (quality check, disease "
                "detection, biomarkers, and a recommendation summary).</p>",
                unsafe_allow_html=True,
            )
            st.markdown('<div style="height: 0.5rem"></div>', unsafe_allow_html=True)
            with st.container(key="vdx-intake-toggle-row"):
                label_col, toggle_col = st.columns([4, 1], vertical_alignment="center")
                with label_col:
                    st.markdown(
                        '<div class="vdx-intake-toggle-label">Demo mode</div>'
                        '<div class="vdx-intake-toggle-sublabel">Use a local sample image</div>',
                        unsafe_allow_html=True,
                    )
                with toggle_col:
                    st.toggle(
                        "Demo mode",
                        value=False,
                        key="demo_mode",
                        label_visibility="collapsed",
                        help="Try the app on a locally available sample image instead of uploading your own.",
                    )

        with right_col:
            st.markdown('<span class="vdx-field-label">Patient ID / reference</span>', unsafe_allow_html=True)
            st.text_input(
                "Patient ID / reference",
                value="",
                placeholder="e.g. DEMO-001",
                key="patient_id",
                label_visibility="collapsed",
            )
            effective_patient_id, image = _resolve_image_source(container=st)

            st.markdown('<div style="height: 0.75rem"></div>', unsafe_allow_html=True)
            initialize_clicked = st.button(
                "Initialize analysis",
                type="primary",
                width="stretch",
                icon=":material/rocket_launch:",
                disabled=image is None,
                key="initialize_btn",
            )

        st.divider()
        # The reference mockup pairs this with a right-aligned "Lat: 40ms /
        # v4.2.1-stable" readout -- fabricated telemetry with no real
        # backing value here, so it's dropped rather than faked; this
        # status dot is left as the one genuinely meaningful signal (the
        # page loaded and its components registered).
        st.markdown(
            '<div class="vdx-header-status">'
            '<span class="vdx-status-dot"></span>'
            "<span>Core engine ready</span>"
            "</div>",
            unsafe_allow_html=True,
        )

    return effective_patient_id, image, initialize_clicked


_started = st.session_state.get("_vdx_started", False)

if not _started:
    effective_patient_id, image, initialize_clicked = render_intake_screen()
    if image is not None and initialize_clicked:
        st.session_state["_vdx_started"] = True
        st.rerun()
    st.stop()

st.header("Results")

# No persistent sidebar in this design (removed per a design-review pass --
# it was the one surface never re-themed to match everything else, and a
# permanent side panel doesn't earn its keep once the wide layout below
# doesn't need to share the viewport with it on every screen). Patient ID /
# demo mode / the image source -- the same session-state-keyed controls
# render_intake_screen() uses -- live in this collapsed-by-default expander
# instead, reachable without bringing back the full-page intake panel.
with st.expander("Change patient / image"):
    st.text_input("Patient ID / reference", value="", placeholder="e.g. DEMO-001", key="patient_id")
    st.toggle(
        "Demo mode",
        value=False,
        help="Try the app on a locally available sample image instead of uploading your own.",
        key="demo_mode",
    )
    effective_patient_id, image = _resolve_image_source(container=st)

if image is None:
    st.info('Upload a fundus photo or turn on demo mode under "Change patient / image" above to get started.')
    st.stop()

# Explainability method also used to live in the sidebar -- read here
# (session_state already holds its current value from the previous run,
# same pattern Streamlit widgets always rely on) since the pipeline needs
# it before the actual selectbox widget renders further down, next to the
# Image Comparison viewer its choice actually affects.
cam_method = st.session_state.get("cam_method", next(iter(CAM_METHODS)))

cache_key = (hashlib.md5(image.tobytes()).hexdigest(), effective_patient_id, cam_method)
is_new_computation = st.session_state.get("_vdx_cache_key") != cache_key

if is_new_computation:
    # The banner is created FIRST, immediately after the "Results" header
    # and before any section placeholder -- position: sticky keeps it
    # pinned to the viewport top once scrolled past, but only from
    # wherever it sits in DOM order onward. Creating it after the section
    # placeholders (tried first, caught live) put the grid above it, so
    # scrolling past the header still left the banner off-screen below
    # real content -- the exact bug this page exists to fix, just moved.
    # It has to be the first thing in the results flow.
    banner = ProgressBanner()

    placeholders = _create_stage_placeholders()

    def on_stage(stage_name, value):
        banner.advance(stage_name)
        args = value if isinstance(value, tuple) else (value,)
        with placeholders[stage_name].container(key=f"vdx-section-{stage_name}-content"):
            _RENDER_BY_STAGE[stage_name](*args)

    try:
        result = run_pipeline(
            image, patient_id=effective_patient_id, cam_method=cam_method, on_stage=on_stage
        )
    except Exception as exc:
        banner.finish()
        render_error_card(exc)
        st.stop()

    banner.finish()
    st.session_state["_vdx_result"] = result
    st.session_state["_vdx_cache_key"] = cache_key
else:
    result = st.session_state["_vdx_result"]
    _render_all_sections(result)

render_image_comparison(result)

st.divider()

# The Recommendation text is report/content.py's one synthesized, cross-
# field summary (severity phrasing, the CDR-vs-classifier-disagreement
# note when present, the disclaimer) -- not reproducible from the raw
# pipeline dicts without duplicating that logic here, so it's still
# sourced from build_report_content() exactly like the PDF is. Everything
# else that Section list carries (quality metrics, before/after images,
# per-detection text/tables/images, biomarker metrics/images) is NOT
# re-rendered below -- see this module's docstring for why: it was a full
# second copy of content already shown above.
content = build_report_content(result)
recommendation = next(s for s in content.sections if s.title == "Recommendation")
render_recommendation_card(recommendation.body)

pdf_bytes = generate_pdf(content)
st.download_button(
    "Download PDF report",
    data=pdf_bytes,
    file_name=f"visiondx_report_{_safe_filename(content.patient_id)}.pdf",
    mime="application/pdf",
)
st.caption(
    "The PDF includes everything above (quality, detections with "
    "probability breakdowns, biomarkers, recommendation) in a print-formatted "
    "layout, or use your browser's Print (Ctrl/Cmd+P) on this page."
)
# Reserves room at the very bottom of the page so the fixed disclaimer
# footer doesn't sit on top of this last caption when scrolled all the way
# down -- the footer floats over whatever's currently at the bottom of the
# viewport, this just keeps that from being real content.
st.markdown('<div class="vdx-footer-spacer"></div>', unsafe_allow_html=True)
