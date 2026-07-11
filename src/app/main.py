"""Phase 9 / redesign: Streamlit dashboard.

Ties every pipeline stage together: upload (or demo mode) -> quality ->
preprocessing preview -> DR/glaucoma/AMD detection + Grad-CAM each ->
vessel biomarkers -> optic disc/cup/CDR -> an in-app report preview -> PDF
download.

Redesign pass: the page is now a dense multi-column dashboard instead of
seven full-width stacked sections. Three ROW groups (Overview / Disease
Screening / Biomarkers) each hold their stages side by side in columns.
The Disease Screening row in particular replaces what used to be three
near-identical full sections (subheader + pill + ring + datagrid +
full-size Grad-CAM image, once each for DR/glaucoma/AMD) with three
compact tiles -- the Grad-CAM images moved to the existing Image Comparison
viewer at the bottom instead of repeating three times inline.

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

from src.app.charts import probability_bar_chart
from src.app.components import render_datagrid, render_ring, render_stat_tile
from src.app.demo_data import list_demo_images, load_demo_image
from src.app.progress import ProgressBanner, render_error_card, render_skeleton
from src.app.render_preview import render_streamlit
from src.app.theme import inject_ambient_cursor, inject_css
from src.explainability.gradcam import CAM_METHODS
from src.report import overlays
from src.report.content import build_report_content
from src.report.pdf import generate_pdf
from src.report.pipeline import run_pipeline

st.set_page_config(page_title="VisionDx", page_icon="\U0001f441", layout="wide")
inject_css()
inject_ambient_cursor()

# CSS custom-property references, not hex literals -- theme.py's :root is
# the single source of truth for these two accent colors; every caller here
# just points at it rather than duplicating hex values.
_TEAL = "var(--vdx-teal)"
_COPPER = "var(--vdx-copper)"


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
    color = _TEAL if quality["passed"] else _COPPER
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
    # Only the "after" image at this density -- the original is still one
    # tap away in the Image Comparison viewer below, so showing both here
    # too would just repeat it.
    _tile_label("Preprocessing")
    st.image(
        _to_rgb(preview["after"]),
        caption="Illumination + CLAHE + color normalization",
        width="stretch",
    )


def render_detection_section(detection: dict | None, cam_overlay) -> None:
    if detection is None:
        _unavailable_tile("Diabetic Retinopathy")
        return
    variant = "normal" if detection["class_idx"] == 0 else "attention"
    color = _TEAL if variant == "normal" else _COPPER
    render_stat_tile(
        "Diabetic Retinopathy",
        detection["label"],
        variant,
        f"{detection['probability'] * 100:.0f}%",
        detection["probability"] * 100,
        ring_color=color,
    )
    # DR is 5-class (not binary like glaucoma/AMD) -- genuinely has more to
    # show, so it alone gets an optional expander for the full distribution
    # rather than forcing every disease tile into an identical shape.
    with st.expander("Severity breakdown"):
        st.plotly_chart(probability_bar_chart(detection), width="stretch", config={"displayModeBar": False})


def render_glaucoma_section(glaucoma: dict | None, cam_overlay) -> None:
    if glaucoma is None:
        _unavailable_tile("Glaucoma")
        return
    variant = "normal" if glaucoma["class_idx"] == 0 else "attention"
    color = _TEAL if variant == "normal" else _COPPER
    render_stat_tile(
        "Glaucoma",
        glaucoma["label"],
        variant,
        f"{glaucoma['probability'] * 100:.0f}%",
        glaucoma["probability"] * 100,
        ring_color=color,
    )


def render_amd_section(amd: dict | None, cam_overlay) -> None:
    if amd is None:
        _unavailable_tile("AMD")
        return
    variant = "normal" if amd["class_idx"] == 0 else "attention"
    color = _TEAL if variant == "normal" else _COPPER
    render_stat_tile(
        "AMD",
        amd["label"],
        variant,
        f"{amd['probability'] * 100:.0f}%",
        amd["probability"] * 100,
        ring_color=color,
    )


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
        render_ring("Density", f"{vessel_result['vessel_density']:.1f}%", vessel_result["vessel_density"], color=_TEAL)
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
    ring_color = _COPPER if cdr >= 0.5 else _TEAL
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


def render_image_comparison(result: dict) -> None:
    """A single unified image viewer switching between every view the
    pipeline produced, via st.pills (real Streamlit widget state/rerun
    handling -- not inert custom HTML buttons, which can't communicate
    back to Python without a full custom component). Reuses arrays
    already computed elsewhere in `result`/overlays.* rather than
    recomputing anything. Now also the ONLY place Grad-CAM overlays are
    shown at full size -- the disease tiles above no longer repeat them.

    Deliberately called only once the FULL pipeline result is available
    (see main flow below) rather than from inside on_stage -- some of
    these images (Grad-CAM, the two overlays) don't exist yet mid-pipeline,
    and threading "which images are ready so far" through a progressively
    updating pills widget isn't worth the complexity for a comparison view
    that's naturally a "look at everything together" step anyway, same
    reasoning as why the Report Preview section only appears once
    `result` is final.
    """
    st.subheader("Image Comparison")
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
    selected = st.pills("Compare views", options, selection_mode="single", default=options[0], key="image_compare")
    if selected is None:
        selected = options[0]
    st.image(_to_rgb(images[selected]), width="stretch")


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


st.title("VisionDx")
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

# --- Sidebar: every input control lives here, so the main column stays
# results-only. Also what the @media print rule in theme.py hides. ---
st.sidebar.header("Input")
patient_id_input = st.sidebar.text_input(
    "Patient ID / reference", value="", placeholder="e.g. DEMO-001", key="patient_id"
)
demo_mode = st.sidebar.toggle(
    "Demo mode",
    value=False,
    help="Try the app on a locally available sample image instead of uploading your own.",
    key="demo_mode",
)
cam_method = st.sidebar.selectbox("Explainability method", options=list(CAM_METHODS), index=0, key="cam_method")

image = None
effective_patient_id = patient_id_input

if demo_mode:
    demo_images = list_demo_images()
    if not demo_images:
        st.sidebar.info("No local demo images found — download APTOS 2019 per the README to use demo mode.")
    else:
        options = {f"{item['label']} — {item['id_code']}": item for item in demo_images}
        choice = st.sidebar.selectbox("Sample image", list(options), key="demo_sample")
        selected = options[choice]
        image = load_demo_image(selected["path"])
        if not effective_patient_id:
            effective_patient_id = f"DEMO-{selected['id_code']}"
else:
    uploaded = st.sidebar.file_uploader(
        "Upload a fundus photo", type=["png", "jpg", "jpeg"], key="file_uploader"
    )
    if uploaded is not None:
        image = _decode_upload(uploaded)

if image is None:
    st.info("Upload a fundus photo or turn on demo mode in the sidebar to get started.")
    st.stop()

st.header("Results")

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

# --- The "generation preview before export": a WYSIWYG mirror of the PDF,
# built from and rendering the exact same ReportContent report/pdf.py
# renders, not a separate description of it. ---
content = build_report_content(result)
render_streamlit(content)

pdf_bytes = generate_pdf(content)
st.download_button(
    "Download PDF report",
    data=pdf_bytes,
    file_name=f"visiondx_report_{_safe_filename(content.patient_id)}.pdf",
    mime="application/pdf",
)
st.caption(
    "To print: download the PDF above and print it (it's A4-formatted for "
    "clean printing), or use your browser's Print (Ctrl/Cmd+P) on this page."
)
# Reserves room at the very bottom of the page so the fixed disclaimer
# footer doesn't sit on top of this last caption when scrolled all the way
# down -- the footer floats over whatever's currently at the bottom of the
# viewport, this just keeps that from being real content.
st.markdown('<div class="vdx-footer-spacer"></div>', unsafe_allow_html=True)
