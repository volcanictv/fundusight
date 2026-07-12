"""Streamlit dashboard: ties every pipeline stage together (upload or demo
mode -> quality -> preprocessing preview -> DR/glaucoma/AMD detection with
Grad-CAM -> vessel biomarkers -> optic disc/cup/CDR -> recommendation
summary -> PDF download).

The results area is a dense multi-column grid (Overview / Disease
Screening / Biomarkers rows, see _ROWS below), not stacked full-width
sections. Each stage gets its own st.empty() placeholder that fills in
place as report/pipeline.py's run_pipeline() finishes it (see on_stage
below and app/progress.py's ProgressBanner), so the page doesn't sit
behind one opaque spinner.

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
import os
import re
import sys
import time

# --- TEMPORARY DIAGNOSTIC INSTRUMENTATION -----------------------------
# Bisecting a segfault in the deployed container (run-streamlit.sh's
# "Segmentation fault", no Python traceback -- SIGSEGV kills the process
# before Python-level error handling ever runs, so the only visibility
# into WHERE it happens is whatever got printed and flushed beforehand).
# Remove this whole block (and the _diag() calls below) once the crash is
# isolated to a specific import/step -- this is not meant to stay.
_diag_t0 = time.time()


def _diag(msg: str) -> None:
    print(f"[DIAG +{time.time() - _diag_t0:.2f}s] {msg}", flush=True)


_diag("main.py execution started")
# --- end temporary instrumentation (continued below, inline) ----------

# Streamlit Community Cloud runs the `streamlit` executable directly rather
# than `python -m streamlit`, so unlike the local dev invocation (see the
# docstring above), the repo root never lands on sys.path automatically --
# only src/app (this file's own directory) does, via Streamlit's own
# bootstrap. Every `from src...` import below needs the repo root on
# sys.path, so add it explicitly before they run.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Set before any numpy/opencv/torch import below -- native libraries read
# these once, at load time, so setting them later has no effect. Both
# mitigate a well-documented native-crash class when OpenCV and PyTorch run
# in the same process (see explainability/gradcam.py's generate_cam(),
# which calls into both within one function: a torch forward/backward pass
# immediately followed by cv2 resize/color-convert calls for the overlay).
# Each library can bundle/initialize its own OpenMP runtime and its own
# internal thread pool, and two live at once in one process is a known
# segfault source -- not something guardable from Python once it happens,
# only preventable by not letting the conflict arise in the first place.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

_diag("importing numpy")
import numpy as np

_diag(f"numpy {np.__version__} imported -- dumping np.show_runtime() (SIMD/CPU dispatch info)")
try:
    np.show_runtime()
except Exception as exc:  # pragma: no cover -- diagnostic only
    _diag(f"np.show_runtime() raised (non-fatal): {exc!r}")
_diag("np.show_runtime() returned without crashing")

_diag("importing cv2")
import cv2

_diag(f"cv2 {cv2.__version__} imported")

_diag("importing streamlit")
import streamlit as st

_diag("streamlit imported")

# Disables OpenCV's own internal thread pool, for the same reason as the
# env vars above. A global, process-wide setting -- runs once here even
# though cv2 is imported in many other modules throughout this pipeline.
cv2.setNumThreads(0)

_diag("importing torch")
import torch

_diag(f"torch {torch.__version__} imported")

_diag("importing torchvision")
import torchvision

_diag(f"torchvision {torchvision.__version__} imported")

_diag("importing monai")
import monai

_diag(f"monai {monai.__version__} imported")

_diag("importing src.app.charts")
from src.app.charts import binary_probability_chart, probability_bar_chart

_diag("importing src.app.checkpoints")
from src.app.checkpoints import fetch_checkpoints

_diag("importing src.app.components")
from src.app.components import render_datagrid, render_recommendation_card, render_ring, render_stat_tile

_diag("importing src.app.demo_data")
from src.app.demo_data import list_demo_images, load_demo_image

_diag("importing src.app.progress")
from src.app.progress import ProgressBanner, render_error_card, render_skeleton

_diag("importing src.app.theme")
from src.app.theme import inject_ambient_cursor, inject_css, inject_image_zoom, render_header

_diag("importing src.detection.amd_infer")
from src.detection.amd_infer import AMD_LABELS

_diag("importing src.detection.glaucoma_infer")
from src.detection.glaucoma_infer import GLAUCOMA_LABELS

_diag("importing src.explainability.gradcam")
from src.explainability.gradcam import CAM_METHODS

_diag("importing src.report.overlays")
from src.report import overlays

_diag("importing src.report.content")
from src.report.content import build_report_content

_diag("importing src.report.pdf")
from src.report.pdf import generate_pdf

_diag("importing src.report.pipeline")
from src.report.pipeline import run_pipeline

_diag("ALL IMPORTS COMPLETE")


@st.cache_resource(show_spinner="Fetching trained model checkpoints...")
def _ensure_checkpoints() -> list[str]:
    """Runs once per process (st.cache_resource) — a fresh deployment with no
    local checkpoints/ fetches them from the GitHub Release on first load; a
    dev machine that already has them locally makes no network calls at all.
    """
    return fetch_checkpoints()


# set_page_config must be the first Streamlit command in the script, so the
# (cached, spinner-showing) checkpoint fetch has to come after it.
st.set_page_config(page_title="Fundusight", page_icon="\U0001f7e3", layout="wide")
_diag("set_page_config done, fetching checkpoints")
_ensure_checkpoints()
_diag("checkpoints ensured, injecting CSS/JS")
inject_css()
inject_ambient_cursor()
inject_image_zoom()
render_header()
_diag("header rendered -- reached the intake/results branch")

# CSS custom-property references, not hex literals -- theme.py's :root is
# the single source of truth for these two accent colors; every caller here
# just points at it rather than duplicating hex values.
_PRIMARY = "var(--fdx-primary)"
_TERTIARY = "var(--fdx-tertiary)"


def _to_rgb(array: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(array, cv2.COLOR_BGR2RGB)


def _decode_upload(uploaded_file) -> np.ndarray:
    file_bytes = np.frombuffer(uploaded_file.getvalue(), np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


def _safe_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_") or "report"


def _tile_label(text: str) -> None:
    # Matches render_stat_tile()'s title style (theme.py's
    # .fdx-stat-tile-title) so every column in the dense grid below reads
    # as one consistent family, not a mix of st.subheader and ad-hoc labels.
    st.markdown(f'<div class="fdx-stat-tile-title">{html.escape(text)}</div>', unsafe_allow_html=True)


def _unavailable_tile(title: str) -> None:
    st.markdown(
        f'<div class="fdx-stat-tile"><div class="fdx-stat-tile-title">{html.escape(title)}</div>'
        f'<div class="fdx-caption">Model not available in this build — no trained checkpoint was found.</div></div>',
        unsafe_allow_html=True,
    )


# Per-section renderers, shared by both the progressive (cache-miss) and
# direct (cache-hit) render paths below.


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
    # Side-by-side before/after in the same glass card as its Overview-row
    # neighbor -- Image Comparison below still covers each image
    # individually at full size for closer inspection.
    with st.container(key="fdx-preprocessing-card"):
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
    # Always visible (not behind an expander) since the probability
    # distribution is the most informative part of a disease tile. DR is
    # 5-class, unlike glaucoma/AMD's binary charts, so it gets its own
    # 5-row chart.
    with st.container(key="fdx-chart-dr"):
        st.plotly_chart(
            probability_bar_chart(detection),
            width="stretch",
            config={"displayModeBar": False, "staticPlot": True},
        )
        if cam_overlay is not None:
            # Grad-CAM heatmap itself lives in Image Comparison further
            # down the page (see render_image_comparison()), not repeated here.
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
    # Binary classifiers get a 2-row chart instead of DR's 5-row ordinal
    # one, always visible for the same reason as render_detection_section().
    with st.container(key="fdx-chart-glaucoma"):
        st.plotly_chart(
            binary_probability_chart(glaucoma, GLAUCOMA_LABELS),
            width="stretch",
            config={"displayModeBar": False, "staticPlot": True},
        )
        if cam_overlay is not None:
            st.caption('Grad-CAM overlay: see "Grad-CAM (Glaucoma)" in Image Comparison below.')


_AMD_TITLE = "Age-Related Macular Degeneration (AMD)"


def render_amd_section(amd: dict | None, cam_overlay) -> None:
    if amd is None:
        _unavailable_tile(_AMD_TITLE)
        return
    variant = "normal" if amd["class_idx"] == 0 else "attention"
    color = _PRIMARY if variant == "normal" else _TERTIARY
    render_stat_tile(
        _AMD_TITLE,
        amd["label"],
        variant,
        f"{amd['probability'] * 100:.0f}%",
        amd["probability"] * 100,
        ring_color=color,
    )
    # staticPlot=True on all three Disease Screening charts (here and in
    # render_detection_section/render_glaucoma_section above): every value
    # already has an on-bar label (see charts.py), so Plotly's default
    # interactive zoom/hover would only add affordance, no information.
    with st.container(key="fdx-chart-amd"):
        st.plotly_chart(
            binary_probability_chart(amd, AMD_LABELS),
            width="stretch",
            config={"displayModeBar": False, "staticPlot": True},
        )
        if cam_overlay is not None:
            st.caption('Grad-CAM overlay: see "Grad-CAM (AMD)" in Image Comparison below.')


def render_vessel_section(vessel_result: dict, working_image: np.ndarray) -> None:
    # Vessel mask overlay lives in Image Comparison below, not repeated here.
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
    # Disc/cup/macula overlay lives in Image Comparison below, not repeated here.
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
        # disc_found only reflects classical localization succeeding --
        # segmentation can independently come back empty on out-of-domain
        # input (see ROADMAP.md's Phase 6 note), which disc_found alone
        # wouldn't catch.
        st.warning("Optic disc could not be confidently segmented in this image — cup/disc measurements above are not meaningful.")


_MAX_COMPARISON_IMAGES_PER_ROW = 3


def render_image_comparison(result: dict) -> None:
    """Multi-select image viewer: pick two or more views via st.pills
    (multi-selection mode) and see them side by side. Reuses arrays already
    computed elsewhere in `result`/overlays.* rather than recomputing
    anything. This is the only place Grad-CAM overlays are shown at full
    size -- the disease tiles above don't repeat them.

    Also hosts the "Explainability method" selectbox, since this is the
    one place its choice has a visible effect (which Grad-CAM overlays
    exist to pick from below).

    Called only once the full pipeline result is available (not from
    on_stage), since some of these images (Grad-CAM, the two overlays)
    don't exist yet mid-pipeline.
    """
    st.subheader("Image Comparison")

    original = result["preprocessing_preview"]["before"]
    images = {
        "Original": original,
        "Preprocessed": result["preprocessing_preview"]["after"],
    }
    # Grad-CAM overlays come back at the model's fixed square input
    # resolution (see explainability/gradcam.py's IMAGE_SIZE resize), not
    # the photo's own aspect ratio, unlike Original/Preprocessed/the vessel
    # and optic-disc overlays -- stretch back to the original (height,
    # width) so rows in this grid line up. Display-only; the model/CAM
    # generation itself is untouched.
    _target_hw = (original.shape[1], original.shape[0])  # cv2.resize wants (width, height)

    def _match_original_aspect(overlay: np.ndarray) -> np.ndarray:
        if overlay.shape[:2] == original.shape[:2]:
            return overlay
        return cv2.resize(overlay, _target_hw, interpolation=cv2.INTER_LINEAR)

    if result["cam_overlay"] is not None:
        images["Grad-CAM (DR)"] = _match_original_aspect(result["cam_overlay"])
    if result["glaucoma_cam_overlay"] is not None:
        images["Grad-CAM (Glaucoma)"] = _match_original_aspect(result["glaucoma_cam_overlay"])
    if result["amd_cam_overlay"] is not None:
        images["Grad-CAM (AMD)"] = _match_original_aspect(result["amd_cam_overlay"])
    images["Vessel mask"] = overlays.vessel_mask_overlay(result["working_image"], result["vessels"])
    images["Optic disc"] = overlays.optic_disc_overlay(result["working_image"], result["optic_disc"])

    options = list(images)
    # A sensible default -- one raw view, one explainability heatmap, one
    # biomarker overlay -- rather than dict-insertion order's first two
    # entries (always "Original"/"Preprocessed", leaving the viewer never
    # showing anything the pipeline actually detected until the user
    # picked something themselves). Exactly one name per category, first
    # available in each -- not a single flat priority list, which would
    # pick two Grad-CAMs over a biomarker overlay whenever more than one
    # detector's checkpoint is present.
    default_selection = []
    if "Original" in images:
        default_selection.append("Original")
    for name in ("Grad-CAM (DR)", "Grad-CAM (Glaucoma)", "Grad-CAM (AMD)"):
        if name in images:
            default_selection.append(name)
            break
    for name in ("Optic disc", "Vessel mask"):
        if name in images:
            default_selection.append(name)
            break
    if len(default_selection) < 3:
        default_selection = options[:3]
    # Explainability method sits directly beside the pills it controls,
    # rather than up near the "Image Comparison" heading, so the
    # relationship reads as clearly connected.
    caption_col, method_col = st.columns([3, 1], vertical_alignment="center")
    with caption_col:
        st.caption("Select two or more views to compare them side by side. Hover an image to magnify.")
    with method_col:
        st.selectbox("Explainability method", options=list(CAM_METHODS), index=0, key="cam_method")
    selected = st.pills(
        "Compare views", options, selection_mode="multi", default=default_selection, key="image_compare"
    )
    if not selected:
        selected = default_selection

    for start in range(0, len(selected), _MAX_COMPARISON_IMAGES_PER_ROW):
        chunk = selected[start : start + _MAX_COMPARISON_IMAGES_PER_ROW]
        # Always allocate a full _MAX_COMPARISON_IMAGES_PER_ROW columns,
        # even for a short last row -- st.columns divides available width
        # by however many columns you ask for, so a shorter row would
        # render its images wider than a full row's.
        columns = st.columns(_MAX_COMPARISON_IMAGES_PER_ROW)
        for column, name in zip(columns, chunk):
            with column:
                st.image(_to_rgb(images[name]), caption=name, width="stretch")


# stage_name -> (render function, extractor from the finished result dict).
# Single source of truth for both render paths below and for on_stage's
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
# many columns that row gets.
_ROWS = [
    ("Overview", ["quality", "preprocessing"]),
    ("Disease Screening", ["detection", "glaucoma", "amd"]),
    ("Biomarkers", ["vessels", "optic_disc"]),
]


def _create_stage_placeholders() -> dict:
    """One st.empty() per stage, arranged into the _ROWS grid, filled
    immediately with a skeleton so the whole results area shows its
    loading shape at once.
    """
    placeholders = {}
    for title, stages in _ROWS:
        st.subheader(title)
        cols = st.columns(len(stages))
        for col, stage in zip(cols, stages):
            with col:
                placeholders[stage] = st.empty()
                with placeholders[stage].container(key=f"fdx-section-{stage}-skeleton"):
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
                with st.container(key=f"fdx-section-{stage}-content"):
                    _RENDER_BY_STAGE[stage](*_EXTRACT_BY_STAGE[stage](result))


# Fixed + centered footer, not an inline caption, so it doesn't interrupt
# the page's flow (see theme.py's .fdx-progress-banner comment for why
# position: fixed, not sticky, is needed here).
st.markdown(
    '<div class="fdx-disclaimer-footer">AI-assisted retinal disease analysis pipeline '
    "— educational/portfolio demonstration, not a diagnostic device.</div>",
    unsafe_allow_html=True,
)


def _resolve_image_source() -> tuple[str, np.ndarray | None]:
    """Demo-sample picker or file uploader, whichever `demo_mode` (already
    rendered by the caller) selects. Shared by render_intake_screen() and
    the post-intake "Change patient / image" expander -- same two widgets
    either way.
    """
    patient_id_input = st.session_state.get("patient_id", "")
    effective_patient_id = patient_id_input
    image = None

    if st.session_state.get("demo_mode"):
        demo_images = list_demo_images()
        if not demo_images:
            st.info("No demo images bundled with this build — upload your own fundus photo instead.")
        else:
            options = {f"{item['label']} — {item['id_code']}": item for item in demo_images}
            choice = st.selectbox("Sample image", list(options), key="demo_sample")
            selected = options[choice]
            image = load_demo_image(selected["path"])
            if not effective_patient_id:
                effective_patient_id = f"DEMO-{selected['id_code']}"
    else:
        # The intake panel composes its own icon/heading/caption inside the
        # dropzone's dashed box (see .st-key-fdx-dropzone-wrapper in
        # theme.py, which strips the native dropzone's own border/
        # background) instead of a separately-bordered native widget.
        # Streamlit's own instructional text is hidden by that same CSS
        # since it would otherwise repeat this heading.
        with st.container(key="fdx-dropzone-wrapper"):
            st.markdown(
                '<div style="text-align: center;">'
                '<span class="material-symbols-outlined" style="font-size: 2.25rem; color: var(--fdx-primary);">cloud_upload</span>'
                '<p style="font-family: var(--fdx-font-display); font-weight: 700; '
                'font-size: 1.05rem; margin: 0.5rem 0 0.15rem;">Drop a fundus photo here</p>'
                '<p class="fdx-caption" style="margin-bottom: 0.75rem;">PNG, JPG, or JPEG</p>'
                "</div>",
                unsafe_allow_html=True,
            )
            uploaded = st.file_uploader(
                "Upload a fundus photo", type=["png", "jpg", "jpeg"], key="file_uploader", label_visibility="collapsed"
            )
        if uploaded is not None:
            image = _decode_upload(uploaded)

    return effective_patient_id, image


def render_intake_screen() -> tuple[str, np.ndarray | None, bool]:
    """The centered "Patient Intake & Signal Acquisition" glass panel,
    shown only before the user has both picked an image source AND clicked
    "Initialize analysis" for the first time this session; after that, the
    same session-state-keyed controls move into a collapsed "Change
    patient / image" expander instead (see the call site below) so they
    stay reachable without this panel taking over the page every time
    settings change.

    Returns (effective_patient_id, image, initialize_clicked).
    """
    with st.container(key="fdx-intake-panel"):
        left_col, right_col = st.columns([5, 7], gap="large")

        with left_col:
            st.markdown(
                '<span class="fdx-intake-eyebrow">System entry portal</span>'
                '<h2 style="margin-top: 0.4rem;">Patient Intake &amp; Signal Acquisition</h2>'
                '<p class="fdx-intake-description">Upload a fundus photo or enter patient '
                "details to run the automated screening pipeline (quality check, disease "
                "detection, biomarkers, and a recommendation summary).</p>",
                unsafe_allow_html=True,
            )
            st.markdown('<div style="height: 0.5rem"></div>', unsafe_allow_html=True)
            with st.container(key="fdx-intake-toggle-row"):
                label_col, toggle_col = st.columns([4, 1], vertical_alignment="center")
                with label_col:
                    st.markdown(
                        '<div class="fdx-intake-toggle-label">Demo mode</div>'
                        '<div class="fdx-intake-toggle-sublabel">Use a local sample image</div>',
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
            st.markdown('<span class="fdx-field-label">Patient ID / reference</span>', unsafe_allow_html=True)
            st.text_input(
                "Patient ID / reference",
                value="",
                placeholder="e.g. DEMO-001",
                key="patient_id",
                label_visibility="collapsed",
            )
            effective_patient_id, image = _resolve_image_source()

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
        # No fabricated telemetry (e.g. fake latency/version numbers) --
        # this status dot is the one genuinely meaningful signal here (the
        # page loaded and its components registered).
        st.markdown(
            '<div class="fdx-header-status">'
            '<span class="fdx-status-dot"></span>'
            "<span>Core engine ready</span>"
            "</div>",
            unsafe_allow_html=True,
        )

    return effective_patient_id, image, initialize_clicked


_started = st.session_state.get("_fdx_started", False)

if not _started:
    effective_patient_id, image, initialize_clicked = render_intake_screen()
    if image is not None and initialize_clicked:
        st.session_state["_fdx_started"] = True
        st.rerun()
    st.stop()

st.header("Results")
_diag("'Results' header rendered, entering 'Change patient / image' expander")

# Same session-state-keyed controls render_intake_screen() uses, collapsed
# by default so they're reachable without bringing back the full-page
# intake panel.
with st.expander("Change patient / image", icon=":material/edit:"):
    st.text_input("Patient ID / reference", value="", placeholder="e.g. DEMO-001", key="patient_id")
    st.toggle(
        "Demo mode",
        value=False,
        help="Try the app on a locally available sample image instead of uploading your own.",
        key="demo_mode",
    )
    _diag("about to call _resolve_image_source() (2nd call site)")
    effective_patient_id, image = _resolve_image_source()
    _diag(f"_resolve_image_source() returned, image is None: {image is None}")

if image is None:
    st.info('Upload a fundus photo or turn on demo mode under "Change patient / image" above to get started.')
    st.stop()

# Read here (session_state already holds its current value from the
# previous run) since the pipeline needs it before the actual selectbox
# further down, next to the Image Comparison viewer its choice affects,
# renders.
cam_method = st.session_state.get("cam_method", next(iter(CAM_METHODS)))
_diag(f"cam_method={cam_method!r}, image.shape={image.shape}, image.dtype={image.dtype} -- about to hash")

cache_key = (hashlib.md5(image.tobytes()).hexdigest(), effective_patient_id, cam_method)
_diag(f"cache_key computed: {cache_key}")
is_new_computation = st.session_state.get("_fdx_cache_key") != cache_key
_diag(f"is_new_computation={is_new_computation}")

if is_new_computation:
    # Created first, before any section placeholder: position: fixed keeps
    # it pinned to the viewport regardless of scroll, but creating it after
    # the placeholders instead put the grid above it in DOM order, leaving
    # the banner off-screen below real content once scrolled past the header.
    _diag("is_new_computation True -- constructing ProgressBanner()")
    banner = ProgressBanner()
    _diag("ProgressBanner() constructed -- calling _create_stage_placeholders()")

    placeholders = _create_stage_placeholders()
    _diag("_create_stage_placeholders() returned")

    def on_stage(stage_name, value):
        _diag(f"pipeline stage completed: {stage_name}")
        banner.advance(stage_name)
        args = value if isinstance(value, tuple) else (value,)
        with placeholders[stage_name].container(key=f"fdx-section-{stage_name}-content"):
            _RENDER_BY_STAGE[stage_name](*args)

    _diag("calling run_pipeline() -- this is the 'analysis starts' trigger")
    try:
        result = run_pipeline(
            image, patient_id=effective_patient_id, cam_method=cam_method, on_stage=on_stage
        )
    except Exception as exc:
        banner.finish()
        render_error_card(exc)
        st.stop()

    banner.finish()
    st.session_state["_fdx_result"] = result
    st.session_state["_fdx_cache_key"] = cache_key
else:
    result = st.session_state["_fdx_result"]
    _render_all_sections(result)

render_image_comparison(result)

st.divider()

# The Recommendation text is report/content.py's one synthesized,
# cross-field summary (severity phrasing, the CDR-vs-classifier-
# disagreement note when present, the disclaimer) -- sourced from
# build_report_content() exactly like the PDF is, so the two can't drift
# apart. Nothing else from its Section list is re-rendered below since
# it's all already shown above (quality metrics, before/after images,
# per-detection text/tables/images, biomarker metrics/images).
content = build_report_content(result)
recommendation = next(s for s in content.sections if s.title == "Recommendation")
render_recommendation_card(recommendation.body)

pdf_bytes = generate_pdf(content)
st.download_button(
    "Download PDF report",
    data=pdf_bytes,
    file_name=f"fundusight_report_{_safe_filename(content.patient_id)}.pdf",
    mime="application/pdf",
)
st.caption(
    "The PDF includes everything above (quality, detections with "
    "probability breakdowns, biomarkers, recommendation) in a print-formatted "
    "layout, or use your browser's Print (Ctrl/Cmd+P) on this page."
)
# Reserves space so the fixed disclaimer footer doesn't overlap the last
# real content when scrolled all the way down.
st.markdown('<div class="fdx-footer-spacer"></div>', unsafe_allow_html=True)
