"""Phase 9: Streamlit dashboard.

Ties every pipeline stage together: upload (or demo mode) -> quality ->
preprocessing preview -> DR detection + Grad-CAM -> vessel biomarkers ->
optic disc/cup/CDR -> an in-app report preview -> PDF download.

Run with (from the repo root, matching this project's Windows venv
convention -- see README):

    .venv\\Scripts\\python.exe -m streamlit run src/app/main.py

Everything here only reads the documented dict/dataclass keys the pipeline
modules expose (see report/pipeline.py, report/content.py) -- never a
specific accuracy number or checkpoint detail, so a future retrained
checkpoint (see ROADMAP.md's Phase 6 note) needs no changes on this page.
"""

import hashlib
import re

import cv2
import numpy as np
import streamlit as st

from src.app.charts import probability_bar_chart
from src.app.demo_data import list_demo_images, load_demo_image
from src.app.render_preview import render_streamlit
from src.app.theme import inject_css
from src.explainability.gradcam import CAM_METHODS
from src.report import overlays
from src.report.content import build_report_content
from src.report.pdf import generate_pdf
from src.report.pipeline import run_pipeline

st.set_page_config(page_title="VisionDx", page_icon="\U0001f441", layout="wide")
inject_css()


def _to_rgb(array: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(array, cv2.COLOR_BGR2RGB)


def _decode_upload(uploaded_file) -> np.ndarray:
    file_bytes = np.frombuffer(uploaded_file.getvalue(), np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


def _safe_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_") or "report"


st.title("VisionDx")
st.caption(
    "AI-assisted retinal disease analysis pipeline — educational/portfolio "
    "demonstration, not a diagnostic device."
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

# --- Run the pipeline once per (image, patient id, CAM method) combination,
# cached in session_state -- unrelated widget interactions on a rerun
# (Streamlit reruns the whole script top-to-bottom on every interaction)
# shouldn't recompute a several-second analysis pipeline. The DR detection
# model itself is cached at a lower level already (pipeline._cached_detection_model),
# so this cache is about skipping the whole run, not just model loading. ---
cache_key = (hashlib.md5(image.tobytes()).hexdigest(), effective_patient_id, cam_method)
if st.session_state.get("_vdx_cache_key") != cache_key:
    with st.spinner("Running analysis pipeline…"):
        st.session_state["_vdx_result"] = run_pipeline(image, patient_id=effective_patient_id, cam_method=cam_method)
    st.session_state["_vdx_cache_key"] = cache_key

result = st.session_state["_vdx_result"]

st.header("Results")

# Quality
st.subheader("Image Quality")
quality = result["quality"]
cols = st.columns(3)
cols[0].metric("Quality score", f"{quality['score']:.0f}/100")
cols[1].metric("Focus", f"{quality['checks']['focus']['score']:.0f}/100")
cols[2].metric("Exposure", f"{quality['checks']['exposure']['score']:.0f}/100")
if quality["passed"]:
    st.success("Image quality passed both checks.")
else:
    st.warning("Image quality did not pass — findings below may be less reliable.")

# Preprocessing preview
st.subheader("Preprocessing")
before_col, after_col = st.columns(2)
before_col.image(_to_rgb(result["preprocessing_preview"]["before"]), caption="Original", width="stretch")
after_col.image(
    _to_rgb(result["preprocessing_preview"]["after"]),
    caption="Illumination + CLAHE + color normalization",
    width="stretch",
)

# Detection + explainability
st.subheader("Diabetic Retinopathy Detection")
detection = result["detection"]
if detection is None:
    st.info("Detection model not available in this build — no trained checkpoint was found at the expected path.")
else:
    st.metric("Top estimate", detection["label"], f"{detection['probability'] * 100:.1f}% confidence")
    st.plotly_chart(probability_bar_chart(detection), width="stretch", config={"displayModeBar": False})
    if result["cam_overlay"] is not None:
        st.image(_to_rgb(result["cam_overlay"]), caption="Grad-CAM attention map", width="stretch")

# Vessel biomarkers
st.subheader("Vessel Biomarkers")
vessel_result = result["vessels"]
cols = st.columns(4)
cols[0].metric("Vessel density", f"{vessel_result['vessel_density']:.2f}%")
cols[1].metric("Branch points", vessel_result["branch_count"])
cols[2].metric("Tortuosity", f"{vessel_result['tortuosity']:.3f}")
cols[3].metric("Avg. width", f"{vessel_result['average_width']:.2f}px")
st.image(
    _to_rgb(overlays.vessel_mask_overlay(result["working_image"], vessel_result)),
    caption="Vessel mask overlay",
    width="stretch",
)

# Optic disc / cup / macula
st.subheader("Optic Disc / Cup / Macula")
optic_disc_result = result["optic_disc"]
cols = st.columns(3)
cols[0].metric("Vertical CDR", f"{optic_disc_result['vertical_cdr']:.3f}")
cols[1].metric("Disc diameter", f"{optic_disc_result['disc_diameter_px']}px")
cols[2].metric("Cup diameter", f"{optic_disc_result['cup_diameter_px']}px")
if not optic_disc_result["disc_found"] or optic_disc_result["disc_diameter_px"] == 0:
    # disc_found only reflects Stage 6.1's classical localization succeeding
    # -- Stage 6.2's segmentation can still independently come back empty
    # (a real, observed failure mode on out-of-domain input with the
    # current provisional checkpoint, see ROADMAP.md's Phase 6 note), which
    # disc_found alone wouldn't catch.
    st.warning("Optic disc could not be confidently segmented in this image — cup/disc measurements above are not meaningful.")
st.image(
    _to_rgb(overlays.optic_disc_overlay(result["working_image"], optic_disc_result)),
    caption="Disc (yellow) / cup (red) / macula (green)",
    width="stretch",
)

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
