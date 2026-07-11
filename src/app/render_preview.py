"""Phase 9: renders a report/content.py ReportContent as the in-app
"generation preview before export" -- what the user sees in the dashboard
before clicking Download PDF. Walks the exact same Section list
report/pdf.py renders, so the preview and the exported PDF can never drift
apart in *content*, only in how each medium presents it (Streamlit widgets
here vs. ReportLab flowables there).
"""

import html

import cv2
import streamlit as st

from src.report.content import ReportContent


def _to_rgb(array):
    # st.image expects RGB; every array flowing through this pipeline is
    # BGR (cv2.imread's convention, matching quality.py/enhance.py/etc.).
    return cv2.cvtColor(array, cv2.COLOR_BGR2RGB)


_MAX_METRICS_PER_ROW = 3

# A lone image stretched to the full ~1140px block-container width (the old
# behavior) mostly just adds scroll height, not information -- these overlays
# are already visible at full size in the Image Comparison viewer above this
# section. Capping width keeps the Report Preview dense without cropping or
# shrinking multi-image rows, which already split the width sensibly.
_SINGLE_IMAGE_MAX_WIDTH = 480

# Deliberately smaller than _SINGLE_IMAGE_MAX_WIDTH: this image sits beside
# a metrics column that's usually only 3-4 short rows tall, so stretching
# the image to fill the full column width (the first version of this fix)
# just made a differently-shaped dead gap -- the image became the tall
# element and the metrics column stopped well short of it. A smaller,
# roughly metrics-column-height image keeps the row's overall height close
# to what the metrics actually need.
_METRIC_GRID_IMAGE_MAX_WIDTH = 300


def _render_metric_grid(body: dict) -> None:
    rows = body["rows"]
    image = body.get("image")
    if image is not None:
        # Metrics beside the image instead of metrics-then-image stacked --
        # cuts this section's vertical footprint (Vessel Biomarkers, Optic
        # Disc/Cup/Macula) since a stacked layout left the metrics column
        # mostly empty above a full-width image.
        metrics_col, image_col = st.columns([1, 1])
        with metrics_col:
            for start in range(0, len(rows), _MAX_METRICS_PER_ROW):
                chunk = rows[start : start + _MAX_METRICS_PER_ROW]
                columns = st.columns(min(_MAX_METRICS_PER_ROW, len(chunk)))
                for column, (label, value) in zip(columns, chunk):
                    column.metric(label, value)
        with image_col:
            st.image(_to_rgb(image["array"]), caption=image["caption"], width=_METRIC_GRID_IMAGE_MAX_WIDTH)
        return

    # Chunked into rows of at most 3 -- keeps each stat.metric tile roomy
    # rather than squeezing an arbitrary count into one cramped row.
    for start in range(0, len(rows), _MAX_METRICS_PER_ROW):
        chunk = rows[start : start + _MAX_METRICS_PER_ROW]
        columns = st.columns(_MAX_METRICS_PER_ROW)
        for column, (label, value) in zip(columns, chunk):
            column.metric(label, value)


def _render_images(images: list) -> None:
    if len(images) == 1:
        _, center, _ = st.columns([1, 2, 1])
        with center:
            st.image(_to_rgb(images[0]["array"]), caption=images[0]["caption"], width=_SINGLE_IMAGE_MAX_WIDTH)
        return
    columns = st.columns(len(images))
    for column, image in zip(columns, images):
        column.image(_to_rgb(image["array"]), caption=image["caption"], width="stretch")


def _render_table(body: dict) -> None:
    header_cells = "".join(
        f"<th style='text-align:left;padding:6px 12px;border-bottom:1px solid var(--vdx-rule);'>{html.escape(str(h))}</th>"
        for h in body["headers"]
    )
    row_html = "".join(
        "<tr>"
        + "".join(
            f"<td style='padding:6px 12px;border-bottom:1px solid var(--vdx-rule);'>{html.escape(str(cell))}</td>"
            for cell in row
        )
        + "</tr>"
        for row in body["rows"]
    )
    st.markdown(
        f"<table style='width:100%;border-collapse:collapse;'><thead><tr>{header_cells}</tr></thead>"
        f"<tbody>{row_html}</tbody></table>",
        unsafe_allow_html=True,
    )


def render_streamlit(content: ReportContent) -> None:
    st.header("Report Preview")
    st.caption("This is what will be included in the downloadable PDF report.")
    st.caption(f"Patient: {content.patient_id}  ·  Generated: {content.timestamp}")

    for section in content.sections:
        st.markdown(f"#### {section.title}")
        if section.kind == "text":
            st.markdown(section.body)
        elif section.kind == "metric_grid":
            _render_metric_grid(section.body)
        elif section.kind == "table":
            _render_table(section.body)
        elif section.kind == "image":
            _render_images(section.body)
        else:
            raise ValueError(f"Unknown section kind: {section.kind!r}")

    st.markdown(f"<div class='vdx-disclaimer'>{html.escape(content.disclaimer)}</div>", unsafe_allow_html=True)
