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


def _render_metric_grid(body: dict) -> None:
    # Chunked into rows of at most 3 -- keeps each stat.metric tile roomy
    # rather than squeezing an arbitrary count into one cramped row.
    rows = body["rows"]
    for start in range(0, len(rows), _MAX_METRICS_PER_ROW):
        chunk = rows[start : start + _MAX_METRICS_PER_ROW]
        columns = st.columns(_MAX_METRICS_PER_ROW)
        for column, (label, value) in zip(columns, chunk):
            column.metric(label, value)

    image = body.get("image")
    if image is not None:
        st.image(_to_rgb(image["array"]), caption=image["caption"], width="stretch")


def _render_images(images: list) -> None:
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
