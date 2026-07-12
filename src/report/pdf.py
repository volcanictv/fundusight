"""Phase 8: PDF report renderer.

Turns a report/content.py ReportContent into a print-ready PDF, returned as
raw bytes (an in-memory BytesIO buffer, not a temp file -- both a file save
and Streamlit's st.download_button just want bytes).

A4 page size with fixed margins and no pixel-fixed layout is what makes
this print-friendly by construction -- see the project plan's "print
support" note for why this is the primary print path, rather than trying
to trigger a browser print of the in-app preview.
"""

import io
from xml.sax.saxutils import escape as _esc

import cv2
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Table, TableStyle

from src.report.content import ReportContent

# Apple-like restrained palette: near-black text, one blue accent (used
# sparingly, e.g. rule under table headers), muted gray for secondary text
# -- everything else is white space, not more color.
_TEXT_COLOR = colors.HexColor("#1D1D1F")
_ACCENT_COLOR = colors.HexColor("#0071E3")
_MUTED_COLOR = colors.HexColor("#6E6E73")
_RULE_COLOR = colors.HexColor("#D2D2D7")

_PAGE_MARGIN = 2 * cm
_CONTENT_WIDTH = A4[0] - 2 * _PAGE_MARGIN
_MAX_IMAGE_HEIGHT = 9 * cm


def _styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("FdxTitle", parent=base["Title"], textColor=_TEXT_COLOR, fontSize=22, spaceAfter=4),
        "meta": ParagraphStyle("FdxMeta", parent=base["Normal"], textColor=_MUTED_COLOR, fontSize=9, spaceAfter=4),
        "heading": ParagraphStyle(
            "FdxHeading", parent=base["Heading2"], textColor=_TEXT_COLOR, fontSize=13, spaceBefore=20, spaceAfter=8
        ),
        "body": ParagraphStyle("FdxBody", parent=base["Normal"], textColor=_TEXT_COLOR, fontSize=10.5, leading=15),
        "caption": ParagraphStyle(
            "FdxCaption", parent=base["Normal"], textColor=_MUTED_COLOR, fontSize=8.5, spaceBefore=4, alignment=1
        ),
    }


def _p(text, style, bold: bool = False) -> Paragraph:
    # Every dynamic string (patient_id is free-text user input) goes
    # through here before reaching ReportLab's mini-XML Paragraph markup --
    # otherwise a stray "<" or "&" in the input could break parsing.
    escaped = _esc(str(text))
    if bold:
        escaped = f"<b>{escaped}</b>"
    return Paragraph(escaped, style)


def _fit_dimensions(h: int, w: int, max_width: float, max_height: float) -> tuple:
    scale = min(max_width / w, max_height / h)
    return w * scale, h * scale


def _images_row(images: list, styles: dict, max_height: float = _MAX_IMAGE_HEIGHT) -> Table:
    """Lay out 1-N images side by side in equal columns, each scaled down
    to fit its column width and a shared max height -- used both for a
    single Grad-CAM/mask thumbnail and for the two-up before/after
    preprocessing pair, so a two-image section doesn't blow past one page.
    """
    per_width = _CONTENT_WIDTH / len(images)
    image_cells, caption_cells = [], []
    for image in images:
        h, w = image["array"].shape[:2]
        width, height = _fit_dimensions(h, w, per_width - 0.3 * cm, max_height)
        ok, encoded = cv2.imencode(".png", image["array"])
        image_cells.append(Image(io.BytesIO(encoded.tobytes()), width=width, height=height))
        caption_cells.append(_p(image["caption"], styles["caption"]))

    table = Table([image_cells, caption_cells], colWidths=[per_width] * len(images))
    table.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, 0), "BOTTOM"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    return table


def _metric_table(rows: list, styles: dict) -> Table:
    data = [[_p(label, styles["body"], bold=True), _p(value, styles["body"])] for label, value in rows]
    table = Table(data, colWidths=[_CONTENT_WIDTH * 0.5, _CONTENT_WIDTH * 0.5])
    table.setStyle(
        TableStyle(
            [
                ("LINEBELOW", (0, 0), (-1, -1), 0.5, _RULE_COLOR),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def _data_table(body: dict, styles: dict) -> Table:
    header_row = [_p(h, styles["body"], bold=True) for h in body["headers"]]
    data_rows = [[_p(cell, styles["body"]) for cell in row] for row in body["rows"]]
    table = Table([header_row, *data_rows], colWidths=[_CONTENT_WIDTH * 0.65, _CONTENT_WIDTH * 0.35])
    table.setStyle(
        TableStyle(
            [
                ("LINEBELOW", (0, 0), (-1, 0), 1, _ACCENT_COLOR),
                ("LINEBELOW", (0, 1), (-1, -1), 0.5, _RULE_COLOR),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def _render_section(section, styles: dict) -> KeepTogether:
    flowables = [_p(section.title, styles["heading"], bold=True)]

    if section.kind == "text":
        flowables.append(_p(section.body, styles["body"]))
    elif section.kind == "metric_grid":
        flowables.append(_metric_table(section.body["rows"], styles))
        image = section.body.get("image")
        if image is not None:
            flowables.append(_images_row([image], styles))
    elif section.kind == "table":
        flowables.append(_data_table(section.body, styles))
    elif section.kind == "image":
        flowables.append(_images_row(section.body, styles))
    else:
        raise ValueError(f"Unknown section kind: {section.kind!r}")

    return KeepTogether(flowables)


def _make_footer(content: ReportContent):
    def _on_page(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(_MUTED_COLOR)
        canvas.drawString(_PAGE_MARGIN, 1.2 * cm, content.disclaimer)
        canvas.drawRightString(A4[0] - _PAGE_MARGIN, 1.2 * cm, f"Page {doc.page}")
        canvas.restoreState()

    return _on_page


def generate_pdf(content: ReportContent) -> bytes:
    """Render `content` to a print-ready PDF and return it as raw bytes."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=_PAGE_MARGIN,
        rightMargin=_PAGE_MARGIN,
        topMargin=_PAGE_MARGIN,
        bottomMargin=_PAGE_MARGIN,
        title=f"Fundusight Report - {content.patient_id}",
    )
    styles = _styles()

    flowables = [
        _p("Fundusight Analysis Report", styles["title"]),
        _p(f"Patient: {content.patient_id}  ·  Generated: {content.timestamp}", styles["meta"]),
        _p(content.disclaimer, styles["meta"]),
    ]
    for section in content.sections:
        flowables.append(_render_section(section, styles))

    on_page = _make_footer(content)
    doc.build(flowables, onFirstPage=on_page, onLaterPages=on_page)
    return buffer.getvalue()
