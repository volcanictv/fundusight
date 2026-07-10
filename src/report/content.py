"""Phase 8/9: shared report content model.

`build_report_content()` turns the raw dict pipeline.run_pipeline() returns
into a renderer-agnostic `ReportContent` -- a flat, ordered list of
`Section`s. Both report/pdf.py (ReportLab) and app/render_preview.py
(Streamlit) walk this SAME list and only worry about rendering each
`Section.kind` in their own idiom -- neither one re-derives number
formatting or recommendation text, so the PDF and the in-app preview can
never visually drift apart in what they say, only in how they say it.

This is also the one place "this isn't a diagnosis" framing lives (see
CLAUDE.md: generated copy must never imply this is a clinical/diagnostic
tool) -- every downstream renderer inherits that framing for free just by
walking these sections.
"""

from dataclasses import dataclass, field
from typing import Any

from src.detection.model import SEVERITY_LABELS
from src.report import overlays

DISCLAIMER = (
    "Educational/portfolio demonstration only — not a medical device "
    "and not a substitute for professional diagnosis."
)

# Non-diagnostic, "educational observation" phrasing per DR severity class --
# deliberately never phrased as "you have X". Keyed by class_idx, matching
# detection/model.py's SEVERITY_LABELS.
_SEVERITY_RECOMMENDATIONS = {
    0: "The automated estimate found no signs of diabetic retinopathy in this image.",
    1: (
        "The automated estimate suggests mild non-proliferative diabetic "
        "retinopathy signs are present; periodic monitoring is a reasonable "
        "educational takeaway."
    ),
    2: (
        "The automated estimate suggests moderate non-proliferative diabetic "
        "retinopathy signs are present — this estimate suggests further "
        "review may be warranted."
    ),
    3: (
        "The automated estimate suggests severe non-proliferative diabetic "
        "retinopathy signs are present — this estimate suggests further "
        "review may be warranted."
    ),
    4: (
        "The automated estimate suggests proliferative diabetic retinopathy "
        "signs are present — this estimate suggests further review may "
        "be warranted."
    ),
}

# Vertical CDR above this is flagged as an educational observation only --
# not a glaucoma diagnosis. 0.5-0.6 is a commonly cited screening cutoff in
# the literature; treated here purely as a talking point, not a threshold
# this project claims clinical validity for.
_ELEVATED_CDR_THRESHOLD = 0.5


@dataclass
class Section:
    title: str
    kind: str  # "text" | "metric_grid" | "image" | "table"
    body: Any


@dataclass
class ReportContent:
    patient_id: str
    timestamp: str
    disclaimer: str
    sections: list[Section] = field(default_factory=list)


def _quality_section(quality: dict) -> Section:
    focus, exposure = quality["checks"]["focus"], quality["checks"]["exposure"]
    rows = [
        ("Overall score", f"{quality['score']:.1f} / 100"),
        ("Passed", "Yes" if quality["passed"] else "No"),
        ("Focus score", f"{focus['score']:.1f} / 100"),
        ("Exposure score", f"{exposure['score']:.1f} / 100"),
    ]
    return Section(title="Image Quality", kind="metric_grid", body={"rows": rows})


def _preprocessing_section(preprocessing_preview: dict) -> Section:
    images = [
        {"caption": "Original", "array": preprocessing_preview["before"]},
        {"caption": "Preprocessed (illumination + CLAHE + color norm)", "array": preprocessing_preview["after"]},
    ]
    return Section(title="Preprocessing Preview", kind="image", body=images)


def _detection_sections(detection: dict | None, cam_overlay) -> list[Section]:
    if detection is None:
        text = (
            "The diabetic retinopathy detection model is not available in "
            "this build — no trained checkpoint was found. Quality, "
            "vessel, and optic disc measurements below are independent of "
            "detection and unaffected."
        )
        return [Section(title="Diabetic Retinopathy Detection", kind="text", body=text)]

    headers = ["Severity", "Probability"]
    rows = [
        [SEVERITY_LABELS[i], f"{p * 100:.1f}%"]
        for i, p in enumerate(detection["probabilities"])
    ]
    top_line = f"Top estimate: {detection['label']} ({detection['probability'] * 100:.1f}% confidence)"
    sections = [
        Section(title="Diabetic Retinopathy Detection", kind="text", body=top_line),
        Section(title="Severity Probabilities", kind="table", body={"headers": headers, "rows": rows}),
    ]
    if cam_overlay is not None:
        cam_images = [{"caption": "Grad-CAM attention map", "array": cam_overlay}]
        sections.append(Section(title="Explainability", kind="image", body=cam_images))
    return sections


def _vessel_section(vessels_result: dict, working_image) -> Section:
    rows = [
        ("Vessel density", f"{vessels_result['vessel_density']:.2f}%"),
        ("Branch point count", str(vessels_result["branch_count"])),
        ("Tortuosity", f"{vessels_result['tortuosity']:.3f}"),
        ("Average width", f"{vessels_result['average_width']:.2f} px"),
    ]
    overlay = overlays.vessel_mask_overlay(working_image, vessels_result)
    return Section(
        title="Vessel Biomarkers",
        kind="metric_grid",
        body={"rows": rows, "image": {"caption": "Vessel mask overlay", "array": overlay}},
    )


def _optic_disc_section(optic_disc_result: dict, working_image) -> Section:
    macula = optic_disc_result["macula_location"]
    rows = [
        ("Vertical cup-to-disc ratio", f"{optic_disc_result['vertical_cdr']:.3f}"),
        ("Disc diameter", f"{optic_disc_result['disc_diameter_px']} px"),
        ("Cup diameter", f"{optic_disc_result['cup_diameter_px']} px"),
        ("Disc located", "Yes" if optic_disc_result["disc_found"] else "No"),
        ("Macula located", "Yes" if optic_disc_result["macula_found"] else "No"),
        ("Macula position", f"{macula}" if macula is not None else "Not found"),
    ]
    overlay = overlays.optic_disc_overlay(working_image, optic_disc_result)
    return Section(
        title="Optic Disc / Cup / Macula",
        kind="metric_grid",
        body={"rows": rows, "image": {"caption": "Disc (yellow) / cup (red) / macula (green)", "array": overlay}},
    )


def _build_recommendation(quality: dict, detection: dict | None, optic_disc_result: dict) -> str:
    parts = []
    if not quality["passed"]:
        parts.append(
            f"Image quality did not meet the pass threshold (score "
            f"{quality['score']:.0f}/100) — consider retaking the photo "
            f"with better focus or exposure before relying on the findings below."
        )

    if detection is None:
        parts.append(
            "No diabetic retinopathy severity estimate is available for this "
            "image (detection model not loaded)."
        )
    else:
        parts.append(_SEVERITY_RECOMMENDATIONS[detection["class_idx"]])

    if optic_disc_result["disc_found"] and optic_disc_result["vertical_cdr"] >= _ELEVATED_CDR_THRESHOLD:
        parts.append(
            f"The estimated vertical cup-to-disc ratio "
            f"({optic_disc_result['vertical_cdr']:.2f}) is on the higher end "
            f"of the typical range — an educational observation only, "
            f"not a glaucoma diagnosis."
        )

    parts.append(DISCLAIMER)
    return " ".join(parts)


def build_report_content(pipeline_result: dict) -> ReportContent:
    """Convert pipeline.run_pipeline()'s raw dict into a ReportContent both
    renderers can walk. Section order here IS the report's reading order.
    """
    working_image = pipeline_result["working_image"]

    sections = [
        _quality_section(pipeline_result["quality"]),
        _preprocessing_section(pipeline_result["preprocessing_preview"]),
        *_detection_sections(pipeline_result["detection"], pipeline_result["cam_overlay"]),
        _vessel_section(pipeline_result["vessels"], working_image),
        _optic_disc_section(pipeline_result["optic_disc"], working_image),
        Section(
            title="Recommendation",
            kind="text",
            body=_build_recommendation(
                pipeline_result["quality"], pipeline_result["detection"], pipeline_result["optic_disc"]
            ),
        ),
    ]

    return ReportContent(
        patient_id=pipeline_result["patient_id"] or "Unspecified",
        timestamp=pipeline_result["timestamp"],
        disclaimer=DISCLAIMER,
        sections=sections,
    )
