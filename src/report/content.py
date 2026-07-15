"""Phase 8/9: shared report content model.

`build_report_content()` turns the raw dict pipeline.run_pipeline() returns
into a renderer-agnostic `ReportContent` -- a flat, ordered list of
`Section`s. Both report/pdf.py (ReportLab) and app/render_preview.py
(Streamlit) walk this SAME list and only worry about rendering each
`Section.kind` in their own idiom -- neither one re-derives number
formatting or recommendation text, so the PDF and the in-app preview can
never visually drift apart in what they say, only in how they say it.

This is also the one place "this isn't a diagnosis" framing lives -- generated
copy must never imply this is a clinical/diagnostic tool, and every downstream
renderer inherits that framing for free just by walking these sections.
"""

from dataclasses import dataclass, field
from typing import Any

from src.detection.amd_infer import AMD_LABELS
from src.detection.glaucoma_infer import GLAUCOMA_LABELS
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

# Same non-diagnostic phrasing convention as _SEVERITY_RECOMMENDATIONS
# above, keyed by class_idx (0=absent, 1=present) matching
# glaucoma_infer.GLAUCOMA_LABELS / amd_infer.AMD_LABELS. Deliberately does
# NOT repeat "educational observation only, not a diagnosis" per finding --
# when multiple findings are positive in one report (DR + glaucoma + AMD +
# an elevated-CDR note), that meta-caveat stacking up to 3-4 times in one
# paragraph is itself the repetition problem; the single DISCLAIMER
# appended once at the end of every recommendation already carries that
# framing (see _build_recommendation()), so each finding sentence below
# only needs to state the finding itself.
_GLAUCOMA_RECOMMENDATIONS = {
    0: "The automated glaucoma estimate found no signs of glaucoma in this image.",
    1: "The automated glaucoma estimate suggests glaucoma signs may be present; further review may be warranted.",
}
_AMD_RECOMMENDATIONS = {
    0: "The automated AMD estimate found no signs of age-related macular degeneration in this image.",
    1: (
        "The automated AMD estimate suggests signs of age-related macular "
        "degeneration may be present; further review may be warranted."
    ),
}


def _confidence_phrase(detection: dict) -> str:
    """'87.0% ± 4.2%' when Monte-Carlo Dropout uncertainty is present (see
    detection/mc_dropout.py), else the plain '87.0% confidence'. The ± figure
    is the 1-sigma spread of the top class's probability across MC passes -- an
    approximate (epistemic) uncertainty, not calibrated probability."""
    pct = detection["probability"] * 100
    std = detection.get("uncertainty_std")
    if std is not None:
        return f"{pct:.1f}% ± {std * 100:.1f}%"
    return f"{pct:.1f}% confidence"


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
        # The raw measurements each 0-100 score above is derived from
        # (preprocessing/quality.py's check_focus/check_exposure) were
        # computed every run but never surfaced anywhere -- not the app,
        # not the PDF -- only the normalized score was. A reader deciding
        # whether to trust a borderline score benefits from seeing the
        # actual measurement it came from, not just the 0-100 result.
        ("Focus (Laplacian variance)", f"{focus['laplacian_variance']:.1f}"),
        ("Exposure score", f"{exposure['score']:.1f} / 100"),
        ("Exposure (mean brightness)", f"{exposure['mean_brightness']:.1f} / 255"),
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
    top_line = f"Top estimate: {detection['label']} ({_confidence_phrase(detection)})"
    sections = [
        Section(title="Diabetic Retinopathy Detection", kind="text", body=top_line),
        Section(title="Severity Probabilities", kind="table", body={"headers": headers, "rows": rows}),
    ]
    if cam_overlay is not None:
        cam_images = [{"caption": "Grad-CAM attention map", "array": cam_overlay}]
        sections.append(Section(title="Explainability", kind="image", body=cam_images))
    return sections


def _binary_classifier_sections(
    title: str, detection: dict | None, cam_overlay, labels: dict, unavailable_text: str
) -> list[Section]:
    """Shared by the glaucoma and AMD sections -- unlike DR's 5-class
    _detection_sections(), both are genuinely identical in shape (binary
    label, same probability/table/optional-CAM layout), so factoring the
    two into one function here is justified rather than premature.
    """
    if detection is None:
        return [Section(title=title, kind="text", body=unavailable_text)]

    headers = ["Finding", "Probability"]
    rows = [[labels[i], f"{p * 100:.1f}%"] for i, p in enumerate(detection["probabilities"])]
    top_line = f"Top estimate: {detection['label']} ({_confidence_phrase(detection)})"
    sections = [
        Section(title=title, kind="text", body=top_line),
        Section(title=f"{title} Probabilities", kind="table", body={"headers": headers, "rows": rows}),
    ]
    if cam_overlay is not None:
        cam_images = [{"caption": "Grad-CAM attention map", "array": cam_overlay}]
        sections.append(Section(title=f"{title} Explainability", kind="image", body=cam_images))
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
    confident = optic_disc_result["disc_confident"]
    rows = [
        ("Vertical cup-to-disc ratio", f"{optic_disc_result['vertical_cdr']:.3f}"),
        ("Disc diameter", f"{optic_disc_result['disc_diameter_px']} px"),
        ("Cup diameter", f"{optic_disc_result['cup_diameter_px']} px"),
        ("Disc located", "Yes" if optic_disc_result["disc_found"] else "No"),
        ("Localization confidence", "OK" if confident else "Low — CDR unreliable"),
        ("Macula located", "Yes" if optic_disc_result["macula_found"] else "No"),
        ("Macula position", f"{macula}" if macula is not None else "Not found"),
    ]
    overlay = overlays.optic_disc_overlay(working_image, optic_disc_result)
    return Section(
        title="Optic Disc / Cup / Macula",
        kind="metric_grid",
        body={"rows": rows, "image": {"caption": "Disc (yellow) / cup (red) / macula (green)", "array": overlay}},
    )


def _build_recommendation(
    quality: dict, detection: dict | None, glaucoma: dict | None, amd: dict | None, optic_disc_result: dict
) -> str:
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

    if glaucoma is None:
        parts.append("No glaucoma estimate is available for this image (glaucoma model not loaded).")
    else:
        parts.append(_GLAUCOMA_RECOMMENDATIONS[glaucoma["class_idx"]])

    if amd is None:
        parts.append("No AMD estimate is available for this image (AMD model not loaded).")
    else:
        parts.append(_AMD_RECOMMENDATIONS[amd["class_idx"]])

    # A CDR is only as trustworthy as the crop it was measured from. Stage
    # 6.1's classical localizer can land on a large hemorrhage or a dense
    # exudate cluster instead of the disc (on ADAM's ground truth it does so
    # on 38/270 images); Stage 6.2 will then happily segment a "disc" and
    # "cup" out of that wrong crop and Stage 6.3 will report a perfectly
    # confident-looking ratio measured off the wrong anatomy. So when the
    # geometric plausibility checks reject the localization, the CDR is
    # reported as unreliable AND withheld from the elevated-CDR observation
    # below -- an elevated CDR derived from a hemorrhage is not a finding,
    # it's an artifact, and stating it would be worse than saying nothing.
    disc_localized_well = optic_disc_result["disc_found"] and optic_disc_result["disc_confident"]
    if optic_disc_result["disc_found"] and not optic_disc_result["disc_confident"]:
        warnings = optic_disc_result["disc_localization_warnings"]
        detail = f" ({warnings[0]})" if warnings else ""
        parts.append(
            f"The optic disc could not be localized with confidence on this "
            f"image{detail}, so the cup-to-disc ratio above should not be "
            f"relied on — it may have been measured from a bright lesion "
            f"rather than the disc itself."
        )

    cdr_elevated = disc_localized_well and optic_disc_result["vertical_cdr"] >= _ELEVATED_CDR_THRESHOLD
    if cdr_elevated:
        # No repeated "educational observation only, not a diagnosis" here
        # either -- see the module-level comment above _GLAUCOMA_
        # RECOMMENDATIONS for why; the disclaimer at the end already
        # covers it once.
        parts.append(
            f"The estimated vertical cup-to-disc ratio "
            f"({optic_disc_result['vertical_cdr']:.2f}) is on the higher end "
            f"of the typical range."
        )

    # Two independent glaucoma-relevant signals now exist on this report
    # (the CDR threshold above and the glaucoma classifier). Surface it
    # explicitly when they point different directions rather than leaving
    # the reader to notice the tension themselves -- both are approximate
    # estimates, neither should silently override the other.
    # Gated on disc_localized_well, not just disc_found: a disagreement
    # between the classifier and a CDR we already know is untrustworthy isn't
    # a real disagreement worth surfacing, it's just the bad crop talking.
    if disc_localized_well and glaucoma is not None:
        classifier_flags_glaucoma = glaucoma["class_idx"] == 1
        if cdr_elevated != classifier_flags_glaucoma:
            parts.append(
                "Note: the cup-to-disc ratio observation and the glaucoma "
                "classifier's estimate point in different directions for "
                "this image — both are approximate, independent signals; "
                "neither should be treated as a confirmed finding on its own."
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
        *_binary_classifier_sections(
            "Glaucoma Detection",
            pipeline_result["glaucoma"],
            pipeline_result["glaucoma_cam_overlay"],
            GLAUCOMA_LABELS,
            "The glaucoma detection model is not available in this build — "
            "no trained checkpoint was found.",
        ),
        *_binary_classifier_sections(
            "Age-Related Macular Degeneration (AMD) Detection",
            pipeline_result["amd"],
            pipeline_result["amd_cam_overlay"],
            AMD_LABELS,
            "The AMD detection model is not available in this build — no "
            "trained checkpoint was found.",
        ),
        _vessel_section(pipeline_result["vessels"], working_image),
        _optic_disc_section(pipeline_result["optic_disc"], working_image),
        Section(
            title="Recommendation",
            kind="text",
            body=_build_recommendation(
                pipeline_result["quality"],
                pipeline_result["detection"],
                pipeline_result["glaucoma"],
                pipeline_result["amd"],
                pipeline_result["optic_disc"],
            ),
        ),
    ]

    return ReportContent(
        patient_id=pipeline_result["patient_id"] or "Unspecified",
        timestamp=pipeline_result["timestamp"],
        disclaimer=DISCLAIMER,
        sections=sections,
    )
