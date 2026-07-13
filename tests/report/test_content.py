import numpy as np

from src.report.content import DISCLAIMER, build_report_content


def _pipeline_result(
    detection=None,
    cam_overlay=None,
    glaucoma=None,
    glaucoma_cam_overlay=None,
    amd=None,
    amd_cam_overlay=None,
    cdr=0.3,
    disc_found=True,
    disc_confident=True,
    disc_localization_warnings=None,
):
    size = 50
    working_image = np.zeros((size, size, 3), dtype=np.uint8)
    mask = np.zeros((size, size), dtype=bool)
    mask[10:20, 10:20] = True

    return {
        "quality": {
            "score": 88.0,
            "passed": True,
            "checks": {
                "focus": {"passed": True, "score": 90.0, "laplacian_variance": 50.0},
                "exposure": {"passed": True, "score": 86.0, "mean_brightness": 70.0},
            },
        },
        "preprocessing_preview": {"before": working_image, "after": working_image},
        "detection": detection,
        "cam_overlay": cam_overlay,
        "glaucoma": glaucoma,
        "glaucoma_cam_overlay": glaucoma_cam_overlay,
        "amd": amd,
        "amd_cam_overlay": amd_cam_overlay,
        "vessels": {
            "vessel_density": 12.345,
            "branch_count": 7,
            "tortuosity": 1.023,
            "average_width": 3.4,
            "mask": mask,
            "skeleton": mask,
        },
        "optic_disc": {
            "disc_mask": mask,
            "cup_mask": mask,
            "vertical_cdr": cdr,
            "disc_diameter_px": 40,
            "cup_diameter_px": 12,
            "macula_location": (5, 5) if disc_found else None,
            "disc_found": disc_found,
            "disc_confident": disc_confident,
            "disc_localization_warnings": disc_localization_warnings or [],
            "macula_found": disc_found,
        },
        "working_image": working_image,
        "patient_id": "P-42",
        "timestamp": "2026-07-10T00:00:00",
    }


def test_build_report_content_without_detection_explains_unavailable():
    content = build_report_content(_pipeline_result(detection=None))

    detection_section = next(s for s in content.sections if s.title == "Diabetic Retinopathy Detection")
    assert detection_section.kind == "text"
    assert "not available" in detection_section.body
    assert not any(s.title == "Severity Probabilities" for s in content.sections)
    assert not any(s.title == "Explainability" for s in content.sections)


def test_build_report_content_with_detection_includes_probabilities_and_cam():
    detection = {
        "label": "Moderate NPDR",
        "probability": 0.7,
        "probabilities": [0.05, 0.1, 0.7, 0.1, 0.05],
        "class_idx": 2,
    }
    cam = np.zeros((10, 10, 3), dtype=np.uint8)

    content = build_report_content(_pipeline_result(detection=detection, cam_overlay=cam))

    prob_section = next(s for s in content.sections if s.title == "Severity Probabilities")
    assert len(prob_section.body["rows"]) == 5
    assert any(s.title == "Explainability" for s in content.sections)


def _binary_detection(class_idx, label):
    probabilities = [0.0, 0.0]
    probabilities[class_idx] = 0.9
    probabilities[1 - class_idx] = 0.1
    return {"label": label, "probability": 0.9, "probabilities": probabilities, "class_idx": class_idx}


def test_build_report_content_without_glaucoma_explains_unavailable():
    content = build_report_content(_pipeline_result(glaucoma=None))

    section = next(s for s in content.sections if s.title == "Glaucoma Detection")
    assert section.kind == "text"
    assert "not available" in section.body
    assert not any(s.title == "Glaucoma Detection Probabilities" for s in content.sections)


def test_build_report_content_with_glaucoma_includes_probabilities_and_cam():
    glaucoma = _binary_detection(1, "Glaucoma Signs Present")
    cam = np.zeros((10, 10, 3), dtype=np.uint8)

    content = build_report_content(_pipeline_result(glaucoma=glaucoma, glaucoma_cam_overlay=cam, cdr=0.65))

    prob_section = next(s for s in content.sections if s.title == "Glaucoma Detection Probabilities")
    assert len(prob_section.body["rows"]) == 2
    assert any(s.title == "Glaucoma Detection Explainability" for s in content.sections)


def test_build_report_content_without_amd_explains_unavailable():
    content = build_report_content(_pipeline_result(amd=None))

    section = next(s for s in content.sections if "AMD" in s.title and s.kind == "text")
    assert "not available" in section.body


def test_build_report_content_with_amd_includes_probabilities_and_cam():
    amd = _binary_detection(1, "AMD Signs Present")
    cam = np.zeros((10, 10, 3), dtype=np.uint8)

    content = build_report_content(_pipeline_result(amd=amd, amd_cam_overlay=cam))

    prob_section = next(s for s in content.sections if "AMD" in s.title and s.kind == "table")
    assert len(prob_section.body["rows"]) == 2
    assert any("AMD" in s.title and s.kind == "image" for s in content.sections)


def test_recommendation_flags_elevated_cdr():
    content = build_report_content(_pipeline_result(cdr=0.65))
    rec = next(s for s in content.sections if s.title == "Recommendation")
    assert "cup-to-disc" in rec.body.lower()


def test_recommendation_flags_disagreement_between_cdr_and_glaucoma_classifier():
    # Elevated CDR but classifier says no glaucoma signs -- the two signals
    # disagree, so the recommendation should say so explicitly.
    glaucoma = _binary_detection(0, "No Glaucoma Signs")
    content = build_report_content(_pipeline_result(glaucoma=glaucoma, cdr=0.65))

    rec = next(s for s in content.sections if s.title == "Recommendation")
    assert "different directions" in rec.body


def test_recommendation_no_disagreement_note_when_signals_agree():
    # Elevated CDR and classifier both flag glaucoma -- no disagreement to report.
    glaucoma = _binary_detection(1, "Glaucoma Signs Present")
    content = build_report_content(_pipeline_result(glaucoma=glaucoma, cdr=0.65))

    rec = next(s for s in content.sections if s.title == "Recommendation")
    assert "different directions" not in rec.body


def test_recommendation_warns_when_disc_localization_is_low_confidence():
    content = build_report_content(
        _pipeline_result(disc_confident=False, disc_localization_warnings=["not disc-shaped (circularity 0.05 < 0.19)"])
    )
    rec = next(s for s in content.sections if s.title == "Recommendation")

    assert "could not be localized with confidence" in rec.body
    assert "not disc-shaped" in rec.body


def test_low_confidence_localization_suppresses_elevated_cdr_observation():
    # The whole point of the plausibility check: an "elevated" CDR measured
    # off a bright lesion instead of the disc is an artifact, not a finding.
    # It must not be stated as an observation just because the number is high.
    content = build_report_content(_pipeline_result(cdr=0.65, disc_confident=False))
    rec = next(s for s in content.sections if s.title == "Recommendation")

    assert "higher end" not in rec.body
    assert "should not be relied on" in rec.body


def test_low_confidence_localization_suppresses_glaucoma_disagreement_note():
    # A disagreement between the classifier and a CDR already known to be
    # untrustworthy isn't a real disagreement -- it's the bad crop talking.
    glaucoma = _binary_detection(0, "No Glaucoma Signs")
    content = build_report_content(_pipeline_result(glaucoma=glaucoma, cdr=0.65, disc_confident=False))

    rec = next(s for s in content.sections if s.title == "Recommendation")
    assert "different directions" not in rec.body


def test_optic_disc_section_reports_localization_confidence():
    confident = build_report_content(_pipeline_result(disc_confident=True))
    low = build_report_content(_pipeline_result(disc_confident=False))

    def _rows(content):
        section = next(s for s in content.sections if s.title == "Optic Disc / Cup / Macula")
        return dict(section.body["rows"])

    assert _rows(confident)["Localization confidence"] == "OK"
    assert "Low" in _rows(low)["Localization confidence"]


def test_recommendation_always_includes_disclaimer():
    content = build_report_content(_pipeline_result())
    rec = next(s for s in content.sections if s.title == "Recommendation")
    assert DISCLAIMER in rec.body


def test_patient_id_defaults_when_blank():
    result = _pipeline_result()
    result["patient_id"] = ""

    content = build_report_content(result)

    assert content.patient_id == "Unspecified"
