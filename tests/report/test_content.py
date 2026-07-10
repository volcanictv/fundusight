import numpy as np

from src.report.content import DISCLAIMER, build_report_content


def _pipeline_result(detection=None, cam_overlay=None, cdr=0.3, disc_found=True):
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


def test_recommendation_flags_elevated_cdr():
    content = build_report_content(_pipeline_result(cdr=0.65))
    rec = next(s for s in content.sections if s.title == "Recommendation")
    assert "cup-to-disc" in rec.body.lower()


def test_recommendation_always_includes_disclaimer():
    content = build_report_content(_pipeline_result())
    rec = next(s for s in content.sections if s.title == "Recommendation")
    assert DISCLAIMER in rec.body


def test_patient_id_defaults_when_blank():
    result = _pipeline_result()
    result["patient_id"] = ""

    content = build_report_content(result)

    assert content.patient_id == "Unspecified"
