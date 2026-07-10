import numpy as np

from src.report.content import build_report_content
from src.report.pdf import generate_pdf


def _minimal_pipeline_result():
    size = 40
    image = np.zeros((size, size, 3), dtype=np.uint8)
    mask = np.zeros((size, size), dtype=bool)
    mask[5:15, 5:15] = True

    return {
        "quality": {
            "score": 72.0,
            "passed": True,
            "checks": {
                "focus": {"passed": True, "score": 80.0, "laplacian_variance": 40.0},
                "exposure": {"passed": True, "score": 65.0, "mean_brightness": 60.0},
            },
        },
        "preprocessing_preview": {"before": image, "after": image},
        "detection": {
            "label": "Mild NPDR",
            "probability": 0.6,
            "probabilities": [0.1, 0.6, 0.1, 0.1, 0.1],
            "class_idx": 1,
        },
        "cam_overlay": image,
        "vessels": {
            "vessel_density": 10.0,
            "branch_count": 3,
            "tortuosity": 1.1,
            "average_width": 2.5,
            "mask": mask,
            "skeleton": mask,
        },
        "optic_disc": {
            "disc_mask": mask,
            "cup_mask": mask,
            "vertical_cdr": 0.4,
            "disc_diameter_px": 20,
            "cup_diameter_px": 8,
            "macula_location": (30, 30),
            "disc_found": True,
            "macula_found": True,
        },
        "working_image": image,
        # Deliberately includes XML-special characters -- patient_id is
        # free-text user input in the app, and _p()'s escaping must keep
        # ReportLab's Paragraph markup parser from choking on it.
        "patient_id": "Test <Patient> & Co",
        "timestamp": "2026-07-10T00:00:00",
    }


def test_generate_pdf_returns_nonempty_pdf_bytes():
    content = build_report_content(_minimal_pipeline_result())

    pdf_bytes = generate_pdf(content)

    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 1000


def test_generate_pdf_handles_missing_detection():
    result = _minimal_pipeline_result()
    result["detection"] = None
    result["cam_overlay"] = None
    content = build_report_content(result)

    pdf_bytes = generate_pdf(content)

    assert pdf_bytes.startswith(b"%PDF")
