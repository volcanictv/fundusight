import cv2
import numpy as np
import torch

from src.detection.model import build_model
from src.report.pipeline import run_pipeline

_EXPECTED_KEYS = {
    "quality",
    "preprocessing_preview",
    "detection",
    "cam_overlay",
    "vessels",
    "optic_disc",
    "working_image",
    "patient_id",
    "timestamp",
}


def _fundus_image(size=700):
    image = np.zeros((size, size, 3), dtype=np.uint8)
    cv2.circle(image, (size // 2, size // 2), int(size * 0.45), (90, 90, 90), -1)
    return image


def test_run_pipeline_without_detection_checkpoint_degrades_gracefully(tmp_path):
    # No classical fallback exists for DR detection (unlike vessels/optic
    # disc) -- a missing checkpoint should leave detection/cam_overlay
    # None rather than raise, and everything else should still compute.
    missing_weights = str(tmp_path / "no_such_checkpoint.pth")

    result = run_pipeline(_fundus_image(), patient_id="P-1", detection_weights_path=missing_weights)

    assert set(result.keys()) == _EXPECTED_KEYS
    assert result["detection"] is None
    assert result["cam_overlay"] is None
    assert result["patient_id"] == "P-1"
    assert result["vessels"] is not None
    assert result["optic_disc"] is not None


def test_run_pipeline_with_detection_checkpoint_populates_detection(tmp_path):
    weights_path = str(tmp_path / "dr_model.pth")
    torch.save(build_model(pretrained=False).state_dict(), weights_path)

    result = run_pipeline(_fundus_image(), detection_weights_path=weights_path)

    assert result["detection"] is not None
    assert set(result["detection"].keys()) == {"label", "probability", "probabilities", "class_idx"}
    assert result["cam_overlay"] is not None


def test_run_pipeline_preprocessing_preview_keeps_raw_image_separate(tmp_path):
    # enhance.preprocess() output is for display only and must never be
    # what detection sees -- this checks the preview pairs the raw image
    # alongside the enhanced one (the "never fed to the model" part is
    # structural in pipeline.py's source, not observable from the return
    # value alone).
    missing_weights = str(tmp_path / "missing.pth")
    raw = _fundus_image()

    result = run_pipeline(raw, detection_weights_path=missing_weights)

    preview = result["preprocessing_preview"]
    assert preview["before"].shape == raw.shape
    assert preview["after"].shape == raw.shape
    assert not np.array_equal(preview["before"], preview["after"])
