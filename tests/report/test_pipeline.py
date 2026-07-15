import cv2
import numpy as np
import torch

from src.detection.model import build_model
from src.report.pipeline import STAGE_NAMES, run_pipeline

_EXPECTED_KEYS = {
    "quality",
    "preprocessing_preview",
    "detection",
    "cam_overlay",
    "glaucoma",
    "glaucoma_cam_overlay",
    "amd",
    "amd_cam_overlay",
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


def _missing(tmp_path, name):
    return str(tmp_path / name)


def _dummy_binary_checkpoint(tmp_path, name):
    # Glaucoma/AMD are both binary (num_classes=2), unlike DR's 5-class
    # build_model() default -- these are deliberately fresh/untrained
    # checkpoints, only testing plumbing (a checkpoint exists -> a
    # prediction + CAM come back), not accuracy.
    path = str(tmp_path / name)
    torch.save(build_model(num_classes=2, pretrained=False).state_dict(), path)
    return path


# Every test below explicitly controls all three classifier weights paths
# (never relying on the real trained checkpoints this repo happens to have
# on disk) -- keeps these tests hermetic/portable to a fresh clone with no
# checkpoints downloaded, same principle the original DR-only tests already
# followed by never relying on detection's own real default checkpoint.


def test_run_pipeline_without_any_classifier_checkpoint_degrades_gracefully(tmp_path):
    # No classical fallback exists for any of the three classifiers (unlike
    # vessels/optic disc) -- a missing checkpoint should leave that
    # classifier's detection/cam_overlay None rather than raise, and
    # everything else should still compute.
    result = run_pipeline(
        _fundus_image(),
        patient_id="P-1",
        detection_weights_path=_missing(tmp_path, "no_dr.pth"),
        glaucoma_weights_path=_missing(tmp_path, "no_glaucoma.pth"),
        amd_weights_path=_missing(tmp_path, "no_amd.pth"),
    )

    assert set(result.keys()) == _EXPECTED_KEYS
    assert result["detection"] is None
    assert result["cam_overlay"] is None
    assert result["glaucoma"] is None
    assert result["glaucoma_cam_overlay"] is None
    assert result["amd"] is None
    assert result["amd_cam_overlay"] is None
    assert result["patient_id"] == "P-1"
    assert result["vessels"] is not None
    assert result["optic_disc"] is not None


def test_run_pipeline_with_detection_checkpoint_populates_detection(tmp_path):
    weights_path = str(tmp_path / "dr_model.pth")
    torch.save(build_model(pretrained=False).state_dict(), weights_path)

    result = run_pipeline(
        _fundus_image(),
        detection_weights_path=weights_path,
        glaucoma_weights_path=_missing(tmp_path, "no_glaucoma.pth"),
        amd_weights_path=_missing(tmp_path, "no_amd.pth"),
    )

    assert result["detection"] is not None
    # uncertainty_std: run_pipeline runs MC-Dropout by default (see mc_dropout.py).
    assert set(result["detection"].keys()) == {"label", "probability", "probabilities", "class_idx", "uncertainty_std"}
    assert 0.0 <= result["detection"]["uncertainty_std"] <= 1.0
    assert result["cam_overlay"] is not None


def test_run_pipeline_with_glaucoma_checkpoint_populates_glaucoma(tmp_path):
    weights_path = _dummy_binary_checkpoint(tmp_path, "glaucoma_model.pth")

    result = run_pipeline(
        _fundus_image(),
        detection_weights_path=_missing(tmp_path, "no_dr.pth"),
        glaucoma_weights_path=weights_path,
        amd_weights_path=_missing(tmp_path, "no_amd.pth"),
    )

    assert result["glaucoma"] is not None
    assert set(result["glaucoma"].keys()) == {"label", "probability", "probabilities", "class_idx", "uncertainty_std"}
    assert result["glaucoma"]["class_idx"] in (0, 1)
    assert result["glaucoma_cam_overlay"] is not None
    assert result["amd"] is None


def test_run_pipeline_with_amd_checkpoint_populates_amd(tmp_path):
    weights_path = _dummy_binary_checkpoint(tmp_path, "amd_model.pth")

    result = run_pipeline(
        _fundus_image(),
        detection_weights_path=_missing(tmp_path, "no_dr.pth"),
        glaucoma_weights_path=_missing(tmp_path, "no_glaucoma.pth"),
        amd_weights_path=weights_path,
    )

    assert result["amd"] is not None
    assert set(result["amd"].keys()) == {"label", "probability", "probabilities", "class_idx", "uncertainty_std"}
    assert result["amd"]["class_idx"] in (0, 1)
    assert result["amd_cam_overlay"] is not None
    assert result["glaucoma"] is None


def test_run_pipeline_preprocessing_preview_keeps_raw_image_separate(tmp_path):
    # enhance.preprocess() output is for display only and must never be
    # what detection sees -- this checks the preview pairs the raw image
    # alongside the enhanced one (the "never fed to the model" part is
    # structural in pipeline.py's source, not observable from the return
    # value alone).
    raw = _fundus_image()

    result = run_pipeline(
        raw,
        detection_weights_path=_missing(tmp_path, "missing_dr.pth"),
        glaucoma_weights_path=_missing(tmp_path, "missing_glaucoma.pth"),
        amd_weights_path=_missing(tmp_path, "missing_amd.pth"),
    )

    preview = result["preprocessing_preview"]
    assert preview["before"].shape == raw.shape
    assert preview["after"].shape == raw.shape
    assert not np.array_equal(preview["before"], preview["after"])


def test_on_stage_fires_for_every_stage_in_order_without_any_classifier_checkpoint(tmp_path):
    # A determinate progress bar (src/app/progress.py) needs a FIXED,
    # known-upfront stage count -- every classifier stage must still fire
    # (with a (None, None) value) even with no checkpoint, so the total
    # never varies at runtime.
    observed = []

    run_pipeline(
        _fundus_image(),
        detection_weights_path=_missing(tmp_path, "missing_dr.pth"),
        glaucoma_weights_path=_missing(tmp_path, "missing_glaucoma.pth"),
        amd_weights_path=_missing(tmp_path, "missing_amd.pth"),
        on_stage=lambda stage, value: observed.append((stage, value)),
    )

    assert [stage for stage, _ in observed] == list(STAGE_NAMES)
    stage_values = dict(observed)
    assert stage_values["detection"] == (None, None)
    assert stage_values["glaucoma"] == (None, None)
    assert stage_values["amd"] == (None, None)
    assert stage_values["quality"] is not None
    vessels_value, working_image = stage_values["vessels"]
    assert vessels_value is not None
    assert working_image is not None


def test_on_stage_fires_for_every_stage_in_order_with_all_classifier_checkpoints(tmp_path):
    dr_weights = str(tmp_path / "dr_model.pth")
    torch.save(build_model(pretrained=False).state_dict(), dr_weights)
    glaucoma_weights = _dummy_binary_checkpoint(tmp_path, "glaucoma_model.pth")
    amd_weights = _dummy_binary_checkpoint(tmp_path, "amd_model.pth")
    observed = []

    run_pipeline(
        _fundus_image(),
        detection_weights_path=dr_weights,
        glaucoma_weights_path=glaucoma_weights,
        amd_weights_path=amd_weights,
        on_stage=lambda stage, value: observed.append((stage, value)),
    )

    assert [stage for stage, _ in observed] == list(STAGE_NAMES)
    stage_values = dict(observed)
    detection, cam_overlay = stage_values["detection"]
    assert detection is not None
    assert cam_overlay is not None
    glaucoma, glaucoma_cam_overlay = stage_values["glaucoma"]
    assert glaucoma is not None
    assert glaucoma_cam_overlay is not None
    amd, amd_cam_overlay = stage_values["amd"]
    assert amd is not None
    assert amd_cam_overlay is not None


def test_on_stage_none_by_default_does_not_raise():
    # The default no-op path -- existing callers that don't pass on_stage
    # must be unaffected.
    result = run_pipeline(_fundus_image())
    assert result["quality"] is not None
