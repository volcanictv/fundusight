"""Phase 8/9: pipeline orchestrator.

Runs every independent pipeline stage (quality, preprocessing preview,
DR detection + Grad-CAM, vessel biomarkers, optic disc/cup/CDR + macula) on
one uploaded fundus photo and assembles a single dict. Both the PDF report
(report/pdf.py, via report/content.py) and the Streamlit dashboard
(app/main.py) call this ONE function rather than each re-deriving the same
sequence of calls -- keeps them from ever drifting out of sync with each
other or with the underlying inference contracts.

Only reads the documented dict/array contracts each stage already exposes
(see e.g. vessel_infer.compute_biomarkers_auto, optic_disc_infer.
compute_optic_biomarkers_auto) -- never a specific accuracy number or
checkpoint detail, so swapping in a retrained checkpoint later (see
ROADMAP.md's Phase 6 note) requires no changes here.
"""

import datetime
import functools
import os
from typing import Callable

import numpy as np
import torch

from src.detection.infer import DEFAULT_WEIGHTS_PATH as DETECTION_DEFAULT_WEIGHTS_PATH
from src.detection.infer import load_model, predict
from src.explainability.gradcam import generate_cam
from src.preprocessing.enhance import preprocess
from src.preprocessing.quality import assess_quality
from src.segmentation import vessels
from src.segmentation.optic_disc_infer import compute_optic_biomarkers_auto
from src.segmentation.vessel_infer import compute_biomarkers_auto

# Single source of truth for progress-callback stage count/order -- the app
# layer (src/app/main.py, src/app/progress.py) imports this rather than
# duplicating the list, so a determinate progress bar always knows its true
# total. Detection and Grad-CAM are deliberately ONE stage here (not two) so
# this count stays fixed regardless of whether a DR checkpoint exists --
# splitting them would make the total vary at runtime, which breaks a
# determinate (not spinner-style) progress bar.
STAGE_NAMES = ("quality", "preprocessing", "detection", "vessels", "optic_disc")


@functools.lru_cache(maxsize=1)
def _cached_detection_model(weights_path: str, device: str) -> torch.nn.Module:
    # DR detection has no built-in caching of its own (unlike vessel_infer/
    # optic_disc_infer's _cached_model) -- this is the equivalent for
    # run_pipeline(), which otherwise reloads the checkpoint on every call.
    return load_model(weights_path, device)


def _run_detection(image: np.ndarray, weights_path: str, cam_method: str, device: str) -> tuple:
    """Returns (detection, cam_overlay), both None if no DR checkpoint is
    available -- there's no classical fallback for DR grading, so this is
    the one stage the rest of the pipeline must be able to run without.
    """
    if not os.path.exists(weights_path):
        return None, None

    model = _cached_detection_model(weights_path, device)
    detection = predict(model, image, device)
    cam_overlay = generate_cam(model, image, method=cam_method, target_class=detection["class_idx"])
    return detection, cam_overlay


def run_pipeline(
    image: np.ndarray,
    patient_id: str = "",
    cam_method: str = "gradcam",
    detection_weights_path: str = DETECTION_DEFAULT_WEIGHTS_PATH,
    device: str = "cpu",
    on_stage: Callable[[str, object], None] | None = None,
) -> dict:
    """Run the full analysis pipeline on one BGR fundus photo (cv2.imread
    convention, matching every stage this calls into).

    Returns a dict with stable top-level keys regardless of which trained
    checkpoints are present: "quality", "preprocessing_preview" (dict with
    "before"/"after" BGR arrays), "detection" (dict or None if unavailable),
    "cam_overlay" (BGR array or None, paired with "detection"), "vessels",
    "optic_disc", "working_image" (the shared VESSEL_WORKING_WIDTH-resolution
    copy vessel/optic-disc masks are already aligned to -- see
    vessels._resize_to_working_width), "patient_id", "timestamp".

    `preprocess()`'s output is for display only (a before/after panel) --
    it is deliberately NOT fed into detection or any other stage, since
    dataset.py documents that the classifier trains on plain resize +
    ImageNet normalization, and enhance.py's color normalization would
    create a train/inference mismatch.

    `on_stage`, if given, is called once per STAGE_NAMES entry, in order,
    right after that stage finishes, as `on_stage(stage_name, value)` --
    this module stays Streamlit-agnostic (no import of streamlit here); the
    caller (src/app/main.py) is the one that turns these callbacks into a
    progress bar and progressive section rendering. `value` matches the
    shape each stage's render function needs directly: the quality dict,
    the preprocessing preview dict, `(detection, cam_overlay)`,
    `(vessel_result, working_image)`, `(optic_disc_result, working_image)`.
    working_image is computed up front (cheap, pure -- no dependency on any
    other stage) specifically so the vessels/optic-disc callbacks can carry
    it alongside their own result.
    """

    def _emit(stage: str, value: object) -> None:
        if on_stage is not None:
            on_stage(stage, value)

    working_image = vessels._resize_to_working_width(image)

    quality = assess_quality(image)
    _emit("quality", quality)

    preprocessing_preview = {"before": image, "after": preprocess(image)}
    _emit("preprocessing", preprocessing_preview)

    detection, cam_overlay = _run_detection(image, detection_weights_path, cam_method, device)
    _emit("detection", (detection, cam_overlay))

    vessel_result = compute_biomarkers_auto(image, device=device)
    _emit("vessels", (vessel_result, working_image))

    optic_disc_result = compute_optic_biomarkers_auto(image, device=device)
    _emit("optic_disc", (optic_disc_result, working_image))

    return {
        "quality": quality,
        "preprocessing_preview": preprocessing_preview,
        "detection": detection,
        "cam_overlay": cam_overlay,
        "vessels": vessel_result,
        "optic_disc": optic_disc_result,
        "working_image": working_image,
        "patient_id": patient_id,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }
