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
    """
    quality = assess_quality(image)
    preprocessing_preview = {"before": image, "after": preprocess(image)}

    detection, cam_overlay = _run_detection(image, detection_weights_path, cam_method, device)

    vessel_result = compute_biomarkers_auto(image, device=device)
    optic_disc_result = compute_optic_biomarkers_auto(image, device=device)
    working_image = vessels._resize_to_working_width(image)

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
