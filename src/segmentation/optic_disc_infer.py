"""Phase 6 (Stages 6.2/6.3): local inference with the trained optic
disc/cup model.

Mirrors vessel_infer.py's split from vessels.py: this is the only file in
src/segmentation/ that imports torch for the optic disc/cup *hybrid* path,
keeping optic_disc.py itself classical and torch-free (see its module
docstring) for anyone who only needs the classical baseline.
"""

import functools
import os

import numpy as np
import torch

from src.segmentation import optic_disc, vessels
from src.segmentation.optic_disc_model import build_optic_disc_model

# Matches optic_disc_train.py's --output default and scripts/demo_optic_disc.py's
# WEIGHTS_PATH -- the one place downstream callers should look for a
# trained checkpoint by default.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_WEIGHTS_PATH = os.path.join(_PROJECT_ROOT, "checkpoints", "optic_disc_unet.pth")

# Two independent attempts at a post-hoc probability-threshold
# recalibration (replacing plain argmax) were tried here to counter the
# systematic disc/cup over-segmentation investigate_cdr_segmentation_
# bias.py found (predicted disc/cup masks ~1.5x/~3x the ground-truth area
# even with perfect Stage 6.1 localization):
#   1. Calibrated against ground-truth ROI crops: looked good in isolation
#      (mean |CDR error| 0.117 -> 0.101 on that condition) but REGRESSED
#      the real held-out test pipeline (0.241 -> 0.289).
#   2. Calibrated against the REAL Stage 6.1+6.2 pipeline on the
#      validation split instead (the methodologically correct fix for
#      attempt 1's flaw): looked even better on validation (0.130 ->
#      0.107) but REGRESSED the real held-out test pipeline WORSE than
#      attempt 1 (0.241 -> 0.339).
# Both reverted. Two independent, methodologically-corrected calibration
# attempts failing to transfer from validation to test -- with the second,
# more careful attempt failing worse than the first -- points at a
# validation/test distribution mismatch large enough that post-hoc
# threshold tuning on 400 validation images isn't a reliable fix here, not
# a tuning-methodology problem. See scripts/calibrate_optic_disc_
# thresholds.py and scripts/investigate_cdr_segmentation_bias.py for the
# full investigation; a retrain (loss reweighting, boundary-aware loss
# term, or more regularization against this over-segmentation tendency)
# is the more promising next step, not further threshold search.


def load_optic_disc_model(weights_path: str, device: str = "cpu") -> torch.nn.Module:
    """Build the architecture and load trained weights, ready for inference."""
    model = build_optic_disc_model()
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def segment_disc_cup_hybrid(roi_image: np.ndarray, model: torch.nn.Module, device: str = "cpu") -> tuple:
    """Hybrid disc/cup mask within an already-cropped ROI:
    optic_disc.extract_color_features()'s 7-channel input feeds the
    trained U-Net directly, argmax over the class logits picks a class per
    pixel, disc = rim-or-cup classes, cup = cup class alone. Both run
    through optic_disc.clean_disc_cup_masks() before returning -- a
    per-pixel argmax decision can leave stray disc fragments or edge-
    bleeding speckles disconnected from the true disc (observed on real
    APTOS demo images), and a cup blob that's drifted away from the disc
    center. This is the post-processing step applied right after
    thresholding the raw class predictions and before any CDR geometry is
    computed from them -- same single choke point classical
    (segment_disc_cup_classical()) and hybrid both route through.

    Plain argmax is deliberate here, not an oversight -- two independent
    probability-threshold recalibration attempts were tried and both
    regressed the held-out test pipeline worse than this. See the module
    docstring above DEFAULT_WEIGHTS_PATH for the full investigation.
    """
    input_arr = optic_disc.extract_color_features(roi_image)
    input_tensor = torch.from_numpy(input_arr).unsqueeze(0).to(device)

    logits = model(input_tensor)
    predicted_class = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

    disc_mask = predicted_class != 0  # class 1 (rim) or class 2 (cup)
    cup_mask = predicted_class == 2
    return optic_disc.clean_disc_cup_masks(disc_mask, cup_mask)


def compute_optic_biomarkers_hybrid(image: np.ndarray, model: torch.nn.Module, device: str = "cpu") -> dict:
    """Same structure and return contract as
    optic_disc.compute_optic_biomarkers(), swapping the classical
    intensity-threshold disc/cup mask for the trained model's mask. Stage
    6.1's classical localization/crop always runs first regardless -- there
    is no trained replacement for it, see ROADMAP.md.
    """
    working = vessels._resize_to_working_width(image)
    disc_info = optic_disc.locate_disc_classical(working)
    roi_image, bbox_meta = optic_disc.crop_disc_roi(working, disc_info["center_xy"], disc_info["diameter_px"])
    roi_disc_mask, roi_cup_mask = segment_disc_cup_hybrid(roi_image, model, device)

    disc_mask = optic_disc.reproject_roi_mask_to_working(roi_disc_mask, bbox_meta, working.shape)
    cup_mask = optic_disc.reproject_roi_mask_to_working(roi_cup_mask, bbox_meta, working.shape)
    cdr_info = optic_disc.compute_cdr(disc_mask, cup_mask)
    macula_info = optic_disc.locate_macula_classical(working, disc_info["center_xy"], disc_info["diameter_px"])

    return {
        "disc_mask": cdr_info["disc_mask"],
        "cup_mask": cdr_info["cup_mask"],
        "vertical_cdr": cdr_info["vertical_cdr"],
        "disc_diameter_px": cdr_info["disc_diameter_px"],
        "cup_diameter_px": cdr_info["cup_diameter_px"],
        "macula_location": macula_info["location_xy"],
        "disc_found": disc_info["found"],
        "macula_found": macula_info["found"],
    }


@functools.lru_cache(maxsize=1)
def _cached_model(weights_path: str, device: str) -> torch.nn.Module:
    # Downstream callers (report generation, the app) call
    # compute_optic_biomarkers_auto() once per uploaded image -- cache the
    # loaded model so repeated calls with the same (weights_path, device)
    # don't re-read the checkpoint off disk and rebuild the network every
    # time. Same pattern as vessel_infer._cached_model().
    return load_optic_disc_model(weights_path, device)


def compute_optic_biomarkers_auto(image: np.ndarray, weights_path: str = DEFAULT_WEIGHTS_PATH, device: str = "cpu") -> dict:
    """compute_optic_biomarkers_hybrid() if a trained checkpoint exists at
    weights_path, otherwise optic_disc.compute_optic_biomarkers()'s
    classical fallback. Same checkpoint-exists/fallback shape as
    vessel_infer.compute_biomarkers_auto() -- this is what downstream
    stages (report generation, the app) should call.
    """
    if os.path.exists(weights_path):
        model = _cached_model(weights_path, device)
        return compute_optic_biomarkers_hybrid(image, model, device)
    return optic_disc.compute_optic_biomarkers(image)
