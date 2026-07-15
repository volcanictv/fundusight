"""Optic-nerve-head cropping for the glaucoma classifier.

Glaucoma is defined at the optic disc, but the full-image classifier learned to
shortcut on edge artifacts and hemorrhages instead (domain-expert review; full
story in DEEP_DIVE.md). Cropping to the ONH before classifying removes that
option. Reuses Phase 6's Stage 6.1 localizer + ROI crop rather than inventing a
second notion of "where the disc is".

This is the SINGLE shared crop definition, imported by both glaucoma_dataset.py
(training) and glaucoma_infer.py (inference) -- if the two diverge, the model is
evaluated on inputs it never trained on and the metrics lie. The crop is
deliberately 3 disc diameters wide (optic_disc._DISC_ROI_CROP_MULTIPLE), not tight
to the rim: peripapillary RNFL defects and disc hemorrhages just outside the rim
are real glaucoma signs.
"""

import numpy as np

from src.segmentation import optic_disc, vessels


def crop_to_onh(image: np.ndarray) -> tuple[np.ndarray, dict]:
    """Crop a BGR fundus photo (cv2.imread convention) to its optic nerve head.

    Returns (crop_bgr, disc_info), where `crop_bgr` is a square
    DISC_ROI_WIDTH x DISC_ROI_WIDTH BGR image and `disc_info` is
    locate_disc_classical()'s dict -- including "confident", so a caller can
    tell whether the crop is trustworthy (see optic_disc.
    assess_disc_plausibility: on ADAM's ground truth, ~14% of localizations
    land outside the true disc, and the geometric checks catch all of them).

    `disc_info` additionally carries "bbox" (crop_disc_roi()'s bounds in
    working-image coordinates) and "working_shape", which together let a caller
    project something else -- a ground-truth mask, say -- into the same crop
    space the model sees, without re-deriving the crop and risking measuring a
    slightly different one than was actually used.

    On a degenerate image the localizer still returns a fallback center and
    diameter rather than failing, so this always returns a usable square crop
    -- a training pipeline must not raise on one bad image, and an inference
    path must not crash on one bad upload. `disc_info["confident"]` is how a
    caller distinguishes "cropped the disc" from "cropped something".
    """
    working = vessels._resize_to_working_width(image)
    disc_info = optic_disc.locate_disc_classical(working)
    crop, bbox = optic_disc.crop_disc_roi(working, disc_info["center_xy"], disc_info["diameter_px"])
    return crop, {**disc_info, "bbox": bbox, "working_shape": working.shape[:2]}
