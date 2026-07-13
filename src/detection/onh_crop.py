"""Phase 7 (revision): optic-nerve-head cropping for the glaucoma classifier.

Domain-expert review of the full-image glaucoma classifier found it attending
to edge artifacts and hemorrhages rather than the optic disc. That is a
plausible shortcut for it to have learned: glaucoma is diagnosed almost
entirely from the optic nerve head (cup-to-disc ratio, neuroretinal rim
thinning, RNFL defects), but the classifier was being handed an entire fundus
photo in which the disc occupies only a few percent of the pixels. Once the
image is squashed to 224x224 (see dataset.build_transforms), the disc is a
~25px blob, while a big hemorrhage or a bright edge artifact is large,
high-contrast, and -- in a dataset where glaucoma prevalence correlates with
being photographed at a particular site/camera -- potentially predictive
enough to shortcut on.

The fix is to crop to the ONH before classifying, so the model can only look
at the anatomy the disease is actually defined by. This reuses Phase 6's
existing Stage 6.1 classical localizer + ROI crop (optic_disc.
locate_disc_classical / crop_disc_roi) rather than introducing a second,
independent notion of "where the disc is".

THIS MODULE IS THE SINGLE SHARED DEFINITION of that crop, imported by both
glaucoma_dataset.py (training) and glaucoma_infer.py (inference). A crop that
differed between the two would be a silent train/inference distribution
mismatch -- the model would be evaluated on inputs it never trained on, and
the metrics would not describe the deployed behavior. Same reasoning that
keeps optic_disc.extract_color_features() shared between the disc/cup U-Net's
training and inference paths.

Note the crop is deliberately WIDER than the disc itself
(optic_disc._DISC_ROI_CROP_MULTIPLE = 3.0 disc diameters, the same ROI Stage
6.2 segments in): glaucoma's signs include peripapillary RNFL defects and
disc hemorrhages just outside the rim, so a crop tight to the disc margin
would cut away real signal. Three disc diameters keeps the disc dominant in
frame while retaining the peripapillary ring.
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
