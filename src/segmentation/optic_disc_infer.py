"""Phase 6 (Stages 6.2/6.3): local inference with the trained optic
disc/cup model.

Mirrors vessel_infer.py's split from vessels.py: this is the only file in
src/segmentation/ that imports torch for the optic disc/cup *hybrid* path,
keeping optic_disc.py itself classical and torch-free (see its module
docstring) for anyone who only needs the classical baseline.
"""

import functools
import os

import cv2
import numpy as np
import torch

from src.segmentation import optic_disc, vessels
from src.segmentation.disc_locator_model import LOCATOR_INPUT_SIZE, build_disc_locator_model
from src.segmentation.optic_disc_model import build_optic_disc_model

# Matches optic_disc_train.py's --output default and scripts/demo_optic_disc.py's
# WEIGHTS_PATH -- the one place downstream callers should look for a
# trained checkpoint by default.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_WEIGHTS_PATH = os.path.join(_PROJECT_ROOT, "checkpoints", "optic_disc_unet.pth")

# Stage 6.0's coarse full-frame locator (see disc_locator_model.py). Optional:
# every function below falls back to the classical-only behaviour when this
# checkpoint is absent, so the pipeline still runs on a fresh clone.
DEFAULT_LOCATOR_WEIGHTS_PATH = os.path.join(_PROJECT_ROOT, "checkpoints", "disc_locator.pth")

# Two centers count as agreeing if they are within this many EXPECTED DISC
# DIAMETERS of each other. One diameter is the natural unit: crop_disc_roi()
# cuts a window 3 disc diameters wide, so two centers within one diameter
# produce heavily overlapping crops that both contain the disc, and the
# distinction does not matter downstream. Beyond that they are cropping
# genuinely different anatomy.
_LOCATOR_AGREEMENT_DIAMETERS = 1.0

# !!! THE OVER-SEGMENTATION THIS SECTION IS ABOUT NO LONGER EXISTS (measured
# 2026-07-14). Everything below is retained as a record of an investigation,
# NOT as a description of the current model -- read it as history.
#
# The ~1.5x/~3x over-segmentation was a property of the model trained on
# REFUGE2's OFFICIAL (three-way camera/domain) split. The pooled re-split
# retrain that took mean Dice 0.5599 -> 0.8756 also eliminated the size bias.
# Re-measured on the 180 held-out pooled test images, Stage 6.2 in isolation on
# ground-truth ROI crops:
#
#     disc area ratio (pred/GT)  mean 1.002   (was ~1.5)
#     cup  area ratio (pred/GT)  mean 1.029   (was ~3.0)
#     CDR bias (pred - GT)             -0.0000
#     CDR mean |error|                  0.0436
#
# There is nothing left for a threshold recalibration -- or for a boundary-aware
# / anti-over-segmentation loss -- to correct. The residual 0.0436 is VARIANCE,
# not bias, and it sits well inside the ~0.1-0.2 inter-observer variability that
# trained ophthalmologists show on cup-to-disc ratio: the model disagrees with
# the label by less than two humans disagree with each other. That is a label
# noise floor, and no loss function fixes label noise. Do not spend a training
# run chasing it.
#
# (Historical, for the record:) Two independent attempts at a post-hoc
# probability-threshold recalibration (replacing plain argmax) were tried here
# to counter the then-real over-segmentation:
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


def load_disc_locator_model(weights_path: str, device: str = "cpu") -> torch.nn.Module:
    """Build the Stage 6.0 coarse locator and load its trained weights."""
    model = build_disc_locator_model()
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_disc_bbox(working_image: np.ndarray, model: torch.nn.Module, device: str = "cpu") -> dict:
    """Run the coarse full-frame locator on a WHOLE fundus frame and return
    its disc estimate in working-image pixel coordinates:
    `{"center_xy": (x, y), "diameter_px": float}`.

    Note this takes the FULL frame, never an ONH crop -- that is the entire
    reason this model exists as a separate network rather than as a head on
    OpticDiscUNet (see disc_locator_model.py's docstring). Feeding it a crop
    would be the out-of-distribution mistake it was built to avoid.
    """
    h, w = working_image.shape[:2]
    resized = cv2.resize(working_image, (LOCATOR_INPUT_SIZE, LOCATOR_INPUT_SIZE), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(np.transpose(rgb, (2, 0, 1))).unsqueeze(0).to(device)

    cx, cy, bw, bh = model(tensor).squeeze(0).cpu().numpy().astype(float)

    return {
        "center_xy": (cx * w, cy * h),
        # The network predicts a box; the pipeline downstream wants a single
        # diameter. max() of the two sides, matching how
        # optic_disc_dataset._disc_bbox_from_mask() collapses a box to a
        # diameter -- so a slightly elliptical prediction still yields a crop
        # that contains the whole disc rather than clipping its long axis.
        "diameter_px": float(max(bw * w, bh * h)),
    }


def _fov_center(working_image: np.ndarray) -> tuple:
    """Centroid of the retina's field of view -- the safe fallback ROI center
    for when every localizer has failed.

    Deliberately NOT the raw image center and NOT whatever coordinate a failed
    localizer last produced: a failed brightness search characteristically
    lands on a frame-edge vignette or a corner artifact, and cropping there
    hands the downstream U-Net (and the Grad-CAM that explains it) a window of
    pure canvas/border, which is where the hallucinated border attention comes
    from. The FOV centroid is guaranteed to be inside the retina, so a crop
    around it is at worst uninformative rather than actively misleading.
    """
    green = vessels.extract_vessel_channel(working_image)
    fov = vessels._fov_mask(green)
    h, w = green.shape[:2]
    if not fov.any():
        return (w / 2.0, h / 2.0)
    ys, xs = np.nonzero(fov)
    return (float(xs.mean()), float(ys.mean()))


def locate_disc_arbitrated(
    working_image: np.ndarray, locator_model: torch.nn.Module | None = None, device: str = "cpu"
) -> dict:
    """Stage 6.1 with a structural fail-safe: the classical
    brightness+convergence search, arbitrated against the Stage 6.0 coarse
    full-frame locator, with a guaranteed-safe fallback if both fail.

    The arbitration policy is asymmetric, and that asymmetry is the whole
    design -- it is set by what the two components are actually measured to be
    good at (ADAM, 270 ground-truth discs; see
    scripts/evaluate_disc_localization.py):

    1. **Classical says confident -> classical wins, unconditionally.** When
       the geometric plausibility checks pass, the classical center was correct
       on 199/199 of ADAM's confident cases -- zero wrong crops slipped
       through. It is also pixel-accurate, where the coarse locator is a
       256px-input estimate and inherently blurrier. A coarse model is not
       allowed to overrule a signal with a perfect observed precision, so it is
       not even consulted here (which also keeps the common, healthy case as
       fast as it was before).

    2. **Classical says NOT confident -> the coarse locator gets to speak.**
       This is the population that contains every wrong crop (16 of them) plus
       55 needlessly-flagged good ones. The locator's center is re-measured
       with the same geometric checks; if the shape under IT is plausible, the
       crop is recovered and reported confident, tagged source="coarse_locator".

    3. **Both fail -> keep the unverified center if it is inside the retina;
       fall back to the FOV centroid only if it is not.** Low confidence is NOT
       the same as unusable: the guard over-flags deliberately (~20% of correct
       localizations get flagged), so the low-confidence pool is mostly good
       crops. Replacing all of them with the FOV centroid was measured to
       destroy 44 correct centers on ADAM to rescue 10 -- so the fallback now
       fires only on the failure it exists to prevent, namely a crop of frame
       edge / black canvas (the thing that makes downstream Grad-CAMs light up
       on image borders). A center inside the retina is kept, unverified, and
       reported `confident=False`.

    Returns locate_disc_classical()'s dict, plus:
      "source": "classical" | "coarse_locator" | "safe_fallback"
      "coarse_center_xy": the locator's center, or None if it wasn't consulted
      "locator_agrees": bool | None -- whether the two centers agreed

    With `locator_model=None` (no checkpoint on disk) this reduces exactly to
    locate_disc_classical() + the safe fallback, so the pipeline still works on
    a fresh clone with no Stage 6.0 weights.
    """
    classical = optic_disc.locate_disc_classical(working_image)
    result = {**classical, "source": "classical", "coarse_center_xy": None, "locator_agrees": None}

    if classical["confident"]:
        return result

    if locator_model is None:
        # No arbitration available. Still refuse to hand downstream a crop
        # centered on whatever the failed search landed on -- if it didn't even
        # find a candidate, fall back to the FOV centroid.
        if not classical["found"]:
            return {**result, "center_xy": _fov_center(working_image), "source": "safe_fallback"}
        return result

    coarse = predict_disc_bbox(working_image, locator_model, device)
    result["coarse_center_xy"] = coarse["center_xy"]

    w = working_image.shape[1]
    expected_diameter = w * optic_disc._EXPECTED_DISC_DIAMETER_FRACTION
    distance = float(np.hypot(*(np.subtract(coarse["center_xy"], classical["center_xy"]))))
    result["locator_agrees"] = bool(distance <= _LOCATOR_AGREEMENT_DIAMETERS * expected_diameter)

    # Re-measure the geometry UNDER the locator's proposed center. Accepting the
    # locator's coordinate on faith would defeat the purpose: it would replace a
    # known-unreliable estimate with an unverified one, which is how a caught
    # failure becomes a silent failure again. It has to clear the same bar.
    green = vessels.extract_vessel_channel(working_image)
    fov = vessels._fov_mask(green)
    geometry = optic_disc._disc_candidate_geometry(green, fov, coarse["center_xy"], expected_diameter)
    plausibility = optic_disc.assess_disc_plausibility(geometry, expected_diameter, image_width=w)

    if plausibility["plausible"]:
        diameter = geometry["diameter_px"] if geometry["measured"] else coarse["diameter_px"]
        return {
            **result,
            "center_xy": coarse["center_xy"],
            "diameter_px": float(diameter),
            "found": True,
            "confident": True,
            "circularity": geometry["circularity"],
            "solidity": geometry["solidity"],
            "diameter_fraction": geometry["diameter_px"] / w if geometry["measured"] else float("nan"),
            "implausible_reasons": [],
            "source": "coarse_locator",
        }

    # Neither candidate's shape holds up. The center stays UNVERIFIED -- but
    # unverified is not the same as unusable, and the difference is worth 44
    # images on ADAM.
    #
    # The first version of this branch replaced the center with the FOV centroid
    # whenever confidence was low. That was wrong, and measurably so: the
    # plausibility guard over-flags on purpose (~20% of CORRECT localizations
    # are flagged), so "low confidence" is dominated by good crops, not bad
    # ones. Overwriting them all with the middle of the retina cut localization
    # accuracy 254 -> 210 while rescuing only 10 -- actively destroying 44
    # correct centers to guard against a failure most of them did not have.
    # It also leaks: crop_to_onh() (the GLAUCOMA input) uses this center and
    # does NOT consult `confident`, so those 61 images would have been
    # classified on a crop of the central retina.
    #
    # So the fallback now fires on the thing it was actually meant to prevent --
    # a crop of frame edge / black canvas, which is what makes downstream
    # Grad-CAMs hallucinate on borders -- rather than on mere lack of
    # confidence. A center that is inside the retina is kept as the best
    # available estimate and simply reported unconfident; a center that is
    # outside the FOV (or was never found at all) is replaced.
    green = vessels.extract_vessel_channel(working_image)
    fov = vessels._fov_mask(green)
    h, w_img = fov.shape[:2]
    cx, cy = int(round(classical["center_xy"][0])), int(round(classical["center_xy"][1]))
    center_in_retina = classical["found"] and 0 <= cx < w_img and 0 <= cy < h and bool(fov[cy, cx])

    if center_in_retina:
        return {
            **result,
            "confident": False,
            "implausible_reasons": classical["implausible_reasons"] + ["coarse locator did not verify it"],
            "source": "classical",
        }

    return {
        **result,
        "center_xy": _fov_center(working_image),
        "diameter_px": float(expected_diameter),
        "confident": False,
        "implausible_reasons": classical["implausible_reasons"] + ["candidate fell outside the field of view"],
        "source": "safe_fallback",
    }


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


def compute_optic_biomarkers_hybrid(
    image: np.ndarray, model: torch.nn.Module, device: str = "cpu", locator_model: torch.nn.Module | None = None
) -> dict:
    """Same structure and return contract as
    optic_disc.compute_optic_biomarkers(), swapping the classical
    intensity-threshold disc/cup mask for the trained model's mask, and
    routing localization through locate_disc_arbitrated() so a low-confidence
    classical crop can be rescued by the Stage 6.0 coarse locator (or, failing
    that, degrade to a safe in-retina ROI) instead of being handed to the
    U-Net as-is.

    `locator_model=None` reduces this to the previous classical-localization
    behaviour, so the absence of a Stage 6.0 checkpoint is never fatal.
    """
    working = vessels._resize_to_working_width(image)
    disc_info = locate_disc_arbitrated(working, locator_model, device)
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
        "disc_confident": disc_info["confident"],
        "disc_localization_warnings": disc_info["implausible_reasons"],
        "macula_found": macula_info["found"],
        # Which stage actually produced the crop the CDR above was measured
        # from -- "classical", "coarse_locator" (the classical search failed
        # and Stage 6.0 recovered it), or "safe_fallback" (both failed; the
        # CDR is not meaningful and disc_confident is False).
        "disc_localization_source": disc_info["source"],
    }


@functools.lru_cache(maxsize=1)
def _cached_model(weights_path: str, device: str) -> torch.nn.Module:
    # Downstream callers (report generation, the app) call
    # compute_optic_biomarkers_auto() once per uploaded image -- cache the
    # loaded model so repeated calls with the same (weights_path, device)
    # don't re-read the checkpoint off disk and rebuild the network every
    # time. Same pattern as vessel_infer._cached_model().
    return load_optic_disc_model(weights_path, device)


@functools.lru_cache(maxsize=1)
def _cached_locator_model(weights_path: str, device: str) -> torch.nn.Module:
    return load_disc_locator_model(weights_path, device)


def compute_optic_biomarkers_auto(
    image: np.ndarray,
    weights_path: str = DEFAULT_WEIGHTS_PATH,
    device: str = "cpu",
    locator_weights_path: str = DEFAULT_LOCATOR_WEIGHTS_PATH,
) -> dict:
    """compute_optic_biomarkers_hybrid() if a trained checkpoint exists at
    weights_path, otherwise optic_disc.compute_optic_biomarkers()'s
    classical fallback. Same checkpoint-exists/fallback shape as
    vessel_infer.compute_biomarkers_auto() -- this is what downstream
    stages (report generation, the app) should call.

    The Stage 6.0 coarse locator is wired in the same optional way: present ->
    it arbitrates low-confidence localizations; absent -> the pipeline behaves
    exactly as it did before Stage 6.0 existed. Two independent checkpoints,
    two independent graceful degradations, no caller-side fallback logic.
    """
    if os.path.exists(weights_path):
        model = _cached_model(weights_path, device)
        locator = _cached_locator_model(locator_weights_path, device) if os.path.exists(locator_weights_path) else None
        return compute_optic_biomarkers_hybrid(image, model, device, locator)
    return optic_disc.compute_optic_biomarkers(image)
