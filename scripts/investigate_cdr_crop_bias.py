"""One-off diagnostic: correlate CDR prediction error against Stage 6.1's
ROI-crop tightness/centering across the REFUGE2 held-out test set, to test
the hypothesis (from evaluate_optic_disc_full_pipeline.py's result --
mean predicted CDR 0.68 vs. ground-truth 0.47) that a mis-centered or
too-tight classical crop is what's driving the systematic CDR
over-estimation: if part of the true disc falls outside the ROI crop
entirely, Stage 6.2 never sees it, shrinking the measured disc diameter
(the CDR denominator) while the cup estimate stays roughly right.

Investigation only -- doesn't change any pipeline code. Run with:

    .venv\\Scripts\\python.exe scripts\\investigate_cdr_crop_bias.py
"""

import os
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from src.segmentation import optic_disc, vessels
from src.segmentation.optic_disc_dataset import _disc_bbox_from_mask, _remap_mask_to_class_indices, build_pairs
from src.segmentation.optic_disc_infer import DEFAULT_WEIGHTS_PATH, compute_optic_biomarkers_hybrid, load_optic_disc_model

REFUGE_ROOT = os.path.join(PROJECT_ROOT, "REFUGE2")


def _ground_truth_working_mask(image: np.ndarray, mask_path: str) -> np.ndarray:
    """Same working-resolution class-index remap as
    evaluate_optic_disc_full_pipeline.py's _ground_truth_working_masks(),
    just returning the raw class-index array so both the disc/cup boolean
    masks AND _disc_bbox_from_mask() (which expects class indices, not a
    boolean mask) can be derived from the one array.
    """
    mask_raw = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask_raw.ndim == 3:
        mask_raw = cv2.cvtColor(mask_raw, cv2.COLOR_BGR2GRAY)
    class_idx = _remap_mask_to_class_indices(mask_raw).astype(np.uint8)

    h, w = image.shape[:2]
    scale = vessels.VESSEL_WORKING_WIDTH / w
    return cv2.resize(class_idx, (vessels.VESSEL_WORKING_WIDTH, round(h * scale)), interpolation=cv2.INTER_NEAREST)


def _disc_clipped_fraction(gt_disc_mask: np.ndarray, bbox_meta: dict) -> float:
    """Fraction of the TRUE disc's pixels that fall outside the predicted
    ROI crop's bounding box -- the direct test of "did the crop cut off
    part of the real disc before the network ever saw it."
    """
    total = int(gt_disc_mask.sum())
    if total == 0:
        return 0.0
    ys, xs = np.nonzero(gt_disc_mask)
    inside = (xs >= bbox_meta["x0"]) & (xs < bbox_meta["x1"]) & (ys >= bbox_meta["y0"]) & (ys < bbox_meta["y1"])
    return 1.0 - (int(inside.sum()) / total)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_optic_disc_model(DEFAULT_WEIGHTS_PATH, device=str(device))
    test_pairs = build_pairs(REFUGE_ROOT)["test"]
    print(f"Analyzing Stage 6.1 crop quality vs. CDR error on {len(test_pairs)} held-out test images...")

    centering_errors_rel, diameter_ratios, clipped_fractions = [], [], []
    signed_cdr_errors, abs_cdr_errors = [], []

    for image_path, mask_path in tqdm(test_pairs):
        image = cv2.imread(image_path)
        working = vessels._resize_to_working_width(image)

        gt_working_mask = _ground_truth_working_mask(image, mask_path)
        gt_disc_mask = gt_working_mask != 0
        gt_cup_mask = gt_working_mask == 2
        gt_bbox = _disc_bbox_from_mask(gt_working_mask)

        disc_info = optic_disc.locate_disc_classical(working)
        pred_center, pred_diameter = disc_info["center_xy"], disc_info["diameter_px"]

        centering_error = float(np.hypot(pred_center[0] - gt_bbox["center_xy"][0], pred_center[1] - gt_bbox["center_xy"][1]))
        centering_errors_rel.append(centering_error / gt_bbox["diameter_px"] if gt_bbox["diameter_px"] > 0 else 0.0)
        diameter_ratios.append(pred_diameter / gt_bbox["diameter_px"] if gt_bbox["diameter_px"] > 0 else 1.0)

        _, bbox_meta = optic_disc.crop_disc_roi(working, pred_center, pred_diameter)
        clipped_fractions.append(_disc_clipped_fraction(gt_disc_mask, bbox_meta))

        result = compute_optic_biomarkers_hybrid(image, model, device=str(device))
        gt_cdr = optic_disc.compute_cdr(gt_disc_mask, gt_cup_mask)["vertical_cdr"]
        signed_error = result["vertical_cdr"] - gt_cdr
        signed_cdr_errors.append(signed_error)
        abs_cdr_errors.append(abs(signed_error))

    centering_errors_rel = np.array(centering_errors_rel)
    diameter_ratios = np.array(diameter_ratios)
    clipped_fractions = np.array(clipped_fractions)
    signed_cdr_errors = np.array(signed_cdr_errors)
    abs_cdr_errors = np.array(abs_cdr_errors)

    print(f"\n=== Correlation of CDR error against Stage 6.1 crop quality ({len(test_pairs)} images) ===")
    print(f"corr(|CDR error|, relative centering error)      = {np.corrcoef(abs_cdr_errors, centering_errors_rel)[0, 1]:.3f}")
    print(f"corr(signed CDR error, relative centering error) = {np.corrcoef(signed_cdr_errors, centering_errors_rel)[0, 1]:.3f}")
    print(f"corr(|CDR error|, disc-clipped fraction)          = {np.corrcoef(abs_cdr_errors, clipped_fractions)[0, 1]:.3f}")
    print(f"corr(signed CDR error, disc-clipped fraction)     = {np.corrcoef(signed_cdr_errors, clipped_fractions)[0, 1]:.3f}")
    print(f"corr(signed CDR error, diameter ratio pred/true)  = {np.corrcoef(signed_cdr_errors, diameter_ratios)[0, 1]:.3f}")

    clipped = clipped_fractions > 0
    print(f"\nImages with any part of the true disc clipped by the ROI crop: {clipped.sum()} / {len(test_pairs)} ({100 * clipped.mean():.1f}%)")
    if clipped.any():
        print(f"  mean signed CDR error, clipped images     = {signed_cdr_errors[clipped].mean():+.4f}")
    if (~clipped).any():
        print(f"  mean signed CDR error, non-clipped images = {signed_cdr_errors[~clipped].mean():+.4f}")

    print(f"\nmean diameter ratio (Stage 6.1 estimate / true disc diameter) = {diameter_ratios.mean():.4f}")
    print(f"mean relative centering error (as a fraction of true diameter) = {centering_errors_rel.mean():.4f}")


if __name__ == "__main__":
    main()
