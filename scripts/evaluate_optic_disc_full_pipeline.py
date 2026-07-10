"""One-off diagnostic: evaluate the FULL optic disc/cup inference pipeline
(Stage 6.1 classical ONH localization -> Stage 6.2 hybrid segmentation ->
clean_disc_cup_masks() post-processing -> Stage 6.3 CDR) against REFUGE2's
held-out test split.

This is a different (harder, more realistic) number than
optic_disc_train.py's evaluate(): that measures the network alone on
ground-truth-derived ROI crops (perfect localization, no post-processing).
This script instead runs the same code path a real deployment would --
including Stage 6.1's classical localizer, which has its own error, and the
largest-connected-component/cup-clipping cleanup added after the demo
script surfaced stray-fragment and cup-drift artifacts. Run with:

    .venv\\Scripts\\python.exe scripts\\evaluate_optic_disc_full_pipeline.py

Prints per-class Dice and CDR agreement statistics; doesn't write any files.
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
from src.segmentation.optic_disc_dataset import _remap_mask_to_class_indices, build_pairs
from src.segmentation.optic_disc_infer import DEFAULT_WEIGHTS_PATH, compute_optic_biomarkers_hybrid, load_optic_disc_model

REFUGE_ROOT = os.path.join(PROJECT_ROOT, "REFUGE2")


def _hard_dice(pred: np.ndarray, target: np.ndarray) -> float:
    pred_sum, target_sum = int(pred.sum()), int(target.sum())
    if pred_sum == 0 and target_sum == 0:
        return 1.0  # both correctly empty -- perfect agreement, not undefined
    intersection = int(np.logical_and(pred, target).sum())
    return 2.0 * intersection / (pred_sum + target_sum)


def _ground_truth_working_masks(image: np.ndarray, mask_path: str) -> tuple:
    """Ground-truth disc/cup masks in the SAME working-image coordinate
    space compute_optic_biomarkers_hybrid()'s output is in, so Dice is
    comparing pixel-for-pixel-aligned masks. Mirrors
    vessels._resize_to_working_width()'s resize exactly, but with
    INTER_NEAREST (this is a label mask, not a photo -- no interpolation
    between class indices).
    """
    mask_raw = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask_raw.ndim == 3:
        mask_raw = cv2.cvtColor(mask_raw, cv2.COLOR_BGR2GRAY)
    class_idx = _remap_mask_to_class_indices(mask_raw).astype(np.uint8)

    h, w = image.shape[:2]
    scale = vessels.VESSEL_WORKING_WIDTH / w
    working_class_idx = cv2.resize(
        class_idx, (vessels.VESSEL_WORKING_WIDTH, round(h * scale)), interpolation=cv2.INTER_NEAREST
    )
    gt_disc = working_class_idx != 0  # rim (1) union cup (2)
    gt_cup = working_class_idx == 2
    return gt_disc, gt_cup


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = load_optic_disc_model(DEFAULT_WEIGHTS_PATH, device=str(device))

    test_pairs = build_pairs(REFUGE_ROOT)["test"]
    print(f"Evaluating full pipeline on {len(test_pairs)} held-out REFUGE2 test images...")

    dice_rim_scores, dice_cup_scores = [], []
    pred_cdrs, gt_cdrs, cdr_abs_errors = [], [], []

    for image_path, mask_path in tqdm(test_pairs):
        image = cv2.imread(image_path)
        result = compute_optic_biomarkers_hybrid(image, model, device=str(device))

        gt_disc, gt_cup = _ground_truth_working_masks(image, mask_path)
        gt_cdr_info = optic_disc.compute_cdr(gt_disc, gt_cup)

        # "rim" = disc-minus-cup, matching class index 1's meaning in
        # optic_disc_train.py's per-class Dice -- keeps this number directly
        # comparable to the ground-truth-crop-based training Dice.
        pred_rim = result["disc_mask"] & ~result["cup_mask"]
        gt_rim = gt_disc & ~gt_cup

        dice_rim_scores.append(_hard_dice(pred_rim, gt_rim))
        dice_cup_scores.append(_hard_dice(result["cup_mask"], gt_cup))
        pred_cdrs.append(result["vertical_cdr"])
        gt_cdrs.append(gt_cdr_info["vertical_cdr"])
        cdr_abs_errors.append(abs(result["vertical_cdr"] - gt_cdr_info["vertical_cdr"]))

    print(f"\nFull-pipeline held-out test results ({len(test_pairs)} images):")
    print(f"  dice_rim (disc-minus-cup)  = {np.mean(dice_rim_scores):.4f}")
    print(f"  dice_cup                   = {np.mean(dice_cup_scores):.4f}")
    print(f"  mean predicted vertical CDR = {np.mean(pred_cdrs):.4f}")
    print(f"  mean ground-truth vertical CDR = {np.mean(gt_cdrs):.4f}")
    print(f"  mean absolute CDR error     = {np.mean(cdr_abs_errors):.4f}")
    print(f"  median absolute CDR error   = {np.median(cdr_abs_errors):.4f}")


if __name__ == "__main__":
    main()
