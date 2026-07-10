"""One-off diagnostic: isolate whether Stage 6.2's segmentation model has
its own systematic CDR bias, independent of Stage 6.1's classical crop.

investigate_cdr_crop_bias.py found the CDR bias in
evaluate_optic_disc_full_pipeline.py (mean predicted CDR 0.68 vs.
ground-truth 0.47) does NOT correlate with crop centering/tightness (all
correlations <0.1, disc clipping affects only 1.5% of images). This script
tests the next hypothesis directly: run the model on ROI crops derived
from the GROUND-TRUTH disc bounding box (OpticDiscDataset's eval-mode crop
-- what the network was trained against, with zero Stage 6.1 localization
error) and see whether the CDR bias persists. If it does at a similar
magnitude, the bias lives in the segmentation model itself, not in Stage
6.1's localization/cropping.

Investigation only -- doesn't change any pipeline code. Run with:

    .venv\\Scripts\\python.exe scripts\\investigate_cdr_segmentation_bias.py
"""

import os
import sys

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from src.segmentation import optic_disc
from src.segmentation.optic_disc_dataset import DISC_ROI_WIDTH, OpticDiscDataset, build_pairs
from src.segmentation.optic_disc_infer import DEFAULT_WEIGHTS_PATH, load_optic_disc_model

REFUGE_ROOT = os.path.join(PROJECT_ROOT, "REFUGE2")


def _hard_dice(pred: np.ndarray, target: np.ndarray) -> float:
    pred_sum, target_sum = int(pred.sum()), int(target.sum())
    if pred_sum == 0 and target_sum == 0:
        return 1.0
    intersection = int(np.logical_and(pred, target).sum())
    return 2.0 * intersection / (pred_sum + target_sum)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_optic_disc_model(DEFAULT_WEIGHTS_PATH, device=str(device))

    test_pairs = build_pairs(REFUGE_ROOT)["test"]
    # train=False -> deterministic crop from the GROUND-TRUTH disc bbox
    # (_disc_bbox_from_mask), not Stage 6.1's classical localize_disc_
    # classical() -- this is precisely the "perfect localization" condition.
    dataset = OpticDiscDataset(test_pairs, roi_width=DISC_ROI_WIDTH, train=False)
    print(f"Evaluating Stage 6.2 alone (ground-truth crop, zero Stage 6.1 localization error) on {len(dataset)} images...")

    dice_rim_scores, dice_cup_scores = [], []
    pred_cdrs, gt_cdrs, signed_cdr_errors = [], [], []
    disc_diameter_deltas, cup_diameter_deltas = [], []

    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            input_tensor, target = dataset[i]
            input_tensor = input_tensor.unsqueeze(0).to(device)

            logits = model(input_tensor)
            predicted_class = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
            pred_disc, pred_cup = optic_disc.clean_disc_cup_masks(predicted_class != 0, predicted_class == 2)

            target_np = target.numpy()
            gt_disc, gt_cup = target_np != 0, target_np == 2

            dice_rim_scores.append(_hard_dice(pred_disc & ~pred_cup, gt_disc & ~gt_cup))
            dice_cup_scores.append(_hard_dice(pred_cup, gt_cup))

            pred_cdr_info = optic_disc.compute_cdr(pred_disc, pred_cup)
            gt_cdr_info = optic_disc.compute_cdr(gt_disc, gt_cup)
            pred_cdrs.append(pred_cdr_info["vertical_cdr"])
            gt_cdrs.append(gt_cdr_info["vertical_cdr"])
            signed_cdr_errors.append(pred_cdr_info["vertical_cdr"] - gt_cdr_info["vertical_cdr"])
            disc_diameter_deltas.append(pred_cdr_info["disc_diameter_px"] - gt_cdr_info["disc_diameter_px"])
            cup_diameter_deltas.append(pred_cdr_info["cup_diameter_px"] - gt_cdr_info["cup_diameter_px"])

    dice_rim_scores = np.array(dice_rim_scores)
    dice_cup_scores = np.array(dice_cup_scores)
    signed_cdr_errors = np.array(signed_cdr_errors)
    disc_diameter_deltas = np.array(disc_diameter_deltas)
    cup_diameter_deltas = np.array(cup_diameter_deltas)

    print(f"\n=== Stage 6.2 alone, ground-truth crop, {len(dataset)} images ===")
    print(f"dice_rim (disc-minus-cup)      = {dice_rim_scores.mean():.4f}")
    print(f"dice_cup                       = {dice_cup_scores.mean():.4f}")
    print(f"mean predicted vertical CDR    = {np.mean(pred_cdrs):.4f}")
    print(f"mean ground-truth vertical CDR = {np.mean(gt_cdrs):.4f}")
    print(f"mean signed CDR error (pred - gt) = {signed_cdr_errors.mean():+.4f}")
    print(f"mean |CDR error|                  = {np.abs(signed_cdr_errors).mean():.4f}")
    print(f"median |CDR error|                = {np.median(np.abs(signed_cdr_errors)):.4f}")

    print(f"\nmean disc diameter delta (pred - gt), ROI px = {disc_diameter_deltas.mean():+.2f}")
    print(f"mean cup diameter delta (pred - gt), ROI px  = {cup_diameter_deltas.mean():+.2f}")
    print(f"fraction of images where predicted cup is larger than ground truth  = {100 * (cup_diameter_deltas > 0).mean():.1f}%")
    print(f"fraction of images where predicted disc is smaller than ground truth = {100 * (disc_diameter_deltas < 0).mean():.1f}%")


if __name__ == "__main__":
    main()
