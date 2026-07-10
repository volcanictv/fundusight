"""One-off calibration: investigate_cdr_shape_vs_area_bias.py found genuine
AREA over-segmentation (predicted disc ~1.48x, predicted cup ~2.98x the
ground-truth area), not just a bounding-box shape artifact -- the model's
raw argmax decision is including too many borderline pixels in the
disc-rim/cup classes. That's a calibration problem, not necessarily a "the
model learned nothing useful" problem, so a NO-RETRAIN fix is worth trying:
replace plain argmax (implicitly "whichever class has the highest softmax
probability wins") with independent probability thresholds per class --
disc = P(not background) > disc_threshold, cup = P(cup) > cup_threshold --
searched to shrink the over-segmented regions back toward the true
boundary.

FIRST ATTEMPT (superseded): calibrated on ground-truth ROI crops (Stage
6.2 alone, for speed) and looked like a clear win there (mean |CDR error|
0.117 -> 0.101, both Dice scores improved). Confirming via
evaluate_optic_disc_full_pipeline.py on the real held-out test pipeline
(Stage 6.1's classical crop included) showed a REGRESSION instead (mean
|CDR error| 0.241 -> 0.289) -- the calibration didn't transfer. Root
cause: Stage 6.1's real crop tends to be TIGHTER than a ground-truth crop
(investigate_cdr_crop_bias.py found Stage 6.1's diameter estimate runs
~10% under the true diameter on average), so the disc already appears
more zoomed-in within a real ROI than within a ground-truth one --
thresholds tuned to shrink an appropriately-sized crop over-shrink an
already-tight one.

THIS VERSION: calibrates against the REAL Stage 6.1 + Stage 6.2 pipeline
on the VALIDATION split (never test) instead of ground-truth crops, so
whatever thresholds are found already account for Stage 6.1's crop-size
tendency. Still must be CONFIRMED on the held-out test split via
evaluate_optic_disc_full_pipeline.py before being trusted -- this script
only searches. Run with:

    .venv\\Scripts\\python.exe scripts\\calibrate_optic_disc_thresholds.py
"""

import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from src.segmentation import optic_disc, vessels
from src.segmentation.optic_disc_dataset import build_pairs
from src.segmentation.optic_disc_infer import DEFAULT_WEIGHTS_PATH, load_optic_disc_model
from scripts.evaluate_optic_disc_full_pipeline import _ground_truth_working_masks, _hard_dice

REFUGE_ROOT = os.path.join(PROJECT_ROOT, "REFUGE2")

DISC_THRESHOLD_GRID = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
CUP_THRESHOLD_GRID = [0.02, 0.05, 0.1, 0.15, 0.2, 0.33, 0.45, 0.55, 0.65]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_optic_disc_model(DEFAULT_WEIGHTS_PATH, device=str(device))

    val_pairs = build_pairs(REFUGE_ROOT)["val"]
    print(f"Running the REAL Stage 6.1+6.2 pipeline once for {len(val_pairs)} validation images...")

    # Phase A: run the full pipeline's classical crop + model forward pass
    # once per image, caching everything needed to re-derive masks for any
    # threshold combo without repeating Stage 6.1 or the GPU forward pass.
    cached = []
    with torch.no_grad():
        for image_path, mask_path in tqdm(val_pairs, desc="forward pass"):
            image = cv2.imread(image_path)
            working = vessels._resize_to_working_width(image)
            disc_info = optic_disc.locate_disc_classical(working)
            roi_image, bbox_meta = optic_disc.crop_disc_roi(working, disc_info["center_xy"], disc_info["diameter_px"])

            input_arr = optic_disc.extract_color_features(roi_image)
            input_tensor = torch.from_numpy(input_arr).unsqueeze(0).to(device)
            probs = F.softmax(model(input_tensor), dim=1).squeeze(0).cpu().numpy()  # (3, H, W)

            gt_disc, gt_cup = _ground_truth_working_masks(image, mask_path)
            gt_cdr = optic_disc.compute_cdr(gt_disc, gt_cup)["vertical_cdr"]

            cached.append((probs, bbox_meta, working.shape, gt_disc, gt_cup, gt_cdr))

    # Phase B: grid search -- cheap, just reprojection + numpy math on the
    # cached arrays, no repeated Stage 6.1 or GPU calls.
    print(f"\nGrid-searching {len(DISC_THRESHOLD_GRID)}x{len(CUP_THRESHOLD_GRID)} threshold combinations...")
    results = []
    for disc_threshold in DISC_THRESHOLD_GRID:
        for cup_threshold in CUP_THRESHOLD_GRID:
            abs_cdr_errors, dice_rim_scores, dice_cup_scores = [], [], []
            for probs, bbox_meta, working_shape, gt_disc, gt_cup, gt_cdr in cached:
                disc_roi = (probs[1] + probs[2]) > disc_threshold
                cup_roi = probs[2] > cup_threshold
                disc_mask = optic_disc.reproject_roi_mask_to_working(disc_roi, bbox_meta, working_shape)
                cup_mask = optic_disc.reproject_roi_mask_to_working(cup_roi, bbox_meta, working_shape)

                cdr_info = optic_disc.compute_cdr(disc_mask, cup_mask)  # also cleans defensively
                abs_cdr_errors.append(abs(cdr_info["vertical_cdr"] - gt_cdr))
                dice_rim_scores.append(_hard_dice(cdr_info["disc_mask"] & ~cdr_info["cup_mask"], gt_disc & ~gt_cup))
                dice_cup_scores.append(_hard_dice(cdr_info["cup_mask"], gt_cup))

            results.append(
                {
                    "disc_threshold": disc_threshold,
                    "cup_threshold": cup_threshold,
                    "mean_abs_cdr_error": float(np.mean(abs_cdr_errors)),
                    "dice_rim": float(np.mean(dice_rim_scores)),
                    "dice_cup": float(np.mean(dice_cup_scores)),
                }
            )

    results.sort(key=lambda r: r["mean_abs_cdr_error"])
    print("\n=== Top 10 threshold combinations by validation mean |CDR error| (real Stage 6.1 pipeline) ===")
    print(f"{'disc_thr':>9} {'cup_thr':>8} {'|CDR err|':>10} {'dice_rim':>9} {'dice_cup':>9}")
    for r in results[:10]:
        print(f"{r['disc_threshold']:>9.2f} {r['cup_threshold']:>8.2f} {r['mean_abs_cdr_error']:>10.4f} {r['dice_rim']:>9.4f} {r['dice_cup']:>9.4f}")

    # Baseline for comparison: true argmax through the same cached
    # real-pipeline probabilities.
    argmax_abs_errors, argmax_dice_rim, argmax_dice_cup = [], [], []
    for probs, bbox_meta, working_shape, gt_disc, gt_cup, gt_cdr in cached:
        predicted_class = probs.argmax(axis=0)
        disc_mask = optic_disc.reproject_roi_mask_to_working(predicted_class != 0, bbox_meta, working_shape)
        cup_mask = optic_disc.reproject_roi_mask_to_working(predicted_class == 2, bbox_meta, working_shape)
        cdr_info = optic_disc.compute_cdr(disc_mask, cup_mask)
        argmax_abs_errors.append(abs(cdr_info["vertical_cdr"] - gt_cdr))
        argmax_dice_rim.append(_hard_dice(cdr_info["disc_mask"] & ~cdr_info["cup_mask"], gt_disc & ~gt_cup))
        argmax_dice_cup.append(_hard_dice(cdr_info["cup_mask"], gt_cup))

    print(f"\nBaseline (true argmax, current production behavior, real pipeline):")
    print(f"  mean |CDR error| = {np.mean(argmax_abs_errors):.4f}   dice_rim = {np.mean(argmax_dice_rim):.4f}   dice_cup = {np.mean(argmax_dice_cup):.4f}")


if __name__ == "__main__":
    main()
