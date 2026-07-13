"""Evaluate the optic disc/cup pipeline on RIGA — a true EXTERNAL test set.

Every disc/cup number in this repo until now came from REFUGE2, the dataset the
U-Net was trained on. RIGA was never trained on, comes from different cameras and
clinics, and — crucially — carries its own six-annotator consensus label, so it
gives two things no REFUGE2 number can:

1. **A cross-dataset CDR error.** Does the pipeline still measure a sensible
   cup-to-disc ratio on photographs from an unseen source?
2. **A reference for what "good" even means.** RIGA's six ophthalmologists
   disagree with each other on CDR by a mean of 0.166 (see
   scripts/validate_riga_extraction.py). A model error is only interpretable
   against that spread. An error well inside it is not a defect to be optimised
   away — it is a model performing within the ambiguity of its own ground truth.

This runs the REAL pipeline end to end (compute_optic_biomarkers_auto: Stage 6.1
localization + the vascular convergence prior + Stage 6.0 arbitration + the Stage
6.2 U-Net), NOT the U-Net on ground-truth crops. Localization error is therefore
included, which is the point — it is what a deployed instance actually does.

Requires the RIGA mask cache. Build it first:
    .venv\\Scripts\\python.exe -c "from src.segmentation.riga_dataset import build_riga_mask_cache; print(build_riga_mask_cache('data','data/riga_masks'))"

Run with:
    .venv\\Scripts\\python.exe scripts\\evaluate_on_riga.py
"""

import argparse
import os
import sys

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from src.segmentation import vessels
from src.segmentation.optic_disc import _vertical_extent
from src.segmentation.optic_disc_dataset import _remap_mask_to_class_indices
from src.segmentation.optic_disc_infer import compute_optic_biomarkers_auto
from src.segmentation.riga_dataset import build_riga_pairs

# Measured on RIGA itself (scripts/validate_riga_extraction.py) -- the mean
# spread in vertical CDR between the six ophthalmologists grading the SAME
# photograph. Quoted here so the model's error is never reported without the
# yardstick that makes it meaningful.
HUMAN_CDR_DISAGREEMENT = 0.1662


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    total = a.sum() + b.sum()
    return float(2.0 * (a & b).sum() / total) if total else 1.0


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--riga-root", default=os.path.join(PROJECT_ROOT, "data"))
    parser.add_argument("--cache-root", default=os.path.join(PROJECT_ROOT, "data", "riga_masks"))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    pairs = build_riga_pairs(args.riga_root, args.cache_root)
    if not pairs:
        sys.exit(f"No RIGA masks in {args.cache_root} -- build the cache first (see this script's docstring).")
    if args.limit:
        pairs = pairs[:: max(len(pairs) // args.limit, 1)][: args.limit]

    print(f"Evaluating the FULL pipeline on {len(pairs)} RIGA images (never trained on).\n")

    rim_dice, cup_dice, cdr_err, confident_cdr_err = [], [], [], []
    pred_cdrs, true_cdrs = [], []
    per_source: dict[str, list] = {}
    n_confident = 0

    for image_path, mask_path, source in tqdm(pairs, desc="RIGA"):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if image is None or raw is None:
            continue

        result = compute_optic_biomarkers_auto(image)
        pred_disc, pred_cup = result["disc_mask"], result["cup_mask"]

        # Ground truth is full-frame; the pipeline returns masks at working
        # resolution. Resize GT per-image with its OWN dimensions (RIGA subsets
        # differ in size, so any fixed scale factor would silently corrupt part
        # of the comparison) -- same convention as the ADAM eval scripts.
        classes = _remap_mask_to_class_indices(raw)
        h, w = pred_disc.shape[:2]
        classes = cv2.resize(classes.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        true_disc, true_cup = classes != 0, classes == 2
        if not true_disc.any() or not true_cup.any():
            continue

        rim_dice.append(_dice(pred_disc & ~pred_cup, true_disc & ~true_cup))
        cup_dice.append(_dice(pred_cup, true_cup))

        td = _vertical_extent(true_disc)
        true_cdr = _vertical_extent(true_cup) / td if td else 0.0
        pred_cdr = result["vertical_cdr"]
        error = abs(pred_cdr - true_cdr)

        cdr_err.append(error)
        pred_cdrs.append(pred_cdr)
        true_cdrs.append(true_cdr)
        per_source.setdefault(source, []).append(error)
        if result["disc_confident"]:
            n_confident += 1
            confident_cdr_err.append(error)

    n = len(cdr_err)
    err = np.array(cdr_err)
    conf_err = np.array(confident_cdr_err)
    pred_cdrs, true_cdrs = np.array(pred_cdrs), np.array(true_cdrs)

    print(f"\n{'=' * 74}")
    print(f"CROSS-DATASET RESULT — {n} RIGA images, model never trained on any of them")
    print(f"{'=' * 74}")
    print(f"  dice_rim : {np.mean(rim_dice):.4f}")
    print(f"  dice_cup : {np.mean(cup_dice):.4f}")
    print()
    print(f"  mean predicted CDR    : {pred_cdrs.mean():.4f}")
    print(f"  mean ground-truth CDR : {true_cdrs.mean():.4f}   (bias {pred_cdrs.mean() - true_cdrs.mean():+.4f})")
    print(f"  mean |CDR error|      : {err.mean():.4f}")
    print(f"  median |CDR error|    : {np.median(err):.4f}")
    print()
    print(f"  localization confident on {n_confident}/{n} ({n_confident / n:.1%})")
    if conf_err.size:
        print(f"  mean |CDR error| WHERE CONFIDENT : {conf_err.mean():.4f}  (median {np.median(conf_err):.4f})")
        print("     ^ the number a user actually sees -- a low-confidence CDR is suppressed in the report")

    print(f"\n{'=' * 74}")
    print("PUT IT IN CONTEXT")
    print(f"{'=' * 74}")
    print(f"  Six ophthalmologists disagree with EACH OTHER on CDR by : {HUMAN_CDR_DISAGREEMENT:.4f}")
    print(f"  This model disagrees with their consensus by            : {err.mean():.4f}")
    ratio = HUMAN_CDR_DISAGREEMENT / err.mean() if err.mean() else float("inf")
    print(f"  -> the model is {ratio:.1f}x closer to the consensus than the humans are to each other")
    if err.mean() < HUMAN_CDR_DISAGREEMENT:
        print("\n  The model's error is INSIDE the human noise floor. There is no CDR")
        print("  accuracy problem left to solve; the ground truth itself is this fuzzy.")

    print("\n  Per source (mean |CDR error|):")
    for source, errors in sorted(per_source.items()):
        print(f"    {source:<20} n={len(errors):<4} {np.mean(errors):.4f}")


if __name__ == "__main__":
    main()
