"""Phase 6: validate Stage 6.1's classical disc localizer against real
ground truth, and calibrate its geometric plausibility checks.

Stage 6.1 (`locate_disc_classical()`) finds the optic disc as the brightest
disc-sized compact patch in the field of view. That answers "where is the
brightest disc-sized patch", which is a strictly weaker question than "where
is the optic disc" -- a large hemorrhage or a dense exudate cluster can win
the brightness search outright. Until now nothing downstream could tell the
difference: a wrong crop would feed Stage 6.2, which would dutifully segment
something disc-shaped out of it, and Stage 6.3 would report a
confident-looking CDR computed from the wrong anatomy.

ADAM ships 400 ground-truth optic disc masks (`Disc_Masks/*.bmp`, one per
Training400 image, 89 AMD / 311 Non-AMD), so this is the first real check of
how often Stage 6.1 actually lands on the disc -- and, critically, the AMD
subset is exactly the pathological population (hemorrhages, exudate) where
the brightness search is expected to fail, so the two subsets can be
compared directly.

This script measures two things:

1. **Localization accuracy** -- is the predicted center actually inside the
   ground-truth disc? Reported overall and split AMD vs Non-AMD.
2. **How well the geometric plausibility checks
   (`assess_disc_plausibility()`) separate the hits from the misses** -- i.e.
   whether "confident=False" is a useful signal or just noise. Printed as a
   confusion matrix of confident-vs-correct, plus the per-metric
   (circularity / solidity / size) distributions for hits and misses, which
   is what the thresholds in optic_disc.py were calibrated from.

The point of (2) is that a plausibility check is only worth having if it
fires on the bad crops and stays quiet on the good ones. A check that
rejects everything is as useless as one that rejects nothing.

Ground-truth masks are compared in VESSEL_WORKING_WIDTH working-image space
(the space locate_disc_classical() returns coordinates in), resizing each
mask with INTER_NEAREST from its own native resolution -- ADAM ships two
(2056x2124 and 1444x1444), so a fixed scale factor would silently corrupt
part of the comparison. Same per-image-actual-dimensions convention as
evaluate_macula_localization.py and evaluate_optic_disc_full_pipeline.py.

Run with:
    .venv\\Scripts\\python.exe scripts\\evaluate_disc_localization.py
"""

import argparse
import csv
import os
import sys

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from src.segmentation import optic_disc, vessels

ADAM_ROOT = os.path.join(PROJECT_ROOT, "ADAM", "Training400")
DISC_MASK_DIR = os.path.join(ADAM_ROOT, "Disc_Masks")


def _image_path(img_name: str) -> str:
    folder = "AMD" if img_name.startswith("A") else "Non-AMD"
    return os.path.join(ADAM_ROOT, folder, img_name)


def _ground_truth_disc_mask(stem: str, working_shape: tuple) -> np.ndarray | None:
    """ADAM's Disc_Masks are 8-bit BMPs where the DISC is 0 (black) and
    background is 255 (white) -- the inverse of the usual convention, so this
    is asserted rather than assumed (a silent polarity flip here would turn
    every hit into a miss and vice versa, and the resulting numbers would
    still look plausible).
    """
    path = os.path.join(DISC_MASK_DIR, f"{stem}.bmp")
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None

    disc = mask == 0
    # Sanity: the disc must be a small minority of the frame. If this trips,
    # the polarity assumption above is wrong for this file.
    if disc.mean() > 0.25:
        raise ValueError(f"{path}: 'disc' (pixel==0) covers {disc.mean():.1%} of the frame -- polarity assumption wrong?")

    h, w = working_shape[:2]
    return cv2.resize(disc.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)


def _summarize(name: str, values: list) -> str:
    if not values:
        return f"  {name:<14} (none)"
    arr = np.array([v for v in values if not np.isnan(v)])
    if arr.size == 0:
        return f"  {name:<14} (all NaN)"
    return (
        f"  {name:<14} n={arr.size:<4} mean={arr.mean():.3f}  median={np.median(arr):.3f}  "
        f"p10={np.percentile(arr, 10):.3f}  p90={np.percentile(arr, 90):.3f}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dump-csv", default=None, help="Write per-image metrics to this CSV (used for threshold calibration).")
    args = parser.parse_args()

    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(DISC_MASK_DIR) if f.endswith(".bmp"))
    print(f"Evaluating Stage 6.1 disc localization against {len(stems)} ADAM ground-truth disc masks...\n")

    records = []  # (stem, is_amd, hit, confident, circularity, solidity, diameter_fraction, reasons)
    unannotated = []

    for stem in tqdm(stems, total=len(stems)):
        image = cv2.imread(_image_path(f"{stem}.jpg"))
        if image is None:
            raise FileNotFoundError(_image_path(f"{stem}.jpg"))

        working = vessels._resize_to_working_width(image)
        gt_disc = _ground_truth_disc_mask(stem, working.shape)
        if gt_disc is None or not gt_disc.any():
            # 130 of ADAM's 400 Disc_Masks are entirely white -- i.e. the disc
            # is simply not annotated for those images, not "annotated as
            # absent". They carry no ground truth to score against, so they're
            # excluded from every number below rather than silently counted as
            # misses (which would understate accuracy by a third of the set).
            unannotated.append(stem)
            continue

        info = optic_disc.locate_disc_classical(working)
        cx, cy = info["center_xy"]
        h, w = gt_disc.shape[:2]
        xi, yi = int(round(cx)), int(round(cy))

        # "Hit" = the predicted center falls inside the true disc. This is the
        # decision that actually matters downstream: crop_disc_roi() centers
        # the Stage 6.2 ROI on this point, so a center inside the disc yields
        # a usable crop and one outside it yields a crop of the wrong anatomy.
        hit = bool(0 <= xi < w and 0 <= yi < h and gt_disc[yi, xi])

        records.append(
            (
                stem,
                stem.startswith("A"),
                hit,
                bool(info["confident"]),
                info["circularity"],
                info["solidity"],
                info["diameter_fraction"],
                info["implausible_reasons"],
            )
        )

    if args.dump_csv:
        with open(args.dump_csv, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["stem", "is_amd", "hit", "confident", "circularity", "solidity", "diameter_fraction"])
            for stem, is_amd, hit, confident, circ, sol, frac, _reasons in records:
                writer.writerow([stem, int(is_amd), int(hit), int(confident), circ, sol, frac])
        print(f"\nWrote per-image metrics for {len(records)} images to {args.dump_csv}")

    total = len(records)
    hits = [r for r in records if r[2]]
    misses = [r for r in records if not r[2]]
    amd = [r for r in records if r[1]]
    non_amd = [r for r in records if not r[1]]

    print(f"\n{'=' * 72}")
    print(f"Scored {total} images with a real disc annotation.")
    print(f"Excluded {len(unannotated)} of {len(stems)} ADAM masks that are entirely blank (no disc annotated).")
    print(f"{'=' * 72}")

    print(f"\n{'=' * 72}")
    print("1. LOCALIZATION ACCURACY (predicted center inside ground-truth disc)")
    print(f"{'=' * 72}")
    print(f"  overall:   {len(hits)}/{total} ({len(hits) / total:.1%})")
    for label, subset in (("AMD", amd), ("Non-AMD", non_amd)):
        sub_hits = sum(1 for r in subset if r[2])
        print(f"  {label:<9} {sub_hits}/{len(subset)} ({sub_hits / len(subset):.1%})" if subset else f"  {label}: none")

    print(f"\n{'=' * 72}")
    print("2. GEOMETRY METRICS, correct vs incorrect localizations")
    print("   (this is what the thresholds in optic_disc.py were calibrated from)")
    print(f"{'=' * 72}")
    for label, subset in (("CORRECT", hits), ("INCORRECT", misses)):
        print(f"\n {label} localizations (n={len(subset)}):")
        print(_summarize("circularity", [r[4] for r in subset]))
        print(_summarize("solidity", [r[5] for r in subset]))
        print(_summarize("diam_fraction", [r[6] for r in subset]))

    print(f"\n{'=' * 72}")
    print("3. DOES THE PLAUSIBILITY CHECK CATCH THE BAD CROPS?")
    print(f"{'=' * 72}")
    # A useful check is one that fires (confident=False) on the misses and
    # stays quiet (confident=True) on the hits.
    hit_conf = sum(1 for r in records if r[2] and r[3])
    hit_flag = sum(1 for r in records if r[2] and not r[3])
    miss_conf = sum(1 for r in records if not r[2] and r[3])
    miss_flag = sum(1 for r in records if not r[2] and not r[3])

    print("                      confident=True   confident=False (flagged)")
    print(f"  correct localization  {hit_conf:>8}         {hit_flag:>8}   <- flagged good crops (cost: false alarms)")
    print(f"  WRONG localization    {miss_conf:>8}         {miss_flag:>8}   <- caught bad crops (the point)")

    if misses:
        print(f"\n  Recall on bad crops:   {miss_flag}/{len(misses)} ({miss_flag / len(misses):.1%}) of wrong localizations flagged")
    if hits:
        print(f"  False-alarm rate:      {hit_flag}/{len(hits)} ({hit_flag / len(hits):.1%}) of correct localizations needlessly flagged")

    silent_failures = miss_conf
    print(f"\n  SILENT failures remaining (wrong crop, still confident): {silent_failures}/{total} ({silent_failures / total:.1%})")
    print(f"  (before this check, ALL {len(misses)} wrong localizations were silent)")

    reason_counts: dict[str, int] = {}
    for r in records:
        if not r[3]:
            for reason in r[7]:
                key = reason.split(" (")[0]
                reason_counts[key] = reason_counts.get(key, 0) + 1
    if reason_counts:
        print("\n  Flag reasons (a candidate can fail more than one check):")
        for reason, count in sorted(reason_counts.items(), key=lambda kv: -kv[1]):
            print(f"    {count:>4}  {reason}")


if __name__ == "__main__":
    main()
