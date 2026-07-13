"""Phase 6 (Stage 6.0): does the coarse full-frame locator actually rescue the
localizations the classical search gets wrong?

This is the test that decides whether Stage 6.0 earns its place in the
pipeline, and it is deliberately a HARD one: the locator is trained on
REFUGE2 (a glaucoma set) and evaluated here on ADAM (an AMD set, with the
hemorrhages and exudate that break the classical search). That is a real
cross-dataset generalization test, not a re-run on the training distribution
-- if it only worked in-domain it would be worthless, since in-domain is
exactly where the classical search already works.

Three numbers matter, in this order:

1. SILENT FAILURES (wrong crop, still reported confident). Must be ZERO. The
   entire value of the plausibility guard is that a wrong crop is never
   reported as a trustworthy CDR. A "fix" that raises accuracy while
   reintroducing silent failures is a regression, not an improvement, and
   this script is built to catch that trade being made accidentally.
2. USABLE CDR YIELD (correct AND confident). This is what a user actually
   gets: an image whose CDR is both right and reported. Raising it is the
   point of the whole exercise.
3. Raw localization accuracy. Necessary but not sufficient -- accuracy that
   arrives as unflagged wrong crops is worse than useless.

Run with:
    .venv\\Scripts\\python.exe scripts\\evaluate_disc_locator.py
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from src.segmentation import optic_disc, optic_disc_infer, vessels

ADAM_ROOT = os.path.join(PROJECT_ROOT, "ADAM", "Training400")
DISC_MASK_DIR = os.path.join(ADAM_ROOT, "Disc_Masks")


def _ground_truth_disc_mask(stem: str, working_shape: tuple):
    """ADAM's Disc_Masks: the DISC is 0 (black), background 255 -- the inverse
    of the usual convention, so it is asserted rather than assumed (a silent
    polarity flip would turn every hit into a miss and the numbers would still
    look plausible). Same convention as evaluate_disc_localization.py.
    """
    mask = cv2.imread(os.path.join(DISC_MASK_DIR, f"{stem}.bmp"), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    disc = mask == 0
    if disc.mean() > 0.25:
        raise ValueError(f"{stem}: 'disc' covers {disc.mean():.1%} of frame -- polarity assumption wrong?")
    h, w = working_shape[:2]
    return cv2.resize(disc.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)


def _hit(center_xy: tuple, gt_disc: np.ndarray) -> bool:
    h, w = gt_disc.shape[:2]
    x, y = int(round(center_xy[0])), int(round(center_xy[1]))
    return bool(0 <= x < w and 0 <= y < h and gt_disc[y, x])


def _report(title: str, records: list, total: int):
    correct = sum(1 for r in records if r["hit"])
    silent = sum(1 for r in records if not r["hit"] and r["confident"])
    usable = sum(1 for r in records if r["hit"] and r["confident"])
    flagged_good = sum(1 for r in records if r["hit"] and not r["confident"])

    print(f"\n{title}")
    print(f"  localization accuracy : {correct}/{total} ({correct / total:.1%})")
    print(f"  SILENT FAILURES       : {silent}/{total} ({silent / total:.1%})   <- must be 0")
    print(f"  usable CDR yield      : {usable}/{total} ({usable / total:.1%})   <- correct AND confident")
    print(f"  needlessly flagged    : {flagged_good}/{total} ({flagged_good / total:.1%})")
    return {"correct": correct, "silent": silent, "usable": usable}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--locator-weights", default=optic_disc_infer.DEFAULT_LOCATOR_WEIGHTS_PATH)
    args = parser.parse_args()

    if not os.path.exists(args.locator_weights):
        sys.exit(f"No Stage 6.0 checkpoint at {args.locator_weights} -- train it with src/segmentation/disc_locator_train.py")

    locator = optic_disc_infer.load_disc_locator_model(args.locator_weights, device="cpu")
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(DISC_MASK_DIR) if f.endswith(".bmp"))

    classical_records, arbitrated_records = [], []
    source_counts: dict[str, int] = {}
    rescued, broken = [], []

    for stem in tqdm(stems, desc="ADAM"):
        folder = "AMD" if stem.startswith("A") else "Non-AMD"
        image = cv2.imread(os.path.join(ADAM_ROOT, folder, f"{stem}.jpg"))
        if image is None:
            continue
        working = vessels._resize_to_working_width(image)
        gt_disc = _ground_truth_disc_mask(stem, working.shape)
        if gt_disc is None or not gt_disc.any():
            continue  # 130 of ADAM's 400 masks carry no disc annotation at all

        classical = optic_disc.locate_disc_classical(working)
        arbitrated = optic_disc_infer.locate_disc_arbitrated(working, locator, device="cpu")

        c_hit, a_hit = _hit(classical["center_xy"], gt_disc), _hit(arbitrated["center_xy"], gt_disc)
        classical_records.append({"hit": c_hit, "confident": classical["confident"]})
        arbitrated_records.append({"hit": a_hit, "confident": arbitrated["confident"]})
        source_counts[arbitrated["source"]] = source_counts.get(arbitrated["source"], 0) + 1

        # A "rescue" is only a rescue if it is USABLE: the crop must become
        # both correct AND confident. Turning a wrong-but-flagged crop into a
        # correct-but-still-flagged one changes nothing a user can see.
        if not (c_hit and classical["confident"]) and (a_hit and arbitrated["confident"]):
            rescued.append((stem, arbitrated["source"]))
        if (c_hit and classical["confident"]) and not (a_hit and arbitrated["confident"]):
            broken.append((stem, arbitrated["source"]))

    total = len(classical_records)
    print(f"\n{'=' * 72}")
    print(f"Stage 6.0 coarse locator, evaluated on {total} ADAM images with a real disc annotation")
    print("(cross-dataset: locator trained on REFUGE2, tested here on ADAM)")
    print(f"{'=' * 72}")

    before = _report("BEFORE  (classical brightness + vascular convergence only)", classical_records, total)
    after = _report("AFTER   (+ Stage 6.0 coarse locator arbitration)", arbitrated_records, total)

    print(f"\n{'=' * 72}")
    print("NET EFFECT")
    print(f"{'=' * 72}")
    print(f"  localization accuracy : {before['correct']} -> {after['correct']}  ({after['correct'] - before['correct']:+d})")
    print(f"  usable CDR yield      : {before['usable']} -> {after['usable']}  ({after['usable'] - before['usable']:+d})")
    print(f"  silent failures       : {before['silent']} -> {after['silent']}  ({after['silent'] - before['silent']:+d})")
    print(f"\n  images RESCUED (became correct+confident) : {len(rescued)}")
    print(f"  images BROKEN  (lost correct+confident)   : {len(broken)}")
    if broken:
        print("    " + ", ".join(f"{s} ({src})" for s, src in broken[:10]))

    print("\n  Which stage produced the final crop:")
    for source, count in sorted(source_counts.items(), key=lambda kv: -kv[1]):
        print(f"    {count:>4}  {source}")

    if after["silent"] > 0:
        print("\n  *** WARNING: silent failures are non-zero. Stage 6.0 is handing downstream")
        print("      a wrong crop while reporting it as confident. Do not ship this. ***")


if __name__ == "__main__":
    main()
