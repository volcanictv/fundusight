"""Phase 6: validate the macula/fovea heuristic against real ground truth.

`locate_macula_classical()` (src/segmentation/optic_disc.py) has never been
checked against anything but "looks about right relative to the disc" --
REFUGE2 (its own training/eval dataset for disc/cup) ships no fovea
coordinate labels at all. ADAM does (Fovea_location.xlsx, one (Fovea_X,
Fovea_Y) row per of all 400 Training400 images), so this is the first real
ground-truth check.

Runs Stage 6.1's classical disc localizer (locate_disc_classical()) +
locate_macula_classical() -- the disc-relative darkest-point heuristic --
on all 400 images. Both return coordinates in VESSEL_WORKING_WIDTH-resized
working-image space, which have to be converted back to each image's own
ORIGINAL pixel space before comparing against Fovea_location.xlsx (already
in original pixel space): ADAM ships images at two different native
resolutions (2056x2124 and 1444x1444), so a single fixed scale factor would
silently corrupt the comparison for whichever subset doesn't match it. Same
per-image-actual-dimensions scaling convention as
evaluate_optic_disc_full_pipeline.py's _ground_truth_working_masks() and
vessels._resize_to_working_width() itself.

Reports raw Euclidean pixel error (mean + median, same both-numbers
convention as evaluate_optic_disc_full_pipeline.py's CDR error report) and
disc-diameter-normalized error (using Stage 6.1's own disc_diameter_px --
the same scale unit locate_macula_classical()'s own search window is
defined in, so the normalized number is directly interpretable against the
heuristic's own geometry), plus the full error distribution and worst
outliers -- a heuristic that's fine on average but wildly wrong on a
handful of images is a real risk once its output crops a downstream ROI or
lands on a report.

Run with:
    .venv\\Scripts\\python.exe scripts\\evaluate_macula_localization.py
"""

import os
import sys

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from src.segmentation import optic_disc, vessels

ADAM_ROOT = os.path.join(PROJECT_ROOT, "ADAM", "Training400")


def _image_path(img_name: str) -> str:
    folder = "AMD" if img_name.startswith("A") else "Non-AMD"
    return os.path.join(ADAM_ROOT, folder, img_name)


def main():
    fovea_df = pd.read_excel(os.path.join(ADAM_ROOT, "Fovea_location.xlsx"))
    print(f"Evaluating macula/fovea heuristic against {len(fovea_df)} ADAM ground-truth labels...")

    pixel_errors = []
    normalized_errors = []
    per_image = []  # (imgName, pixel_error, normalized_error, disc_found)
    disc_not_found = []
    macula_not_found = []

    for row in tqdm(fovea_df.itertuples(index=False), total=len(fovea_df)):
        path = _image_path(row.imgName)
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(path)
        orig_h, orig_w = image.shape[:2]

        working = vessels._resize_to_working_width(image)
        disc_info = optic_disc.locate_disc_classical(working)
        if not disc_info["found"]:
            disc_not_found.append(row.imgName)

        macula_info = optic_disc.locate_macula_classical(working, disc_info["center_xy"], disc_info["diameter_px"])
        if not macula_info["found"]:
            macula_not_found.append(row.imgName)
            continue

        # Same per-image scale (not a fixed constant) _resize_to_working_width
        # itself uses -- ADAM ships two different native resolutions.
        scale = vessels.VESSEL_WORKING_WIDTH / orig_w
        pred_x = macula_info["location_xy"][0] / scale
        pred_y = macula_info["location_xy"][1] / scale
        disc_diameter_original = disc_info["diameter_px"] / scale

        pixel_error = float(np.hypot(pred_x - row.Fovea_X, pred_y - row.Fovea_Y))
        normalized_error = pixel_error / disc_diameter_original if disc_diameter_original > 0 else float("inf")

        # The heuristic itself has no eye-laterality info (see
        # locate_macula_classical()'s docstring) and tries both sides of the
        # disc, picking whichever is darker -- diagnostic for whether a
        # "wrong side" pick (fovea is only ever temporal to the disc on ONE
        # side, never both) is a major error driver, distinct from the
        # heuristic finding the right side but the wrong point on it.
        disc_x_original = disc_info["center_xy"][0] / scale
        same_side = np.sign(pred_x - disc_x_original) == np.sign(row.Fovea_X - disc_x_original)

        pixel_errors.append(pixel_error)
        normalized_errors.append(normalized_error)
        per_image.append((row.imgName, pixel_error, normalized_error, disc_info["found"], bool(same_side)))

    pixel_errors = np.array(pixel_errors)
    normalized_errors = np.array(normalized_errors)

    print(
        f"\nEvaluated {len(pixel_errors)}/{len(fovea_df)} images with a macula prediction "
        f"({len(macula_not_found)} macula-search failures -- no prediction at all, "
        f"{len(disc_not_found)} of the evaluated images had Stage 6.1 disc localization "
        f"fall back to a low-confidence estimate)"
    )
    if macula_not_found:
        print(f"  macula search failed on: {macula_not_found}")
    if disc_not_found:
        print(f"  disc localization fell back on: {disc_not_found}")

    print("\nRaw Euclidean pixel error (original per-image pixel space):")
    print(f"  mean   = {pixel_errors.mean():.1f} px")
    print(f"  median = {np.median(pixel_errors):.1f} px")

    print("\nDisc-diameter-normalized error (pixel error / Stage 6.1 disc diameter):")
    print(f"  mean   = {normalized_errors.mean():.3f} disc diameters")
    print(f"  median = {np.median(normalized_errors):.3f} disc diameters")

    print("\nNormalized-error distribution (percentiles):")
    for p in (10, 25, 50, 75, 90, 95, 99):
        print(f"  p{p:<3} = {np.percentile(normalized_errors, p):.3f}")
    print(f"  max    = {normalized_errors.max():.3f}")

    same_side_flags = np.array([t[4] for t in per_image])
    same_side_errors = normalized_errors[same_side_flags]
    opposite_side_errors = normalized_errors[~same_side_flags]
    print(
        f"\nSide-of-disc check (fovea is only ever temporal to the disc on ONE "
        f"side; the heuristic has no eye-laterality info and tries both):"
    )
    print(f"  same side as ground truth:     {same_side_flags.sum()}/{len(same_side_flags)}")
    print(f"    -> median normalized error = {np.median(same_side_errors):.3f}")
    print(f"  opposite side from ground truth: {(~same_side_flags).sum()}/{len(same_side_flags)}")
    if opposite_side_errors.size:
        print(f"    -> median normalized error = {np.median(opposite_side_errors):.3f}")

    worst = sorted(per_image, key=lambda t: t[2], reverse=True)[:10]
    print("\nWorst 10 outliers (imgName, pixel_error, normalized_error, disc_found, same_side):")
    for name, px_err, norm_err, disc_found, same_side in worst:
        print(f"  {name}: {px_err:.1f} px, {norm_err:.3f} disc diameters, disc_found={disc_found}, same_side={same_side}")


if __name__ == "__main__":
    main()
