"""Run the optic disc/cup/macula pipeline on a few real APTOS images so the
disc mask, cup mask, macula marker, and vertical CDR can be eyeballed
instead of just trusting the numbers. Shows the classical (Stage 6.1 crop +
intensity-threshold) pipeline side by side with the trained hybrid
(REFUGE2 U-Net) pipeline, when a checkpoint is available. Run with:

    .venv\\Scripts\\python.exe scripts\\demo_optic_disc.py

Writes optic_disc_grid.png next to this script and prints biomarkers to stdout.
"""

import os
import sys

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from src.segmentation.optic_disc import compute_optic_biomarkers
from src.segmentation.optic_disc_infer import DEFAULT_WEIGHTS_PATH, compute_optic_biomarkers_hybrid, load_optic_disc_model
from src.segmentation.vessels import VESSEL_WORKING_WIDTH

DATA_DIR = os.path.join(PROJECT_ROOT, "APTOS 2019")
IMG_DIR = os.path.join(DATA_DIR, "train_images", "train_images")
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "optic_disc_grid.png")
# REFUGE2 is training data only -- demoed here on APTOS, the same kind of
# arbitrary fundus photo the real app will see, same reasoning
# demo_vessels.py demos on APTOS despite training on DRIVE/STARE/CHASE_DB1.
WEIGHTS_PATH = DEFAULT_WEIGHTS_PATH
N_SAMPLES = 3
DISPLAY_WIDTH = 900

_DISC_COLOR = (0, 255, 255)  # yellow, BGR
_CUP_COLOR = (0, 0, 255)  # red, BGR
_MACULA_COLOR = (0, 255, 0)  # green, BGR
_MACULA_MARKER_RADIUS = 12


def _overlay(image, result):
    out = image.copy()
    out[result["disc_mask"]] = _DISC_COLOR
    out[result["cup_mask"]] = _CUP_COLOR  # drawn after disc -- cup wins any overlap
    if result["macula_location"] is not None:
        cv2.circle(out, result["macula_location"], _MACULA_MARKER_RADIUS, _MACULA_COLOR, 2)
    return out


def _print_biomarkers(label, id_code, result):
    print(
        f"  [{label}] {id_code}: "
        f"vertical_cdr={result['vertical_cdr']:.3f}  "
        f"disc_diameter={result['disc_diameter_px']}px  "
        f"cup_diameter={result['cup_diameter_px']}px  "
        f"macula={result['macula_location']}  "
        f"disc_found={result['disc_found']}  macula_found={result['macula_found']}"
    )


def main():
    # The hybrid model is optional -- compute_optic_biomarkers_hybrid() is a
    # drop-in swap for the classical pipeline once trained, but Stage 6.1's
    # classical localization/crop always runs regardless (see ROADMAP.md),
    # and the classical intensity-threshold disc/cup segmentation remains
    # the fallback when no checkpoint exists yet.
    model = None
    if os.path.exists(WEIGHTS_PATH):
        model = load_optic_disc_model(WEIGHTS_PATH, device="cpu")
        print(f"Loaded hybrid model from {WEIGHTS_PATH}")
    else:
        print(f"No checkpoint at {WEIGHTS_PATH} -- showing classical pipeline only.")

    df = pd.read_csv(os.path.join(DATA_DIR, "train_1.csv"))
    sample = df.sample(n=N_SAMPLES, random_state=7)

    rows = []
    for _, row in sample.iterrows():
        id_code = row["id_code"]
        path = os.path.join(IMG_DIR, id_code + ".png")
        image = cv2.imread(path)

        # Both compute_optic_biomarkers() and compute_optic_biomarkers_hybrid()
        # canonicalize resolution internally -- pass the native image
        # straight through. For the overlay images below we need a version
        # at that same VESSEL_WORKING_WIDTH resolution so it lines up with
        # the returned mask shape.
        h, w = image.shape[:2]
        working = cv2.resize(image, (VESSEL_WORKING_WIDTH, round(h * VESSEL_WORKING_WIDTH / w)))

        classical = compute_optic_biomarkers(image)
        classical_overlay = _overlay(working, classical)
        _print_biomarkers("classical", id_code, classical)

        row_images = [working, classical_overlay]
        if model is not None:
            hybrid = compute_optic_biomarkers_hybrid(image, model, device="cpu")
            hybrid_overlay = _overlay(working, hybrid)
            _print_biomarkers("hybrid   ", id_code, hybrid)
            row_images.append(hybrid_overlay)

        rows.append(np.hstack(row_images))

    max_h = max(r.shape[0] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, max_h - r.shape[0], 0, 0, cv2.BORDER_CONSTANT) for r in rows]
    grid = np.vstack(rows)
    grid = cv2.resize(grid, (DISPLAY_WIDTH, round(grid.shape[0] * DISPLAY_WIDTH / grid.shape[1])), interpolation=cv2.INTER_AREA)

    cv2.imwrite(OUT_PATH, grid)
    print(f"Saved {grid.shape[1]}x{grid.shape[0]} grid to {OUT_PATH}")
    if model is not None:
        print("Columns: raw | classical overlay (disc=yellow, cup=red, macula=green) | hybrid overlay")
    else:
        print("Columns: raw | classical overlay (disc=yellow, cup=red, macula=green)")


if __name__ == "__main__":
    main()
