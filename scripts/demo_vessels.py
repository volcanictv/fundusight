"""Run the vessel segmentation pipeline on a few real APTOS images so the
mask and biomarkers (density, branch count, tortuosity, average width) can
be eyeballed instead of just trusting the numbers. Shows the classical
Frangi+hysteresis pipeline side by side with the trained hybrid
Frangi+U-Net model, when a checkpoint is available. Run with:

    .venv\\Scripts\\python.exe scripts\\demo_vessels.py

Writes vessels_grid.png next to this script and prints biomarkers to stdout.
"""

import os
import sys

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from src.segmentation.vessels import VESSEL_WORKING_WIDTH, compute_biomarkers
from src.segmentation.vessel_infer import DEFAULT_WEIGHTS_PATH, compute_biomarkers_hybrid, load_vessel_model

DATA_DIR = os.path.join(PROJECT_ROOT, "APTOS 2019")
IMG_DIR = os.path.join(DATA_DIR, "train_images", "train_images")
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vessels_grid.png")
WEIGHTS_PATH = DEFAULT_WEIGHTS_PATH
N_SAMPLES = 3
# Final saved grid width -- purely a display concern for a viewable PNG, not
# related to VESSEL_WORKING_WIDTH, which now drives the actual segmentation
# resolution inside compute_biomarkers().
DISPLAY_WIDTH = 900

_MASK_COLOR = (0, 0, 255)  # red, BGR


def _overlay(image, region, color):
    result = image.copy()
    result[region] = color
    return result


def _print_biomarkers(label, id_code, result):
    print(
        f"  [{label}] {id_code}: "
        f"vessel_density={result['vessel_density']:.2f}%  "
        f"branch_count={result['branch_count']}  "
        f"tortuosity={result['tortuosity']:.3f}  "
        f"average_width={result['average_width']:.2f}px"
    )


def main():
    # The hybrid model is optional -- compute_biomarkers_hybrid()/
    # segment_vessels_hybrid() are a drop-in swap for the classical
    # pipeline once trained, but the classical Frangi+hysteresis pipeline
    # remains the fallback when no checkpoint exists yet (see README).
    model = None
    if os.path.exists(WEIGHTS_PATH):
        model = load_vessel_model(WEIGHTS_PATH, device="cpu")
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

        # Both compute_biomarkers() and compute_biomarkers_hybrid()
        # canonicalize resolution internally -- pass the native image
        # straight through. For the overlay images below we need a version
        # at that same VESSEL_WORKING_WIDTH resolution so it lines up with
        # the returned mask shape.
        h, w = image.shape[:2]
        working = cv2.resize(image, (VESSEL_WORKING_WIDTH, round(h * VESSEL_WORKING_WIDTH / w)))

        classical = compute_biomarkers(image)
        classical_overlay = _overlay(working, classical["mask"], _MASK_COLOR)
        _print_biomarkers("classical", id_code, classical)

        row_images = [working, classical_overlay]
        if model is not None:
            hybrid = compute_biomarkers_hybrid(image, model, device="cpu")
            hybrid_overlay = _overlay(working, hybrid["mask"], _MASK_COLOR)
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
        print("Columns: raw | classical (Frangi+hysteresis) mask overlay | hybrid (Frangi+U-Net) mask overlay")
    else:
        print("Columns: raw | classical (Frangi+hysteresis) mask overlay")


if __name__ == "__main__":
    main()
