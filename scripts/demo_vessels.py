"""Run the vessel segmentation pipeline on a few real APTOS images so the
mask/skeleton and the four biomarkers (density, branch count, tortuosity,
average width) can be eyeballed instead of just trusting the numbers. Run
with:

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
from src.segmentation.vessels import compute_biomarkers

DATA_DIR = os.path.join(PROJECT_ROOT, "APTOS 2019")
IMG_DIR = os.path.join(DATA_DIR, "train_images", "train_images")
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vessels_grid.png")
N_SAMPLES = 3
THUMB_WIDTH = 300

_MASK_COLOR = (0, 0, 255)  # red, BGR
_SKELETON_COLOR = (0, 255, 255)  # yellow, BGR
# Skeleton is 1px wide, which is nearly invisible at thumbnail resolution —
# dilate the drawn overlay a couple pixels for visibility only, after the
# biomarkers (computed from the true 1px skeleton) are already in hand.
_SKELETON_DISPLAY_KERNEL = np.ones((3, 3), np.uint8)


def _overlay(image, region, color):
    result = image.copy()
    result[region] = color
    return result


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "train_1.csv"))
    sample = df.sample(n=N_SAMPLES, random_state=7)

    rows = []
    for _, row in sample.iterrows():
        path = os.path.join(IMG_DIR, row["id_code"] + ".png")
        image = cv2.imread(path)
        h, w = image.shape[:2]
        image = cv2.resize(image, (THUMB_WIDTH, int(h * THUMB_WIDTH / w)))

        result = compute_biomarkers(image)
        mask_overlay = _overlay(image, result["mask"], _MASK_COLOR)
        skeleton_display = cv2.dilate(result["skeleton"].astype(np.uint8), _SKELETON_DISPLAY_KERNEL) > 0
        skeleton_overlay = _overlay(image, skeleton_display, _SKELETON_COLOR)

        rows.append(np.hstack([image, mask_overlay, skeleton_overlay]))

        print(
            f"{row['id_code']}: "
            f"vessel_density={result['vessel_density']:.2f}%  "
            f"branch_count={result['branch_count']}  "
            f"tortuosity={result['tortuosity']:.3f}  "
            f"average_width={result['average_width']:.2f}px"
        )

    max_h = max(r.shape[0] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, max_h - r.shape[0], 0, 0, cv2.BORDER_CONSTANT) for r in rows]
    grid = np.vstack(rows)

    cv2.imwrite(OUT_PATH, grid)
    print(f"Saved {grid.shape[1]}x{grid.shape[0]} grid to {OUT_PATH}")
    print("Columns: raw | vessel mask overlay | skeleton overlay")


if __name__ == "__main__":
    main()
