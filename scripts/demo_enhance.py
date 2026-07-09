"""Build a before/after grid (raw | illumination-corrected | CLAHE |
color-normalized) for a few real APTOS images, so the preprocessing steps
in enhance.py can be eyeballed instead of just trusting the numbers. Run
with:

    .venv\\Scripts\\python.exe scripts\\demo_enhance.py

Writes preprocessing_grid.png next to this script.
"""

import os
import sys

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from src.preprocessing.enhance import apply_clahe, correct_illumination, normalize_color

DATA_DIR = os.path.join(PROJECT_ROOT, "APTOS 2019")
IMG_DIR = os.path.join(DATA_DIR, "train_images", "train_images")
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "preprocessing_grid.png")
N_SAMPLES = 3
THUMB_WIDTH = 300


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "train_1.csv"))
    sample = df.sample(n=N_SAMPLES, random_state=7)

    rows = []
    for _, row in sample.iterrows():
        path = os.path.join(IMG_DIR, row["id_code"] + ".png")
        image = cv2.imread(path)
        h, w = image.shape[:2]
        image = cv2.resize(image, (THUMB_WIDTH, int(h * THUMB_WIDTH / w)))

        illuminated = correct_illumination(image)
        contrast_enhanced = apply_clahe(illuminated)
        normalized = normalize_color(contrast_enhanced)

        rows.append(np.hstack([image, illuminated, contrast_enhanced, normalized]))

    max_h = max(r.shape[0] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, max_h - r.shape[0], 0, 0, cv2.BORDER_CONSTANT) for r in rows]
    grid = np.vstack(rows)

    cv2.imwrite(OUT_PATH, grid)
    print(f"Saved {grid.shape[1]}x{grid.shape[0]} grid to {OUT_PATH}")
    print("Columns: raw | illumination-corrected | CLAHE | color-normalized")


if __name__ == "__main__":
    main()
