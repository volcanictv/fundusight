"""Build a grid of Grad-CAM / EigenCAM / LayerCAM heatmaps for real positive
DR cases, overlaid on the original image, so the explainability wiring can be
sanity-checked visually: does the heatmap land on lesions, or is it attending
to the image border? (A common bug worth explicitly checking — see
ROADMAP.md Phase 4.) Run with:

    .venv\\Scripts\\python.exe scripts\\demo_gradcam.py

Writes gradcam_grid.png next to this script.
"""

import os
import sys

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from src.detection.infer import load_model, predict
from src.explainability.gradcam import CAM_METHODS, generate_cam

DATA_DIR = os.path.join(PROJECT_ROOT, "APTOS 2019")
IMG_DIR = os.path.join(DATA_DIR, "test_images", "test_images")
WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "dr_efficientnet_b0.pth")
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gradcam_grid.png")
N_CASES = 3


def main():
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"No trained weights at {WEIGHTS_PATH}. Run src/detection/train.py first.")

    df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    # Positive, higher-severity DR cases only - that's what the roadmap's
    # sanity check is about (does the heatmap find the lesions).
    positive_cases = df[df["diagnosis"] >= 2].sample(n=N_CASES, random_state=1)

    model = load_model(WEIGHTS_PATH)

    rows = []
    for _, row in positive_cases.iterrows():
        path = os.path.join(IMG_DIR, row["id_code"] + ".png")
        image = cv2.imread(path)

        result = predict(model, image)
        original_resized = cv2.resize(image, (224, 224))

        overlays = [generate_cam(model, image, method=m, target_class=result["class_idx"]) for m in CAM_METHODS]
        rows.append(np.hstack([original_resized, *overlays]))
        print(
            f"{row['id_code']}  true={row['diagnosis']}  "
            f"pred={result['class_idx']} ({result['label']}, {result['probability']*100:.1f}%)"
        )

    grid = np.vstack(rows)
    cv2.imwrite(OUT_PATH, grid)
    print(f"\nSaved {grid.shape[1]}x{grid.shape[0]} grid to {OUT_PATH}")
    print(f"Columns: original | {' | '.join(CAM_METHODS)}")


if __name__ == "__main__":
    main()
