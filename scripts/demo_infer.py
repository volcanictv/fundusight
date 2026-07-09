"""Run the trained DR classifier on a sample of real held-out test images and
print true vs. predicted severity, so the model can be eyeballed without
re-running the full training/eval loop. Run with:

    .venv\\Scripts\\python.exe scripts\\demo_infer.py
"""

import os
import sys

import cv2
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from src.detection.infer import load_model, predict

DATA_DIR = os.path.join(PROJECT_ROOT, "APTOS 2019")
IMG_DIR = os.path.join(DATA_DIR, "test_images", "test_images")
WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "dr_efficientnet_b0.pth")
N_SAMPLES = 10


def main():
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(
            f"No trained weights at {WEIGHTS_PATH}. Run src/detection/train.py first."
        )

    df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    sample = df.sample(n=N_SAMPLES, random_state=2)

    model = load_model(WEIGHTS_PATH)

    n_correct = 0
    for _, row in sample.iterrows():
        path = os.path.join(IMG_DIR, row["id_code"] + ".png")
        image = cv2.imread(path)
        result = predict(model, image)

        true_label = int(row["diagnosis"])
        correct = result["class_idx"] == true_label
        n_correct += correct

        marker = "OK" if correct else "MISS"
        print(
            f"{row['id_code']}  {marker:<4}  true={true_label}  "
            f"pred={result['class_idx']} ({result['label']}, {result['probability']*100:.1f}%)"
        )

    print(f"\n{n_correct}/{N_SAMPLES} correct on this sample")


if __name__ == "__main__":
    main()
