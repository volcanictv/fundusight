"""Run assess_quality() against a sample of real APTOS images and print the
results. Not a test (no assertions) — just something to eyeball when
tweaking quality.py's thresholds. Run with:

    .venv\\Scripts\\python.exe scripts\\demo_quality.py
"""

import os
import sys

import cv2
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from src.preprocessing.quality import assess_quality

# Resolved from the script's own location, not the current working
# directory — VS Code's debugger runs with the workspace root as cwd, which
# isn't necessarily this project folder.
DATA_DIR = os.path.join(PROJECT_ROOT, "APTOS 2019")
IMG_DIR = os.path.join(DATA_DIR, "train_images", "train_images")
N_SAMPLES = 10


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "train_1.csv"))
    sample = df.sample(n=N_SAMPLES, random_state=0)

    n_passed = 0
    for _, row in sample.iterrows():
        path = os.path.join(IMG_DIR, row["id_code"] + ".png")
        image = cv2.imread(path)
        result = assess_quality(image)
        n_passed += result["passed"]

        status = "PASS" if result["passed"] else "FAIL"
        focus = result["checks"]["focus"]
        exposure = result["checks"]["exposure"]
        print(
            f"{row['id_code']}  {status}  score={result['score']:>5.1f}  "
            f"focus_var={focus['laplacian_variance']:>6.1f}  "
            f"mean_brightness={exposure['mean_brightness']:>5.1f}"
        )

    print(f"\n{n_passed}/{N_SAMPLES} passed")


if __name__ == "__main__":
    main()
