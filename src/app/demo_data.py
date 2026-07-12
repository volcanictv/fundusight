"""Phase 9: demo-mode sample images.

Demo mode lets a stranger try the dashboard without their own fundus photo
to upload. It reads a small (10-image, 2-per-severity-class) set bundled at
`src/app/demo_images/` -- a deliberate, fixed selection re-encoded as
downscaled JPEGs (see `labels.csv` alongside them) so the demo walkthrough is
repeatable and lightweight enough to commit. These are a handful of APTOS
2019 samples; APTOS's own Kaggle license terms don't extend redistribution
rights, so shipping them here is a deliberate call made by the project owner
for this small educational subset, not a default to repeat for other
datasets without separately checking their terms.
"""

import os

import cv2
import numpy as np
import pandas as pd

from src.detection.model import SEVERITY_LABELS

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_IMG_DIR = os.path.join(_PROJECT_ROOT, "src", "app", "demo_images")
_CSV_PATH = os.path.join(_IMG_DIR, "labels.csv")


def list_demo_images() -> list:
    """One dict per bundled sample: {"id_code", "diagnosis" (int), "label"
    (str), "path"}, sorted by severity so the picker reads low-to-high.
    """
    if not os.path.exists(_CSV_PATH):
        return []

    df = pd.read_csv(_CSV_PATH)
    demo_images = []
    for _, row in df.iterrows():
        path = os.path.join(_IMG_DIR, row["id_code"] + ".jpg")
        if not os.path.exists(path):
            continue
        demo_images.append(
            {
                "id_code": row["id_code"],
                "diagnosis": int(row["diagnosis"]),
                "label": SEVERITY_LABELS[int(row["diagnosis"])],
                "path": path,
            }
        )

    demo_images.sort(key=lambda item: item["diagnosis"])
    return demo_images


def load_demo_image(path: str) -> np.ndarray:
    """cv2.imread wrapper, matching this pipeline's BGR convention
    everywhere else -- kept as its own function so main.py doesn't import
    cv2 just for this one call.
    """
    return cv2.imread(path)
