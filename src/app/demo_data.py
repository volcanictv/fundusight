"""Phase 9: demo-mode sample images.

Demo mode lets a stranger try the dashboard without their own fundus photo
to upload. APTOS 2019 is Kaggle-licensed and not redistributable, so this
deliberately does NOT bundle/ship any images as committed app assets -- it
only references images already downloaded locally per the README (same
`APTOS 2019/train_images/train_images/*.png` + `train_1.csv` pair the demo
scripts under scripts/ already use), and degrades to an empty list with a
helpful message if that folder isn't present yet (e.g. a fresh clone).
"""

import os

import cv2
import numpy as np
import pandas as pd

from src.detection.model import SEVERITY_LABELS

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DATA_DIR = os.path.join(_PROJECT_ROOT, "APTOS 2019")
_IMG_DIR = os.path.join(_DATA_DIR, "train_images", "train_images")
_CSV_PATH = os.path.join(_DATA_DIR, "train_1.csv")

# One example per DR severity class -- a deliberate, fixed selection
# (not re-randomized per page load) so the demo walkthrough is repeatable:
# a stranger clicking through demo mode twice sees the same five cases.
_N_PER_CLASS = 1
_RANDOM_STATE = 7


def list_demo_images() -> list:
    """Returns [] if the APTOS dataset isn't downloaded locally yet,
    otherwise one dict per selected sample:
    {"id_code", "diagnosis" (int), "label" (str), "path"}, sorted by
    severity so the picker reads low-to-high.
    """
    if not os.path.isdir(_IMG_DIR) or not os.path.exists(_CSV_PATH):
        return []

    df = pd.read_csv(_CSV_PATH)
    demo_images = []
    for diagnosis, group in df.groupby("diagnosis"):
        picked = group.sample(n=min(_N_PER_CLASS, len(group)), random_state=_RANDOM_STATE)
        for _, row in picked.iterrows():
            path = os.path.join(_IMG_DIR, row["id_code"] + ".png")
            if not os.path.exists(path):
                continue
            demo_images.append(
                {
                    "id_code": row["id_code"],
                    "diagnosis": int(diagnosis),
                    "label": SEVERITY_LABELS[int(diagnosis)],
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
