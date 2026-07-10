"""Phase 3: DR Detection — local inference.

Loads a trained checkpoint and runs prediction on a single fundus photo.
Training happens on the GPU (see train.py); inference runs on CPU, matching
how the rest of the pipeline (quality.py, enhance.py) and the eventual
Streamlit app operate.
"""

import os

import cv2
import numpy as np
import torch

from src.detection.dataset import build_transforms
from src.detection.model import SEVERITY_LABELS, build_model

# Matches train.py's --output default and scripts/demo_infer.py's
# WEIGHTS_PATH -- the one place downstream callers (report generation, the
# Streamlit app) should look for a trained checkpoint by default. Mirrors
# vessel_infer.DEFAULT_WEIGHTS_PATH / optic_disc_infer.DEFAULT_WEIGHTS_PATH,
# even though (unlike those two) there's no classical fallback here -- a
# missing checkpoint just means detection is unavailable.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_WEIGHTS_PATH = os.path.join(_PROJECT_ROOT, "checkpoints", "dr_efficientnet_b0.pth")


def load_model(weights_path: str, device: str = "cpu") -> torch.nn.Module:
    """Build the architecture and load trained weights, ready for inference."""
    model = build_model(pretrained=False)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict(model: torch.nn.Module, image: np.ndarray, device: str = "cpu") -> dict:
    """Run inference on a single fundus photo.

    `image` is a BGR array as returned by cv2.imread, matching quality.py and
    enhance.py's convention elsewhere in this pipeline.
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = build_transforms(train=False)
    tensor = transform(rgb).unsqueeze(0).to(device)

    logits = model(tensor)
    probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    class_idx = int(probabilities.argmax())

    return {
        "label": SEVERITY_LABELS[class_idx],
        "probability": float(probabilities[class_idx]),
        "probabilities": probabilities.tolist(),
        "class_idx": class_idx,
    }
