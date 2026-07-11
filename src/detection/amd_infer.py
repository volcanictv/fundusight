"""Phase 7: AMD classifier — local inference.

Mirrors src/detection/infer.py's shape exactly (load_model/predict contract),
swapped to the binary AMD checkpoint trained by amd_train.py.
"""

import os

import cv2
import numpy as np
import torch

from src.detection.dataset import build_transforms
from src.detection.model import build_model

# Matches amd_train.py's --output default.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_WEIGHTS_PATH = os.path.join(_PROJECT_ROOT, "checkpoints", "amd_efficientnet_b0.pth")

AMD_LABELS = {0: "No AMD Signs", 1: "AMD Signs Present"}


def load_model(weights_path: str, device: str = "cpu") -> torch.nn.Module:
    """Build the architecture and load trained weights, ready for inference."""
    model = build_model(num_classes=2, pretrained=False)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict(model: torch.nn.Module, image: np.ndarray, device: str = "cpu") -> dict:
    """Run inference on a single fundus photo.

    `image` is a BGR array as returned by cv2.imread, matching infer.py's
    convention elsewhere in this pipeline.
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = build_transforms(train=False)
    tensor = transform(rgb).unsqueeze(0).to(device)

    logits = model(tensor)
    probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    class_idx = int(probabilities.argmax())

    return {
        "label": AMD_LABELS[class_idx],
        "probability": float(probabilities[class_idx]),
        "probabilities": probabilities.tolist(),
        "class_idx": class_idx,
    }
