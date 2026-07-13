"""Phase 7: glaucoma classifier — local inference.

Mirrors src/detection/infer.py's shape exactly (load_model/predict contract),
swapped to the binary glaucoma checkpoint trained by glaucoma_train.py.

Unlike infer.py (DR) and amd_infer.py, predict() here CROPS TO THE OPTIC NERVE
HEAD first (src/detection/onh_crop.py), because that is what the shipped
checkpoint was trained on -- the full-image model this replaced was found to
attend to hemorrhages and edge artifacts rather than the disc. The crop is
applied via the same shared crop_to_onh() the training set was built with, so
the two cannot drift apart; feeding this checkpoint a full fundus photo would
be an out-of-distribution input and its probabilities would be meaningless.
"""

import os

import cv2
import numpy as np
import torch

from src.detection.dataset import build_transforms
from src.detection.model import build_model
from src.detection.onh_crop import crop_to_onh

# Matches glaucoma_train.py's --output default.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_WEIGHTS_PATH = os.path.join(_PROJECT_ROOT, "checkpoints", "glaucoma_efficientnet_b0.pth")

GLAUCOMA_LABELS = {0: "No Glaucoma Signs", 1: "Glaucoma Signs Present"}


def load_model(weights_path: str, device: str = "cpu") -> torch.nn.Module:
    """Build the architecture and load trained weights, ready for inference."""
    model = build_model(num_classes=2, pretrained=False)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def model_input(image: np.ndarray) -> np.ndarray:
    """The exact BGR array this checkpoint expects to classify: the ONH crop of
    a full fundus photo.

    Exposed separately from predict() so a caller that ALSO needs the model's
    input for something else -- report/pipeline.py, which must run Grad-CAM on
    it -- can obtain it without either re-deriving the crop or accidentally
    explaining an input the model never saw. A Grad-CAM computed on the full
    photo while the model classified a crop would be a heatmap of the wrong
    image, which is precisely the kind of silent mismatch that made the
    original attention problem hard to see.
    """
    crop, _disc_info = crop_to_onh(image)
    return crop


@torch.no_grad()
def predict_on_model_input(model: torch.nn.Module, onh_crop: np.ndarray, device: str = "cpu") -> dict:
    """predict() for a caller that already holds the ONH crop (see model_input()).

    Takes the CROP, not the full photo -- passing a full fundus photo here
    silently classifies an out-of-distribution image. Use predict() unless you
    specifically need to share one crop across several calls.
    """
    rgb = cv2.cvtColor(onh_crop, cv2.COLOR_BGR2RGB)
    transform = build_transforms(train=False)
    tensor = transform(rgb).unsqueeze(0).to(device)

    logits = model(tensor)
    probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    class_idx = int(probabilities.argmax())

    return {
        "label": GLAUCOMA_LABELS[class_idx],
        "probability": float(probabilities[class_idx]),
        "probabilities": probabilities.tolist(),
        "class_idx": class_idx,
    }


def predict(model: torch.nn.Module, image: np.ndarray, device: str = "cpu") -> dict:
    """Run inference on a single FULL fundus photo, cropping to the ONH first.

    `image` is a BGR array as returned by cv2.imread, matching infer.py's
    convention elsewhere in this pipeline -- callers hand over a whole fundus
    photo exactly as they do for the DR and AMD models, and the ONH crop this
    checkpoint requires is applied here rather than being every caller's
    responsibility to remember.
    """
    return predict_on_model_input(model, model_input(image), device)
