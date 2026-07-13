"""Phase 4: Explainability.

Class activation heatmaps over the DR classifier, so we can check whether it
looks at actual lesions or is shortcutting on something else — a common
failure mode is attending to the image border/vignette instead of retinal
tissue, which would make the accuracy numbers meaningless.
"""

import cv2
import numpy as np
import torch.nn as nn
from pytorch_grad_cam import EigenCAM, GradCAM, LayerCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.detection.dataset import IMAGE_SIZE, build_transforms

CAM_METHODS = {"gradcam": GradCAM, "eigencam": EigenCAM, "layercam": LayerCAM}


def _target_layer(model: nn.Module) -> list:
    """EfficientNet-B0's last conv block (1x1 projection to 1280 channels,
    right before global pooling) — the standard Grad-CAM target for this
    architecture, since it's the last layer where each spatial position
    still corresponds to a specific region of the input image.
    """
    return [model.features[-1]]


def compute_cam(model: nn.Module, image: np.ndarray, method: str = "gradcam", target_class: int | None = None) -> np.ndarray:
    """The raw class activation map for `image` (BGR, cv2 convention), as a
    float32 (IMAGE_SIZE, IMAGE_SIZE) array in [0, 1] — no overlay, no color.

    Split out from generate_cam() so attention can be MEASURED, not just
    looked at: quantifying how much of a model's attention falls on a given
    anatomical region (see scripts/compare_glaucoma_attention.py) needs the
    heatmap as numbers, and re-deriving it there would risk measuring a
    slightly different CAM than the one the report actually displays.
    """
    if method not in CAM_METHODS:
        raise ValueError(f"Unknown CAM method: {method!r}. Choose from {list(CAM_METHODS)}")

    model.eval()

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = build_transforms(train=False)
    input_tensor = transform(rgb).unsqueeze(0)

    targets = [ClassifierOutputTarget(target_class)] if target_class is not None else None

    cam_class = CAM_METHODS[method]
    with cam_class(model=model, target_layers=_target_layer(model)) as cam:
        return cam(input_tensor=input_tensor, targets=targets)[0]


def generate_cam(model: nn.Module, image: np.ndarray, method: str = "gradcam", target_class: int | None = None) -> np.ndarray:
    """Overlay a class activation heatmap on `image` (BGR, cv2 convention).

    `target_class` selects which class's activation to explain; defaults to
    the model's own top prediction. Returns a BGR uint8 overlay resized to
    IMAGE_SIZE x IMAGE_SIZE — what the model actually saw, not the original
    resolution, so the heatmap lines up exactly with the input.
    """
    grayscale_cam = compute_cam(model, image, method=method, target_class=target_class)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized_rgb = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32) / 255.0
    overlay_rgb = show_cam_on_image(resized_rgb, grayscale_cam, use_rgb=True)

    return cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
