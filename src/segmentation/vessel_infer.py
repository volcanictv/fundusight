"""Phase 5 (hybrid stage): local inference with the trained vessel model.

Mirrors src/detection/infer.py's split from its dataset/model modules:
this is the only file in src/segmentation/ that imports torch for the
*hybrid* path, keeping vessels.py itself classical and torch-free (see its
module docstring) for anyone who only needs the classical baseline.
"""

import numpy as np
import torch

from src.segmentation import vessels
from src.segmentation.vessel_model import build_vessel_model

_MASK_THRESHOLD = 0.5


def load_vessel_model(weights_path: str, device: str = "cpu") -> torch.nn.Module:
    """Build the architecture and load trained weights, ready for inference."""
    model = build_vessel_model()
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def segment_vessels_hybrid(image: np.ndarray, model: torch.nn.Module, device: str = "cpu") -> np.ndarray:
    """Hybrid vessel mask: compute_frangi_response()'s two channels feed the
    trained U-Net directly (no hysteresis threshold on the Frangi response
    itself -- the model learns its own correction), then the model's output
    is thresholded at 0.5. Same signature/return contract as
    vessels.segment_vessels() (boolean mask at VESSEL_WORKING_WIDTH
    resolution, canonicalized internally) -- a drop-in swap once a trained
    checkpoint exists.
    """
    enhanced, vesselness = vessels.compute_frangi_response(image)
    input_arr = np.stack([enhanced, vesselness], axis=0)
    input_tensor = torch.from_numpy(input_arr).unsqueeze(0).to(device)

    logits = model(input_tensor)
    probs = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().numpy()

    return probs > _MASK_THRESHOLD


def compute_biomarkers_hybrid(image: np.ndarray, model: torch.nn.Module, device: str = "cpu") -> dict:
    """Same structure and return contract as vessels.compute_biomarkers(),
    swapping the classical hysteresis-threshold mask for the trained
    model's mask. Resizes once up front (mirroring compute_biomarkers()
    exactly) so the FOV mask stays shape-consistent with the vessel
    mask/skeleton it's paired with below.
    """
    working = vessels._resize_to_working_width(image)
    mask = segment_vessels_hybrid(working, model, device)
    skeleton = vessels.skeletonize_vessels(mask)
    fov = vessels._fov_mask(vessels.extract_vessel_channel(working))

    return {
        "vessel_density": vessels.vessel_density(mask, fov),
        "branch_count": vessels.branch_point_count(skeleton),
        "tortuosity": vessels.tortuosity(skeleton),
        "average_width": vessels.average_vessel_width(mask, skeleton),
        "mask": mask,
        "skeleton": skeleton,
    }
