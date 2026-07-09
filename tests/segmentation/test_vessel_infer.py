import cv2
import numpy as np
import torch

from src.segmentation.vessel_infer import compute_biomarkers_hybrid, load_vessel_model, segment_vessels_hybrid
from src.segmentation.vessel_model import build_vessel_model
from src.segmentation.vessels import VESSEL_WORKING_WIDTH


def _fundus_image(size=700):
    image = np.zeros((size, size, 3), dtype=np.uint8)
    cv2.circle(image, (size // 2, size // 2), int(size * 0.45), (180, 180, 180), -1)
    return image


def test_segment_vessels_hybrid_returns_expected_shape_and_dtype():
    model = build_vessel_model()
    model.eval()

    mask = segment_vessels_hybrid(_fundus_image(), model)

    assert mask.shape == (VESSEL_WORKING_WIDTH, VESSEL_WORKING_WIDTH)
    assert mask.dtype == bool


def test_compute_biomarkers_hybrid_returns_same_keys_as_classical():
    model = build_vessel_model()
    model.eval()

    result = compute_biomarkers_hybrid(_fundus_image(), model)

    assert set(result.keys()) == {
        "vessel_density",
        "branch_count",
        "tortuosity",
        "average_width",
        "mask",
        "skeleton",
    }
    assert result["mask"].shape == (VESSEL_WORKING_WIDTH, VESSEL_WORKING_WIDTH)
    assert result["skeleton"].shape == (VESSEL_WORKING_WIDTH, VESSEL_WORKING_WIDTH)


def test_load_vessel_model_restores_matching_weights(tmp_path):
    original = build_vessel_model()
    weights_path = str(tmp_path / "vessel_unet.pth")
    torch.save(original.state_dict(), weights_path)

    loaded = load_vessel_model(weights_path, device="cpu")

    for (name, orig_param), (_, loaded_param) in zip(original.named_parameters(), loaded.named_parameters()):
        assert torch.equal(orig_param, loaded_param), f"mismatch in {name}"
    assert not loaded.training  # eval() was called
