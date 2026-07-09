import numpy as np
import pytest

from src.detection.dataset import IMAGE_SIZE
from src.detection.model import build_model
from src.explainability.gradcam import CAM_METHODS, generate_cam


@pytest.fixture
def model():
    # pretrained=False - CAM plumbing works the same regardless of whether
    # the weights are trained; no network access needed for this test.
    return build_model(pretrained=False)


@pytest.fixture
def image():
    return np.random.default_rng(0).integers(0, 255, size=(300, 300, 3), dtype=np.uint8)


@pytest.mark.parametrize("method", list(CAM_METHODS))
def test_generate_cam_returns_correct_shape_and_dtype(model, image, method):
    overlay = generate_cam(model, image, method=method)

    assert overlay.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
    assert overlay.dtype == np.uint8


def test_generate_cam_respects_target_class(model, image):
    overlay = generate_cam(model, image, method="gradcam", target_class=2)
    assert overlay.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)


def test_generate_cam_rejects_unknown_method(model, image):
    with pytest.raises(ValueError):
        generate_cam(model, image, method="not-a-real-method")
