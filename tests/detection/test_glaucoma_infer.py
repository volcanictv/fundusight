import numpy as np

from src.detection.glaucoma_infer import model_input, predict, predict_on_model_input
from src.detection.model import build_model
from src.segmentation.optic_disc import DISC_ROI_WIDTH


def _model():
    # A randomly initialized (untrained) model - these only check the
    # input/output plumbing, not prediction accuracy.
    model = build_model(num_classes=2, pretrained=False)
    model.eval()
    return model


def _fundus():
    return np.random.default_rng(0).integers(0, 255, size=(300, 300, 3), dtype=np.uint8)


def test_predict_returns_expected_structure():
    result = predict(_model(), _fundus())

    assert set(result.keys()) == {"label", "probability", "probabilities", "class_idx"}
    assert isinstance(result["label"], str)
    assert result["class_idx"] in (0, 1)
    assert len(result["probabilities"]) == 2
    assert abs(sum(result["probabilities"]) - 1.0) < 1e-4


def test_model_input_returns_a_square_onh_crop():
    # This checkpoint classifies ONH crops, not full fundus photos -- see
    # src/detection/onh_crop.py.
    crop = model_input(_fundus())

    assert crop.shape == (DISC_ROI_WIDTH, DISC_ROI_WIDTH, 3)


def test_predict_matches_predict_on_model_input_with_the_same_crop():
    # The contract report/pipeline.py depends on: it calls model_input() itself
    # (so it can run Grad-CAM on the same array) and then
    # predict_on_model_input(). That path must produce exactly what the
    # convenience predict() produces, or the app and a standalone caller would
    # silently disagree.
    model, image = _model(), _fundus()

    direct = predict(model, image)
    via_crop = predict_on_model_input(model, model_input(image))

    assert direct["class_idx"] == via_crop["class_idx"]
    assert np.allclose(direct["probabilities"], via_crop["probabilities"], atol=1e-6)
