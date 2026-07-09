import numpy as np

from src.detection.infer import predict
from src.detection.model import build_model


def test_predict_returns_expected_structure():
    # A randomly initialized (untrained) model - this only checks the
    # input/output plumbing, not prediction accuracy.
    model = build_model(pretrained=False)
    model.eval()
    image = np.random.default_rng(0).integers(0, 255, size=(300, 300, 3), dtype=np.uint8)

    result = predict(model, image)

    assert set(result.keys()) == {"label", "probability", "probabilities", "class_idx"}
    assert isinstance(result["label"], str)
    assert 0 <= result["class_idx"] < 5
    assert len(result["probabilities"]) == 5
    assert abs(sum(result["probabilities"]) - 1.0) < 1e-4
