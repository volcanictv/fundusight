import numpy as np
import torch

from src.detection.infer import predict
from src.detection.mc_dropout import (
    DEFAULT_MC_SAMPLES,
    enable_dropout,
    mc_dropout_probabilities,
    predicted_class_std,
)
from src.detection.model import build_model


def _model_and_tensor():
    model = build_model(pretrained=False)
    model.eval()
    tensor = torch.randn(1, 3, 224, 224)
    return model, tensor


def test_enable_dropout_flips_only_dropout_layers():
    model, _ = _model_and_tensor()
    enable_dropout(model)
    dropouts = [m for m in model.modules() if isinstance(m, torch.nn.Dropout)]
    assert dropouts, "EfficientNet-B0 head should retain a Dropout layer"
    assert all(m.training for m in dropouts)
    # BatchNorm (and the model as a whole) must stay in eval -- only dropout samples.
    assert all(not m.training for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d))


def test_mc_dropout_probabilities_shape_and_variation_and_restores_eval():
    model, tensor = _model_and_tensor()
    n = 16
    samples = mc_dropout_probabilities(model, tensor, n)

    assert samples.shape == (n, 5)
    np.testing.assert_allclose(samples.sum(axis=1), 1.0, atol=1e-4)
    # Dropout active => the passes are not all identical (this is the whole point).
    assert samples.std(axis=0).max() > 0
    # Model restored to eval afterwards, so a shared cached model is unaffected.
    assert not model.training
    assert all(not m.training for m in model.modules() if isinstance(m, torch.nn.Dropout))


def test_predicted_class_std_is_nonnegative_float():
    model, tensor = _model_and_tensor()
    std = predicted_class_std(model, tensor, class_idx=0, n_samples=16)
    assert isinstance(std, float)
    assert std >= 0.0


def test_predict_adds_uncertainty_only_when_requested():
    model = build_model(pretrained=False)
    model.eval()
    image = np.random.default_rng(0).integers(0, 255, size=(300, 300, 3), dtype=np.uint8)

    baseline = predict(model, image)
    assert "uncertainty_std" not in baseline  # backward compatible: mc_samples=0 default

    with_unc = predict(model, image, mc_samples=DEFAULT_MC_SAMPLES)
    assert "uncertainty_std" in with_unc
    assert isinstance(with_unc["uncertainty_std"], float)
    assert with_unc["uncertainty_std"] >= 0.0
    # The deterministic point prediction is untouched by MC sampling.
    assert with_unc["class_idx"] == baseline["class_idx"]
    assert with_unc["probability"] == baseline["probability"]
