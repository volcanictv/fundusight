import torch

from src.segmentation.optic_disc_model import IN_CHANNELS, OUT_CHANNELS, OpticDiscUNet, build_optic_disc_model


def test_build_optic_disc_model_returns_optic_disc_unet():
    model = build_optic_disc_model()
    assert isinstance(model, OpticDiscUNet)


def test_forward_preserves_spatial_shape_for_roi_input():
    model = build_optic_disc_model()
    x = torch.randn(2, IN_CHANNELS, 512, 512)

    y = model(x)

    assert y.shape == (2, OUT_CHANNELS, 512, 512)


def test_forward_handles_odd_non_power_of_two_spatial_size():
    # crop_disc_roi() always resizes to a fixed DISC_ROI_WIDTH square, but
    # the model itself is fully convolutional and shouldn't assume that --
    # same odd-size robustness vessel_model.py's decoder needed.
    model = build_optic_disc_model()
    x = torch.randn(1, IN_CHANNELS, 401, 355)

    y = model(x)

    assert y.shape == (1, OUT_CHANNELS, 401, 355)


def test_model_is_lightweight():
    model = build_optic_disc_model()
    n_params = sum(p.numel() for p in model.parameters())
    # Well under EfficientNet-B0's 5.3M (src/detection/model.py) -- small
    # and task-specific, same reasoning as vessel_model.py.
    assert n_params < 5_000_000
