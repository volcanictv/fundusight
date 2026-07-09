import torch

from src.segmentation.vessel_model import IN_CHANNELS, OUT_CHANNELS, ShallowDilatedUNet, build_vessel_model


def test_build_vessel_model_returns_shallow_dilated_unet():
    model = build_vessel_model()
    assert isinstance(model, ShallowDilatedUNet)


def test_forward_preserves_spatial_shape_for_patch_input():
    model = build_vessel_model()
    x = torch.randn(2, IN_CHANNELS, 512, 512)

    y = model(x)

    assert y.shape == (2, OUT_CHANNELS, 512, 512)


def test_forward_handles_odd_non_power_of_two_spatial_size():
    # Aspect-ratio-preserving resize to VESSEL_WORKING_WIDTH commonly
    # produces an odd height (e.g. 1005px) -- a fixed-stride transpose-conv
    # decoder can't exactly invert two 2x poolings for a size like this,
    # which broke the skip-connection concat. Full-resolution inference (not
    # just fixed-size patches) depends on this working.
    model = build_vessel_model()
    x = torch.randn(1, IN_CHANNELS, 1400, 1005)

    y = model(x)

    assert y.shape == (1, OUT_CHANNELS, 1400, 1005)


def test_bottleneck_uses_dilated_convolutions():
    model = build_vessel_model()
    dilations = [
        module.dilation[0]
        for module in model.bottleneck.modules()
        if isinstance(module, torch.nn.Conv2d)
    ]
    # Six convs total (two per _ConvBlock, three blocks); dilation should
    # widen through the bottleneck rather than staying at 1 throughout --
    # that's the receptive-field mechanism replacing extra pooling depth.
    assert dilations == [2, 2, 4, 4, 8, 8]


def test_model_is_lightweight():
    model = build_vessel_model()
    n_params = sum(p.numel() for p in model.parameters())
    # Well under EfficientNet-B0's 5.3M (src/detection/model.py) -- this is
    # meant to be small given the ~68-image labeled training set.
    assert n_params < 2_000_000
