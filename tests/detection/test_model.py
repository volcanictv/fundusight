import torch

from src.detection.model import NUM_CLASSES, build_model


def test_build_model_output_shape():
    # pretrained=False skips the ImageNet weight download - this test only
    # needs to check the architecture/shapes, not real features.
    model = build_model(pretrained=False)
    model.eval()
    dummy = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        output = model(dummy)

    assert output.shape == (2, NUM_CLASSES)
