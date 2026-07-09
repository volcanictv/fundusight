"""Phase 3: DR Detection — model architecture.

EfficientNet-B0 fine-tuned for 5-class diabetic retinopathy severity grading.
Chosen over ConvNeXt-Tiny for its smaller parameter count (5.3M vs 28M), which
matters here since the training set is only ~2930 images — fewer parameters
means less room to overfit.
"""

import torch.nn as nn
import torchvision.models as models

NUM_CLASSES = 5

# APTOS diagnosis column: 0-4 DR severity grade, standard clinical naming.
SEVERITY_LABELS = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR",
}


def build_model(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """EfficientNet-B0 with its 1000-class ImageNet head replaced for DR grading.

    `pretrained=False` skips the ImageNet weight download — used in tests and
    anywhere else that only needs to check the architecture/shapes, not real
    features, without requiring network access.
    """
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model
