"""Phase 5 (hybrid stage): vessel refinement model architecture.

A shallow U-Net that takes the CLAHE-enhanced green channel and the raw
Frangi vesselness response (see `vessels.compute_frangi_response()`) as a
2-channel input and learns to refine them into a cleaner vessel mask —
bridging broken thin-vessel segments and suppressing lesion/noise false
positives that a fixed threshold can't tell apart from real vessels.

Only 2 encoder levels deep (not 4-5 like a typical U-Net) to keep spatial
resolution high, since thin vessels are only a few pixels wide even at
VESSEL_WORKING_WIDTH and would be lost to more downsampling. That leaves the
bottleneck's receptive field too small on its own to bridge a broken vessel
segment spanning tens of pixels, so the bottleneck uses a stack of dilated
convolutions (dilation 2, 4, 8) instead of more pooling depth to widen the
receptive field without losing resolution -- the standard alternative to
"just add more layers" for this exact tradeoff.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

IN_CHANNELS = 2  # enhanced green channel + Frangi vesselness response
OUT_CHANNELS = 1  # single-channel vessel probability logits


class _ConvBlock(nn.Module):
    """Two 3x3 convs (with optional dilation) + BatchNorm + ReLU each."""

    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ShallowDilatedUNet(nn.Module):
    """2-level encoder/decoder U-Net with a dilated-convolution bottleneck.

    Fully convolutional (no fixed-size dense layers), so it accepts any
    spatial size -- trained on fixed-size patches, run at inference on the
    full VESSEL_WORKING_WIDTH-canonicalized image directly.
    """

    def __init__(self, in_channels: int = IN_CHANNELS, out_channels: int = OUT_CHANNELS):
        super().__init__()
        self.enc1 = _ConvBlock(in_channels, 32)
        self.enc2 = _ConvBlock(32, 64)
        self.pool = nn.MaxPool2d(2)

        # Dilated bottleneck: widens the receptive field via dilation rather
        # than further pooling depth -- see module docstring.
        self.bottleneck = nn.Sequential(
            _ConvBlock(64, 128, dilation=2),
            _ConvBlock(128, 128, dilation=4),
            _ConvBlock(128, 128, dilation=8),
        )

        # Channel-reduction convs; spatial upsampling is done in forward()
        # via interpolation to the *exact* skip-connection size (see below)
        # rather than a fixed-stride ConvTranspose2d, since VESSEL_WORKING_WIDTH
        # canonicalization preserves aspect ratio and commonly produces odd
        # heights (e.g. 1005px) that a stride-2 transpose conv can't exactly
        # invert after two poolings, which would break the skip concat.
        self.up2 = nn.Conv2d(128, 64, kernel_size=1)
        self.dec2 = _ConvBlock(64 + 64, 64)
        self.up1 = nn.Conv2d(64, 32, kernel_size=1)
        self.dec1 = _ConvBlock(32 + 32, 32)

        self.head = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))

        b = self.bottleneck(self.pool(e2))

        d2 = F.interpolate(self.up2(b), size=e2.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(self.up1(d2), size=e1.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        # Raw logits -- sigmoid is applied by the loss (training) or the
        # caller (inference), not here, matching how detection/model.py
        # returns raw class logits rather than pre-softmaxed probabilities.
        return self.head(d1)


def build_vessel_model() -> nn.Module:
    """Construct the vessel-refinement U-Net. No pretrained-weights option
    (unlike detection/model.py's EfficientNet-B0) -- this architecture is
    small and task-specific enough to train from scratch on the pooled
    DRIVE/STARE/CHASE_DB1 patches, with random-crop + flip augmentation
    compensating for the small (~68 image) labeled dataset.
    """
    return ShallowDilatedUNet()
