"""Phase 6 (Stage 6.2): optic disc/cup segmentation model architecture.

A small U-Net that takes a 512x512 optic-nerve-head ROI crop (see
`optic_disc.crop_disc_roi()`) built from RGB + Lab(a, b) + HSV(H, S) -- 7
channels total -- and predicts a 3-class map (background / disc rim /
cup). Multiple color spaces are used together because the cup/disc
boundary is a pallor difference that's subtle in any single space alone;
combining RGB with Lab's chromaticity channels and HSV's hue/saturation
(deliberately excluding Lab's L and HSV's V, both redundant with the
brightness information RGB already carries) is standard practice in
classical optic-cup segmentation work (e.g. Cheng et al.'s superpixel
classification approach) and carries over well as extra input channels for
a learned model.

Unlike vessels.py's ShallowDilatedUNet, this uses a standard (not dilated)
3-level encoder/decoder: vessels needed dilation specifically to preserve
spatial resolution for thin, elongated structures a few pixels wide. Disc
and cup are compact, roughly circular blobs occupying a sizeable fraction
of the ROI -- they benefit from the wider receptive field and multi-scale
context a deeper pooling encoder gives, not from avoiding pooling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

IN_CHANNELS = 7  # RGB + Lab(a, b) + HSV(H, S) -- see module docstring
OUT_CHANNELS = 3  # background, disc rim, cup


class _ConvBlock(nn.Module):
    """Two 3x3 convs + BatchNorm + ReLU each -- same building block as
    vessels_model.py's _ConvBlock, minus the dilation option this
    architecture doesn't need.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class OpticDiscUNet(nn.Module):
    """3-level encoder/decoder U-Net, standard (non-dilated) pooling for a
    wide receptive field over the compact disc/cup shapes.

    Fully convolutional (no fixed-size dense layers), so it accepts any
    spatial size -- trained on DISC_ROI_WIDTH x DISC_ROI_WIDTH crops, but
    not hard-coded to that exact size.
    """

    def __init__(self, in_channels: int = IN_CHANNELS, out_channels: int = OUT_CHANNELS):
        super().__init__()
        self.enc1 = _ConvBlock(in_channels, 32)
        self.enc2 = _ConvBlock(32, 64)
        self.enc3 = _ConvBlock(64, 128)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = _ConvBlock(128, 256)

        # Channel-reduction convs; spatial upsampling uses interpolation to
        # the exact skip-connection size (not a fixed-stride ConvTranspose2d)
        # -- same reasoning as vessels_model.py: aspect-preserving resizes
        # elsewhere in the pipeline can produce odd spatial dimensions a
        # stride-2 transpose conv can't exactly invert.
        self.up3 = nn.Conv2d(256, 128, kernel_size=1)
        self.dec3 = _ConvBlock(128 + 128, 128)
        self.up2 = nn.Conv2d(128, 64, kernel_size=1)
        self.dec2 = _ConvBlock(64 + 64, 64)
        self.up1 = nn.Conv2d(64, 32, kernel_size=1)
        self.dec1 = _ConvBlock(32 + 32, 32)

        self.head = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        d3 = F.interpolate(self.up3(b), size=e3.shape[2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = F.interpolate(self.up2(d3), size=e2.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(self.up1(d2), size=e1.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        # Raw per-class logits -- softmax is applied by the loss (training)
        # or the caller (inference), not here, same convention as
        # vessels_model.py's sigmoid-outside-the-model choice.
        return self.head(d1)


def build_optic_disc_model() -> nn.Module:
    """Construct the disc/cup segmentation U-Net. No pretrained-weights
    option -- small and task-specific enough to train from scratch on
    REFUGE2's 400 training images, same reasoning as
    vessel_model.build_vessel_model().
    """
    return OpticDiscUNet()
