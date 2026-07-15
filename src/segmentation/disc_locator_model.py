"""Stage 6.0: coarse full-frame optic disc localizer.

A small CNN that regresses the disc's bounding box ([x, y, w, h] as fractions of
the frame) from a downscaled whole fundus photo. It's a second, independent
opinion for Stage 6.1's classical search, for the images where that search locks
onto a hemorrhage or a reflection.

Two non-obvious design choices, both from a failed first attempt (full write-up
in DEEP_DIVE.md, "Stage 6.0"): it's a separate model rather than a bbox head on
OpticDiscUNet (that U-Net only sees an ONH crop which, on the failing images,
doesn't contain the disc -- so a head reading it can't point at the disc), and
position is read out via soft-argmax over a heatmap rather than GAP->MLP (GAP is
translation-invariant and collapses to predicting the mean disc position; size
still uses GAP, where that invariance is what you want).
"""

import torch
import torch.nn as nn

# Side length the full frame is squashed to before the network sees it. Square
# (not aspect-preserving) on purpose: the target bbox is in relative [0, 1]
# coords, so an anisotropic resize maps straight onto it. Kept deliberately low
# so the net can't key on a lesion's fine local texture -- macro layout (arcade
# shape, disc position in the FOV) is all the task needs.
LOCATOR_INPUT_SIZE = 256

IN_CHANNELS = 3  # plain BGR->RGB
OUT_VALUES = 4  # [x_center, y_center, width, height], relative to the frame


def soft_argmax(scores: torch.Tensor) -> torch.Tensor:
    """Differentiable 2-D argmax: softmax an (N, H, W) score map into one
    probability distribution over locations, then take the EXPECTED cell
    coordinate under it. Returns (N, 2) as `[x, y]` fractions of the frame.

    Extracted as a standalone function (rather than inlined in forward) so the
    coordinate convention below can be unit-tested exactly, on a hand-built
    heatmap, instead of only being exercised through a randomly-initialised
    network -- where a near-uniform heatmap puts the readout at ~(0.5, 0.5) no
    matter what and hides any error in it.

    The `+ 0.5` is load-bearing. Cell i covers the band [i/w, (i+1)/w), so its
    CENTER is at (i + 0.5)/w. The tempting `i / (w - 1)` instead pins the first
    and last cells' centers to exactly 0.0 and 1.0 -- implying the grid's
    extreme cells sit ON the image border -- which biases every prediction
    outward by half a cell.
    """
    n, h, w = scores.shape
    probabilities = torch.softmax(scores.view(n, h * w), dim=1).view(n, h, w)

    xs = ((torch.arange(w, device=scores.device, dtype=scores.dtype) + 0.5) / w).view(1, 1, w)
    ys = ((torch.arange(h, device=scores.device, dtype=scores.dtype) + 0.5) / h).view(1, h, 1)

    x_center = (probabilities * xs).sum(dim=(1, 2))
    y_center = (probabilities * ys).sum(dim=(1, 2))
    return torch.stack([x_center, y_center], dim=1)


class _DownBlock(nn.Module):
    """Conv -> BatchNorm -> ReLU -> stride-2 conv. Halves the spatial size."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DiscLocatorNet(nn.Module):
    """256x256 whole-frame in, 4 relative bbox values out
    ([x_center, y_center, width, height], each a fraction of the frame).

    Two heads over one shared encoder -- see the module docstring:
      * `heatmap` -> spatial softmax -> soft-argmax  =>  position
      * global average pool -> MLP -> sigmoid        =>  size
    """

    def __init__(self, in_channels: int = IN_CHANNELS, out_values: int = OUT_VALUES):
        super().__init__()
        self.features = nn.Sequential(
            _DownBlock(in_channels, 16),  # 256 -> 128
            _DownBlock(16, 32),  # 128 -> 64
            _DownBlock(32, 64),  # 64  -> 32
            _DownBlock(64, 128),  # 32  -> 16
        )  # -> (128, 16, 16); stopping at 16x16 keeps the heatmap fine enough
        # that one cell is 16 input px, comfortably under a disc's ~30px width
        # at this scale -- an 8x8 grid would quantise the disc into a single
        # cell and cap achievable precision.

        # Position: a single-channel score map, read out by soft-argmax.
        self.heatmap = nn.Conv2d(128, 1, kernel_size=1)

        # Size: translation-invariant, so pooled features are appropriate here.
        self.size_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 2),
            # Sigmoid: a width/height is a fraction of the frame and so lives in
            # [0, 1] by construction. Letting the net emit a negative width and
            # relying on the loss to discourage it would permit nonsense
            # predictions on an out-of-distribution photo -- exactly the
            # situation this model is called in. Bounding it makes them
            # unrepresentable rather than merely penalised.
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        center = soft_argmax(self.heatmap_logits_from_features(features))  # position
        size = self.size_head(features)  # size
        return torch.cat([center, size], dim=1)

    def heatmap_logits_from_features(self, features: torch.Tensor) -> torch.Tensor:
        n, _c, h, w = features.shape
        return self.heatmap(features).view(n, h, w)

    def heatmap_logits(self, x: torch.Tensor) -> torch.Tensor:
        """The raw (N, H, W) position score map, exposed for debugging and for
        visualising WHERE the locator is looking -- the soft-argmax collapses
        it to a point, which hides a bimodal "torn between the disc and a
        lesion" heatmap that is exactly what you want to see when diagnosing a
        failure."""
        return self.heatmap_logits_from_features(self.features(x))


def build_disc_locator_model() -> nn.Module:
    """Construct the coarse full-frame disc locator. Trained from scratch --
    it is tiny and the task is geometric rather than semantic, so an ImageNet
    backbone buys little, same reasoning as build_optic_disc_model()."""
    return DiscLocatorNet()
