"""Phase 6 (Stage 6.0): coarse full-frame optic disc localizer.

A small CNN that takes a DOWNSCALED WHOLE fundus photo and regresses the
optic disc's bounding box in full-frame coordinates. It exists to give
Stage 6.1's classical brightness+convergence search a second, independent
opinion -- specifically one that can still be right on the images where the
classical search lands on a hemorrhage or a specular reflection.

WHY THIS IS A SEPARATE MODEL AND NOT A HEAD ON OpticDiscUNet
------------------------------------------------------------
The tempting design is to bolt a bounding-box regression head onto
OpticDiscUNet's bottleneck and get localization "for free" as a multi-task
output. That design cannot work, for a reason worth stating plainly so it
isn't re-proposed:

OpticDiscUNet only ever sees a 512x512 ONH CROP that Stage 6.1 already
produced. Its bbox head would therefore only ever learn "where is the disc
inside a crop that already contains the disc" -- and at inference, on the
images that matter, Stage 6.1's crop DOESN'T contain the disc (it contains
a hemorrhage). The disc simply is not in the tensor, so no head reading that
tensor can point at it. Running such a head on a full frame instead would be
out-of-distribution inference: this repo has already been bitten by exactly
that (the glaucoma classifier trained on full images and fed ONH crops
returned confident, meaningless probabilities rather than erroring -- see
src/detection/onh_crop.py and CLAUDE.md). A confident bbox pointing at
nothing is strictly worse than an honest `confident=False`, because it
converts a caught failure back into a silent one.

So the arbitration model has to be trained on the same kind of image it will
be asked about: whole frames. That is what this is.

DESIGN
------
Deliberately small and low-resolution (_LOCATOR_INPUT_SIZE, default 256).
The task is "which part of the frame is the disc in", which needs macro
spatial layout -- the vessel arcade shape, the disc's position relative to
the FOV -- not fine texture. Low resolution is a feature, not a compromise:
it makes the network physically unable to key on a small bright lesion's
local texture, which is the failure mode being defended against, and it
keeps the model cheap enough to run on CPU inside the deployed app.

Outputs 4 numbers: [x_center, y_center, width, height], all as FRACTIONS of
the full frame ([0, 1]), so the prediction is resolution independent and can
be mapped onto any input photo's native size.

WHY THE POSITION HEAD IS A SOFT-ARGMAX AND NOT GAP -> MLP
---------------------------------------------------------
The obvious head -- global average pooling followed by an MLP that emits four
numbers -- was tried first and FAILED, in an instructive way: validation hit
rate collapsed (0.31 -> 0.017 -> 0.011 over three epochs) while the training
loss kept falling. The model was minimising the regression loss by predicting
a constant, roughly the mean disc position over the dataset, and ignoring the
image entirely.

That is not a tuning problem, it is the architecture. GAP averages every
spatial cell into one vector, which makes it (by construction) almost
translation-invariant -- it deliberately throws away WHERE a feature fired and
keeps only WHETHER it fired. Asking the resulting vector to report a
coordinate is asking it for the one thing it was designed to discard; the only
positional signal left is a faint border/padding artifact, so the optimiser
sensibly gives up and regresses to the mean.

The fix is to keep position in the spatial domain end to end:
  * POSITION is read out with a spatial softmax over a predicted 1-channel
    heatmap, then a soft-argmax (the expectation of the pixel coordinate under
    that distribution). This is the standard differentiable keypoint-
    localization readout, and it cannot regress to the mean unless the heatmap
    itself is flat.
  * SIZE (width/height) still uses GAP -> MLP, and that IS the right tool for
    it -- how big the disc is genuinely does not depend on where it is, so a
    translation-invariant pooled descriptor is exactly what you want.

Use GAP for "what/how much", never for "where".
"""

import torch
import torch.nn as nn

# Side length the full frame is squashed to before the network sees it. Square
# (not aspect-preserving) on purpose: the target bbox is expressed in relative
# [0, 1] coordinates, so an anisotropic resize maps exactly onto it, and the
# network never has to reason about letterbox padding.
LOCATOR_INPUT_SIZE = 256

IN_CHANNELS = 3  # plain BGR->RGB; see module docstring on why no fine texture
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
