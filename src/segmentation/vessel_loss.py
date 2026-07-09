"""Phase 5 (hybrid stage): Dice + clDice combined loss.

Plain Dice rewards pixel/area overlap, which under-penalizes small
topological breaks in thin structures — a predicted vessel mask can score a
high Dice while still being disconnected in several places, which is
exactly today's classical-pipeline failure mode (broken thin-vessel
segments). clDice (Shit et al., "clDice - A Novel Topology-Preserving Loss
Function for Tubular Structure Segmentation", CVPR 2021) instead compares
*skeletons*: a prediction only scores well if its centerline actually lies
inside the true mask and vice versa, which directly penalizes breaks and
spurious branches. Combining both keeps Dice's stable area-overlap signal
while adding clDice's connectivity signal.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

_EPSILON = 1e-6

# Number of soft-erosion iterations used to build the soft skeleton --
# enough to thin a vessel a few pixels wide down to close to 1px without
# eroding away entirely; tuned for the vessel calibers _FRANGI_SIGMAS
# targets in vessels.py (roughly 3-15px at VESSEL_WORKING_WIDTH).
_SKELETON_ITERATIONS = 10


def _soft_erode(x: torch.Tensor) -> torch.Tensor:
    """Differentiable morphological erosion via min-pooling (implemented as
    -max_pool(-x)), decomposed into a 3x1 then 1x3 pass -- equivalent to a
    3x3 erosion but cheaper, and the standard clDice reference approach.
    """
    p1 = -F.max_pool2d(-x, kernel_size=(3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(-x, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.min(p1, p2)


def _soft_dilate(x: torch.Tensor) -> torch.Tensor:
    """Differentiable morphological dilation via max-pooling."""
    return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)


def _soft_open(x: torch.Tensor) -> torch.Tensor:
    """Erosion then dilation -- removes thin protrusions smaller than the
    structuring element, leaving the "core" of a shape.
    """
    return _soft_dilate(_soft_erode(x))


def soft_skeletonize(x: torch.Tensor, iterations: int = _SKELETON_ITERATIONS) -> torch.Tensor:
    """Iterative differentiable skeletonization (clDice reference
    algorithm): repeatedly erode the shape, and at each step keep whatever
    the opening operation *would have removed* (`relu(eroded - opened)`) --
    that's the centerline material at that erosion depth. Unioning these
    across iterations traces out the full skeleton as the shape is eroded
    down to nothing.
    """
    x1 = _soft_open(x)
    skeleton = F.relu(x - x1)
    for _ in range(iterations):
        x = _soft_erode(x)
        x1 = _soft_open(x)
        delta = F.relu(x - x1)
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def soft_dice_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Standard differentiable Dice loss on sigmoid probabilities."""
    probs = torch.sigmoid(pred)
    intersection = (probs * target).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * intersection + _EPSILON) / (union + _EPSILON)
    return 1.0 - dice.mean()


def soft_cldice_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """clDice loss: soft precision/sensitivity between each side's soft
    skeleton and the other side's mask.

    - precision: how much of the predicted skeleton actually lies inside
      the true mask (penalizes spurious predicted branches).
    - sensitivity: how much of the true skeleton is covered by the
      predicted mask (penalizes missed/broken vessel segments -- the
      failure mode this loss specifically targets).
    """
    probs = torch.sigmoid(pred)
    pred_skel = soft_skeletonize(probs)
    target_skel = soft_skeletonize(target)

    precision = (pred_skel * target).sum(dim=(1, 2, 3)) + _EPSILON
    precision = precision / (pred_skel.sum(dim=(1, 2, 3)) + _EPSILON)
    sensitivity = (target_skel * probs).sum(dim=(1, 2, 3)) + _EPSILON
    sensitivity = sensitivity / (target_skel.sum(dim=(1, 2, 3)) + _EPSILON)

    cldice = 2 * precision * sensitivity / (precision + sensitivity)
    return 1.0 - cldice.mean()


class DiceClDiceLoss(nn.Module):
    """`(1 - w) * dice_loss + w * cldice_loss`. `w` defaults to 0.5,
    matching the balance the original clDice paper reports.
    """

    def __init__(self, cldice_weight: float = 0.5):
        super().__init__()
        self.cldice_weight = cldice_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dice = soft_dice_loss(pred, target)
        cldice = soft_cldice_loss(pred, target)
        return (1.0 - self.cldice_weight) * dice + self.cldice_weight * cldice
