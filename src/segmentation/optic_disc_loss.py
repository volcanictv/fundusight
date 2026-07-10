"""Phase 6 (Stage 6.2): combined CrossEntropy + multi-class soft Dice loss.

Disc/cup segmentation is a 3-class (background/disc-rim/cup) pixel
classification problem over compact, roughly circular regions -- unlike
vessels.py's clDice loss, there's no thin tubular topology to preserve
here, so clDice (which specifically rewards matching *skeletons*, not just
area overlap) is the wrong tool. CrossEntropy gives a stable per-pixel
classification signal; multi-class Dice adds robustness to the class
imbalance between the large background region and the much smaller
disc-rim/cup regions, which cross-entropy alone tends to under-weight.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

_EPSILON = 1e-6


def multiclass_dice_per_class(logits: torch.Tensor, target: torch.Tensor, num_classes: int = 3) -> torch.Tensor:
    """Soft Dice score (not loss -- higher is better, range [0, 1]) for
    each class, averaged over the batch. Returns a `(num_classes,)` tensor
    -- used both by soft_multiclass_dice_loss() below and directly by
    optic_disc_train.py for validation reporting/checkpointing, where the
    background class needs to be excluded from the metric (it's trivially
    easy and would otherwise dominate a plain mean).
    """
    probs = F.softmax(logits, dim=1)
    target_onehot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()

    intersection = (probs * target_onehot).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))
    dice_per_class_per_sample = (2 * intersection + _EPSILON) / (union + _EPSILON)
    return dice_per_class_per_sample.mean(dim=0)


def soft_multiclass_dice_loss(logits: torch.Tensor, target: torch.Tensor, num_classes: int = 3) -> torch.Tensor:
    """`1 - mean(per-class Dice)` across all classes including background
    (background is excluded only at the evaluation/checkpointing level in
    optic_disc_train.py, not here -- during training it still contributes
    useful gradient signal).
    """
    return 1.0 - multiclass_dice_per_class(logits, target, num_classes).mean()


class DiceCELoss(nn.Module):
    """`dice_weight * dice_loss + ce_weight * cross_entropy_loss`. Defaults
    to an even split, same balance vessels.py's DiceClDiceLoss defaults to.
    """

    def __init__(self, dice_weight: float = 0.5, ce_weight: float = 0.5, num_classes: int = 3):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dice = soft_multiclass_dice_loss(logits, target, self.num_classes)
        ce = F.cross_entropy(logits, target)
        return self.dice_weight * dice + self.ce_weight * ce
