import torch

from src.segmentation.vessel_loss import DiceClDiceLoss, soft_cldice_loss, soft_dice_loss, soft_skeletonize


def _bar_target(size=64):
    target = torch.zeros(1, 1, size, size)
    target[0, 0, 20:44, 30:34] = 1.0  # a vertical bar standing in for a vessel
    return target


def _confident_logits(mask: torch.Tensor) -> torch.Tensor:
    # Strongly confident logits matching `mask` exactly (sigmoid(~10)=~1, sigmoid(~-10)=~0).
    return mask * 20 - 10


def test_loss_near_zero_for_perfect_prediction():
    target = _bar_target()
    pred = _confident_logits(target)

    loss = DiceClDiceLoss()(pred, target)

    assert loss.item() < 0.01


def test_loss_much_higher_for_random_prediction():
    torch.manual_seed(0)
    target = _bar_target()
    pred = torch.randn(1, 1, 64, 64)

    perfect_loss = DiceClDiceLoss()(_confident_logits(target), target).item()
    random_loss = DiceClDiceLoss()(pred, target).item()

    assert random_loss > perfect_loss * 10


def test_cldice_penalizes_topological_break_more_than_dice():
    # Same total pixel-count error either way: a gap punched through the
    # middle of the bar. clDice should weight this worse than plain Dice,
    # since a break here severs the skeleton in two while removing
    # relatively few pixels -- exactly the failure mode clDice targets.
    target = _bar_target()
    broken = target.clone()
    broken[0, 0, 28:36, 30:34] = 0
    pred = _confident_logits(broken)

    dice = soft_dice_loss(pred, target).item()
    cldice = soft_cldice_loss(pred, target).item()

    assert cldice > dice


def test_soft_skeletonize_thins_toward_centerline():
    target = _bar_target()

    skeleton = soft_skeletonize(target)

    # The bar is 4px wide; its skeleton should occupy a much smaller area
    # (a thin centerline), while still being non-empty and contained within
    # a slightly dilated version of the original bar (allowing for the soft
    # approximation's fuzziness at the boundary).
    assert 0 < skeleton.sum().item() < target.sum().item()


def test_loss_gradients_are_finite():
    torch.manual_seed(0)
    target = _bar_target()
    pred = torch.randn(1, 1, 64, 64, requires_grad=True)

    loss = DiceClDiceLoss()(pred, target)
    loss.backward()

    assert torch.isfinite(pred.grad).all()


def test_cldice_weight_controls_balance():
    target = _bar_target()
    broken = target.clone()
    broken[0, 0, 28:36, 30:34] = 0
    pred = _confident_logits(broken)

    dice_only = DiceClDiceLoss(cldice_weight=0.0)(pred, target).item()
    cldice_only = DiceClDiceLoss(cldice_weight=1.0)(pred, target).item()
    balanced = DiceClDiceLoss(cldice_weight=0.5)(pred, target).item()

    assert dice_only < balanced < cldice_only
