import torch

from src.segmentation.optic_disc_loss import DiceCELoss, multiclass_dice_per_class, soft_multiclass_dice_loss

_NUM_CLASSES = 3


def _target(size=32):
    # 0=background everywhere except a disc-rim ring and a cup blob inside it.
    target = torch.zeros(1, size, size, dtype=torch.long)
    target[0, 8:24, 8:24] = 1  # disc rim
    target[0, 13:19, 13:19] = 2  # cup
    return target


def _confident_logits(target: torch.Tensor) -> torch.Tensor:
    # Strongly confident one-hot-like logits matching `target` exactly.
    one_hot = torch.nn.functional.one_hot(target, num_classes=_NUM_CLASSES).permute(0, 3, 1, 2).float()
    return one_hot * 20 - 10


def test_loss_near_zero_for_perfect_prediction():
    target = _target()
    logits = _confident_logits(target)

    loss = DiceCELoss()(logits, target)

    assert loss.item() < 0.05


def test_loss_much_higher_for_random_prediction():
    torch.manual_seed(0)
    target = _target()

    perfect_loss = DiceCELoss()(_confident_logits(target), target).item()
    random_loss = DiceCELoss()(torch.randn(1, _NUM_CLASSES, 32, 32), target).item()

    assert random_loss > perfect_loss * 5


def test_multiclass_dice_per_class_near_one_for_perfect_prediction():
    target = _target()
    logits = _confident_logits(target)

    dice = multiclass_dice_per_class(logits, target, num_classes=_NUM_CLASSES)

    assert dice.shape == (_NUM_CLASSES,)
    assert torch.all(dice > 0.99)


def test_soft_multiclass_dice_loss_is_one_minus_mean_dice():
    target = _target()
    logits = torch.randn(1, _NUM_CLASSES, 32, 32)

    dice = multiclass_dice_per_class(logits, target, num_classes=_NUM_CLASSES)
    loss = soft_multiclass_dice_loss(logits, target, num_classes=_NUM_CLASSES)

    assert abs(loss.item() - (1.0 - dice.mean().item())) < 1e-6


def test_loss_gradients_are_finite():
    target = _target()
    logits = torch.randn(1, _NUM_CLASSES, 32, 32, requires_grad=True)

    loss = DiceCELoss()(logits, target)
    loss.backward()

    assert torch.isfinite(logits.grad).all()


def test_dice_ce_weight_controls_balance():
    torch.manual_seed(0)
    target = _target()
    logits = torch.randn(1, _NUM_CLASSES, 32, 32)

    dice_only = DiceCELoss(dice_weight=1.0, ce_weight=0.0)(logits, target).item()
    ce_only = DiceCELoss(dice_weight=0.0, ce_weight=1.0)(logits, target).item()
    balanced = DiceCELoss(dice_weight=0.5, ce_weight=0.5)(logits, target).item()

    assert abs(balanced - (dice_only + ce_only) / 2) < 1e-5
