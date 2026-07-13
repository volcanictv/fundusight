"""Phase 6 (Stage 6.0): train the coarse full-frame optic disc locator.

Run with (from the project root):

    .venv\\Scripts\\python.exe src\\segmentation\\disc_locator_train.py --epochs 60

Trains DiscLocatorNet to regress the disc bounding box from a downscaled
WHOLE fundus frame, on REFUGE2's pooled/re-stratified split (the same split
and seed optic_disc_train.py uses, so the two models never disagree about
which images are held out). Saves the best checkpoint by validation HIT RATE
to --output, then evaluates once on the held-out test split.

Model selection is on HIT RATE, tie-broken by median center error -- not on
the regression loss. The loss is a proxy; what the pipeline actually needs
from this model is a center that lands INSIDE the true disc, since that
center is what crop_disc_roi() is handed. A model with slightly worse mean L1
but more centers inside the disc is strictly the better model for this job,
and selecting on loss would happily pick the other one. This mirrors
optic_disc_train.py selecting on mean rim/cup Dice rather than on its own loss
value.

THE TIE-BREAK IS NOT COSMETIC. Validation hit rate SATURATES at 1.000 within
about five epochs (the val split is in-domain REFUGE2, and a hit is scored
against the ground-truth BOX, which is generous). A plain `hit_rate > best`
rule therefore stops saving after the first epoch that reaches 1.0 and locks in
whatever weights happened to get there first -- even when a later epoch is
measurably better positioned (observed directly: epoch 5 saved at median error
0.0260 while epoch 6 reached 0.0219 and was discarded). Selecting on a
saturated metric is selection by coin flip. Ranking by (hit_rate,
-median_center_error) keeps the metric that matters primary while letting a
continuous, non-saturating one break the ties it cannot.
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
from src.segmentation.disc_locator_dataset import DiscLocatorDataset
from src.segmentation.disc_locator_model import LOCATOR_INPUT_SIZE, build_disc_locator_model
from src.segmentation.optic_disc_dataset import build_pooled_pairs, split_pooled_pairs


def parse_args():
    parser = argparse.ArgumentParser(description="Train the coarse full-frame optic disc locator.")
    parser.add_argument("--refuge-root", default=os.path.join(PROJECT_ROOT, "REFUGE2"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--input-size", type=int, default=LOCATOR_INPUT_SIZE)
    parser.add_argument("--lr", type=float, default=1e-3)
    # 2, not 4: each worker holds a decoded full-resolution REFUGE2 frame (up
    # to 2056x2124) before it is downscaled. Four of those plus a persistent
    # val pool was enough to exhaust memory on Windows and kill the run without
    # a traceback -- the same hazard optic_disc_train.py documents.
    parser.add_argument("--num-workers", type=int, default=2)
    # Same seed as optic_disc_train.py's default, on purpose: both models then
    # draw the SAME train/val/test partition of the same 1200 pooled images, so
    # the locator can never be trained on an image the disc/cup U-Net holds out
    # (or vice versa) and the end-to-end pipeline evaluation stays honest.
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=os.path.join(PROJECT_ROOT, "checkpoints", "disc_locator.pth"))
    return parser.parse_args()


def _hit_rate(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Fraction of predicted centers that fall inside the ground-truth disc
    BOX. This is the decision the pipeline actually makes -- crop_disc_roi()
    centers the ONH crop on this point, so a center inside the disc yields a
    usable crop and one outside it yields a crop of the wrong anatomy.

    The GT box (rather than the exact disc ellipse) is a slightly generous
    stand-in here because the dataset carries only the box; the honest,
    pixel-exact version of this measurement is done against real disc masks in
    scripts/evaluate_disc_locator.py, which is the number worth quoting.
    """
    pred_cx, pred_cy = predictions[:, 0], predictions[:, 1]
    tgt_cx, tgt_cy, tgt_w, tgt_h = targets[:, 0], targets[:, 1], targets[:, 2], targets[:, 3]
    inside = (
        (pred_cx >= tgt_cx - tgt_w / 2)
        & (pred_cx <= tgt_cx + tgt_w / 2)
        & (pred_cy >= tgt_cy - tgt_h / 2)
        & (pred_cy <= tgt_cy + tgt_h / 2)
    )
    return float(inside.float().mean())


def run_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    progress = tqdm(loader, desc="train", leave=False)
    for images, boxes in progress:
        images, boxes = images.to(device), boxes.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), boxes)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        progress.set_postfix(loss=f"{loss.item():.5f}")
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_predictions, all_targets = [], []
    for images, boxes in tqdm(loader, desc="eval", leave=False):
        all_predictions.append(model(images.to(device)).cpu())
        all_targets.append(boxes)
    predictions, targets = torch.cat(all_predictions), torch.cat(all_targets)

    center_error = torch.hypot(predictions[:, 0] - targets[:, 0], predictions[:, 1] - targets[:, 1])
    return {
        "hit_rate": _hit_rate(predictions, targets),
        # Center error as a fraction of frame width -- directly comparable to
        # _EXPECTED_DISC_DIAMETER_FRACTION (0.12) in optic_disc.py, so "is the
        # error smaller than a disc" can be read straight off it.
        "median_center_error": float(center_error.median()),
        "p90_center_error": float(torch.quantile(center_error, 0.9)),
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    pooled = build_pooled_pairs(args.refuge_root)
    train_pairs, valid_pairs, test_pairs = split_pooled_pairs(pooled, seed=args.seed)
    print(f"Pooled {len(pooled)} labeled pairs, re-split: train={len(train_pairs)}  val={len(valid_pairs)}  test={len(test_pairs)}")

    train_ds = DiscLocatorDataset(train_pairs, input_size=args.input_size, train=True)
    val_ds = DiscLocatorDataset(valid_pairs, input_size=args.input_size, train=False)
    test_ds = DiscLocatorDataset(test_pairs, input_size=args.input_size, train=False)

    loader_kwargs = dict(num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    # Not persistent -- iterated once, at the end. See optic_disc_train.py's
    # note on Windows worker pools exhausting the paging file.
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_disc_locator_model().to(device)
    # SmoothL1 rather than plain MSE: MSE's squared penalty lets a handful of
    # badly-placed frames dominate the gradient, and the disc's WIDTH/HEIGHT
    # outputs have a much smaller dynamic range than its CENTER coordinates, so
    # a squared loss quietly under-weights the size outputs relative to
    # position. SmoothL1 is the standard choice for box regression for exactly
    # these reasons.
    criterion = nn.SmoothL1Loss(beta=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # Steps on median center error (minimise), NOT on hit rate: hit rate pins at
    # 1.0 within a few epochs, which the scheduler would read as a permanent
    # plateau and respond to by decaying the learning rate to nothing while the
    # model was still genuinely improving its precision.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    # Lexicographic: maximise hit rate, then MINIMISE median center error (hence
    # the negation). See the module docstring -- hit rate saturates at 1.0, so
    # without the tie-break this silently freezes the checkpoint at the first
    # epoch to reach it.
    best_key = (-1.0, -float("inf"))

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step(val_metrics["median_center_error"])

        print(
            f"epoch {epoch:>3}/{args.epochs}  train_loss={train_loss:.5f}  "
            f"val_hit_rate={val_metrics['hit_rate']:.4f}  "
            f"val_median_err={val_metrics['median_center_error']:.4f}  "
            f"val_p90_err={val_metrics['p90_center_error']:.4f}"
        )

        key = (val_metrics["hit_rate"], -val_metrics["median_center_error"])
        if key > best_key:
            best_key = key
            torch.save(model.state_dict(), args.output)
            print(
                f"  -> new best (val_hit_rate={key[0]:.4f}, val_median_err={-key[1]:.4f}), saved to {args.output}"
            )

    print("\nLoading best checkpoint for final held-out test evaluation...")
    model.load_state_dict(torch.load(args.output, map_location=device))
    del train_loader, val_loader
    test_metrics = evaluate(model, test_loader, device)

    print(
        f"\nTest set (held out, never used for training/model selection): "
        f"hit_rate={test_metrics['hit_rate']:.4f}  "
        f"median_center_error={test_metrics['median_center_error']:.4f}  "
        f"p90_center_error={test_metrics['p90_center_error']:.4f}"
    )


if __name__ == "__main__":
    main()
