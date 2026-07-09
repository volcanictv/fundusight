"""Phase 5 (hybrid stage): train the vessel-refinement U-Net.

Trains ShallowDilatedUNet on DRIVE + STARE + CHASE_DB1 (pooled, since none
is individually large enough alone -- ~68 labeled images total) with a
Dice + clDice loss. Run with (from the project root):

    .venv\\Scripts\\python.exe src\\segmentation\\vessel_train.py --epochs 150

Saves the best checkpoint (by validation clDice) to --output, then
evaluates once on the held-out test split, which is never touched during
training or model selection -- same discipline as
src/detection/train.py.

Each "epoch" here is a single random patch per training image (~46 patches
with the default split), much smaller than an epoch of Phase 3's ~2930
APTOS images -- hence the much higher default epoch count, and why the
random-crop+flip in vessel_dataset.py doing double duty as data
augmentation matters so much for a dataset this small.
"""

import argparse
import os
import sys
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
from src.segmentation.vessel_dataset import DEFAULT_PATCH_SIZE, VesselDataset, build_pairs, split_pairs
from src.segmentation.vessel_loss import DiceClDiceLoss, soft_cldice_loss, soft_dice_loss
from src.segmentation.vessel_model import build_vessel_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train the hybrid Frangi+U-Net vessel segmentation model.")
    parser.add_argument("--drive-root", default=os.path.join(PROJECT_ROOT, "DRIVE"))
    parser.add_argument("--stare-root", default=os.path.join(PROJECT_ROOT, "STARE"))
    parser.add_argument("--chase-root", default=os.path.join(PROJECT_ROOT, "CHASE_DB1"))
    parser.add_argument("--cache-dir", default=os.path.join(PROJECT_ROOT, ".frangi_cache"))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cldice-weight", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=os.path.join(PROJECT_ROOT, "checkpoints", "vessel_unet.pth"))
    return parser.parse_args()


def run_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    progress = tqdm(loader, desc="train", leave=False)
    for inputs, masks in progress:
        inputs, masks = inputs.to(device), masks.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        progress.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    # Full uncropped images (batch_size=1, see main()) -- matches what
    # inference actually runs on, not training patches.
    model.eval()
    dice_scores, cldice_scores = [], []
    for inputs, masks in tqdm(loader, desc="eval", leave=False):
        inputs, masks = inputs.to(device), masks.to(device)
        logits = model(inputs)
        dice_scores.append(1.0 - soft_dice_loss(logits, masks).item())
        cldice_scores.append(1.0 - soft_cldice_loss(logits, masks).item())
    return {"dice": float(np.mean(dice_scores)), "cldice": float(np.mean(cldice_scores))}


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    pairs = build_pairs(args.drive_root, args.stare_root, args.chase_root)
    counts = Counter(source for _, _, source in pairs)
    print(f"Found {len(pairs)} labeled pairs: {dict(counts)}")

    train_pairs, valid_pairs, test_pairs = split_pairs(pairs, seed=args.seed)
    print(f"Split: train={len(train_pairs)}  valid={len(valid_pairs)}  test={len(test_pairs)}")

    train_ds = VesselDataset(train_pairs, cache_dir=args.cache_dir, train=True, patch_size=args.patch_size)
    valid_ds = VesselDataset(valid_pairs, cache_dir=args.cache_dir, train=False)
    test_ds = VesselDataset(test_pairs, cache_dir=args.cache_dir, train=False)

    # persistent_workers keeps the worker pool alive across epochs instead
    # of respawning it every time (respawning was a large chunk of the I/O
    # bottleneck seen with num_workers=0 in Phase 3). Eval loaders use
    # batch_size=1 since full (uncropped) images vary in size and can't be
    # collated into a batch tensor together.
    common_loader_kwargs = dict(
        num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **common_loader_kwargs)
    valid_loader = DataLoader(valid_ds, batch_size=1, shuffle=False, **common_loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, **common_loader_kwargs)

    model = build_vessel_model().to(device)
    criterion = DiceClDiceLoss(cldice_weight=args.cldice_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    best_cldice = -1.0

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, valid_loader, device)
        scheduler.step(val_metrics["cldice"])

        print(
            f"epoch {epoch:>3}/{args.epochs}  train_loss={train_loss:.4f}  "
            f"val_dice={val_metrics['dice']:.4f}  val_cldice={val_metrics['cldice']:.4f}"
        )

        if val_metrics["cldice"] > best_cldice:
            best_cldice = val_metrics["cldice"]
            torch.save(model.state_dict(), args.output)
            print(f"  -> new best (cldice={best_cldice:.4f}), saved to {args.output}")

    print("\nLoading best checkpoint for final held-out test evaluation...")
    model.load_state_dict(torch.load(args.output, map_location=device))
    test_metrics = evaluate(model, test_loader, device)

    print(
        f"\nTest set (held out, never used for training/model selection): "
        f"dice={test_metrics['dice']:.4f}  cldice={test_metrics['cldice']:.4f}"
    )


if __name__ == "__main__":
    main()
