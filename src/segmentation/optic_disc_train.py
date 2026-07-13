"""Phase 6 (Stage 6.2): train the optic disc/cup segmentation U-Net.

Trains OpticDiscUNet with a combined CrossEntropy + multi-class Dice loss.
Run with (from the project root):

    .venv\\Scripts\\python.exe src\\segmentation\\optic_disc_train.py --epochs 80

Saves the best checkpoint (by validation mean Dice over the disc-rim and
cup classes -- background excluded, see evaluate()) to --output, then
evaluates once on the held-out test split, which is never touched during
training or model selection -- same discipline as
src/detection/train.py and src/segmentation/vessel_train.py.

Trains on a POOLED, re-stratified split (optic_disc_dataset.
build_pooled_pairs()/split_pooled_pairs()), NOT REFUGE2's own official
train/val/test folders -- see optic_disc_dataset.py's module docstring for
why: REFUGE2's official split turned out to be a three-way camera/domain
split (each folder is a single uniform resolution/color-profile source,
confirmed by direct image inspection), not a random sample of one
population. Training on it as-is and evaluating on its own val/test left
the model's validation performance not predictive of test performance,
which broke post-hoc calibration in Phase 6's investigation (see
ROADMAP.md and scripts/calibrate_optic_disc_thresholds.py). Pooling and
re-splitting with stratification by original folder mixes all three
domains into each new split instead.

Don't copy vessel_train.py's 150-epoch default here: that compensated for
a tiny ~46-image pooled training set where each "epoch" was a single random
patch per image. Here an epoch is a real pass over ~840 pooled training
images, so a much lower epoch count (80, with ReduceLROnPlateau backing off
the learning rate as needed) is the appropriate default.
"""

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
from src.segmentation.optic_disc_dataset import (
    DISC_ROI_WIDTH,
    WORKING_CACHE_DIRNAME,
    OpticDiscDataset,
    build_pooled_pairs,
    build_working_cache,
    split_pooled_pairs,
)
from src.segmentation.optic_disc_loss import DiceCELoss, multiclass_dice_per_class
from src.segmentation.optic_disc_model import OUT_CHANNELS, build_optic_disc_model
from src.segmentation.riga_dataset import build_riga_pairs


def parse_args():
    parser = argparse.ArgumentParser(description="Train the REFUGE2-based optic disc/cup segmentation model.")
    parser.add_argument("--refuge-root", default=os.path.join(PROJECT_ROOT, "REFUGE2"))
    parser.add_argument("--epochs", type=int, default=80)
    # 8, NOT 16. At 512x512 with 7 input channels this U-Net peaks at ~11.6 GB of
    # GPU memory at batch 16 -- more than the 8 GB on the RTX 4060 Laptop this
    # repo trains on. It does not OOM cleanly: it spills into shared host memory
    # and thrashes, which makes the step **17x slower** (7.33 s/batch vs 0.42
    # s/batch at batch 8, measured) and turns a ~1.2 min epoch into a ~10 min one.
    #
    # This wasted several training runs, because the symptom is not an error --
    # it is a run that silently takes hours and gets killed before finishing an
    # epoch. If training feels inexplicably slow, measure the GPU step time on a
    # synthetic batch before optimising the data pipeline; the bottleneck here was
    # never data loading.
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--roi-width", type=int, default=DISC_ROI_WIDTH)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=os.path.join(PROJECT_ROOT, "checkpoints", "optic_disc_unet.pth"))
    parser.add_argument(
        "--include-riga",
        action="store_true",
        help="Pool RIGA's ~749 six-annotator-consensus images with REFUGE2 (adds 6 camera domains).",
    )
    parser.add_argument("--riga-root", default=os.path.join(PROJECT_ROOT, "data"))
    parser.add_argument("--riga-cache", default=os.path.join(PROJECT_ROOT, "data", "riga_masks"))
    return parser.parse_args()


def run_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    progress = tqdm(loader, desc="train", leave=False)
    for inputs, targets in progress:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        progress.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    # Unlike vessel_train.py's evaluate(), every OpticDiscDataset item
    # (train or eval mode) is already a fixed roi_width x roi_width crop --
    # no variable-sized full images here -- so eval can use a real batch
    # size instead of being forced to batch_size=1.
    model.eval()
    per_batch_class_dice = []
    for inputs, targets in tqdm(loader, desc="eval", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        per_batch_class_dice.append(multiclass_dice_per_class(logits, targets, num_classes=OUT_CHANNELS))
    class_dice = torch.stack(per_batch_class_dice).mean(dim=0)
    return {
        "dice_background": float(class_dice[0]),
        "dice_rim": float(class_dice[1]),
        "dice_cup": float(class_dice[2]),
        # Checkpoint/scheduler metric: background is trivially easy (it's
        # most of the ROI even after cropping) and would dominate a
        # 3-class average, masking the signal that actually matters --
        # standard REFUGE evaluation practice also reports OD/OC Dice
        # separately rather than folding background in.
        "mean_dice_rim_cup": float(class_dice[1:].mean()),
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # REFUGE2 is split FIRST and on its own, exactly as it always has been (same
    # 1200 pairs, same seed) -- so its train/val/test assignment is byte-for-byte
    # what every previous REFUGE2-only checkpoint was trained and scored against.
    #
    # This is load-bearing for the comparison, not a stylistic choice. Pooling
    # REFUGE2 + RIGA and THEN splitting reshuffles REFUGE2's assignment, which
    # would put images from the OLD model's test split into the NEW model's
    # training set. The two checkpoints could then not be compared at all: the
    # new one would score better on a test set it had partly memorised, and the
    # improvement would be an artifact. Splitting each dataset independently and
    # concatenating keeps REFUGE2's 180 test images clean for BOTH models, which
    # is the only way "did RIGA help?" has a meaningful answer.
    refuge = build_pooled_pairs(args.refuge_root)
    train_pairs, valid_pairs, test_pairs = split_pooled_pairs(refuge, seed=args.seed)

    if args.include_riga:
        # RIGA adds six more camera/clinic domains to REFUGE2's three. This is
        # not about "more data" -- it is specifically about DOMAIN COVERAGE, and
        # the evidence for it is direct (scripts/evaluate_on_riga.py): the
        # REFUGE2-only model holds up on RIGA's MESSIDOR subset (mean |CDR error|
        # 0.065) but collapses on BinRushed1 (0.167) and Magrabia-male (0.180) --
        # at or above the 0.166 that six ophthalmologists disagree with EACH
        # OTHER by, i.e. useless on those cameras. It also carries a systematic
        # +0.0384 CDR bias out-of-domain that is absent in-domain.
        #
        # In-domain CDR accuracy is NOT the target here and must not be used to
        # judge this: it is already below the human noise floor and cannot
        # meaningfully improve (see DEEP_DIVE.md). Out-of-domain ROBUSTNESS is.
        riga = build_riga_pairs(args.riga_root, args.riga_cache)
        if not riga:
            sys.exit(f"--include-riga given but no masks in {args.riga_cache}. Build them with riga_dataset.build_riga_mask_cache().")

        riga_train, riga_valid, riga_test = split_pooled_pairs(riga, seed=args.seed)
        train_pairs += riga_train
        valid_pairs += riga_valid
        test_pairs += riga_test
        print(f"Pooled REFUGE2 ({len(refuge)}) + RIGA ({len(riga)}) = {len(refuge) + len(riga)} pairs, split independently")

    print(f"Splits: train={len(train_pairs)}  val={len(valid_pairs)}  test={len(test_pairs)}")

    # Pre-resize every frame once. Decoding native RIGA/REFUGE2 images every epoch
    # made the data pipeline the bottleneck AND forced DataLoader worker processes
    # that (holding full-resolution frames, under CUDA, on Windows) killed the run
    # silently and repeatedly. With the cache, loading is cheap enough to run
    # single-process. See build_working_cache().
    cache_root = os.path.join(PROJECT_ROOT, "data", WORKING_CACHE_DIRNAME)
    print(f"\nBuilding working-resolution cache in {cache_root} (idempotent; skipped if present)...")
    stats = build_working_cache(train_pairs + valid_pairs + test_pairs, cache_root)
    print(f"  {stats}")

    train_ds = OpticDiscDataset(train_pairs, roi_width=args.roi_width, train=True, working_cache_root=cache_root)
    val_ds = OpticDiscDataset(valid_pairs, roi_width=args.roi_width, train=False, working_cache_root=cache_root)
    test_ds = OpticDiscDataset(test_pairs, roi_width=args.roi_width, train=False, working_cache_root=cache_root)

    # persistent_workers keeps the worker pool alive across epochs instead
    # of respawning it every time -- num_workers=0 caused a severe I/O
    # bottleneck in an earlier phase, so this DataLoader configuration is a
    # hard requirement for train/val, not a tuning knob, same as
    # vessel_train.py. test_loader is deliberately NOT persistent: it's
    # only ever iterated once, at the very end (see the final evaluate()
    # call below) -- on Windows, DataLoader workers are spawned processes
    # that each re-import torch from scratch, and a smoke test showed 3
    # concurrent persistent worker pools (train + val + test, 4 workers
    # each) exhausting the system paging file. Freeing train/val's pools
    # (see `del` below) before spawning test's, and not keeping test's
    # pool alive at all, avoids ever having more than one pool's worth of
    # workers alive at a time.
    persistent_loader_kwargs = dict(num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **persistent_loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **persistent_loader_kwargs)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    model = build_optic_disc_model().to(device)
    criterion = DiceCELoss(dice_weight=args.dice_weight, ce_weight=1.0 - args.dice_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    best_metric = -1.0

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step(val_metrics["mean_dice_rim_cup"])

        print(
            f"epoch {epoch:>3}/{args.epochs}  train_loss={train_loss:.4f}  "
            f"val_dice_rim={val_metrics['dice_rim']:.4f}  val_dice_cup={val_metrics['dice_cup']:.4f}  "
            f"val_mean={val_metrics['mean_dice_rim_cup']:.4f}"
        )

        if val_metrics["mean_dice_rim_cup"] > best_metric:
            best_metric = val_metrics["mean_dice_rim_cup"]
            torch.save(model.state_dict(), args.output)
            print(f"  -> new best (mean_dice_rim_cup={best_metric:.4f}), saved to {args.output}")

    print("\nLoading best checkpoint for final held-out test evaluation...")
    model.load_state_dict(torch.load(args.output, map_location=device))

    # See the persistent_loader_kwargs comment above -- release train/val's
    # persistent worker processes before spawning test's.
    del train_loader, val_loader
    test_metrics = evaluate(model, test_loader, device)

    print(
        f"\nTest set (held out, never used for training/model selection): "
        f"dice_rim={test_metrics['dice_rim']:.4f}  dice_cup={test_metrics['dice_cup']:.4f}  "
        f"mean={test_metrics['mean_dice_rim_cup']:.4f}"
    )


if __name__ == "__main__":
    main()
