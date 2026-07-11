"""Phase 7: AMD classifier — training script.

Fine-tunes EfficientNet-B0 (same architecture/backbone as the DR and
glaucoma classifiers, src/detection/model.py's build_model()) for binary
AMD classification. Run with (from the project root):

    .venv\\Scripts\\python.exe src\\detection\\amd_train.py --epochs 30

Trains on a label-stratified re-split of ADAM's Training400 set (the only
publicly-shipped labeled split -- src.detection.amd_dataset.build_pairs()/
split_pairs()), since no official val/test split exists.

Saves the best checkpoint (by validation AUC-ROC, same metric convention as
glaucoma_train.py) to --output, then evaluates once on the held-out test
split, which is never touched during training or model selection.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
from src.detection.amd_dataset import AMDDataset, build_pairs, compute_class_weights, split_pairs
from src.detection.dataset import build_transforms
from src.detection.model import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune EfficientNet-B0 for binary AMD classification.")
    parser.add_argument("--adam-root", default=os.path.join(PROJECT_ROOT, "ADAM"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=os.path.join(PROJECT_ROOT, "checkpoints", "amd_efficientnet_b0.pth"))
    return parser.parse_args()


def run_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    start = time.perf_counter()
    progress = tqdm(loader, desc="train", leave=False)
    for images, labels in progress:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        progress.set_postfix(loss=f"{loss.item():.4f}")
    elapsed_seconds = time.perf_counter() - start
    return total_loss / len(loader.dataset), elapsed_seconds


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        all_labels.extend(labels.numpy())
        all_preds.extend(probs.argmax(axis=1))
        all_probs.extend(probs[:, 1])

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        # Can happen if a class is entirely absent from a small eval split.
        auc = float("nan")

    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    return {
        "accuracy": accuracy,
        "auc": auc,
        "f1": f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "labels": all_labels,
        "preds": all_preds,
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    pairs = build_pairs(args.adam_root)
    train_pairs, valid_pairs, test_pairs = split_pairs(pairs, seed=args.seed)
    print(
        f"Pooled {len(pairs)} labeled pairs, re-split: "
        f"train={len(train_pairs)}  val={len(valid_pairs)}  test={len(test_pairs)}"
    )

    train_ds = AMDDataset(train_pairs, transform=build_transforms(train=True))
    valid_ds = AMDDataset(valid_pairs, transform=build_transforms(train=False))
    test_ds = AMDDataset(test_pairs, transform=build_transforms(train=False))

    # persistent_workers for train/val only, released (see `del` below)
    # before spawning test's non-persistent pool -- same Windows paging-file
    # exhaustion avoidance as glaucoma_train.py / optic_disc_train.py.
    persistent_loader_kwargs = dict(num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **persistent_loader_kwargs)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, **persistent_loader_kwargs)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    model = build_model(num_classes=2, pretrained=True).to(device)

    # Inverse-frequency class weights: AMD is ~78/22 imbalanced, same
    # rationale as glaucoma_train.py.
    class_weights = compute_class_weights(train_pairs).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    best_auc = -1.0

    for epoch in range(1, args.epochs + 1):
        train_loss, epoch_seconds = run_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, valid_loader, device)
        scheduler.step(val_metrics["auc"])

        print(
            f"epoch {epoch:>2}/{args.epochs}  train_loss={train_loss:.4f}  wall_clock={epoch_seconds:.1f}s  "
            f"val_acc={val_metrics['accuracy']:.4f}  val_auc={val_metrics['auc']:.4f}  "
            f"val_f1={val_metrics['f1']:.4f}  val_sens={val_metrics['sensitivity']:.4f}  "
            f"val_spec={val_metrics['specificity']:.4f}"
        )

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            torch.save(model.state_dict(), args.output)
            print(f"  -> new best (auc={best_auc:.4f}), saved to {args.output}")

    print("\nLoading best checkpoint for final held-out test evaluation...")
    model.load_state_dict(torch.load(args.output, map_location=device))

    # See the persistent_loader_kwargs comment above -- release train/val's
    # persistent worker processes before spawning test's.
    del train_loader, valid_loader
    test_metrics = evaluate(model, test_loader, device)

    print(
        f"\nTest set (held out, never used for training/model selection): "
        f"accuracy={test_metrics['accuracy']:.4f}  auc={test_metrics['auc']:.4f}  "
        f"f1={test_metrics['f1']:.4f}  sensitivity={test_metrics['sensitivity']:.4f}  "
        f"specificity={test_metrics['specificity']:.4f}"
    )
    print("Confusion matrix (rows=true, cols=predicted):")
    print(confusion_matrix(test_metrics["labels"], test_metrics["preds"]))


if __name__ == "__main__":
    main()
