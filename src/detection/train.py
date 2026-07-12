"""Phase 3: DR Detection — training script.

Fine-tunes EfficientNet-B0 on APTOS DR severity labels, locally on a GPU. Run
with (from the project root):

    .venv\\Scripts\\python.exe src\\detection\\train.py --epochs 15

Saves the best checkpoint (by validation quadratic weighted kappa — the
official metric for this exact competition/dataset) to --output, then
evaluates once on the held-out test split, which is never touched during
training or model selection.
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
from src.detection.dataset import AptosDataset, build_transforms, compute_class_weights
from src.detection.model import NUM_CLASSES, build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune EfficientNet-B0 on APTOS DR severity labels.")
    parser.add_argument("--data-dir", default=os.path.join(PROJECT_ROOT, "APTOS 2019"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", default=os.path.join(PROJECT_ROOT, "checkpoints", "dr_efficientnet_b0.pth"))
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def run_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


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
        all_probs.extend(probs)

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    accuracy = accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
    except ValueError:
        # Can happen if a class is entirely absent from a small eval split.
        auc = float("nan")

    return {"accuracy": accuracy, "kappa": kappa, "auc": auc, "labels": all_labels, "preds": all_preds}


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_csv = os.path.join(args.data_dir, "train_1.csv")
    train_dir = os.path.join(args.data_dir, "train_images", "train_images")
    valid_csv = os.path.join(args.data_dir, "valid.csv")
    valid_dir = os.path.join(args.data_dir, "val_images", "val_images")
    test_csv = os.path.join(args.data_dir, "test.csv")
    test_dir = os.path.join(args.data_dir, "test_images", "test_images")

    train_ds = AptosDataset(train_csv, train_dir, transform=build_transforms(train=True))
    valid_ds = AptosDataset(valid_csv, valid_dir, transform=build_transforms(train=False))
    test_ds = AptosDataset(test_csv, test_dir, transform=build_transforms(train=False))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(num_classes=NUM_CLASSES, pretrained=True).to(device)

    class_weights = compute_class_weights(train_csv).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    best_kappa = -1.0

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, valid_loader, device)
        scheduler.step(val_metrics["kappa"])

        print(
            f"epoch {epoch:>2}/{args.epochs}  train_loss={train_loss:.4f}  "
            f"val_acc={val_metrics['accuracy']:.4f}  val_auc={val_metrics['auc']:.4f}  "
            f"val_kappa={val_metrics['kappa']:.4f}"
        )

        if val_metrics["kappa"] > best_kappa:
            best_kappa = val_metrics["kappa"]
            torch.save(model.state_dict(), args.output)
            print(f"  -> new best (kappa={best_kappa:.4f}), saved to {args.output}")

    print("\nLoading best checkpoint for final held-out test evaluation...")
    model.load_state_dict(torch.load(args.output, map_location=device))
    test_metrics = evaluate(model, test_loader, device)

    print(
        f"\nTest set (held out, never used for training/model selection): "
        f"accuracy={test_metrics['accuracy']:.4f}  auc={test_metrics['auc']:.4f}  "
        f"kappa={test_metrics['kappa']:.4f}"
    )
    print("Confusion matrix (rows=true, cols=predicted):")
    print(confusion_matrix(test_metrics["labels"], test_metrics["preds"]))


if __name__ == "__main__":
    main()
