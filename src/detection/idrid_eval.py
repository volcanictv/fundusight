"""Phase 7: cross-dataset DR validation — evaluation only, no training.

Runs the existing APTOS-trained DR checkpoint (checkpoints/dr_efficientnet_b0.pth)
against IDRiD, a completely separate DR dataset the model has never seen, to
demonstrate whether its accuracy/AUC/kappa hold up outside APTOS -- a
single-dataset result is weak evidence on its own. Run with (from the
project root):

    .venv\\Scripts\\python.exe src\\detection\\idrid_eval.py

Reuses train.py's evaluate() directly (no metric-computation duplication)
and dataset.py's build_transforms() (same preprocessing the model was
trained on). There's no training loop here and therefore no train/val/test
split of IDRiD itself -- the whole labeled set (455 images) is evaluated as
one held-out cross-dataset test, since none of it was used for training or
model selection.
"""

import argparse
import os
import sys

import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
from src.detection.dataset import build_transforms
from src.detection.idrid_dataset import IDRiDDataset
from src.detection.model import NUM_CLASSES, build_model
from src.detection.train import evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the APTOS-trained DR model against IDRiD.")
    parser.add_argument("--idrid-root", default=os.path.join(PROJECT_ROOT, "data", "IDRi"))
    parser.add_argument("--checkpoint", default=os.path.join(PROJECT_ROOT, "checkpoints", "dr_efficientnet_b0.pth"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    csv_path = os.path.join(args.idrid_root, "idrid_labels.csv")
    img_dir = os.path.join(args.idrid_root, "Imagenes", "Imagenes")
    dataset = IDRiDDataset(csv_path, img_dir, transform=build_transforms(train=False))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f"Loaded {len(dataset)} IDRiD images (none used during APTOS training)")

    model = build_model(num_classes=NUM_CLASSES, pretrained=False).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    metrics = evaluate(model, loader, device)

    print(
        f"\nIDRiD cross-dataset results (checkpoint: {args.checkpoint}): "
        f"accuracy={metrics['accuracy']:.4f}  auc={metrics['auc']:.4f}  kappa={metrics['kappa']:.4f}"
    )
    print("Confusion matrix (rows=true, cols=predicted):")
    print(confusion_matrix(metrics["labels"], metrics["preds"]))


if __name__ == "__main__":
    main()
