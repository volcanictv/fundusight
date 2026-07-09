"""Phase 3: DR Detection — dataset and transforms.

Loads APTOS-format CSVs (id_code, diagnosis) and their matching PNGs. Kept
separate from enhance.py's preprocessing chain deliberately: the classifier
is trained on plain resize + ImageNet normalization, since the pretrained
backbone's early layers expect ImageNet-like input statistics, and the custom
color normalization in enhance.py shifts away from that (and visibly
amplifies noise) — a real risk to a first model's accuracy.
"""

import os

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(train: bool) -> transforms.Compose:
    """Resize + (for training) augment + normalize. Fundus photos have no
    fixed "up" orientation, so horizontal flips are a physically valid
    augmentation, unlike for e.g. photos of text or faces.
    """
    ops = [transforms.ToPILImage(), transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
    if train:
        ops += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        ]
    ops += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(ops)


class AptosDataset(Dataset):
    """Reads an APTOS-format CSV (id_code, diagnosis) and matching PNGs."""

    def __init__(self, csv_path: str, img_dir: str, transform: transforms.Compose | None = None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = os.path.join(self.img_dir, f"{row['id_code']}.png")
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image)

        return image, int(row["diagnosis"])


def compute_class_weights(csv_path: str) -> torch.Tensor:
    """Inverse-frequency weights for CrossEntropyLoss. DR severity classes
    are imbalanced (roughly half the training set is "No DR") — without
    weighting, the model could get a deceptively good accuracy by mostly
    predicting the majority class.
    """
    df = pd.read_csv(csv_path)
    counts = df["diagnosis"].value_counts().sort_index()
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(counts)
    return torch.tensor(weights.values, dtype=torch.float32)
