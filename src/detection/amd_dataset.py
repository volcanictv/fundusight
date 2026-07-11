"""Phase 7: AMD classifier — dataset and stratified split.

Reads ADAM (iChallenge-AMD) Training400: images are pre-sorted into
`AMD/` (label 1) and `Non-AMD/` (label 0) folders -- that folder membership
*is* the classification label, there's no separate label CSV (only
Fovea_location.xlsx, which is fovea coordinates for a different task).

Unlike REFUGE2 (three camera domains -- see glaucoma_dataset.py), ADAM's
400 images are a single domain, so there's no domain-leakage risk to guard
against here. But only a labeled *training* set ships publicly (no
official val/test), so -- same as the REFUGE2 pooling fix -- a split has
to be carved out ourselves, stratified on the AMD/Non-AMD label alone
(~89/311, a ~22%/78% imbalance).
"""

import glob
import os

import cv2
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

# (image_path, AMD label: 1=AMD, 0=Non-AMD)
Pair = tuple[str, int]


def build_pairs(adam_root: str) -> list[Pair]:
    training_dir = os.path.join(adam_root, "Training400")
    pairs = []
    for folder, label in (("AMD", 1), ("Non-AMD", 0)):
        for path in sorted(glob.glob(os.path.join(training_dir, folder, "*.jpg"))):
            pairs.append((path, label))
    return pairs


def split_pairs(
    pairs: list[Pair], valid_frac: float = 0.15, test_frac: float = 0.15, seed: int = 42
) -> tuple[list[Pair], list[Pair], list[Pair]]:
    """Two-step stratified train_test_split (train+valid vs test, then train
    vs valid), same relative_valid_frac correction as
    glaucoma_dataset.split_pairs() -- stratifying on the label alone here
    since ADAM has no domain dimension to compound it with.
    """
    labels = [label for _, label in pairs]
    train_valid, test = train_test_split(pairs, test_size=test_frac, stratify=labels, random_state=seed)

    train_valid_labels = [label for _, label in train_valid]
    relative_valid_frac = valid_frac / (1 - test_frac)
    train, valid = train_test_split(
        train_valid, test_size=relative_valid_frac, stratify=train_valid_labels, random_state=seed
    )

    return train, valid, test


class AMDDataset(Dataset):
    """Reads a list of (image_path, label) pairs -- see build_pairs()/split_pairs()."""

    def __init__(self, pairs: list[Pair], transform: transforms.Compose | None = None):
        self.pairs = pairs
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        path, label = self.pairs[idx]
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def compute_class_weights(pairs: list[Pair]) -> torch.Tensor:
    """Inverse-frequency weights for CrossEntropyLoss -- AMD is a ~78/22
    imbalanced binary label, same rationale as glaucoma_dataset's
    compute_class_weights().
    """
    labels = [label for _, label in pairs]
    counts = np.bincount(labels, minlength=2)
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(counts)
    return torch.tensor(weights, dtype=torch.float32)
