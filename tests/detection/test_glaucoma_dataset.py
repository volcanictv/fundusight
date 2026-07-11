import os

import cv2
import numpy as np
import pandas as pd
import torch

from src.detection.dataset import build_transforms
from src.detection.glaucoma_dataset import (
    GlaucomaDataset,
    build_pairs,
    compute_class_weights,
    domain_counts,
    split_pairs,
)

_SIZE = 50  # small synthetic image -- real pairing/loading/split logic under test


def _write_fake_image(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image = np.random.randint(0, 255, (_SIZE, _SIZE, 3), dtype=np.uint8)
    cv2.imwrite(path, image)


def _make_fake_refuge_with_labels(root, counts):
    """counts: dict of {(domain, label): n} -- writes n fake images per cell
    plus a matching glaucoma_labels_merged.csv, mirroring the real merged
    CSV's columns (filename, domain, glaucoma, source)."""
    rows = []
    for (domain, label), n in counts.items():
        for i in range(n):
            filename = f"{domain[0]}{label}_{i:04d}.jpg"
            _write_fake_image(os.path.join(root, domain, "images", filename))
            rows.append({"filename": filename, "domain": domain, "glaucoma": label, "source": "test"})
    pd.DataFrame(rows).to_csv(os.path.join(root, "glaucoma_labels_merged.csv"), index=False)


# Small counts make stratified rounding noisy (e.g. round(0.2 * 2) can land on
# either 0 or 1) -- these are large enough for the two chained stratified
# splits to land close to proportional, same order of magnitude as the real
# merged CSV's smallest domain/label cell (32).
_BALANCED_COUNTS = {
    ("train", 0): 80,
    ("train", 1): 20,
    ("val", 0): 80,
    ("val", 1): 20,
    ("test", 0): 80,
    ("test", 1): 20,
}


def test_build_pairs_reads_merged_csv_correctly(tmp_path):
    _make_fake_refuge_with_labels(tmp_path, {("train", 0): 2, ("val", 1): 3})

    pairs = build_pairs(str(tmp_path))

    assert len(pairs) == 5
    for path, label, domain in pairs:
        assert os.path.isfile(path)
        assert label in (0, 1)
        assert domain in ("train", "val")
    assert domain_counts(pairs) == {"train": 2, "val": 3}


def test_split_pairs_is_stratified_and_covers_everything(tmp_path):
    _make_fake_refuge_with_labels(tmp_path, _BALANCED_COUNTS)
    pairs = build_pairs(str(tmp_path))
    overall_positive_rate = sum(label for _, label, _ in pairs) / len(pairs)

    train, valid, test = split_pairs(pairs, valid_frac=0.2, test_frac=0.2, seed=0)

    # Full coverage, no leakage between splits.
    assert len(train) + len(valid) + len(test) == len(pairs)
    all_paths = [p[0] for p in train] + [p[0] for p in valid] + [p[0] for p in test]
    assert len(all_paths) == len(set(all_paths))

    # Every split still has all three domains represented -- the whole point
    # of stratifying on domain, mirroring optic_disc_dataset's Phase 6 fix.
    for split in (train, valid, test):
        assert domain_counts(split).keys() == {"train", "val", "test"}

    # Stratifying on the compound domain+label key should keep each split's
    # glaucoma-positive rate close to the overall rate -- a numeric check
    # beyond what optic_disc_dataset's own split test verifies, since this
    # compound-key stratification is new logic without existing precedent.
    for split in (train, valid, test):
        positive_rate = sum(label for _, label, _ in split) / len(split)
        assert abs(positive_rate - overall_positive_rate) < 0.05


def test_glaucoma_dataset_returns_image_and_label(tmp_path):
    _make_fake_refuge_with_labels(tmp_path, {("train", 1): 1})
    pairs = build_pairs(str(tmp_path))

    # No transform: raw plumbing check -- BGR->RGB decode, correct label.
    image, label = GlaucomaDataset(pairs)[0]
    assert image.shape == (_SIZE, _SIZE, 3)
    assert label == 1

    # With the real transform pipeline (shared with the DR dataset, already
    # tested on its own in tests/detection/test_dataset.py) -- confirms
    # GlaucomaDataset threads it through correctly end-to-end.
    tensor, label = GlaucomaDataset(pairs, transform=build_transforms(train=False))[0]
    assert tensor.shape == (3, 224, 224)
    assert label == 1


def test_glaucoma_dataset_missing_image_raises(tmp_path):
    pairs = [(str(tmp_path / "missing.jpg"), 0, "train")]

    dataset = GlaucomaDataset(pairs)

    try:
        dataset[0]
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_compute_class_weights_favors_minority_class():
    pairs = [("a", 0, "train")] * 90 + [("b", 1, "train")] * 10

    weights = compute_class_weights(pairs)

    assert isinstance(weights, torch.Tensor)
    assert weights[1] > weights[0]
