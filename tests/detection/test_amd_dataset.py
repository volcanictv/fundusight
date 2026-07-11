import os

import cv2
import numpy as np
import torch

from src.detection.amd_dataset import AMDDataset, build_pairs, compute_class_weights, split_pairs
from src.detection.dataset import build_transforms

_SIZE = 50  # small synthetic image -- real pairing/loading/split logic under test


def _write_fake_image(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image = np.random.randint(0, 255, (_SIZE, _SIZE, 3), dtype=np.uint8)
    cv2.imwrite(path, image)


def _make_fake_adam(root, n_amd, n_non_amd):
    """Writes n_amd fake images under Training400/AMD/ and n_non_amd under
    Training400/Non-AMD/ -- mirrors the real ADAM layout where folder
    membership is the label, no separate CSV."""
    for i in range(n_amd):
        _write_fake_image(os.path.join(root, "Training400", "AMD", f"A{i:04d}.jpg"))
    for i in range(n_non_amd):
        _write_fake_image(os.path.join(root, "Training400", "Non-AMD", f"N{i:04d}.jpg"))


def test_build_pairs_reads_folder_membership_as_label(tmp_path):
    _make_fake_adam(tmp_path, n_amd=2, n_non_amd=3)

    pairs = build_pairs(str(tmp_path))

    assert len(pairs) == 5
    labels = sorted(label for _, label in pairs)
    assert labels == [0, 0, 0, 1, 1]
    for path, _label in pairs:
        assert os.path.isfile(path)


def test_split_pairs_is_stratified_and_covers_everything(tmp_path):
    # Same order of magnitude as ADAM's real ~78/22 imbalance.
    _make_fake_adam(tmp_path, n_amd=40, n_non_amd=160)
    pairs = build_pairs(str(tmp_path))
    overall_positive_rate = sum(label for _, label in pairs) / len(pairs)

    train, valid, test = split_pairs(pairs, valid_frac=0.2, test_frac=0.2, seed=0)

    # Full coverage, no leakage between splits.
    assert len(train) + len(valid) + len(test) == len(pairs)
    all_paths = [p[0] for p in train] + [p[0] for p in valid] + [p[0] for p in test]
    assert len(all_paths) == len(set(all_paths))

    # Stratifying on the label should keep each split's AMD-positive rate
    # close to the overall rate.
    for split in (train, valid, test):
        positive_rate = sum(label for _, label in split) / len(split)
        assert abs(positive_rate - overall_positive_rate) < 0.05


def test_amd_dataset_returns_image_and_label(tmp_path):
    _make_fake_adam(tmp_path, n_amd=1, n_non_amd=0)
    pairs = build_pairs(str(tmp_path))

    # No transform: raw plumbing check -- BGR->RGB decode, correct label.
    image, label = AMDDataset(pairs)[0]
    assert image.shape == (_SIZE, _SIZE, 3)
    assert label == 1

    # With the real transform pipeline (shared with the DR/glaucoma datasets,
    # already tested on their own) -- confirms AMDDataset threads it through
    # correctly end-to-end.
    tensor, label = AMDDataset(pairs, transform=build_transforms(train=False))[0]
    assert tensor.shape == (3, 224, 224)
    assert label == 1


def test_amd_dataset_missing_image_raises(tmp_path):
    pairs = [(str(tmp_path / "missing.jpg"), 0)]

    dataset = AMDDataset(pairs)

    try:
        dataset[0]
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_compute_class_weights_favors_minority_class():
    pairs = [("a", 0)] * 80 + [("b", 1)] * 20

    weights = compute_class_weights(pairs)

    assert isinstance(weights, torch.Tensor)
    assert weights[1] > weights[0]
