import os

import cv2
import numpy as np
import torch

from src.segmentation.optic_disc_dataset import (
    CLASS_BACKGROUND,
    CLASS_CUP,
    CLASS_DISC_RIM,
    OpticDiscDataset,
    _disc_bbox_from_mask,
    _refuge_pairs,
    _remap_mask_to_class_indices,
    build_pairs,
    build_pooled_pairs,
    split_pooled_pairs,
)

_SIZE = 200  # small synthetic image -- real pairing/loading/cropping logic under test


def _write_refuge_image(path, size=_SIZE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image = np.random.randint(50, 200, (size, size, 3), dtype=np.uint8)
    cv2.imwrite(path, image)
    return image


def _write_refuge_mask(path, size=_SIZE, disc_center=(120, 100), disc_radius=30, cup_radius=12):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mask = np.full((size, size), 255, dtype=np.uint8)  # background
    cv2.circle(mask, disc_center, disc_radius, 128, -1)  # disc rim
    cv2.circle(mask, disc_center, cup_radius, 0, -1)  # cup, carved out of the rim
    cv2.imwrite(path, mask)
    return mask


def _make_fake_refuge(root, n=2):
    # Real REFUGE2 layout: train/test masks are .bmp, val masks are .png --
    # confirmed by directory audit.
    for split, ext in [("train", ".bmp"), ("val", ".png"), ("test", ".bmp")]:
        for i in range(n):
            stem = f"{split[0]}{i:04d}"
            _write_refuge_image(os.path.join(root, split, "images", f"{stem}.jpg"))
            _write_refuge_mask(os.path.join(root, split, "mask", f"{stem}{ext}"))


def test_refuge_pairs_finds_bmp_and_png_masks(tmp_path):
    _make_fake_refuge(tmp_path, n=2)

    train_pairs = _refuge_pairs(str(tmp_path / "train"))
    val_pairs = _refuge_pairs(str(tmp_path / "val"))

    assert len(train_pairs) == 2
    assert len(val_pairs) == 2
    for _, mask_path in train_pairs:
        assert mask_path.endswith(".bmp")
    for _, mask_path in val_pairs:
        assert mask_path.endswith(".png")


def test_build_pairs_returns_official_train_val_test_split(tmp_path):
    _make_fake_refuge(tmp_path, n=2)

    pairs = build_pairs(str(tmp_path))

    assert set(pairs.keys()) == {"train", "val", "test"}
    assert len(pairs["train"]) == 2
    assert len(pairs["val"]) == 2
    assert len(pairs["test"]) == 2


def test_build_pooled_pairs_pools_all_three_with_source_labels(tmp_path):
    _make_fake_refuge(tmp_path, n=10)

    pooled = build_pooled_pairs(str(tmp_path))

    assert len(pooled) == 30
    sources = {s for _, _, s in pooled}
    assert sources == {"orig_train", "orig_val", "orig_test"}


def test_split_pooled_pairs_is_stratified_and_covers_everything(tmp_path):
    _make_fake_refuge(tmp_path, n=10)
    pooled = build_pooled_pairs(str(tmp_path))

    train, valid, test = split_pooled_pairs(pooled, valid_frac=0.2, test_frac=0.2, seed=0)

    assert len(train) + len(valid) + len(test) == 30
    # No overlap between splits, and the source tag is stripped -- plain
    # (image, mask) pairs matching build_pairs()'s per-split contract.
    all_paths = [p[0] for p in train] + [p[0] for p in valid] + [p[0] for p in test]
    assert len(all_paths) == len(set(all_paths))
    for pairs in (train, valid, test):
        for pair in pairs:
            assert len(pair) == 2


def test_remap_mask_to_class_indices():
    mask = np.array([[0, 128, 255, 77]], dtype=np.uint8)  # 77 = unexpected stray value

    remapped = _remap_mask_to_class_indices(mask)

    # Stray/unexpected pixel values fall back to background, not an
    # uninitialized or out-of-range class index.
    assert remapped.tolist() == [[CLASS_CUP, CLASS_DISC_RIM, CLASS_BACKGROUND, CLASS_BACKGROUND]]


def test_disc_bbox_from_mask_matches_known_region():
    mask = np.full((100, 100), CLASS_BACKGROUND, dtype=np.int64)
    mask[20:60, 30:70] = CLASS_DISC_RIM  # rows 20..59, cols 30..69

    info = _disc_bbox_from_mask(mask)

    assert abs(info["center_xy"][0] - 49.5) < 1.0  # (30 + 69) / 2
    assert abs(info["center_xy"][1] - 39.5) < 1.0  # (20 + 59) / 2
    assert abs(info["diameter_px"] - 40) < 1.0


def test_disc_bbox_from_mask_falls_back_when_empty():
    mask = np.full((100, 100), CLASS_BACKGROUND, dtype=np.int64)

    info = _disc_bbox_from_mask(mask)

    assert info["center_xy"] == (50.0, 50.0)
    assert info["diameter_px"] > 0


def test_optic_disc_dataset_eval_mode_shape_and_dtype(tmp_path):
    _make_fake_refuge(tmp_path, n=1)
    pairs = build_pairs(str(tmp_path))["train"]

    dataset = OpticDiscDataset(pairs, roi_width=64, train=False)
    input_tensor, target_tensor = dataset[0]

    assert input_tensor.shape == (7, 64, 64)
    assert target_tensor.shape == (64, 64)
    assert input_tensor.dtype == torch.float32
    assert target_tensor.dtype == torch.int64
    assert set(target_tensor.unique().tolist()) <= {CLASS_BACKGROUND, CLASS_DISC_RIM, CLASS_CUP}


def test_optic_disc_dataset_eval_mode_is_deterministic(tmp_path):
    _make_fake_refuge(tmp_path, n=1)
    pairs = build_pairs(str(tmp_path))["train"]
    dataset = OpticDiscDataset(pairs, roi_width=64, train=False)

    first_input, first_target = dataset[0]
    second_input, second_target = dataset[0]

    assert torch.equal(first_input, second_input)
    assert torch.equal(first_target, second_target)


def test_optic_disc_dataset_train_mode_varies_across_calls(tmp_path):
    _make_fake_refuge(tmp_path, n=1)
    pairs = build_pairs(str(tmp_path))["train"]
    dataset = OpticDiscDataset(pairs, roi_width=64, train=True)

    # Fresh RNG per __getitem__ call means the ROI jitter/flip/rotation
    # should differ across repeated reads of the same sample -- if this
    # ever came back identical every time, it would mean the jitter isn't
    # actually wired in (or worse, a worker-fork RNG-correlation bug like
    # the one vessel_dataset.py's fresh-rng-per-call convention avoids).
    samples = [dataset[0][0] for _ in range(5)]
    assert not all(torch.equal(samples[0], s) for s in samples[1:])
