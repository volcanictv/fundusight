import gzip
import os

import cv2
import numpy as np
import pytest

from src.segmentation.vessel_dataset import (
    VesselDataset,
    _chase_pairs,
    _drive_pairs,
    _load_mask,
    _stare_pairs,
    build_pairs,
    split_pairs,
)

_SIZE = 64  # small synthetic image size -- real pairing/loading logic under test, not Frangi quality


def _write_image(path, size=_SIZE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
    cv2.imwrite(path, image)
    return image


def _write_mask(path, size=_SIZE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[size // 4 : size // 2, size // 4 : size // 2] = 255

    if path.endswith(".gif"):
        # cv2 can't *write* GIF (no encoder), but real DRIVE .gif masks
        # decode as 3-channel -- write PNG bytes (which cv2 CAN encode) to
        # a .gif-named file to fake that shape; cv2.imread sniffs the real
        # file header rather than trusting the extension, so this reads
        # back correctly and exercises the same 3-channel code path.
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        ok, buf = cv2.imencode(".png", mask_3ch)
        assert ok
        with open(path, "wb") as f:
            f.write(buf.tobytes())
    else:
        cv2.imwrite(path, mask)
    return mask


def _write_gz_ppm(path, size=_SIZE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".ppm", image)
    assert ok
    with gzip.open(path, "wb") as f:
        f.write(buf.tobytes())
    return image


def _make_fake_drive(root, n=3):
    for i in range(n):
        _write_image(os.path.join(root, "training", "images", f"{i}_training.tif"))
        _write_mask(os.path.join(root, "training", "1st_manual", f"{i}_manual1.gif"))
    # A test-set image with NO matching vessel label -- mirrors the real
    # DRIVE download, where DRIVE/test/ has no usable ground truth.
    _write_image(os.path.join(root, "test", "images", "0_test.tif"))


def _make_fake_stare(root, n=3):
    for i in range(n):
        _write_gz_ppm(os.path.join(root, "stare-images", f"im{i:04d}.ppm.gz"))
        _write_gz_ppm(os.path.join(root, "labels-ah", f"im{i:04d}.ah.ppm.gz"))


def _make_fake_chase(root, n=3):
    for i in range(n):
        _write_image(os.path.join(root, "Images", f"Image_{i:02d}L.jpg"))
        _write_mask(os.path.join(root, "Masks", f"Image_{i:02d}L_1stHO.png"))


def test_drive_pairs_finds_training_only_not_unlabeled_test_set(tmp_path):
    _make_fake_drive(tmp_path)

    pairs = _drive_pairs(str(tmp_path))

    assert len(pairs) == 3
    for img_path, mask_path in pairs:
        assert "training" in img_path
        assert os.path.exists(img_path) and os.path.exists(mask_path)


def test_stare_pairs_finds_gz_pairs(tmp_path):
    _make_fake_stare(tmp_path)

    pairs = _stare_pairs(str(tmp_path))

    assert len(pairs) == 3
    for img_path, mask_path in pairs:
        assert img_path.endswith(".ppm.gz")
        assert mask_path.endswith(".ah.ppm.gz")


def test_chase_pairs_finds_matching_masks(tmp_path):
    _make_fake_chase(tmp_path)

    pairs = _chase_pairs(str(tmp_path))

    assert len(pairs) == 3
    for img_path, mask_path in pairs:
        assert mask_path.endswith("_1stHO.png")


def test_build_pairs_pools_all_three_with_source_labels(tmp_path):
    _make_fake_drive(tmp_path / "DRIVE")
    _make_fake_stare(tmp_path / "STARE")
    _make_fake_chase(tmp_path / "CHASE_DB1")

    pairs = build_pairs(str(tmp_path / "DRIVE"), str(tmp_path / "STARE"), str(tmp_path / "CHASE_DB1"))

    assert len(pairs) == 9
    sources = {s for _, _, s in pairs}
    assert sources == {"drive", "stare", "chase"}


def test_split_pairs_is_stratified_and_covers_everything():
    # 10 of each source, enough for a stratified split with all fractions non-trivial.
    pairs = [(f"a_img{i}", f"a_mask{i}", "a") for i in range(10)]
    pairs += [(f"b_img{i}", f"b_mask{i}", "b") for i in range(10)]

    train, valid, test = split_pairs(pairs, valid_frac=0.2, test_frac=0.2, seed=0)

    assert len(train) + len(valid) + len(test) == 20
    # No overlap between splits.
    all_ids = [p[0] for p in train] + [p[0] for p in valid] + [p[0] for p in test]
    assert len(all_ids) == len(set(all_ids))
    # Both sources represented in the held-out test split.
    test_sources = {s for _, _, s in test}
    assert test_sources == {"a", "b"}


def test_load_mask_binarizes_drive_style_3channel_gif(tmp_path):
    path = str(tmp_path / "mask.gif")
    _write_mask(path)

    mask = _load_mask(path)

    assert mask.ndim == 2
    assert set(np.unique(mask).tolist()) <= {0, 1}


def test_vessel_dataset_train_mode_returns_paired_cropped_patch(tmp_path):
    _make_fake_drive(tmp_path, n=1)
    pairs = [(p[0], p[1], "drive") for p in _drive_pairs(str(tmp_path))]
    cache_dir = str(tmp_path / "_frangi_cache")

    dataset = VesselDataset(pairs, cache_dir=cache_dir, train=True, patch_size=32)
    input_tensor, mask_tensor = dataset[0]

    assert input_tensor.shape == (2, 32, 32)
    assert mask_tensor.shape == (1, 32, 32)
    assert input_tensor.dtype == mask_tensor.dtype == __import__("torch").float32


def test_vessel_dataset_eval_mode_returns_full_uncropped_image(tmp_path):
    _make_fake_drive(tmp_path, n=1)
    pairs = [(p[0], p[1], "drive") for p in _drive_pairs(str(tmp_path))]
    cache_dir = str(tmp_path / "_frangi_cache")

    dataset = VesselDataset(pairs, cache_dir=cache_dir, train=False)
    input_tensor, mask_tensor = dataset[0]

    # Not cropped to a fixed patch size -- matches full-image inference.
    assert input_tensor.shape[1:] == mask_tensor.shape[1:]
    assert input_tensor.shape[1] > 32 and input_tensor.shape[2] > 32


def test_vessel_dataset_caches_frangi_features_to_disk(tmp_path):
    _make_fake_drive(tmp_path, n=1)
    pairs = [(p[0], p[1], "drive") for p in _drive_pairs(str(tmp_path))]
    cache_dir = str(tmp_path / "_frangi_cache")

    dataset = VesselDataset(pairs, cache_dir=cache_dir, train=False)
    dataset[0]

    cached_files = list(os.listdir(cache_dir))
    assert len(cached_files) == 1
    assert cached_files[0].endswith(".npz")
