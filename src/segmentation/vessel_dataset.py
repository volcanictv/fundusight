"""Phase 5 (hybrid stage): dataset loading for DRIVE/STARE/CHASE_DB1.

APTOS has no pixel-level vessel labels, so the hybrid U-Net is trained on
the three standard hand-labeled vessel segmentation benchmarks instead, each
with its own directory layout and file-naming convention:

- DRIVE: `DRIVE/training/images/*_training.tif` + `1st_manual/*_manual1.gif`.
  Only the training split has ground-truth vessel masks in this download —
  `DRIVE/test/` has images + FOV masks but no vessel labels, so it's not
  usable for supervised training/eval and is deliberately excluded.
- STARE: `STARE/stare-images/imXXXX.ppm.gz` + `labels-ah/imXXXX.ah.ppm.gz` —
  gzip-compressed PPM. Decompressed and decoded in memory on read (cheap
  enough per-image not to need a separate decompress-to-disk step).
- CHASE_DB1: `CHASE_DB1/Images/Image_NNL.jpg` (or `_NNR`) +
  `Masks/Image_NNL_1stHO.png`.

All three are pooled (68 labeled images total) and split once into
train/valid/test, since none of them is individually large enough to train
on alone.
"""

import gzip
import hashlib
import os

import cv2
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from src.segmentation.vessels import compute_frangi_response

# Random-crop patch size for training (see _random_crop_and_flip) -- chosen
# to comfortably fit an RTX-4060-class 8GB GPU at a reasonable batch size
# with this model's channel widths, while still giving the dilated
# bottleneck (receptive field ~100+px) plenty of context per patch.
DEFAULT_PATCH_SIZE = 512


def _drive_pairs(drive_root: str) -> list[tuple[str, str]]:
    img_dir = os.path.join(drive_root, "training", "images")
    mask_dir = os.path.join(drive_root, "training", "1st_manual")
    pairs = []
    for fname in sorted(os.listdir(img_dir)):
        if not fname.endswith("_training.tif"):
            continue
        stem = fname[: -len("_training.tif")]
        mask_path = os.path.join(mask_dir, f"{stem}_manual1.gif")
        if os.path.exists(mask_path):
            pairs.append((os.path.join(img_dir, fname), mask_path))
    return pairs


def _stare_pairs(stare_root: str) -> list[tuple[str, str]]:
    img_dir = os.path.join(stare_root, "stare-images")
    label_dir = os.path.join(stare_root, "labels-ah")
    pairs = []
    for fname in sorted(os.listdir(img_dir)):
        if not fname.endswith(".ppm.gz"):
            continue
        stem = fname[: -len(".ppm.gz")]  # e.g. "im0001"
        mask_path = os.path.join(label_dir, f"{stem}.ah.ppm.gz")
        if os.path.exists(mask_path):
            pairs.append((os.path.join(img_dir, fname), mask_path))
    return pairs


def _chase_pairs(chase_root: str) -> list[tuple[str, str]]:
    img_dir = os.path.join(chase_root, "Images")
    mask_dir = os.path.join(chase_root, "Masks")
    pairs = []
    for fname in sorted(os.listdir(img_dir)):
        if not fname.endswith(".jpg"):
            continue
        stem = fname[: -len(".jpg")]
        mask_path = os.path.join(mask_dir, f"{stem}_1stHO.png")
        if os.path.exists(mask_path):
            pairs.append((os.path.join(img_dir, fname), mask_path))
    return pairs


def build_pairs(drive_root: str, stare_root: str, chase_root: str) -> list[tuple[str, str, str]]:
    """Pool all three datasets into one `(image_path, mask_path, source)`
    list -- `source` is kept alongside so splitting can be stratified by
    dataset (see split_pairs()) rather than risking a test split drawn
    entirely from one source.
    """
    pairs = []
    pairs += [(img, mask, "drive") for img, mask in _drive_pairs(drive_root)]
    pairs += [(img, mask, "stare") for img, mask in _stare_pairs(stare_root)]
    pairs += [(img, mask, "chase") for img, mask in _chase_pairs(chase_root)]
    return pairs


def split_pairs(
    pairs: list[tuple[str, str, str]],
    valid_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> tuple[list, list, list]:
    """Stratified train/valid/test split by dataset source, so the held-out
    test set isn't accidentally drawn entirely from e.g. just CHASE_DB1.
    None of DRIVE/STARE/CHASE_DB1 ships an official held-out split with
    ground truth we can use directly (see module docstring re: DRIVE test),
    so this is a one-time random split instead, fixed by `seed` for
    reproducibility.
    """
    sources = [s for _, _, s in pairs]
    train_valid, test = train_test_split(pairs, test_size=test_frac, stratify=sources, random_state=seed)

    train_valid_sources = [s for _, _, s in train_valid]
    relative_valid_frac = valid_frac / (1 - test_frac)
    train, valid = train_test_split(
        train_valid, test_size=relative_valid_frac, stratify=train_valid_sources, random_state=seed
    )
    return train, valid, test


def _load_image_any(path: str, flags: int) -> np.ndarray:
    """Read an image, transparently gunzipping first if `path` ends in
    `.gz` (STARE ships gzip-compressed PPM) -- cv2.imdecode handles the
    decompressed PPM bytes directly, no temp file needed.
    """
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            data = f.read()
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), flags)
    else:
        image = cv2.imread(path, flags)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def _load_mask(path: str) -> np.ndarray:
    """Load a ground-truth vessel mask as a clean binary (0/1) 2D array.
    DRIVE's .gif masks decode as 3-channel even though the content is
    binary; STARE/CHASE_DB1 already decode single-channel -- normalize
    both cases the same way.
    """
    mask = _load_image_any(path, cv2.IMREAD_UNCHANGED)
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return (mask > 127).astype(np.uint8)


def _random_crop_and_flip(
    input_arr: np.ndarray, mask_arr: np.ndarray, patch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Crop a random `patch_size` x `patch_size` window out of `input_arr`
    (C, H, W) and `mask_arr` (H, W) at the same location, then apply the
    same random horizontal flip decision to both -- this is also the
    primary data augmentation, given the pooled dataset is only ~68 images.

    Uses a fresh `default_rng()` per call rather than a generator stored on
    the Dataset instance: with num_workers > 0, DataLoader worker processes
    are forked, and a pre-created generator would be copied at fork time,
    giving every worker the identical random sequence (duplicate
    augmentations across workers). A fresh OS-entropy-seeded generator per
    call sidesteps that without needing a custom worker_init_fn.
    """
    rng = np.random.default_rng()
    _, h, w = input_arr.shape
    pad_h, pad_w = max(0, patch_size - h), max(0, patch_size - w)
    if pad_h or pad_w:
        input_arr = np.pad(input_arr, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
        mask_arr = np.pad(mask_arr, ((0, pad_h), (0, pad_w)), mode="reflect")
        h, w = input_arr.shape[1:]

    top = rng.integers(0, h - patch_size + 1)
    left = rng.integers(0, w - patch_size + 1)
    input_patch = input_arr[:, top : top + patch_size, left : left + patch_size]
    mask_patch = mask_arr[top : top + patch_size, left : left + patch_size]

    if rng.random() < 0.5:
        input_patch = input_patch[:, :, ::-1]
        mask_patch = mask_patch[:, ::-1]

    return np.ascontiguousarray(input_patch), np.ascontiguousarray(mask_patch)


class VesselDataset(Dataset):
    """Yields `(input, mask)` tensors: `input` is (2, H, W) float32 --
    stacked (enhanced_green, frangi_response) from
    `vessels.compute_frangi_response()` -- and `mask` is (1, H, W) float32
    in {0, 1}.

    `train=True` random-crops a `patch_size` patch (+ paired flip) from the
    full canonicalized image/mask pair; `train=False` returns the full
    canonicalized image uncropped, matching what inference actually runs on
    (see vessel_infer.py).
    """

    def __init__(
        self,
        pairs: list[tuple[str, str, str]],
        cache_dir: str,
        train: bool = False,
        patch_size: int = DEFAULT_PATCH_SIZE,
    ):
        self.pairs = pairs
        self.cache_dir = cache_dir
        self.train = train
        self.patch_size = patch_size
        os.makedirs(cache_dir, exist_ok=True)

    def __len__(self) -> int:
        return len(self.pairs)

    def _cached_frangi_features(self, image_path: str, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Frangi takes ~3-4s per image at VESSEL_WORKING_WIDTH and doesn't
        # change across epochs (it's a fixed classical computation, not
        # learned) -- computing it fresh on every __getitem__ call would
        # otherwise dominate training time for no benefit, so cache to disk
        # the first time each image is seen.
        cache_path = os.path.join(self.cache_dir, hashlib.md5(image_path.encode()).hexdigest() + ".npz")
        if os.path.exists(cache_path):
            cached = np.load(cache_path)
            return cached["enhanced"], cached["vesselness"]

        enhanced, vesselness = compute_frangi_response(image)
        np.savez(cache_path, enhanced=enhanced, vesselness=vesselness)
        return enhanced, vesselness

    def __getitem__(self, idx: int):
        image_path, mask_path, _source = self.pairs[idx]
        image = _load_image_any(image_path, cv2.IMREAD_COLOR)
        mask = _load_mask(mask_path)

        enhanced, vesselness = self._cached_frangi_features(image_path, image)
        h, w = enhanced.shape
        # Resized to match compute_frangi_response()'s actual output shape
        # (read off the array, not recomputed) so this can never drift out
        # of sync with vessels.py's canonicalization logic. Nearest-neighbor
        # keeps the mask binary.
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.float32)

        input_arr = np.stack([enhanced, vesselness], axis=0)

        if self.train:
            input_arr, mask_resized = _random_crop_and_flip(input_arr, mask_resized, self.patch_size)

        return torch.from_numpy(input_arr.copy()), torch.from_numpy(mask_resized[None].copy())
