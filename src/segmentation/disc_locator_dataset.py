"""Phase 6 (Stage 6.0): dataset for the coarse full-frame disc locator.

Reuses REFUGE2's pooled/re-stratified pairs (optic_disc_dataset.
build_pooled_pairs / split_pooled_pairs) -- same images, same domain-balanced
split discipline, same seed -- so the locator and the disc/cup U-Net are
trained on consistent data and neither leaks the other's test images.

The difference is the TARGET. OpticDiscDataset yields a per-pixel class map
inside an ONH crop; this yields four numbers for the WHOLE frame: the disc's
bounding box as [x_center, y_center, width, height], each a fraction of the
frame. Deriving it from the same ground-truth mask means no extra annotation
is needed -- the label is already there, it just has to be read as geometry
instead of as pixels.

THE DECOY AUGMENTATION IS THE POINT
-----------------------------------
A locator trained on clean REFUGE2 frames alone would learn "the disc is the
bright roundish thing", which is precisely the heuristic that fails on the
pathological images this model exists to rescue -- it would inherit the
classical localizer's bug rather than correct it. REFUGE2 is a glaucoma set;
it is thin on the massive hemorrhages and confluent exudate of a BRVO or
severe DR frame, so that failure would never show up in training loss.

_add_synthetic_decoys() therefore paints bright, avascular blobs into the
frame at random locations away from the true disc, at sizes and intensities
that can outshine it. The box does not move. The model is thus explicitly
supervised to ignore a bright blob that has no vessels converging on it and
to keep pointing at the real disc -- forcing it onto the macro cues (vessel
arcade convergence, FOV geometry) rather than local brightness.
"""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.segmentation.disc_locator_model import LOCATOR_INPUT_SIZE
from src.segmentation.optic_disc_dataset import CLASS_BACKGROUND, _remap_mask_to_class_indices

# How many synthetic bright decoys to paint in, and how likely any are at all.
# Not every training frame gets them -- the model must also stay accurate on
# the clean, unpathological images that make up most real input.
_DECOY_PROBABILITY = 0.5
_DECOY_COUNT_RANGE = (1, 3)

# Decoy size as a fraction of frame width. The upper end deliberately EXCEEDS
# a real disc (~0.09-0.12 of width): a decoy that is always smaller than the
# disc would let the network cheat by simply picking the largest bright blob,
# learning a size rule instead of an anatomical one.
_DECOY_DIAMETER_FRACTION_RANGE = (0.05, 0.20)

# Decoy brightness. The top of the range saturates -- brighter than any real
# optic disc -- so brightness alone can never be a sufficient cue.
_DECOY_INTENSITY_RANGE = (200, 255)

# Keep decoys at least this far (in multiples of the true disc diameter) from
# the real disc center, so a decoy never overlaps the disc and turns the
# target box itself into a lie.
_DECOY_MIN_DISTANCE_FACTOR = 2.0


def disc_bbox_relative(mask_class_idx: np.ndarray) -> np.ndarray:
    """Disc bounding box as [x_center, y_center, width, height], each a
    fraction of the frame, from a {0=bg, 1=rim, 2=cup} class map. The disc is
    rim UNION cup (anything not background) -- the cup is anatomically inside
    the disc, so a box around the disc must contain it.

    Returns the frame center with a nominal 10%-of-width box if the mask has
    no disc at all. Such a sample is degenerate rather than informative, and
    the caller (see DiscLocatorDataset) drops it instead of training on it --
    this fallback exists only so the function never raises.
    """
    disc = mask_class_idx != CLASS_BACKGROUND
    h, w = mask_class_idx.shape
    ys, xs = np.nonzero(disc)
    if ys.size == 0:
        return np.array([0.5, 0.5, 0.1, 0.1], dtype=np.float32)

    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    return np.array(
        [
            ((x0 + x1) / 2.0) / w,
            ((y0 + y1) / 2.0) / h,
            (x1 - x0 + 1) / w,
            (y1 - y0 + 1) / h,
        ],
        dtype=np.float32,
    )


def _add_synthetic_decoys(image: np.ndarray, bbox: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Paint bright avascular blobs (fake exudate clusters / specular
    reflections) into the frame, away from the true disc. See the module
    docstring -- this is what stops the locator from simply learning
    "brightest blob wins". `bbox` is the relative disc box; it is NOT
    modified, which is exactly the supervision signal.
    """
    h, w = image.shape[:2]
    disc_cx, disc_cy = bbox[0] * w, bbox[1] * h
    disc_diameter = max(bbox[2] * w, bbox[3] * h)
    min_distance = _DECOY_MIN_DISTANCE_FACTOR * disc_diameter

    out = image.copy()
    for _ in range(rng.integers(*_DECOY_COUNT_RANGE, endpoint=True)):
        # Rejection-sample a location far enough from the true disc.
        for _attempt in range(20):
            cx, cy = rng.integers(0, w), rng.integers(0, h)
            if np.hypot(cx - disc_cx, cy - disc_cy) >= min_distance:
                break
        else:
            continue

        diameter = rng.uniform(*_DECOY_DIAMETER_FRACTION_RANGE) * w
        intensity = int(rng.integers(*_DECOY_INTENSITY_RANGE, endpoint=True))
        # Irregular ellipse, not a clean circle: real exudate clusters and
        # reflections are ragged, and a perfectly circular decoy would let the
        # network reject decoys on circularity alone -- a cue that would not
        # transfer, since a real hemorrhage is not conveniently non-circular.
        axes = (int(diameter / 2), int(diameter / 2 * rng.uniform(0.5, 1.0)))
        angle = float(rng.uniform(0, 180))
        cv2.ellipse(out, (int(cx), int(cy)), axes, angle, 0, 360, (intensity,) * 3, -1)

    # Blur only the painted result slightly, so decoy edges are not razor-sharp
    # in a way real lesions never are.
    return cv2.GaussianBlur(out, (0, 0), sigmaX=max(w * 0.002, 0.5))


def _augment(image: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> tuple:
    """Geometric + photometric augmentation. The bbox is recomputed FROM the
    augmented mask rather than transformed analytically -- flipping and
    rotating a mask and re-reading its extents is impossible to get subtly
    wrong, whereas hand-deriving the rotated box's new coordinates is easy to
    get wrong in a way that trains the model on quietly-misplaced targets.
    """
    h, w = mask.shape

    if rng.random() < 0.5:
        image = np.ascontiguousarray(image[:, ::-1])
        mask = np.ascontiguousarray(mask[:, ::-1])

    angle = float(rng.uniform(-20, 20))
    rotation = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    image = cv2.warpAffine(image, rotation, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    mask = cv2.warpAffine(
        mask, rotation, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=CLASS_BACKGROUND
    )

    alpha = float(rng.uniform(0.85, 1.15))  # contrast
    beta = float(rng.uniform(-20, 20))  # brightness
    image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    return image, mask


class DiscLocatorDataset(Dataset):
    """Yields `(image, bbox)`: `image` is (3, LOCATOR_INPUT_SIZE,
    LOCATOR_INPUT_SIZE) float32 RGB in [0, 1] of the WHOLE frame; `bbox` is
    (4,) float32 `[x_center, y_center, width, height]` relative to the frame.

    `train=True` applies flip/rotation/photometric augmentation AND the
    synthetic decoy lesions. `train=False` is deterministic and clean --
    validation/test measure accuracy on real, unmodified fundus photos, which
    is what actually gets deployed. (Evaluating on decoy-augmented frames
    would be measuring the model against its own training trick rather than
    against reality; see scripts/evaluate_disc_locator.py for the separate,
    honest robustness check against ADAM's real pathology.)
    """

    def __init__(self, pairs: list[tuple[str, str]], input_size: int = LOCATOR_INPUT_SIZE, train: bool = False):
        self.pairs = pairs
        self.input_size = input_size
        self.train = train

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        image_path, mask_path = self.pairs[idx]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        mask_raw = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask_raw.ndim == 3:
            mask_raw = cv2.cvtColor(mask_raw, cv2.COLOR_BGR2GRAY)
        mask = _remap_mask_to_class_indices(mask_raw)

        # DOWNSCALE FIRST, then augment -- order matters, and not only for
        # speed. REFUGE2's frames are up to 2056x2124, and rotating, painting
        # and blurring those at native resolution in every worker process
        # exhausted memory and killed training outright (silently: the process
        # simply vanished, no traceback). Augmenting at 256x256 instead is the
        # same augmentation on ~65x fewer pixels.
        #
        # It is also free of correctness cost, which is why it is safe to do:
        # the target is a RELATIVE bbox, so it is invariant to the resize, and
        # an anisotropic (square) resize maps onto it exactly with no letterbox
        # bookkeeping. Quantising the mask to 256px shifts a box edge by at most
        # 1/256 = 0.004 of the frame, against a disc ~0.09 wide -- negligible.
        image = cv2.resize(image, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask.astype(np.uint8), (self.input_size, self.input_size), interpolation=cv2.INTER_NEAREST)

        if self.train:
            # Fresh generator per __getitem__ call, never stored on self: with
            # num_workers > 0 the workers are forked/spawned, and a generator
            # created in __init__ would be duplicated into every worker, giving
            # them all the identical random stream. Same load-bearing
            # convention as OpticDiscDataset and vessel_dataset.
            rng = np.random.default_rng()
            image, mask = _augment(image, mask, rng)
            bbox = disc_bbox_relative(mask)
            if rng.random() < _DECOY_PROBABILITY:
                image = _add_synthetic_decoys(image, bbox, rng)
        else:
            bbox = disc_bbox_relative(mask)

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = np.transpose(rgb, (2, 0, 1))

        return torch.from_numpy(tensor.copy()), torch.from_numpy(bbox.copy())
