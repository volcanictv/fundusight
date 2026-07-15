"""Stage 6.2: dataset loading for REFUGE2 (disc/cup segmentation).

    REFUGE2/{train,val,test}/images/*.jpg
    REFUGE2/{train,val,test}/mask/*.{bmp,png}   -- train/test .bmp, val .png

Masks use pixel values {0=cup, 128=disc rim, 255=background}, remapped here to
class indices {0=background, 1=disc rim, 2=cup} to match optic_disc_model.py's
channel ordering and the loss/infer assumption that class 0 is background.

REFUGE2's official split is NOT a random sample: each folder is effectively one
camera domain (uniform resolution + very different colour stats), so a model
trained on it scores wildly differently val vs test and post-hoc calibration
won't transfer. build_pooled_pairs() / split_pooled_pairs() below pool all three
folders and re-split stratified by original folder so every split mixes all three
domains -- optic_disc_train.py uses these, not build_pairs(). Full write-up and
numbers in DEEP_DIVE.md / README.
"""

import glob
import hashlib
import os

import cv2
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from src.segmentation import vessels
from src.segmentation.optic_disc import DISC_ROI_WIDTH, crop_disc_roi, extract_color_features

_MASK_CUP_VALUE = 0
_MASK_DISC_RIM_VALUE = 128
_MASK_BACKGROUND_VALUE = 255

CLASS_BACKGROUND, CLASS_DISC_RIM, CLASS_CUP = 0, 1, 2


def _refuge_pairs(split_root: str) -> list[tuple[str, str]]:
    """Pair up images/*.jpg against mask/*.bmp OR mask/*.png -- confirmed by
    audit that train/test masks are .bmp but val masks are .png, so both
    extensions have to be checked rather than hardcoding one.
    """
    img_dir = os.path.join(split_root, "images")
    mask_dir = os.path.join(split_root, "mask")
    pairs = []
    for img_path in sorted(glob.glob(os.path.join(img_dir, "*.jpg"))):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        for ext in (".bmp", ".png"):
            mask_path = os.path.join(mask_dir, stem + ext)
            if os.path.exists(mask_path):
                pairs.append((img_path, mask_path))
                break
    return pairs


def build_pairs(refuge_root: str) -> dict:
    """REFUGE2's own official split, used directly -- kept for reference
    and its own tests, but see the module docstring: this is NO LONGER
    what optic_disc_train.py trains on, since the official split turned
    out to be a three-way camera/domain split, not a random sample of one
    population. Use build_pooled_pairs()/split_pooled_pairs() instead.
    """
    return {
        "train": _refuge_pairs(os.path.join(refuge_root, "train")),
        "val": _refuge_pairs(os.path.join(refuge_root, "val")),
        "test": _refuge_pairs(os.path.join(refuge_root, "test")),
    }


def build_pooled_pairs(refuge_root: str) -> list[tuple[str, str, str]]:
    """Pool ALL of REFUGE2's images (its own train+val+test folders) into
    one `(image_path, mask_path, source)` list, `source` tagging which
    ORIGINAL folder (and therefore which camera/domain -- see module
    docstring) each pair came from. `source` is kept alongside so
    split_pooled_pairs() can stratify by it, the same reason
    vessel_dataset.build_pairs() tags pairs by dataset source.
    """
    pooled = []
    pooled += [(img, mask, "orig_train") for img, mask in _refuge_pairs(os.path.join(refuge_root, "train"))]
    pooled += [(img, mask, "orig_val") for img, mask in _refuge_pairs(os.path.join(refuge_root, "val"))]
    pooled += [(img, mask, "orig_test") for img, mask in _refuge_pairs(os.path.join(refuge_root, "test"))]
    return pooled


def split_pooled_pairs(
    pairs: list[tuple[str, str, str]], valid_frac: float = 0.15, test_frac: float = 0.15, seed: int = 42
) -> tuple[list, list, list]:
    """Stratified train/valid/test split by ORIGINAL source folder (see
    build_pooled_pairs()), so each new split gets a proportional mix of
    all three camera domains instead of being a single one -- same
    stratified-split approach vessel_dataset.split_pairs() uses for
    DRIVE/STARE/CHASE_DB1, fixed by `seed` for reproducibility. Strips the
    source tag before returning -- OpticDiscDataset expects plain
    `(image_path, mask_path)` pairs, matching build_pairs()'s per-split
    lists.
    """
    sources = [s for _, _, s in pairs]
    train_valid, test = train_test_split(pairs, test_size=test_frac, stratify=sources, random_state=seed)

    train_valid_sources = [s for _, _, s in train_valid]
    relative_valid_frac = valid_frac / (1 - test_frac)
    train, valid = train_test_split(
        train_valid, test_size=relative_valid_frac, stratify=train_valid_sources, random_state=seed
    )

    strip_source = lambda pairs: [(img, mask) for img, mask, _source in pairs]
    return strip_source(train), strip_source(valid), strip_source(test)


WORKING_CACHE_DIRNAME = "optic_disc_working_cache"


def _working_cache_paths(cache_root: str, image_path: str) -> tuple[str, str]:
    """Cache filenames for one pair, keyed by a hash of the SOURCE IMAGE PATH.

    A hash rather than the bare filename because REFUGE2 and RIGA both reuse
    stems across folders (`image1-1.tif` exists in several RIGA subsets, and
    REFUGE2 repeats names across train/val/test) -- keying on the basename would
    silently collide and train the model on another image's mask.
    """
    key = hashlib.sha1(os.path.abspath(image_path).encode("utf-8")).hexdigest()[:16]
    return (
        os.path.join(cache_root, f"{key}_img.png"),
        os.path.join(cache_root, f"{key}_msk.png"),
    )


def build_working_cache(pairs: list[tuple[str, str]], cache_root: str) -> dict:
    """Pre-resize every (image, mask) pair to VESSEL_WORKING_WIDTH and cache it.

    This is purely an I/O optimisation, and it is what makes pooled REFUGE2+RIGA
    training possible at all on this machine. Decoding a native RIGA TIFF
    (2240x1488) costs ~130 ms per sample; the cached working-resolution PNG costs
    ~15 ms. That collapses the data pipeline from the bottleneck into a rounding
    error, which in turn means training no longer needs DataLoader worker
    processes -- and those workers, each holding full-resolution frames, were
    silently killing the run under CUDA on Windows (no traceback, process simply
    vanishes; the same paging-file exhaustion this module's sibling scripts warn
    about).

    Crucially this does NOT bake in the crop: the resized FULL frame is cached,
    and the ROI is still cropped (with jitter) at __getitem__ time. Caching the
    crop itself would freeze the jitter augmentation, which exists to make the
    model robust to Stage 6.1's imperfect localization.

    PNG, not JPEG -- a lossy re-encode would add compression artifacts to the
    exact cup/rim pallor boundary the model is being asked to resolve. Idempotent:
    existing entries are skipped, so an interrupted run resumes.
    """
    os.makedirs(cache_root, exist_ok=True)
    written = skipped = 0

    for image_path, mask_path in pairs:
        img_out, msk_out = _working_cache_paths(cache_root, image_path)
        if os.path.exists(img_out) and os.path.exists(msk_out):
            skipped += 1
            continue

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        mask_raw = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if image is None or mask_raw is None:
            raise FileNotFoundError(f"Could not read {image_path} / {mask_path}")
        if mask_raw.ndim == 3:
            mask_raw = cv2.cvtColor(mask_raw, cv2.COLOR_BGR2GRAY)

        image = vessels._resize_to_working_width(image)
        mask = _remap_mask_to_class_indices(mask_raw)
        mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(img_out, image)
        cv2.imwrite(msk_out, mask)  # already class indices {0,1,2}, not raw values
        written += 1

    return {"total": len(pairs), "written": written, "skipped_existing": skipped}


def _remap_mask_to_class_indices(mask: np.ndarray) -> np.ndarray:
    """{0=cup, 128=disc rim, 255=background} -> {0=background, 1=disc rim,
    2=cup} class-index array. Uses np.select rather than three separate
    boolean-index assignments so any stray pixel value (e.g. JPEG-adjacent
    compression artifacts on a mask that shouldn't have any, or an
    unexpected file) falls back to background instead of silently keeping
    whatever uninitialized value happened to be there.
    """
    conditions = [mask == _MASK_DISC_RIM_VALUE, mask == _MASK_CUP_VALUE]
    choices = [CLASS_DISC_RIM, CLASS_CUP]
    return np.select(conditions, choices, default=CLASS_BACKGROUND).astype(np.int64)


def _disc_bbox_from_mask(mask_class_idx: np.ndarray) -> dict:
    """Ground-truth-derived ROI center/diameter from the disc region (rim
    union cup, i.e. anything not background) bounding box. Deliberately
    NOT optic_disc.locate_disc_classical() -- that classical localizer is
    only ever run at real inference time, when no ground truth is
    available; using it here too would (a) waste time re-deriving what the
    label already tells us directly, and (b) leak the localizer's own
    errors into training, since a mis-centered crop would train the model
    on a systematically wrong disc position for that sample. Ground truth
    is both cheaper and strictly more correct for this purpose.
    """
    disc_region = mask_class_idx != CLASS_BACKGROUND
    ys, xs = np.nonzero(disc_region)
    if ys.size == 0:
        h, w = mask_class_idx.shape
        return {"center_xy": (w / 2.0, h / 2.0), "diameter_px": w * 0.1}

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    center_xy = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
    diameter_px = max(x1 - x0, y1 - y0) + 1
    return {"center_xy": center_xy, "diameter_px": float(diameter_px)}


def _augment(image: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> tuple:
    """Flip + small rotation + brightness/contrast jitter, applied
    identically to `image` and `mask` where geometric (flip/rotation
    preserve label alignment), image-only where photometric
    (brightness/contrast has no meaning for a class-index mask). Rotation
    uses INTER_NEAREST + a background(0) constant border for the mask, so
    no interpolated/invalid class index is ever introduced at the border or
    along a rotated edge.
    """
    h, w = mask.shape

    if rng.random() < 0.5:
        image = image[:, ::-1]
        mask = mask[:, ::-1]

    angle = rng.uniform(-15, 15)
    center = (w / 2.0, h / 2.0)
    rotation = cv2.getRotationMatrix2D(center, angle, 1.0)
    image = cv2.warpAffine(image, rotation, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    mask = cv2.warpAffine(
        mask, rotation, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=CLASS_BACKGROUND
    )

    alpha = rng.uniform(0.9, 1.1)  # contrast
    beta = rng.uniform(-15, 15)  # brightness
    image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(image), np.ascontiguousarray(mask)


class OpticDiscDataset(Dataset):
    """Yields `(input, target)` tensors: `input` is (7, roi_width,
    roi_width) float32 from optic_disc.extract_color_features(), `target`
    is (roi_width, roi_width) int64 class indices in {0, 1, 2}.

    `train=True` derives the ROI crop from the ground-truth mask's bounding
    box (see _disc_bbox_from_mask), JITTERED (center shifted, diameter
    scaled) so the network sees the same kind of imperfectly-centered crop
    Stage 6.1's classical localizer will actually produce at inference
    time -- without this, the model would only ever be trained on
    perfectly-centered crops and could be thrown off by a realistically
    imprecise ROI at inference. `train=False` uses the exact ground-truth
    bounding box, no jitter, no flip/rotation/brightness augmentation --
    deterministic, matching what a real inference-time crop is meant to
    approximate.
    """

    def __init__(
        self,
        pairs: list[tuple[str, str]],
        roi_width: int = DISC_ROI_WIDTH,
        train: bool = False,
        working_resolution: bool = True,
        working_cache_root: str | None = None,
    ):
        self.pairs = pairs
        self.roi_width = roi_width
        self.train = train
        # When set, samples are read from build_working_cache()'s pre-resized
        # PNGs instead of decoding the native image every epoch (~8x cheaper).
        # Implies working_resolution -- the cache IS at working resolution.
        self.working_cache_root = working_cache_root
        # `working_resolution=False` reproduces the pre-2026-07-14 behaviour
        # (crop from the native-resolution image), which is what every checkpoint
        # before optic_disc_unet.riga_pooled.pth was trained with. Kept so that
        # old result can be reproduced, NOT because it is correct -- see the note
        # in __getitem__: it does not match what inference does.
        self.working_resolution = working_resolution

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        image_path, mask_path = self.pairs[idx]

        if self.working_cache_root is not None:
            # Cached entries are ALREADY at working resolution and already
            # remapped to class indices -- see build_working_cache().
            img_path, msk_path = _working_cache_paths(self.working_cache_root, image_path)
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            mask = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
            if image is None or mask is None:
                raise FileNotFoundError(f"Working cache miss for {image_path} -- run build_working_cache() first")
            mask = mask.astype(np.int64)
            bbox_info = _disc_bbox_from_mask(mask)
            return self._finish(image, mask, bbox_info)

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        mask_raw = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask_raw.ndim == 3:
            mask_raw = cv2.cvtColor(mask_raw, cv2.COLOR_BGR2GRAY)
        mask = _remap_mask_to_class_indices(mask_raw)

        if self.working_resolution:
            # Downscale to VESSEL_WORKING_WIDTH *before* cropping, because that is
            # what INFERENCE does (optic_disc_infer.compute_optic_biomarkers_hybrid
            # resizes to the working width and only then calls crop_disc_roi).
            #
            # Without this, training and inference feed the U-Net crops of
            # different sharpness: at REFUGE2's native 2056px a disc is ~185px, so
            # the 3-diameter ROI is ~555px and gets DOWNsampled into the 512 input
            # (sharp); at the 1400px working width the same disc is ~126px, the ROI
            # is ~378px, and it gets UPsampled into 512 (soft). The model then
            # trains on sharp crops and is deployed on blurry ones -- the same
            # family of train/inference mismatch as the glaucoma ONH crop bug, and
            # a plausible part of why full-pipeline Dice (0.855/0.821) trails the
            # Dice reported during training on ground-truth crops (0.894/0.858).
            #
            # It also makes training tractable: the worker processes stop holding
            # full-resolution frames (REFUGE2 2056x2124, RIGA 2240x1488), which is
            # what was exhausting memory and killing the pooled run outright at the
            # epoch->validation boundary, when the second worker pool spawns.
            image = vessels._resize_to_working_width(image)
            mask = cv2.resize(
                mask.astype(np.uint8),
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.int64)

        bbox_info = _disc_bbox_from_mask(mask)
        return self._finish(image, mask, bbox_info)

    def _finish(self, image: np.ndarray, mask: np.ndarray, bbox_info: dict):
        """Crop the ROI (jittered in train mode), augment, and build the tensors.

        Shared by BOTH the cached and uncached paths above, on purpose: a cached
        sample and an uncached one must be identical apart from where the pixels
        were read from. Duplicating this tail is exactly how a cache silently
        starts producing a different distribution than the code it is meant to
        be an optimisation of.
        """
        center_xy, diameter_px = bbox_info["center_xy"], bbox_info["diameter_px"]

        if self.train:
            # Fresh generator per call, not stored on the Dataset instance
            # -- with num_workers > 0, DataLoader worker processes are
            # forked, and a pre-created generator would be copied at fork
            # time, giving every worker the identical random sequence
            # (duplicate augmentations across workers). Same load-bearing
            # convention as vessel_dataset.py's _random_crop_and_flip().
            rng = np.random.default_rng()
            jitter_frac = 0.1  # up to +/-10% of disc diameter, both axes
            center_xy = (
                center_xy[0] + rng.uniform(-jitter_frac, jitter_frac) * diameter_px,
                center_xy[1] + rng.uniform(-jitter_frac, jitter_frac) * diameter_px,
            )
            diameter_px = diameter_px * rng.uniform(0.9, 1.1)

        roi_image, bbox_meta = crop_disc_roi(image, center_xy, diameter_px, self.roi_width)
        mask_crop = mask[bbox_meta["y0"] : bbox_meta["y1"], bbox_meta["x0"] : bbox_meta["x1"]]
        roi_mask = cv2.resize(mask_crop, (self.roi_width, self.roi_width), interpolation=cv2.INTER_NEAREST)

        if self.train:
            roi_image, roi_mask = _augment(roi_image, roi_mask, rng)

        # Color-space conversion happens AFTER crop+resize (and after
        # augmentation), identically to how optic_disc_infer.py builds its
        # input -- crop first, then convert, always in that order, so
        # train and inference never see different interpolation artifacts
        # from converting at different resolutions.
        input_arr = extract_color_features(roi_image)

        return torch.from_numpy(input_arr.copy()), torch.from_numpy(roi_mask.astype(np.int64).copy())
