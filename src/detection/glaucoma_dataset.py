"""Phase 7: glaucoma classifier — dataset and REFUGE2 domain-stratified split.

Reads REFUGE2/glaucoma_labels_merged.csv (built by
scripts/build_glaucoma_labels.py, which merges REFUGE2's own
Refuge2_test.csv with SMDG-19's REFUGE1-train/REFUGE1-val rows — see that
script's docstring and ROADMAP.md's Phase 7 section for how the labels were
assembled and what conflicts were resolved).

Mirrors src/segmentation/optic_disc_dataset.py's pooled/stratified-split
pattern: REFUGE2's train/val/test folders are three different camera
domains (see that module's docstring), so the split here is stratified by
BOTH domain and glaucoma label jointly — glaucoma prevalence (~10-13%) is a
second imbalance a domain-only stratification wouldn't protect against.
"""

import os

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

from src.detection.onh_crop import crop_to_onh

MERGED_LABELS_FILENAME = "glaucoma_labels_merged.csv"

# Where build_onh_crop_cache() writes the precomputed ONH crops. Cached rather
# than cropped on the fly because localization costs ~53ms/image -- trivial
# once, but re-paid on every __getitem__ it would dominate a 30-epoch run over
# 998 images (~30 min of pure repeated localization, recomputing the exact same
# deterministic crop each time). Precomputed once in a single process (see
# glaucoma_train.py) rather than lazily inside __getitem__, which would race
# across DataLoader workers writing the same file.
ONH_CROP_CACHE_DIRNAME = "onh_crops"

# (image_path, glaucoma label, original REFUGE2 domain)
Pair = tuple[str, int, str]


def build_pairs(refuge_root: str) -> list[Pair]:
    csv_path = os.path.join(refuge_root, MERGED_LABELS_FILENAME)
    df = pd.read_csv(csv_path)
    pairs = []
    for row in df.itertuples(index=False):
        img_path = os.path.join(refuge_root, row.domain, "images", row.filename)
        pairs.append((img_path, int(row.glaucoma), row.domain))
    return pairs


def split_pairs(
    pairs: list[Pair], valid_frac: float = 0.15, test_frac: float = 0.15, seed: int = 42
) -> tuple[list[Pair], list[Pair], list[Pair]]:
    """Stratifies on a compound `{domain}_{label}` key so every split gets a
    proportional mix of all three REFUGE2 camera domains AND a proportional
    glaucoma-positive rate — same two-step train_test_split + relative_valid_frac
    correction as optic_disc_dataset.split_pooled_pairs(), extended to a
    compound stratification key instead of domain alone.
    """
    strat_keys = [f"{domain}_{label}" for _, label, domain in pairs]
    train_valid, test = train_test_split(pairs, test_size=test_frac, stratify=strat_keys, random_state=seed)

    train_valid_keys = [f"{domain}_{label}" for _, label, domain in train_valid]
    relative_valid_frac = valid_frac / (1 - test_frac)
    train, valid = train_test_split(
        train_valid, test_size=relative_valid_frac, stratify=train_valid_keys, random_state=seed
    )

    return train, valid, test


def onh_crop_path(cache_root: str, image_path: str, domain: str) -> str:
    """Cached-crop location for one source image. Namespaced by domain because
    REFUGE2's three camera domains reuse filenames across folders -- flattening
    them into one directory would silently overwrite crops.
    """
    stem = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(cache_root, domain, f"{stem}.png")


def build_onh_crop_cache(pairs: list[Pair], cache_root: str) -> dict:
    """Precompute an ONH crop (src/detection/onh_crop.py) for every pair,
    writing PNGs under `cache_root`. Idempotent -- existing crops are left
    alone, so an interrupted run resumes and a second training run costs
    nothing.

    Returns {"total", "written", "confident"}. `confident` counts how many
    crops the Stage 6.1 geometric plausibility checks accepted (see
    optic_disc.assess_disc_plausibility). It is reported rather than acted on:
    on REFUGE2 the localizer lands inside the true disc on ~90% of images, so
    the crops are good enough to train on, and dropping every flagged image
    would throw away more than half the dataset (the flag is deliberately
    tuned to over-flag: it catches 100% of bad crops at a ~20% false-alarm
    rate). Training on the flagged ones anyway is the right call -- the flag's
    job is to stop a bad crop from silently producing a confident CDR at
    inference time, not to curate a training set.

    PNG, not JPEG: these crops are model input, and re-encoding a crop with a
    lossy codec would add compression artifacts to the exact fine texture
    (rim thinning, RNFL striations) the classifier is being asked to look at.
    """
    written = 0
    confident = 0

    for image_path, _label, domain in pairs:
        out_path = onh_crop_path(cache_root, image_path, domain)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        if os.path.exists(out_path):
            continue

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        crop, disc_info = crop_to_onh(image)
        cv2.imwrite(out_path, crop)
        written += 1
        confident += bool(disc_info["confident"])

    return {"total": len(pairs), "written": written, "confident": confident}


class GlaucomaDataset(Dataset):
    """Reads a list of (image_path, label, domain) pairs — see build_pairs()/split_pairs().

    With `onh_crop_root` set, reads the precomputed optic-nerve-head crop for
    each pair (see build_onh_crop_cache()) instead of the full fundus photo.
    That is the whole point of the ONH-cropped model: the full-image classifier
    was found to attend to hemorrhages and edge artifacts rather than the disc,
    and glaucoma is defined by disc anatomy -- see src/detection/onh_crop.py.
    """

    def __init__(
        self,
        pairs: list[Pair],
        transform: transforms.Compose | None = None,
        onh_crop_root: str | None = None,
    ):
        self.pairs = pairs
        self.transform = transform
        self.onh_crop_root = onh_crop_root

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        path, label, domain = self.pairs[idx]

        if self.onh_crop_root is not None:
            path = onh_crop_path(self.onh_crop_root, path, domain)

        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def compute_class_weights(pairs: list[Pair]) -> torch.Tensor:
    """Inverse-frequency weights for CrossEntropyLoss — glaucoma is a ~85/15
    imbalanced binary label, same rationale as detection/dataset.py's
    compute_class_weights() for DR severity, just operating directly on an
    already-loaded pairs list instead of re-reading a CSV.
    """
    labels = [label for _, label, _domain in pairs]
    counts = np.bincount(labels, minlength=2)
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(counts)
    return torch.tensor(weights, dtype=torch.float32)


def domain_counts(pairs: list[Pair]) -> dict[str, int]:
    """Pair count per original REFUGE2 domain — used to print a coverage-by-domain
    report so it's obvious at a glance if a split lost representation of one
    of the three camera domains (the Phase 6 blind spot this whole
    stratification scheme exists to avoid)."""
    counts: dict[str, int] = {}
    for _path, _label, domain in pairs:
        counts[domain] = counts.get(domain, 0) + 1
    return counts
