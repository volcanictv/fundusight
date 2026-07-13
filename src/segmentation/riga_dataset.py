"""RIGA: recover disc/cup masks from 6-annotator contour OVERLAYS.

RIGA (BinRushed + Magrabia + MESSIDOR, ~750 fundus photos) is the only dataset
in this repo carrying BOTH real pathology AND a cup annotation. REFUGE2 has cup
labels but is a clean glaucoma set; ADAM has pathology but ships disc masks with
no cup. That gap is why the CDR benefit of the 2026-07-14 localization work is
currently unmeasurable (see DEEP_DIVE.md). RIGA closes it.

NOTE ON NAMING: "MESSIDOR" here is RIGA's MESSIDOR SUBSET, not the MESSIDOR
diabetic-retinopathy grading dataset. They share a name and nothing else -- the
DR one ships no disc/cup labels at all. RIGA is ONE dataset with three sources.

THE LABELS ARE NOT MASKS
------------------------
RIGA ships no mask files. For each base image `imageNprime.tif` there are six
files `imageN-1..6`, each a COPY OF THE PHOTO with that ophthalmologist's disc
and cup contours drawn on top. The label has to be recovered by differencing an
annotation against its prime and reconstructing the filled regions:

    diff = |annotation - prime|          -> two thin closed curves
    label(diff)                          -> exactly 2 components: disc + cup ring
    fill_holes(each)                     -> two filled disks
    larger = disc, smaller = cup

This is clean in practice, not a heuristic hedge: the contours are drawn in a
solid colour with no antialiasing, so the diff is threshold-INSENSITIVE (a cut at
20, 30 or 50 selects the identical pixel set), and the two curves come out as
exactly two connected components. Anything that does NOT produce exactly two
plausible nested components is REJECTED rather than guessed at -- a silently
mis-reconstructed label is worse than a dropped one, because it poisons training
invisibly.

Masks are cached in REFUGE2's own raw pixel convention ({0=cup, 128=disc rim,
255=background}) precisely so that optic_disc_dataset._remap_mask_to_class_indices()
and OpticDiscDataset consume them with no new loading code, and RIGA pairs can be
pooled with REFUGE2 pairs directly.
"""

import glob
import os

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes, label

# Raw mask pixel values -- MUST match optic_disc_dataset's _MASK_* constants, so
# the cached RIGA masks are indistinguishable from REFUGE2's to every downstream
# consumer.
_MASK_CUP_VALUE = 0
_MASK_DISC_RIM_VALUE = 128
_MASK_BACKGROUND_VALUE = 255

# Any pixel differing from the prime by more than this (max over BGR) is contour.
# The choice is not sensitive -- the contours are solid-coloured and unantialiased,
# so 20/30/50 all select the same pixels (verified on real files). 30 sits in the
# middle of that plateau.
_CONTOUR_DIFF_THRESHOLD = 30

# Contour components smaller than this are compression speckle / stray marks, not
# an anatomical contour, and are dropped before the two-component check.
_MIN_CONTOUR_COMPONENT_PX = 100

# Sanity bounds on the reconstructed geometry. A recovered "cup" that is 98% of
# the disc, or a "disc" covering a third of the frame, means the reconstruction
# went wrong (contours merged, annotator drew only one curve, prime/annotation
# mismatched) -- reject rather than emit a confidently wrong label.
_MIN_CDR_AREA_RATIO = 0.02
_MAX_CDR_AREA_RATIO = 0.95
_MAX_DISC_FRAME_FRACTION = 0.15

# RIGA's folder layout, with the two real-world quirks it ships with:
#   * BinRushed1 contains NO prime images at all -- BinRushed1-Corrected is the
#     usable version of that subset, so BinRushed1 is deliberately NOT listed.
#   * Magrabia's female folder is misspelled "MagrabiFemale" in the distribution
#     (missing the second 'a'). Spelling it correctly finds nothing.
_SUBSET_DIRS = [
    ("MESSIDOR/MESSIDOR", "riga_messidor"),
    ("Magrabia/MagrabiaMale", "riga_magrabia_m"),
    ("Magrabia/MagrabiFemale", "riga_magrabia_f"),
    ("BinRushed/BinRushed1-Corrected", "riga_binrushed1"),
    ("BinRushed/BinRushed2", "riga_binrushed2"),
    ("BinRushed/BinRushed3", "riga_binrushed3"),
    ("BinRushed/BinRushed4", "riga_binrushed4"),
]

RIGA_MASK_CACHE_DIRNAME = "riga_masks"


def build_riga_annotation_sets(riga_root: str) -> list[dict]:
    """Find every base ("prime") image and its annotator overlays.

    Returns `[{"prime": path, "annotations": [path, ...], "source": str,
    "stem": str}, ...]`. Extensions are mixed WITHIN a single subset (BinRushed
    stores most overlays as .jpg but some as .tif), and case is inconsistent
    (`Image1-1.tif` vs `image1-2.tif`), so both are matched case-insensitively
    rather than hardcoded -- assuming one extension silently drops annotators.
    """
    sets = []
    for subdir, source in _SUBSET_DIRS:
        directory = os.path.join(riga_root, subdir)
        if not os.path.isdir(directory):
            continue

        by_name = {f.lower(): f for f in os.listdir(directory) if not f.lower().endswith(".db")}
        for lower, actual in sorted(by_name.items()):
            if "prime" not in lower:
                continue
            stem = lower.split("prime")[0]  # "image100prime.tif" -> "image100"

            annotations = []
            for index in range(1, 7):
                for ext in (".tif", ".jpg", ".jpeg", ".png"):
                    candidate = f"{stem}-{index}{ext}"
                    if candidate in by_name:
                        annotations.append(os.path.join(directory, by_name[candidate]))
                        break

            if annotations:
                sets.append(
                    {
                        "prime": os.path.join(directory, actual),
                        "annotations": annotations,
                        "source": source,
                        "stem": f"{source}_{stem}",
                    }
                )
    return sets


def extract_disc_cup_from_annotation(prime: np.ndarray, annotation: np.ndarray) -> tuple:
    """Recover one annotator's (disc_mask, cup_mask) by differencing their overlay
    against the clean image. Returns `(None, None, reason)` on any reconstruction
    the geometry does not support -- see the module docstring on why a rejected
    label beats a guessed one.

    Returns `(disc_mask, cup_mask, None)` on success.
    """
    if prime is None or annotation is None:
        return None, None, "unreadable file"
    if prime.shape != annotation.shape:
        return None, None, f"shape mismatch {prime.shape} vs {annotation.shape}"

    contour = cv2.absdiff(annotation, prime).max(axis=2) > _CONTOUR_DIFF_THRESHOLD
    if not contour.any():
        return None, None, "no contour drawn (annotation identical to prime)"

    labelled, count = label(contour)
    if count == 0:
        return None, None, "no contour components"

    sizes = np.bincount(labelled.ravel())
    sizes[0] = 0
    keep = [i for i in range(1, count + 1) if sizes[i] >= _MIN_CONTOUR_COMPONENT_PX]
    if len(keep) != 2:
        return None, None, f"expected 2 contour components (disc + cup), found {len(keep)}"

    # Fill each ring. The larger filled region is the disc, the smaller the cup --
    # they are nested by anatomy, and that nesting is verified below rather than
    # assumed, because a merged or mis-drawn pair would still produce two blobs.
    filled = [binary_fill_holes(labelled == i) for i in keep]
    filled.sort(key=lambda m: m.sum(), reverse=True)
    disc, cup = filled[0], filled[1]

    disc_area, cup_area = int(disc.sum()), int(cup.sum())
    if disc_area == 0 or cup_area == 0:
        return None, None, "degenerate (empty) region after filling"

    # The cup must lie inside the disc. If it does not, the two curves were not a
    # nested disc/cup pair at all and everything downstream would be wrong.
    outside = int((cup & ~disc).sum())
    if outside > 0.05 * cup_area:
        return None, None, f"cup not nested inside disc ({outside / cup_area:.0%} outside)"

    ratio = cup_area / disc_area
    if not (_MIN_CDR_AREA_RATIO <= ratio <= _MAX_CDR_AREA_RATIO):
        return None, None, f"implausible cup/disc area ratio {ratio:.3f}"
    if disc_area > _MAX_DISC_FRAME_FRACTION * disc.size:
        return None, None, f"disc covers {disc_area / disc.size:.1%} of frame"

    return disc, cup & disc, None


def fuse_annotations(masks: list, min_votes: int | None = None) -> tuple:
    """Majority-vote fusion of several annotators' (disc, cup) masks.

    `min_votes` defaults to a simple majority (more than half of the annotators
    who were successfully reconstructed -- NOT of the six who exist, since some
    may have been rejected). Fusing is the point of RIGA: a single
    ophthalmologist's contour carries the observer variability this dataset was
    built to expose, and the consensus is a strictly better target than any one
    of them.

    The cup is re-nested inside the fused disc at the end: majority-voting the
    two masks INDEPENDENTLY can, at the margin, admit a cup pixel where fewer
    than half the annotators put disc, which is anatomically impossible.
    """
    if not masks:
        return None, None
    if min_votes is None:
        min_votes = len(masks) // 2 + 1

    disc_votes = np.sum([m[0] for m in masks], axis=0)
    cup_votes = np.sum([m[1] for m in masks], axis=0)

    disc = disc_votes >= min_votes
    cup = (cup_votes >= min_votes) & disc
    return disc, cup


def masks_to_refuge_raw(disc: np.ndarray, cup: np.ndarray) -> np.ndarray:
    """Encode (disc, cup) into REFUGE2's raw mask convention, so the cached file
    is byte-compatible with what optic_disc_dataset already reads."""
    raw = np.full(disc.shape, _MASK_BACKGROUND_VALUE, dtype=np.uint8)
    raw[disc] = _MASK_DISC_RIM_VALUE
    raw[cup] = _MASK_CUP_VALUE
    return raw


def build_riga_mask_cache(riga_root: str, cache_root: str) -> dict:
    """Reconstruct and cache a fused mask per base image. Idempotent -- an
    existing mask is left alone, so an interrupted run resumes.

    Returns counts plus a breakdown of WHY annotations were rejected, which is
    the number to look at before trusting any of this: a high rejection rate
    means the reconstruction assumptions do not hold and the cache should not be
    trained on.
    """
    os.makedirs(cache_root, exist_ok=True)
    stats = {"images": 0, "written": 0, "skipped_existing": 0, "failed_images": 0}
    reasons: dict[str, int] = {}
    annotator_counts: list[int] = []

    for entry in build_riga_annotation_sets(riga_root):
        stats["images"] += 1
        out_path = os.path.join(cache_root, f"{entry['stem']}.png")
        if os.path.exists(out_path):
            stats["skipped_existing"] += 1
            continue

        prime = cv2.imread(entry["prime"], cv2.IMREAD_COLOR)
        recovered = []
        for annotation_path in entry["annotations"]:
            annotation = cv2.imread(annotation_path, cv2.IMREAD_COLOR)
            disc, cup, reason = extract_disc_cup_from_annotation(prime, annotation)
            if reason is not None:
                reasons[reason.split(" (")[0].split(" [")[0]] = reasons.get(reason.split(" (")[0].split(" [")[0], 0) + 1
                continue
            recovered.append((disc, cup))

        if not recovered:
            stats["failed_images"] += 1
            continue

        annotator_counts.append(len(recovered))
        disc, cup = fuse_annotations(recovered)
        cv2.imwrite(out_path, masks_to_refuge_raw(disc, cup))
        stats["written"] += 1

    stats["rejection_reasons"] = reasons
    stats["mean_annotators_per_image"] = float(np.mean(annotator_counts)) if annotator_counts else 0.0
    return stats


def build_riga_pairs(riga_root: str, cache_root: str) -> list[tuple[str, str, str]]:
    """`(image_path, mask_path, source)` triples for every base image with a
    cached fused mask -- the SAME shape optic_disc_dataset.build_pooled_pairs()
    returns, so RIGA can be concatenated with REFUGE2 and handed to
    split_pooled_pairs() for a source-stratified split with no new code.
    """
    pairs = []
    for entry in build_riga_annotation_sets(riga_root):
        mask_path = os.path.join(cache_root, f"{entry['stem']}.png")
        if os.path.exists(mask_path):
            pairs.append((entry["prime"], mask_path, entry["source"]))
    return pairs
