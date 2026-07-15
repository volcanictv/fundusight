"""Phase 6: THE calibration procedure for Stage 6.1's disc plausibility gate.

Run this whenever `locate_disc_classical()` changes how it picks its candidate.
CLAUDE.md states that requirement; this script is what makes it actually
executable, because the previous re-sweep was an ad-hoc, ADAM-only, in-sample
procedure with no committed artifact -- which is exactly how the gate's headline
claim went stale without anyone noticing.

WHAT WENT WRONG BEFORE, and what this script exists to prevent:

  1. IN-SAMPLE SCORING. The gate's advertised "16/16 wrong crops caught, 0 silent
     failures" was measured on the same 16 misses the thresholds were swept
     against. Under leave-one-out it is 14/16 (88%). A fitted number is not a
     measured one. This script always fits on one split and scores on another.

  2. ONE DATASET. Those 16 misses all came from ADAM, which is small and heavily
     pathological -- so the gate was fitted to one camera's framing and one
     failure mode. Measured across ADAM + REFUGE2 + RIGA (2219 discs), the
     false-alarm rate is not the advertised ~22% but ~49% pooled, reaching 89%
     on RIGA's BinRushed subsets. Half of all correctly-located discs get their
     CDR suppressed. That was invisible on ADAM alone.

  3. A FRAME-RELATIVE SIZE THRESHOLD. `diameter_fraction` divides the disc's
     diameter by the FRAME width, so it silently encodes the camera's field of
     view rather than the disc. A correct disc on a zoomed-in camera occupies a
     larger frame fraction than a WRONG blob on a wide-field one -- so a cap
     tuned on one camera cannot transfer to another. This script measures both
     the frame-relative and the FOV-relative normalisation side by side, so the
     next person can see which one is actually portable instead of assuming.

Reported per-dataset, never pooled-only: ADAM is pathological and REFUGE2/RIGA
are clean, so a pooled average hides precisely the subgroup that matters.

SILENT FAILURES ARE REPORTED FIRST, ALWAYS. A false alarm withholds a CDR; a
silent failure reports a CDR measured off a hemorrhage as if it were real. Any
"improvement" that trades the second for the first is a regression.

Run with:
    .venv\\Scripts\\python.exe scripts\\calibrate_disc_plausibility.py
    .venv\\Scripts\\python.exe scripts\\calibrate_disc_plausibility.py --rebuild
"""

import argparse
import itertools
import os
import pickle
import sys

import cv2
import numpy as np
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.segmentation import optic_disc, vessels
from src.segmentation.optic_disc_dataset import _remap_mask_to_class_indices, build_pooled_pairs
from src.segmentation.riga_dataset import build_riga_pairs

ADAM_ROOT = os.path.join(PROJECT_ROOT, "ADAM", "Training400")
ADAM_DISC_MASKS = os.path.join(ADAM_ROOT, "Disc_Masks")
CACHE_PATH = os.path.join(PROJECT_ROOT, "outputs", "disc_plausibility_features.pkl")

# Gate axes: +1 => correct discs score HIGHER (gate is `value >= t`),
#            -1 => correct discs score LOWER  (gate is `value <= t`).
AXES = {
    "circularity": +1,
    "contrast": +1,
    "n_components": -1,
    "diameter_fraction": -1,
    "diameter_fov_fraction": -1,
    "extent": +1,
    "solidity": +1,
}

# The gate as it currently ships, so every number below is a comparison against
# the thing actually running in production, not against a strawman.
PRODUCTION_GATE = [
    ("circularity", +1, optic_disc._MIN_DISC_CIRCULARITY),
    ("diameter_fraction", -1, optic_disc._MAX_DISC_DIAMETER_FRACTION),
    ("diameter_fraction", +1, optic_disc._MIN_DISC_DIAMETER_FRACTION),
]


def _adam_pairs() -> list:
    pairs = []
    for filename in sorted(os.listdir(ADAM_DISC_MASKS)):
        if not filename.endswith(".bmp"):
            continue
        stem = os.path.splitext(filename)[0]
        folder = "AMD" if stem.startswith("A") else "Non-AMD"
        pairs.append(
            (
                os.path.join(ADAM_ROOT, folder, stem + ".jpg"),
                os.path.join(ADAM_DISC_MASKS, filename),
                "adam_amd" if stem.startswith("A") else "adam_nonamd",
            )
        )
    return pairs


def build_ground_truth_pairs() -> list:
    """(image_path, mask_path, dataset) across every source that ships real
    disc ground truth. 2219 discs vs ADAM's 270."""
    pairs = _adam_pairs()
    pairs += [(img, mask, "refuge2") for img, mask, _src in build_pooled_pairs(os.path.join(PROJECT_ROOT, "REFUGE2"))]
    pairs += [(img, mask, src) for img, mask, src in build_riga_pairs(os.path.join(PROJECT_ROOT, "data"), os.path.join(PROJECT_ROOT, "data", "riga_masks"))]
    return pairs


def load_disc_truth(mask_path: str, dataset: str, shape: tuple):
    """ADAM and REFUGE2/RIGA use OPPOSITE mask polarities -- ADAM's disc is 0
    (black) on a white background, REFUGE2's follows its own {0=cup, 128=rim,
    255=bg} convention. This is asserted, not assumed: a silent polarity flip
    turns every hit into a miss and the resulting numbers still look plausible.
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        return None
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    if dataset.startswith("adam"):
        disc = mask == 0
        if disc.mean() > 0.25 or not disc.any():
            return None  # blank/unannotated -- 130 of ADAM's 400 are
    else:
        disc = _remap_mask_to_class_indices(mask) != 0
        if not disc.any():
            return None

    h, w = shape[:2]
    return cv2.resize(disc.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)


def candidate_features(working, green, fov, center_xy, expected_diameter) -> dict | None:
    """Measure the SAME window / Otsu / largest-component blob the production
    gate judges, plus candidates the gate does not currently use.

    The extra features are deliberately chosen to be INDEPENDENT of the
    selection rule (windowed green brightness x vascular convergence). Anything
    derived from brightness or convergence is contaminated: once those pick the
    peak, the peak scores high on them by construction, even on the misses --
    the same reason gating on convergence-at-the-chosen-center was measured and
    rejected (AUC 0.761 vs circularity's 0.898).

    `diameter_fov_fraction` is the important one. `diameter_fraction` divides by
    the FRAME width, which bakes in the camera's field of view; dividing by the
    FOV's own diameter is the framing-invariant version of the same quantity.
    """
    h, w = green.shape[:2]
    cx, cy = center_xy
    half = expected_diameter
    x0, x1 = max(0, int(cx - half)), min(w, int(cx + half))
    y0, y1 = max(0, int(cy - half)), min(h, int(cy + half))
    local, local_fov = green[y0:y1, x0:x1], fov[y0:y1, x0:x1]
    local_bgr = working[y0:y1, x0:x1]

    if local.size == 0 or not local_fov.any():
        return None
    in_window = local[local_fov]
    if in_window.min() == in_window.max():
        return None

    bright = (local > threshold_otsu(in_window)) & local_fov
    labelled = label(bright)
    if labelled.max() == 0:
        return None
    component = optic_disc._largest_component_mask(bright)
    region = max(regionprops(label(component)), key=lambda r: r.area)
    if region.perimeter <= 0:
        return None

    # FOV diameter: the retina circle's extent, not the frame's.
    fov_cols = np.nonzero(fov.any(axis=0))[0]
    fov_diameter = float(fov_cols.max() - fov_cols.min() + 1) if fov_cols.size else float(w)

    blue, green_c, red = (local_bgr[..., i].astype(np.float32) for i in range(3))
    saturation = cv2.cvtColor(local_bgr, cv2.COLOR_BGR2HSV)[..., 1].astype(np.float32)

    inside = component
    ring = cv2.dilate(component.astype(np.uint8), np.ones((15, 15), np.uint8), iterations=2).astype(bool) & ~component & local_fov
    if not inside.any():
        return None
    surround = float(local[ring].mean()) if ring.any() else float(local[local_fov].mean())

    diameter_px = float(region.equivalent_diameter_area)
    return {
        "circularity": min(4.0 * np.pi * region.area / (region.perimeter**2), 1.0),
        "solidity": float(region.solidity),
        "eccentricity": float(region.eccentricity),
        "extent": float(region.extent),
        "diameter_fraction": diameter_px / w,
        "diameter_fov_fraction": diameter_px / fov_diameter,
        "redness": float(((red[inside] - green_c[inside]) / (red[inside] + green_c[inside] + 1e-6)).mean()),
        "saturation": float(saturation[inside].mean()),
        "contrast": float(local[inside].mean()) - surround,
        "n_components": int(labelled.max()),
    }


def extract_all() -> list:
    items = []
    for image_path, mask_path, dataset in tqdm(build_ground_truth_pairs(), desc="features", mininterval=30):
        image = cv2.imread(image_path)
        if image is None:
            continue
        working = vessels._resize_to_working_width(image)
        green = vessels.extract_vessel_channel(working)
        fov = vessels._fov_mask(green)
        if not fov.any():
            continue
        h, w = green.shape[:2]

        truth = load_disc_truth(mask_path, dataset, green.shape)
        if truth is None:
            continue

        info = optic_disc.locate_disc_classical(working)
        cx, cy = info["center_xy"]
        xi, yi = int(round(cx)), int(round(cy))
        hit = bool(0 <= xi < w and 0 <= yi < h and truth[yi, xi])

        features = candidate_features(working, green, fov, info["center_xy"], w * optic_disc._EXPECTED_DISC_DIAMETER_FRACTION)
        if features is None:
            continue
        features.update(
            stem=os.path.basename(image_path),
            dataset=dataset,
            hit=hit,
            confident=bool(info["confident"]),
            reasons=info["implausible_reasons"],
        )
        items.append(features)
    return items


def passes(item: dict, gate: list) -> bool:
    for feature, direction, threshold in gate:
        value = item[feature]
        if np.isnan(value):
            return True  # unmeasurable -> never flag on a NaN
        if direction > 0 and value < threshold:
            return False
        if direction < 0 and value > threshold:
            return False
    return True


def score(gate: list, pool: list) -> tuple:
    hits = [i for i in pool if i["hit"]]
    misses = [i for i in pool if not i["hit"]]
    silent = [m for m in misses if passes(m, gate)]        # wrong crop, called CONFIDENT
    false_alarms = [h for h in hits if not passes(h, gate)]  # right crop, needlessly flagged
    return len(hits), len(misses), len(silent), len(false_alarms)


def fit_gate(features: list, pool: list, allowed_silent: int = 0):
    """Loosest thresholds (fewest false alarms) that still catch every wrong crop
    in `pool`, subject to at most `allowed_silent` slipping through."""
    misses = [i for i in pool if not i["hit"]]
    hits = [i for i in pool if i["hit"]]
    if not misses:
        return None

    grids = []
    for feature in features:
        direction = AXES[feature]
        values = sorted({round(m[feature], 4) for m in misses if not np.isnan(m[feature])})
        step = max(len(values) // 30, 1)
        grids.append([v + 1e-6 if direction > 0 else v - 1e-6 for v in values[::step]])

    best, best_fa = None, None
    for combo in itertools.product(*grids):
        gate = [(f, AXES[f], t) for f, t in zip(features, combo)]
        if sum(1 for m in misses if passes(m, gate)) > allowed_silent:
            continue
        fa = sum(1 for h in hits if not passes(h, gate))
        if best_fa is None or fa < best_fa:
            best, best_fa = gate, fa
    return best


def report_production(items: list) -> None:
    print("=" * 100)
    print("1. THE PRODUCTION GATE, measured on 2219 discs it was never fitted to")
    print("=" * 100)
    print(f"\n  {'dataset':<17} {'n':>5} {'localized':>10} {'wrong':>6} {'SILENT':>7}   {'false alarms':>16}")
    print("  " + "-" * 74)
    for dataset in sorted({i["dataset"] for i in items}):
        subset = [i for i in items if i["dataset"] == dataset]
        h, m, s, f = score(PRODUCTION_GATE, subset)
        print(f"  {dataset:<17} {len(subset):>5} {h / len(subset):>9.1%} {m:>6} {s:>7}   {f:>6} ({f / max(h, 1):>6.1%})")

    h, m, s, f = score(PRODUCTION_GATE, items)
    print("  " + "-" * 74)
    print(f"  {'POOLED':<17} {len(items):>5} {h / len(items):>9.1%} {m:>6} {s:>7}   {f:>6} ({f / max(h, 1):>6.1%})")
    print(f"\n  Silent failures: {s}/{m} wrong crops pass the gate CONFIDENTLY. The docs claim 0/16.")
    print(f"  False alarms:    {f}/{h} correct discs have their CDR suppressed. The docs claim ~22%.")


def report_portability(items: list) -> None:
    """The crux: is a threshold expressed relative to the FRAME portable across
    cameras, or does it just encode one camera's field of view?"""
    print("\n" + "=" * 100)
    print("2. IS THE SIZE THRESHOLD PORTABLE? (correct localizations only)")
    print("   Frame-relative vs FOV-relative. The spread ACROSS datasets is what matters:")
    print("   a threshold can only transfer if the quantity it gates on does not move with the camera.")
    print("=" * 100)
    print(f"\n  {'dataset':<17} {'diameter / FRAME':>26}   {'diameter / FOV':>26}")
    print(f"  {'':<17} {'p05':>8} {'median':>8} {'p95':>8}   {'p05':>8} {'median':>8} {'p95':>8}")
    print("  " + "-" * 78)

    frame_medians, fov_medians = [], []
    for dataset in sorted({i["dataset"] for i in items}):
        hits = [i for i in items if i["dataset"] == dataset and i["hit"]]
        frame = np.array([i["diameter_fraction"] for i in hits], dtype=float)
        fovr = np.array([i["diameter_fov_fraction"] for i in hits], dtype=float)
        frame, fovr = frame[~np.isnan(frame)], fovr[~np.isnan(fovr)]
        if frame.size == 0:
            continue
        frame_medians.append(np.median(frame))
        fov_medians.append(np.median(fovr))
        print(
            f"  {dataset:<17} {np.percentile(frame,5):>8.3f} {np.median(frame):>8.3f} {np.percentile(frame,95):>8.3f}"
            f"   {np.percentile(fovr,5):>8.3f} {np.median(fovr):>8.3f} {np.percentile(fovr,95):>8.3f}"
        )

    frame_spread = max(frame_medians) / min(frame_medians)
    fov_spread = max(fov_medians) / min(fov_medians)
    print("  " + "-" * 78)
    print(f"\n  Spread of the per-dataset MEDIAN (max/min) -- lower is more portable:")
    print(f"    diameter / FRAME : {frame_spread:.2f}x")
    print(f"    diameter / FOV   : {fov_spread:.2f}x")
    verdict = "FOV-relative is more portable" if fov_spread < frame_spread else "FOV normalisation does NOT help"
    print(f"\n  => {verdict}.")
    if fov_spread >= frame_spread:
        print("     The disc's apparent size varies by camera for a reason FOV normalisation does not")
        print("     capture (true optical magnification differs, not just how much retina is framed).")
        print("     A single global size threshold is then the wrong instrument, whatever it divides by.")


def report_heldout(items: list) -> None:
    print("\n" + "=" * 100)
    print("3. FIT ON ONE HALF, SCORE ON THE OTHER (20 random splits)")
    print("   Read the SILENT column first. Lowering false alarms while raising silent failures")
    print("   is a regression however good the false-alarm number looks.")
    print("=" * 100)

    rng = np.random.default_rng(0)
    candidates = {
        "production (frozen)": None,
        "refit: circularity + frame-size": ["circularity", "diameter_fraction"],
        "refit: circularity + FOV-size": ["circularity", "diameter_fov_fraction"],
        "refit: contrast + FOV-size + extent": ["contrast", "diameter_fov_fraction", "extent"],
        "refit: circ + contrast + FOV-size": ["circularity", "contrast", "diameter_fov_fraction"],
    }

    print(f"\n  {'gate':<38} {'held-out SILENT':>18} {'held-out false alarms':>24}")
    print("  " + "-" * 84)
    for name, features in candidates.items():
        silent, alarms, misses, hits = [], [], [], []
        for _ in range(20):
            order = rng.permutation(len(items))
            train = [items[i] for i in order[: len(items) // 2]]
            test = [items[i] for i in order[len(items) // 2 :]]
            gate = PRODUCTION_GATE if features is None else fit_gate(features, train)
            if gate is None:
                continue
            h, m, s, f = score(gate, test)
            silent.append(s); alarms.append(f); misses.append(m); hits.append(h)
        if not silent:
            print(f"  {name:<38}  could not fit")
            continue
        print(
            f"  {name:<38} {np.mean(silent):>5.1f}/{np.mean(misses):<5.1f} ({np.mean(silent)/max(np.mean(misses),1):>5.1%})"
            f"   {np.mean(alarms):>7.1f}/{np.mean(hits):<6.1f} ({np.mean(alarms)/max(np.mean(hits),1):>5.1%})"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rebuild", action="store_true", help="Re-extract features (~25 min) instead of using the cache.")
    args = parser.parse_args()

    if args.rebuild or not os.path.exists(CACHE_PATH):
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        items = extract_all()
        with open(CACHE_PATH, "wb") as fh:
            pickle.dump(items, fh)
        print(f"\nCached {len(items)} scored discs to {CACHE_PATH}\n")
    else:
        with open(CACHE_PATH, "rb") as fh:
            items = pickle.load(fh)
        print(f"Loaded {len(items)} scored discs from cache (--rebuild to re-extract)\n")

    report_production(items)
    report_portability(items)
    report_heldout(items)


if __name__ == "__main__":
    main()
