"""Validate the RIGA disc/cup reconstruction BEFORE training anything on it.

RIGA's labels are not masks -- they are six ophthalmologists' contours drawn on
top of copies of the photo, and the masks have to be reconstructed by
differencing (see src/segmentation/riga_dataset.py). A silent bug in that
reconstruction would poison the training set in a way no loss curve would ever
reveal: the model would faithfully learn wrong labels and report a healthy Dice
against them.

So this script does the three things that would actually catch that:

1. **Rejection audit.** How many annotator overlays fail to reconstruct, and
   WHY. A high or oddly-distributed rejection rate means the assumptions in
   riga_dataset.py do not hold on this data.
2. **Inter-annotator agreement.** Pairwise Dice between the six annotators, per
   structure. This is a property of the DATA, not of any model, and it is worth
   having for its own sake: it is the empirical noise floor for CDR, i.e. the
   number that says how well any model could possibly do. (DEEP_DIVE.md argues
   the REFUGE2 CDR error is already below human inter-observer variability --
   this measures that variability directly instead of citing the literature.)
3. **Eyeball evidence.** Renders the reconstructed disc/cup over the photo for a
   sample of images. Statistics can look fine while the masks are subtly wrong;
   looking at them is not optional.

Run with:
    .venv\\Scripts\\python.exe scripts\\validate_riga_extraction.py
"""

import argparse
import os
import sys

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from src.segmentation.optic_disc import _vertical_extent
from src.segmentation.riga_dataset import (
    build_riga_annotation_sets,
    extract_disc_cup_from_annotation,
    fuse_annotations,
)

RIGA_ROOT = os.path.join(PROJECT_ROOT, "data")


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    total = a.sum() + b.sum()
    return float(2.0 * (a & b).sum() / total) if total else 1.0


def _cdr(disc: np.ndarray, cup: np.ndarray) -> float:
    d = _vertical_extent(disc)
    return float(_vertical_extent(cup)) / d if d else 0.0


def _overlay(image: np.ndarray, disc: np.ndarray, cup: np.ndarray) -> np.ndarray:
    """Crop to the disc neighbourhood and draw the fused disc (green) and cup
    (red) outlines. Cropped, because the disc is a few percent of a RIGA frame
    and a full-frame thumbnail would show nothing useful."""
    ys, xs = np.nonzero(disc)
    if ys.size == 0:
        return cv2.resize(image, (256, 256))
    cy, cx = int(ys.mean()), int(xs.mean())
    half = max(int(2.0 * max(ys.max() - ys.min(), xs.max() - xs.min())), 60)
    h, w = image.shape[:2]
    y0, y1 = max(cy - half, 0), min(cy + half, h)
    x0, x1 = max(cx - half, 0), min(cx + half, w)

    view = image[y0:y1, x0:x1].copy()
    for mask, colour in ((disc, (0, 255, 0)), (cup, (0, 0, 255))):
        sub = mask[y0:y1, x0:x1].astype(np.uint8)
        contours, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(view, contours, -1, colour, 2)
    return cv2.resize(view, (256, 256))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--riga-root", default=RIGA_ROOT)
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N images (0 = all).")
    parser.add_argument("--contact-sheet", default=os.path.join(PROJECT_ROOT, "outputs", "riga_extraction_check.png"))
    args = parser.parse_args()

    entries = build_riga_annotation_sets(args.riga_root)
    if args.limit:
        # Stride rather than truncate, so the sample spans all subsets instead of
        # being entirely the alphabetically-first one.
        entries = entries[:: max(len(entries) // args.limit, 1)][: args.limit]
    print(f"Found {len(entries)} RIGA base images with annotator overlays.\n")

    reasons: dict[str, int] = {}
    per_source: dict[str, dict] = {}
    recovered_counts, disc_dices, cup_dices, cdr_spreads, fused_cdrs = [], [], [], [], []
    tiles = []

    for entry in tqdm(entries, desc="reconstructing"):
        prime = cv2.imread(entry["prime"], cv2.IMREAD_COLOR)
        recovered = []
        for path in entry["annotations"]:
            disc, cup, reason = extract_disc_cup_from_annotation(prime, cv2.imread(path, cv2.IMREAD_COLOR))
            if reason is not None:
                key = reason.split(" (")[0]
                reasons[key] = reasons.get(key, 0) + 1
                continue
            recovered.append((disc, cup))

        source = entry["source"]
        stats = per_source.setdefault(source, {"images": 0, "ok": 0, "annotators": 0})
        stats["images"] += 1
        if not recovered:
            continue
        stats["ok"] += 1
        stats["annotators"] += len(recovered)
        recovered_counts.append(len(recovered))

        # Inter-annotator agreement: every pair, per structure.
        for i in range(len(recovered)):
            for j in range(i + 1, len(recovered)):
                disc_dices.append(_dice(recovered[i][0], recovered[j][0]))
                cup_dices.append(_dice(recovered[i][1], recovered[j][1]))

        cdrs = [_cdr(d, c) for d, c in recovered]
        if len(cdrs) > 1:
            cdr_spreads.append(max(cdrs) - min(cdrs))

        disc, cup = fuse_annotations(recovered)
        fused_cdrs.append(_cdr(disc, cup))
        if len(tiles) < 24 and len(tiles) * 7 <= len(fused_cdrs):
            tiles.append(_overlay(prime, disc, cup))

    total_overlays = sum(s["images"] for s in per_source.values()) * 6
    print(f"\n{'=' * 74}")
    print("1. RECONSTRUCTION AUDIT")
    print(f"{'=' * 74}")
    print(f"  base images                 : {sum(s['images'] for s in per_source.values())}")
    print(f"  images with >=1 usable label: {sum(s['ok'] for s in per_source.values())}")
    print(f"  mean annotators recovered   : {np.mean(recovered_counts):.2f} / 6")
    rejected = sum(reasons.values())
    print(f"  annotator overlays rejected : {rejected} / ~{total_overlays} ({rejected / max(total_overlays, 1):.1%})")
    if reasons:
        print("\n  Rejection reasons (a rejected overlay is DROPPED, never guessed at):")
        for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"    {count:>5}  {reason}")

    print("\n  Per source:")
    for source, s in sorted(per_source.items()):
        mean_ann = s["annotators"] / s["ok"] if s["ok"] else 0
        print(f"    {source:<20} {s['ok']:>4}/{s['images']:<4} images usable   {mean_ann:.2f} annotators/image")

    print(f"\n{'=' * 74}")
    print("2. INTER-ANNOTATOR AGREEMENT  (a property of the DATA -- the noise floor")
    print("   any model is measured against, and cannot meaningfully beat)")
    print(f"{'=' * 74}")
    for name, values in (("disc", disc_dices), ("cup", cup_dices)):
        arr = np.array(values)
        print(f"  pairwise Dice, {name:<5}: mean={arr.mean():.4f}  median={np.median(arr):.4f}  p10={np.percentile(arr, 10):.4f}")
    spread = np.array(cdr_spreads)
    print(f"\n  CDR spread across the 6 annotators on the SAME image:")
    print(f"    mean={spread.mean():.4f}  median={np.median(spread):.4f}  p90={np.percentile(spread, 90):.4f}  max={spread.max():.4f}")
    print("    ^ this is the human disagreement on the exact quantity the pipeline reports.")

    fused = np.array(fused_cdrs)
    print(f"\n  Fused (consensus) CDR distribution: mean={fused.mean():.4f}  sd={fused.std():.4f}")

    if tiles:
        os.makedirs(os.path.dirname(args.contact_sheet), exist_ok=True)
        cols = 6
        rows = int(np.ceil(len(tiles) / cols))
        sheet = np.zeros((rows * 256, cols * 256, 3), dtype=np.uint8)
        for idx, tile in enumerate(tiles):
            r, c = divmod(idx, cols)
            sheet[r * 256 : (r + 1) * 256, c * 256 : (c + 1) * 256] = tile
        cv2.imwrite(args.contact_sheet, sheet)
        print(f"\n  Contact sheet (green=disc, red=cup) written to {args.contact_sheet}")
        print("  LOOK AT IT. The statistics above cannot tell you the masks are on the disc.")


if __name__ == "__main__":
    main()
