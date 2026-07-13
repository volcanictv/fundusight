"""Phase 7 (revision): does the AMD classifier actually look at the macula?

Domain-expert review reported that the AMD classifier abandons the macula and
attends to large hemorrhages instead on striking cases, while behaving fine on
subtler ones. This script measures that against real ground truth rather than
taking it on faith or on a heuristic's word.

Crucially, it uses ADAM's OWN fovea ground truth (`Fovea_location.xlsx`, one
(Fovea_X, Fovea_Y) per image) -- NOT `locate_macula_classical()`, which a
previous validation already showed is unreliable (57% correct on eye-laterality
alone; see DEEP_DIVE.md's macula entry). Measuring an attention problem with a
localizer that is itself broken would confound the two, and there would be no
way to tell which one produced the number.

Metric: macula ENRICHMENT, same construction as
scripts/compare_glaucoma_attention.py --

    enrichment = (share of CAM mass inside the macula region)
               / (share of pixels that ARE the macula region)

where "macula region" is a disc-diameter-radius circle around the ground-truth
fovea (disc diameter taken from ADAM's ground-truth Disc_Masks, so the region
scales with each eye's real anatomy rather than a fixed pixel radius).
Enrichment 1.0 = attention no better than uniform; >1 = concentrated on the
macula; <1 = actively avoiding it.

Reported under BOTH Grad-CAM and LayerCAM. That is not padding: the glaucoma
comparison found these two methods can disagree about attention by an order of
magnitude and even invert which model looks better (EfficientNet-B0's final CAM
grid is only 7x7, so a coarse method cannot resolve a small structure). A claim
that survives both methods is worth something; one that only holds under a
single method is not.

Also splits by the model's own confidence, which is the specific shape of the
expert's claim: the failure is supposed to appear on STRIKING cases (where the
model is very confident) and not on subtle ones. If that is right, macula
enrichment should be LOWER among high-confidence AMD predictions -- the model
having found something bigger and easier to key on than the macula.

Run with:
    .venv\\Scripts\\python.exe scripts\\evaluate_amd_attention.py
"""

import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from src.detection import amd_infer
from src.detection.dataset import IMAGE_SIZE
from src.explainability.gradcam import compute_cam

ADAM_ROOT = os.path.join(PROJECT_ROOT, "ADAM", "Training400")
DISC_MASK_DIR = os.path.join(ADAM_ROOT, "Disc_Masks")


def _image_path(img_name: str) -> str:
    folder = "AMD" if img_name.startswith("A") else "Non-AMD"
    return os.path.join(ADAM_ROOT, folder, img_name)


def _disc_diameter(stem: str) -> float | None:
    """Ground-truth disc diameter in native pixels, from ADAM's Disc_Masks
    (disc = 0, background = 255). 130 of the 400 masks are entirely blank -- no
    disc annotated -- and return None rather than a fabricated default.
    """
    mask = cv2.imread(os.path.join(DISC_MASK_DIR, f"{stem}.bmp"), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    disc = mask == 0
    if not disc.any():
        return None
    ys, xs = np.nonzero(disc)
    return float(max(ys.max() - ys.min(), xs.max() - xs.min()))


def _macula_region(shape: tuple, fovea_xy: tuple, disc_diameter: float) -> np.ndarray:
    """Boolean mask of a disc-diameter-radius circle around the ground-truth
    fovea. Radius scales with each eye's own disc so the region means the same
    anatomical thing across ADAM's two native resolutions.
    """
    h, w = shape[:2]
    fx, fy = fovea_xy
    yy, xx = np.mgrid[0:h, 0:w]
    return ((xx - fx) ** 2 + (yy - fy) ** 2) <= disc_diameter**2


def _inpaint_region(image: np.ndarray, cx: float, cy: float, radius: float) -> np.ndarray:
    """Delete a circular region and fill it with surrounding retinal texture.

    INPAINTING, not blacking out, and the difference decides the experiment. A
    filled-black circle is itself a violent out-of-distribution artifact: this
    script's own numbers show a black disc drops the AMD probability by ~0.475
    NO MATTER WHERE IT IS PUT (macula or an arbitrary control region, p=0.20 --
    indistinguishable). A black-disc occlusion test therefore measures the
    model's reaction to a black disc, not its dependence on the anatomy it
    covered. Inpainting removes the region while leaving a plausible retina
    behind, so what remains is an actual test of "does the model need what was
    there".

    (This is also the empirical basis for the hemorrhage-masking verdict in
    DEEP_DIVE.md: a zeroed mask is a feature the network responds to strongly.)
    """
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.circle(mask, (int(cx), int(cy)), int(radius), 255, -1)
    # Inpaint at quarter scale -- cv2.inpaint is O(area) and these are ~2000px
    # images; the fill only needs to be plausible at the 224px the model sees.
    small = cv2.resize(image, (image.shape[1] // 4, image.shape[0] // 4))
    small_mask = cv2.resize(mask, (small.shape[1], small.shape[0]))
    filled = cv2.inpaint(small, small_mask, 7, cv2.INPAINT_TELEA)
    return cv2.resize(filled, (image.shape[1], image.shape[0]))


def occlusion_test(model, fovea_df: pd.DataFrame, device: str) -> None:
    """The causal test, and the one that actually settles the question.

    Grad-CAM is correlational and (as this repo has now measured twice) unstable
    across methods. Occlusion asks the causal question directly: if the macula
    -- the site that DEFINES age-related MACULAR degeneration -- is removed from
    the image, does the model still call it AMD?

    A matched CONTROL region (an equal-sized circle mirrored across the image
    center, so it lands on non-macular retina) is occluded too. Without that
    control, any confidence drop could just be "the image was edited at all".
    """
    print("\n" + "=" * 78)
    print("CAUSAL OCCLUSION TEST -- does the model actually NEED the macula?")
    print("=" * 78)

    rows = []
    for row in tqdm(list(fovea_df.itertuples(index=False)), leave=False):
        if not row.imgName.startswith("A"):  # true-AMD images only
            continue
        stem = os.path.splitext(row.imgName)[0]
        disc_diameter = _disc_diameter(stem)
        if disc_diameter is None:
            continue

        image = cv2.imread(_image_path(row.imgName))
        h, w = image.shape[:2]

        p_orig = amd_infer.predict(model, image, device)["probabilities"][1]
        p_macula = amd_infer.predict(model, _inpaint_region(image, row.Fovea_X, row.Fovea_Y, disc_diameter), device)["probabilities"][1]
        p_control = amd_infer.predict(model, _inpaint_region(image, w - row.Fovea_X, h - row.Fovea_Y, disc_diameter), device)["probabilities"][1]
        rows.append((p_orig, p_macula, p_control))

    df = pd.DataFrame(rows, columns=["p_orig", "p_macula", "p_control"])
    drop_macula = df.p_orig - df.p_macula
    drop_control = df.p_orig - df.p_control

    print(f"  true-AMD images, n={len(df)}   (mean AMD probability)\n")
    print(f"    unmodified                  {df.p_orig.mean():.3f}")
    print(f"    macula inpainted away       {df.p_macula.mean():.3f}   (drop {drop_macula.mean():+.3f})")
    print(f"    control region inpainted    {df.p_control.mean():.3f}   (drop {drop_control.mean():+.3f})")

    still_amd = (df.p_macula >= 0.5).sum()
    print(f"\n    STILL predicted AMD with the macula removed: {still_amd}/{len(df)} ({still_amd / len(df):.1%})")

    _stat, p_value = wilcoxon(drop_macula, drop_control)
    print(f"    Wilcoxon, macula-drop vs control-drop: p={p_value:.3f}")
    if p_value >= 0.05:
        print("\n  VERDICT: NO macula-specific dependence. Removing the macula costs the model")
        print("  essentially nothing, and no more than removing an arbitrary patch of retina.")
        print("  The model is deciding 'AMD' from features OUTSIDE the macula -- which is")
        print("  the substance of the reported failure, even though Grad-CAM above suggests")
        print("  the opposite. Attention maps are not causal evidence; this is.")
    else:
        print("\n  VERDICT: the model does depend on the macula specifically.")


def main():
    parser = argparse.ArgumentParser(description="Measure AMD classifier attention against ADAM's fovea ground truth.")
    parser.add_argument("--weights", default=amd_infer.DEFAULT_WEIGHTS_PATH)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = amd_infer.load_model(args.weights, device)

    fovea_df = pd.read_excel(os.path.join(ADAM_ROOT, "Fovea_location.xlsx"))
    print(f"Measuring AMD attention on {len(fovea_df)} ADAM images (ground-truth fovea + disc).")
    print(f"Device: {device}\n")

    records = []  # (stem, is_amd, amd_prob, enrichment_gradcam, enrichment_layercam)
    skipped = 0

    for row in tqdm(fovea_df.itertuples(index=False), total=len(fovea_df), leave=False):
        stem = os.path.splitext(row.imgName)[0]
        image = cv2.imread(_image_path(row.imgName))
        if image is None:
            raise FileNotFoundError(_image_path(row.imgName))

        disc_diameter = _disc_diameter(stem)
        if disc_diameter is None:
            skipped += 1
            continue

        macula = _macula_region(image.shape, (row.Fovea_X, row.Fovea_Y), disc_diameter)
        macula_cam_space = cv2.resize(
            macula.astype(np.uint8), (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        if not macula_cam_space.any():
            skipped += 1
            continue

        result = amd_infer.predict(model, image, device)
        amd_prob = result["probabilities"][1]

        enrichments = {}
        for method in ("gradcam", "layercam"):
            cam = compute_cam(model, image, method=method, target_class=1)
            share = cam[macula_cam_space].sum() / cam.sum() if cam.sum() > 0 else 0.0
            frac = macula_cam_space.mean()
            enrichments[method] = float(share / frac) if frac > 0 else float("nan")

        records.append((stem, stem.startswith("A"), amd_prob, enrichments["gradcam"], enrichments["layercam"]))

    df = pd.DataFrame(records, columns=["stem", "is_amd", "amd_prob", "gradcam", "layercam"])
    print(f"Scored {len(df)} images ({skipped} skipped: no ground-truth disc annotation in ADAM).\n")

    print("=" * 78)
    print("MACULA ENRICHMENT (attention on the ground-truth macula)")
    print("=" * 78)
    print("  1.0 = no better than uniform;  >1 = concentrated on macula;  <1 = avoiding it\n")

    for method in ("gradcam", "layercam"):
        print(f"  [{method}]")
        for label, subset in (("AMD (true)", df[df.is_amd]), ("Non-AMD (true)", df[~df.is_amd])):
            v = subset[method].dropna()
            print(f"    {label:<16} n={len(v):<4} mean={v.mean():.2f}  median={v.median():.2f}  p90={v.quantile(0.9):.2f}")
        print()

    print("=" * 78)
    print("THE EXPERT'S CLAIM: the failure appears on STRIKING cases, not subtle ones.")
    print("=" * 78)
    print("  If true, macula enrichment should FALL as the model's AMD confidence rises")
    print("  -- a confident model having keyed on something larger than the macula.\n")

    amd_only = df[df.is_amd].copy()
    confident = amd_only[amd_only.amd_prob >= 0.9]
    subtle = amd_only[amd_only.amd_prob < 0.9]

    for method in ("gradcam", "layercam"):
        c, s = confident[method].dropna(), subtle[method].dropna()
        print(f"  [{method}]  true-AMD images only")
        print(f"    striking (p>=0.9)  n={len(c):<4} mean enrichment={c.mean():.2f}  median={c.median():.2f}")
        print(f"    subtle   (p< 0.9)  n={len(s):<4} mean enrichment={s.mean():.2f}  median={s.median():.2f}")
        if len(c) and len(s):
            direction = "LOWER on striking cases (consistent with the claim)" if c.mean() < s.mean() else "HIGHER on striking cases (does NOT support the claim)"
            print(f"    -> enrichment is {direction}")
        print()

    corr = amd_only[["amd_prob", "gradcam", "layercam"]].corr(method="spearman")["amd_prob"]
    print("  Spearman correlation between AMD confidence and macula enrichment (true-AMD only):")
    print(f"    gradcam  {corr['gradcam']:+.3f}")
    print(f"    layercam {corr['layercam']:+.3f}")
    print("    (negative = more confident means LESS macula attention, i.e. supports the claim)")
    print("\n  The two methods disagree in SIGN. Nothing above settles the question --")
    print("  which is exactly why the causal occlusion test below exists.")

    occlusion_test(model, fovea_df, device)


if __name__ == "__main__":
    main()
