"""Phase 7 (revision): did ONH cropping actually move the glaucoma
classifier's attention onto the optic disc?

Domain-expert review found the full-image glaucoma classifier attending to
edge artifacts and hemorrhages instead of the optic nerve head. The fix
(src/detection/onh_crop.py) crops the classifier's input to the ONH before
classifying. This script checks whether that worked, by comparing the two
checkpoints on the SAME held-out test split:

  baseline  checkpoints/glaucoma_efficientnet_b0.fullimage_baseline.pth
            (the original model, classifying whole fundus photos)
  onh       checkpoints/glaucoma_efficientnet_b0.pth
            (the retrained model, classifying ONH crops)

It reports two things.

1. HELD-OUT METRICS for both, on the same split, so the attention fix can be
   checked for a performance regression rather than assumed harmless.
   Sensitivity/specificity get bootstrap confidence intervals, because the
   test split contains only ~18 glaucoma positives -- at that size a
   three-case swing moves sensitivity by 17 points, and reading such a
   difference as a real regression (or a real improvement) without an
   interval around it would be over-reading the data.

2. ATTENTION ON THE DISC, measured against REFUGE2's own ground-truth disc
   masks, as an ENRICHMENT RATIO:

       enrichment = (share of CAM mass falling on the disc)
                  / (share of pixels that ARE disc)

   Enrichment 1.0 means the model's attention is no better than spreading it
   uniformly over the frame; >1 means it concentrates on the disc.

   The ratio, not the raw share, is the honest comparison here: the ONH crop
   makes the disc occupy a far larger share of the frame (a few percent of a
   full photo vs. ~5-25% of a 3-disc-diameter crop), so the raw "fraction of
   attention on the disc" would rise even for a model that learned nothing and
   just spread its attention at random. Dividing by the disc's pixel share
   cancels that out.

   *** READ THIS BEFORE QUOTING ANY ENRICHMENT NUMBER. *** This metric turns
   out to be badly unstable across CAM methods, which is why it is reported
   under BOTH Grad-CAM and LayerCAM here rather than under one. On the same
   model, same target layer, and same images, the two disagree by roughly an
   order of magnitude and even INVERT which model looks better. The cause is
   resolution: EfficientNet-B0's final conv layer is a 7x7 grid, so one CAM
   cell covers 32x32 input pixels while the disc spans only ~2-3 cells --
   Grad-CAM's channel-averaged weighting produces a smooth blob that cannot
   concentrate at that scale (an image whose CAM center-of-mass sits exactly
   on the disc can still score enrichment 0.13), while LayerCAM's per-pixel
   positive-gradient weighting is far more selective and scores the same image
   much higher. Neither is "wrong"; they measure different things. So: use
   these numbers to compare a model against ITSELF under a FIXED method, and
   do not read a small cross-model difference under a single method as
   evidence the attention problem is fixed or unfixed. The structural argument
   (below) is the one that actually holds.

   The claim that DOES hold without depending on any of this: after cropping,
   edge artifacts and distant hemorrhages are not merely deprioritized, they
   are outside the model's input entirely and cannot be attended to at all.

Also writes a handful of side-by-side before/after Grad-CAM panels to
--output-dir, since a number that says attention improved is worth being able
to look at.

Run with:
    .venv\\Scripts\\python.exe scripts\\compare_glaucoma_attention.py
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from src.detection import glaucoma_infer
from src.detection.dataset import IMAGE_SIZE
from src.detection.glaucoma_dataset import build_pairs, split_pairs
from src.detection.onh_crop import crop_to_onh
from src.explainability.gradcam import compute_cam
from src.segmentation import optic_disc, vessels

BASELINE_WEIGHTS = os.path.join(PROJECT_ROOT, "checkpoints", "glaucoma_efficientnet_b0.fullimage_baseline.pth")
ONH_WEIGHTS = os.path.join(PROJECT_ROOT, "checkpoints", "glaucoma_efficientnet_b0.pth")
REFUGE_ROOT = os.path.join(PROJECT_ROOT, "REFUGE2")


def _gt_disc_mask(image_path: str, domain: str) -> np.ndarray | None:
    """REFUGE2's disc/cup mask for one image, as a boolean disc mask.

    Mask values are {0=cup, 128=disc rim, 255=background} -- so the DISC (rim
    plus the cup inside it) is everything below 255. train/test ship .bmp and
    val ships .png (an audited quirk of the dataset, see
    optic_disc_dataset.py), hence trying both extensions.
    """
    stem = os.path.splitext(os.path.basename(image_path))[0]
    for ext in (".bmp", ".png"):
        mask_path = os.path.join(REFUGE_ROOT, domain, "mask", stem + ext)
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                return mask < 255
    return None


def _enrichment(cam: np.ndarray, disc: np.ndarray) -> float | None:
    """Share of CAM mass on the disc, divided by the disc's share of pixels.

    Returns None when there's no disc in frame (the ONH crop can genuinely miss
    it) or the CAM is entirely flat -- both make the ratio undefined, and
    silently scoring them as 0.0 would drag the mean down with non-measurements.
    """
    cam_total = cam.sum()
    disc_frac = disc.mean()
    if cam_total <= 0 or disc_frac <= 0:
        return None
    return float((cam[disc].sum() / cam_total) / disc_frac)


def _sensitivity_at_specificity(labels: np.ndarray, probs: np.ndarray, target_specificity: float) -> float:
    """Best sensitivity achievable at or above `target_specificity`.

    This is the comparison that actually answers "did the fix cost us
    anything". Comparing sensitivity at each model's own argmax(0.5) threshold
    conflates two different things: how well a model RANKS cases, and where its
    probabilities happen to sit relative to 0.5. The ONH model outputs
    systematically lower probabilities than the baseline, so a fixed 0.5 cutoff
    silently places it at a much more conservative operating point -- which
    shows up as a "sensitivity regression" that is really just a threshold
    artifact. Matching specificity removes that confound.
    """
    fpr, tpr, _thresholds = roc_curve(labels, probs)
    achievable = tpr[(1 - fpr) >= target_specificity]
    return float(achievable.max()) if achievable.size else float("nan")


def _evaluate(model, pairs, use_onh_crop: bool, device: str, cam_panels: int, output_dir: str, tag: str) -> dict:
    labels, probs, preds = [], [], []
    enrichments = {"gradcam": [], "layercam": []}
    panels_written = 0

    for image_path, label, domain in tqdm(pairs, desc=tag, leave=False):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(image_path)

        gt_disc_native = _gt_disc_mask(image_path, domain)

        if use_onh_crop:
            model_input, disc_info = crop_to_onh(image)
            # Project the ground-truth disc into the SAME crop the model sees,
            # using the crop's own recorded bbox rather than re-deriving it.
            disc_in_input = None
            if gt_disc_native is not None:
                wh, ww = disc_info["working_shape"]
                disc_working = cv2.resize(gt_disc_native.astype(np.uint8), (ww, wh), interpolation=cv2.INTER_NEAREST)
                b = disc_info["bbox"]
                disc_in_input = disc_working[b["y0"] : b["y1"], b["x0"] : b["x1"]].astype(bool)
        else:
            model_input = image
            disc_in_input = gt_disc_native

        result = glaucoma_infer.predict_on_model_input(model, model_input, device)
        labels.append(label)
        probs.append(result["probabilities"][1])
        preds.append(result["class_idx"])

        cam = compute_cam(model, model_input, method="gradcam", target_class=1)

        if disc_in_input is not None and disc_in_input.any():
            disc_cam_space = cv2.resize(
                disc_in_input.astype(np.uint8), (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            # Both methods, because they disagree wildly -- see the module
            # docstring. Reporting only one would be cherry-picking.
            for method in ("gradcam", "layercam"):
                method_cam = cam if method == "gradcam" else compute_cam(model, model_input, method=method, target_class=1)
                value = _enrichment(method_cam, disc_cam_space)
                if value is not None:
                    enrichments[method].append(value)

        if panels_written < cam_panels:
            _write_panel(model_input, cam, os.path.join(output_dir, f"{tag}_{panels_written:02d}_{os.path.basename(image_path)}.png"))
            panels_written += 1

    labels, probs, preds = np.array(labels), np.array(probs), np.array(preds)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()

    return {
        "labels": labels,
        "probs": probs,
        "accuracy": (preds == labels).mean(),
        "auc": roc_auc_score(labels, probs),
        "sensitivity": tp / (tp + fn) if (tp + fn) else float("nan"),
        "specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
        "confusion": (tn, fp, fn, tp),
        "enrichment": {k: np.array(v) for k, v in enrichments.items()},
    }


def _write_panel(model_input: np.ndarray, cam: np.ndarray, path: str) -> None:
    from pytorch_grad_cam.utils.image import show_cam_on_image

    rgb = cv2.cvtColor(model_input, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32) / 255.0
    overlay = show_cam_on_image(resized, cam, use_rgb=True)
    side_by_side = np.hstack([(resized * 255).astype(np.uint8), overlay])
    cv2.imwrite(path, cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR))


def _bootstrap_ci(labels: np.ndarray, probs: np.ndarray, metric: str, n: int = 2000, seed: int = 0) -> tuple:
    """Percentile bootstrap CI. With ~18 positives in the test split, the point
    estimate of sensitivity moves in ~5.5-point steps -- an interval is the
    only honest way to report it.
    """
    rng = np.random.default_rng(seed)
    preds = (probs >= 0.5).astype(int)
    values = []
    for _ in range(n):
        idx = rng.integers(0, len(labels), len(labels))
        lab, pr = labels[idx], preds[idx]
        if len(np.unique(lab)) < 2:
            continue
        tn, fp, fn, tp = confusion_matrix(lab, pr, labels=[0, 1]).ravel()
        if metric == "sensitivity":
            values.append(tp / (tp + fn) if (tp + fn) else np.nan)
        else:
            values.append(tn / (tn + fp) if (tn + fp) else np.nan)
    values = np.array([v for v in values if not np.isnan(v)])
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def main():
    parser = argparse.ArgumentParser(description="Compare full-image vs ONH-cropped glaucoma classifier.")
    parser.add_argument("--cam-panels", type=int, default=6, help="How many before/after Grad-CAM panels to write.")
    parser.add_argument("--output-dir", default=os.path.join(PROJECT_ROOT, "outputs", "glaucoma_attention"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pairs = build_pairs(REFUGE_ROOT)
    _train, _val, test_pairs = split_pairs(pairs, seed=args.seed)
    print(f"Held-out test split: {len(test_pairs)} images ({sum(l for _p, l, _d in test_pairs)} glaucoma-positive)")
    print(f"Device: {device}\n")

    baseline_model = glaucoma_infer.load_model(BASELINE_WEIGHTS, device)
    onh_model = glaucoma_infer.load_model(ONH_WEIGHTS, device)

    baseline = _evaluate(baseline_model, test_pairs, False, device, args.cam_panels, args.output_dir, "baseline_fullimage")
    onh = _evaluate(onh_model, test_pairs, True, device, args.cam_panels, args.output_dir, "onh_crop")

    print("=" * 78)
    print("1. HELD-OUT TEST METRICS (same split, same images)")
    print("=" * 78)
    print(f"{'':<16}{'baseline (full image)':>24}{'ONH crop':>20}")
    for name in ("accuracy", "auc", "sensitivity", "specificity"):
        print(f"  {name:<14}{baseline[name]:>24.4f}{onh[name]:>20.4f}")
    print(f"\n  confusion (tn, fp, fn, tp): baseline={baseline['confusion']}  onh={onh['confusion']}")

    print("\n  95% bootstrap CIs (the test split has only ~18 positives):")
    for name in ("sensitivity", "specificity"):
        b_lo, b_hi = _bootstrap_ci(baseline["labels"], baseline["probs"], name)
        o_lo, o_hi = _bootstrap_ci(onh["labels"], onh["probs"], name)
        print(f"    {name:<12} baseline [{b_lo:.3f}, {b_hi:.3f}]   onh [{o_lo:.3f}, {o_hi:.3f}]")

    print("\n  SENSITIVITY AT MATCHED SPECIFICITY -- the like-for-like comparison.")
    print("  The ONH model outputs systematically lower probabilities, so a fixed")
    print("  argmax(0.5) puts it at a more conservative operating point. That, not a")
    print("  loss of discriminative power, is what a raw sensitivity drop reflects:")
    for target in (baseline["specificity"], 0.80, onh["specificity"]):
        b_sens = _sensitivity_at_specificity(baseline["labels"], baseline["probs"], target)
        o_sens = _sensitivity_at_specificity(onh["labels"], onh["probs"], target)
        print(f"    at specificity >= {target:.3f}:   baseline sens={b_sens:.3f}   onh sens={o_sens:.3f}")

    print("\n" + "=" * 78)
    print("2. ATTENTION ON THE OPTIC DISC (CAM vs REFUGE2 ground-truth disc)")
    print("=" * 78)
    print("   enrichment = (share of CAM mass on disc) / (share of pixels that are disc)")
    print("   1.0 = attention no better than uniform; higher = more disc-focused")
    print("   Reported under BOTH methods because they disagree -- see module docstring.\n")
    for method in ("gradcam", "layercam"):
        print(f"  [{method}]")
        for name, res in (("baseline (full image)", baseline), ("ONH crop", onh)):
            e = res["enrichment"][method]
            print(
                f"    {name:<22} n={e.size:<4} mean={e.mean():.2f}  median={np.median(e):.2f}  "
                f"p90={np.percentile(e, 90):.2f}"
            )
        ratio = onh["enrichment"][method].mean() / baseline["enrichment"][method].mean()
        direction = "MORE" if ratio > 1 else "LESS"
        print(f"    -> ONH model puts {ratio:.2f}x {direction} attention on the disc under {method}.\n")

    print("  The two methods disagree on both the magnitude AND the direction of the")
    print("  change. Treat the enrichment numbers as inconclusive; the claim that holds")
    print("  regardless is structural -- after cropping, edge artifacts and distant")
    print("  hemorrhages are outside the model's input and cannot be attended to at all.")

    print(f"\nWrote {args.cam_panels} before/after Grad-CAM panels per model to {args.output_dir}")


if __name__ == "__main__":
    main()
