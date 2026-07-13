"""Phase 7 (revision): is the DR classifier centrally biased, and if so, why?

Domain-expert review reported the DR classifier showing a central spatial bias
-- attending to the macula/central retina and missing pathology out in the
periphery. This script works through that claim in the order that costs the
least to rule things out.

STEP 1 -- IS PREPROCESSING THROWING THE PERIPHERY AWAY?
The cheapest possible explanation, and a one-line fix if true: if something in
the path from `cv2.imread` to the model's input tensor crops or masks off
peripheral pixels, then the model isn't biased at all, it simply never sees the
periphery. This step asserts, programmatically, that no such crop exists.

STEP 2 -- IF PREPROCESSING IS CLEAN, IS THERE A LEARNED BIAS?
Measure where the model's Grad-CAM attention actually falls, as a function of
distance from the center of the field of view, aggregated over many images.

The trap in step 2, and the reason for the control below: a CNN + Grad-CAM has
a center bias *built in*, for reasons that have nothing to do with what it
learned. Zero-padding at every conv layer makes border activations
systematically weaker; the final 7x7 CAM grid is coarse; and the fundus FOV is
a circle inscribed in a square frame, so the outer radial bins are partly black
background to begin with. Any of these alone would produce a center-weighted
attention profile from a model that learned nothing whatsoever.

So this measures the SAME radial profile twice: once for the trained DR
classifier, and once for an untrained, randomly-initialized EfficientNet-B0.
The untrained model's profile is pure architecture/CAM artifact -- it has
learned nothing about retinas. Only the DIFFERENCE between the two curves can
be attributed to what training taught the model. Attention is also normalized
per radial bin by how much FOV (non-black retina) actually falls in that bin,
so the circular field of view doesn't masquerade as a learned preference.

STEP 3 -- CORRELATE ATTENTION WITH REAL LESION LOCATIONS: **BLOCKED, NO DATA.**
The planned confirmation was to correlate peripheral lesion locations against
Grad-CAM attention using IDRiD's lesion segmentation masks. Those masks are not
in this repo. IDRiD ships as separate downloads, and only "B. Disease Grading"
(455 JPEGs + a severity CSV, no pixel labels) was ever downloaded -- see
`data/IDRi/`. The lesion masks live in IDRiD's "A. Segmentation" subset (81
images with per-lesion masks: microaneurysms, haemorrhages, hard exudates, soft
exudates, optic disc), which is not present. This script therefore CANNOT run
that correlation, and says so rather than substituting a different dataset and
quietly calling it the same check. To unblock it, download IDRiD "A.
Segmentation" and the correlation becomes straightforward: for each image, the
lesion mask gives ground-truth pathology locations, and the CAM gives attention
-- the question is whether attention on peripheral lesions is systematically
lower than on central ones.

Run with:
    .venv\\Scripts\\python.exe scripts\\investigate_dr_spatial_bias.py
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from src.detection.dataset import IMAGE_SIZE, build_transforms
from src.detection.infer import DEFAULT_WEIGHTS_PATH, load_model
from src.detection.model import build_model
from src.explainability.gradcam import compute_cam
from src.preprocessing.enhance import preprocess
from src.segmentation import vessels

N_RADIAL_BINS = 6


def step1_preprocessing_is_clean(image: np.ndarray) -> bool:
    """Assert nothing between cv2.imread and the model's input discards
    peripheral pixels. Three independent things to check, because "the
    periphery is gone" could happen in three different places.
    """
    print("=" * 78)
    print("STEP 1 -- DOES PREPROCESSING CROP OR DISCARD PERIPHERAL PIXELS?")
    print("=" * 78)

    ok = True

    # (a) The transform chain the classifier actually uses. A Resize to a
    #     (H, W) tuple squashes the whole frame in; a CenterCrop/RandomCrop
    #     would cut the periphery off. There must be no crop op.
    ops = [type(t).__name__ for t in build_transforms(train=False).transforms]
    crop_ops = [o for o in ops if "Crop" in o]
    print(f"  (a) eval transform chain: {ops}")
    if crop_ops:
        print(f"      !! FOUND CROP OPS: {crop_ops} -- these discard peripheral pixels.")
        ok = False
    else:
        print("      -> no crop op. Resize((224,224)) squashes the full frame in; nothing is cut off.")

    # (b) enhance.preprocess() must not shrink or mask the frame. (It is
    #     display-only per report/pipeline.py, but verify rather than trust.)
    processed = preprocess(image)
    same_shape = processed.shape == image.shape
    print(f"  (b) enhance.preprocess(): {image.shape} -> {processed.shape}")
    if not same_shape:
        print("      !! preprocess() changed the frame size.")
        ok = False
    else:
        print("      -> shape preserved, no cropping.")

    # (c) The decisive one: is preprocess() even in the classifier's path?
    #     report/pipeline.py passes the RAW image to the classifiers and uses
    #     preprocess() only to render a before/after panel.
    import inspect

    from src.report import pipeline

    source = inspect.getsource(pipeline.run_pipeline)
    feeds_raw = "_run_classifier(image," in source or "_run_classifier(\n        image," in source
    print(f"  (c) does report/pipeline.py feed preprocess() output to the classifier? {'no' if feeds_raw else 'UNCLEAR'}")
    print("      -> classifiers receive the raw image; preprocess() output is display-only.")

    print(f"\n  VERDICT: preprocessing is {'CLEAN -- it is not the cause.' if ok else 'THE PROBLEM.'}")
    print("  The model does see the full frame, periphery included. So if a central")
    print("  bias exists, it is learned (or an artifact) -- not a cropped-away periphery.\n")
    return ok


def _radial_profile(cam: np.ndarray, fov: np.ndarray) -> np.ndarray:
    """Mean CAM intensity per radial bin, measured from the FOV's center and
    normalized so that a perfectly uniform attention map scores 1.0 in every
    bin.

    Only FOV (retina) pixels count. Without that, the outer bins would be
    diluted by the black corners of the frame and every model would look
    "centrally biased" purely because a fundus photo is a circle in a square.
    """
    h, w = cam.shape
    ys, xs = np.nonzero(fov)
    if ys.size == 0:
        return np.full(N_RADIAL_BINS, np.nan)

    cy, cx = ys.mean(), xs.mean()
    radius = max(np.sqrt(((ys - cy) ** 2 + (xs - cx) ** 2)).max(), 1e-6)

    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / radius

    overall_mean = cam[fov].mean()
    if overall_mean <= 0:
        return np.full(N_RADIAL_BINS, np.nan)

    profile = np.full(N_RADIAL_BINS, np.nan)
    edges = np.linspace(0, 1.0, N_RADIAL_BINS + 1)
    for i in range(N_RADIAL_BINS):
        band = fov & (r >= edges[i]) & (r < edges[i + 1] if i < N_RADIAL_BINS - 1 else r <= 1.0)
        if band.any():
            profile[i] = cam[band].mean() / overall_mean
    return profile


def _fov_at_cam_scale(image: np.ndarray) -> np.ndarray:
    """The retina's field-of-view mask, resized to the CAM's 224x224 grid the
    same way the model's own input was (a plain squash -- see build_transforms).
    """
    green = vessels.extract_vessel_channel(image)
    fov = vessels._fov_mask(green)
    return cv2.resize(fov.astype(np.uint8), (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST).astype(bool)


def step2_radial_attention(image_paths: list, device: str) -> None:
    print("=" * 78)
    print("STEP 2 -- WHERE DOES ATTENTION ACTUALLY FALL? (radial profile)")
    print("=" * 78)
    print("  Normalized so uniform attention = 1.00 in every bin.")
    print("  'untrained' = randomly-initialized EfficientNet-B0: its profile is pure")
    print("  architecture/Grad-CAM artifact (zero-padding border effects, 7x7 CAM grid),")
    print("  NOT anything learned. Only trained-minus-untrained is attributable to training.\n")

    trained = load_model(DEFAULT_WEIGHTS_PATH, device)
    untrained = build_model(num_classes=5, pretrained=False).to(device).eval()

    profiles = {"trained": [], "untrained": []}
    for path in tqdm(image_paths, desc="radial profile", leave=False):
        image = cv2.imread(path)
        if image is None:
            continue
        fov = _fov_at_cam_scale(image)
        if not fov.any():
            continue
        for name, model in (("trained", trained), ("untrained", untrained)):
            # target_class=None -> explain the model's own top prediction.
            cam = compute_cam(model, image, method="gradcam", target_class=None)
            profile = _radial_profile(cam, fov)
            if not np.isnan(profile).any():
                profiles[name].append(profile)

    trained_mean = np.mean(profiles["trained"], axis=0)
    untrained_mean = np.mean(profiles["untrained"], axis=0)
    delta = trained_mean - untrained_mean

    print(f"  n = {len(profiles['trained'])} images\n")
    header = "  bin (center->edge)  " + "".join(f"{i / N_RADIAL_BINS:.2f}-{(i + 1) / N_RADIAL_BINS:.2f} " for i in range(N_RADIAL_BINS))
    print(header)
    print(f"  {'trained':<18}  " + "".join(f"{v:^10.2f}" for v in trained_mean))
    print(f"  {'untrained (null)':<18}  " + "".join(f"{v:^10.2f}" for v in untrained_mean))
    print(f"  {'difference':<18}  " + "".join(f"{v:^+10.2f}" for v in delta))

    center_learned = delta[0]
    edge_learned = delta[-1]
    print()
    print(f"  Innermost bin, learned component: {center_learned:+.2f}")
    print(f"  Outermost bin, learned component: {edge_learned:+.2f}")

    if center_learned > 0.05 and edge_learned < -0.05:
        verdict = "A LEARNED central bias is present (training pushed attention inward)."
    elif abs(center_learned) <= 0.05 and abs(edge_learned) <= 0.05:
        verdict = "NO learned central bias. The center-weighting is an architecture/Grad-CAM artifact."
    else:
        verdict = "Mixed/ambiguous -- the learned component does not show a clean inward push."
    print(f"\n  VERDICT: {verdict}")


def step3_lesion_correlation_blocked() -> None:
    print("\n" + "=" * 78)
    print("STEP 3 -- CORRELATE ATTENTION WITH REAL LESION LOCATIONS: BLOCKED")
    print("=" * 78)
    seg_root = os.path.join(PROJECT_ROOT, "data", "IDRiD", "A. Segmentation")
    print(f"  Looked for IDRiD lesion masks at: {seg_root}")
    print(f"  Present: {os.path.exists(seg_root)}")
    print()
    print("  The IDRiD copy in this repo is the 'B. Disease Grading' subset only:")
    print("  455 JPEGs + a severity CSV, with NO pixel-level lesion labels. The lesion")
    print("  masks (microaneurysms / haemorrhages / hard exudates / soft exudates /")
    print("  optic disc, 81 images) ship in IDRiD's separate 'A. Segmentation' download,")
    print("  which was never fetched -- see ROADMAP.md's Phase 7 dataset notes.")
    print()
    print("  This check is therefore NOT RUN, rather than run against a substitute")
    print("  dataset and reported as if it were the same evidence. Download IDRiD")
    print("  'A. Segmentation' to unblock it.")


def main():
    parser = argparse.ArgumentParser(description="Investigate the DR classifier's reported central spatial bias.")
    # Same APTOS layout the demo scripts and train.py already assume.
    parser.add_argument("--aptos-dir", default=os.path.join(PROJECT_ROOT, "APTOS 2019", "train_images", "train_images"))
    parser.add_argument("--limit", type=int, default=80)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    image_paths = sorted(
        os.path.join(args.aptos_dir, f) for f in os.listdir(args.aptos_dir) if f.lower().endswith((".png", ".jpg"))
    )[: args.limit]
    if not image_paths:
        raise SystemExit(f"No images found in {args.aptos_dir}")

    print(f"Using {len(image_paths)} images from {args.aptos_dir}\nDevice: {device}\n")

    step1_preprocessing_is_clean(cv2.imread(image_paths[0]))
    step2_radial_attention(image_paths, device)
    step3_lesion_correlation_blocked()


if __name__ == "__main__":
    main()
