# VisionDx — Project Roadmap

AI-assisted retinal disease analysis pipeline. Educational/portfolio project — not a diagnostic device, not for clinical use.

## Ground rules

- Build one vertical slice end-to-end before adding breadth. A working DR-only pipeline beats five half-built disease models.
- Train on a local NVIDIA GPU via `src/detection/train.py`. Don't try to fine-tune CNNs on a CPU.
- Use Claude Code with Plan Mode (Shift+Tab) for anything touching more than 2-3 files. Sonnet for implementation, Opus for planning/architecture calls (`/model opusplan`).
- Commit to git after every working milestone. `/clear` your Claude Code session between unrelated phases.
- Every phase below ends with something you can run and look at — not just code that compiles.

---

## Phase 0 — Setup (few days)

- Install Claude Code, connect it to this repo, read through its basics (`/help`, Plan Mode, `/model`).
- Set up Python env (3.10+), install `requirements.txt`.
- Create a free Kaggle account, download APTOS 2019 Blindness Detection dataset.

**Done when:** you can load and display a handful of APTOS images locally.

## Phase 1 — Image Quality Assessment (weeks 1-2)

Classical CV, no ML/GPU needed.

- Focus: variance of Laplacian (blurry images score low).
- Exposure/illumination: histogram statistics (over/under-exposed detection).
- Combine into a single quality score (0-100%) with pass/fail recommendation.

**Done when:** feeding an image in returns a quality score + which checks passed/failed.

## Phase 2 — Preprocessing (week 2-3)

- Illumination correction (e.g., background subtraction via large Gaussian blur).
- CLAHE (contrast-limited adaptive histogram equalization) on the green/luminance channel.
- Color normalization across the dataset.

**Done when:** you can show a before/after grid of raw vs. preprocessed images and the difference is visible.

## Phase 3 — DR Detection (weeks 3-5)

- Fine-tune a pretrained EfficientNet-B0 on APTOS labels (5-class DR severity) —
  locally on a GPU if you have one (`src/detection/train.py`), otherwise in Colab.
- Track accuracy/AUC/quadratic weighted kappa on a held-out validation split during
  training, then report the same on the untouched test split — don't skip this,
  it's your first real result.
- Load the trained weights in the local inference pipeline (`src/detection/infer.py`).
- Output: probability + severity label (e.g., "Moderate NPDR, 94.2%").

**Done when:** you can run inference on a new image locally and get a probability + severity out.

## Phase 4 — Explainability (week 5-6)

- Wire up `pytorch-grad-cam`: Grad-CAM first, then LayerCAM/EigenCAM.
- Overlay heatmap on the original fundus image.
- Sanity-check: does the heatmap actually land on lesions, or is it attending to the image border? (A common bug — worth explicitly checking.)

**Done when:** for a positive DR case, the heatmap visibly highlights a lesion region, not noise.

## Phase 5 — Vessel Segmentation (weeks 6-8)

Started as classical CV (can run in parallel with Phase 3/4 training), now
upgraded to a hybrid classical+learned pipeline once the classical
Frangi-only mask proved to under-segment thin peripheral vessels.

- Classical stage (done): green channel → CLAHE → multi-scale Frangi
  vesselness filter → hysteresis threshold → skeletonize. Still the
  fallback path when no trained weights are available.
- Hybrid stage: the raw Frangi vesselness response (not thresholded) is fed
  as an extra input channel, alongside the CLAHE'd green channel, into a
  small dilated-convolution U-Net trained on DRIVE/STARE/CHASE_DB1 (APTOS
  has no pixel-level vessel labels) with a Dice + clDice loss, which learns
  to refine the Frangi response rather than segmenting from raw pixels
  alone.
- Compute vessel density, branching count, tortuosity, average width from
  the skeleton — same four biomarkers, now computed from whichever mask
  (classical or hybrid) is in use.

**Done when:** the hybrid model's vessel mask visibly recovers thin
peripheral branches the classical Frangi-only mask missed, on the same
sample image, with a held-out Dice/clDice score reported from training on
DRIVE/STARE/CHASE_DB1.

## Phase 6 — Optic Disc / Cup / Macula Detection (week 8-9)

A two-stage hybrid pipeline, same shape as Phase 5's classical+learned
split: a cheap classical stage handles localization/cropping, a trained
model handles the hard segmentation decision within that crop.

- Stage 6.1 (classical): locate the optic nerve head (ONH) as the center
  of the brightest disc-sized *compact* patch within the field-of-view
  mask (a windowed-average-brightness peak, not a global brightness
  threshold — a global threshold breaks on real fundus photos, where
  diffuse lesions or an illumination gradient can out-connect the actual
  disc), then crop a square region-of-interest (ROI) around it sized as a
  multiple of the estimated disc diameter. This ROI crop is what makes
  Stage 6.2 tractable — the disc is a small fraction of a full fundus
  photo, and training a segmenter directly on full images would be
  dominated by background/easy negatives. Still the fallback path
  (including a simple intensity-based disc/cup estimate) when no trained
  weights are available.
- Stage 6.2 (hybrid): a small U-Net trained on the REFUGE2 dataset (1200
  labeled images, pooled and re-split — see "Known issue" below, not used
  via its own official train/val/test folders) performs 3-class semantic
  segmentation (background / disc rim / cup) on the ROI crop. Input
  channels combine RGB with Lab (a, b) and HSV (H, S) — multiple color
  spaces sharpen the pallor-based cup/disc boundary that's subtle in RGB
  alone, a standard technique in the optic cup segmentation literature.
  Trained with combined cross-entropy + multi-class soft Dice loss (not
  clDice — that's specifically for thin tubular structures like vessels,
  not compact blobs like disc/cup).
- Stage 6.3 (geometric): compute the vertical cup-to-disc ratio (CDR) from
  the predicted disc/cup masks, enforcing the structural constraint that
  the cup mask must reside entirely within the disc mask (independent
  per-class postprocessing could otherwise violate this — verified by a
  test that deliberately constructs a violating cup mask). Macula/fovea
  location is estimated with a classical heuristic — REFUGE2 has no fovea
  coordinate labels, so this stays unlearned: the darkest compact region
  within an annular band at ~2.5x disc diameter from the disc center,
  searched along the horizontal meridian.

**Known issue — REFUGE2's official split is a 3-way camera/domain split,
not a random sample:** direct image inspection (no metadata file ships
with the dataset) found each of REFUGE2's train/val/test folders is a
single UNIFORM resolution with zero mixing (train 2056x2124, val
1940x1940, test 1634x1634, 400/400 each) and dramatically different mean
color statistics (mean blue channel: train 21.4, val 14.9, test 56.5) —
each folder is effectively one clinical site/camera, not a mixture. This
explains why the first trained checkpoint's validation performance wasn't
predictive of its test performance, and why two independent, correctly
cross-validated post-hoc probability-threshold recalibration attempts
(see `scripts/calibrate_optic_disc_thresholds.py`) both failed to
transfer from validation's domain to test's domain — the second, more
careful attempt failed *worse* than the first. Ground-truth CDR
(disease-severity proxy) is comparable across splits, so this is a
camera/domain issue, not a severity-composition one.

Fix in progress: `optic_disc_dataset.build_pooled_pairs()`/
`split_pooled_pairs()` now pool all 1200 REFUGE2 images and re-split with
stratification by original folder, so every new split gets a
proportional mix of all three camera domains; `optic_disc_train.py`
trains on this pooled split by default. A 1-epoch smoke test on the new
pipeline completed cleanly (train/val/test batch counts matching the new
~840/180/180 split, no runtime errors). **Retraining the production
checkpoint on the pooled split is still outstanding** — deliberately not
run yet (compute not committed). The current `checkpoints/
optic_disc_unet.pth` in README was trained on the old, domain-split data
and should be treated as provisional until the retrain runs and its
held-out numbers (from the new, domain-mixed test split) are reported here.

**Done when:** disc mask, cup mask, and macula location are overlaid on a
sample image, a vertical cup-disc ratio is printed alongside them, the
cup-within-disc structural check passes (verified by a test), the
production model is retrained on the pooled/re-split data, and a
held-out test-split Dice score (from that new split) is reported for the
disc/cup segmentation model.

## Phase 7 — Multi-disease + Multi-dataset (weeks 9-11)

- Extend classifier to glaucoma (REFUGE) and AMD.
- Cross-validate DR model against a second dataset (MESSIDOR or IDRiD) to demonstrate generalization — this is the detail that shows you understand why single-dataset results are weak evidence.

**Done when:** you have probability scores for all three diseases from one uploaded image.

## Phase 8 — Report Generation (week 11-12)

- ReportLab PDF: patient ID, quality score, disease probabilities, vessel/CDR measurements, attention map thumbnails, recommendation text.

**Done when:** uploading an image produces a downloadable PDF report.

## Phase 9 — Dashboard (weeks 12-13)

- Streamlit or Gradio app tying everything together: upload → quality → detection → heatmap → measurements → report download.
- Plotly for the probability bars / metrics.

**Done when:** a stranger can upload a fundus photo and get the full pipeline output through the UI, no code required.

---

## Stretch goals (ongoing, after v1 works)

- Model comparison view (EfficientNet vs. DenseNet vs. ConvNeXt vs. ViT — accuracy/AUC/F1/precision/recall side by side).
- Explainability comparison grid (original / Grad-CAM / EigenCAM / LayerCAM).
- Uncertainty estimation (MC Dropout or a small ensemble) instead of a bare probability.
- Longitudinal tracking if you simulate/collect multi-visit data (lesion count, vessel density, CDR, severity over time).

## Datasets

| Dataset | Use for | Notes |
|---|---|---|
| APTOS 2019 | DR grading | Best starting point — single CSV, ~3.6k images, Kaggle |
| EyePACS | DR grading | Much larger, use once pipeline works on APTOS |
| REFUGE2 | Optic disc/cup segmentation, CDR | 1200 images; official train/val/test folders (400 each) are each a single camera/site domain (see Phase 6's "Known issue"), so pooled and re-split instead; masks use pixel values {0=cup, 128=disc rim, 255=background}; no fovea/macula coordinate labels |
| MESSIDOR | DR | Good cross-dataset validation set |
| IDRiD | Lesion segmentation + grading | More granular labels |
| DDR | Multi-lesion DR | Broader lesion types |
| DRIVE | Vessel segmentation (pixel labels) | 40 images; only the 20-image training split ships vessel ground truth in the standard download — the test split has images + FOV masks but no vessel labels |
| STARE | Vessel segmentation (pixel labels) | 20 images, hand-labeled by two independent experts; ships gzip-compressed (`.ppm.gz`) |
| CHASE_DB1 | Vessel segmentation (pixel labels) | 28 images, two independent manual segmentations |

## Compute notes

- Training: local NVIDIA GPU — install the CUDA build of torch/torchvision
  (see `requirements.txt`) and run `src/detection/train.py` /
  `src/segmentation/vessel_train.py` directly.
- Inference/app: runs fine on CPU once models are trained — no GPU needed for the Streamlit demo.
