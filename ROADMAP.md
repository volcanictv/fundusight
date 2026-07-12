# Fundusight — Project Roadmap

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

Fix applied: `optic_disc_dataset.build_pooled_pairs()`/
`split_pooled_pairs()` pool all 1200 REFUGE2 images and re-split with
stratification by original folder, so every new split gets a
proportional mix of all three camera domains; `optic_disc_train.py`
trains on this pooled split by default. **Retrain complete** (2026-07-11,
80 epochs, pooled/re-split data: train=840 val=180 test=180): held-out
test Dice — `dice_rim=0.8937  dice_cup=0.8576  mean=0.8756` — vs. the old
domain-split checkpoint's held-out test Dice of `dice_rim=0.6696
dice_cup=0.4502  mean=0.5599`. The large jump (especially on the cup
class) confirms the camera/domain split, not model capacity, was the
bottleneck. `checkpoints/optic_disc_unet.pth` now holds the pooled-split
weights; the prior domain-split checkpoint is kept at
`checkpoints/optic_disc_unet.provisional_domainsplit.pth` for reference/
comparison, not used by inference.

The Dice numbers above are network-only, on ground-truth ROI crops (no
localization error, no post-processing) -- `scripts/
evaluate_optic_disc_full_pipeline.py` measures the harder, more realistic
number: the FULL pipeline (Stage 6.1 classical ONH localization -> Stage
6.2 -> mask cleanup -> Stage 6.3 CDR) against held-out images, and reports
CDR agreement, not just Dice. That script still pointed at REFUGE2's own
official test folder (`build_pairs()`) -- fixed to use the same pooled/
re-split held-out set (`seed=42`) the retrain actually used, since the
official folder is no longer a valid held-out set for this checkpoint
(most of its images are now inside the pooled training set). Full-pipeline
results (180 held-out images, 2026-07-11): `dice_rim=0.8414
dice_cup=0.8149` (lower than the network-only numbers above, as expected
once localization error and post-processing are included), mean predicted
vertical CDR=0.4766 vs. mean ground-truth CDR=0.4722, **mean absolute CDR
error=0.0571** (median=0.0368) -- close agreement, no major systematic
bias, unlike the old domain-split checkpoint (whose cup Dice of 0.45 would
have made its CDR unreliable, consistent with why the earlier calibration
attempts in this section failed).

**Macula/fovea heuristic validated against real ground truth for the first
time (2026-07-12) — result: unreliable, root cause identified.**
`locate_macula_classical()` had only ever been sanity-checked as "looks
about right relative to the disc," since REFUGE2 (its own training/eval
dataset) ships no fovea coordinate labels. ADAM does
(`Fovea_location.xlsx`, one (Fovea_X, Fovea_Y) per image), so
`scripts/evaluate_macula_localization.py` runs Stage 6.1's classical disc
localizer + the macula heuristic on all 400 ADAM Training400 images and
compares against it — converting predicted working-image-space coordinates
back to each image's own *original* pixel space first (ADAM ships two
different native resolutions, 2056x2124 and 1444x1444, so a fixed scale
factor would silently corrupt part of the comparison; same per-image-actual-
dimensions convention `evaluate_optic_disc_full_pipeline.py` and
`vessels._resize_to_working_width()` already use).

```
Raw Euclidean pixel error (original pixel space):     mean=634.6px  median=485.3px
Disc-diameter-normalized error:                        mean=3.679   median=3.327
Percentiles (normalized): p10=0.273 p25=0.988 p50=3.327 p75=5.942 p90=7.194 p95=8.125 p99=9.733 max=18.220
```

The heuristic's own docstring already admits it: "no reliable eye-laterality
info available," so it tries both sides of the disc and picks whichever is
darker. That guess is the dominant error source, confirmed by a side-of-disc
breakdown: it picks the **correct side in only 229/400 images (57%)** —
barely better than a coin flip — and all 10 worst outliers (9-18 disc
diameters of error) are wrong-side picks. Even restricted to the 229
correct-side images, median normalized error is still 1.318 disc diameters
— a real miss, not just noise, so the "darkest point in the window" logic
itself is imprecise even when pointed the right direction (plausibly pulled
off-target by vessels or, on the 89 true-AMD images specifically, AMD
lesions — which are themselves dark, macula-adjacent, and exactly what
Stage 6.1/6.2 were never trained to distinguish from the fovea). This is the
concrete version of the risk the heuristic's docstring already flagged in
the abstract: it was tuned/eyeballed only on REFUGE2-like framing and
degrades outside it. Not fixed here — this is a validation result, not a
fix; a real fix would need either eye-laterality metadata (not available in
REFUGE2 or ADAM) or a learned macula localizer, which no available dataset
currently has the labels to train.

**Done when:** disc mask, cup mask, and macula location are overlaid on a
sample image, a vertical cup-disc ratio is printed alongside them, the
cup-within-disc structural check passes (verified by a test), the
production model is retrained on the pooled/re-split data, and a
held-out test-split Dice score (from that new split) is reported for the
disc/cup segmentation model. **Status: done** — see retrain results above.
The macula/fovea heuristic itself is now known to be unreliable outside
REFUGE2-like framing (see validation above) — worth flagging in the app/
report if macula location is ever surfaced as more than an approximate
overlay.

## Phase 7 — Multi-disease + Multi-dataset (weeks 9-11)

- Extend classifier to glaucoma (REFUGE) and AMD.
- Cross-validate DR model against a second dataset (MESSIDOR or IDRiD) to demonstrate generalization — this is the detail that shows you understand why single-dataset results are weak evidence.

**Dataset decisions (locked in, don't re-litigate without asking):** AMD uses
ADAM (iChallenge-AMD, downloaded to `ADAM/Training400/`) — no separate
val/test ships publicly, only a 400-image labeled training set, so a split
will need to be carved out same as REFUGE2's Phase 6 pooling fix. Cross-dataset
DR validation uses IDRiD, not MESSIDOR (`data/IDRi/`, 455 of the official
516-image "B. Disease Grading" set — 88% complete, user has confirmed working
with what's present is fine). **Note:** a full duplicate of the IDRiD data
also exists at `IDRiD/` (repo root) — same 456 files as `data/IDRi/`, byte
identical `idrid_labels.csv`. Only `data/IDRi/` is referenced by any code;
the root-level `IDRiD/` duplicate hasn't been cleaned up (ask the user before
deleting either copy).

**Glaucoma classifier — code built and smoke-tested, real training NOT yet
run.** Architecture: EfficientNet-B0, same pattern as the DR classifier (not
the disc/cup U-Net — that's a separate Phase 6 model). Labels come from
`REFUGE2/glaucoma_labels_merged.csv`, built by `scripts/build_glaucoma_labels.py`
(merges REFUGE2's own `Refuge2_test.csv` — despite the name, it covers all
three REFUGE2 domains — with SMDG-19's REFUGE1-train/REFUGE1-val rows,
resolving internally-conflicting REFUGE2-csv duplicates via SMDG-19 where
covered; see that script's docstring for the exact merge rules). If
`REFUGE2/glaucoma_labels_merged.csv` is missing (REFUGE2/ is gitignored, so a
fresh clone won't have it), regenerate with:
```
.venv\Scripts\python.exe scripts\build_glaucoma_labels.py
```
requires `REFUGE2/` and `SMDG, A Standardized Fundus Glaucoma Dataset/` both
downloaded first (same non-committed-data convention as every other dataset
here — see README).

Code: `src/detection/glaucoma_dataset.py` (`build_pairs()`/`split_pairs()` —
mirrors `src/segmentation/optic_disc_dataset.py`'s pooled/stratified-split
pattern, but stratifies on a compound `{domain}_{label}` key, since glaucoma
prevalence (~85/15) is a second imbalance beyond REFUGE2's three-camera-domain
issue) and `src/detection/glaucoma_train.py` (mirrors
`src/segmentation/optic_disc_train.py`'s structure; reuses
`src/detection/model.py`'s `build_model(num_classes=2)` and
`src/detection/dataset.py`'s `build_transforms()` directly — no duplicated
model/transform code, which is also what keeps this swappable for the
RETFound stretch goal mentioned below). Tests: `tests/detection/test_glaucoma_dataset.py`
(5 tests, passing).

Real-data split (verified 2026-07-11): 998 pooled labeled pairs → train=698,
val=150, test=150, all three REFUGE2 camera domains represented in every
split.

**Real 30-epoch training run completed (2026-07-11, GPU: RTX 4060, ~5.5 min
total).** Model selection is by validation AUC-ROC per epoch; the best
checkpoint was reached at **epoch 6/30** (val AUC 0.7271) — after that, train
loss kept falling (0.40 → 0.10) while val AUC plateaued/declined, a clear
overfitting signal on the small 150-image validation split, so the later
epochs' weights were correctly never selected. Held-out **test set** results
(never touched during training or model selection):

```
accuracy=0.7400  auc=0.8304  f1=0.4179  sensitivity=0.7778  specificity=0.7348
confusion matrix (rows=true, cols=predicted):
[[97 35]   TN=97 FP=35
 [ 4 14]]  FN=4  TP=14
```

Sensitivity (77.8%) is notably higher than accuracy would suggest — the
inverse-frequency class-weighted loss trades precision (more false positives,
35) for fewer missed glaucoma cases (only 4 false negatives out of 18
positives), which is the appropriate tradeoff for a screening task. Test AUC
(0.8304) exceeding the best val AUC (0.7271) is plausible given how small and
noisy both splits are (150 images each, ~12% positive).
`checkpoints/glaucoma_efficientnet_b0.pth` now holds this real, fully-trained
checkpoint (not a smoke test).

**To retrain from a fresh session:**
```
.venv\Scripts\python.exe src\detection\glaucoma_train.py --epochs 30
```
(30 is the script's default — override with `--epochs N` for a different
budget; compare against the epoch-6 val AUC of 0.7271 / test AUC of 0.8304
above).

**AMD classifier — trained (2026-07-12, GPU: RTX 4060, ~2 min total).**
Architecture: EfficientNet-B0, same `build_model()`/`build_transforms()`
pattern as DR and glaucoma. Labels come straight from ADAM's own folder
structure — `Training400/AMD/` (89 images, label 1) vs `Training400/Non-AMD/`
(311 images, label 0) — no separate CSV needed. Since ADAM (unlike REFUGE2)
is a single camera domain, `src/detection/amd_dataset.py`'s `split_pairs()`
stratifies on the AMD/Non-AMD label alone (no domain-compound key needed).
Code: `src/detection/amd_dataset.py` + `src/detection/amd_train.py` (mirrors
`glaucoma_train.py`'s structure exactly). Tests: `tests/detection/test_amd_dataset.py`
(5 tests, passing).

Real-data split: 400 labeled images → train=279, val=61, test=60 (carved out
ourselves since ADAM ships no official val/test — same pooling-fix pattern as
REFUGE2's Phase 6 and Phase 7's glaucoma split). Val AUC climbed steadily
across all 30 epochs (0.6398 → 0.9666) with no overfitting plateau — unlike
glaucoma's early epoch-6 peak, this is a cleaner training curve, likely
because AMD/Non-AMD is a more visually separable task than glaucoma cup/disc
subtleties, and/or ADAM's images are more homogeneous (single domain vs
REFUGE2's three). Best checkpoint landed at **epoch 30/30** (val AUC 0.9666).
Held-out **test set** results (never touched during training or model
selection):

```
accuracy=0.9167  auc=0.8887  f1=0.8000  sensitivity=0.7692  specificity=0.9574
confusion matrix (rows=true, cols=predicted):
[[45  2]   TN=45 FP=2
 [ 3 10]]  FN=3  TP=10
```

`checkpoints/amd_efficientnet_b0.pth` holds this real, fully-trained
checkpoint. To retrain: `.venv\Scripts\python.exe src\detection\amd_train.py --epochs 30`.

**IDRiD DR cross-dataset validation — done (2026-07-12), evaluation only, no
training.** Runs the existing APTOS-trained `checkpoints/dr_efficientnet_b0.pth`
unmodified against IDRiD (`data/IDRi/`, 455 images, both IDRiD's official
train and test sets pooled — see `src/detection/idrid_dataset.py`'s docstring
for how the "test"-suffixed id_codes disambiguate IDRiD's own restarted
numbering). IDRiD uses the same 0-4 ICDR severity scale as APTOS, so class
indices transfer directly with no relabeling. Code:
`src/detection/idrid_dataset.py` (a small standalone loader, not a reuse of
`AptosDataset` — IDRiD ships JPEGs not PNGs, and its CSV carries extra
unnamed columns) + `src/detection/idrid_eval.py` (reuses `train.py`'s
`evaluate()` and `dataset.py`'s `build_transforms()` directly — no
metric-computation duplication). Tests: `tests/detection/test_idrid_dataset.py`
(3 tests, passing). Run with:
```
.venv\Scripts\python.exe src\detection\idrid_eval.py
```

**Results — the whole point of this exercise, a real generalization gap:**

```
                APTOS (in-domain, from README.md)   IDRiD (cross-dataset)
accuracy        83.9%                                54.3%
AUC (macro ovr) 0.925                                0.840
kappa (quad.)   0.889                                0.764
```

Confusion matrix (IDRiD, rows=true, cols=predicted):
```
[[ 73  46  10   0   0]
 [  4  16   2   0   0]
 [  4  20 118   6   8]
 [  0   1  59  13  11]
 [  0   0  23  14  27]]
```

Raw accuracy drops ~30 points out-of-domain, but AUC (0.840) and quadratic
weighted kappa (0.764, still "substantial agreement" on the Landis-Koch
scale) hold up much better — kappa in particular discounts near-miss errors
(e.g. predicting severity 2 for a true 3), and the confusion matrix shows
most of IDRiD's errors are exactly that kind of adjacent-class confusion,
not wild misses. This is the demonstration the roadmap called for: a
single-dataset accuracy number is weak evidence of real-world performance,
and the degradation pattern (ranking/ordinal signal surviving better than
hard classification accuracy) is itself informative about what a fundus-photo
domain shift (different camera hardware, population, lighting) does to a
CNN classifier.

**Stretch goal reminder (not a Phase 7 blocker):** later, add RETFound
(ViT-Large, MAE-pretrained on retinal images) as a comparison arm against
EfficientNet-B0 across DR/glaucoma/AMD, via partial fine-tuning (freeze most
of the backbone, unfreeze last few blocks + head — fits 8GB VRAM). Don't
build it yet, but the glaucoma classifier code above was deliberately kept
model-agnostic (`build_model(num_classes=...)` as the only task-specific
call) so RETFound can slot in later without a rewrite.

**Glaucoma + AMD wired into inference and the app (2026-07-12).** Both
classifiers were trained but only reachable via their training scripts
until now. `src/detection/glaucoma_infer.py` / `amd_infer.py` mirror
`infer.py`'s `load_model()`/`predict()` contract exactly. `report/pipeline.py`
runs all three classifiers (DR/glaucoma/AMD) through one shared
`_run_classifier()` helper (each gets its own Grad-CAM overlay, same as DR)
and `STAGE_NAMES` now includes `"glaucoma"`/`"amd"`. `report/content.py`
gained a shared `_binary_classifier_sections()` (glaucoma and AMD are
identical in shape, unlike DR's 5-class layout) plus a **disagreement
check**: the optic-disc section's existing elevated-CDR observation and the
new glaucoma classifier are now two independent glaucoma-relevant signals
on the same report, so the Recommendation text explicitly flags it when
they point different directions rather than silently favoring one. No
changes were needed to `report/pdf.py` or `app/render_preview.py` — both
are already `Section.kind`-driven and disease-agnostic, confirming that
architecture choice paid off. Verified end-to-end in the real running app
(not just tests): both sections render correctly with real predictions,
correct pill colors, and Grad-CAM overlays; the disagreement note correctly
appears/doesn't appear depending on whether the two signals agree; the
downloaded PDF includes both new sections.

**Done when:** you have probability scores for all three diseases from one uploaded image. **Status: done.**

## Phase 8 — Report Generation (week 11-12)

- ReportLab PDF: patient ID, quality score, disease probabilities, vessel/CDR measurements, attention map thumbnails, recommendation text.

**Done when:** uploading an image produces a downloadable PDF report. **Status:
done** — `src/report/pipeline.py` orchestrates quality/preprocessing-preview/
detection+Grad-CAM/vessel/optic-disc into one dict (degrading gracefully, not
crashing, when no DR checkpoint is present — there's no classical fallback for
detection unlike vessels/optic-disc); `src/report/content.py` turns that into a
renderer-agnostic `ReportContent`; `src/report/pdf.py` renders it to a
print-ready A4 PDF (ReportLab, fixed margins, embedded thumbnails, page
footer/disclaimer).

## Phase 9 — Dashboard (weeks 12-13)

- Streamlit or Gradio app tying everything together: upload → quality → detection → heatmap → measurements → report download.
- Plotly for the probability bars / metrics.

**Done when:** a stranger can upload a fundus photo and get the full pipeline
output through the UI, no code required. **Status: done** — see
`src/app/main.py`: upload or a demo-mode toggle (reads locally-downloaded
APTOS sample images only, never bundled — see the Datasets table's licensing
note) → quality → before/after preprocessing → detection + Grad-CAM + an
ordinal-ramp Plotly probability chart → vessel biomarkers → optic disc/CDR →
an in-app "generation preview" (`src/app/render_preview.py`) that mirrors the
PDF exactly, built from the same `ReportContent` → PDF download. The PDF's A4
layout is the primary print path; a light `@media print` rule
(`src/app/theme.py`) also cleans up the live preview screen for a direct
browser print.

---

## Stretch goals (ongoing, after v1 works)

- Model comparison view (EfficientNet vs. DenseNet vs. ConvNeXt vs. ViT — accuracy/AUC/F1/precision/recall side by side).
- Explainability comparison grid (original / Grad-CAM / EigenCAM / LayerCAM).
- Uncertainty estimation (MC Dropout or a small ensemble) instead of a bare probability.
- Longitudinal tracking if you simulate/collect multi-visit data (lesion count, vessel density, CDR, severity over time).

## Datasets

| Dataset | Use for | Notes |
|---|---|---|
| APTOS 2019 | DR grading | Best starting point — single CSV, ~3.6k images, Kaggle-licensed and not redistributable, so it's gitignored locally and the dashboard's demo mode (Phase 9) only ever reads it off disk, never bundles a copy |
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
