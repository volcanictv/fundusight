# Fundusight

An AI-assisted retinal disease analysis pipeline: fundus photo → image
quality check → preprocessing → disease detection (diabetic retinopathy,
glaucoma, AMD) → explainability (Grad-CAM/EigenCAM/LayerCAM) → vessel and
optic-disc/cup biomarkers → PDF report, tied together by a Streamlit
dashboard.

**Live demo:** https://fundusight-main.streamlit.app/ (try it with "Demo
mode" — no upload needed)

**This is an educational/portfolio project, not a diagnostic device.** No
output here should be treated as clinical advice.

This README is a snapshot of what's built and what it actually measures.
`DEEP_DIVE.md` has longer write-ups of specific investigations referenced
below.

## What this project demonstrates

Three findings came out of building this that are more interesting than any
single accuracy number:

1. **A dataset's official train/val/test split can silently be a
   camera/domain split, not a random sample — and that alone can make a
   model look far worse than it is.** REFUGE2's three official folders
   turned out to be three different cameras/sites (uniform resolutions per
   folder: 2056x2124 / 1940x1940 / 1634x1634, and very different color
   statistics — mean blue channel 21.4 / 14.9 / 56.5). Training the optic
   disc/cup U-Net on the official split gave test Dice of only 0.560 (cup
   Dice 0.450) despite a healthy validation score, because validation and
   test were effectively different domains. Pooling all 1200 images and
   re-splitting with stratification by original folder (so every split gets
   a proportional mix of all three cameras) took mean Dice from 0.560 to
   **0.876** with the exact same architecture and training budget — the
   split, not the model, was the bottleneck. Two independent post-hoc
   threshold-recalibration attempts on the original split both failed to
   fix this (the second, more careful one failed *worse*), which is itself
   informative: recalibration can't undo a domain mismatch baked into the
   split.

2. **A classical vessel filter's tuning constants are coupled to a specific
   working resolution, and that coupling has to be enforced in code, not
   left as a convention.** Frangi vesselness sigmas are absolute pixel
   scales, but incoming fundus photos arrive at wildly different native
   resolutions (e.g. 1736x2416 vs 1050x1050). The same sigma range means a
   different physical vessel width depending on which photo came in unless
   every image is first resized to one canonical working width
   (`VESSEL_WORKING_WIDTH = 1400` in `src/segmentation/vessels.py`) before
   the filter runs — every other pixel-unit constant in that module
   (Frangi sigma range, minimum object size for speckle removal) is tuned
   for that specific resolution and doesn't rescale automatically if it
   changes.

3. **Raw accuracy is a weak signal for cross-dataset generalization —
   ranking/ordinal metrics survive a domain shift much better.** The
   APTOS-trained DR classifier drops from 83.9% to 54.3% accuracy when
   evaluated unmodified on IDRiD (different camera hardware/population/
   lighting), which looks like the model barely works out-of-domain. But
   AUC only drops from 0.925 to 0.840 and quadratic weighted kappa from
   0.889 to 0.764 (still "substantial agreement" on the Landis-Koch scale)
   — the confusion matrix shows most of the damage is adjacent-class
   confusion (e.g. predicting severity 2 for a true 3), not wild misses.
   A single-dataset accuracy number would have hidden this distinction
   entirely.

A fourth result worth knowing about even though it isn't a "fix": the
classical macula/fovea localization heuristic was validated against real
ADAM ground truth (2026-07-12) for the first time and found to be
**unreliable outside REFUGE2-like framing** — it guesses which side of the
optic disc the fovea is on (no eye-laterality metadata is available in
either REFUGE2 or ADAM) and gets it right only 57% of the time, barely
better than chance. See `DEEP_DIVE.md` for the full investigation. This is
flagged in-repo, not silently shipped as if it were reliable.

## Pipeline stages and real results

| Stage | Approach | Result |
|---|---|---|
| Image quality | Classical CV: variance of Laplacian (blur) + exposure histogram stats | Pass/fail + 0-100 score per image |
| Preprocessing | Illumination correction (Gaussian background subtraction) + CLAHE on green/luminance channel | Visible before/after contrast improvement |
| DR detection | EfficientNet-B0 fine-tuned on APTOS 2019 (5-class severity) | In-domain (APTOS): accuracy 83.9%, AUC 0.925, quadratic weighted kappa 0.889. Cross-dataset (IDRiD, no retraining): accuracy 54.3%, AUC 0.840, kappa 0.764 — see finding #3 above |
| Explainability | Grad-CAM / EigenCAM / LayerCAM (`pytorch-grad-cam`) over the DR classifier | Heatmaps verified to land on lesions, not image border/vignette artifacts |
| Uncertainty | Monte-Carlo Dropout: 20 stochastic forward passes (dropout left active at inference) per classifier, in one batched pass | Reports the top class's probability ± 1σ (e.g. "74% ± 4%") in the app tiles and PDF. Approximate *epistemic* uncertainty from the model's own disagreement under dropout — not calibrated probability, and labelled as such |
| Vessel segmentation | Hybrid: classical Frangi vesselness feeds a small dilated-conv U-Net trained on DRIVE/STARE/CHASE_DB1 (Dice+clDice loss); classical-only is the fallback with no checkpoint | Held-out test Dice 0.663, clDice 0.832 (hybrid). See finding #2 above for the resolution-coupling bug this pipeline has to guard against |
| Optic disc/cup + CDR | Hybrid: classical ONH localization/crop feeds a U-Net trained on pooled/re-split REFUGE2 (RGB+Lab+HSV channels, CrossEntropy+Dice loss); classical intensity-threshold is the fallback | Network-only held-out test Dice: rim 0.894, cup 0.858 (mean 0.876). Full pipeline (localization + network + postprocessing): Dice rim 0.841, cup 0.815, mean absolute CDR error 0.057. See finding #1 above for the domain-split bug this result depends on |
| Macula/fovea | Classical heuristic only (no dataset used here ships fovea labels at training volume) | Validated against ADAM ground truth: median error 3.3 disc diameters, root cause = coin-flip eye-laterality guessing (57% correct). Known-unreliable outside REFUGE2-like framing — see "What this project demonstrates" above |
| Glaucoma detection | EfficientNet-B0 fine-tuned on pooled REFUGE2 + SMDG-19 glaucoma labels, classifying an **optic-nerve-head crop** (not the full photo) so attention stays on the disc rather than edge/hemorrhage artifacts | Held-out test (n=150, 18 positives): accuracy 86.7%, AUC 0.827, F1 0.444, sensitivity 44.4%, specificity 92.4%. The small positive count makes threshold-0.5 point metrics noisy — AUC (ranking) is the more stable signal here; see `DEEP_DIVE.md` for the ONH-crop attention fix this row depends on |
| AMD detection | EfficientNet-B0 fine-tuned on ADAM (AMD vs Non-AMD) | Held-out test: accuracy 91.7%, AUC 0.889, F1 0.800, sensitivity 76.9%, specificity 95.7% |
| Report generation | ReportLab PDF, driven by one shared `ReportContent` model | Patient ID, quality score, all three disease probabilities, vessel/CDR measurements, Grad-CAM thumbnails, recommendation text (including a disagreement flag when the CDR-based glaucoma signal and the glaucoma classifier disagree) |
| Dashboard | Streamlit + Plotly | Upload or demo mode → quality → preprocessing preview → detection+Grad-CAM → biomarkers → in-app report preview → PDF download |

All metrics above are held-out test-set results, not cherry-picked from
validation.

## Trained weights

Checkpoints are gitignored (large binary files, regenerable). Regenerate
any of them locally with the training scripts below.

| Checkpoint | Regenerate with | Held-out test result |
|---|---|---|
| `checkpoints/dr_efficientnet_b0.pth` | `src\detection\train.py --epochs 15` | accuracy 83.9%, AUC 0.925, kappa 0.889 |
| `checkpoints/glaucoma_efficientnet_b0.pth` | `src\detection\glaucoma_train.py --epochs 30` | accuracy 86.7%, AUC 0.827, F1 0.444 |
| `checkpoints/amd_efficientnet_b0.pth` | `src\detection\amd_train.py --epochs 30` | accuracy 91.7%, AUC 0.889, F1 0.800 |
| `checkpoints/vessel_unet.pth` | `src\segmentation\vessel_train.py --epochs 150` | Dice 0.663, clDice 0.832 |
| `checkpoints/optic_disc_unet.pth` | `src\segmentation\optic_disc_train.py --epochs 80` | dice_rim 0.894, dice_cup 0.858 |

Without a given checkpoint, vessel and optic-disc/cup biomarkers fall back
to their classical pipelines automatically; DR/glaucoma/AMD detection have
no classical fallback and simply don't appear in the report/app.

## Running the app

```
.venv\Scripts\python.exe -m streamlit run src\app\main.py
```
Opens the Streamlit dashboard. Upload a fundus photo, or turn on "Demo
mode" to try it against a locally available APTOS 2019 sample instead.

## Running tests

```
.venv\Scripts\python.exe -m pytest
```
Runs the full suite (`tests/`, mirrors `src/` structure). Tests use small
real or synthetic sample images and don't require the full datasets to be
downloaded or hit the network.

## Project structure

```
src/
  preprocessing/   quality assessment, CLAHE, illumination correction
  detection/       model loading, inference, training; MC-Dropout uncertainty
  explainability/  Grad-CAM / EigenCAM / LayerCAM wrappers
  segmentation/    vessel biomarkers + optic disc/cup localization & CDR
  report/          PDF report generation + shared content model
  app/             Streamlit dashboard
data/              not committed — datasets are downloaded locally, gitignored
tests/             unit tests, mirrors src/ structure
```
