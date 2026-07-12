# Fundusight

An AI-assisted retinal disease analysis pipeline: fundus photo → image
quality check → preprocessing → disease detection (diabetic retinopathy,
glaucoma, AMD) → explainability (Grad-CAM/EigenCAM/LayerCAM) → vessel and
optic-disc/cup biomarkers → PDF report, tied together by a Streamlit
dashboard.

**This is an educational/portfolio project, not a diagnostic device.** No
output here should be treated as clinical advice.

This README is a snapshot of what's built and what it actually measures.
`ROADMAP.md` has the full phase-by-phase build log, and `DEEP_DIVE.md` has
longer write-ups of specific investigations referenced below. `CLAUDE.md` is
a dev-history/conventions doc from when this was built with Claude Code —
useful background, not an active instruction file.

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
| Vessel segmentation | Hybrid: classical Frangi vesselness feeds a small dilated-conv U-Net trained on DRIVE/STARE/CHASE_DB1 (Dice+clDice loss); classical-only is the fallback with no checkpoint | Held-out test Dice 0.663, clDice 0.832 (hybrid). See finding #2 above for the resolution-coupling bug this pipeline has to guard against |
| Optic disc/cup + CDR | Hybrid: classical ONH localization/crop feeds a U-Net trained on pooled/re-split REFUGE2 (RGB+Lab+HSV channels, CrossEntropy+Dice loss); classical intensity-threshold is the fallback | Network-only held-out test Dice: rim 0.894, cup 0.858 (mean 0.876). Full pipeline (localization + network + postprocessing): Dice rim 0.841, cup 0.815, mean absolute CDR error 0.057. See finding #1 above for the domain-split bug this result depends on |
| Macula/fovea | Classical heuristic only (no dataset used here ships fovea labels at training volume) | Validated against ADAM ground truth: median error 3.3 disc diameters, root cause = coin-flip eye-laterality guessing (57% correct). Known-unreliable outside REFUGE2-like framing — see "What this project demonstrates" above |
| Glaucoma detection | EfficientNet-B0 fine-tuned on pooled REFUGE2 + SMDG-19 glaucoma labels | Held-out test: accuracy 74.0%, AUC 0.830, F1 0.418, sensitivity 77.8%, specificity 73.5% (class-weighted loss deliberately trades precision for fewer missed positives — appropriate for a screening task) |
| AMD detection | EfficientNet-B0 fine-tuned on ADAM (AMD vs Non-AMD) | Held-out test: accuracy 91.7%, AUC 0.889, F1 0.800, sensitivity 76.9%, specificity 95.7% |
| Report generation | ReportLab PDF, driven by one shared `ReportContent` model | Patient ID, quality score, all three disease probabilities, vessel/CDR measurements, Grad-CAM thumbnails, recommendation text (including a disagreement flag when the CDR-based glaucoma signal and the glaucoma classifier disagree) |
| Dashboard | Streamlit + Plotly | Upload or demo mode → quality → preprocessing preview → detection+Grad-CAM → biomarkers → in-app report preview → PDF download |

All metrics above are held-out test-set results, reported alongside their
training/validation numbers in `ROADMAP.md` — not cherry-picked from
validation.

## Setup

0. Clone the repo:
   ```
   git clone https://github.com/volcanictv/fundusight.git
   cd fundusight
   ```
1. Python 3.10+
   ```
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Get a free Kaggle account and download the [APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection) dataset into `APTOS 2019/` at the repo root, with `train_1.csv`/`valid.csv`/`test.csv` and matching `train_images/`/`val_images/`/`test_images/` folders (not committed to git — see `.gitignore`).
3. For the hybrid vessel segmentation model, download DRIVE, STARE, and
   CHASE_DB1 into `DRIVE/`, `STARE/`, and `CHASE_DB1/` at the repo root
   (also not committed to git). Expected layout:
   - `DRIVE/training/images/*.tif` + `DRIVE/training/1st_manual/*.gif`
     (the `DRIVE/test/` split has no vessel ground truth in the standard
     download, so it isn't used for training).
   - `STARE/stare-images/*.ppm.gz` + `STARE/labels-ah/*.ppm.gz` (loaded
     directly from the gzip-compressed originals — no manual decompression
     needed).
   - `CHASE_DB1/Images/*.jpg` + `CHASE_DB1/Masks/*_1stHO.png`.
4. For the hybrid optic disc/cup segmentation model, download REFUGE2 into
   `REFUGE2/` at the repo root (also not committed to git). Expected
   layout: `REFUGE2/{train,val,test}/images/*.jpg` +
   `REFUGE2/{train,val,test}/mask/*.bmp` (train/test) or `*.png` (val).
   Masks use pixel values `{0=cup, 128=disc rim, 255=background}`; REFUGE2
   ships no fovea/macula coordinate labels, so macula localization stays a
   classical heuristic (see "What this project demonstrates" above).
5. For the glaucoma classifier, also download SMDG-19 ("SMDG, A
   Standardized Fundus Glaucoma Dataset") to the repo root, then generate
   merged labels:
   ```
   .venv\Scripts\python.exe scripts\build_glaucoma_labels.py
   ```
6. For the AMD classifier, download ADAM (iChallenge-AMD) into `ADAM/` at
   the repo root — labels come directly from its `Training400/AMD/` vs
   `Training400/Non-AMD/` folder structure, no separate CSV needed.
7. For cross-dataset DR validation, download IDRiD into `data/IDRi/`.
8. For model training: install the CUDA build of torch/torchvision on a
   local NVIDIA GPU machine (see the comment at the top of
   `requirements.txt`) and run `src/detection/train.py` /
   `src/detection/glaucoma_train.py` / `src/detection/amd_train.py` /
   `src/segmentation/vessel_train.py` /
   `src/segmentation/optic_disc_train.py` directly. Training on CPU is not
   practical.

## Trained weights

Checkpoints are gitignored (large binary files, regenerable — see
`CLAUDE.md`). Regenerate any of them locally with the training scripts
below, or see "Deployment" below for how a deployed instance fetches
pre-trained weights instead of retraining from scratch.

| Checkpoint | Regenerate with | Held-out test result |
|---|---|---|
| `checkpoints/dr_efficientnet_b0.pth` | `src\detection\train.py --epochs 15` | accuracy 83.9%, AUC 0.925, kappa 0.889 |
| `checkpoints/glaucoma_efficientnet_b0.pth` | `src\detection\glaucoma_train.py --epochs 30` | accuracy 74.0%, AUC 0.830, F1 0.418 |
| `checkpoints/amd_efficientnet_b0.pth` | `src\detection\amd_train.py --epochs 30` | accuracy 91.7%, AUC 0.889, F1 0.800 |
| `checkpoints/vessel_unet.pth` | `src\segmentation\vessel_train.py --epochs 150` | Dice 0.663, clDice 0.832 |
| `checkpoints/optic_disc_unet.pth` | `src\segmentation\optic_disc_train.py --epochs 80` | dice_rim 0.894, dice_cup 0.858 |

Without a given checkpoint, vessel and optic-disc/cup biomarkers fall back
to their classical pipelines automatically; DR/glaucoma/AMD detection have
no classical fallback and simply don't appear in the report/app.

## Deployment

This is meant to be deployed once, as a live portfolio demo, via [Streamlit
Community Cloud](https://share.streamlit.io), which builds an app straight
from a GitHub repository URL — no separate server or hosting account to
manage. Since checkpoints are never committed to git (see "Trained
weights" above), a fresh deploy starts with none locally; the app fetches
them itself from this repo's GitHub Releases assets.

**How the checkpoint fetch works:** `src/app/main.py` calls
`fetch_checkpoints()` (`src/app/checkpoints.py`) once per process on
startup, which downloads the five checkpoints inference needs into
`checkpoints/` — existing local files are never re-fetched, so a dev
machine that already trained its own checkpoints makes zero network calls.
A failed fetch (offline, release not published yet) degrades the same way a
missing checkpoint always has — the affected section falls back or
disappears, not a crash.

**Steps to deploy this repo yourself, end to end:**

1. Push the repo to GitHub (already done here:
   `https://github.com/volcanictv/fundusight`).
2. Publish trained checkpoints as GitHub Release assets — one-time, this is
   what step "1" above downloads from:
   ```
   gh release create v1.0.0 checkpoints/dr_efficientnet_b0.pth \
       checkpoints/glaucoma_efficientnet_b0.pth checkpoints/amd_efficientnet_b0.pth \
       checkpoints/vessel_unet.pth checkpoints/optic_disc_unet.pth \
       --title "v1.0.0" --notes "Fundusight v1.0.0 trained checkpoints"
   ```
   (Requires the [`gh` CLI](https://cli.github.com/), logged in, run from
   the repo root with the checkpoints present locally.)
3. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
   GitHub.
4. Click **"New app"** and point it at this repo by URL:
   - **Repository:** `volcanictv/fundusight`
   - **Branch:** `master`
   - **Main file path:** `src/app/main.py`
5. Click **"Deploy"**. Streamlit Cloud clones the repo from that GitHub
   URL, installs `requirements.txt`, and runs the app; on first load it
   fetches the five checkpoints from the GitHub Release created in step 2
   automatically — nothing to configure by hand beyond the repo URL itself.

To pre-fetch checkpoints outside the app (e.g. to test the fetch locally
before deploying) instead of relying on the app's own startup fetch:
```
.venv\Scripts\python.exe scripts\fetch_checkpoints.py
```

GitHub Releases was chosen over Hugging Face Hub for hosting the
checkpoints: the repo already lives on GitHub, every checkpoint here is
well under GitHub's 2GB per-asset limit, and it avoids adding a second
hosting account/dependency for five files.

## Running the app

```
.venv\Scripts\python.exe -m streamlit run src\app\main.py
```
Opens the Streamlit dashboard. Upload a fundus photo, or turn on "Demo
mode" to try it against a locally available APTOS 2019 sample instead —
demo mode needs that dataset already downloaded per setup step 2 above.

## Running tests

```
.venv\Scripts\python.exe -m pytest
```
Runs the full suite (`tests/`, mirrors `src/` structure). Tests use small
real or synthetic sample images and don't require the full datasets to be
downloaded or hit the network.

## Project structure

See the "Repo layout" section in `CLAUDE.md`.
