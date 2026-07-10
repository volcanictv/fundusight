# VisionDx

AI-assisted retinal disease analysis pipeline. Educational/portfolio project — not a diagnostic device.

See `ROADMAP.md` for the phased build plan and `CLAUDE.md` for project conventions (used automatically by Claude Code).

## Setup

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
   classical heuristic (see `CLAUDE.md`).
5. For model training: install the CUDA build of torch/torchvision on a local
   NVIDIA GPU machine (see the comment at the top of `requirements.txt`) and
   run `src/detection/train.py` / `src/segmentation/vessel_train.py` /
   `src/segmentation/optic_disc_train.py` directly. Training on CPU is not
   practical.

## Trained weights

Not committed to git (see `CLAUDE.md` conventions) — regenerate with:
```
.venv\Scripts\python.exe src\detection\train.py --epochs 15
```
Saves the best checkpoint (by validation quadratic weighted kappa) to
`checkpoints/dr_efficientnet_b0.pth`. Current baseline (EfficientNet-B0, 15
epochs, RTX 4060): held-out test accuracy 83.9%, AUC 0.925, quadratic weighted
kappa 0.889.

The hybrid vessel segmentation model (see "Repo layout" — `src/segmentation/`)
is regenerated with:
```
.venv\Scripts\python.exe src\segmentation\vessel_train.py --epochs 150
```
Saves the best checkpoint (by validation clDice) to
`checkpoints/vessel_unet.pth`, trained on the pooled DRIVE/STARE/CHASE_DB1
labeled images (~68 total, 46/11/11 train/valid/test). Current baseline
(ShallowDilatedUNet, 150 epochs, RTX 4060): held-out test Dice 0.663,
clDice 0.832. Without this checkpoint, `vessels.segment_vessels()`'s
classical Frangi + hysteresis-threshold pipeline is used as a fallback.

The hybrid optic disc/cup segmentation model is regenerated with:
```
.venv\Scripts\python.exe src\segmentation\optic_disc_train.py --epochs 80
```
Saves the best checkpoint (by validation mean Dice over the disc-rim and
cup classes, background excluded) to `checkpoints/optic_disc_unet.pth`,
trained on REFUGE2's official 400-image train split (400/400 val/test).
Current baseline (OpticDiscUNet, 80 epochs, RTX 4060): held-out test
dice_rim 0.670, dice_cup 0.450. Without this checkpoint,
`optic_disc.compute_optic_biomarkers()`'s classical ONH-crop +
intensity-threshold pipeline is used as a fallback.

## Running Claude Code on this repo

From the repo root:
```
claude
```
Claude Code will automatically pick up `CLAUDE.md` for project context. Use Plan Mode (Shift+Tab) before large changes, and `/model opusplan` when you want Opus to plan and Sonnet to execute.

## Project structure

See the "Repo layout" section in `CLAUDE.md`.
