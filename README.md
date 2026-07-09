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
3. For model training: if you have a local NVIDIA GPU, install the CUDA build of
   torch/torchvision (see the comment at the top of `requirements.txt`) and run
   `src/detection/train.py` directly. Otherwise use a Colab or Kaggle notebook
   with a free GPU runtime — see `notebooks/`. Training locally on CPU is not
   practical either way.

## Trained weights

Not committed to git (see `CLAUDE.md` conventions) — regenerate with:
```
.venv\Scripts\python.exe src\detection\train.py --epochs 15
```
Saves the best checkpoint (by validation quadratic weighted kappa) to
`checkpoints/dr_efficientnet_b0.pth`. Current baseline (EfficientNet-B0, 15
epochs, RTX 4060): held-out test accuracy 83.9%, AUC 0.925, quadratic weighted
kappa 0.889.

## Running Claude Code on this repo

From the repo root:
```
claude
```
Claude Code will automatically pick up `CLAUDE.md` for project context. Use Plan Mode (Shift+Tab) before large changes, and `/model opusplan` when you want Opus to plan and Sonnet to execute.

## Project structure

See the "Repo layout" section in `CLAUDE.md`.
