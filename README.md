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
2. Get a free Kaggle account and download the [APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection) dataset into `data/aptos2019/` (not committed to git — see `.gitignore`).
3. For model training, use a Colab or Kaggle notebook with a free GPU runtime — see `notebooks/`. Training locally on CPU is not practical.

## Running Claude Code on this repo

From the repo root:
```
claude
```
Claude Code will automatically pick up `CLAUDE.md` for project context. Use Plan Mode (Shift+Tab) before large changes, and `/model opusplan` when you want Opus to plan and Sonnet to execute.

## Project structure

See the "Repo layout" section in `CLAUDE.md`.
