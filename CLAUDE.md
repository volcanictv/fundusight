# VisionDx

AI-assisted retinal disease analysis pipeline (fundus photo → quality check → preprocessing → disease detection → explainability → biomarker extraction → report). Educational/portfolio project, not a clinical/diagnostic tool — don't let generated copy (UI text, report language, docstrings) imply otherwise.

See ROADMAP.md for the full phased plan. Update the "Current phase" line below as you progress.

**Current phase:** Phase 3 — DR Detection

## Tech stack

- PyTorch for deep learning; pretrained EfficientNet/ConvNeXt/DenseNet/Swin as backbones, fine-tuned, not trained from scratch.
- OpenCV + scikit-image for classical CV (CLAHE, Frangi filter, skeletonization).
- pytorch-grad-cam for explainability (Grad-CAM, EigenCAM, LayerCAM).
- Streamlit for the app UI, Plotly for charts.
- ReportLab for PDF report generation.
- Training happens in Colab/Kaggle notebooks (GPU); inference and the app run locally on CPU.

## Repo layout

```
src/
  preprocessing/     quality assessment, CLAHE, illumination correction
  detection/          model loading, inference, training scripts (mirrors Colab notebooks)
  explainability/     Grad-CAM / EigenCAM / LayerCAM wrappers
  segmentation/       vessel segmentation, optic disc/cup, macula detection
  report/             PDF report generation
  app/                Streamlit dashboard
notebooks/            Colab training notebooks (source of truth for trained weights)
data/                 not committed — see README for dataset download instructions
tests/                unit tests, mirrors src/ structure
```

## Conventions

- I'm learning to code — when a change involves a new concept or pattern I haven't used yet, add a short comment explaining *why*, not just what.
- Prefer small, single-purpose functions over large ones; this is a pipeline of independent stages, keep them independently testable.
- Every new pipeline stage (quality check, preprocessing step, detector, segmenter) gets at least one test with a real or synthetic sample image before moving to the next phase.
- Don't silently swap in a different dataset/model than what's specified in the current roadmap phase — flag it and ask first.
- Trained model weights are large — never commit them to git. Reference download/regeneration instructions in README instead.

## Working with Claude Code on this repo

- Use Plan Mode before touching more than 2-3 files.
- `/model opusplan` for anything architectural (e.g., designing the inference pipeline interface); default Sonnet for implementation.
- `/clear` between phases — a preprocessing session and a Grad-CAM session don't need to share context.
- When in doubt about a phase's scope, check ROADMAP.md's "Done when" criteria before considering it finished.
