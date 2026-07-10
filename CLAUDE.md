# VisionDx

AI-assisted retinal disease analysis pipeline (fundus photo → quality check → preprocessing → disease detection → explainability → biomarker extraction → report). Educational/portfolio project, not a clinical/diagnostic tool — don't let generated copy (UI text, report language, docstrings) imply otherwise.

See ROADMAP.md for the full phased plan. Update the "Current phase" line below as you progress.

**Current phase:** Phase 8/9 done — PDF report generation (`src/report/`) + Streamlit dashboard (`src/app/`). Phase 6's optic-disc/cup retrain on the pooled/re-split REFUGE2 data is still parked (see ROADMAP.md's Phase 6 "Known issue"), and Phase 7 (multi-disease) hasn't been started.

**Known issue — dashboard visual design is a first pass, not final:** the current look (bento cards, ring/pill/data-grid components in `src/app/components.py`, Inter/JetBrains Mono typography, `src/app/theme.py`'s CSS) is a functional, verified-working iteration, not a finished design. It still needs a real design pass — spacing/hierarchy consistency across sections, mobile/narrow-viewport behavior (never checked, the app was only verified at desktop widths), and a general polish pass beyond "the individual pieces work" — before treating the visual layer as done the way the underlying pipeline/report logic is.

## Tech stack

- PyTorch for deep learning; pretrained EfficientNet/ConvNeXt/DenseNet/Swin as backbones, fine-tuned, not trained from scratch.
- OpenCV + scikit-image for classical CV (CLAHE, Frangi filter, skeletonization).
- Vessel segmentation is a hybrid classical+learned pipeline: classical Frangi vesselness feeds a small dilated-convolution U-Net (trained on DRIVE/STARE/CHASE_DB1 with a Dice+clDice loss) that refines the final mask — see `src/segmentation/`. Downstream stages (report generation, the app) should call `vessel_infer.compute_biomarkers_auto()`, not `vessels.compute_biomarkers()` directly — it picks the hybrid model when a checkpoint exists and falls back to the classical pipeline otherwise, so callers don't need their own fallback logic.
- Optic disc/cup segmentation is also a hybrid classical+learned pipeline: a classical stage locates and crops the optic nerve head (ONH) region to correct for class imbalance (the disc is a small fraction of a full fundus photo), feeding a small U-Net (trained on REFUGE2 with combined RGB/Lab/HSV color channels and a CrossEntropy+Dice loss) that performs 3-class (background/disc rim/cup) segmentation — see `src/segmentation/optic_disc*.py`. Downstream stages should call `optic_disc_infer.compute_optic_biomarkers_auto()`, not `optic_disc.compute_optic_biomarkers()` directly, mirroring the vessel pipeline's fallback convention. Macula/fovea location uses a classical heuristic only — REFUGE2 ships no fovea coordinate labels.
- pytorch-grad-cam for explainability (Grad-CAM, EigenCAM, LayerCAM).
- Streamlit for the app UI, Plotly for charts.
- ReportLab for PDF report generation.
- Report generation and the dashboard share one pipeline orchestrator (`src/report/pipeline.run_pipeline()`) and one renderer-agnostic content model (`src/report/content.py`), so the PDF (`src/report/pdf.py`) and the in-app "preview before export" (`src/app/render_preview.py`) can't drift apart in content, only in presentation. `src/app/main.py` is the Streamlit entrypoint; run it with `.venv\Scripts\python.exe -m streamlit run src/app/main.py`.
- Training happens locally on a local NVIDIA GPU via `src/detection/train.py` / `src/segmentation/vessel_train.py` (CUDA-enabled torch — see `requirements.txt` for the install command). Inference and the app run locally on CPU.

## Repo layout

```
src/
  preprocessing/     quality assessment, CLAHE, illumination correction
  detection/          model loading, inference, local GPU training script
  explainability/     Grad-CAM / EigenCAM / LayerCAM wrappers
  segmentation/       vessel biomarkers (classical Frangi baseline + trained hybrid U-Net); optic disc/cup localization + CDR (classical ONH crop + REFUGE2-trained U-Net) + classical macula heuristic
  report/             PDF report generation
  app/                Streamlit dashboard
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
