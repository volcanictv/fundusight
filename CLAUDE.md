# Fundusight

AI-assisted retinal disease analysis pipeline (fundus photo → quality check → preprocessing → disease detection → explainability → biomarker extraction → report). Educational/portfolio project, not a clinical/diagnostic tool — don't let generated copy (UI text, report language, docstrings) imply otherwise.

See ROADMAP.md for the full phased plan and DEEP_DIVE.md for longer write-ups of specific investigations/results (e.g. the Phase 6 macula heuristic validation). Update the "Current phase" line below as you progress.

**v1.0.0 ship-prep (2026-07-12):** project renamed from its dev codename
VisionDx to Fundusight (README/ROADMAP/app UI/PDF report), git history
audited (nothing large or dataset/checkpoint-related was ever committed —
clean), `.gitignore` tightened, dead code/over-explaining comments cleaned
up, `requirements.txt` pinned to tested versions, and a GitHub-Releases
checkpoint-fetch mechanism (`src/app/checkpoints.py`,
`scripts/fetch_checkpoints.py`) added so a deployed instance with no local
checkpoints can still run inference. Tagged `v1.0.0`.

**Current phase:** Phase 8/9 done — PDF report generation (`src/report/`) + Streamlit dashboard (`src/app/`). Phase 6 is now done — the optic-disc/cup U-Net has been retrained on the pooled/re-split REFUGE2 data (2026-07-11): held-out test Dice `dice_rim=0.8937 dice_cup=0.8576 mean=0.8756`, up from the old domain-split checkpoint's `mean=0.5599` (see ROADMAP.md's Phase 6, which also now covers a 2026-07-12 validation of the macula/fovea heuristic against real ADAM ground truth — see DEEP_DIVE.md for the full write-up: it's unreliable outside REFUGE2-like framing, root cause identified). Phase 7 (multi-disease + multi-dataset) is fully done, including app integration. Glaucoma classifier (2026-07-11, EfficientNet-B0): held-out test `accuracy=0.7400 auc=0.8304 f1=0.4179 sensitivity=0.7778 specificity=0.7348`, best checkpoint at epoch 6/30 by val AUC. AMD classifier (2026-07-12, EfficientNet-B0): held-out test `accuracy=0.9167 auc=0.8887 f1=0.8000 sensitivity=0.7692 specificity=0.9574`, best checkpoint at epoch 30/30 by val AUC. IDRiD cross-dataset DR validation (2026-07-12, evaluation only): the APTOS-trained DR model scores accuracy=0.5429/auc=0.8398/kappa=0.7640 on IDRiD, vs its 83.9%/0.925/0.889 on APTOS itself — a real, expected generalization gap where AUC/kappa (ranking/ordinal signal) hold up much better than raw accuracy. Both classifiers are now wired into `src/detection/glaucoma_infer.py`/`amd_infer.py`, `report/pipeline.py`, `report/content.py`, and the Streamlit app (2026-07-12) — verified end-to-end in the real running app, not just tests. See ROADMAP.md's Phase 7 section for full breakdowns, confusion matrices, and the app-integration details.

**Dashboard visual design — redesigned (2026-07-12).** The bento-card/Inter-only look is gone. Now: a dense, glassmorphic dashboard (`src/app/theme.py`) — frosted white glass cards (backdrop-filter blur + saturation boost) over a light two-tone gradient background, a copper/teal semantic accent duo (copper = "a finding is present," teal = "normal/calm," replacing the old flat blue/emerald/amber trio), Fraunces (serif) for headings/verdict lines paired with Inter (UI) and JetBrains Mono (data), unchanged. The three disease-detection sections (DR/glaucoma/AMD), which used to each render a full subheader + pill + ring + datagrid + full-size Grad-CAM image, are now one "Disease Screening" row of three compact tiles (`render_stat_tile()` in `src/app/components.py`) — the Grad-CAM images moved to the existing shared Image Comparison pills viewer instead of repeating three times inline. Quality/Preprocessing and Vessel/Optic-disc are similarly paired into dense side-by-side rows instead of full-width stacked sections. `report/content.py`'s recommendation text was also tightened — the non-diagnostic "educational observation only" framing is stated once (via the existing `DISCLAIMER`) instead of once per finding. `report/pdf.py` deliberately untouched (separate print-optimized renderer — glass/blur doesn't print well). Verified end-to-end in the real running app via Playwright, not just tests. Still not done: mobile/narrow-viewport behavior (only ever checked at desktop widths).

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

- Prefer small, single-purpose functions over large ones; this is a pipeline of independent stages, keep them independently testable.
- Every new pipeline stage (quality check, preprocessing step, detector, segmenter) gets at least one test with a real or synthetic sample image before moving to the next phase.
- Don't silently swap in a different dataset/model than what's specified in the current roadmap phase — flag it and ask first.
- Trained model weights are large — never commit them to git. Reference download/regeneration instructions in README instead.

## Working with Claude Code on this repo

- Use Plan Mode before touching more than 2-3 files.
- `/model opusplan` for anything architectural (e.g., designing the inference pipeline interface); default Sonnet for implementation.
- `/clear` between phases — a preprocessing session and a Grad-CAM session don't need to share context.
- When in doubt about a phase's scope, check ROADMAP.md's "Done when" criteria before considering it finished.
