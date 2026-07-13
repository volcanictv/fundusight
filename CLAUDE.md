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

**Post-review model-failure fixes (2026-07-13).** A domain-expert review of the
trained models surfaced four failure modes; all four were investigated against
real ground truth (see ROADMAP.md's "Post-review model-failure fixes" and
DEEP_DIVE.md for full write-ups). Two produced fixes, one is a documented
limitation, one is blocked on missing data:
1. **Glaucoma classifier attended to edge artifacts/hemorrhages, not the disc — fixed.** It now classifies an **ONH crop**, not a full fundus photo. `src/detection/onh_crop.py` is the single shared crop definition imported by BOTH `glaucoma_dataset.py` (training) and `glaucoma_infer.py` (inference) — do not crop in one without the other, and note `glaucoma_infer.predict()` takes a FULL photo and crops internally, while `predict_on_model_input()` takes an already-cropped ROI (that's the pair `report/pipeline.py` uses so Grad-CAM explains the same array the prediction came from). Retrained: no regression (the apparent sensitivity drop is a threshold artifact — at matched specificity both models hit 0.778). Pre-fix checkpoint kept at `checkpoints/glaucoma_efficientnet_b0.fullimage_baseline.pth`.
2. **Stage 6.1's classical disc localizer could mistake a hemorrhage for the disc — fixed.** `optic_disc.assess_disc_plausibility()` adds geometric (circularity + size) checks, calibrated against ADAM's ground-truth disc masks; it flags 38/38 wrong crops (0 silent failures, previously all 38 silent). `disc_confident`/`disc_localization_warnings` now flow through the biomarker dicts into the report/app, and a low-confidence localization **suppresses the elevated-CDR observation** rather than reporting a CDR measured off a lesion.
3. **DR classifier has a real, LEARNED central spatial bias.** Preprocessing was ruled out (no crop anywhere in the path — the classifier gets the raw image; `preprocess()` is display-only). Confirming it against lesion locations is **blocked**: IDRiD's lesion masks ("A. Segmentation") are not downloaded, only "B. Disease Grading".
4. **AMD classifier does not use the macula** — proven causally (remove the macula and 91.7% of AMD cases are still called AMD, p=0.979 vs a control region). **Deliberately not fixed** (the obvious fix depends on the unreliable macula heuristic). Hemorrhage-masking was investigated and is **not recommended** — see ROADMAP.md.

**Beware CAM-based attention metrics in this repo.** Grad-CAM and LayerCAM were
found to disagree by ~10x, and to *invert* which model looks better, on the same
model/layer/images — EfficientNet-B0's final CAM grid is 7x7 (one cell = 32x32
input px), too coarse to resolve the disc or fovea. Grad-CAM also pointed the
*wrong way* on AMD, where a causal occlusion test contradicted it outright. Don't
build a claim on a single CAM method; prefer causal tests (occlusion — and
inpaint, don't black out: a black region shifts output by ~0.475 regardless of
what it covers).

**Current phase:** Phase 8/9 done — PDF report generation (`src/report/`) + Streamlit dashboard (`src/app/`). Phase 6 is now done — the optic-disc/cup U-Net has been retrained on the pooled/re-split REFUGE2 data (2026-07-11): held-out test Dice `dice_rim=0.8937 dice_cup=0.8576 mean=0.8756`, up from the old domain-split checkpoint's `mean=0.5599` (see ROADMAP.md's Phase 6, which also now covers a 2026-07-12 validation of the macula/fovea heuristic against real ADAM ground truth — see DEEP_DIVE.md for the full write-up: it's unreliable outside REFUGE2-like framing, root cause identified). Phase 7 (multi-disease + multi-dataset) is fully done, including app integration. Glaucoma classifier (retrained 2026-07-13 on **ONH crops**, EfficientNet-B0 — see the post-review fixes above): held-out test `accuracy=0.8533 auc=0.8110 f1=0.5000 sensitivity=0.6111 specificity=0.8864`. Read that sensitivity with care: it is *not* a regression against the retired full-image checkpoint's `sensitivity=0.7778`, it's a threshold artifact — at matched specificity both score 0.778, and AUC is statistically indistinguishable. AMD classifier (2026-07-12, EfficientNet-B0): held-out test `accuracy=0.9167 auc=0.8887 f1=0.8000 sensitivity=0.7692 specificity=0.9574`, best checkpoint at epoch 30/30 by val AUC. IDRiD cross-dataset DR validation (2026-07-12, evaluation only): the APTOS-trained DR model scores accuracy=0.5429/auc=0.8398/kappa=0.7640 on IDRiD, vs its 83.9%/0.925/0.889 on APTOS itself — a real, expected generalization gap where AUC/kappa (ranking/ordinal signal) hold up much better than raw accuracy. Both classifiers are now wired into `src/detection/glaucoma_infer.py`/`amd_infer.py`, `report/pipeline.py`, `report/content.py`, and the Streamlit app (2026-07-12) — verified end-to-end in the real running app, not just tests. See ROADMAP.md's Phase 7 section for full breakdowns, confusion matrices, and the app-integration details.

**Dashboard visual design — redesigned (2026-07-12).** The bento-card/Inter-only look is gone. Now: a dense, glassmorphic dashboard (`src/app/theme.py`) — frosted white glass cards (backdrop-filter blur + saturation boost) over a light two-tone gradient background, a copper/teal semantic accent duo (copper = "a finding is present," teal = "normal/calm," replacing the old flat blue/emerald/amber trio), Fraunces (serif) for headings/verdict lines paired with Inter (UI) and JetBrains Mono (data), unchanged. The three disease-detection sections (DR/glaucoma/AMD), which used to each render a full subheader + pill + ring + datagrid + full-size Grad-CAM image, are now one "Disease Screening" row of three compact tiles (`render_stat_tile()` in `src/app/components.py`) — the Grad-CAM images moved to the existing shared Image Comparison pills viewer instead of repeating three times inline. Quality/Preprocessing and Vessel/Optic-disc are similarly paired into dense side-by-side rows instead of full-width stacked sections. `report/content.py`'s recommendation text was also tightened — the non-diagnostic "educational observation only" framing is stated once (via the existing `DISCLAIMER`) instead of once per finding. `report/pdf.py` deliberately untouched (separate print-optimized renderer — glass/blur doesn't print well). Verified end-to-end in the real running app via Playwright, not just tests. Still not done: mobile/narrow-viewport behavior (only ever checked at desktop widths).

## Tech stack

- PyTorch for deep learning; pretrained EfficientNet/ConvNeXt/DenseNet/Swin as backbones, fine-tuned, not trained from scratch.
- OpenCV + scikit-image for classical CV (CLAHE, Frangi filter, skeletonization).
- Vessel segmentation is a hybrid classical+learned pipeline: classical Frangi vesselness feeds a small dilated-convolution U-Net (trained on DRIVE/STARE/CHASE_DB1 with a Dice+clDice loss) that refines the final mask — see `src/segmentation/`. Downstream stages (report generation, the app) should call `vessel_infer.compute_biomarkers_auto()`, not `vessels.compute_biomarkers()` directly — it picks the hybrid model when a checkpoint exists and falls back to the classical pipeline otherwise, so callers don't need their own fallback logic.
- Optic disc/cup segmentation is also a hybrid classical+learned pipeline: a classical stage locates and crops the optic nerve head (ONH) region to correct for class imbalance (the disc is a small fraction of a full fundus photo), feeding a small U-Net (trained on REFUGE2 with combined RGB/Lab/HSV color channels and a CrossEntropy+Dice loss) that performs 3-class (background/disc rim/cup) segmentation — see `src/segmentation/optic_disc*.py`. Downstream stages should call `optic_disc_infer.compute_optic_biomarkers_auto()`, not `optic_disc.compute_optic_biomarkers()` directly, mirroring the vessel pipeline's fallback convention. The classical ONH-localization stage is guarded by geometric plausibility checks (`assess_disc_plausibility()`) — a candidate that isn't disc-shaped or is the wrong size sets `disc_confident=False`, which callers must honour rather than reporting the CDR anyway. Macula/fovea location uses a classical heuristic only — REFUGE2 ships no fovea coordinate labels — and is **known unreliable** (57% correct on eye-laterality); don't build a fix on top of it.
- pytorch-grad-cam for explainability (Grad-CAM, EigenCAM, LayerCAM).
- Streamlit for the app UI, Plotly for charts.
- ReportLab for PDF report generation.
- Report generation and the dashboard share one pipeline orchestrator (`src/report/pipeline.run_pipeline()`) and one renderer-agnostic content model (`src/report/content.py`), so the PDF (`src/report/pdf.py`) and the in-app "preview before export" (`src/app/render_preview.py`) can't drift apart in content, only in presentation. `src/app/main.py` is the Streamlit entrypoint; run it with `.venv\Scripts\python.exe -m streamlit run src/app/main.py`.
- Training happens locally on a local NVIDIA GPU via `src/detection/train.py` / `src/segmentation/vessel_train.py` (CUDA-enabled torch — see `requirements.txt` for the install command). Inference, the app, and deployment all run on CPU — `requirements.txt` itself is pinned to the CPU build of torch/torchvision since Streamlit Community Cloud has no GPU; see "Git workflow" below before merging any `requirements.txt` change to `master`.

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

## Git workflow

- `master` is the stable branch: what Streamlit Community Cloud deploys from and what recruiters/portfolio viewers see live. It should always work.
- `dev` is the active-work branch — commit and experiment there. Merge (or PR) into `master` only once a change is tested and polished.
- Exception: a live-production incident (e.g. the deployed app crashing) gets fixed directly on `master`, since that's the only branch Streamlit Cloud actually redeploys — fixes pushed to `dev` alone won't get tested against the real crash. Sync `dev` back up (fast-forward is usually enough) once `master` is stable again.
- **`requirements.txt` must keep `torch`/`torchvision` pinned to the `+cpu` build** (`--extra-index-url https://download.pytorch.org/whl/cpu`, `torch==2.11.0+cpu`, `torchvision==0.26.0+cpu`). Streamlit Community Cloud has no GPU — a bare `torch==2.11.0` resolves to the CUDA-bundled default PyPI wheel, and CUDA runtime init/device probing with no driver present segfaults the deployed app on the very first analysis run (this actually happened — see the `+cpu` pin's own comment in `requirements.txt` for the full incident and the exact commands). Local GPU training reverses the usual order because of this pin: run `pip install -r requirements.txt` first (installs the CPU build), *then* install the CUDA build over it for training — re-running `pip install -r requirements.txt` afterward reverts back to CPU, so do that last. **Before merging `dev` → `master`, if `requirements.txt` was touched, double-check the `+cpu` pin is still intact** — installing the CUDA build for local training and then committing without reverting would silently push it back to `master` and reintroduce the deploy crash.

## Working with Claude Code on this repo

- Use Plan Mode before touching more than 2-3 files.
- `/model opusplan` for anything architectural (e.g., designing the inference pipeline interface); default Sonnet for implementation.
- `/clear` between phases — a preprocessing session and a Grad-CAM session don't need to share context.
- When in doubt about a phase's scope, check ROADMAP.md's "Done when" criteria before considering it finished.
