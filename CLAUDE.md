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

**Localization hardening (2026-07-14).** Stage 6.1's disc localizer was failing
on exactly the images that matter (hemorrhages, exudate, reflections). Two
changes, both in DEEP_DIVE.md:
1. **Vascular convergence prior** — `optic_disc.compute_vascular_convergence()`
   maps where retinal vessels *converge* (a directional Hough-style accumulator
   over vessel orientations, not a density map — density peaks along the whole
   arcade, convergence only at the hub). `locate_disc_classical()` now scores
   `brightness * ((1 - w) + w * convergence)`. Localization accuracy on ADAM's
   270 ground-truth discs: **85.9% → 94.1%** (**83.3% → 91.7%** on the
   pathological AMD subset); usable (correct AND confident) CDRs 68.5% → 73.7%;
   silent failures still **0**. The `(1 - w)` floor is deliberate — a bare
   product fails catastrophically when vessel extraction itself breaks down.
2. **Stage 6.0 coarse full-frame locator** (`src/segmentation/disc_locator_*.py`)
   arbitrates when Stage 6.1 says `confident=False`, and falls back to the FOV
   centroid only when the candidate is *outside the retina*
   (`optic_disc_infer.locate_disc_arbitrated()`). A confident classical
   localization is **never** overruled by it (it was correct 199/199 on ADAM's
   confident cases). End-to-end on ADAM: localization 254 → 257, usable CDRs
   199 → 209, **0 broken, 0 silent failures**. Trained on REFUGE2, so ADAM is a
   real cross-dataset test (held-out REFUGE2: hit rate 0.994, median center
   error 0.0163 of frame width).
   It is a **separate model, not a head on the Stage 6.2 U-Net** — that U-Net
   only ever sees a crop Stage 6.1 already made, and on the failing images that
   crop doesn't contain the disc at all, so no head reading it could point at
   the disc. Running such a head on a full frame would be OOD inference, exactly
   the full-image-glaucoma mistake.
3. **Glaucoma was retrained** (2026-07-13 → 2026-07-14) because the prior moved
   **100% of the ONH crops**: `accuracy=0.8667 auc=0.8274 sens=0.4444
   spec=0.9242`. Read as a **wash, not a win** — AUC's 95% CI is [0.713, 0.925],
   and at matched specificity sensitivity is 0.556 vs the old 0.611, a
   difference of *one patient* out of 18 positives. The retrain is a
   **correctness** fix, not a performance gain.

**HARD COUPLING — `locate_disc_classical()` has two downstream dependents that
break silently if you change it:**
- The **plausibility thresholds** (`_MIN_DISC_CIRCULARITY` etc.) are calibrated
  against a specific localizer's hit/miss distribution. Re-sweep them.
- The **glaucoma ONH crop cache** (`REFUGE2/onh_crops/`) and the glaucoma
  checkpoint trained on it. Rebuild the cache and retrain, or the classifier is
  fed crops it never trained on — and it will not error, it will just be
  confidently wrong. (Old cache kept at `REFUGE2/onh_crops.brightness_only_baseline/`.)

**Two traps that generalize, learned the hard way here:**
- **The plausibility thresholds are a property of the LOCALIZER, not of optic
  discs.** Improving the localizer silently invalidated them — false alarms rose
  20.3% → 31.5% with nothing failing, because the newly-rescued discs are the
  hard, raggeder ones the old circularity gate rejected. Re-swept 0.19 → 0.10.
  **Any change to how the disc candidate is picked requires re-sweeping them.**
- **Never use a selection signal as its own confidence check.** Gating
  plausibility on convergence-at-the-chosen-center looks obvious and is useless
  (AUC 0.761 vs circularity's 0.945): once convergence *picks* the peak, the peak
  is high-convergence by construction, even on the misses. An independent guard
  must measure something the selection rule did not use — which is why the
  *shape* gates work.
- **Use GAP for "what/how much", never for "where".** The Stage 6.0 locator's
  first design regressed coordinates from a global-average-pooled vector; GAP is
  translation-invariant by construction, so it collapsed to predicting the mean
  disc position (val hit rate 0.31 → 0.011 *while the training loss fell*).
  Position now comes from a soft-argmax over a heatmap; GAP is retained only for
  the size output, where translation-invariance is actually correct.

**Optic disc/cup U-Net retrained on REFUGE2 + RIGA (2026-07-14).** RIGA
(`src/segmentation/riga_dataset.py`, ~749 images, 6-annotator consensus) adds six
camera domains to REFUGE2's three. This fixed a real **domain-shift bias**: the
REFUGE2-only model over-estimated CDR by up to **+0.20** on unseen cameras.
Pooled model: out-of-domain CDR error **0.0875 → 0.0420 (−52%)**, bias now within
±0.06 everywhere, **no in-domain regression** (REFUGE2 dice_rim 0.8556 → 0.8592).
Held-out pooled test `dice_rim=0.9058 dice_cup=0.8691 mean=0.8874`. Old checkpoint
kept at `optic_disc_unet.refuge2_only.pth`.
- RIGA ships **no masks** — labels are 6 ophthalmologists' contours drawn on copies
  of the photo, recovered by differencing against the `prime` image. Note two data
  quirks: `BinRushed1` has no primes (use `BinRushed1-Corrected`), and Magrabia's
  female folder is misspelled `MagrabiFemale`.
- **Split each dataset independently, THEN concatenate** (`optic_disc_train.py`).
  Pooling first and splitting after reshuffles REFUGE2's assignment and leaks the
  old model's test set into the new model's training set — the comparison then
  silently flatters whatever is new.
- **`--batch-size` must stay at 8.** At 16 this U-Net peaks at ~11.6 GB on an 8 GB
  RTX 4060 — it does NOT OOM, it spills to host memory and runs **17x slower**
  (7.33 vs 0.42 s/batch). The symptom is a run that inexplicably takes hours and
  gets killed mid-epoch. If training feels slow, **time a synthetic GPU step before
  touching the data pipeline** — this cost six training runs and an unnecessary
  caching layer.
- Training now crops at **working resolution**, matching inference. Previously it
  cropped from the native image (sharp) while inference cropped from the 1400px
  working image (soft) — a real train/inference mismatch.

**The disc/cup CDR is at its label-noise floor IN-DOMAIN — but bias is a different
story (2026-07-14).** Label noise explains *variance*, never *bias*. In-domain CDR
error (0.044) is far below the measured human inter-observer disagreement (0.166,
measured on RIGA's 6 annotators — see DEEP_DIVE), so no loss function will improve
it. But a *systematic* offset on unseen cameras was never a noise-floor story — it
was domain shift, and pooling RIGA fixed it. Don't let "we're at the noise floor"
become a reason to stop looking at out-of-distribution behaviour.

**Original note (still true, in-domain only):**
The long-standing note that predicted disc/cup masks run ~1.5x/~3x ground-truth
area is **stale**; it described the pre-pooled-split model. Measured on the current
checkpoint: disc area ratio **1.002**, cup **1.029**, CDR bias **−0.0000**, mean
|CDR error| **0.0436**. That residual is *variance, not bias*, and it is smaller
than the ~0.1–0.2 inter-observer variability trained ophthalmologists show on
vertical CDR. A boundary-aware loss, a Tversky/FP-weighted loss, or another
threshold sweep would all target a systematic error that no longer exists. The
binding constraint is annotation quality, not the objective.

**Re-measure before acting on any recorded empirical claim.** Two stale
conclusions were found in one session (the over-segmentation above, and
`calibrate_optic_disc_thresholds.py`'s "threshold tuning doesn't transfer" verdict,
which still reads REFUGE2's *old domain-split* folders and whose confound the
pooled re-split already removed). Both were true when written. A retrain
invalidates every empirical comment about the model — treat them as hypotheses,
not facts.

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
