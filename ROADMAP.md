# Fundusight — Project Roadmap

AI-assisted retinal disease analysis pipeline. Educational/portfolio project — not a diagnostic device, not for clinical use.

## Ground rules

- Build one vertical slice end-to-end before adding breadth. A working DR-only pipeline beats five half-built disease models.
- Train on a local NVIDIA GPU via `src/detection/train.py`. Don't try to fine-tune CNNs on a CPU.
- Use Claude Code with Plan Mode (Shift+Tab) for anything touching more than 2-3 files. Sonnet for implementation, Opus for planning/architecture calls (`/model opusplan`).
- Commit to git after every working milestone. `/clear` your Claude Code session between unrelated phases.
- Every phase below ends with something you can run and look at — not just code that compiles.

---

## Phase 0 — Setup (few days)

- Install Claude Code, connect it to this repo, read through its basics (`/help`, Plan Mode, `/model`).
- Set up Python env (3.10+), install `requirements.txt`.
- Create a free Kaggle account, download APTOS 2019 Blindness Detection dataset.

**Done when:** you can load and display a handful of APTOS images locally.

## Phase 1 — Image Quality Assessment (weeks 1-2)

Classical CV, no ML/GPU needed.

- Focus: variance of Laplacian (blurry images score low).
- Exposure/illumination: histogram statistics (over/under-exposed detection).
- Combine into a single quality score (0-100%) with pass/fail recommendation.

**Done when:** feeding an image in returns a quality score + which checks passed/failed.

## Phase 2 — Preprocessing (week 2-3)

- Illumination correction (e.g., background subtraction via large Gaussian blur).
- CLAHE (contrast-limited adaptive histogram equalization) on the green/luminance channel.
- Color normalization across the dataset.

**Done when:** you can show a before/after grid of raw vs. preprocessed images and the difference is visible.

## Phase 3 — DR Detection (weeks 3-5)

- Fine-tune a pretrained EfficientNet-B0 on APTOS labels (5-class DR severity) —
  locally on a GPU if you have one (`src/detection/train.py`), otherwise in Colab.
- Track accuracy/AUC/quadratic weighted kappa on a held-out validation split during
  training, then report the same on the untouched test split — don't skip this,
  it's your first real result.
- Load the trained weights in the local inference pipeline (`src/detection/infer.py`).
- Output: probability + severity label (e.g., "Moderate NPDR, 94.2%").

**Done when:** you can run inference on a new image locally and get a probability + severity out.

## Phase 4 — Explainability (week 5-6)

- Wire up `pytorch-grad-cam`: Grad-CAM first, then LayerCAM/EigenCAM.
- Overlay heatmap on the original fundus image.
- Sanity-check: does the heatmap actually land on lesions, or is it attending to the image border? (A common bug — worth explicitly checking.)

**Done when:** for a positive DR case, the heatmap visibly highlights a lesion region, not noise.

## Phase 5 — Vessel Segmentation (weeks 6-8)

Started as classical CV (can run in parallel with Phase 3/4 training), now
upgraded to a hybrid classical+learned pipeline once the classical
Frangi-only mask proved to under-segment thin peripheral vessels.

- Classical stage (done): green channel → CLAHE → multi-scale Frangi
  vesselness filter → hysteresis threshold → skeletonize. Still the
  fallback path when no trained weights are available.
- Hybrid stage: the raw Frangi vesselness response (not thresholded) is fed
  as an extra input channel, alongside the CLAHE'd green channel, into a
  small dilated-convolution U-Net trained on DRIVE/STARE/CHASE_DB1 (APTOS
  has no pixel-level vessel labels) with a Dice + clDice loss, which learns
  to refine the Frangi response rather than segmenting from raw pixels
  alone.
- Compute vessel density, branching count, tortuosity, average width from
  the skeleton — same four biomarkers, now computed from whichever mask
  (classical or hybrid) is in use.

**Done when:** the hybrid model's vessel mask visibly recovers thin
peripheral branches the classical Frangi-only mask missed, on the same
sample image, with a held-out Dice/clDice score reported from training on
DRIVE/STARE/CHASE_DB1.

## Phase 6 — Optic Disc / Cup / Macula Detection (week 8-9)

A two-stage hybrid pipeline, same shape as Phase 5's classical+learned
split: a cheap classical stage handles localization/cropping, a trained
model handles the hard segmentation decision within that crop.

> **Localization hardening (2026-07-14).** Stage 6.1's brightness search was
> failing on exactly the images that matter (pathology), so two things were
> added. See DEEP_DIVE.md for both write-ups.
>
> 1. **Vascular convergence prior** (`optic_disc.compute_vascular_convergence`)
>    — the disc is where the retinal vessels converge, which exudates,
>    reflections and hemorrhages are not. Multiplying the brightness map by a
>    directional vessel-voting accumulator took localization accuracy
>    **85.9% → 94.1%** on ADAM's 270 ground-truth discs (**83.3% → 91.7%** on
>    the pathological AMD subset), wrong crops 38 → 16, silent failures still 0.
>    Usable (correct *and* confident) CDRs: **68.5% → 73.7%**.
>    **This invalidated the plausibility thresholds** — they are a property of
>    the localizer, not of discs — and re-sweeping moved circularity 0.19 → 0.10.
>    Any future change to how the candidate is picked requires re-sweeping them.
> 2. **Stage 6.0, a coarse full-frame locator** (`disc_locator_model.py`) — a
>    small CNN that regresses the disc bbox from a downscaled WHOLE frame, used
>    to arbitrate when Stage 6.1 reports low confidence, with a safe in-retina
>    fallback ROI when both fail (`optic_disc_infer.locate_disc_arbitrated`).
>    Deliberately a **separate model**, not a multi-task head on the Stage 6.2
>    U-Net: that U-Net only ever sees an ONH crop Stage 6.1 already produced, so
>    a head on it could only learn "where is the disc inside a crop that already
>    contains the disc" — and on the failing images the crop does not contain the
>    disc at all. Running such a head on a full frame would be out-of-distribution
>    inference, the same mistake the full-image glaucoma classifier made.
>    Its position head is a **soft-argmax over a heatmap, not GAP → MLP**: GAP is
>    translation-invariant by construction and cannot report *where* a feature
>    fired. The GAP version was built first and failed exactly that way (val hit
>    rate collapsed 0.31 → 0.011 while train loss fell — it regressed to
>    predicting the mean disc position). Use GAP for "what/how much", never for
>    "where".

- Stage 6.1 (classical): locate the optic nerve head (ONH) as the center
  of the brightest disc-sized *compact* patch within the field-of-view
  mask (a windowed-average-brightness peak, not a global brightness
  threshold — a global threshold breaks on real fundus photos, where
  diffuse lesions or an illumination gradient can out-connect the actual
  disc), then crop a square region-of-interest (ROI) around it sized as a
  multiple of the estimated disc diameter. This ROI crop is what makes
  Stage 6.2 tractable — the disc is a small fraction of a full fundus
  photo, and training a segmenter directly on full images would be
  dominated by background/easy negatives. Still the fallback path
  (including a simple intensity-based disc/cup estimate) when no trained
  weights are available.
- Stage 6.2 (hybrid): a small U-Net trained on the REFUGE2 dataset (1200
  labeled images, pooled and re-split — see "Known issue" below, not used
  via its own official train/val/test folders) performs 3-class semantic
  segmentation (background / disc rim / cup) on the ROI crop. Input
  channels combine RGB with Lab (a, b) and HSV (H, S) — multiple color
  spaces sharpen the pallor-based cup/disc boundary that's subtle in RGB
  alone, a standard technique in the optic cup segmentation literature.
  Trained with combined cross-entropy + multi-class soft Dice loss (not
  clDice — that's specifically for thin tubular structures like vessels,
  not compact blobs like disc/cup).
- Stage 6.3 (geometric): compute the vertical cup-to-disc ratio (CDR) from
  the predicted disc/cup masks, enforcing the structural constraint that
  the cup mask must reside entirely within the disc mask (independent
  per-class postprocessing could otherwise violate this — verified by a
  test that deliberately constructs a violating cup mask). Macula/fovea
  location is estimated with a classical heuristic — REFUGE2 has no fovea
  coordinate labels, so this stays unlearned: the darkest compact region
  within an annular band at ~2.5x disc diameter from the disc center,
  searched along the horizontal meridian.

**Known issue — REFUGE2's official split is a 3-way camera/domain split,
not a random sample:** direct image inspection (no metadata file ships
with the dataset) found each of REFUGE2's train/val/test folders is a
single UNIFORM resolution with zero mixing (train 2056x2124, val
1940x1940, test 1634x1634, 400/400 each) and dramatically different mean
color statistics (mean blue channel: train 21.4, val 14.9, test 56.5) —
each folder is effectively one clinical site/camera, not a mixture. This
explains why the first trained checkpoint's validation performance wasn't
predictive of its test performance, and why two independent, correctly
cross-validated post-hoc probability-threshold recalibration attempts
(see `scripts/calibrate_optic_disc_thresholds.py`) both failed to
transfer from validation's domain to test's domain — the second, more
careful attempt failed *worse* than the first. Ground-truth CDR
(disease-severity proxy) is comparable across splits, so this is a
camera/domain issue, not a severity-composition one.

Fix applied: `optic_disc_dataset.build_pooled_pairs()`/
`split_pooled_pairs()` pool all 1200 REFUGE2 images and re-split with
stratification by original folder, so every new split gets a
proportional mix of all three camera domains; `optic_disc_train.py`
trains on this pooled split by default. **Retrain complete** (2026-07-11,
80 epochs, pooled/re-split data: train=840 val=180 test=180): held-out
test Dice — `dice_rim=0.8937  dice_cup=0.8576  mean=0.8756` — vs. the old
domain-split checkpoint's held-out test Dice of `dice_rim=0.6696
dice_cup=0.4502  mean=0.5599`. The large jump (especially on the cup
class) confirms the camera/domain split, not model capacity, was the
bottleneck. `checkpoints/optic_disc_unet.pth` now holds the pooled-split
weights; the prior domain-split checkpoint is kept at
`checkpoints/optic_disc_unet.provisional_domainsplit.pth` for reference/
comparison, not used by inference.

The Dice numbers above are network-only, on ground-truth ROI crops (no
localization error, no post-processing) -- `scripts/
evaluate_optic_disc_full_pipeline.py` measures the harder, more realistic
number: the FULL pipeline (Stage 6.1 classical ONH localization -> Stage
6.2 -> mask cleanup -> Stage 6.3 CDR) against held-out images, and reports
CDR agreement, not just Dice. That script still pointed at REFUGE2's own
official test folder (`build_pairs()`) -- fixed to use the same pooled/
re-split held-out set (`seed=42`) the retrain actually used, since the
official folder is no longer a valid held-out set for this checkpoint
(most of its images are now inside the pooled training set). Full-pipeline
results (180 held-out images, 2026-07-11): `dice_rim=0.8414
dice_cup=0.8149` (lower than the network-only numbers above, as expected
once localization error and post-processing are included), mean predicted
vertical CDR=0.4766 vs. mean ground-truth CDR=0.4722, **mean absolute CDR
error=0.0571** (median=0.0368) -- close agreement, no major systematic
bias, unlike the old domain-split checkpoint (whose cup Dice of 0.45 would
have made its CDR unreliable, consistent with why the earlier calibration
attempts in this section failed).

**Macula/fovea heuristic validated against real ground truth for the first
time (2026-07-12) — result: unreliable, root cause identified.**
`locate_macula_classical()` had only ever been sanity-checked as "looks
about right relative to the disc," since REFUGE2 (its own training/eval
dataset) ships no fovea coordinate labels. ADAM does
(`Fovea_location.xlsx`, one (Fovea_X, Fovea_Y) per image), so
`scripts/evaluate_macula_localization.py` runs Stage 6.1's classical disc
localizer + the macula heuristic on all 400 ADAM Training400 images and
compares against it — converting predicted working-image-space coordinates
back to each image's own *original* pixel space first (ADAM ships two
different native resolutions, 2056x2124 and 1444x1444, so a fixed scale
factor would silently corrupt part of the comparison; same per-image-actual-
dimensions convention `evaluate_optic_disc_full_pipeline.py` and
`vessels._resize_to_working_width()` already use).

```
Raw Euclidean pixel error (original pixel space):     mean=634.6px  median=485.3px
Disc-diameter-normalized error:                        mean=3.679   median=3.327
Percentiles (normalized): p10=0.273 p25=0.988 p50=3.327 p75=5.942 p90=7.194 p95=8.125 p99=9.733 max=18.220
```

The heuristic's own docstring already admits it: "no reliable eye-laterality
info available," so it tries both sides of the disc and picks whichever is
darker. That guess is the dominant error source, confirmed by a side-of-disc
breakdown: it picks the **correct side in only 229/400 images (57%)** —
barely better than a coin flip — and all 10 worst outliers (9-18 disc
diameters of error) are wrong-side picks. Even restricted to the 229
correct-side images, median normalized error is still 1.318 disc diameters
— a real miss, not just noise, so the "darkest point in the window" logic
itself is imprecise even when pointed the right direction (plausibly pulled
off-target by vessels or, on the 89 true-AMD images specifically, AMD
lesions — which are themselves dark, macula-adjacent, and exactly what
Stage 6.1/6.2 were never trained to distinguish from the fovea). This is the
concrete version of the risk the heuristic's docstring already flagged in
the abstract: it was tuned/eyeballed only on REFUGE2-like framing and
degrades outside it. Not fixed here — this is a validation result, not a
fix; a real fix would need either eye-laterality metadata (not available in
REFUGE2 or ADAM) or a learned macula localizer, which no available dataset
currently has the labels to train.

**Stage 6.1's classical localizer can mistake a hemorrhage for the disc —
found, quantified, and now guarded (2026-07-13).** The brightness search
answers "where is the brightest disc-sized patch", which is strictly weaker
than "where is the optic disc": a large hemorrhage or dense exudate cluster
can win it outright, and nothing downstream noticed — the wrong crop fed
Stage 6.2, which segmented a plausible "disc"/"cup" out of it, and Stage 6.3
reported an ordinary-looking CDR measured off the wrong anatomy. Validated
against ADAM's 400 ground-truth `Disc_Masks` (130 are blank/unannotated and
excluded, leaving 270 real ones) by `scripts/evaluate_disc_localization.py`:
the predicted center lands **outside the true disc on 38/270 images (14%)**,
and pathological eyes fail more (AMD 83.3% correct vs Non-AMD 87.1%).

Fix: `assess_disc_plausibility()` in `src/segmentation/optic_disc.py` now
checks the candidate's *shape* rather than its brightness — circularity and
size, both **calibrated against those 270 masks, not guessed**. Result: **38/38
(100%) of wrong crops flagged, 0 silent failures** (previously all 38 were
silent), at a deliberate cost of 47/232 (20.3%) correct crops needlessly
flagged — a false alarm only annotates the CDR as low-confidence, while a miss
silently reports a ratio measured off a hemorrhage. Three calibration results
were counterintuitive enough to be worth knowing (see DEEP_DIVE.md): a
correctly-located disc scores only ~0.34 circularity (not ~0.9 — the blob is a
raw Otsu threshold with vessels cutting through it, and a textbook 0.65
threshold flagged 95.7% of *correct* crops); wrong crops are systematically
**larger** than real discs, so the size check earns its keep as a max, not a
min; and solidity, though individually discriminative, is entirely redundant
and was deliberately **not** shipped as a gate.

`disc_confident` / `disc_localization_warnings` now flow through
`compute_optic_biomarkers{,_hybrid}()` into `report/content.py` and the app.
When localization is rejected the CDR is labelled unreliable **and withheld
from the elevated-CDR observation** (an "elevated" CDR derived from a
hemorrhage is an artifact, not a finding), and the glaucoma-vs-CDR
disagreement note is suppressed too. Verified end-to-end on a real image (ADAM
`A0001`): the localizer grabs a lesion, the check fires
(`circularity 0.05 < 0.19`, `diameter 0.153 of image width`), and a CDR of
0.536 that would previously have been reported as an elevated-CDR finding is
now flagged instead.

**Done when:** disc mask, cup mask, and macula location are overlaid on a
sample image, a vertical cup-disc ratio is printed alongside them, the
cup-within-disc structural check passes (verified by a test), the
production model is retrained on the pooled/re-split data, and a
held-out test-split Dice score (from that new split) is reported for the
disc/cup segmentation model. **Status: done** — see retrain results above.
The macula/fovea heuristic itself is now known to be unreliable outside
REFUGE2-like framing (see validation above) — worth flagging in the app/
report if macula location is ever surfaced as more than an approximate
overlay. Stage 6.1's localizer is now guarded by geometric plausibility
checks (see above), so a bad crop can no longer silently produce a CDR.

## Phase 7 — Multi-disease + Multi-dataset (weeks 9-11)

- Extend classifier to glaucoma (REFUGE) and AMD.
- Cross-validate DR model against a second dataset (MESSIDOR or IDRiD) to demonstrate generalization — this is the detail that shows you understand why single-dataset results are weak evidence.

**Dataset decisions (locked in, don't re-litigate without asking):** AMD uses
ADAM (iChallenge-AMD, downloaded to `ADAM/Training400/`) — no separate
val/test ships publicly, only a 400-image labeled training set, so a split
will need to be carved out same as REFUGE2's Phase 6 pooling fix. Cross-dataset
DR validation uses IDRiD, not MESSIDOR (`data/IDRi/`, 455 of the official
516-image "B. Disease Grading" set — 88% complete, user has confirmed working
with what's present is fine). **Note:** a full duplicate of the IDRiD data
also exists at `IDRiD/` (repo root) — same 456 files as `data/IDRi/`, byte
identical `idrid_labels.csv`. Only `data/IDRi/` is referenced by any code;
the root-level `IDRiD/` duplicate hasn't been cleaned up (ask the user before
deleting either copy).

**Glaucoma classifier — code built and smoke-tested, real training NOT yet
run.** Architecture: EfficientNet-B0, same pattern as the DR classifier (not
the disc/cup U-Net — that's a separate Phase 6 model). Labels come from
`REFUGE2/glaucoma_labels_merged.csv`, built by `scripts/build_glaucoma_labels.py`
(merges REFUGE2's own `Refuge2_test.csv` — despite the name, it covers all
three REFUGE2 domains — with SMDG-19's REFUGE1-train/REFUGE1-val rows,
resolving internally-conflicting REFUGE2-csv duplicates via SMDG-19 where
covered; see that script's docstring for the exact merge rules). If
`REFUGE2/glaucoma_labels_merged.csv` is missing (REFUGE2/ is gitignored, so a
fresh clone won't have it), regenerate with:
```
.venv\Scripts\python.exe scripts\build_glaucoma_labels.py
```
requires `REFUGE2/` and `SMDG, A Standardized Fundus Glaucoma Dataset/` both
downloaded first (same non-committed-data convention as every other dataset
here — see README).

Code: `src/detection/glaucoma_dataset.py` (`build_pairs()`/`split_pairs()` —
mirrors `src/segmentation/optic_disc_dataset.py`'s pooled/stratified-split
pattern, but stratifies on a compound `{domain}_{label}` key, since glaucoma
prevalence (~85/15) is a second imbalance beyond REFUGE2's three-camera-domain
issue) and `src/detection/glaucoma_train.py` (mirrors
`src/segmentation/optic_disc_train.py`'s structure; reuses
`src/detection/model.py`'s `build_model(num_classes=2)` and
`src/detection/dataset.py`'s `build_transforms()` directly — no duplicated
model/transform code, which is also what keeps this swappable for the
RETFound stretch goal mentioned below). Tests: `tests/detection/test_glaucoma_dataset.py`
(5 tests, passing).

Real-data split (verified 2026-07-11): 998 pooled labeled pairs → train=698,
val=150, test=150, all three REFUGE2 camera domains represented in every
split.

**Real 30-epoch training run completed (2026-07-11, GPU: RTX 4060, ~5.5 min
total).** Model selection is by validation AUC-ROC per epoch; the best
checkpoint was reached at **epoch 6/30** (val AUC 0.7271) — after that, train
loss kept falling (0.40 → 0.10) while val AUC plateaued/declined, a clear
overfitting signal on the small 150-image validation split, so the later
epochs' weights were correctly never selected. Held-out **test set** results
(never touched during training or model selection):

```
accuracy=0.7400  auc=0.8304  f1=0.4179  sensitivity=0.7778  specificity=0.7348
confusion matrix (rows=true, cols=predicted):
[[97 35]   TN=97 FP=35
 [ 4 14]]  FN=4  TP=14
```

Sensitivity (77.8%) is notably higher than accuracy would suggest — the
inverse-frequency class-weighted loss trades precision (more false positives,
35) for fewer missed glaucoma cases (only 4 false negatives out of 18
positives), which is the appropriate tradeoff for a screening task. Test AUC
(0.8304) exceeding the best val AUC (0.7271) is plausible given how small and
noisy both splits are (150 images each, ~12% positive).
`checkpoints/glaucoma_efficientnet_b0.pth` now holds this real, fully-trained
checkpoint (not a smoke test).

**To retrain from a fresh session:**
```
.venv\Scripts\python.exe src\detection\glaucoma_train.py --epochs 30
```
(30 is the script's default — override with `--epochs N` for a different
budget; compare against the epoch-6 val AUC of 0.7271 / test AUC of 0.8304
above).

**AMD classifier — trained (2026-07-12, GPU: RTX 4060, ~2 min total).**
Architecture: EfficientNet-B0, same `build_model()`/`build_transforms()`
pattern as DR and glaucoma. Labels come straight from ADAM's own folder
structure — `Training400/AMD/` (89 images, label 1) vs `Training400/Non-AMD/`
(311 images, label 0) — no separate CSV needed. Since ADAM (unlike REFUGE2)
is a single camera domain, `src/detection/amd_dataset.py`'s `split_pairs()`
stratifies on the AMD/Non-AMD label alone (no domain-compound key needed).
Code: `src/detection/amd_dataset.py` + `src/detection/amd_train.py` (mirrors
`glaucoma_train.py`'s structure exactly). Tests: `tests/detection/test_amd_dataset.py`
(5 tests, passing).

Real-data split: 400 labeled images → train=279, val=61, test=60 (carved out
ourselves since ADAM ships no official val/test — same pooling-fix pattern as
REFUGE2's Phase 6 and Phase 7's glaucoma split). Val AUC climbed steadily
across all 30 epochs (0.6398 → 0.9666) with no overfitting plateau — unlike
glaucoma's early epoch-6 peak, this is a cleaner training curve, likely
because AMD/Non-AMD is a more visually separable task than glaucoma cup/disc
subtleties, and/or ADAM's images are more homogeneous (single domain vs
REFUGE2's three). Best checkpoint landed at **epoch 30/30** (val AUC 0.9666).
Held-out **test set** results (never touched during training or model
selection):

```
accuracy=0.9167  auc=0.8887  f1=0.8000  sensitivity=0.7692  specificity=0.9574
confusion matrix (rows=true, cols=predicted):
[[45  2]   TN=45 FP=2
 [ 3 10]]  FN=3  TP=10
```

`checkpoints/amd_efficientnet_b0.pth` holds this real, fully-trained
checkpoint. To retrain: `.venv\Scripts\python.exe src\detection\amd_train.py --epochs 30`.

**IDRiD DR cross-dataset validation — done (2026-07-12), evaluation only, no
training.** Runs the existing APTOS-trained `checkpoints/dr_efficientnet_b0.pth`
unmodified against IDRiD (`data/IDRi/`, 455 images, both IDRiD's official
train and test sets pooled — see `src/detection/idrid_dataset.py`'s docstring
for how the "test"-suffixed id_codes disambiguate IDRiD's own restarted
numbering). IDRiD uses the same 0-4 ICDR severity scale as APTOS, so class
indices transfer directly with no relabeling. Code:
`src/detection/idrid_dataset.py` (a small standalone loader, not a reuse of
`AptosDataset` — IDRiD ships JPEGs not PNGs, and its CSV carries extra
unnamed columns) + `src/detection/idrid_eval.py` (reuses `train.py`'s
`evaluate()` and `dataset.py`'s `build_transforms()` directly — no
metric-computation duplication). Tests: `tests/detection/test_idrid_dataset.py`
(3 tests, passing). Run with:
```
.venv\Scripts\python.exe src\detection\idrid_eval.py
```

**Results — the whole point of this exercise, a real generalization gap:**

```
                APTOS (in-domain, from README.md)   IDRiD (cross-dataset)
accuracy        83.9%                                54.3%
AUC (macro ovr) 0.925                                0.840
kappa (quad.)   0.889                                0.764
```

Confusion matrix (IDRiD, rows=true, cols=predicted):
```
[[ 73  46  10   0   0]
 [  4  16   2   0   0]
 [  4  20 118   6   8]
 [  0   1  59  13  11]
 [  0   0  23  14  27]]
```

Raw accuracy drops ~30 points out-of-domain, but AUC (0.840) and quadratic
weighted kappa (0.764, still "substantial agreement" on the Landis-Koch
scale) hold up much better — kappa in particular discounts near-miss errors
(e.g. predicting severity 2 for a true 3), and the confusion matrix shows
most of IDRiD's errors are exactly that kind of adjacent-class confusion,
not wild misses. This is the demonstration the roadmap called for: a
single-dataset accuracy number is weak evidence of real-world performance,
and the degradation pattern (ranking/ordinal signal surviving better than
hard classification accuracy) is itself informative about what a fundus-photo
domain shift (different camera hardware, population, lighting) does to a
CNN classifier.

---

### Post-review model-failure fixes (2026-07-13)

Domain-expert review of the trained models surfaced four failure modes. All
four were investigated against real ground truth; two produced fixes, one is
documented as a known limitation, one is blocked on missing data. Full
write-ups in DEEP_DIVE.md.

**1. Glaucoma classifier was attending to edge artifacts and hemorrhages, not
the disc — FIXED by cropping to the ONH.** Reproduced first (Grad-CAM on the
old checkpoint puts its heat on the frame edge/vignette on `T0041` and on a
bright lesion cluster on `T0022`, disc cold in both). Glaucoma is diagnosed
from the optic nerve head, but the classifier saw a whole fundus photo where
the disc is a few percent of pixels — a ~25px blob once squashed to 224x224.
`src/detection/onh_crop.py` now crops to the ONH (3 disc diameters wide, so
peripapillary RNFL defects and disc hemorrhages just outside the rim survive)
by reusing Phase 6's Stage 6.1 localizer. It is the **single shared crop
definition**, imported by both `glaucoma_dataset.py` (training) and
`glaucoma_infer.py` (inference) so the two cannot drift; `report/pipeline.py`
applies it once and feeds it to both the prediction and Grad-CAM, so the
heatmap explains the array the prediction came from. Retrained via
`.venv\Scripts\python.exe src\detection\glaucoma_train.py --epochs 30`
(`--full-image` reproduces the old baseline). The pre-fix checkpoint is kept at
`checkpoints/glaucoma_efficientnet_b0.fullimage_baseline.pth` for comparison,
not used by inference.

**No regression — but only when compared correctly:**

```
                  baseline (full image)   ONH crop
accuracy                0.7400             0.8533
auc                     0.8308             0.8110
sensitivity             0.7778             0.6111
specificity             0.7348             0.8864
```

The 17-point sensitivity "collapse" is a **threshold artifact, not a capability
loss**. The ONH model outputs systematically lower probabilities (mean 0.295 vs
0.393), so a fixed argmax(0.5) parks it at a more conservative operating point.
At **matched specificity** the two are identical at the baseline's own point
(both `sens=0.778 @ spec>=0.735`) and the ONH model is *better* at high
specificity (`0.611` vs `0.556 @ spec>=0.886`). AUC is statistically
indistinguishable (bootstrap CIs `[0.722, 0.926]` vs `[0.688, 0.919]`) — the
test split has only 18 positives, where three cases move sensitivity 17 points.
Lower the threshold if the old sensitivity is wanted.

**On attention, the honest result:** the CAM "enrichment on the disc" metric is
*not* trustworthy here — Grad-CAM says the ONH model puts 1.45x MORE attention
on the disc, LayerCAM says 0.65x LESS, disagreeing on magnitude *and direction*
(EfficientNet-B0's final CAM grid is 7x7, one cell = 32x32 input px, too coarse
to resolve the disc). No numeric claim is made. What holds regardless is
structural: after cropping, edge artifacts and distant hemorrhages are outside
the model's input and **cannot** be attended to. Attention now lands on
rim/peripapillary anatomy rather than the frame edge — a partial fix, not a
solved problem. Run `scripts\compare_glaucoma_attention.py`.

**2. Classical disc localizer accepting hemorrhages as discs — FIXED.** See
Phase 6 above (geometric plausibility checks; 38/38 bad crops now caught, 0
silent failures).

**3. DR classifier's central spatial bias — CONFIRMED REAL AND LEARNED;
lesion-mask confirmation BLOCKED.** `scripts/investigate_dr_spatial_bias.py`.
Preprocessing was ruled out first (cheapest explanation, one-line fix if true):
the eval transform chain has **no crop op** — `Resize((224,224))` squashes the
whole frame in — `enhance.preprocess()` preserves frame size, and it isn't in
the classifier's path anyway (`pipeline.py` passes the **raw** image to the
classifiers; `preprocess()` output is display-only). So the model does see the
periphery. The bias is then measured against a **randomly-initialized control**,
since a CNN + Grad-CAM has a center bias built in (zero-padding, coarse grid,
circular FOV in a square frame) that would make even an untrained model look
centrally biased:

```
bin (center -> edge)   0.00-0.17  0.17-0.33  0.33-0.50  0.50-0.67  0.67-0.83  0.83-1.00
trained                  2.52       2.19       1.54       0.85       0.40       0.24
untrained (null)         0.82       1.05       0.74       0.63       1.24       1.68
difference              +1.70      +1.14      +0.81      +0.22      -0.84      -1.44
```

The untrained net actually attends **more at the edge** — so training didn't
just fail to overcome an architectural center bias, it *reversed an edge bias
into a strong central one*. Attention falls off 10x from center to periphery.
**Blocked:** the planned lesion-location correlation needs IDRiD's lesion
segmentation masks, which are **not in this repo** — only IDRiD's "B. Disease
Grading" subset (455 JPEGs + severity CSV, no pixel labels) was downloaded. The
masks live in IDRiD's separate **"A. Segmentation"** download (81 images).
Reported as blocked rather than substituting another dataset and calling it the
same evidence.

**4. AMD classifier ignores the macula — CONFIRMED (causally), NOT FIXED
(deliberately).** `scripts/evaluate_amd_attention.py`, measured against ADAM's
real fovea ground truth (`Fovea_location.xlsx`), *not* the unreliable
`locate_macula_classical()`. Grad-CAM is *misleading* here: it suggests the
model looks hard at the macula (3.49x enrichment on AMD images, and harder when
more confident), and LayerCAM disagrees in sign. The **causal occlusion test**
settles it — inpaint the macula away (inpainting, not blacking out: a black disc
costs the model ~0.475 confidence *regardless of where it is put*, so a
black-out test measures reaction to a black disc, not dependence on anatomy) and
compare against a matched control region:

```
true-AMD, n=84 (mean AMD probability)
  unmodified                 0.923
  macula inpainted away      0.879   (drop +0.043)
  control region inpainted   0.899   (drop +0.024)
  STILL predicted AMD with the macula removed: 77/84 (91.7%)
  Wilcoxon macula-drop vs control-drop: p=0.979  -> no macula-specific dependence
```

Remove the site that *defines* the disease and the model still calls it AMD
91.7% of the time, no worse than removing a random patch of retina. **Not fixed
on purpose:** the obvious fix (crop to the macula) depends on
`locate_macula_classical()`, already known unreliable (57% correct on
eye-laterality) — stacking a real fix on a broken localizer would just move the
error. What the model *is* using is unidentified; ADAM ships no lesion masks, so
"not the macula" is as far as the available data goes.

**Hemorrhage-masking as a lower-risk mitigation — investigated, NOT
recommended.** Three reasons, in order of severity: (a) **nothing to mask
with** — ADAM has no lesion annotations at all, and the nearest source (IDRiD
"A. Segmentation") is undownloaded *and* a diabetic-retinopathy population whose
hemorrhages differ from AMD's submacular ones; a segmenter trained there and
applied to ADAM crosses the exact domain gap this repo measured at 83.9% ->
54.3%. (b) **Masking leaks the label, and the occlusion experiment proves it
empirically** — a masked region shifts this model's output by ~0.475 regardless
of content, so if masks appear only where hemorrhages are (and hemorrhages
correlate with AMD), the model learns "black blob => AMD": a *worse* shortcut
than the one being removed. Avoiding that needs inpainting **plus** matched
decoy masks on the negatives — substantial, delicate work, not a cheap
mitigation. (c) **It may delete real signal** — submacular hemorrhage is a
legitimate wet-AMD finding, not a spurious correlate. A real fix needs either a
trustworthy fovea localizer or lesion-level labels for ADAM; until then this is
a documented limitation, not a queued fix.

---

**Stretch goal reminder (not a Phase 7 blocker):** later, add RETFound
(ViT-Large, MAE-pretrained on retinal images) as a comparison arm against
EfficientNet-B0 across DR/glaucoma/AMD, via partial fine-tuning (freeze most
of the backbone, unfreeze last few blocks + head — fits 8GB VRAM). Don't
build it yet, but the glaucoma classifier code above was deliberately kept
model-agnostic (`build_model(num_classes=...)` as the only task-specific
call) so RETFound can slot in later without a rewrite.

**Glaucoma + AMD wired into inference and the app (2026-07-12).** Both
classifiers were trained but only reachable via their training scripts
until now. `src/detection/glaucoma_infer.py` / `amd_infer.py` mirror
`infer.py`'s `load_model()`/`predict()` contract exactly. `report/pipeline.py`
runs all three classifiers (DR/glaucoma/AMD) through one shared
`_run_classifier()` helper (each gets its own Grad-CAM overlay, same as DR)
and `STAGE_NAMES` now includes `"glaucoma"`/`"amd"`. `report/content.py`
gained a shared `_binary_classifier_sections()` (glaucoma and AMD are
identical in shape, unlike DR's 5-class layout) plus a **disagreement
check**: the optic-disc section's existing elevated-CDR observation and the
new glaucoma classifier are now two independent glaucoma-relevant signals
on the same report, so the Recommendation text explicitly flags it when
they point different directions rather than silently favoring one. No
changes were needed to `report/pdf.py` or `app/render_preview.py` — both
are already `Section.kind`-driven and disease-agnostic, confirming that
architecture choice paid off. Verified end-to-end in the real running app
(not just tests): both sections render correctly with real predictions,
correct pill colors, and Grad-CAM overlays; the disagreement note correctly
appears/doesn't appear depending on whether the two signals agree; the
downloaded PDF includes both new sections.

**Done when:** you have probability scores for all three diseases from one uploaded image. **Status: done.**

## Phase 8 — Report Generation (week 11-12)

- ReportLab PDF: patient ID, quality score, disease probabilities, vessel/CDR measurements, attention map thumbnails, recommendation text.

**Done when:** uploading an image produces a downloadable PDF report. **Status:
done** — `src/report/pipeline.py` orchestrates quality/preprocessing-preview/
detection+Grad-CAM/vessel/optic-disc into one dict (degrading gracefully, not
crashing, when no DR checkpoint is present — there's no classical fallback for
detection unlike vessels/optic-disc); `src/report/content.py` turns that into a
renderer-agnostic `ReportContent`; `src/report/pdf.py` renders it to a
print-ready A4 PDF (ReportLab, fixed margins, embedded thumbnails, page
footer/disclaimer).

## Phase 9 — Dashboard (weeks 12-13)

- Streamlit or Gradio app tying everything together: upload → quality → detection → heatmap → measurements → report download.
- Plotly for the probability bars / metrics.

**Done when:** a stranger can upload a fundus photo and get the full pipeline
output through the UI, no code required. **Status: done** — see
`src/app/main.py`: upload or a demo-mode toggle (reads locally-downloaded
APTOS sample images only, never bundled — see the Datasets table's licensing
note) → quality → before/after preprocessing → detection + Grad-CAM + an
ordinal-ramp Plotly probability chart → vessel biomarkers → optic disc/CDR →
an in-app "generation preview" (`src/app/render_preview.py`) that mirrors the
PDF exactly, built from the same `ReportContent` → PDF download. The PDF's A4
layout is the primary print path; a light `@media print` rule
(`src/app/theme.py`) also cleans up the live preview screen for a direct
browser print.

---

## Stretch goals (ongoing, after v1 works)

- Model comparison view (EfficientNet vs. DenseNet vs. ConvNeXt vs. ViT — accuracy/AUC/F1/precision/recall side by side).
- Explainability comparison grid (original / Grad-CAM / EigenCAM / LayerCAM).
- Uncertainty estimation (MC Dropout or a small ensemble) instead of a bare probability.
- Longitudinal tracking if you simulate/collect multi-visit data (lesion count, vessel density, CDR, severity over time).

## Datasets

| Dataset | Use for | Notes |
|---|---|---|
| APTOS 2019 | DR grading | Best starting point — single CSV, ~3.6k images, Kaggle-licensed and not redistributable, so it's gitignored locally and the dashboard's demo mode (Phase 9) only ever reads it off disk, never bundles a copy |
| EyePACS | DR grading | Much larger, use once pipeline works on APTOS |
| REFUGE2 | Optic disc/cup segmentation, CDR | 1200 images; official train/val/test folders (400 each) are each a single camera/site domain (see Phase 6's "Known issue"), so pooled and re-split instead; masks use pixel values {0=cup, 128=disc rim, 255=background}; no fovea/macula coordinate labels |
| MESSIDOR | DR | Good cross-dataset validation set |
| IDRiD | Lesion segmentation + grading | More granular labels |
| DDR | Multi-lesion DR | Broader lesion types |
| DRIVE | Vessel segmentation (pixel labels) | 40 images; only the 20-image training split ships vessel ground truth in the standard download — the test split has images + FOV masks but no vessel labels |
| STARE | Vessel segmentation (pixel labels) | 20 images, hand-labeled by two independent experts; ships gzip-compressed (`.ppm.gz`) |
| CHASE_DB1 | Vessel segmentation (pixel labels) | 28 images, two independent manual segmentations |

## Compute notes

- Training: local NVIDIA GPU — install the CUDA build of torch/torchvision
  (see `requirements.txt`) and run `src/detection/train.py` /
  `src/segmentation/vessel_train.py` directly.
- Inference/app: runs fine on CPU once models are trained — no GPU needed for the Streamlit demo.
