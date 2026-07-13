# Deep Dive

`ROADMAP.md` tracks what got built and its headline numbers, phase by phase.
This doc is the companion to it: longer write-ups of specific investigations
that turned up something worth explaining in more depth than a roadmap
bullet — a validation result, a surprising failure mode, a root-cause
analysis. Each entry is dated and points at the script(s) that produced it,
so the numbers can be reproduced, not just read.

## Phase 7 — Glaucoma classifier retrained on ONH crops (2026-07-13)

**Summary: the reported failure is real and visible; cropping to the optic nerve
head fixes it structurally. No performance regression — the apparent sensitivity
drop is a threshold artifact, and disappears at matched specificity.**

### The failure, reproduced

Domain-expert review reported the glaucoma classifier attending to edge
artifacts and hemorrhages rather than the optic disc. This is exactly the
shortcut the setup invites: glaucoma is diagnosed almost entirely from the optic
nerve head, but the classifier was handed a whole fundus photo in which the disc
is a few percent of the pixels — and once squashed to 224x224, a ~25px blob,
while a bright edge artifact or a large lesion is big, high-contrast, and
plausibly correlated with site/camera.

Grad-CAM on the original checkpoint reproduces it plainly. On `T0041` the heat
sits on the **right frame edge and outer vignette** with the disc stone-cold; on
`T0022` it is a large blob squarely on a **bright lesion cluster**, disc cold.
Panels are written by `scripts/compare_glaucoma_attention.py`.

### The fix

Crop the classifier's input to the ONH before classifying, reusing Phase 6's
Stage 6.1 localizer + ROI crop rather than inventing a second notion of "where
the disc is" (`src/detection/onh_crop.py`). The crop is 3 disc diameters wide —
deliberately wider than the disc, since peripapillary RNFL defects and disc
hemorrhages just outside the rim are real glaucoma signs a tight crop would cut
away.

`onh_crop.py` is the **single shared definition**, imported by both
`glaucoma_dataset.py` (training) and `glaucoma_infer.py` (inference). A crop that
differed between the two would be a silent train/inference mismatch — the model
evaluated on inputs it never trained on. `report/pipeline.py` applies it once and
feeds the result to *both* the prediction and Grad-CAM, so the heatmap explains
the array the prediction actually came from.

Sanity-checked before committing to it: on REFUGE2, Stage 6.1 lands inside the
true disc on **90%** of images (97% among crops its plausibility check accepts),
so the crops are good enough to train on.

### No regression — but only if you compare correctly

```
                  baseline (full image)   ONH crop
accuracy                0.7400             0.8533
auc                     0.8308             0.8110
sensitivity             0.7778             0.6111
specificity             0.7348             0.8864
```

Read at face value that is a 17-point sensitivity collapse — the worst thing
that could happen to a screening model. It isn't real. The ONH model outputs
systematically lower probabilities (mean 0.295 vs 0.393), so a fixed argmax(0.5)
silently parks it at a much more conservative operating point. Comparing at
**matched specificity** removes that confound:

```
at specificity >= 0.735 (baseline's own point):  baseline 0.778   onh 0.778
at specificity >= 0.800:                         baseline 0.778   onh 0.667
at specificity >= 0.886 (onh's own point):       baseline 0.556   onh 0.611
```

At the baseline's own operating point the two are **identical** (0.778), and at
high specificity the ONH model is better. AUC is statistically indistinguishable
(bootstrap CIs [0.722, 0.926] vs [0.688, 0.919]), as it should be — the test
split has only 18 positives, where a three-case swing moves sensitivity 17
points. If the old sensitivity is wanted, lower the decision threshold; nothing
was lost.

### Attention: the honest version

The tempting claim is "attention is now on the disc, here's the number". It does
not survive scrutiny. Measuring CAM enrichment on the ground-truth disc:

```
Grad-CAM:   baseline mean=0.40   onh mean=0.58   -> ONH 1.45x MORE disc attention
LayerCAM:   baseline mean=4.55   onh mean=2.95   -> ONH 0.65x LESS disc attention
```

The two methods disagree on the **magnitude and the direction**. EfficientNet-B0's
final CAM grid is 7x7 (one cell = 32x32 input pixels), so a coarse
channel-averaged method cannot concentrate on a structure spanning ~2-3 cells —
an image whose Grad-CAM center-of-mass sits exactly on the disc can still score
0.13 enrichment. Neither method is "wrong"; they measure different things, and
quoting whichever one flatters the change would be cherry-picking.

**The claim that does hold is structural, and needs no metric:** after cropping,
edge artifacts and distant hemorrhages are not merely deprioritized — they are
outside the model's input entirely and *cannot* be attended to. The panels show
the remaining attention landing on rim/peripapillary anatomy rather than the
frame edge. It is not tightly centered on the cup/disc, so this is a partial fix,
not a solved problem.

(The same instability recurred in the AMD investigation below, where a causal
occlusion test ultimately contradicted Grad-CAM outright. The lesson generalizes:
in this repo, CAM enrichment is a weak instrument, and causal tests beat attention
maps.)

## Phase 3/7 — DR classifier's central spatial bias is real and learned (2026-07-13)

**Summary: preprocessing is clean — it is not the cause. The bias is genuinely
learned. The lesion-mask confirmation is blocked on data this repo doesn't have.**

`scripts/investigate_dr_spatial_bias.py`.

### Step 1 — preprocessing is not throwing the periphery away

The cheapest explanation and a one-line fix if true, so it was ruled out first,
three ways: the eval transform chain is `[ToPILImage, Resize, ToTensor,
Normalize]` with **no crop op** (a `Resize((224,224))` squashes the whole frame
in — nothing is cut); `enhance.preprocess()` preserves the frame size; and it
isn't in the classifier's path anyway — `report/pipeline.py` passes the **raw**
image to the classifiers and uses `preprocess()` output only for a display
panel. The model sees the full frame, periphery included.

### Step 2 — the bias is learned, and the control is what proves it

A CNN + Grad-CAM has a center bias *built in*: zero-padding weakens border
activations, the CAM grid is coarse, and the fundus FOV is a circle in a square
frame. Any of those alone would make a model that learned nothing look
"centrally biased". So the radial attention profile was measured twice — once
for the trained DR classifier, once for a **randomly-initialized** EfficientNet-B0
whose profile is pure architecture/CAM artifact — with attention normalized per
radial bin by how much retina actually falls in that bin.

```
bin (center -> edge)   0.00-0.17  0.17-0.33  0.33-0.50  0.50-0.67  0.67-0.83  0.83-1.00
trained                  2.52       2.19       1.54       0.85       0.40       0.24
untrained (null)         0.82       1.05       0.74       0.63       1.24       1.68
difference              +1.70      +1.14      +0.81      +0.22      -0.84      -1.44
```

The result is stronger than "training failed to overcome an architectural center
bias". The untrained network actually attends **more at the edge** (1.68 in the
outermost bin) — so training didn't merely fail to fight a center bias, it
**reversed an edge bias into a strong central one**. The trained model's
attention falls off 10x from center (2.52) to periphery (0.24). The domain
expert's observation is confirmed, and it is learned, not an artifact.

### Step 3 — lesion-location correlation: BLOCKED, no data

The planned confirmation — correlating peripheral lesion locations against
attention using IDRiD's lesion segmentation masks — **cannot be run.** The IDRiD
copy here is the "B. Disease Grading" subset only: 455 JPEGs + a severity CSV,
no pixel labels. The lesion masks (microaneurysms / haemorrhages / hard exudates
/ soft exudates, 81 images) ship in IDRiD's separate **"A. Segmentation"**
download, which was never fetched. This is reported as blocked rather than
silently substituting a different dataset and calling it the same evidence.
Downloading IDRiD "A. Segmentation" unblocks it.

## Phase 7 — AMD classifier does not use the macula (2026-07-13)

**Summary: confirmed, by a causal test — not the one the Grad-CAM heatmaps
suggested. The model decides "AMD" from features outside the macula, and
removing the macula entirely barely changes its mind.**

### Background

Domain-expert review reported that the AMD classifier "abandons the macula and
attends to large hemorrhages instead on striking cases, while working fine on
subtler ones." Age-related *macular* degeneration is defined at the macula, so
a model ignoring it would be right for the wrong reasons — the kind of shortcut
that survives a good test-set score (this one's is `accuracy=0.9167
auc=0.8887`) and fails on any population where the shortcut doesn't hold.

ADAM ships real fovea ground truth (`Fovea_location.xlsx`) and real disc masks
(`Disc_Masks/`), so for once the claim could be checked against labels rather
than against `locate_macula_classical()` — which is itself unreliable (57%
correct on eye-laterality, see the entry below). Measuring an attention problem
with a broken localizer would confound the two beyond recovery.
`scripts/evaluate_amd_attention.py` runs the whole investigation.

### Attention maps say the OPPOSITE of the truth — a cautionary result

Macula enrichment = (share of CAM mass inside the macula) / (share of pixels
that are macula); a disc-diameter-radius circle around the ground-truth fovea.
1.0 = no better than uniform.

```
                    Grad-CAM              LayerCAM
AMD (true)      mean=3.49 median=3.60   mean=4.49 median=4.57
Non-AMD (true)  mean=0.09 median=0.00   mean=5.14 median=5.36

true-AMD only, striking (p>=0.9) vs subtle (p<0.9):
  Grad-CAM   striking=3.79  subtle=1.49   -> HIGHER when confident
  LayerCAM   striking=4.39  subtle=5.17   -> LOWER when confident
Spearman(confidence, enrichment):  Grad-CAM +0.342   LayerCAM -0.177
```

Read naively, Grad-CAM says the model looks *hard* at the macula on AMD images
(3.49x enrichment) and looks *harder* the more confident it is — i.e. the
expert is wrong. Visual inspection agrees: on the three most confident cases
(p=1.000) the heat sits squarely on the ground-truth fovea.

But the two CAM methods disagree in **sign** on the confidence trend, which is
the tell that neither is load-bearing. (The glaucoma investigation hit the same
wall: on identical inputs the two methods disagreed about disc attention by an
order of magnitude and inverted which model looked better. EfficientNet-B0's
final CAM grid is 7x7 — one cell covers 32x32 input pixels — so a coarse method
cannot resolve a structure this small, and "which method" silently becomes
"which answer".)

### The causal test, which settles it

Attention maps are correlational. Occlusion is not: delete the macula and see
whether the model still says AMD.

Two details make this a real experiment rather than a vibe check:

1. **A matched control region** — an equal-sized circle mirrored across the
   image center, landing on non-macular retina. Without it, any confidence drop
   could just mean "the image was edited at all".
2. **Inpainting, not blacking out.** This mattered enormously. The first attempt
   filled the region with black and produced a large, convincing-looking drop:

   ```
   original 0.923 -> macula blacked out 0.448  (drop 0.475)
                  -> CONTROL blacked out 0.463  (drop 0.460)
                  Wilcoxon macula-vs-control: p=0.20  (indistinguishable)
   ```

   A black disc costs the model ~0.475 **no matter where it is put**. The test
   was measuring the model's reaction to a black disc, not its dependence on
   the anatomy underneath. Refilling the region with surrounding retinal texture
   (`cv2.inpaint`) removes that artifact:

```
INPAINTED, true-AMD images, n=84 (mean AMD probability)
  unmodified                  0.923
  macula inpainted away       0.879   (drop +0.043)
  control region inpainted    0.899   (drop +0.024)

  STILL predicted AMD with the macula removed: 77/84 (91.7%)
  Wilcoxon, macula-drop vs control-drop: p=0.979
```

**Remove the macula — the site that defines the disease — and the model still
calls it AMD 91.7% of the time, at a cost of 0.043 confidence, which is
statistically no worse than removing a random patch of retina (p=0.98).** The
model has no macula-specific dependence at all. The expert's claim is correct
in substance, and Grad-CAM actively pointed the wrong way.

What the model *is* keying on is not identified here. It could be hemorrhages,
as reported; it could equally be global colour/texture cues or an ADAM-specific
acquisition artifact. Distinguishing those needs lesion masks, and **ADAM ships
none** (only AMD/Non-AMD labels, disc masks, and fovea coordinates), so this
entry stops at "not the macula" rather than guessing at what replaced it.

### Not fixed here, and why

The obvious fix — crop to the macula, as was done for glaucoma's ONH — is
blocked: it depends on `locate_macula_classical()`, which is the unreliable
component documented below. Stacking a real fix on a broken localizer would
produce a model whose inputs are wrong in a new way, and the cropped-glaucoma
retrain only worked because Stage 6.1's disc localizer is ~90% accurate on
REFUGE2. There is no equivalently trustworthy macula localizer to build on.

### Would hemorrhage-masking during training be a lower-risk mitigation?

Investigated, not implemented — the answer is **no, it is not currently
feasible, and the naive form is actively worse than the problem it treats.**
Three independent reasons, in order of how fatal they are:

1. **There is nothing to mask with.** Masking hemorrhages during AMD training
   requires hemorrhage annotations on ADAM. ADAM has no lesion masks of any
   kind. The nearest available source is IDRiD's "A. Segmentation" subset
   (81 images with haemorrhage/exudate/microaneurysm masks) — which is *not
   downloaded* (this repo has only IDRiD's "B. Disease Grading" images + CSV),
   and which is a **diabetic retinopathy** population: DR hemorrhages differ
   from AMD's submacular hemorrhages in morphology, distribution, and cause.
   Training a segmenter on IDRiD and applying it to ADAM crosses exactly the
   domain gap this repo has already measured as severe (the DR classifier drops
   83.9% -> 54.3% accuracy from APTOS to IDRiD). A mask driven by an unreliable
   cross-domain segmenter deletes the wrong pixels.

2. **Masking leaks the label — and the occlusion experiment above proves it
   empirically, not theoretically.** A blacked-out region shifts this model's
   output by ~0.475 *regardless of what it covers*. A mask is therefore an
   enormously salient feature in its own right. If masks are applied only where
   hemorrhages exist, and hemorrhages correlate with AMD, then the mask itself
   becomes a near-perfect predictor and the model learns "black blob => AMD" —
   a shortcut strictly worse than the one being removed, because it is
   guaranteed and has no clinical content at all. Avoiding this requires
   inpainting with plausible retinal texture (never zeroing) *and* applying
   matched decoy masks to the negatives so the mask carries no label
   information. That is a substantial and delicate piece of work, not a cheap
   mitigation.

3. **It may delete real signal.** In wet AMD, submacular hemorrhage is a
   legitimate diagnostic finding, not a spurious correlate. Masking it would
   train the model to ignore a genuine feature of the disease. Only lesions that
   are *not* diagnostic of AMD would be worth masking — and telling those apart
   is precisely the annotation work that doesn't exist for ADAM.

**Recommended instead:** the occlusion probe above is the cheap, label-free
diagnostic and it already ran — no masks needed, and it answers the question
that matters ("does the model need the macula?") more directly than any CAM. A
real fix needs either fovea-localization good enough to crop on (currently
absent) or lesion-level labels for ADAM. Until one of those exists, this is a
documented limitation, not a queued fix.

See `ROADMAP.md`'s Phase 7 section for the short version.

## Phase 6 — Classical disc localizer accepts hemorrhages as discs (2026-07-13)

**Summary: confirmed against ADAM's ground-truth disc masks — Stage 6.1 lands
outside the true disc on 38/270 images, and every one of those used to produce a
confident-looking CDR. Geometric plausibility checks now catch 38/38.**

### The failure

Stage 6.1 (`locate_disc_classical()`) finds the disc as the brightest
disc-sized compact patch in the field of view. That answers *"where is the
brightest disc-sized patch"*, which is strictly weaker than *"where is the optic
disc"*: a large hemorrhage or a dense exudate cluster can win the brightness
search outright. Nothing downstream could tell — `crop_disc_roi()` would crop
around the lesion, the Stage 6.2 U-Net would dutifully segment something
disc-shaped out of whatever it was handed, and Stage 6.3 would report a
perfectly ordinary-looking cup-to-disc ratio measured off the wrong anatomy.

ADAM ships 400 ground-truth disc masks (130 are blank — no disc annotated — and
are excluded rather than scored as misses, leaving **270** real ones).
`scripts/evaluate_disc_localization.py` measures how often the predicted center
actually falls inside the true disc:

```
overall   232/270 (85.9%)
AMD        70/84  (83.3%)     <- pathological eyes fail more, as expected
Non-AMD   162/186 (87.1%)
```

### The fix: check the candidate's SHAPE, not its brightness

Shape is the one property the confusers don't share with a disc — a disc is
compact and near-circular; hemorrhages are irregular and exudate clusters are
ragged. Thresholds were **calibrated against the 270 ground-truth masks**, not
guessed:

```
                   correct localizations   incorrect localizations
circularity        mean=0.344 median=0.335  mean=0.068 median=0.049   AUC 0.945
solidity           mean=0.871 median=0.891  mean=0.754 median=0.759   AUC 0.851
diameter_fraction  mean=0.092 median=0.089  mean=0.136 median=0.129   AUC 0.159
```

Three things in that table are worth stating outright, because each contradicts
an intuition that would otherwise have shipped a broken check:

- **A correctly-located disc scores only ~0.34 circularity, not ~0.9.** The blob
  being measured is a raw local Otsu threshold, not a clean disc outline —
  vessels cut through it and the edge is ragged, and a ragged edge inflates the
  perimeter that circularity squares in its denominator. The first attempt used a
  textbook threshold of 0.65 and flagged **95.7% of correct localizations**. The
  number is only meaningful *relatively* (0.34 vs 0.07), so `_MIN_DISC_CIRCULARITY`
  is set to 0.19.
- **Size discriminates in the MAX direction, not the min** (AUC 0.159 — i.e.
  *lower* is correct). Wrong crops are systematically **larger** than real discs;
  a confluent exudate/hemorrhage patch outgrows a disc. The intuition that a
  spurious blob would be *small* is backwards.
- **Solidity is redundant and was dropped as a gate.** It separates well on its
  own (AUC 0.851) but adds nothing marginal: gating on it at any threshold from
  0.60–0.75 changed neither the 38 caught nor the 47 false alarms. It is still
  computed and returned as a diagnostic, but gating on it would have been dead
  weight that merely *looked* like extra rigor.

Final operating point (`circularity < 0.19`, or diameter outside 4–12% of image
width):

```
                      confident=True   confident=False (flagged)
correct localization       185               47      <- 20.3% false alarms
WRONG localization           0               38      <- 100% of bad crops caught

SILENT failures remaining: 0/270   (before this check: all 38 were silent)
```

The asymmetry is deliberate. A false alarm merely annotates the CDR as
low-confidence; a miss silently reports a ratio measured off a hemorrhage. At
100% recall on bad crops, the cost is that ~1 in 5 good crops is needlessly
flagged — worth it.

### What it changes downstream

`disc_confident` / `disc_localization_warnings` now flow through
`compute_optic_biomarkers{,_hybrid}()` into the report and app. When
localization is rejected, the CDR is labelled unreliable **and withheld from the
elevated-CDR observation** — an "elevated" CDR derived from a hemorrhage is an
artifact, not a finding, and stating it would be worse than saying nothing. The
glaucoma-vs-CDR disagreement note is suppressed for the same reason: a
disagreement with a CDR already known to be untrustworthy isn't a disagreement,
it's the bad crop talking.

Confirmed end-to-end on a real image (ADAM `A0001`), where the localizer grabs a
lesion and the check fires:

```
cdr=0.536  found=True  confident=False
warnings=['not disc-shaped (circularity 0.05 < 0.19)',
          'implausible size (diameter 0.153 of image width, expected 0.04-0.12)']
```

That CDR of 0.536 would previously have been reported as an elevated-CDR
finding.

## Phase 6 — Macula/fovea heuristic validated against real ground truth (2026-07-12)

**Summary: unreliable outside REFUGE2-like framing, with an identified root cause.**

### Background

`locate_macula_classical()` (`src/segmentation/optic_disc.py`) estimates the
macula/fovea as the darkest compact region within a search window
`~2.5x` the optic disc's diameter from the disc center, tried on both sides
of the disc along the horizontal meridian. It's a classical heuristic, not a
trained model, because REFUGE2 — the dataset Stage 6.2's disc/cup U-Net was
trained on — ships no fovea coordinate labels at all. Until now, it had only
ever been checked by eye ("looks about right relative to the disc").

ADAM (iChallenge-AMD) is the first dataset in this repo that ships real
fovea ground truth: `Fovea_location.xlsx`, one `(Fovea_X, Fovea_Y)` row per
image across all 400 `Training400` images (89 AMD, 311 Non-AMD).
`scripts/evaluate_macula_localization.py` runs Stage 6.1's classical disc
localizer plus the macula heuristic on all 400 images and compares against
it.

**A scaling detail that mattered:** the heuristic's output coordinates are
in `VESSEL_WORKING_WIDTH`-resized working-image space, not original pixel
space, and ADAM ships images at two different native resolutions
(2056x2124 and 1444x1444). Converting predictions back with a single fixed
scale factor would have silently corrupted the comparison for whichever
resolution didn't match it — so the script reads each image's actual
dimensions and computes a per-image scale factor, the same convention
`evaluate_optic_disc_full_pipeline.py`'s `_ground_truth_working_masks()` and
`vessels._resize_to_working_width()` itself already use.

### Results

```
Raw Euclidean pixel error (original pixel space):     mean=634.6px  median=485.3px
Disc-diameter-normalized error:                        mean=3.679   median=3.327
Percentiles (normalized): p10=0.273 p25=0.988 p50=3.327 p75=5.942 p90=7.194 p95=8.125 p99=9.733 max=18.220
```

Disc-diameter normalization matters here more than raw pixels: it's the
same scale unit the heuristic's own search window is defined in, and it's
comparable across ADAM's two native resolutions, unlike a raw pixel count.

A median error of over 3 disc diameters is not "close, but imprecise" — it
means the heuristic is landing somewhere that isn't the macula at all on a
majority of images.

### Root cause: it's guessing which side of the disc the fovea is on

`locate_macula_classical()`'s own docstring already admits the gap: "no
reliable eye-laterality info available." The fovea sits on exactly *one*
side of the disc (temporal to it — nasal for the left eye's disc position,
or vice versa depending on which eye), never both, but without knowing
whether a given photo is a left or right eye, the heuristic searches both
sides and picks whichever is darker.

A side-of-disc breakdown confirms this guess is the dominant error source:

```
same side as ground truth:      229/400 (57%) -> median normalized error = 1.318
opposite side from ground truth: 171/400 (43%) -> median normalized error (much larger)
```

57% correct is barely better than a coin flip. And **all 10 worst
outliers** (9-18 disc diameters of error — the heuristic landing on
something like a blood vessel or a dark corner of the frame, nowhere near
the macula) **are wrong-side picks.**

Even restricted to the 229 images where the heuristic guessed the correct
side, median normalized error is still **1.318 disc diameters** — a real
miss, not noise. So there are two compounding problems, not one:
1. Coin-flip-odds side selection (dominant driver of the worst outliers).
2. Imprecise "darkest point" localization even when pointed the right
   direction — plausibly pulled off-target by blood vessels, or, on the 89
   true-AMD images specifically, AMD lesions themselves (which are dark and
   macula-adjacent by definition, and which neither Stage 6.1 nor Stage 6.2
   were ever trained to distinguish from the fovea).

### What this doesn't fix, and why

This was a validation pass, not a fix. A real fix would need either
eye-laterality metadata (not available in REFUGE2 or ADAM — neither dataset
records which eye a photo is of) or a trained macula localizer, and no
dataset currently in this repo has fovea labels at the volume needed to
train one (ADAM's 400 images, run once as a validation set here, is thin
for that). The practical takeaway: macula location from this pipeline
should be treated as an approximate overlay, not a reliable coordinate —
worth flagging explicitly if it's ever surfaced as more than that in the
report or app.

See `ROADMAP.md`'s Phase 6 section for the shorter version of this result
alongside the rest of Phase 6's history.
