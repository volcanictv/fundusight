# Deep Dive

`ROADMAP.md` tracks what got built and its headline numbers, phase by phase.
This doc is the companion to it: longer write-ups of specific investigations
that turned up something worth explaining in more depth than a roadmap
bullet — a validation result, a surprising failure mode, a root-cause
analysis. Each entry is dated and points at the script(s) that produced it,
so the numbers can be reproduced, not just read.

## Phase 6 — Pooling RIGA fixes the domain-shift failure (2026-07-14)

**Summary: the REFUGE2-only disc/cup model carried a systematic CDR bias of up to
+0.20 on unseen cameras — it was not noisy on those images, it was WRONG in a
consistent direction. Retraining on REFUGE2 + RIGA (nine camera domains instead
of three) cuts out-of-domain CDR error by 52% and collapses the bias to within
±0.06 everywhere, with NO in-domain regression. Promoted.**

### The failure this fixes

`scripts/evaluate_on_riga.py` first exposed it: the model held up on RIGA's
MESSIDOR subset but fell apart on BinRushed and Magrabia. Crucially the failure
was a **bias**, not variance — and that distinction is the whole point.

The entry below establishes that in-domain CDR error (0.0436) sits far under the
human inter-observer floor (0.166), so in-domain accuracy is finished and no loss
function can improve it. **Label noise explains variance. It never explains bias.**
A consistent +0.20 offset on BinRushed images was therefore never a noise-floor
story — it was domain shift, and it was invisible because REFUGE2 was the only
dataset with cup labels, so the model was marking its own homework.

### Result — head-to-head, same images, full pipeline

REFUGE2's 180 test images are byte-identical for both checkpoints and provably
absent from the new model's training set (the two datasets are split
INDEPENDENTLY and then concatenated; pooling first and splitting after would have
leaked the old model's test set into the new model's training set and quietly
rigged the comparison).

**In-domain (REFUGE2, n=180):**

| | dice_rim | dice_cup | \|CDR err\| | bias |
|---|---|---|---|---|
| REFUGE2-only | 0.8556 | 0.8244 | 0.0573 | +0.0066 |
| **+RIGA** | **0.8592** | **0.8258** | **0.0561** | **−0.0004** |

**Out-of-domain (RIGA, n=113 held out):**

| subset | \|CDR err\| old → new | bias old → new |
|---|---|---|
| binrushed1 | 0.1582 → **0.0615** | +0.154 → +0.060 |
| binrushed2 | 0.2059 → **0.0697** | +0.201 → −0.049 |
| binrushed3 | 0.1945 → **0.0328** | +0.195 → +0.025 |
| binrushed4 | 0.1313 → **0.0467** | +0.115 → +0.007 |
| magrabia_f | 0.1107 → **0.0677** | +0.102 → +0.035 |
| magrabia_m | 0.0639 → **0.0466** | +0.009 → −0.018 |
| messidor | 0.0514 → **0.0343** | +0.018 → −0.001 |

Overall RIGA CDR error **0.0875 → 0.0420 (−52%)**; RIGA Dice ~0.85 → ~0.91.
Held-out pooled test: `dice_rim=0.9058 dice_cup=0.8691 mean=0.8874`.

### Two process traps this run walked into, both worth remembering

**1. A 20-epoch run said RIGA HURTS in-domain. It was wrong.** The first pooled
model (20 epochs) regressed REFUGE2 Dice 0.8556 → 0.8096, which reads exactly like
"pooling dilutes in-domain performance". It was simply UNDER-TRAINED — validation
Dice was still climbing when it stopped. At 60 epochs the same recipe *beats* the
baseline in-domain. **A negative result from a model that has not converged is not
a result.** The REFUGE2-only baseline had 80 epochs; the comparison was never fair.

**2. The GPU was silently thrashing, and it cost six training runs.** At the old
default `--batch-size 16` this U-Net peaks at ~11.6 GB — more than the 8 GB on the
RTX 4060 Laptop. It does not OOM cleanly; it spills to shared host memory and runs
**17x slower** (7.33 s/batch vs 0.42 s/batch at batch 8). The symptom is not an
error message — it is a run that mysteriously takes hours and dies before finishing
an epoch, which was misdiagnosed three separate ways (job concurrency, DataLoader
worker memory, image decode cost) and prompted an entire caching layer to fix a
bottleneck that was never there.
> **If training is inexplicably slow, time a synthetic GPU step BEFORE optimising
> the data pipeline.** One measurement would have found this immediately.
(The caching layer and the working-resolution fix were kept anyway — the latter is
a genuine train/inference correctness bug, see optic_disc_dataset.__getitem__.)

## Phase 6 — RIGA: the CDR label-noise floor, measured (2026-07-14)

**Summary: six ophthalmologists, grading the same photographs, disagree with each
other on vertical CDR by a mean of 0.166. The model's mean absolute CDR error is
0.0436 — roughly FOUR TIMES SMALLER than the disagreement between two human
graders. The CDR is not an open problem. It is finished, and this is the evidence
that closes it.**

Module: `src/segmentation/riga_dataset.py`. Script:
`scripts/validate_riga_extraction.py`.

### Getting labels out of RIGA at all

RIGA ships **no masks**. Each base image `imageNprime.tif` has six companions
`imageN-1..6`, which are *copies of the photograph* with one ophthalmologist's
disc and cup contours drawn on top. The label must be reconstructed:

```
diff = |annotation - prime|   ->  two thin closed curves
label(diff)                   ->  exactly 2 connected components (disc, cup ring)
fill_holes(each)              ->  two filled disks; larger = disc, smaller = cup
```

This is clean, not a hedge. The contours are drawn in a solid colour with no
antialiasing, so the diff is **threshold-insensitive** (cuts at 20, 30 and 50
select an identical pixel set), and the two curves emerge as exactly two
components. Anything that does not reconstruct to two plausible *nested*
components is **rejected, never guessed at** — a silently mis-reconstructed label
would poison training in a way no loss curve could reveal.

Reconstruction audit over the full set: **0.4% of overlays rejected**, 5.97 of 6
annotators recovered per image, every one of the seven subsets usable. Two data
quirks worth knowing: `BinRushed1` contains **no prime images** (use
`BinRushed1-Corrected`), and Magrabia's female folder is misspelled
`MagrabiFemale` in the distribution. Masks are cached in REFUGE2's own raw pixel
convention (`{0=cup, 128=rim, 255=bg}`) so they drop straight into
`OpticDiscDataset` with no new loading code.

**The masks were also looked at, not just counted** (`outputs/riga_extraction_check.png`).
Statistics cannot tell you a mask is on the disc.

### The number that matters

| | value |
|---|---|
| pairwise inter-annotator Dice, **disc** | 0.9534 |
| pairwise inter-annotator Dice, **cup** | 0.8105 |
| **CDR spread across the 6 annotators, same image** | **mean 0.1662**, median 0.1566, p90 0.2544, max 0.4730 |
| consensus CDR distribution | mean 0.4716, sd 0.0963 |

Set against the model's measured performance:

> **Model mean absolute CDR error: 0.0436.
> Human–human CDR disagreement: 0.1662.**
>
> The model agrees with the consensus roughly **four times more closely than the
> experts who produced that consensus agree with each other.**

DEEP_DIVE previously argued this from published inter-observer figures (~0.1–0.2).
That argument is now unnecessary: the floor is measured, on this repo's own data,
and it is *higher* than the literature range's midpoint.

### Two corroborating details

**The model fails where the humans fail.** Annotators agree on the disc (Dice
0.953) and argue about the cup (0.811) — cup margin is genuinely ambiguous, which
is well known clinically. The model's Dice splits the same way (rim 0.855 vs cup
0.821). A model whose error concentrates in exactly the structure human experts
cannot agree on is exhibiting irreducible ambiguity, not a modelling defect.

**The consensus CDR distribution matches REFUGE2's** (RIGA mean 0.4716 sd 0.0963;
REFUGE2 GT mean 0.4723). Two independently-annotated datasets agreeing on the
population statistic is a strong check that the reconstruction is not systematically
skewed.

### Consequence: stop optimizing the CDR

Any further work aimed at the CDR number — boundary-aware loss, Tversky /
false-positive weighting, threshold recalibration, a bigger backbone — is
optimizing against a target that is itself uncertain to ±0.17. The binding
constraint is **annotation**, and it is not close. This supersedes every "the CDR
could be tightened" thread in this repo.

## Phase 6 — The disc/cup over-segmentation is gone, and the CDR is at the label-noise floor (2026-07-14)

**Summary: the repo's recorded "predicted disc/cup masks run ~1.5x/~3x ground-truth
area" is STALE. Measured on the current model it is 1.002x / 1.029x, with a CDR
bias of -0.0000. The pooled re-split retrain fixed it. The residual CDR error
(0.0436) is variance, not bias, and it is smaller than the disagreement between
two human graders — so there is nothing left for a threshold recalibration or a
boundary-aware loss to correct.**

This entry exists because a stale conclusion nearly caused a pointless retrain.
It is worth reading as a process lesson as much as a result.

### The measurement

Stage 6.2 in isolation (ground-truth ROI crops, so localization error is excluded),
on the 180 held-out pooled test images — never used for training or model selection:

| | recorded claim | measured 2026-07-14 |
|---|---|---|
| disc area ratio (pred / GT) | ~1.5x | **1.002** (median 1.004) |
| cup area ratio (pred / GT) | ~3.0x | **1.029** (median 1.020) |
| disc vertical-extent ratio | — | 1.006 |
| cup vertical-extent ratio | — | 1.016 |
| CDR bias (pred − GT) | biased high | **−0.0000** |
| CDR mean \|error\| | — | 0.0436 (median 0.0349) |

### Why the old number was true and is now false

The ~1.5x/~3x figure was measured on the model trained against REFUGE2's
**official** split — which turned out to be a three-way camera/domain split, not a
sample of one population. The pooled re-split retrain (mean Dice 0.5599 → 0.8756)
did not merely raise Dice; it removed the size bias as a side effect. Nobody
re-measured, so the note outlived the model it described.

**This is the second stale conclusion found in one session.** The other:
`calibrate_optic_disc_thresholds.py` still reads `build_pairs(...)["val"]` — the
old domain-split folders — and the verdict recorded from it ("post-hoc threshold
tuning isn't reliable here; it's a validation/test distribution mismatch") was
drawn from an experiment whose confound the pooled split later eliminated. Both
notes were correct when written and misleading when read.

> **Process lesson: a recorded conclusion is only valid against the model and data
> that produced it.** When the model is retrained, every empirical claim in the
> comments becomes a hypothesis again. Date them, name the checkpoint, and
> re-measure before acting on them — this repo now has two documented cases of a
> stale note nearly directing real work at a problem that no longer existed.

### There is no fix left to apply here

The residual **0.0436 mean absolute CDR error is variance, not bias** (the bias is
zero to four decimals). Published inter-observer variability on vertical CDR
between trained ophthalmologists is roughly **0.1–0.2** — i.e. the model disagrees
with its label by *less than two humans disagree with each other*.

That is a **label-noise floor**, and no loss function fixes label noise. A
boundary-aware loss, a Tversky/false-positive-weighted loss, or another threshold
sweep would all be aimed at a systematic error that is no longer measurable. If
the CDR number is to improve further, the constraint is the **annotation**, not
the architecture or the objective.

## Phase 6 — A vascular convergence prior fixes disc localization on pathology (2026-07-14)

**Summary: the classical disc localizer's brightness search was asking the wrong
question. Multiplying it by a map of where retinal vessels converge takes
localization accuracy from 85.9% to 94.1% (and from 83.3% to 91.7% on the
pathological AMD subset), while keeping silent failures at zero. Two follow-on
findings mattered as much as the fix: the plausibility thresholds silently went
stale when the localizer improved, and the convergence signal is useless as a
confidence check on its own output.**

Scripts: `scripts/evaluate_disc_localization.py` (accuracy + threshold sweep),
`scripts/evaluate_disc_locator.py` (Stage 6.0 arbitration).

### The problem: "brightest disc-sized patch" ≠ "optic disc"

Stage 6.1 located the disc as the peak of a box-filtered brightness map — the
brightest compact, disc-sized patch inside the field of view. That is a strictly
weaker question than "where is the optic disc", and on real pathology the two
answers diverge: a dense exudate cluster, a specular camera reflection, or a
sprawling hemorrhage can all win the brightness search outright. On ADAM's 270
annotated discs it lost 38 times (14.1%), and disproportionately on the AMD
subset (83.3% correct vs 87.1% on Non-AMD) — i.e. it failed exactly where
pathology lives, which is exactly where it matters.

### The fix: vessels converge on the disc, and on nothing else

The optic disc is not defined by being bright. It is anatomically the hub where
every primary retinal vessel enters and exits the eye — and that property is one
the confusers do **not** share. Exudates and reflections are avascular; a
hemorrhage has no vessels *converging* on it. So vascular convergence is a nearly
orthogonal source of evidence to brightness, and it is precisely orthogonal in
the direction that matters.

`optic_disc.compute_vascular_convergence()` builds a directional
Hough-style accumulator:

1. Downscale to 1/4, CLAHE, small multi-scale Frangi → coarse vesselness
   dominated by the major arcades.
2. Estimate each strong vessel pixel's local direction from the structure tensor
   of the vesselness (gradient is *across* a ridge, so the vessel direction is
   the perpendicular one).
3. Every such pixel casts a weighted vote **along its own direction**, in both
   directions (a ridge orientation is sign-ambiguous), out to half a frame.
4. Blur, normalize.

Then `locate_disc_classical()` scores
`brightness * ((1 - w) + w * convergence)` and takes the peak.

**Voting, not density.** A vessel *density* map (blur the vessel mask) was the
obvious cheaper option and is wrong: density is high all along the arcades, so it
peaks in a broad band and would happily rank a dense mid-arcade region above the
disc. Convergence is the property unique to the disc — vessels radiate *from* it,
so their direction lines all pass *through* it and intersect there, while
elsewhere only a couple of near-parallel lines overlap.

**Why it is a weighted blend and not a bare product.** At `w=1.0` a pixel with
zero convergence scores zero, so an image where vessel extraction itself breaks
down (blur, over-exposure, media opacity) fails catastrophically rather than
merely badly. The `(1 - w)` floor keeps brightness as a fallback that can still
win when there is no vascular evidence anywhere, so a vessel-extraction failure
degrades to the *old* behaviour instead of to garbage.

### Result (ADAM, 270 ground-truth discs)

| weight | overall | AMD subset |
|---|---|---|
| 0.00 (brightness only — the old behaviour) | 85.9% | 83.3% |
| 0.30 | 93.3% | 90.5% |
| **0.50 (shipped)** | **94.1%** | **91.7%** |
| 0.70 | 93.7% | 91.7% |
| 1.00 (bare product, no fallback floor) | 94.1% | 91.7% |

Accuracy is flat from 0.5 upward, so the choice is insensitive; 0.5 is the most
conservative point on that plateau (largest brightness fallback for the
extraction-failure case, still at the plateau maximum). Wrong crops: **38 → 16**.
The gain lands where it was designed to: the pathological subset improves most.

Cost: ~178 ms, against a pipeline that already spends ~6.5 s on vessel
segmentation. ~5% overhead, so no caching was needed.

### Finding 1: the plausibility thresholds silently went stale

The geometric gates (circularity ≥ 0.19, diameter ≤ 0.12 of width) were
calibrated in the 2026-07-13 work against the *old* localizer's hit/miss
distribution. Improving the localizer **invalidated them**, and did so quietly:
false alarms rose 20.3% → **31.5%** while recall stayed at 100%. Nothing failed;
the guard just got worse.

The cause is that the 22 newly-rescued localizations are the *hard* ones — the
pathological discs whose raw Otsu blobs are raggeder (lower circularity) and
larger than the easy discs the thresholds were fitted on. The old circularity
gate of 0.19 rejected them as implausible. Re-sweeping for 100% recall at minimum
false alarms moved circularity **0.19 → 0.10** (size cap unchanged), restoring
false alarms to 21.7%.

**The generalizable lesson: these thresholds are a property of the LOCALIZER, not
of optic discs.** Any future change to how the candidate is picked invalidates
them and requires a re-sweep. They are not anatomical constants and must not be
read as such.

Net effect on the number that actually matters — images yielding a **usable**
(correct *and* confident) CDR: **185/270 (68.5%) → 199/270 (73.7%)**. More usable
CDRs *and* fewer wrong crops, with silent failures still at **0**.

### Finding 2: you cannot use the selection signal as its own confidence check

The obvious next idea — gate plausibility on "how much convergence is there at
the chosen center?" — was tested and **rejected**. It separates correct from
incorrect localizations with AUC 0.761, far worse than circularity's 0.945, and
adds *nothing* to the joint gate.

The reason is structural and worth remembering beyond this repo: once the
convergence map is used to **pick** the peak, the peak is high-convergence *by
construction* — including on the misses (max 0.957 among them). **Selecting on a
signal destroys that signal's value as an independent check on the selection.**
An independent guard has to measure something the selection rule did not use,
which is exactly why the *shape* gates work: brightness and convergence pick the
location, geometry audits it.

### What this does NOT improve, and the measurement gap behind it

Better localization does **not** show up as a better CDR on the data where a CDR
can actually be scored. Full-pipeline, on REFUGE2's 180 held-out test images
(`scripts/evaluate_optic_disc_full_pipeline.py`):

| | brightness-only (baseline) | + convergence prior + Stage 6.0 |
|---|---|---|
| dice_rim | 0.8414 | 0.8554 |
| dice_cup | 0.8149 | 0.8206 |
| mean abs CDR error | **0.0571** | **0.0577** |
| median abs CDR error | **0.0368** | **0.0385** |

Dice moves slightly; **CDR error does not move at all** (it is a hair worse, well
inside noise). This is not a contradiction, and it is important to state plainly
rather than to quietly quote the ADAM numbers instead:

**REFUGE2 is a clean glaucoma set. The brightness localizer already worked on it.**
There is almost no confluent exudate, no sprawling hemorrhage, no specular
reflection outshining the disc — that is, none of the pathology the prior was
built to survive. So on REFUGE2 there is nothing for the prior to fix, and it
correctly changes nothing.

The population where localization *did* improve is ADAM's — and **ADAM ships disc
masks but no CUP masks**, so a cup-to-disc ratio cannot be scored there at all.
The result is a real measurement gap:

> The downstream CDR benefit of this work is **unmeasured, and not measurable with
> the data currently in the repo.** We can prove the crop lands on the disc far
> more often on pathology (ADAM, ground truth); we cannot prove the resulting CDR
> is more accurate, because no dataset here has both the pathology and the cup
> annotation.

What IS demonstrated, and is worth having on its own terms, is a **correctness and
safety** property rather than a precision one: 22 fewer CDRs measured off the
wrong anatomy, 10 more usable CDRs, and still zero silent failures. Do not let
this get written up as "the CDR got more accurate" — it did not, on any evidence
available.

(A dataset with both severe pathology and cup annotation — RIGA's 6-annotator
disc+cup contours are the obvious candidate, already downloaded — would close
this gap. See the RIGA notes in ROADMAP.md.)

### Known thin margin, deliberately documented rather than tidied away

5 of the 16 remaining wrong crops are caught by the **size cap alone**, and two
(N0159 at 0.126, N0201 at 0.130) clear the 0.12 cap by less than 0.01. Raising
`_MAX_DISC_DIAMETER_FRACTION` even slightly would convert them into silent
failures. The 100% recall figure is real, but it is not comfortable, and it rests
on 16 examples.

## Phase 6 — Stage 6.0: a coarse full-frame disc locator as a fail-safe (2026-07-14)

**Summary: a second, independent localizer — trained on WHOLE frames, not crops —
arbitrates when Stage 6.1 reports low confidence. On ADAM it rescues 10
localizations, breaks 0, and keeps silent failures at 0. Getting there required
killing two designs that looked reasonable and were not: a multi-task head on the
Stage 6.2 U-Net, and a GAP→MLP coordinate regressor.**

Files: `src/segmentation/disc_locator_{model,dataset,train}.py`,
`optic_disc_infer.locate_disc_arbitrated()`. Script:
`scripts/evaluate_disc_locator.py`.

### Why NOT a multi-task head on OpticDiscUNet

The natural design — bolt a bbox regression head onto the Stage 6.2 U-Net's
bottleneck, get localization as a free auxiliary output — **cannot work**, and
the reason is worth stating so it isn't re-proposed.

OpticDiscUNet only ever sees a 512×512 ONH crop that Stage 6.1 already produced.
A head on it could therefore only learn *"where is the disc inside a crop that
already contains the disc"*. But on the images that matter — the ones this whole
effort is about — **Stage 6.1's crop does not contain the disc**; it contains a
hemorrhage. The disc is not in the tensor, so no head reading that tensor can
point at it.

Running such a head on a *full frame* instead is out-of-distribution inference —
precisely the mistake this repo already made once, with the full-image glaucoma
classifier that returned confident, meaningless probabilities when fed ONH crops.
And it would be worse than useless: a confident bbox pointing at nothing
**converts a caught failure back into a silent one**, destroying the only property
of the localization guard that is actually worth having.

So the arbitration model must be trained on the kind of image it will be asked
about: whole frames. It is a separate network. That is not an implementation
detail, it is the entire point.

### Why NOT a GAP → MLP coordinate head (a real, instructive failure)

The first locator pooled its features globally and regressed 4 numbers from the
pooled vector. It **failed**, and diagnostically:

```
epoch 1  train_loss=0.131  val_hit_rate=0.311  val_median_err=0.117
epoch 2  train_loss=0.086  val_hit_rate=0.017  val_median_err=0.221
epoch 3  train_loss=0.084  val_hit_rate=0.011  val_median_err=0.288
```

Loss falling, hit rate collapsing. The model was minimising the regression loss
by predicting a constant — roughly the dataset-mean disc position — and ignoring
the image.

That is not a tuning problem, it is the architecture. **Global average pooling is
translation-invariant by construction**: it averages every spatial cell into one
vector, deliberately discarding *where* a feature fired and keeping only
*whether* it fired. Asking the pooled vector for a coordinate asks it for the one
thing it exists to throw away, so the optimiser sensibly gives up and regresses
to the mean.

The fix keeps position in the spatial domain: a **soft-argmax over a predicted
1-channel heatmap** (the standard differentiable keypoint readout), which cannot
regress to the mean unless the heatmap itself goes flat. Size (width/height)
still uses GAP → MLP, and there it is *correct* — how big the disc is genuinely
does not depend on where it is.

Same encoder, same data, same loss; only the readout changed:

```
epoch 2  val_hit_rate=0.989  val_median_err=0.024
```

**Generalizable rule: use GAP for "what / how much", never for "where".**

### The arbitration policy, and the bug in its first version

The policy is deliberately asymmetric, set by what each component is *measured*
to be good at:

1. **Classical confident → classical wins, unconditionally.** It was correct on
   199/199 of ADAM's confident cases, and it is pixel-accurate where the locator
   is a 256px estimate. A coarse model does not get to overrule a signal with a
   perfect observed precision — so it isn't even consulted, which also keeps the
   common healthy case as fast as before.
2. **Classical not confident → the locator speaks**, and its center is re-checked
   against the same geometric gates. Accepting its coordinate on faith would
   replace a known-unreliable estimate with an unverified one — that is how a
   caught failure becomes a silent failure again.
3. **Neither verifies → keep the unverified center if it is inside the retina.**

Step 3's first version was **wrong, and the evaluation caught it.** It replaced
the center with the FOV centroid whenever confidence was low. But the
plausibility guard over-flags *on purpose* (~20% of correct localizations get
flagged), so the low-confidence pool is dominated by **good** crops. Result:

| | first version | fixed |
|---|---|---|
| localization accuracy | 210/270 (−44) | **257/270 (+3)** |
| usable CDR yield | 209 (+10) | **209 (+10)** |
| silent failures | 0 | **0** |

It destroyed 44 correct centers to rescue 10 — while the headline "usable yield"
number went *up*, which is exactly how this sort of regression hides. It also
leaked beyond the CDR: `crop_to_onh()` feeds the **glaucoma classifier** and never
consults `confident`, so 61/270 images would have been classified on a crop of the
central retina.

The fallback now fires on the failure it was actually built to prevent — a crop of
frame edge / black canvas, the thing that makes downstream Grad-CAMs light up on
borders — not on mere lack of confidence. On ADAM it never fires at all, which is
the right frequency for a guard against degenerate frames.

### Result

Trained on REFUGE2, evaluated on **ADAM** — a genuine cross-dataset test, since
in-domain is exactly where the classical search already works.

- Held-out REFUGE2 test: `hit_rate=0.9944  median_center_error=0.0163
  p90=0.0285` (center error as a fraction of frame width, against a disc ~0.09
  wide).
- ADAM, end-to-end vs. the classical localizer alone: localization accuracy
  **254 → 257**, usable CDR yield **199 → 209**, needlessly-flagged **55 → 48**,
  silent failures **0 → 0**, images broken **0**.

### Model selection was a coin flip, and that was a bug too

Validation hit rate **saturates at 1.000** by epoch 5 (in-domain REFUGE2, scored
against the ground-truth box, which is generous). A plain `hit_rate > best` rule
therefore stops saving after the first epoch to reach it, locking in whichever
weights got there first — observed directly: epoch 5 was saved at median error
0.0260 while epoch 6 reached 0.0219 and was discarded. **Selecting on a saturated
metric is selection by coin flip.** Ranking on `(hit_rate, -median_center_error)`
keeps the metric that matters primary while letting a continuous one break the
ties it cannot. The LR scheduler had the same defect (it read the saturated
metric as a permanent plateau) and now steps on center error.

## Phase 7 — Glaucoma retrained again, for correctness not for gain (2026-07-14)

**Summary: adding the vascular convergence prior moved 100% of the ONH crops, so
the glaucoma checkpoint was silently mismatched to its own inference path. It was
retrained. Performance is statistically indistinguishable — the retrain is a
correctness fix, not an improvement, and is documented as such.**

`crop_to_onh()` calls `optic_disc.locate_disc_classical()`. Changing that
function changed the crops — a direct comparison of the rebuilt crop cache against
the old one found **300/300 sampled crops changed**. The glaucoma checkpoint had
been trained on crops the code no longer produces: the same train/inference
mismatch the ONH work of 2026-07-13 existed to eliminate, one level subtler,
because the crops still *look* like ONH crops. They are simply not the ones the
model learned on.

Retrained on the rebuilt cache (the old cache is preserved at
`REFUGE2/onh_crops.brightness_only_baseline/` for comparison):

| | old (brightness-only crops) | new (convergence-prior crops) |
|---|---|---|
| accuracy | 0.8533 | 0.8667 |
| AUC | 0.8110 | 0.8274 (95% CI [0.713, 0.925]) |
| sensitivity @ 0.5 | 0.6111 | 0.4444 |
| specificity @ 0.5 | 0.8864 | 0.9242 |
| **sensitivity @ matched specificity 0.89** | **0.6111** | **0.5556** |

**Read this as a wash, not a win.** AUC rose, but the old point estimate sits well
inside the new CI. At matched specificity the new model is nominally *worse* —
but that gap is **one patient**: the test split has 18 positives, so 0.611 = 11/18
and 0.556 = 10/18. Nothing here is distinguishable from noise on 18 positives.

The justification for the retrain is that the alternative was shipping a model
whose input distribution had silently changed underneath it. Note this cuts both
ways as a warning: **any future change to `locate_disc_classical()` requires
rebuilding the ONH crop cache and retraining glaucoma**, or the mismatch returns.

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

> **Re-measured 2026-07-15 (see CLAUDE.md's "Standardized disc-plausibility
> calibration").** The `38/38` / `0 silent failures` / `~22% false alarms` figures
> below are all ADAM **in-sample** — the misses the thresholds were swept against.
> On a pooled, never-fitted 2219-disc set (ADAM+REFUGE2+RIGA) the shipped gate is
> **1/53 silent (~2%)** and **~49% false alarms** (up to 88.6% on RIGA
> BinRushed3). FOV-relative size normalization was tested and does **not** help
> (portability spread 3.13x FOV vs 2.43x frame). Reproduce with
> `scripts/calibrate_disc_plausibility.py`.

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
