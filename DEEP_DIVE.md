# Deep Dive

`ROADMAP.md` tracks what got built and its headline numbers, phase by phase.
This doc is the companion to it: longer write-ups of specific investigations
that turned up something worth explaining in more depth than a roadmap
bullet — a validation result, a surprising failure mode, a root-cause
analysis. Each entry is dated and points at the script(s) that produced it,
so the numbers can be reproduced, not just read.

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
