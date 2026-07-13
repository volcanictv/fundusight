"""Phase 6: Optic Disc / Cup / Macula Detection.

Classical CV module -- stays torch-free on purpose, same reasoning as
vessels.py: `locate_disc_classical()`, `crop_disc_roi()`, `compute_cdr()`,
and `locate_macula_classical()` here are pure numpy/opencv/skimage, so both
the classical-only fallback and the hybrid path in `optic_disc_infer.py`
can call into them without pulling in torch.

Pipeline (see ROADMAP.md Phase 6 for the three-stage breakdown):
  Stage 6.1 (this file): locate the optic nerve head (ONH) as the largest
    bright connected region in the field of view, crop a square region of
    interest (ROI) around it. This crop is what corrects for class
    imbalance -- the disc is a small fraction of a full fundus photo, so
    segmenting it directly on the full image would be dominated by
    background/easy negatives.
  Stage 6.2 (optic_disc_model.py / optic_disc_infer.py): a trained U-Net
    segments the ROI crop into background/disc-rim/cup. This file also
    provides a classical intensity-threshold fallback for when no trained
    checkpoint exists (segment_disc_cup_classical()), mirroring
    vessels.segment_vessels()'s role for the vessel pipeline.
  Stage 6.3 (this file): compute_cdr() turns disc/cup masks into a
    vertical cup-to-disc ratio; locate_macula_classical() finds the
    macula/fovea heuristically, since REFUGE2 (the disc/cup training
    dataset) ships no fovea coordinate labels -- that part stays unlearned.

Reuses vessels.VESSEL_WORKING_WIDTH / vessels._resize_to_working_width /
vessels._fov_mask / vessels.extract_vessel_channel directly rather than
re-deriving resolution-canonicalization or FOV-exclusion logic -- both
pipelines work on the same kind of fundus photo, so there's no reason for
a second, independent notion of "canonical resolution."
"""

import cv2
import numpy as np
from skimage.filters import frangi, threshold_otsu
from skimage.measure import label, regionprops

from src.segmentation import vessels

# Size of the square crop fed to the Stage 6.2 network. COUPLED TO the
# assumption (baked into optic_disc_dataset.py and optic_disc_train.py too)
# that the network always sees a DISC_ROI_WIDTH x DISC_ROI_WIDTH image --
# changing this requires retraining, it doesn't rescale automatically.
DISC_ROI_WIDTH = 512

# The ROI crop's side length is this many multiples of the estimated disc
# diameter -- wide enough to comfortably contain the full disc (plus a
# margin for a slightly-off centroid estimate) without cropping so wide
# that the disc becomes a small fraction of the ROI again, which is
# exactly the class-imbalance problem this crop exists to avoid.
_DISC_ROI_CROP_MULTIPLE = 3.0

# Typical optic disc diameter as a fraction of a well-framed fundus photo's
# width -- used both as the disc-locating brightness-window size (see
# locate_disc_classical) and as the fallback/default diameter estimate.
# Real fundus photos vary in exact framing, but this is a reasonable prior
# to search around rather than trust blindly -- locate_disc_classical
# refines the actual diameter locally once it has found a candidate center.
_EXPECTED_DISC_DIAMETER_FRACTION = 0.12

# A locally-refined diameter estimate is only trusted within this multiple
# of the expected diameter above/below -- guards against the same failure
# mode a global brightness threshold has (a diffuse lesion cluster or an
# illumination gradient can locally look "large and bright" too), just
# scoped down from the whole image to a small window so it's far less
# likely to trigger, and cheap to sanity-clamp when it still does.
_DIAMETER_REFINEMENT_MIN_FACTOR = 0.4
_DIAMETER_REFINEMENT_MAX_FACTOR = 2.5

# Geometric plausibility thresholds for the located disc candidate (see
# assess_disc_plausibility). The brightness-window peak in
# locate_disc_classical() answers "where is the brightest disc-sized patch",
# which is NOT the same question as "is that patch actually a disc" -- a
# large hemorrhage or a dense exudate cluster can win the brightness search
# outright. These checks test the candidate blob's SHAPE instead of its
# brightness, which is the one property the confusers don't share with a
# real optic disc: the disc is a compact, near-circular, solid blob of a
# fairly predictable size, while exudate clusters are scattered/ragged and
# hemorrhages are irregular.
#
# Calibrated (not guessed) against ADAM's 270 annotated ground-truth
# Disc_Masks -- see scripts/evaluate_disc_localization.py and DEEP_DIVE.md.
# At these values the checks flag 16/16 (100%) of the localizations that
# actually land outside the true disc, at the cost of flagging 55/254
# (21.7%) of correct ones. That asymmetry is deliberate: a false alarm just
# annotates the CDR as low-confidence, while a miss silently reports a CDR
# measured off the wrong anatomy, which is the failure this exists to stop.
#
# RECALIBRATED when the vascular convergence prior landed. These thresholds
# are a property of the LOCALIZER, not of optic discs in the abstract, so
# changing how locate_disc_classical() picks its candidate invalidates them
# and they must be re-swept -- do not treat them as fixed anatomical
# constants. Concretely, the prior cut wrong crops 38 -> 16, and the 22
# newly-correct localizations are the HARD, pathological ones, whose Otsu
# blobs are raggeder (lower circularity) and larger than the easy discs the
# old thresholds were fitted on. Left at the old circularity gate of 0.19,
# those newly-correct crops got flagged as implausible and the false-alarm
# rate rose to 31.5% -- i.e. the localizer improved while the guard silently
# got worse. Re-sweeping (100% recall on the misses, minimising false
# alarms) moved circularity 0.19 -> 0.10 and left the size cap at 0.12.
#
# Net effect on what actually matters, the share of images that yield a
# usable (correct AND confident) CDR: 185/270 (68.5%) before the prior,
# 199/270 (73.7%) after -- more usable CDRs AND fewer wrong crops.
#
# The size cap earns its keep in the MAX direction, not the min: wrong crops
# are systematically LARGER than real discs (a hemorrhage or confluent
# exudate patch outgrows a disc), the opposite of the intuition that a
# spurious blob would be small. The min gate catches nothing on this data
# (the smallest wrong crop measures 0.093) and is retained only as a guard
# against degenerate blobs.
#
# KNOWN THIN MARGIN, do not "tidy" this away: 5 of the 16 wrong crops are
# caught by the size cap ALONE, and two of them (N0159 at 0.126, N0201 at
# 0.130) clear the 0.12 cap by less than 0.01. Raising _MAX_DISC_DIAMETER_
# FRACTION even slightly would convert them into silent failures. The 100%
# recall figure is real but it is not comfortable, and it rests on 16
# examples.
#
# Two things measured and deliberately NOT gated on, because gating on them
# would be dead weight that only looks like extra rigor:
#   - Solidity: entirely redundant with the two gates below (a solidity gate
#     anywhere in 0.60-0.75 changed neither the caught set nor the false
#     alarms).
#   - Convergence at the chosen center: a weak discriminator (AUC 0.761 vs
#     circularity's 0.945) that adds nothing to the joint gate, for a
#     structural reason worth remembering -- once the convergence map is used
#     to PICK the peak, the peak is high-convergence by construction, even on
#     the misses (max 0.957). Selecting on a signal destroys that signal's
#     value as an independent check on the selection.
_MIN_DISC_CIRCULARITY = 0.10
_MIN_DISC_DIAMETER_FRACTION = 0.04
_MAX_DISC_DIAMETER_FRACTION = 0.12

# --- Vascular convergence prior (see compute_vascular_convergence) ----------
#
# The optic disc is not merely "bright" -- anatomically it is the hub where
# every primary retinal vessel enters and exits the eye. That is a property
# the two things which beat it in a pure brightness search do NOT share: a
# dense exudate cluster and a specular camera reflection are avascular, and
# a sprawling hemorrhage has no vessels *converging* on it either. So a map
# of "how strongly do vessel trajectories point at this pixel" is a nearly
# orthogonal source of evidence to brightness, and multiplying the two
# suppresses exactly the confusers brightness alone cannot reject.
#
# The prior is computed at a REDUCED resolution on purpose. Only the major
# arcades carry convergence information -- thin peripheral vessels contribute
# noise, not signal, to a "where do the trunks point" accumulator -- so
# running a small Frangi at 1/4 scale is both more appropriate and ~10x
# cheaper than reusing vessels.compute_frangi_response()'s 7-scale filter at
# the full 1400px working width, which would roughly double pipeline cost for
# a map that gets blurred anyway.
_VASCULAR_SCALE = 0.25

# Frangi scales for the reduced-resolution pass. These are vessels.py's
# _FRANGI_SIGMAS multiplied by _VASCULAR_SCALE and truncated to the coarse
# end: at 1/4 resolution a major arcade is only a few pixels across, and the
# thin-vessel sigmas would resolve nothing but noise.
_VASCULAR_FRANGI_SIGMAS = (1, 2, 3)

# Only vessel pixels above this percentile of in-FOV vesselness get to vote.
# A vote is a claim about a vessel's *direction*, and direction is only
# meaningfully estimable where there is a real ridge -- letting weak/ambiguous
# response vote would spray near-random directions into the accumulator.
_VASCULAR_VOTE_PERCENTILE = 90.0

# How far (as a fraction of image width) a vessel pixel casts its directional
# vote. The disc sits within roughly half a frame of any arcade pixel; voting
# further just lets far-side vessels smear votes across the whole image.
_VASCULAR_VOTE_DISTANCE_FRACTION = 0.5

# Gaussian smoothing of the raw accumulator, as a fraction of image width.
# The accumulator is a sum of thin rays, so it is spiky; blurring at roughly
# half a disc diameter turns "lines crossed near here" into a smooth basin
# whose peak is stable, and matches the spatial precision the downstream
# 3-disc-diameter crop actually needs.
_VASCULAR_BLUR_FRACTION = 0.05

# How much the prior is allowed to modulate the brightness score. The
# combination is `brightness * ((1 - w) + w * convergence)`, NOT a bare
# product: at w=1.0 a pixel with zero convergence scores zero, which makes the
# localizer fail catastrophically (rather than merely badly) on any image
# where vessel extraction itself breaks down -- a blurred, over-exposed, or
# heavily media-opacified photo. The floor of (1 - w) keeps brightness as a
# fallback signal that can still win when there is no vascular evidence
# anywhere, so a vessel-extraction failure degrades to the OLD behaviour
# instead of to garbage.
#
# Calibrated against ADAM's 270 annotated ground-truth discs, not guessed
# (see scripts/evaluate_disc_localization.py --sweep-vascular-prior).
# Localization accuracy vs. this weight:
#
#     w=0.00 (brightness only, the old behaviour)  85.9%   AMD 83.3%
#     w=0.30                                       93.3%   AMD 90.5%
#     w=0.50                                       94.1%   AMD 91.7%
#     w=0.70                                       93.7%   AMD 91.7%
#     w=0.85                                       94.1%   AMD 91.7%
#     w=1.00 (pure product, no fallback floor)     94.1%   AMD 91.7%
#
# Accuracy is FLAT from 0.5 upward, so the choice is not sensitive -- 0.5 is
# picked as the most conservative point on that plateau, since it retains the
# largest brightness fallback (a 0.5 floor) for the vessel-extraction-failure
# case above while still scoring at the plateau maximum. Note the gain is
# largest exactly where it was designed to be: the AMD subset (hemorrhages,
# exudate) improves 83.3% -> 91.7%, halving the wrong crops on the
# pathological population the brightness search was failing on.
_VASCULAR_PRIOR_WEIGHT = 0.5

# How far from the disc center (in multiples of disc diameter) to search
# for the macula/fovea -- clinically the macula sits roughly 2-2.5 disc
# diameters temporal to the disc. Searched on BOTH sides of the disc along
# the horizontal meridian since APTOS doesn't reliably indicate which eye
# (left/right) a photo is of, so "temporal" could be either direction.
_MACULA_SEARCH_RADIUS_FACTOR = 2.5

# Radius (in multiples of disc diameter) of the disc region excluded from
# the macula search, so a dark pixel right at the disc's own edge doesn't
# get mistaken for the macula.
_MACULA_DISC_EXCLUSION_FACTOR = 0.75


def _largest_component_mask(binary: np.ndarray) -> np.ndarray:
    """Boolean mask of just the largest connected component of `binary`,
    empty if there are none. Used to turn a noisy brightness/darkness
    threshold into a single coherent blob (the disc, or the cup) rather
    than a scatter of small candidate regions.
    """
    labeled = label(binary)
    if labeled.max() == 0:
        return np.zeros_like(binary, dtype=bool)
    counts = np.bincount(labeled.ravel())
    counts[0] = 0  # background label -- never the "largest component"
    return labeled == counts.argmax()


def _unlocated_disc(center_xy: tuple, expected_diameter: float, reason: str) -> dict:
    """The degenerate-image return for locate_disc_classical(): a fallback
    center/diameter so callers never crash, with found=False. Never confident
    -- if the disc couldn't be located at all, its crop can't be trusted
    either, so this shares the same "confident" contract as a candidate that
    was located but failed the geometric checks.
    """
    return {
        "center_xy": center_xy,
        "diameter_px": float(expected_diameter),
        "found": False,
        "confident": False,
        "circularity": float("nan"),
        "solidity": float("nan"),
        "diameter_fraction": float("nan"),
        "implausible_reasons": [reason],
        "vascular_prior_used": False,
    }


def _vessel_direction_field(vesselness: np.ndarray, sigma: float) -> tuple:
    """Per-pixel unit vector pointing ALONG the local vessel ridge, estimated
    from the structure tensor of the vesselness response.

    The structure tensor J = G_sigma * (grad v)(grad v)^T has its dominant
    eigenvector along the direction of greatest intensity change. On a ridge
    that direction is ACROSS the vessel, so the vessel's own direction is the
    perpendicular one -- which is what we return.

    Building the tensor from explicit Sobel gradients rather than
    skimage.feature.structure_tensor is deliberate: skimage's `order='rc'`
    returns row/column-ordered elements, and silently mixing that up with an
    (x, y) convention would rotate every direction by 90 degrees -- producing
    an accumulator that votes ACROSS vessels instead of along them, which
    would look like a plausible map while being exactly wrong. Doing the two
    Sobels by hand keeps the axis convention explicit and auditable.

    Returns `(dx, dy)`, both float32 arrays the shape of `vesselness`. The
    sign of the vector is arbitrary (a ridge has no head or tail), which is
    why the caller votes in BOTH the +d and -d directions.
    """
    gx = cv2.Sobel(vesselness, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(vesselness, cv2.CV_32F, 0, 1, ksize=5)

    jxx = cv2.GaussianBlur(gx * gx, (0, 0), sigmaX=sigma)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), sigmaX=sigma)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), sigmaX=sigma)

    # Orientation of the dominant gradient (across the vessel).
    theta = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)
    # Rotate 90 degrees to get the along-vessel direction.
    return (-np.sin(theta)).astype(np.float32), np.cos(theta).astype(np.float32)


def compute_vascular_convergence(image: np.ndarray) -> np.ndarray:
    """Map of "how strongly do retinal vessel trajectories converge here",
    normalized to [0, 1] and returned at the caller's working-image
    resolution. This is the anatomical prior that tells the optic disc apart
    from a bright thing that merely LOOKS like one.

    Method (a directional Hough-style accumulator, not a density map):
      1. Downscale, CLAHE, and run a small multi-scale Frangi filter to get a
         coarse vesselness response dominated by the major arcades.
      2. Estimate each strong vessel pixel's local direction from the
         structure tensor (see _vessel_direction_field).
      3. Have every such pixel cast a weighted vote along its own direction
         -- in both directions, since a ridge orientation is sign-ambiguous
         -- out to _VASCULAR_VOTE_DISTANCE_FRACTION of the frame.
      4. Blur the accumulator and normalize.

    Why VOTING rather than simply blurring the vessel mask into a density
    map: vessel *density* is high all along the arcades, so a density map
    peaks in a broad band rather than at the disc, and it would happily rank
    a dense mid-arcade region above the disc itself. Convergence is the
    property that is unique to the disc -- vessels radiate FROM it, so their
    direction lines all pass THROUGH it and intersect there, while elsewhere
    only a couple of near-parallel lines overlap. Isolated exudates and
    specular reflections cast no votes at all, since they have no vessels.

    Returns an all-zero map when vessel extraction finds nothing to work with
    (no FOV, no vesselness response). Callers must treat an all-zero map as
    "no vascular evidence available" and fall back to brightness alone rather
    than multiplying their score to zero everywhere -- see
    locate_disc_classical().
    """
    working = vessels._resize_to_working_width(image)
    full_h, full_w = working.shape[:2]

    small = cv2.resize(
        working, (max(int(full_w * _VASCULAR_SCALE), 1), max(int(full_h * _VASCULAR_SCALE), 1)), interpolation=cv2.INTER_AREA
    )
    h, w = small.shape[:2]

    green = vessels.extract_vessel_channel(small)
    fov = vessels._fov_mask(green)
    if not fov.any():
        return np.zeros((full_h, full_w), dtype=np.float32)

    enhanced = vessels.enhance_vessel_contrast(green).astype(np.float64) / 255.0
    # black_ridges=True: vessels are DARKER than surrounding tissue in the
    # green channel (hemoglobin absorbs green) -- same convention as
    # vessels.compute_frangi_response().
    vesselness = frangi(enhanced, sigmas=_VASCULAR_FRANGI_SIGMAS, black_ridges=True).astype(np.float32)
    vesselness *= fov

    in_fov_response = vesselness[fov]
    if in_fov_response.size == 0 or in_fov_response.max() <= 0:
        return np.zeros((full_h, full_w), dtype=np.float32)

    vote_threshold = np.percentile(in_fov_response, _VASCULAR_VOTE_PERCENTILE)
    voters = (vesselness >= vote_threshold) & fov & (vesselness > 0)
    if not voters.any():
        return np.zeros((full_h, full_w), dtype=np.float32)

    dx, dy = _vessel_direction_field(vesselness, sigma=max(w * 0.01, 1.0))

    ys, xs = np.nonzero(voters)
    weights = vesselness[ys, xs].astype(np.float32)
    vx, vy = dx[ys, xs], dy[ys, xs]

    accumulator = np.zeros((h, w), dtype=np.float32)
    max_steps = max(int(w * _VASCULAR_VOTE_DISTANCE_FRACTION), 1)

    # Walk every voter one step at a time along +/- its own direction,
    # accumulating with np.add.at (which, unlike fancy-index assignment,
    # correctly ACCUMULATES rather than overwrites when several voters land on
    # the same pixel -- the overwrite bug would silently turn this from a vote
    # count into a "last writer wins" map and destroy the whole method).
    for step in range(1, max_steps + 1):
        for direction in (1.0, -1.0):
            px = np.round(xs + direction * step * vx).astype(np.int32)
            py = np.round(ys + direction * step * vy).astype(np.int32)
            inside = (px >= 0) & (px < w) & (py >= 0) & (py < h)
            np.add.at(accumulator, (py[inside], px[inside]), weights[inside])

    accumulator *= fov
    blurred = cv2.GaussianBlur(accumulator, (0, 0), sigmaX=max(w * _VASCULAR_BLUR_FRACTION, 1.0))

    peak = blurred.max()
    if peak <= 0:
        return np.zeros((full_h, full_w), dtype=np.float32)
    normalized = blurred / peak

    return cv2.resize(normalized, (full_w, full_h), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def locate_disc_classical(image: np.ndarray, use_vascular_prior: bool = True) -> dict:
    """Locate the optic disc as the center of the brightest disc-sized
    *compact* patch within the field of view -- found via the average
    brightness within a sliding window sized to the expected disc diameter
    (cv2.boxFilter), not a per-pixel brightness threshold.

    A global threshold (Otsu or otherwise) + largest-connected-component
    looks appealing but breaks on real fundus photos: diffuse bright
    lesions (e.g. diabetic retinopathy exudates) or a smooth illumination
    gradient across the frame can each form a connected "bright" region far
    larger than the actual disc, which a global threshold can't
    distinguish from the real thing. Windowed *average* brightness instead
    directly measures "is this a solid, compact, uniformly bright patch
    the size of a disc" -- scattered lesions with gaps between them, or a
    gradual gradient with no strong local peak, score much lower than the
    disc's genuinely solid bright area does.

    Resizes to VESSEL_WORKING_WIDTH internally (see vessels.py's module
    docstring for why this must happen inside the function, not be left to
    the caller) -- safe to call with either a raw native-resolution image
    or an already-working-resolution one, since resizing to the same size
    is a no-op.

    Brightness alone is not enough to be sure the patch found IS the disc,
    so the candidate's geometry is then checked (assess_disc_plausibility()):
    "confident"=False flags a candidate whose shape doesn't look like a disc
    (most importantly, a large hemorrhage or dense exudate cluster that won
    the brightness search) so downstream stages can degrade rather than
    silently compute a CDR from the wrong anatomy.

    Returns {"center_xy": (x, y), "diameter_px": float, "found": bool,
    "confident": bool, "circularity": float, "solidity": float,
    "diameter_fraction": float, "implausible_reasons": [str, ...]} in
    working-image coordinates. "found"=False (with a fallback center/
    diameter estimate so callers never crash on a degenerate image, e.g. a
    synthetic test image or a photo with no visible disc) means no FOV --
    or no brightness variation at all -- was found to search within; the
    shape metrics are NaN in that case. found=True with confident=False is
    the interesting one: a disc-sized bright patch WAS found, it just doesn't
    look like a disc.
    """
    working = vessels._resize_to_working_width(image)
    green = vessels.extract_vessel_channel(working)
    fov = vessels._fov_mask(green)
    h, w = green.shape[:2]
    expected_diameter = w * _EXPECTED_DISC_DIAMETER_FRACTION

    if not fov.any():
        return _unlocated_disc((w / 2.0, h / 2.0), expected_diameter, "no field of view found")

    fov_pixels = green[fov]
    if fov_pixels.min() == fov_pixels.max():
        ys, xs = np.nonzero(fov)
        return _unlocated_disc((float(xs.mean()), float(ys.mean())), expected_diameter, "no brightness variation in field of view")

    window_size = max(int(expected_diameter), 3)
    windowed_brightness = cv2.boxFilter(green.astype(np.float32), ddepth=-1, ksize=(window_size, window_size))

    # Anatomical prior: the disc is where the vessels converge, not merely the
    # brightest compact patch (see compute_vascular_convergence). An all-zero
    # map means vessel extraction found nothing to work with -- multiplying by
    # it would zero the whole score surface and make the argmax below
    # meaningless, so in that case brightness is used alone, exactly as before
    # this prior existed.
    score = windowed_brightness
    convergence_used = False
    if use_vascular_prior:
        convergence = compute_vascular_convergence(working)
        if convergence.max() > 0:
            w_prior = _VASCULAR_PRIOR_WEIGHT
            score = windowed_brightness * ((1.0 - w_prior) + w_prior * convergence)
            convergence_used = True

    candidate = np.where(fov, score, -1.0)
    peak_y, peak_x = np.unravel_index(np.argmax(candidate), candidate.shape)
    center_xy = (float(peak_x), float(peak_y))

    geometry = _disc_candidate_geometry(green, fov, center_xy, expected_diameter)
    plausibility = assess_disc_plausibility(geometry, expected_diameter, image_width=w)

    diameter_px = geometry["diameter_px"] if geometry["measured"] else expected_diameter
    in_range = (
        _DIAMETER_REFINEMENT_MIN_FACTOR * expected_diameter
        <= diameter_px
        <= _DIAMETER_REFINEMENT_MAX_FACTOR * expected_diameter
    )
    if not in_range:
        diameter_px = expected_diameter

    return {
        "center_xy": center_xy,
        "diameter_px": float(diameter_px),
        "found": True,
        "confident": plausibility["plausible"],
        "circularity": geometry["circularity"],
        "solidity": geometry["solidity"],
        "diameter_fraction": geometry["diameter_px"] / w if geometry["measured"] else float("nan"),
        "implausible_reasons": plausibility["reasons"],
        "vascular_prior_used": convergence_used,
    }


def _disc_candidate_geometry(green: np.ndarray, fov: np.ndarray, center_xy: tuple, expected_diameter: float) -> dict:
    """Measure the SHAPE of the bright blob sitting under the brightness-peak
    candidate center, using Otsu + largest-connected-component within a small
    window CENTERED ON THE ALREADY-FOUND peak, rather than across the whole
    image -- local enough that a distant lesion cluster or the far side of an
    illumination gradient can't reach in and distort it, unlike the same
    technique applied globally.

    Returns {"diameter_px", "circularity", "solidity", "measured"}.
    "measured"=False (with NaN shape metrics) means the window was degenerate
    -- no FOV, no brightness variation, or no blob above the local threshold
    -- so there is nothing whose shape could be assessed; callers should fall
    back to the expected-diameter prior and treat the candidate as
    unverified rather than as verified-implausible.

    Circularity is 4*pi*area / perimeter^2: 1.0 for a perfect circle, falling
    toward 0 as a blob gets more ragged or elongated. It is the single most
    discriminative cue here (AUC 0.945 for separating correct from incorrect
    localizations on ADAM), because a real optic disc is close to circular
    while the two things that beat it in a pure brightness search -- a
    hemorrhage and a dense exudate cluster -- are not.

    IMPORTANT, and unintuitive: a correctly-located disc scores only ~0.34
    here on real photos, not ~0.9. The blob being measured is a raw local
    Otsu threshold, not a clean disc outline -- vessels cut through it and
    the threshold leaves a ragged edge, and a ragged edge inflates the
    perimeter, which circularity squares in the denominator. So the useful
    reading of this number is RELATIVE (correct ~0.34 vs incorrect ~0.07),
    and _MIN_DISC_CIRCULARITY is set accordingly low. Anyone "fixing" the
    threshold up toward a textbook 0.8 would flag essentially every image.
    """
    unmeasured = {"diameter_px": float(expected_diameter), "circularity": float("nan"), "solidity": float("nan"), "measured": False}

    h, w = green.shape[:2]
    cx, cy = center_xy
    half = expected_diameter
    x0, x1 = max(0, int(cx - half)), min(w, int(cx + half))
    y0, y1 = max(0, int(cy - half)), min(h, int(cy + half))
    local, local_fov = green[y0:y1, x0:x1], fov[y0:y1, x0:x1]

    if local.size == 0 or not local_fov.any():
        return unmeasured
    local_pixels = local[local_fov]
    if local_pixels.min() == local_pixels.max():
        return unmeasured

    local_bright = (local > threshold_otsu(local_pixels)) & local_fov
    component = _largest_component_mask(local_bright)
    if not component.any():
        return unmeasured

    region = max(regionprops(label(component)), key=lambda r: r.area)
    perimeter = region.perimeter
    if perimeter <= 0:
        return unmeasured

    # A pixelated blob's perimeter estimate is slightly biased, which can push
    # circularity marginally above 1.0 for a very small near-perfect circle --
    # clamp so downstream comparisons stay on a well-defined [0, 1] scale.
    circularity = min(4.0 * np.pi * region.area / (perimeter**2), 1.0)

    return {
        "diameter_px": float(region.equivalent_diameter_area),
        "circularity": float(circularity),
        "solidity": float(region.solidity),
        "measured": True,
    }


def assess_disc_plausibility(geometry: dict, expected_diameter: float, image_width: int) -> dict:
    """Decide whether a located disc candidate is geometrically plausible as
    an actual optic disc, given _disc_candidate_geometry()'s shape metrics.

    This exists because locate_disc_classical()'s brightness search answers
    "where is the brightest disc-sized patch in the field of view" -- which
    is a strictly weaker question than "where is the optic disc". On a fundus
    photo with a large hemorrhage or a dense exudate cluster, the brightest
    disc-sized patch can be the lesion, not the disc, and nothing downstream
    would notice: crop_disc_roi() would happily crop around the lesion, the
    Stage 6.2 U-Net would segment *something* disc-shaped out of whatever it
    was handed, and compute_cdr() would report a confident-looking CDR
    computed from the wrong anatomy entirely. A wrong crop must not silently
    produce a wrong CDR, so the shape checks here give callers a signal to
    degrade on.

    Returns {"plausible": bool, "reasons": [str, ...]} -- `reasons` is empty
    when plausible, and otherwise names each failed check, so a caller (or a
    report) can say *why* localization is low-confidence rather than just
    that it is.
    """
    if not geometry["measured"]:
        return {"plausible": False, "reasons": ["shape not measurable (degenerate window)"]}

    reasons = []
    circularity = geometry["circularity"]
    diameter_fraction = geometry["diameter_px"] / image_width if image_width > 0 else 0.0

    if circularity < _MIN_DISC_CIRCULARITY:
        reasons.append(f"not disc-shaped (circularity {circularity:.2f} < {_MIN_DISC_CIRCULARITY})")
    if not (_MIN_DISC_DIAMETER_FRACTION <= diameter_fraction <= _MAX_DISC_DIAMETER_FRACTION):
        reasons.append(
            f"implausible size (diameter {diameter_fraction:.3f} of image width, "
            f"expected {_MIN_DISC_DIAMETER_FRACTION}-{_MAX_DISC_DIAMETER_FRACTION})"
        )

    return {"plausible": not reasons, "reasons": reasons}


def crop_disc_roi(working_image: np.ndarray, center_xy: tuple, diameter_px: float, roi_width: int = DISC_ROI_WIDTH) -> tuple:
    """Crop a square window of side `_DISC_ROI_CROP_MULTIPLE * diameter_px`
    around `center_xy`, resized to `roi_width` x `roi_width`. If the window
    would extend past the image bounds (common for an off-center disc near
    the frame edge), it's SHIFTED to stay in bounds rather than truncated,
    so the result is always exactly square -- a non-square crop would break
    every downstream assumption that ROI-space is a fixed roi_width square.

    Returns (roi_image, bbox_meta) where bbox_meta = {"x0", "y0", "x1",
    "y1"} are the crop's bounds in working-image coordinates -- the only
    information reproject_roi_mask_to_working() needs to invert the crop.
    """
    h, w = working_image.shape[:2]
    cx, cy = center_xy
    half = max((_DISC_ROI_CROP_MULTIPLE * diameter_px) / 2.0, 1.0)

    x0, y0, x1, y1 = cx - half, cy - half, cx + half, cy + half
    if x0 < 0:
        x1 -= x0
        x0 = 0.0
    if y0 < 0:
        y1 -= y0
        y0 = 0.0
    if x1 > w:
        x0 -= x1 - w
        x1 = float(w)
    if y1 > h:
        y0 -= y1 - h
        y1 = float(h)
    x0, y0 = max(x0, 0.0), max(y0, 0.0)

    x0i, y0i = int(round(x0)), int(round(y0))
    # min() here (rather than trusting x1/y1 directly) guards the
    # degenerate case where the requested window is larger than the image
    # itself -- always produces a valid, in-bounds, square crop.
    side = max(min(int(round(x1)) - x0i, int(round(y1)) - y0i, w - x0i, h - y0i), 1)
    x1i, y1i = x0i + side, y0i + side

    roi = working_image[y0i:y1i, x0i:x1i]
    roi_resized = cv2.resize(roi, (roi_width, roi_width), interpolation=cv2.INTER_LINEAR)
    return roi_resized, {"x0": x0i, "y0": y0i, "x1": x1i, "y1": y1i}


def reproject_roi_mask_to_working(roi_mask: np.ndarray, bbox_meta: dict, working_shape: tuple) -> np.ndarray:
    """Inverse of crop_disc_roi(): resize a ROI-resolution mask back down
    to the crop's original pixel size and paste it into a working-image-
    sized canvas at the recorded bbox location. INTER_NEAREST keeps the
    result boolean (no interpolated in-between values at the mask edge).
    """
    canvas = np.zeros(working_shape[:2], dtype=bool)
    x0, y0, x1, y1 = bbox_meta["x0"], bbox_meta["y0"], bbox_meta["x1"], bbox_meta["y1"]
    target_w, target_h = x1 - x0, y1 - y0
    if target_w <= 0 or target_h <= 0:
        return canvas
    resized = cv2.resize(roi_mask.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    canvas[y0:y1, x0:x1] = resized.astype(bool)
    return canvas


def extract_color_features(roi_image: np.ndarray) -> np.ndarray:
    """Build the 7-channel input Stage 6.2's model consumes from a BGR ROI
    crop: RGB + Lab(a, b) + HSV(H, S), each normalized to roughly [0, 1]
    float32, stacked as (7, H, W) -- see optic_disc_model.py's module
    docstring for why these specific channels. Shared building block for
    both training (optic_disc_dataset.py) and inference
    (optic_disc_infer.py) so the channel order/normalization can never
    drift between the two -- the same role vessels.compute_frangi_response()
    plays for the vessel pipeline.
    """
    rgb = cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    lab = cv2.cvtColor(roi_image, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV).astype(np.float32)

    # OpenCV encodes Lab's a/b as uint8 in [0, 255] representing the signed
    # range [-128, 127] -- normalize onto roughly the same [0, 1] scale as
    # the other channels rather than leaving them on a different range.
    lab_ab = lab[:, :, 1:3] / 255.0
    # OpenCV's 8-bit HSV convention: H in [0, 179], S in [0, 255].
    hsv_hs = np.stack([hsv[:, :, 0] / 179.0, hsv[:, :, 1] / 255.0], axis=-1)

    stacked = np.concatenate([rgb, lab_ab, hsv_hs], axis=-1)  # (H, W, 7)
    return np.transpose(stacked, (2, 0, 1)).astype(np.float32)  # (7, H, W)


def segment_disc_cup_classical(roi_image: np.ndarray) -> tuple:
    """Classical disc/cup mask within an already-cropped ROI: the disc is
    the largest bright connected region (same idea as locate_disc_classical,
    now at ROI scale); the cup is the palest sub-region *within* the disc,
    found via Otsu thresholding restricted to disc pixels (the neuroretinal
    rim is more vascular/reddish, the cup is pale from tissue loss -- the
    same pallor cue Stage 6.2's trained model also relies on, just captured
    here with a single global threshold instead of a learned boundary).

    This is a genuinely weaker estimate than the trained model -- it exists
    only so compute_optic_biomarkers() (and compute_optic_biomarkers_auto()
    in optic_disc_infer.py) keep working with no checkpoint, mirroring
    vessels.segment_vessels()'s role as vessel_infer.py's fallback.
    """
    green = vessels.extract_vessel_channel(roi_image)
    if green.min() == green.max():
        return np.zeros_like(green, dtype=bool), np.zeros_like(green, dtype=bool)

    threshold = threshold_otsu(green)
    disc_candidate = green > threshold

    if not disc_candidate.any():
        return np.zeros_like(green, dtype=bool), np.zeros_like(green, dtype=bool)

    disc_pixels = green[disc_candidate]
    if disc_pixels.min() == disc_pixels.max():
        # A perfectly flat disc region has no pallor contrast to threshold
        # on -- there's no basis to claim a cup exists.
        return _largest_component_mask(disc_candidate), np.zeros_like(green, dtype=bool)

    cup_threshold = threshold_otsu(disc_pixels)
    cup_candidate = (green >= cup_threshold) & disc_candidate
    return clean_disc_cup_masks(disc_candidate, cup_candidate)


def enforce_cup_within_disc(disc_mask: np.ndarray, cup_mask: np.ndarray) -> np.ndarray:
    """The structural nesting guarantee: the cup is anatomically part of
    the disc, so a cup mask can never legitimately extend outside the disc
    mask. Independent per-class postprocessing (e.g. largest-connected-
    component cleanup applied separately to each class) could otherwise
    produce a cup mask that pokes outside the disc mask -- this is the
    single choke point every code path routes through before returning a
    cup mask, so that violation can never leak out.
    """
    return cup_mask & disc_mask


def clean_disc_cup_masks(disc_mask: np.ndarray, cup_mask: np.ndarray) -> tuple:
    """Post-processing step applied to a RAW disc/cup mask -- straight off
    a per-pixel threshold (classical) or an argmax over class logits
    (hybrid, see optic_disc_infer.segment_disc_cup_hybrid()) -- before any
    geometry (compute_cdr(), overlay drawing) is computed from it:

    1. Keep only the disc's largest connected component. A per-pixel
       decision (threshold or argmax) can leave stray bright fragments or
       edge-bleeding speckles disconnected from the true disc -- these
       corrupt _vertical_extent()-based measurements badly, since even a
       single stray pixel far from the real disc inflates the bounding-box
       height used for CDR.
    2. Restrict the cup to the now-CLEANED disc via enforce_cup_within_disc()
       -- doing this after step 1, not before, matters: a cup fragment that
       only looked "inside the disc" because of a stray disc fragment
       (since discarded) must not survive either.
    3. Keep only the cup's largest connected component too, so a cup that's
       drifted into multiple disconnected pieces collapses to one coherent
       central region instead of leaving orphaned speckles of its own.

    Returns (disc_mask, cup_mask), both boolean, each a single connected
    component (or empty).
    """
    disc_mask = _largest_component_mask(disc_mask)
    cup_mask = _largest_component_mask(enforce_cup_within_disc(disc_mask, cup_mask))
    return disc_mask, cup_mask


def _vertical_extent(mask: np.ndarray) -> int:
    """Height in pixels of the mask's bounding box (max row - min row + 1
    over True pixels), 0 if the mask is empty. The basis for vertical CDR,
    the clinically standard cup-to-disc ratio definition (as opposed to a
    horizontal or area-based ratio).
    """
    rows_with_mask = np.any(mask, axis=1)
    if not rows_with_mask.any():
        return 0
    row_indices = np.nonzero(rows_with_mask)[0]
    return int(row_indices[-1] - row_indices[0] + 1)


def compute_cdr(disc_mask: np.ndarray, cup_mask: np.ndarray) -> dict:
    """Vertical cup-to-disc ratio from a disc mask and cup mask (of the
    same shape). Always runs both through clean_disc_cup_masks() first --
    keeping only the disc's largest connected component and restricting
    the cup to it -- so the returned "disc_mask"/"cup_mask", and every
    ratio computed from them, satisfy the structural guarantees regardless
    of what the caller passed in. segment_disc_cup_classical() and
    segment_disc_cup_hybrid() already do this same cleanup themselves
    before compute_cdr() is ever called in the normal pipeline -- this is
    a second, defensive application (cheap and a no-op on an
    already-clean mask) so a stray disc fragment can never inflate
    _vertical_extent()'s bounding-box measurement even for a caller that
    doesn't route through them.

    Returns {"vertical_cdr", "disc_diameter_px", "cup_diameter_px",
    "disc_mask", "cup_mask"}. 0.0/0/0 if disc_mask is empty (mirrors
    vessels.average_vessel_width()'s degenerate-input handling).
    """
    disc_mask, cup_mask = clean_disc_cup_masks(disc_mask, cup_mask)
    disc_diameter_px = _vertical_extent(disc_mask)
    cup_diameter_px = _vertical_extent(cup_mask)
    vertical_cdr = float(cup_diameter_px) / disc_diameter_px if disc_diameter_px > 0 else 0.0
    return {
        "vertical_cdr": vertical_cdr,
        "disc_diameter_px": disc_diameter_px,
        "cup_diameter_px": cup_diameter_px,
        "disc_mask": disc_mask,
        "cup_mask": cup_mask,
    }


def locate_macula_classical(working_image: np.ndarray, disc_center_xy: tuple, disc_diameter_px: float) -> dict:
    """Locate the macula/fovea heuristically: the darkest point within a
    search window centered `_MACULA_SEARCH_RADIUS_FACTOR` disc-diameters
    from the disc center, tried on both sides along the horizontal
    meridian (no reliable eye-laterality info available), excluding the
    disc region itself. A Gaussian blur first (scaled to disc size) avoids
    a single dark vessel pixel winning over the true, more diffusely dark
    macula/fovea region.

    Returns {"location_xy": (x, y) | None, "found": bool} in working-image
    coordinates. REFUGE2 (the Stage 6.2 training dataset) has no fovea
    coordinate labels, so unlike disc/cup segmentation, this stays a
    classical heuristic rather than a trained model -- see ROADMAP.md.
    """
    h, w = working_image.shape[:2]
    green = vessels.extract_vessel_channel(working_image)
    fov = vessels._fov_mask(green)
    cx, cy = disc_center_xy

    blur_sigma = max(disc_diameter_px * 0.15, 3.0)
    blurred = cv2.GaussianBlur(green, (0, 0), sigmaX=blur_sigma).astype(np.float64)

    search_dist = _MACULA_SEARCH_RADIUS_FACTOR * disc_diameter_px
    search_radius = max(disc_diameter_px * 0.75, 1.0)

    best_location = None
    best_darkness = None
    for direction in (-1.0, 1.0):
        sx, sy = cx + direction * search_dist, cy
        x_lo, x_hi = int(round(max(sx - search_radius, 0))), int(round(min(sx + search_radius, w)))
        y_lo, y_hi = int(round(max(sy - search_radius, 0))), int(round(min(sy + search_radius, h)))
        if x_hi <= x_lo or y_hi <= y_lo:
            continue

        window = blurred[y_lo:y_hi, x_lo:x_hi]
        window_fov = fov[y_lo:y_hi, x_lo:x_hi].copy()
        # Exclude anything still within reach of the disc itself.
        yy, xx = np.mgrid[y_lo:y_hi, x_lo:x_hi]
        window_fov &= (xx - cx) ** 2 + (yy - cy) ** 2 > (disc_diameter_px * _MACULA_DISC_EXCLUSION_FACTOR) ** 2
        if not window_fov.any():
            continue

        candidate = np.where(window_fov, window, np.inf)
        min_idx = np.unravel_index(np.argmin(candidate), candidate.shape)
        min_val = candidate[min_idx]
        if min_val == np.inf:
            continue
        if best_darkness is None or min_val < best_darkness:
            best_darkness = min_val
            best_location = (x_lo + int(min_idx[1]), y_lo + int(min_idx[0]))

    if best_location is None:
        return {"location_xy": None, "found": False}
    return {"location_xy": best_location, "found": True}


def compute_optic_biomarkers(image: np.ndarray) -> dict:
    """Full classical-only pipeline: locate disc -> crop ROI -> classical
    disc/cup segmentation -> reproject masks back to working-image space ->
    CDR -> macula. This is what optic_disc_infer.compute_optic_biomarkers_
    auto() falls back to when no trained checkpoint is available -- same
    role vessels.compute_biomarkers() plays for the vessel pipeline.

    Resizes to VESSEL_WORKING_WIDTH once up front so the same working image
    is reused for disc localization, ROI cropping, mask reprojection, and
    macula search -- avoids each sub-step independently (and redundantly)
    re-deriving it, and keeps every returned mask in one consistent
    coordinate space.
    """
    working = vessels._resize_to_working_width(image)
    disc_info = locate_disc_classical(working)
    roi_image, bbox_meta = crop_disc_roi(working, disc_info["center_xy"], disc_info["diameter_px"])
    roi_disc_mask, roi_cup_mask = segment_disc_cup_classical(roi_image)

    disc_mask = reproject_roi_mask_to_working(roi_disc_mask, bbox_meta, working.shape)
    cup_mask = reproject_roi_mask_to_working(roi_cup_mask, bbox_meta, working.shape)
    cdr_info = compute_cdr(disc_mask, cup_mask)
    macula_info = locate_macula_classical(working, disc_info["center_xy"], disc_info["diameter_px"])

    return {
        "disc_mask": cdr_info["disc_mask"],
        "cup_mask": cdr_info["cup_mask"],
        "vertical_cdr": cdr_info["vertical_cdr"],
        "disc_diameter_px": cdr_info["disc_diameter_px"],
        "cup_diameter_px": cdr_info["cup_diameter_px"],
        "macula_location": macula_info["location_xy"],
        "disc_found": disc_info["found"],
        "disc_confident": disc_info["confident"],
        "disc_localization_warnings": disc_info["implausible_reasons"],
        "macula_found": macula_info["found"],
        # Always "classical" here: Stage 6.0's coarse locator is a torch model,
        # so it lives in optic_disc_infer.py and cannot be reached from this
        # deliberately torch-free module. Reported anyway so this function and
        # compute_optic_biomarkers_hybrid() return the SAME KEYS -- a caller
        # switching between the two paths must not have to guard for a missing
        # key, which is the whole point of the shared return contract.
        "disc_localization_source": "classical",
    }
