"""Phase 5: Vessel Segmentation.

Classical CV module that extracts the retinal vessel tree from a fundus
photo and computes four biomarkers from it: vessel density, branch point
count, tortuosity, and average width. Independent of the DR detection model
in `src/detection/` — this is a separate, parallel branch of the pipeline.

This module stays classical and torch-free on purpose: `compute_biomarkers()`
and `segment_vessels()` here only ever produce a mask via
`compute_frangi_response()` + a hysteresis threshold, no trained model
involved, so anything that only needs the classical baseline (tests, the
demo script) never pulls in torch. The hybrid classical+learned pipeline —
where `compute_frangi_response()`'s *unthresholded* Frangi response is fed
as an input channel to a trained U-Net instead of being thresholded
directly — lives in `vessel_infer.py`, which imports from here rather than
the other way around.

Pipeline: resize to a canonical working resolution -> green channel ->
CLAHE -> multi-scale Frangi vesselness filter -> hysteresis threshold ->
small-object removal -> skeletonize.

APTOS images arrive at inconsistent native resolutions (e.g. 1736x2416 vs
1050x1050), and Frangi's sigmas are absolute pixel scales — the same sigma
range corresponds to a different physical vessel width depending on which
photo came in. `compute_frangi_response()` resizes every input to
`VESSEL_WORKING_WIDTH` internally so the sigma range (and everything
downstream, classical or hybrid) means the same thing across every image,
not just for well-behaved callers.
"""

import cv2
import numpy as np
from scipy import ndimage
from skimage.filters import apply_hysteresis_threshold, frangi, threshold_otsu
from skimage.measure import label
from skimage.morphology import remove_small_objects, skeletonize

_CLAHE_CLIP_LIMIT = 2.0
_CLAHE_TILE_GRID_SIZE = (8, 8)

# Same idea as preprocessing.quality._fundus_mask: these photos have a large
# black background around the circular retina field of view (FOV), and the
# hard edge at that boundary is itself a strong ridge that the Frangi filter
# will happily mistake for a vessel if it isn't masked out first.
_FOV_MIN_BRIGHTNESS = 10

# Canonical resolution segment_vessels() resizes every input to before
# running Frangi — see module docstring for why this must happen inside the
# function rather than being left to the caller. Every other constant below
# that's in pixel units (_FRANGI_SIGMAS, _MIN_VESSEL_OBJECT_SIZE) is tuned
# for images at this width specifically.
VESSEL_WORKING_WIDTH = 1400

# Vessels span a range of calibers (thin peripheral vessels to thick vessels
# near the optic disc), so Frangi is run across multiple scales rather than
# one fixed width. COUPLED TO VESSEL_WORKING_WIDTH=1400: these are absolute
# pixel scales tuned to match real vessel calibers at that resolution —
# changing VESSEL_WORKING_WIDTH requires re-tuning this range to match, it
# does not rescale automatically.
_FRANGI_SIGMAS = (3, 5, 7, 9, 11, 13, 15)

# Hysteresis threshold (see segment_vessels): pixels >= _HIGH seed the mask,
# pixels >= _HIGH * this fraction are kept only if connected to a seed. This
# lets faint-but-connected thin-vessel response survive without admitting
# isolated noise, unlike a single global cutoff.
_HYSTERESIS_LOW_FRACTION = 0.6

# Drops speckle noise left over after thresholding the Frangi response —
# real vessel segments are long and thin, isolated blobs this small are not.
# COUPLED TO VESSEL_WORKING_WIDTH=1400: a "vessel-sized" component is a much
# larger pixel count at this resolution than at a small thumbnail — this
# must scale with VESSEL_WORKING_WIDTH too.
_MIN_VESSEL_OBJECT_SIZE = 500

# A "component" of only a few skeleton pixels is noise left over from
# thresholding, not an actual vessel segment — excluded from the
# branch/tortuosity biomarkers so a handful of stray pixels don't skew them.
_MIN_TORTUOSITY_COMPONENT_SIZE = 5


def _resize_to_working_width(image: np.ndarray) -> np.ndarray:
    """Resize to VESSEL_WORKING_WIDTH, preserving aspect ratio. INTER_AREA is
    the correct choice for shrinking (avoids aliasing thin vessels away);
    INTER_LINEAR for enlarging, since APTOS's smaller native images (e.g.
    1050x1050) need to be upsized to reach the working resolution.
    """
    h, w = image.shape[:2]
    scale = VESSEL_WORKING_WIDTH / w
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(image, (VESSEL_WORKING_WIDTH, round(h * scale)), interpolation=interpolation)


def extract_vessel_channel(image: np.ndarray) -> np.ndarray:
    """Isolate the green channel of a BGR fundus photo. Hemoglobin absorbs
    green light strongly, so vessels show up as their darkest, highest
    contrast against background tissue in this channel — the standard first
    step in classical retinal vessel segmentation.
    """
    return image[:, :, 1]


def enhance_vessel_contrast(channel: np.ndarray) -> np.ndarray:
    """CLAHE on the (grayscale) green channel, boosting local contrast so
    thin/faint vessels stand out enough for the Frangi filter to pick up.
    Same clip limit/tile size as `preprocessing.enhance.apply_clahe`.
    """
    clahe = cv2.createCLAHE(clipLimit=_CLAHE_CLIP_LIMIT, tileGridSize=_CLAHE_TILE_GRID_SIZE)
    return clahe.apply(channel)


def _fov_mask(gray: np.ndarray) -> np.ndarray:
    """Boolean mask of the circular retina FOV, excluding the black
    background border — see module docstring for why this matters here.
    """
    _, mask = cv2.threshold(gray, _FOV_MIN_BRIGHTNESS, 255, cv2.THRESH_BINARY)
    return mask.astype(bool)


def compute_frangi_response(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Resize to VESSEL_WORKING_WIDTH, then compute the two feature channels
    vessel segmentation is built from: the CLAHE-enhanced green channel and
    the raw multi-scale Frangi vesselness response — WITHOUT thresholding
    the latter into a mask. Returns `(enhanced_green, vesselness)`, both
    float32 arrays at VESSEL_WORKING_WIDTH resolution, scaled to roughly
    [0, 1].

    This is the shared building block for both the classical mask (see
    segment_vessels(), which thresholds `vesselness` below) and the hybrid
    model (see vessel_infer.py, which instead feeds both arrays as input
    channels to a trained U-Net that learns its own correction). Resizing
    happens inside this function — not left to the caller — so the Frangi
    sigma range means the same physical vessel width regardless of a given
    image's native resolution (see module docstring).
    """
    image = _resize_to_working_width(image)
    green = extract_vessel_channel(image)
    enhanced = enhance_vessel_contrast(green)

    # Vessels are ridges *darker* than surrounding tissue in the green
    # channel, hence black_ridges=True. frangi() expects a float image.
    vesselness = frangi(enhanced.astype(np.float64) / 255.0, sigmas=_FRANGI_SIGMAS, black_ridges=True)

    return enhanced.astype(np.float32) / 255.0, vesselness.astype(np.float32)


def segment_vessels(image: np.ndarray) -> np.ndarray:
    """Classical vessel mask: compute_frangi_response() -> hysteresis
    threshold (inside the FOV only) -> small-object removal. Returns a
    boolean mask at VESSEL_WORKING_WIDTH resolution — NOT the same shape as
    the input image, since the input is canonicalized first (see module
    docstring).

    This is the no-trained-model fallback/baseline — see vessel_infer.py's
    segment_vessels_hybrid() for the trained-U-Net alternative, which has
    the same signature/return contract and can be swapped in directly.
    """
    working = _resize_to_working_width(image)
    fov = _fov_mask(extract_vessel_channel(working))
    _, vesselness = compute_frangi_response(working)

    fov_response = vesselness[fov]
    if fov_response.size == 0 or fov_response.max() <= 0:
        return np.zeros_like(fov, dtype=bool)

    # A single global threshold (e.g. Otsu alone) is biased toward the
    # strong response of thick arcade vessels and cuts off the weaker
    # response of thin ones. Hysteresis instead seeds the mask from
    # high-confidence pixels and grows into connected weaker response,
    # picking up thin vessels attached to the tree without admitting
    # disconnected noise.
    high = threshold_otsu(fov_response)
    low = high * _HYSTERESIS_LOW_FRACTION
    mask = apply_hysteresis_threshold(vesselness, low, high) & fov

    # max_size removes objects <= this size, so subtract 1 to keep objects
    # of exactly _MIN_VESSEL_OBJECT_SIZE pixels.
    return remove_small_objects(mask, max_size=_MIN_VESSEL_OBJECT_SIZE - 1)


def skeletonize_vessels(mask: np.ndarray) -> np.ndarray:
    """Thin a vessel mask down to a 1-pixel-wide centerline skeleton, the
    basis for the branch count, tortuosity, and width biomarkers below.
    """
    return skeletonize(mask)


def vessel_density(mask: np.ndarray, fov_mask: np.ndarray) -> float:
    """Fraction of the field of view occupied by vessels, as a percentage —
    a standard proxy for overall retinal vascular coverage.
    """
    fov_pixels = fov_mask.sum()
    if fov_pixels == 0:
        return 0.0
    return float(mask.sum()) / float(fov_pixels) * 100.0


def branch_point_count(skeleton: np.ndarray) -> int:
    """Count vessel branch points: skeleton pixels with 3+ skeleton
    neighbors (a straight/curved segment has exactly 2; an endpoint has 1;
    a branch or crossing has 3+). Computed via a 3x3 neighbor-sum
    convolution rather than walking the skeleton pixel by pixel.
    """
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    neighbor_counts = ndimage.convolve(skeleton.astype(np.uint8), kernel, mode="constant")
    return int(np.count_nonzero(skeleton & (neighbor_counts >= 3)))


def tortuosity(skeleton: np.ndarray) -> float:
    """Mean arc-length-over-chord-length ratio across vessel segments — 1.0
    for a dead-straight segment, higher for segments that wind. Arc length
    is approximated as the segment's pixel count (each skeleton step is
    ~1px); chord length is the max pairwise distance between the segment's
    convex-hull points (an efficient stand-in for its two farthest-apart
    points). Segments below `_MIN_TORTUOSITY_COMPONENT_SIZE` pixels are
    skipped as noise, not real vessel structure.
    """
    labeled, num_components = label(skeleton, connectivity=2, return_num=True)

    ratios = []
    for component_id in range(1, num_components + 1):
        ys, xs = np.nonzero(labeled == component_id)
        arc_length = len(ys)
        if arc_length < _MIN_TORTUOSITY_COMPONENT_SIZE:
            continue

        points = np.column_stack([xs, ys]).astype(np.float32)
        hull = cv2.convexHull(points).reshape(-1, 2)
        if len(hull) < 2:
            continue

        diffs = hull[:, None, :] - hull[None, :, :]
        chord_length = float(np.sqrt((diffs**2).sum(axis=-1)).max())
        if chord_length < 1e-6:
            continue

        # A run of diagonal steps (each really sqrt(2) apart) can make the
        # pixel-count arc length come out slightly under the chord length,
        # even though a segment can never be shorter than the straight line
        # between its own endpoints — floor at 1.0 (perfectly straight).
        ratios.append(max(arc_length / chord_length, 1.0))

    return float(np.mean(ratios)) if ratios else 1.0


def average_vessel_width(mask: np.ndarray, skeleton: np.ndarray) -> float:
    """Average vessel width in pixels: the distance-to-background transform
    of the mask, sampled at skeleton (centerline) locations, gives each
    point's local vessel radius; doubling and averaging gives mean width.
    """
    if not skeleton.any():
        return 0.0
    if mask.all():
        # cv2.distanceTransform has no background pixel to measure distance
        # to when the mask covers the entire image, and silently returns
        # FLT_MAX sentinels instead of erroring -- summing even a few of
        # those overflows float32. A fully-saturated mask only happens for a
        # degenerate/failed segmentation (e.g. an untrained or early hybrid
        # model), never a real vessel mask, so there's no meaningful width.
        return 0.0
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    return float(distance[skeleton].mean()) * 2.0


def compute_biomarkers(image: np.ndarray) -> dict:
    """Run the full vessel segmentation pipeline and compute all four
    biomarkers. Returns the mask and skeleton alongside the numbers so
    callers (e.g. a demo/visualization script) don't need to recompute them.

    Resizes to VESSEL_WORKING_WIDTH once up front (matching what
    segment_vessels() does internally) so the FOV mask lines up with the
    vessel mask/skeleton it's paired with below — segment_vessels() would
    otherwise canonicalize its own input independently, leaving this
    function's FOV mask at the original, different resolution.
    """
    working = _resize_to_working_width(image)
    mask = segment_vessels(working)
    skeleton = skeletonize_vessels(mask)
    fov = _fov_mask(extract_vessel_channel(working))

    return {
        "vessel_density": vessel_density(mask, fov),
        "branch_count": branch_point_count(skeleton),
        "tortuosity": tortuosity(skeleton),
        "average_width": average_vessel_width(mask, skeleton),
        "mask": mask,
        "skeleton": skeleton,
    }
