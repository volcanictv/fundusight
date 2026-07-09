"""Phase 5: Vessel Segmentation.

Classical CV pipeline that extracts the retinal vessel tree from a fundus
photo and computes four biomarkers from it: vessel density, branch point
count, tortuosity, and average width. Independent of the DR detection model
in `src/detection/` — this is a separate, parallel branch of the pipeline.

Pipeline: green channel -> CLAHE -> Frangi vesselness filter -> Otsu
threshold -> small-object removal -> skeletonize.
"""

import cv2
import numpy as np
from scipy import ndimage
from skimage.filters import frangi, threshold_otsu
from skimage.measure import label
from skimage.morphology import remove_small_objects, skeletonize

_CLAHE_CLIP_LIMIT = 2.0
_CLAHE_TILE_GRID_SIZE = (8, 8)

# Same idea as preprocessing.quality._fundus_mask: these photos have a large
# black background around the circular retina field of view (FOV), and the
# hard edge at that boundary is itself a strong ridge that the Frangi filter
# will happily mistake for a vessel if it isn't masked out first.
_FOV_MIN_BRIGHTNESS = 10

# Vessels span a range of calibers (thin peripheral vessels to thick vessels
# near the optic disc), so Frangi is run across multiple scales rather than
# one fixed width.
_FRANGI_SIGMAS = range(1, 5)

# Drops speckle noise left over after thresholding the Frangi response —
# real vessel segments are long and thin, isolated blobs this small are not.
_MIN_VESSEL_OBJECT_SIZE = 30

# A "component" of only a few skeleton pixels is noise left over from
# thresholding, not an actual vessel segment — excluded from the
# branch/tortuosity biomarkers so a handful of stray pixels don't skew them.
_MIN_TORTUOSITY_COMPONENT_SIZE = 5


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


def segment_vessels(image: np.ndarray) -> np.ndarray:
    """Full vessel segmentation: green channel -> CLAHE -> Frangi vesselness
    filter -> Otsu threshold (inside the FOV only) -> small-object removal.
    Returns a boolean mask the same shape as the input image.
    """
    green = extract_vessel_channel(image)
    fov = _fov_mask(green)
    enhanced = enhance_vessel_contrast(green)

    # Vessels are ridges *darker* than surrounding tissue in the green
    # channel, hence black_ridges=True. frangi() expects a float image.
    vesselness = frangi(enhanced.astype(np.float64) / 255.0, sigmas=_FRANGI_SIGMAS, black_ridges=True)

    fov_response = vesselness[fov]
    if fov_response.size == 0 or fov_response.max() <= 0:
        return np.zeros_like(fov, dtype=bool)

    threshold = threshold_otsu(fov_response)
    mask = (vesselness > threshold) & fov

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
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    return float(distance[skeleton].mean()) * 2.0


def compute_biomarkers(image: np.ndarray) -> dict:
    """Run the full vessel segmentation pipeline and compute all four
    biomarkers. Returns the mask and skeleton alongside the numbers so
    callers (e.g. a demo/visualization script) don't need to recompute them.
    """
    mask = segment_vessels(image)
    skeleton = skeletonize_vessels(mask)
    fov = _fov_mask(extract_vessel_channel(image))

    return {
        "vessel_density": vessel_density(mask, fov),
        "branch_count": branch_point_count(skeleton),
        "tortuosity": tortuosity(skeleton),
        "average_width": average_vessel_width(mask, skeleton),
        "mask": mask,
        "skeleton": skeleton,
    }
