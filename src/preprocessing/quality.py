"""Phase 1: Image Quality Assessment.

Determine whether a fundus photo is usable before running any model on it.
Checks: focus (Laplacian variance), exposure/illumination (histogram stats).
"""

import cv2
import numpy as np

# Fundus photos vary a lot in resolution in this dataset (~600px to ~3200px
# wide). Laplacian variance scales with resolution — more pixels means more
# edges to measure — so scores from a small and a huge image aren't
# comparable unless everything is resized to the same width first.
_FOCUS_RESIZE_WIDTH = 512

# Thresholds below were calibrated empirically against a 150-image sample of
# the real APTOS dataset (see notebooks/) rather than guessed, because these
# photos have a large black background around the circular retina field of
# view (FOV) that would otherwise skew brightness/focus stats toward "bad".
_FOV_MIN_BRIGHTNESS = 10  # pixels darker than this are background, not FOV

_FOCUS_POOR = 5.0
_FOCUS_GOOD = 100.0
_FOCUS_PASS_THRESHOLD = 15.0

_EXPOSURE_POOR_LOW = 25.0
_EXPOSURE_GOOD_LOW = 50.0
_EXPOSURE_GOOD_HIGH = 120.0
_EXPOSURE_POOR_HIGH = 140.0


def _fundus_mask(gray: np.ndarray) -> np.ndarray:
    """Boolean mask of the circular retina FOV, excluding the black
    background. Brightness/focus stats are only meaningful inside the FOV —
    a photo with a tiny visible retina and huge black borders would
    otherwise look "underexposed" even though the retina itself is fine.
    """
    _, mask = cv2.threshold(gray, _FOV_MIN_BRIGHTNESS, 255, cv2.THRESH_BINARY)
    return mask.astype(bool)


def _lerp_score(value: float, poor: float, good: float) -> float:
    """Map `value` onto a 0-100 scale: at/beyond `poor` -> 0, at/beyond
    `good` -> 100, linear in between. Gives a graded score instead of a
    hard pass/fail cutoff. Works whether `good` > `poor` or `good` < `poor`.
    """
    if good == poor:
        return 100.0 if value >= good else 0.0
    fraction = (value - poor) / (good - poor)
    return float(np.clip(fraction, 0.0, 1.0) * 100)


def check_focus(gray: np.ndarray, mask: np.ndarray) -> dict:
    """Variance of the Laplacian, measured inside the FOV only: sharp edges
    produce large second derivatives and high variance; blur smooths them
    out and variance collapses toward zero.
    """
    h, w = gray.shape
    scale = _FOCUS_RESIZE_WIDTH / w
    size = (_FOCUS_RESIZE_WIDTH, int(h * scale))
    resized = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)

    laplacian = cv2.Laplacian(resized, cv2.CV_64F)
    variance = float(laplacian[resized_mask].var()) if resized_mask.any() else 0.0

    return {
        "passed": variance >= _FOCUS_PASS_THRESHOLD,
        "score": _lerp_score(variance, _FOCUS_POOR, _FOCUS_GOOD),
        "laplacian_variance": variance,
    }


def check_exposure(gray: np.ndarray, mask: np.ndarray) -> dict:
    """Flags under/over-exposed images via mean brightness inside the FOV."""
    fov_pixels = gray[mask]
    mean_brightness = float(fov_pixels.mean()) if fov_pixels.size else 0.0

    if mean_brightness < _EXPOSURE_GOOD_LOW:
        score = _lerp_score(mean_brightness, _EXPOSURE_POOR_LOW, _EXPOSURE_GOOD_LOW)
    elif mean_brightness > _EXPOSURE_GOOD_HIGH:
        score = _lerp_score(mean_brightness, _EXPOSURE_POOR_HIGH, _EXPOSURE_GOOD_HIGH)
    else:
        score = 100.0

    passed = _EXPOSURE_POOR_LOW <= mean_brightness <= _EXPOSURE_POOR_HIGH

    return {
        "passed": passed,
        "score": score,
        "mean_brightness": mean_brightness,
    }


def assess_quality(image: np.ndarray) -> dict:
    """Score a fundus photo 0-100 and report pass/fail per check.

    `image` is a BGR array as returned by cv2.imread (or a single-channel
    grayscale array).
    """
    if image is None or image.size == 0:
        raise ValueError("assess_quality received an empty image")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    mask = _fundus_mask(gray)

    checks = {
        "focus": check_focus(gray, mask),
        "exposure": check_exposure(gray, mask),
    }
    score = round(sum(c["score"] for c in checks.values()) / len(checks), 1)

    return {
        "score": score,
        "passed": all(c["passed"] for c in checks.values()),
        "checks": checks,
    }
