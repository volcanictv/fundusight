"""Phase 2: Preprocessing.

Three independent, chainable steps applied to a raw fundus photo before it
goes to any model: illumination correction, local contrast enhancement
(CLAHE), and color normalization.
"""

import cv2
import numpy as np

# Gaussian sigma for the illumination background estimate, as a fraction of
# the image's longer side. Scaling by image size (rather than a fixed pixel
# value) keeps behavior consistent across this dataset's wide range of
# resolutions (~600px to ~3200px).
_ILLUMINATION_SIGMA_FRACTION = 1 / 30

_CLAHE_CLIP_LIMIT = 2.0
_CLAHE_TILE_GRID_SIZE = (8, 8)

_COLOR_NORM_TARGET_MEAN = 128.0
_COLOR_NORM_TARGET_STD = 50.0


def correct_illumination(image: np.ndarray) -> np.ndarray:
    """Flatten uneven lighting (e.g. a brighter center, dimmer edges) by
    estimating the low-frequency background with a large Gaussian blur and
    subtracting it out. A big-enough blur washes out fine detail like
    vessels and lesions, leaving just the slow-varying lighting pattern —
    subtracting that leaves the detail with even illumination. The +128
    offset re-centers the result on mid-gray so subtracted regions don't
    clip to black.
    """
    h, w = image.shape[:2]
    sigma = max(h, w) * _ILLUMINATION_SIGMA_FRACTION
    background = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)
    corrected = cv2.addWeighted(image, 1, background, -1, 128)
    return corrected


def apply_clahe(image: np.ndarray) -> np.ndarray:
    """Boost local contrast via CLAHE on the L (lightness) channel of LAB
    color space. LAB separates lightness from color, so enhancing contrast
    there (rather than per RGB channel) avoids introducing color shifts —
    important since disease severity in later phases is partly judged by
    color (e.g. hemorrhage redness).
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=_CLAHE_CLIP_LIMIT, tileGridSize=_CLAHE_TILE_GRID_SIZE)
    l_channel = clahe.apply(l_channel)

    enhanced_lab = cv2.merge((l_channel, a_channel, b_channel))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def normalize_color(image: np.ndarray) -> np.ndarray:
    """Rescale each color channel to a fixed mean/std so brightness and
    contrast are comparable across photos taken with different cameras or
    settings, instead of the model having to learn to ignore that variation.
    """
    normalized = np.empty_like(image, dtype=np.float32)
    for c in range(image.shape[2]):
        channel = image[:, :, c].astype(np.float32)
        mean, std = channel.mean(), channel.std()
        if std < 1e-6:
            normalized[:, :, c] = _COLOR_NORM_TARGET_MEAN
            continue
        rescaled = (channel - mean) / std * _COLOR_NORM_TARGET_STD + _COLOR_NORM_TARGET_MEAN
        normalized[:, :, c] = rescaled

    return np.clip(normalized, 0, 255).astype(np.uint8)


def preprocess(image: np.ndarray) -> np.ndarray:
    """Run the full preprocessing chain: illumination correction, then
    CLAHE, then color normalization. Order matters — CLAHE works on
    already-flattened lighting, and normalization is a final pass over the
    fully enhanced image.
    """
    corrected = correct_illumination(image)
    contrast_enhanced = apply_clahe(corrected)
    return normalize_color(contrast_enhanced)
