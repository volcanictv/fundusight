import cv2
import numpy as np
import pytest

from src.preprocessing.quality import assess_quality, check_exposure, check_focus, _fundus_mask


def _synthetic_fundus(brightness=90, size=400, pattern=False):
    """A black square with a bright circular disc in the middle, mimicking
    a fundus photo's FOV against its black background. `pattern=True` adds
    high-frequency detail inside the disc so focus checks have something to
    measure (a flat-color disc has ~zero Laplacian variance regardless of
    blur, which would make focus tests meaningless).
    """
    img = np.zeros((size, size), dtype=np.uint8)
    center = size // 2
    radius = size // 2 - 10
    cv2.circle(img, (center, center), radius, brightness, thickness=-1)
    if pattern:
        rng = np.random.default_rng(0)
        noise = rng.integers(-40, 40, size=img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        # re-mask so the background stays pure black after adding noise
        disc_mask = np.zeros_like(img)
        cv2.circle(disc_mask, (center, center), radius, 255, thickness=-1)
        img = np.where(disc_mask > 0, img, 0)
    return img


def test_sharp_image_passes_focus_check():
    sharp = _synthetic_fundus(pattern=True)
    mask = _fundus_mask(sharp)
    result = check_focus(sharp, mask)
    assert result["passed"]
    assert result["score"] > 50


def test_blurry_image_fails_focus_check():
    sharp = _synthetic_fundus(pattern=True)
    blurry = cv2.GaussianBlur(sharp, (31, 31), sigmaX=10)
    mask = _fundus_mask(blurry)
    result = check_focus(blurry, mask)
    assert not result["passed"]


def test_normal_brightness_passes_exposure_check():
    img = _synthetic_fundus(brightness=90)
    mask = _fundus_mask(img)
    result = check_exposure(img, mask)
    assert result["passed"]


def test_dark_image_fails_exposure_check():
    # 18 is above the FOV-mask cutoff (10) so the disc still counts as
    # foreground, but below the underexposed threshold (25).
    img = _synthetic_fundus(brightness=18)
    mask = _fundus_mask(img)
    result = check_exposure(img, mask)
    assert not result["passed"]


def test_bright_image_fails_exposure_check():
    img = _synthetic_fundus(brightness=250)
    mask = _fundus_mask(img)
    result = check_exposure(img, mask)
    assert not result["passed"]


def test_assess_quality_end_to_end():
    bgr = cv2.cvtColor(_synthetic_fundus(pattern=True), cv2.COLOR_GRAY2BGR)
    result = assess_quality(bgr)
    assert set(result.keys()) == {"score", "passed", "checks"}
    assert set(result["checks"].keys()) == {"focus", "exposure"}
    assert result["passed"]
    assert 0 <= result["score"] <= 100


def test_assess_quality_rejects_empty_image():
    with pytest.raises(ValueError):
        assess_quality(np.empty((0, 0, 3), dtype=np.uint8))
