import cv2
import numpy as np

from src.preprocessing.enhance import apply_clahe, correct_illumination, normalize_color, preprocess


def test_correct_illumination_flattens_gradient():
    # A left-to-right brightness gradient stands in for uneven lighting
    # (e.g. a photo brighter on one side than the other).
    size = 300
    gradient = np.tile(np.linspace(50, 220, size, dtype=np.uint8), (size, 1))
    image = cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)

    corrected = correct_illumination(image)

    def side_difference(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        quarter = size // 4
        return abs(gray[:, -quarter:].mean() - gray[:, :quarter].mean())

    # Correction should shrink the left/right brightness gap substantially,
    # not eliminate it perfectly (a large Gaussian blur is an approximation).
    assert side_difference(corrected) < side_difference(image) * 0.3


def test_apply_clahe_increases_contrast():
    # A two-tone image with a narrow brightness range mimics a low-contrast
    # photo; CLAHE should stretch that range out locally.
    size = 200
    low_contrast = np.zeros((size, size), dtype=np.uint8)
    low_contrast[:, : size // 2] = 110
    low_contrast[:, size // 2 :] = 130
    image = cv2.cvtColor(low_contrast, cv2.COLOR_GRAY2BGR)

    enhanced = apply_clahe(image)

    before_std = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).std()
    after_std = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY).std()
    assert after_std > before_std


def test_normalize_color_matches_target_stats():
    rng = np.random.default_rng(0)
    image = rng.normal(loc=180, scale=10, size=(200, 200, 3)).clip(0, 255).astype(np.uint8)

    normalized = normalize_color(image)

    for c in range(3):
        channel = normalized[:, :, c].astype(np.float32)
        assert abs(channel.mean() - 128.0) < 5
        assert abs(channel.std() - 50.0) < 5


def test_normalize_color_handles_flat_channel():
    # A perfectly flat channel (std=0) shouldn't divide by zero.
    image = np.full((50, 50, 3), 100, dtype=np.uint8)
    normalized = normalize_color(image)
    assert normalized.shape == image.shape


def test_preprocess_preserves_shape_and_dtype():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, size=(150, 200, 3), dtype=np.uint8)

    result = preprocess(image)

    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert result.min() >= 0 and result.max() <= 255
