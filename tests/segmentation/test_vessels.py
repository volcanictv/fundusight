import cv2
import numpy as np

from src.segmentation.vessels import (
    average_vessel_width,
    branch_point_count,
    compute_biomarkers,
    enhance_vessel_contrast,
    extract_vessel_channel,
    segment_vessels,
    skeletonize_vessels,
    tortuosity,
    vessel_density,
)


def _fundus_image(size=200, fov_value=180, line=True):
    # A filled disc on a black background stands in for a fundus photo's
    # circular field of view (FOV) against the black border; an optional
    # dark curved stroke through it stands in for a vessel (vessels are
    # darker than surrounding tissue in the green channel).
    image = np.zeros((size, size, 3), dtype=np.uint8)
    center = size // 2
    cv2.circle(image, (center, center), int(size * 0.45), (fov_value,) * 3, -1)
    if line:
        cv2.line(image, (int(size * 0.2), center), (int(size * 0.8), center), (40, 40, 40), 3)
    return image


def test_extract_vessel_channel_returns_green_channel():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    image[:, :, 0] = 10  # B
    image[:, :, 1] = 200  # G
    image[:, :, 2] = 50  # R

    channel = extract_vessel_channel(image)

    assert channel.shape == (10, 10)
    assert np.all(channel == 200)


def test_enhance_vessel_contrast_increases_contrast():
    # A two-tone channel with a narrow brightness range mimics a low-contrast
    # green channel; CLAHE should stretch that range out locally.
    size = 200
    channel = np.zeros((size, size), dtype=np.uint8)
    channel[:, : size // 2] = 110
    channel[:, size // 2 :] = 130

    enhanced = enhance_vessel_contrast(channel)

    assert enhanced.std() > channel.std()


def test_segment_vessels_detects_line_and_ignores_flat_fov():
    with_line = segment_vessels(_fundus_image(line=True))
    without_line = segment_vessels(_fundus_image(line=False))

    assert with_line.shape == (200, 200)
    assert with_line.dtype == bool
    # A perfectly flat FOV (no texture at all) has zero Frangi response
    # everywhere, so nothing should be flagged as a vessel.
    assert without_line.sum() == 0
    # The drawn line should be picked up as vessel-like structure.
    assert with_line.sum() > 0


def test_skeletonize_vessels_thins_mask():
    mask = np.zeros((50, 50), dtype=bool)
    mask[10:30, 10:30] = True  # a thick filled blob

    skeleton = skeletonize_vessels(mask)

    assert skeleton.sum() < mask.sum()
    # Skeleton must lie entirely within the original mask.
    assert not np.any(skeleton & ~mask)


def test_vessel_density_computes_exact_fraction():
    mask = np.zeros((10, 10), dtype=bool)
    mask.flat[:15] = True  # exactly 15 of the 100 pixels are "vessel"
    fov_mask = np.ones((10, 10), dtype=bool)

    assert vessel_density(mask, fov_mask) == 15.0


def test_branch_point_count_zero_on_straight_line():
    skeleton = np.zeros((10, 10), dtype=bool)
    skeleton[5, 2:8] = True  # a plain horizontal segment, no junctions

    assert branch_point_count(skeleton) == 0


def test_branch_point_count_detects_y_junction():
    skeleton = np.zeros((10, 10), dtype=bool)
    # Three arms radiating from a shared center pixel (5, 5): left,
    # upper-right diagonal, lower-right diagonal.
    for x in (2, 3, 4):
        skeleton[5, x] = True
    for offset in (1, 2, 3):
        skeleton[5 - offset, 5 + offset] = True
        skeleton[5 + offset, 5 + offset] = True
    skeleton[5, 5] = True

    assert branch_point_count(skeleton) == 1


def test_tortuosity_higher_for_zigzag_than_straight():
    straight = np.zeros((20, 70), dtype=np.uint8)
    cv2.line(straight, (5, 10), (65, 10), 1, 1)

    # A path that repeatedly reverses direction in x while slowly
    # progressing in y: much longer arc length than the width/height of its
    # bounding box (its chord), unlike a straight or gently curved vessel.
    zigzag = np.zeros((20, 15), dtype=np.uint8)
    points = []
    for row in range(6):
        y = row * 3
        if row % 2 == 0:
            points += [(0, y), (10, y)]
        else:
            points += [(10, y), (0, y)]
    cv2.polylines(zigzag, [np.array(points, dtype=np.int32)], isClosed=False, color=1, thickness=1)

    straight_tortuosity = tortuosity(straight.astype(bool))
    zigzag_tortuosity = tortuosity(zigzag.astype(bool))

    assert straight_tortuosity < 1.3
    assert zigzag_tortuosity > straight_tortuosity * 2


def test_average_vessel_width_larger_for_thick_mask():
    thin_mask = np.zeros((10, 20), dtype=bool)
    thin_mask[5, 2:18] = True
    thin_skeleton = thin_mask  # already 1px wide

    thick_mask = np.zeros((15, 20), dtype=bool)
    thick_mask[3:12, 2:18] = True  # a 9px-tall strip
    thick_skeleton = skeletonize_vessels(thick_mask)

    thin_width = average_vessel_width(thin_mask, thin_skeleton)
    thick_width = average_vessel_width(thick_mask, thick_skeleton)

    assert thick_width > thin_width


def test_compute_biomarkers_returns_expected_keys_and_types():
    result = compute_biomarkers(_fundus_image(line=True))

    assert set(result.keys()) == {
        "vessel_density",
        "branch_count",
        "tortuosity",
        "average_width",
        "mask",
        "skeleton",
    }
    assert isinstance(result["vessel_density"], float) and result["vessel_density"] >= 0
    assert isinstance(result["branch_count"], int) and result["branch_count"] >= 0
    assert isinstance(result["tortuosity"], float)
    assert isinstance(result["average_width"], float) and result["average_width"] >= 0
    assert result["mask"].shape == (200, 200)
    assert result["skeleton"].shape == (200, 200)
    assert result["mask"].dtype == bool
    assert result["skeleton"].dtype == bool
