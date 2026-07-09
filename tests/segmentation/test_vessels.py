import cv2
import numpy as np

from src.segmentation.vessels import (
    VESSEL_WORKING_WIDTH,
    average_vessel_width,
    branch_point_count,
    compute_biomarkers,
    compute_frangi_response,
    enhance_vessel_contrast,
    extract_vessel_channel,
    segment_vessels,
    skeletonize_vessels,
    tortuosity,
    vessel_density,
)

# Matches the caliber _FRANGI_SIGMAS (3..15) is tuned to detect, in pixels,
# at VESSEL_WORKING_WIDTH resolution.
_TARGET_LINE_WIDTH_AT_WORKING_RES = 12


def _fundus_image(native_width, fov_value=180, line=True):
    # Built at `native_width` -- deliberately different from
    # VESSEL_WORKING_WIDTH in the tests that use this -- with vessel
    # thickness scaled so that AFTER segment_vessels()'s internal resize to
    # VESSEL_WORKING_WIDTH, the line lands at a realistic width for the
    # tuned sigma range. This exercises the actual internal-resize code
    # path rather than only testing pre-canonicalized input.
    #
    # A filled disc on a black background stands in for a fundus photo's
    # circular field of view (FOV) against the black border; an optional
    # dark curved stroke through it stands in for a vessel (vessels are
    # darker than surrounding tissue in the green channel).
    image = np.zeros((native_width, native_width, 3), dtype=np.uint8)
    center = native_width // 2
    cv2.circle(image, (center, center), int(native_width * 0.45), (fov_value,) * 3, -1)
    if line:
        thickness = max(1, round(_TARGET_LINE_WIDTH_AT_WORKING_RES * native_width / VESSEL_WORKING_WIDTH))
        cv2.line(image, (int(native_width * 0.2), center), (int(native_width * 0.8), center), (40, 40, 40), thickness)
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
    # Native resolution is 2x VESSEL_WORKING_WIDTH, so segment_vessels()
    # must downsize internally before Frangi runs.
    native_width = VESSEL_WORKING_WIDTH * 2
    with_line = segment_vessels(_fundus_image(native_width, line=True))
    without_line = segment_vessels(_fundus_image(native_width, line=False))

    # Output is at the canonical working resolution, not the native input
    # resolution -- proof the internal resize actually ran.
    assert with_line.shape == (VESSEL_WORKING_WIDTH, VESSEL_WORKING_WIDTH)
    assert with_line.dtype == bool
    # A perfectly flat FOV (no texture at all) has zero Frangi response
    # everywhere, so nothing should be flagged as a vessel.
    assert without_line.sum() == 0
    # The drawn line should be picked up as vessel-like structure.
    assert with_line.sum() > 0


def test_compute_frangi_response_shapes_and_range():
    # Native resolution differs from VESSEL_WORKING_WIDTH, same as the
    # segment_vessels test above -- confirms the internal resize applies
    # here too, since segment_vessels() now depends on this function for it.
    native_width = VESSEL_WORKING_WIDTH * 2
    enhanced, vesselness = compute_frangi_response(_fundus_image(native_width, line=True))

    assert enhanced.shape == (VESSEL_WORKING_WIDTH, VESSEL_WORKING_WIDTH)
    assert vesselness.shape == (VESSEL_WORKING_WIDTH, VESSEL_WORKING_WIDTH)
    assert enhanced.dtype == np.float32
    assert vesselness.dtype == np.float32
    assert enhanced.min() >= 0.0 and enhanced.max() <= 1.0
    # The drawn line should produce a non-trivial (unthresholded) response.
    assert vesselness.max() > 0.0


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


def test_average_vessel_width_handles_fully_saturated_mask():
    # cv2.distanceTransform has no background pixel to measure distance to
    # when the mask covers the entire image, and returns FLT_MAX sentinels
    # rather than erroring -- summing even a few overflows float32. A
    # degenerate all-True mask (e.g. from an untrained hybrid model) isn't a
    # real vessel mask, so this should return a defined, finite value rather
    # than triggering that overflow.
    mask = np.ones((20, 20), dtype=bool)
    skeleton = np.zeros((20, 20), dtype=bool)
    skeleton[10, 10] = True

    assert average_vessel_width(mask, skeleton) == 0.0


def test_compute_biomarkers_returns_expected_keys_and_types():
    # Native resolution is half of VESSEL_WORKING_WIDTH here (vs. 2x in the
    # segment_vessels test above), so this exercises the internal-resize
    # upsize path -- APTOS's smaller native images (e.g. 1050x1050) need
    # this same path in real use.
    native_width = VESSEL_WORKING_WIDTH // 2
    result = compute_biomarkers(_fundus_image(native_width, line=True))

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
    assert result["mask"].shape == (VESSEL_WORKING_WIDTH, VESSEL_WORKING_WIDTH)
    assert result["skeleton"].shape == (VESSEL_WORKING_WIDTH, VESSEL_WORKING_WIDTH)
    assert result["mask"].dtype == bool
    assert result["skeleton"].dtype == bool
