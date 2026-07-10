import cv2
import numpy as np

from src.segmentation.optic_disc import (
    DISC_ROI_WIDTH,
    _largest_component_mask,
    _vertical_extent,
    clean_disc_cup_masks,
    compute_cdr,
    compute_optic_biomarkers,
    crop_disc_roi,
    enforce_cup_within_disc,
    extract_color_features,
    locate_disc_classical,
    locate_macula_classical,
    reproject_roi_mask_to_working,
    segment_disc_cup_classical,
)
from src.segmentation.vessels import VESSEL_WORKING_WIDTH

# Tissue/disc/cup/macula brightness levels for the synthetic fundus below --
# spread far enough apart that percentile/Otsu thresholding cleanly
# separates them, the same way a real fundus photo's disc pallor stands out
# against surrounding tissue.
_TISSUE_VALUE = 90
_DISC_VALUE = 200
_CUP_VALUE = 240
_MACULA_VALUE = 20


def _fundus_with_disc(native_width, disc_offset_frac=(0.15, -0.05), disc_diameter_frac=0.14, cup_frac=0.4, macula=True):
    """A filled FOV circle (tissue) with a brighter disc circle, an even
    brighter cup circle inside it, and (optionally) a dark macula-like blob
    -- built at `native_width`, deliberately different from
    VESSEL_WORKING_WIDTH in the tests that use this, so locate_disc_classical
    and locate_macula_classical exercise the actual internal-resize code
    path. Since the canvas is square, resizing to VESSEL_WORKING_WIDTH scales
    both axes by the same factor, so fractional offsets/sizes here map
    directly onto fractions of VESSEL_WORKING_WIDTH after resize.
    """
    image = np.zeros((native_width, native_width, 3), dtype=np.uint8)
    center = native_width // 2
    cv2.circle(image, (center, center), int(native_width * 0.45), (_TISSUE_VALUE,) * 3, -1)

    disc_center = (int(center + native_width * disc_offset_frac[0]), int(center + native_width * disc_offset_frac[1]))
    disc_diameter = native_width * disc_diameter_frac
    cv2.circle(image, disc_center, int(disc_diameter / 2), (_DISC_VALUE,) * 3, -1)
    cv2.circle(image, disc_center, int(disc_diameter * cup_frac / 2), (_CUP_VALUE,) * 3, -1)

    if macula:
        # Placed on the opposite side of the disc from its offset, at the
        # ~2.5x-disc-diameter distance locate_macula_classical searches.
        macula_x = int(center - native_width * disc_offset_frac[0] - 2.5 * disc_diameter * np.sign(disc_offset_frac[0] or 1))
        macula_y = disc_center[1]
        cv2.circle(image, (macula_x, macula_y), int(disc_diameter * 0.3), (_MACULA_VALUE,) * 3, -1)

    return image, disc_center, disc_diameter


def test_largest_component_mask_picks_largest():
    binary = np.zeros((30, 30), dtype=bool)
    binary[2:5, 2:5] = True  # small blob, 9 px
    binary[10:20, 10:20] = True  # large blob, 100 px

    result = _largest_component_mask(binary)

    assert result.sum() == 100
    assert not np.any(result[2:5, 2:5])


def test_largest_component_mask_empty_input_returns_empty():
    binary = np.zeros((10, 10), dtype=bool)

    assert not _largest_component_mask(binary).any()


def test_locate_disc_classical_finds_bright_region():
    native_width = VESSEL_WORKING_WIDTH * 2
    image, disc_center, disc_diameter = _fundus_with_disc(native_width, macula=False)
    scale = VESSEL_WORKING_WIDTH / native_width
    expected_center = (disc_center[0] * scale, disc_center[1] * scale)
    expected_diameter = disc_diameter * scale

    result = locate_disc_classical(image)

    assert result["found"]
    assert abs(result["center_xy"][0] - expected_center[0]) < VESSEL_WORKING_WIDTH * 0.05
    assert abs(result["center_xy"][1] - expected_center[1]) < VESSEL_WORKING_WIDTH * 0.05
    assert abs(result["diameter_px"] - expected_diameter) < expected_diameter * 0.3


def test_locate_disc_classical_handles_empty_fov():
    image = np.zeros((300, 300, 3), dtype=np.uint8)

    result = locate_disc_classical(image)

    assert not result["found"]
    assert result["diameter_px"] > 0  # still a usable fallback, never zero/NaN


def test_crop_disc_roi_shape_and_squareness():
    working = np.zeros((VESSEL_WORKING_WIDTH, VESSEL_WORKING_WIDTH, 3), dtype=np.uint8)

    roi, bbox = crop_disc_roi(working, center_xy=(700, 700), diameter_px=150)

    assert roi.shape == (DISC_ROI_WIDTH, DISC_ROI_WIDTH, 3)
    assert bbox["x1"] - bbox["x0"] == bbox["y1"] - bbox["y0"]  # exactly square


def test_crop_disc_roi_shifts_instead_of_truncating_near_edge():
    working = np.zeros((VESSEL_WORKING_WIDTH, VESSEL_WORKING_WIDTH, 3), dtype=np.uint8)

    # Disc center right in the corner -- a naive centered crop would extend
    # far past the image bounds on two sides at once.
    roi, bbox = crop_disc_roi(working, center_xy=(5, 5), diameter_px=150)

    assert roi.shape == (DISC_ROI_WIDTH, DISC_ROI_WIDTH, 3)
    assert bbox["x0"] >= 0 and bbox["y0"] >= 0
    assert bbox["x1"] <= VESSEL_WORKING_WIDTH and bbox["y1"] <= VESSEL_WORKING_WIDTH
    assert bbox["x1"] - bbox["x0"] == bbox["y1"] - bbox["y0"]  # still exactly square


def test_reproject_roi_mask_to_working_places_mask_at_bbox():
    working_shape = (VESSEL_WORKING_WIDTH, VESSEL_WORKING_WIDTH)
    roi_mask = np.ones((DISC_ROI_WIDTH, DISC_ROI_WIDTH), dtype=bool)
    bbox = {"x0": 100, "y0": 200, "x1": 300, "y1": 400}

    canvas = reproject_roi_mask_to_working(roi_mask, bbox, working_shape)

    assert canvas.shape == working_shape
    assert canvas[200:400, 100:300].all()  # fully-True ROI fills its bbox
    assert not canvas[0:100, 0:100].any()  # nothing leaks outside the bbox


def test_extract_color_features_shape_and_range():
    roi = np.zeros((DISC_ROI_WIDTH, DISC_ROI_WIDTH, 3), dtype=np.uint8)
    roi[:, :, 0] = 60  # B
    roi[:, :, 1] = 120  # G
    roi[:, :, 2] = 200  # R

    features = extract_color_features(roi)

    assert features.shape == (7, DISC_ROI_WIDTH, DISC_ROI_WIDTH)
    assert features.dtype == np.float32
    assert features.min() >= 0.0 and features.max() <= 1.0
    # RGB channels (0, 1, 2) should reflect the R/G/B values set above,
    # confirming the BGR->RGB conversion, not just BGR passed through.
    assert abs(features[0].mean() - 200 / 255.0) < 1e-3  # R
    assert abs(features[1].mean() - 120 / 255.0) < 1e-3  # G
    assert abs(features[2].mean() - 60 / 255.0) < 1e-3  # B


def test_segment_disc_cup_classical_detects_bright_and_paler_regions():
    roi = np.full((DISC_ROI_WIDTH, DISC_ROI_WIDTH, 3), _TISSUE_VALUE, dtype=np.uint8)
    center = DISC_ROI_WIDTH // 2
    cv2.circle(roi, (center, center), DISC_ROI_WIDTH // 3, (_DISC_VALUE,) * 3, -1)
    cv2.circle(roi, (center, center), DISC_ROI_WIDTH // 6, (_CUP_VALUE,) * 3, -1)

    disc_mask, cup_mask = segment_disc_cup_classical(roi)

    assert disc_mask.any()
    assert cup_mask.any()
    assert not np.any(cup_mask & ~disc_mask)  # nesting guarantee holds here too


def test_clean_disc_cup_masks_discards_stray_disc_fragments():
    disc_mask = np.zeros((100, 100), dtype=bool)
    disc_mask[20:60, 20:60] = True  # true disc, 1600 px
    disc_mask[80:85, 80:85] = True  # stray disconnected speckle, 25 px
    cup_mask = np.zeros((100, 100), dtype=bool)
    cup_mask[35:45, 35:45] = True  # inside the true disc

    cleaned_disc, cleaned_cup = clean_disc_cup_masks(disc_mask, cup_mask)

    assert cleaned_disc.sum() == 1600
    assert not np.any(cleaned_disc[80:85, 80:85])  # speckle discarded
    assert cleaned_cup.sum() == 100


def test_clean_disc_cup_masks_drops_cup_fragment_orphaned_by_disc_cleanup():
    disc_mask = np.zeros((100, 100), dtype=bool)
    disc_mask[20:60, 20:60] = True  # true disc
    disc_mask[80:85, 80:85] = True  # stray disconnected disc speckle
    cup_mask = np.zeros((100, 100), dtype=bool)
    cup_mask[35:45, 35:45] = True  # real cup, inside the true disc
    cup_mask[80:85, 80:85] = True  # only "inside disc" because of the stray speckle

    cleaned_disc, cleaned_cup = clean_disc_cup_masks(disc_mask, cup_mask)

    # Once the stray disc speckle is discarded, the cup pixels that only
    # overlapped it must go too -- not survive as an orphaned fragment.
    assert not np.any(cleaned_cup[80:85, 80:85])
    assert cleaned_cup.sum() == 100


def test_clean_disc_cup_masks_collapses_multi_component_cup_to_largest():
    disc_mask = np.zeros((100, 100), dtype=bool)
    disc_mask[10:90, 10:90] = True
    cup_mask = np.zeros((100, 100), dtype=bool)
    cup_mask[30:50, 30:50] = True  # larger cup fragment, 400 px
    cup_mask[60:65, 60:65] = True  # smaller, disconnected cup fragment, 25 px -- both inside disc

    _, cleaned_cup = clean_disc_cup_masks(disc_mask, cup_mask)

    assert cleaned_cup.sum() == 400
    assert not np.any(cleaned_cup[60:65, 60:65])


def test_enforce_cup_within_disc_removes_out_of_bounds_pixels():
    disc_mask = np.zeros((20, 20), dtype=bool)
    disc_mask[5:15, 5:15] = True
    cup_mask = np.zeros((20, 20), dtype=bool)
    cup_mask[8:18, 8:18] = True  # deliberately pokes outside disc_mask

    result = enforce_cup_within_disc(disc_mask, cup_mask)

    assert not np.any(result & ~disc_mask)
    assert result.sum() == 7 * 7  # only the overlapping 7x7 region survives


def test_vertical_extent_measures_bounding_box_height():
    mask = np.zeros((20, 20), dtype=bool)
    mask[3:9, :] = True  # rows 3..8 -> height 6

    assert _vertical_extent(mask) == 6


def test_vertical_extent_zero_for_empty_mask():
    assert _vertical_extent(np.zeros((10, 10), dtype=bool)) == 0


def test_compute_cdr_basic_ratio():
    disc_mask = np.zeros((100, 100), dtype=bool)
    disc_mask[10:90, 10:90] = True  # vertical extent 80
    cup_mask = np.zeros((100, 100), dtype=bool)
    cup_mask[30:70, 30:70] = True  # vertical extent 40, fully inside disc

    result = compute_cdr(disc_mask, cup_mask)

    assert result["disc_diameter_px"] == 80
    assert result["cup_diameter_px"] == 40
    assert abs(result["vertical_cdr"] - 0.5) < 1e-6


def test_compute_cdr_enforces_cup_within_disc():
    disc_mask = np.zeros((100, 100), dtype=bool)
    disc_mask[20:60, 20:60] = True
    cup_mask = np.zeros((100, 100), dtype=bool)
    cup_mask[10:90, 10:90] = True  # deliberately extends far outside disc_mask

    result = compute_cdr(disc_mask, cup_mask)

    assert not np.any(result["cup_mask"] & ~disc_mask)


def test_compute_cdr_handles_empty_disc():
    disc_mask = np.zeros((50, 50), dtype=bool)
    cup_mask = np.zeros((50, 50), dtype=bool)

    result = compute_cdr(disc_mask, cup_mask)

    assert result == {
        "vertical_cdr": 0.0,
        "disc_diameter_px": 0,
        "cup_diameter_px": 0,
        "disc_mask": result["disc_mask"],
        "cup_mask": result["cup_mask"],
    }
    assert not result["disc_mask"].any()
    assert not result["cup_mask"].any()


def test_compute_cdr_discards_stray_disc_fragment_from_measurement():
    # A stray fragment far from the true disc would otherwise inflate the
    # vertical-extent-based diameter measurement -- this is the CDR-
    # correctness motivation for compute_cdr() cleaning defensively, not
    # just cosmetic overlay quality.
    disc_mask = np.zeros((100, 100), dtype=bool)
    disc_mask[40:60, 40:60] = True  # true disc, vertical extent 20
    disc_mask[95:97, 95:97] = True  # tiny stray fragment near the far edge
    cup_mask = np.zeros((100, 100), dtype=bool)
    cup_mask[45:55, 45:55] = True  # vertical extent 10, inside the true disc

    result = compute_cdr(disc_mask, cup_mask)

    assert result["disc_diameter_px"] == 20  # not inflated by the stray fragment
    assert result["cup_diameter_px"] == 10
    assert abs(result["vertical_cdr"] - 0.5) < 1e-6


def test_locate_macula_classical_finds_dark_region():
    native_width = VESSEL_WORKING_WIDTH // 2
    image, disc_center, disc_diameter = _fundus_with_disc(native_width, macula=True)
    scale = VESSEL_WORKING_WIDTH / native_width
    working = cv2.resize(image, (VESSEL_WORKING_WIDTH, VESSEL_WORKING_WIDTH))
    disc_center_working = (disc_center[0] * scale, disc_center[1] * scale)
    disc_diameter_working = disc_diameter * scale

    result = locate_macula_classical(working, disc_center_working, disc_diameter_working)

    assert result["found"]
    # The macula blob is dark (value 20) against tissue (value 90) -- the
    # found location should sample as dark in the green channel.
    x, y = result["location_xy"]
    assert working[y, x, 1] < _TISSUE_VALUE


def test_compute_optic_biomarkers_returns_expected_keys_and_shapes():
    native_width = VESSEL_WORKING_WIDTH // 2
    image, _, _ = _fundus_with_disc(native_width)

    result = compute_optic_biomarkers(image)

    assert set(result.keys()) == {
        "disc_mask",
        "cup_mask",
        "vertical_cdr",
        "disc_diameter_px",
        "cup_diameter_px",
        "macula_location",
        "disc_found",
        "macula_found",
    }
    assert result["disc_mask"].shape == (VESSEL_WORKING_WIDTH, VESSEL_WORKING_WIDTH)
    assert result["cup_mask"].shape == (VESSEL_WORKING_WIDTH, VESSEL_WORKING_WIDTH)
    assert result["disc_mask"].dtype == bool
    assert result["cup_mask"].dtype == bool
    assert not np.any(result["cup_mask"] & ~result["disc_mask"])
    assert isinstance(result["vertical_cdr"], float) and result["vertical_cdr"] >= 0
    assert result["disc_found"]
