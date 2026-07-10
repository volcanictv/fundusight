import numpy as np

from src.report.overlays import optic_disc_overlay, vessel_mask_overlay


def _working_image(size=200):
    return np.full((size, size, 3), 40, dtype=np.uint8)


def test_vessel_mask_overlay_paints_mask_pixels_red():
    image = _working_image()
    mask = np.zeros(image.shape[:2], dtype=bool)
    mask[10:20, 10:20] = True

    overlay = vessel_mask_overlay(image, {"mask": mask})

    assert overlay.shape == image.shape
    assert tuple(overlay[15, 15]) == (0, 0, 255)
    assert tuple(overlay[0, 0]) == tuple(image[0, 0])


def test_optic_disc_overlay_draws_disc_cup_and_macula():
    image = _working_image()
    disc_mask = np.zeros(image.shape[:2], dtype=bool)
    disc_mask[50:150, 50:150] = True
    cup_mask = np.zeros(image.shape[:2], dtype=bool)
    cup_mask[80:120, 80:120] = True
    result = {"disc_mask": disc_mask, "cup_mask": cup_mask, "macula_location": (170, 100)}

    overlay = optic_disc_overlay(image, result)

    assert tuple(overlay[60, 60]) == (0, 255, 255)  # disc-only pixel, yellow
    assert tuple(overlay[100, 100]) == (0, 0, 255)  # cup pixel, red -- wins over disc
    # macula marker is an unfilled circle around (170, 100) -- check the
    # bounding region contains the green marker color somewhere rather than
    # pinning an exact anti-aliased pixel.
    region = overlay[88:113, 158:183]
    assert np.any(np.all(region == (0, 255, 0), axis=-1))


def test_optic_disc_overlay_handles_no_macula_found():
    image = _working_image()
    mask = np.zeros(image.shape[:2], dtype=bool)
    result = {"disc_mask": mask, "cup_mask": mask, "macula_location": None}

    overlay = optic_disc_overlay(image, result)

    assert overlay.shape == image.shape
