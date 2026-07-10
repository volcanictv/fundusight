"""Phase 8/9: overlay thumbnails for the report/dashboard.

Draws the vessel mask and disc/cup/macula results on top of the shared
working-resolution image (see pipeline.run_pipeline's "working_image" key).
Reuses the exact color conventions already established in
scripts/demo_vessels.py and scripts/demo_optic_disc.py (vessel mask = red,
disc = yellow, cup = red, macula marker = green circle) so a reader who has
already seen those demo grids recognizes the same visual language here.
"""

import cv2
import numpy as np

_VESSEL_MASK_COLOR = (0, 0, 255)  # red, BGR -- matches demo_vessels.py
_DISC_COLOR = (0, 255, 255)  # yellow, BGR -- matches demo_optic_disc.py
_CUP_COLOR = (0, 0, 255)  # red, BGR -- matches demo_optic_disc.py
_MACULA_COLOR = (0, 255, 0)  # green, BGR -- matches demo_optic_disc.py
_MACULA_MARKER_RADIUS = 12


def vessel_mask_overlay(working_image: np.ndarray, vessel_result: dict) -> np.ndarray:
    """working_image with vessel_result["mask"] painted on top in solid
    red. `working_image` must already be at the mask's resolution (both
    vessel_infer.compute_biomarkers_auto() and the classical fallback
    return a mask sized to vessels.VESSEL_WORKING_WIDTH).
    """
    overlay = working_image.copy()
    overlay[vessel_result["mask"]] = _VESSEL_MASK_COLOR
    return overlay


def optic_disc_overlay(working_image: np.ndarray, optic_disc_result: dict) -> np.ndarray:
    """working_image with the disc mask, cup mask, and macula marker drawn
    on top. Cup is drawn after disc so it wins any overlap (same order
    demo_optic_disc.py uses) -- masks are already nested (cup within disc)
    by the time they reach here, see optic_disc.enforce_cup_within_disc().
    """
    overlay = working_image.copy()
    overlay[optic_disc_result["disc_mask"]] = _DISC_COLOR
    overlay[optic_disc_result["cup_mask"]] = _CUP_COLOR
    if optic_disc_result["macula_location"] is not None:
        cv2.circle(overlay, optic_disc_result["macula_location"], _MACULA_MARKER_RADIUS, _MACULA_COLOR, 2)
    return overlay
