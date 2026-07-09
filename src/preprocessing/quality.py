"""Phase 1: Image Quality Assessment.

Determine whether a fundus photo is usable before running any model on it.
Checks: focus (Laplacian variance), exposure/illumination (histogram stats).
"""

def assess_quality(image) -> dict:
    """Return a quality score (0-100) and pass/fail per-check breakdown.

    TODO (Phase 1): implement focus, exposure, and illumination checks.
    """
    raise NotImplementedError


