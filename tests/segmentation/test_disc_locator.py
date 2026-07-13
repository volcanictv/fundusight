"""Stage 6.0: the coarse full-frame disc locator and the arbitration/fail-safe
layer that sits between it and Stage 6.1.

The arbitration tests deliberately use a STUB locator rather than the trained
checkpoint. What is being tested here is the POLICY -- who wins when the two
localizers agree, disagree, or both fail -- and that policy must hold no matter
what the network happens to predict. Binding these tests to a real checkpoint
would make them re-fail every time the model is retrained, for reasons that
have nothing to do with the logic they exist to pin down.
"""

import cv2
import numpy as np
import torch

from src.segmentation import optic_disc, optic_disc_infer
from src.segmentation.disc_locator_dataset import disc_bbox_relative
from src.segmentation.disc_locator_model import LOCATOR_INPUT_SIZE, build_disc_locator_model, soft_argmax
from src.segmentation.optic_disc_dataset import CLASS_BACKGROUND, CLASS_CUP, CLASS_DISC_RIM
from src.segmentation.vessels import VESSEL_WORKING_WIDTH


class _StubLocator(torch.nn.Module):
    """Always predicts the same relative box. Lets each arbitration test place
    the 'coarse locator' exactly where the scenario needs it."""

    def __init__(self, cx, cy, w=0.09, h=0.09):
        super().__init__()
        self.box = torch.tensor([[cx, cy, w, h]], dtype=torch.float32)

    def forward(self, x):
        return self.box.repeat(x.shape[0], 1)


def _working_fundus(disc_center_frac=(0.7, 0.5), disc_diameter_frac=0.09):
    """A working-resolution fundus: FOV disc of tissue, one bright round disc."""
    w = VESSEL_WORKING_WIDTH
    h = int(w * 0.75)
    image = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.circle(image, (w // 2, h // 2), int(h * 0.48), (90,) * 3, -1)
    center = (int(w * disc_center_frac[0]), int(h * disc_center_frac[1]))
    cv2.circle(image, center, int(w * disc_diameter_frac / 2), (200,) * 3, -1)
    return image, center


def _fundus_that_defeats_the_classical_localizer():
    """A fundus where the classical search finds a candidate but must REJECT it:
    a big, over-bright blob (0.16 of frame width -- larger than any real disc, so
    it trips the size gate) outshines a smaller, correctly-proportioned real disc
    elsewhere.

    This is the only fixture that puts locate_disc_classical() into the
    found=True / confident=False state the whole arbitration layer exists to
    handle. Building it is fiddly, which is exactly why the earlier version of
    these tests guarded their assertions behind `if not confident:` -- and then
    passed vacuously, asserting nothing at all, because the fixture never
    actually reached that state.

    Returns (image, blob_center, real_disc_center).
    """
    w = VESSEL_WORKING_WIDTH
    h = int(w * 0.75)
    image = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.circle(image, (w // 2, h // 2), int(h * 0.48), (90,) * 3, -1)

    # The confuser: oversized and brighter -- wins the brightness search, fails
    # the geometric size check.
    blob_center = (int(w * 0.68), int(h * 0.5))
    cv2.circle(image, blob_center, int(w * 0.16 / 2), (255,) * 3, -1)

    # The real disc: correctly proportioned, dimmer, so brightness never picks it.
    disc_center = (int(w * 0.30), int(h * 0.5))
    cv2.circle(image, disc_center, int(w * 0.085 / 2), (185,) * 3, -1)

    return image, blob_center, disc_center


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------


def test_disc_locator_output_shape_and_bounds():
    model = build_disc_locator_model()
    out = model(torch.randn(3, 3, LOCATOR_INPUT_SIZE, LOCATOR_INPUT_SIZE))

    assert out.shape == (3, 4)
    # Every output is a fraction of the frame -- a center or size outside [0, 1]
    # must be structurally unrepresentable, not merely discouraged by the loss.
    assert bool((out >= 0).all() and (out <= 1).all())


def test_soft_argmax_reads_the_peak_of_a_heatmap():
    # THE regression test for the GAP bug. The first version of this model
    # pooled globally before the coordinate head, which is translation-INVARIANT
    # by construction: it physically could not report WHERE a feature fired, so
    # it collapsed to predicting the dataset-mean position and ignoring the
    # image (val hit rate fell 0.31 -> 0.011 while train loss kept dropping).
    #
    # This pins the replacement readout exactly, on a hand-built heatmap. It is
    # deliberately NOT phrased as "an untrained network moves its prediction
    # when the input moves": an untrained heatmap is near-uniform, so that
    # readout sits at ~(0.5, 0.5) regardless and the effect size is pure
    # random-init luck -- a test that passes or fails on the seed, which is
    # exactly the flakiness this replaces.
    scores = torch.full((1, 16, 16), -10.0)
    scores[0, 12, 4] = 10.0  # row 12, col 4 -> y index 12, x index 4

    x, y = soft_argmax(scores)[0]

    # Cell centers, not cell edges: (4 + 0.5)/16 and (12 + 0.5)/16.
    assert abs(float(x) - (4 + 0.5) / 16) < 1e-3
    assert abs(float(y) - (12 + 0.5) / 16) < 1e-3


def test_soft_argmax_is_symmetric_for_a_uniform_heatmap():
    # A flat heatmap must read out as the exact frame center. If the +0.5
    # cell-center convention were dropped for `i / (w - 1)`, this would still
    # pass -- but test_soft_argmax_reads_the_peak_of_a_heatmap above would not,
    # which is why both exist.
    x, y = soft_argmax(torch.zeros(1, 16, 16))[0]

    assert abs(float(x) - 0.5) < 1e-5
    assert abs(float(y) - 0.5) < 1e-5


def test_disc_locator_position_depends_on_the_heatmap_not_on_pooled_features():
    # The architectural property, tested where it is actually decidable: two
    # different heatmaps over the SAME pooled features must yield different
    # centers. A GAP -> MLP head cannot satisfy this by construction, which is
    # precisely why it failed.
    features = torch.randn(1, 128, 16, 16)
    model = build_disc_locator_model()

    left = model.heatmap_logits_from_features(features).clone()
    left[0, :, :8] += 20.0  # pile all the mass on the left half
    right = model.heatmap_logits_from_features(features).clone()
    right[0, :, 8:] += 20.0  # ... and on the right half

    shift = (soft_argmax(right)[0, 0] - soft_argmax(left)[0, 0]).detach()
    assert float(shift) > 0.3


# --------------------------------------------------------------------------
# dataset target derivation
# --------------------------------------------------------------------------


def test_disc_bbox_relative_covers_rim_and_cup():
    mask = np.full((100, 200), CLASS_BACKGROUND, dtype=np.int64)
    mask[40:60, 80:120] = CLASS_DISC_RIM
    mask[45:55, 90:110] = CLASS_CUP  # cup is INSIDE the disc, must not shrink it

    cx, cy, w, h = disc_bbox_relative(mask)

    assert cx == (80 + 119) / 2 / 200
    assert cy == (40 + 59) / 2 / 100
    assert w == 40 / 200
    assert h == 20 / 100


def test_disc_bbox_relative_handles_mask_with_no_disc():
    mask = np.full((50, 50), CLASS_BACKGROUND, dtype=np.int64)

    box = disc_bbox_relative(mask)

    assert box.shape == (4,)
    assert not np.isnan(box).any()  # degenerate, but never NaN/raising


# --------------------------------------------------------------------------
# arbitration policy
# --------------------------------------------------------------------------


def test_confident_classical_wins_and_locator_is_not_consulted():
    # Policy 1. A confident classical localization was correct on 199/199 of
    # ADAM's confident cases -- a coarse 256px model must never be allowed to
    # overrule a signal with that observed precision. The stub here points
    # somewhere absurd on purpose; it must be ignored entirely.
    image, center = _working_fundus()
    absurd_locator = _StubLocator(0.05, 0.05)

    result = optic_disc_infer.locate_disc_arbitrated(image, absurd_locator)

    assert result["confident"]
    assert result["source"] == "classical"
    assert result["coarse_center_xy"] is None  # not even consulted
    assert abs(result["center_xy"][0] - center[0]) < image.shape[1] * 0.05


def test_locator_rescues_a_low_confidence_classical_localization():
    # Policy 2. The classical search lands on the oversized blob and rejects it;
    # the locator points at the real disc, whose shape DOES hold up, so the crop
    # is recovered and reported confident, tagged as coming from Stage 6.0.
    image, _blob, disc_center = _fundus_that_defeats_the_classical_localizer()
    h, w = image.shape[:2]

    classical = optic_disc.locate_disc_classical(image)
    # Assert the precondition rather than guarding on it. If the fixture stops
    # producing found=True/confident=False, this test must FAIL loudly, not slip
    # into passing vacuously -- which is precisely what the earlier `if`-guarded
    # version did.
    assert classical["found"] and not classical["confident"], "fixture no longer sets up the scenario under test"

    locator = _StubLocator(disc_center[0] / w, disc_center[1] / h)
    result = optic_disc_infer.locate_disc_arbitrated(image, locator)

    assert result["source"] == "coarse_locator"
    assert result["confident"]
    assert abs(result["center_xy"][0] - disc_center[0]) < w * 0.05


def test_both_localizers_failing_degrades_to_a_safe_in_retina_roi():
    # Policy 3. The whole point: a failed localization must degrade to a boring
    # in-FOV crop, NOT to whatever frame-corner artifact the failed search
    # landed on. A canvas/border crop is what makes downstream Grad-CAMs
    # hallucinate on image edges.
    image = np.zeros((600, 800, 3), dtype=np.uint8)
    cv2.circle(image, (400, 300), 280, (90,) * 3, -1)  # flat tissue, no disc
    # Locator also points at a corner -- nothing there is disc-shaped.
    locator = _StubLocator(0.02, 0.02)

    result = optic_disc_infer.locate_disc_arbitrated(image, locator)

    assert not result["confident"]
    assert result["source"] == "safe_fallback"
    cx, cy = result["center_xy"]
    # The returned center must be inside the retina, not out at the canvas edge.
    assert np.hypot(cx - 400, cy - 300) < 280


def test_an_unverified_but_in_retina_center_is_KEPT_not_overwritten():
    # Regression test for a real bug this design had. The fallback originally
    # replaced the center with the FOV centroid whenever confidence was low.
    # But the plausibility guard OVER-FLAGS on purpose (~20% of *correct*
    # localizations get flagged), so the low-confidence pool is mostly good
    # crops -- and overwriting them all destroyed 44 correct centers on ADAM to
    # rescue 10, dropping localization accuracy from 254/270 to 210/270.
    #
    # It also leaked past the CDR: crop_to_onh() (the glaucoma classifier's
    # input) uses this center and never consults `confident`, so those images
    # would have been classified on a crop of the central retina.
    #
    # So: an unconfident center that is still inside the retina must be RETAINED
    # as the best available estimate and merely reported unconfident. Only a
    # center outside the FOV gets replaced.
    image, _blob, _disc = _fundus_that_defeats_the_classical_localizer()
    # Locator points at flat tissue, where nothing is disc-shaped -- so neither
    # candidate verifies, which is the branch under test.
    useless_locator = _StubLocator(0.5, 0.85)

    classical = optic_disc.locate_disc_classical(image)
    assert classical["found"] and not classical["confident"], "fixture no longer sets up the scenario under test"

    result = optic_disc_infer.locate_disc_arbitrated(image, useless_locator)

    assert not result["confident"]
    assert result["source"] == "classical"
    # The center must be untouched -- NOT swapped for the FOV centroid. This is
    # the assertion that pins the 44-correct-localizations regression.
    assert result["center_xy"] == classical["center_xy"]


def test_no_locator_checkpoint_reduces_to_classical_behaviour():
    # The pipeline must still run on a fresh clone with no Stage 6.0 weights.
    image, center = _working_fundus()

    result = optic_disc_infer.locate_disc_arbitrated(image, locator_model=None)
    classical = optic_disc.locate_disc_classical(image)

    assert result["center_xy"] == classical["center_xy"]
    assert result["confident"] == classical["confident"]
    assert result["coarse_center_xy"] is None
