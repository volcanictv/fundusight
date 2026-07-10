import cv2
import numpy as np
import torch

from src.segmentation import optic_disc
from src.segmentation.optic_disc_infer import (
    _cached_model,
    compute_optic_biomarkers_auto,
    compute_optic_biomarkers_hybrid,
    load_optic_disc_model,
    segment_disc_cup_hybrid,
)
from src.segmentation.optic_disc_model import build_optic_disc_model
from src.segmentation.vessels import VESSEL_WORKING_WIDTH

_ROI_WIDTH = optic_disc.DISC_ROI_WIDTH


def _fundus_image(size=700):
    image = np.zeros((size, size, 3), dtype=np.uint8)
    cv2.circle(image, (size // 2, size // 2), int(size * 0.45), (90, 90, 90), -1)
    cv2.circle(image, (size // 2, size // 2), int(size * 0.1), (200, 200, 200), -1)
    return image


class _FixedLogitsModel(torch.nn.Module):
    """Test double: ignores its input entirely and always returns a fixed
    logits map -- a deterministic way to verify segment_disc_cup_hybrid()
    actually applies clean_disc_cup_masks() to whatever the model predicts,
    rather than just trusting an untrained model's already-well-behaved
    output (which the other tests here use, but can't exercise the
    fragment-discarding behavior itself).
    """

    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self._logits = logits

    def forward(self, x):
        return self._logits.expand(x.shape[0], -1, -1, -1)


def test_segment_disc_cup_hybrid_discards_stray_disc_fragment():
    size = 40
    logits = torch.zeros(1, 3, size, size)
    logits[0, 0] = 5.0  # background everywhere by default
    logits[0, 0, 10:25, 10:25], logits[0, 1, 10:25, 10:25] = -5.0, 5.0  # true disc rim blob, 225px
    logits[0, 0, 35:38, 35:38], logits[0, 1, 35:38, 35:38] = -5.0, 5.0  # stray disconnected speckle, 9px

    model = _FixedLogitsModel(logits)
    roi = np.zeros((size, size, 3), dtype=np.uint8)

    disc_mask, _ = segment_disc_cup_hybrid(roi, model)

    assert disc_mask.sum() == 225
    assert not np.any(disc_mask[35:38, 35:38])


def test_segment_disc_cup_hybrid_returns_expected_shape_and_dtype():
    model = build_optic_disc_model()
    model.eval()
    roi = np.zeros((_ROI_WIDTH, _ROI_WIDTH, 3), dtype=np.uint8)

    disc_mask, cup_mask = segment_disc_cup_hybrid(roi, model)

    assert disc_mask.shape == (_ROI_WIDTH, _ROI_WIDTH)
    assert cup_mask.shape == (_ROI_WIDTH, _ROI_WIDTH)
    assert disc_mask.dtype == bool
    assert cup_mask.dtype == bool


def test_segment_disc_cup_hybrid_enforces_nesting_even_untrained():
    # An untrained (random-weight) model's argmax output is internally
    # consistent by construction, so this mostly verifies
    # enforce_cup_within_disc() is actually wired into
    # segment_disc_cup_hybrid(), not just defined and unused -- if someone
    # accidentally dropped that call, this would only show up as a subtle
    # bug on real (imperfect) model output, not in this synthetic check.
    model = build_optic_disc_model()
    model.eval()
    roi = np.random.randint(0, 255, (_ROI_WIDTH, _ROI_WIDTH, 3), dtype=np.uint8)

    disc_mask, cup_mask = segment_disc_cup_hybrid(roi, model)

    assert not np.any(cup_mask & ~disc_mask)


def test_compute_optic_biomarkers_hybrid_returns_same_keys_as_classical():
    model = build_optic_disc_model()
    model.eval()

    result = compute_optic_biomarkers_hybrid(_fundus_image(), model)

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
    assert not np.any(result["cup_mask"] & ~result["disc_mask"])


def test_load_optic_disc_model_restores_matching_weights(tmp_path):
    original = build_optic_disc_model()
    weights_path = str(tmp_path / "optic_disc_unet.pth")
    torch.save(original.state_dict(), weights_path)

    loaded = load_optic_disc_model(weights_path, device="cpu")

    for (name, orig_param), (_, loaded_param) in zip(original.named_parameters(), loaded.named_parameters()):
        assert torch.equal(orig_param, loaded_param), f"mismatch in {name}"
    assert not loaded.training  # eval() was called


def test_compute_optic_biomarkers_auto_falls_back_to_classical_without_checkpoint(tmp_path):
    image = _fundus_image()
    missing_weights = str(tmp_path / "does_not_exist.pth")

    result = compute_optic_biomarkers_auto(image, weights_path=missing_weights)
    expected = optic_disc.compute_optic_biomarkers(image)

    assert np.array_equal(result["disc_mask"], expected["disc_mask"])
    assert np.array_equal(result["cup_mask"], expected["cup_mask"])


def test_compute_optic_biomarkers_auto_uses_hybrid_when_checkpoint_exists(tmp_path):
    original = build_optic_disc_model()
    weights_path = str(tmp_path / "optic_disc_unet.pth")
    torch.save(original.state_dict(), weights_path)
    image = _fundus_image()

    result = compute_optic_biomarkers_auto(image, weights_path=weights_path)
    model = load_optic_disc_model(weights_path, device="cpu")
    expected = compute_optic_biomarkers_hybrid(image, model)

    assert np.array_equal(result["disc_mask"], expected["disc_mask"])
    assert np.array_equal(result["cup_mask"], expected["cup_mask"])


def test_cached_model_reuses_same_instance_for_same_weights_path(tmp_path):
    original = build_optic_disc_model()
    weights_path = str(tmp_path / "optic_disc_unet.pth")
    torch.save(original.state_dict(), weights_path)

    first = _cached_model(weights_path, "cpu")
    second = _cached_model(weights_path, "cpu")

    assert first is second
