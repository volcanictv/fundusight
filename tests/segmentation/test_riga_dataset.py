"""RIGA label reconstruction.

These tests build synthetic "prime + annotator overlay" pairs, because the whole
risk with RIGA is a reconstruction that is silently WRONG rather than one that
crashes: a mis-recovered mask trains the model on a lie and still reports a
healthy Dice against that lie. So the tests assert the recovered geometry, and
assert that malformed input is REJECTED rather than guessed at.
"""

import cv2
import numpy as np

from src.segmentation.riga_dataset import (
    _MASK_BACKGROUND_VALUE,
    _MASK_CUP_VALUE,
    _MASK_DISC_RIM_VALUE,
    extract_disc_cup_from_annotation,
    fuse_annotations,
    masks_to_refuge_raw,
)

_CONTOUR_COLOUR = (30, 200, 30)


def _prime(size=400):
    """A plain fundus-ish photo with no annotation on it."""
    image = np.zeros((size, size, 3), dtype=np.uint8)
    cv2.circle(image, (size // 2, size // 2), int(size * 0.45), (60, 90, 160), -1)
    return image


def _annotated(prime, disc_r=40, cup_r=20, center=None, draw_cup=True):
    """The same photo with disc (and optionally cup) contours drawn on it -- the
    form RIGA actually ships."""
    out = prime.copy()
    c = center or (prime.shape[1] // 2, prime.shape[0] // 2)
    cv2.circle(out, c, disc_r, _CONTOUR_COLOUR, 2)
    if draw_cup:
        cv2.circle(out, c, cup_r, _CONTOUR_COLOUR, 2)
    return out


def test_extracts_nested_disc_and_cup():
    prime = _prime()
    annotation = _annotated(prime, disc_r=40, cup_r=20)

    disc, cup, reason = extract_disc_cup_from_annotation(prime, annotation)

    assert reason is None
    # Areas should be close to pi*r^2 (the filled contour includes its own 2px
    # stroke, so allow a little slack rather than demanding exactness).
    assert abs(disc.sum() - np.pi * 40**2) / (np.pi * 40**2) < 0.15
    assert abs(cup.sum() - np.pi * 20**2) / (np.pi * 20**2) < 0.25
    # The defining anatomical constraint: the cup lies inside the disc.
    assert not (cup & ~disc).any()
    assert cup.sum() < disc.sum()


def test_larger_contour_becomes_the_disc_regardless_of_draw_order():
    # The two curves come back as unordered connected components -- the code must
    # decide which is which by SIZE, not by whichever happened to be found first.
    prime = _prime()
    disc, cup, reason = extract_disc_cup_from_annotation(prime, _annotated(prime, disc_r=55, cup_r=15))

    assert reason is None
    assert disc.sum() > cup.sum()


def test_rejects_an_overlay_with_only_one_contour():
    # An annotator who drew the disc but no cup. There is no cup to recover, and
    # inventing one would be far worse than dropping the overlay.
    prime = _prime()
    annotation = _annotated(prime, draw_cup=False)

    disc, cup, reason = extract_disc_cup_from_annotation(prime, annotation)

    assert disc is None and cup is None
    assert "2 contour components" in reason


def test_rejects_an_unannotated_overlay():
    prime = _prime()

    disc, cup, reason = extract_disc_cup_from_annotation(prime, prime.copy())

    assert disc is None and cup is None
    assert "no contour" in reason


def test_rejects_non_nested_contours():
    # Two contours side by side rather than nested -- not a disc/cup pair at all.
    prime = _prime()
    annotation = prime.copy()
    cv2.circle(annotation, (140, 200), 40, _CONTOUR_COLOUR, 2)
    cv2.circle(annotation, (280, 200), 25, _CONTOUR_COLOUR, 2)

    disc, cup, reason = extract_disc_cup_from_annotation(prime, annotation)

    assert disc is None and cup is None
    assert "not nested" in reason


def test_fuse_annotations_takes_a_majority_not_a_union():
    # 3 of 5 annotators mark a pixel -> majority (>= 3) keeps it. 2 of 5 -> dropped.
    # A union would keep both, inflating every mask; an intersection would drop
    # both, shrinking them. Majority is the point of having six graders.
    shape = (10, 10)
    def mask(cols):
        m = np.zeros(shape, dtype=bool)
        m[:, cols] = True
        return m

    masks = [
        (mask(slice(0, 6)), mask(slice(0, 2))),
        (mask(slice(0, 6)), mask(slice(0, 2))),
        (mask(slice(0, 6)), mask(slice(0, 2))),
        (mask(slice(0, 4)), mask(slice(0, 1))),
        (mask(slice(0, 4)), mask(slice(0, 1))),
    ]

    disc, cup = fuse_annotations(masks)

    assert disc[:, 5].all()  # 3/5 voted -> majority, kept
    assert not disc[:, 6].any()  # 0/5
    assert cup[:, 1].all()  # 3/5
    assert not cup[:, 2].any()


def test_fused_cup_is_re_nested_inside_the_fused_disc():
    # Voting the two masks independently can admit a cup pixel where fewer than
    # half the annotators put disc -- anatomically impossible. The fusion must
    # re-nest rather than emit it.
    shape = (6, 6)
    big_cup = np.zeros(shape, dtype=bool)
    big_cup[:, :5] = True
    small_disc = np.zeros(shape, dtype=bool)
    small_disc[:, :2] = True

    disc, cup = fuse_annotations([(small_disc, big_cup)] * 3)

    assert not (cup & ~disc).any()


def test_masks_to_refuge_raw_uses_refuge2s_own_pixel_convention():
    # The cached RIGA mask must be byte-indistinguishable from a REFUGE2 mask, so
    # optic_disc_dataset._remap_mask_to_class_indices() reads it with no changes.
    disc = np.zeros((4, 4), dtype=bool)
    disc[1:3, 1:3] = True
    cup = np.zeros((4, 4), dtype=bool)
    cup[2, 2] = True

    raw = masks_to_refuge_raw(disc, cup)

    assert raw[0, 0] == _MASK_BACKGROUND_VALUE
    assert raw[1, 1] == _MASK_DISC_RIM_VALUE
    assert raw[2, 2] == _MASK_CUP_VALUE
