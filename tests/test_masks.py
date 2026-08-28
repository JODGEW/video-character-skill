"""Unit tests for the SAM 3 RLE decoder.

The format is start/length pairs, 1-based, row-major — see
``video_character_skill.masks`` for the evidence. These tests pin every part
of that contract, plus every way a string can fail it.
"""

from __future__ import annotations

import numpy as np
import pytest

from video_character_skill.masks import (
    MaskDecodeError,
    decode_object_mask,
    decode_rle,
    mask_area_ratio,
    mask_bbox,
)
from video_character_skill.schemas import MaskFrame, ObjectMask, VideoMaskTrack

# First four pairs of a real mask from driving_video_o1.mp4 (1080x1920).
REAL_PREFIX = "393654 12 394733 14 395800 34 396878 42"
SOURCE_WIDTH = 1080
SOURCE_HEIGHT = 1920


def track(width: int = 4, height: int = 3, rle: str = "2 3 6 2") -> VideoMaskTrack:
    return VideoMaskTrack(
        frames=(MaskFrame(frame_index=0, objects=(ObjectMask(track_id=0, rle=rle),)),),
        width=width,
        height=height,
        num_frames=1,
    )


# -- the contract -------------------------------------------------------


def test_known_rle_decodes_to_the_expected_mask() -> None:
    mask = decode_rle("2 3 6 2", 4, 3)

    assert mask.shape == (3, 4)
    assert mask.dtype == np.bool_
    np.testing.assert_array_equal(
        mask,
        np.array(
            [
                [False, True, True, True],
                [False, True, True, False],
                [False, False, False, False],
            ]
        ),
    )


def test_starts_are_one_based() -> None:
    """Start 1 is the top-left pixel; there is no start 0."""
    first = decode_rle("1 1", 4, 3)
    assert first[0, 0]
    assert np.count_nonzero(first) == 1

    second = decode_rle("2 1", 4, 3)
    assert not second[0, 0]
    assert second[0, 1]


def test_runs_flatten_row_major_and_wrap_onto_the_next_row() -> None:
    """A run longer than the remaining row continues at column 0 below it."""
    mask = decode_rle("3 4", 4, 3)

    np.testing.assert_array_equal(
        mask,
        np.array(
            [
                [False, False, True, True],
                [True, True, False, False],
                [False, False, False, False],
            ]
        ),
    )


def test_a_column_is_encoded_as_one_pair_per_row_a_width_apart() -> None:
    """Row-major means a vertical stripe's starts step by the frame width."""
    mask = decode_rle("2 1 6 1 10 1", 4, 3)

    np.testing.assert_array_equal(mask[:, 1], np.array([True, True, True]))
    assert np.count_nonzero(mask) == 3


def test_a_run_may_cover_the_whole_frame() -> None:
    assert decode_rle("1 12", 4, 3).all()


def test_real_prefix_lands_on_consecutive_rows_at_the_same_columns() -> None:
    """The starts sit ~1080 apart, so each pair is one row's span."""
    mask = decode_rle(REAL_PREFIX, SOURCE_WIDTH, SOURCE_HEIGHT)

    rows = np.flatnonzero(mask.any(axis=1))
    np.testing.assert_array_equal(rows, np.array([364, 365, 366, 367]))
    assert mask_bbox(mask) == (517, 364, 559, 368)
    assert np.count_nonzero(mask) == 12 + 14 + 34 + 42


# -- rejections ---------------------------------------------------------


def test_odd_token_count_is_rejected() -> None:
    with pytest.raises(MaskDecodeError, match="even count"):
        decode_rle("2 3 6", 4, 3)


@pytest.mark.parametrize("rle", ["0 3", "0 0"])
def test_zero_start_is_rejected(rle: str) -> None:
    with pytest.raises(MaskDecodeError, match="1-based"):
        decode_rle(rle, 4, 3)


def test_zero_length_is_rejected() -> None:
    with pytest.raises(MaskDecodeError, match="at least 1"):
        decode_rle("2 0", 4, 3)


@pytest.mark.parametrize("rle", ["-2 3", "2 -3", "+2 3"])
def test_signed_values_are_rejected(rle: str) -> None:
    with pytest.raises(MaskDecodeError, match="unsigned integers"):
        decode_rle(rle, 4, 3)


@pytest.mark.parametrize("rle", ["2.5 3", "two three", "2,3 4", "2 3;6 2"])
def test_non_integer_tokens_are_rejected(rle: str) -> None:
    with pytest.raises(MaskDecodeError, match="unsigned integers"):
        decode_rle(rle, 4, 3)


@pytest.mark.parametrize("rle", ["12 2", "13 1", "1 13"])
def test_runs_past_the_end_of_the_frame_are_rejected_not_clipped(rle: str) -> None:
    with pytest.raises(MaskDecodeError, match="past the 4x3 frame"):
        decode_rle(rle, 4, 3)


def test_the_last_pixel_is_still_in_bounds() -> None:
    mask = decode_rle("12 1", 4, 3)
    assert mask[2, 3]
    assert np.count_nonzero(mask) == 1


@pytest.mark.parametrize("rle", ["2 3 3 2", "2 3 4 1"])
def test_overlapping_runs_are_rejected(rle: str) -> None:
    with pytest.raises(MaskDecodeError, match="overlap"):
        decode_rle(rle, 4, 3)


def test_descending_runs_are_rejected() -> None:
    with pytest.raises(MaskDecodeError, match="ascending"):
        decode_rle("6 2 2 3", 4, 3)


def test_adjacent_runs_are_allowed() -> None:
    """A run ending at pixel n and the next starting at n+1 do not overlap."""
    np.testing.assert_array_equal(decode_rle("1 2 3 2", 4, 3), decode_rle("1 4", 4, 3))


@pytest.mark.parametrize("rle", ["", "   ", "\n\t "])
def test_blank_rle_is_rejected_rather_than_read_as_an_empty_mask(rle: str) -> None:
    """SAM 3 leaves an absent object out of ``objects``; it never sends "".

    fal documents ``MaskFrame.objects`` as "empty when none", so a blank RLE
    means a truncated payload, and returning an all-false mask would hide it.
    """
    with pytest.raises(MaskDecodeError, match="blank"):
        decode_rle(rle, 4, 3)


@pytest.mark.parametrize(("width", "height"), [(0, 3), (4, 0), (-4, 3)])
def test_non_positive_frame_dimensions_are_rejected(width: int, height: int) -> None:
    with pytest.raises(ValueError, match="dimensions must be positive"):
        decode_rle("1 1", width, height)


# -- helpers ------------------------------------------------------------


def test_decode_object_mask_uses_the_tracks_dimensions() -> None:
    subject = track()

    mask = decode_object_mask(subject.frames[0].objects[0], subject)

    np.testing.assert_array_equal(mask, decode_rle("2 3 6 2", 4, 3))


def test_decode_object_mask_propagates_a_bad_rle() -> None:
    subject = track(rle="99 1")

    with pytest.raises(MaskDecodeError):
        decode_object_mask(subject.frames[0].objects[0], subject)


def test_mask_bbox_is_half_open_and_slices_the_mask() -> None:
    mask = decode_rle("2 3 6 2", 4, 3)

    box = mask_bbox(mask)
    assert box == (1, 0, 4, 2)

    x_min, y_min, x_max, y_max = box
    assert mask[y_min:y_max, x_min:x_max].shape == (2, 3)


def test_mask_bbox_of_an_empty_mask_is_none() -> None:
    assert mask_bbox(np.zeros((3, 4), dtype=np.bool_)) is None


def test_mask_area_ratio_counts_the_masked_fraction() -> None:
    assert mask_area_ratio(decode_rle("2 3 6 2", 4, 3)) == pytest.approx(5 / 12)
    assert mask_area_ratio(decode_rle("1 12", 4, 3)) == 1.0
    assert mask_area_ratio(np.zeros((3, 4), dtype=np.bool_)) == 0.0
