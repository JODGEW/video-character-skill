"""Decode SAM 3's run-length encoded masks into arrays.

``fal-ai/sam-3/video-rle-objects`` documents its ``rle`` field only as
"Run-length encoding (Kaggle/COCO order) of the mask", which names two
incompatible formats. Two real runs over ``driving_video_o1.mp4`` settled it:
the strings are **start/length pairs**, not COCO's alternating background /
foreground run lengths.

A real value begins::

    393654 12 394733 14 395800 34 396878 42 ...

Read as start/length pairs this is consistent: consecutive starts sit roughly
1080 apart — one source frame width — so each pair is one horizontal span of
one row, and decoding with 1-based starts in row-major order puts the mask
where the subject actually is in the source frame. Read as COCO alternating
runs it is impossible: the values sum to 1,065,761,049 for a frame of
1080x1920 = 2,073,600 pixels, ~514x too many.

The decoder is therefore deterministic and strict. Anything that does not
match that contract exactly — an odd token count, a zero or negative value, a
run past the end of the frame, runs that overlap or run backwards — raises
:class:`MaskDecodeError` rather than being clipped or repaired, because a mask
that is quietly wrong would corrupt a composite without ever failing.
"""

from __future__ import annotations

import re

import numpy as np
from numpy.typing import NDArray

from video_character_skill.schemas import ObjectMask, VideoMaskTrack

__all__ = [
    "MaskDecodeError",
    "decode_object_mask",
    "decode_rle",
    "mask_area_ratio",
    "mask_bbox",
]

BoolMask = NDArray[np.bool_]

# Unsigned decimal integers separated by whitespace, and nothing else: this
# rejects signs, decimal points and stray characters in one pass, so the
# integer conversion below cannot silently reinterpret anything.
_TOKENS = re.compile(r"\d+(?:\s+\d+)*")


class MaskDecodeError(ValueError):
    """An RLE string does not match SAM 3's documented start/length format."""


def decode_rle(rle: str, width: int, height: int) -> BoolMask:
    """Decode one SAM 3 RLE string into a ``(height, width)`` boolean mask.

    Args:
        rle: whitespace-separated ``start length`` pairs. ``start`` is a
            **1-based** index into the frame flattened in **row-major (C)
            order**; ``length`` is the number of pixels in the run.
        width: frame width in pixels, from :attr:`VideoMaskTrack.width`.
        height: frame height in pixels, from :attr:`VideoMaskTrack.height`.

    Returns:
        A boolean array, ``True`` inside the object. ``mask[y, x]`` is the
        pixel at column ``x`` of row ``y``.

    Raises:
        MaskDecodeError: on a blank string, a non-integer or signed token, an
            odd token count, a zero or negative start or length, a run that
            ends past ``width * height``, or runs that overlap or are not in
            ascending order.
        ValueError: if ``width`` or ``height`` is not positive.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"frame dimensions must be positive, got {width}x{height}")

    text = rle.strip()
    if not text:
        # SAM 3 signals "this object has no mask in this frame" by leaving it
        # out of MaskFrame.objects, so a blank RLE is a truncated payload, not
        # an empty mask. Callers who want the empty mask should skip decoding.
        raise MaskDecodeError("rle is blank; an absent mask is an absent object, not an empty rle")
    if _TOKENS.fullmatch(text) is None:
        raise MaskDecodeError(f"rle is not whitespace-separated unsigned integers: {text[:60]!r}")

    values = np.array(text.split(), dtype=np.int64)
    if values.size % 2:
        raise MaskDecodeError(
            f"rle has {values.size} tokens; start/length pairs need an even count"
        )

    starts = values[0::2]
    lengths = values[1::2]
    if not starts.size:
        raise MaskDecodeError("rle has no start/length pairs")
    if starts.min() < 1:
        raise MaskDecodeError("rle starts are 1-based; found a start below 1")
    if lengths.min() < 1:
        raise MaskDecodeError("rle run lengths must be at least 1; found a length below 1")

    pixels = width * height
    begins = starts - 1
    ends = begins + lengths  # exclusive
    if ends.max() > pixels:
        raise MaskDecodeError(
            f"rle run ends at pixel {int(ends.max())}, past the {width}x{height} "
            f"frame's {pixels} pixels"
        )
    if np.any(begins[1:] < ends[:-1]):
        raise MaskDecodeError("rle runs overlap or are not in ascending order")

    # Mark each run's edges, then integrate: O(pixels) with no Python loop.
    # Runs are disjoint and ascending by here, so `begins` and `ends` are each
    # free of duplicates — plain fancy indexing is safe — and the running sum
    # is only ever 0 or 1.
    edges = np.zeros(pixels + 1, dtype=np.int8)
    edges[begins] += 1
    edges[ends] -= 1
    flat = np.cumsum(edges[:pixels], dtype=np.int8).astype(np.bool_)
    return flat.reshape(height, width)


def decode_object_mask(mask: ObjectMask, track: VideoMaskTrack) -> BoolMask:
    """Decode one tracked object's mask against its track's frame dimensions.

    The RLE carries no dimensions of its own, so it is only meaningful next to
    the :class:`VideoMaskTrack` it came from.
    """
    return decode_rle(mask.rle, track.width, track.height)


def mask_bbox(mask: BoolMask) -> tuple[int, int, int, int] | None:
    """Return the mask's bounding box as half-open ``(x_min, y_min, x_max, y_max)``.

    Half-open so it slices directly: ``mask[y_min:y_max, x_min:x_max]``.
    Returns ``None`` when nothing is masked.
    """
    rows = np.flatnonzero(mask.any(axis=1))
    if not rows.size:
        return None
    columns = np.flatnonzero(mask.any(axis=0))
    return int(columns[0]), int(rows[0]), int(columns[-1]) + 1, int(rows[-1]) + 1


def mask_area_ratio(mask: BoolMask) -> float:
    """Fraction of the frame the mask covers, in ``[0.0, 1.0]``.

    A cheap sanity signal: a person filling a portrait clip lands in the low
    tenths, so a ratio near 0 or near 1 means the segmentation missed.
    """
    return float(np.count_nonzero(mask)) / float(mask.size)
