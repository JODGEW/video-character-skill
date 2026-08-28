"""Composite a replacement subject back onto the original background.

The pipeline's last step. Kling O1 gives us a video in which the person has
been replaced but the room was regenerated; VEED gives us that same clip's
person as an alpha matte. This module puts the original background back::

    output = alpha * replacement + (1 - alpha) * source

The point of the whole masked architecture is that background pixels survive
untouched, so the blend is not applied uniformly. Pixels are partitioned by
their matte alpha:

``alpha == 0``
    Copied byte-for-byte from the *source* frame. No arithmetic runs on them
    at all — the output buffer starts life as a copy of the source, so a
    transparent pixel is preserved by construction rather than by a blend that
    happens to round back to the same value.

``alpha == 255``
    Copied byte-for-byte from the *replacement* frame.

``0 < alpha < 255``
    The only pixels that are computed, using integer arithmetic (see
    :func:`composite_frame`). On the real matte this is 1.5-2.35 % of each
    frame.

What that guarantee does and does not cover
-------------------------------------------
The invariant is **pre-encode**, and only pre-encode. In the RGB24 frame handed
to the encoder:

* every ``alpha == 0`` pixel holds the source frame's exact RGB24 bytes;
* every ``alpha == 255`` pixel holds the replacement frame's exact RGB24 bytes;
* only ``0 < alpha < 255`` pixels have been computed at all.

Nothing is claimed about the encoded file. This pipeline is
``RGB24 -> libx264 -> yuv420p``, and the RGB->YUV 4:2:0 conversion happens
*before* x264 ever runs: it rounds every pixel through a colour-space matrix
and throws away three quarters of the chroma resolution. Decoding back to RGB
cannot undo either step. So decoded background pixels of the output ``.mp4``
are **not** guaranteed to match the source byte-for-byte, and lowering the
quantiser does not change that. ``crf=0`` makes x264 mathematically lossless
*with respect to the yuv420p frames it was given* — which are already not the
composited RGB — so it does not buy bit-exact RGB either.

Verifying the invariant therefore means checking the composited frames, not the
encoded file: compare what :func:`composite_frame` returns against the source
frame, or compare raw RGB24 piped out of the compositor. A lossless delivery
path is a separate concern and is not implemented here.

Decoding VP9 alpha
------------------
FFmpeg's *native* ``vp9`` decoder silently drops alpha: ``ffprobe`` reports
``pix_fmt=yuv420p`` for ``out/o1_matte.webm`` even though the file carries
``ALPHA_MODE=1``. Only ``-c:v libvpx-vp9`` surfaces it, as ``yuva420p``. A
matte decoded the wrong way would come back fully opaque and the compositor
would emit the replacement clip unchanged — a silent, total loss of the
background. :data:`ALPHA_DECODERS` forces the right decoder, and validation
fails closed if the decoded pixel format has no alpha channel.

Dual-matte union (v2 POC)
-------------------------
The single-matte path keys every pixel off the *replacement* matte. That leaks
the original person wherever the old silhouette pokes out from under the new
one. A read-only overlap analysis of the real mattes (foreground =
``alpha >= 128``, 225 frames) measured that ``source_only`` region — source
foreground, replacement background — at a mean 0.372 % of the frame, peaking
at 0.954 %, with a mean source alpha of 219/255 and replacement alpha of
17/255 inside it: solid old-person interior, not edge noise. Foreground IoU
between the two mattes is 97.02 %.

:func:`composite_video_union` composites under the pixel-wise maximum of the
two mattes instead::

    effective_alpha = max(source_alpha, replacement_alpha)
    output = composite_frame(source_rgb, replacement_rgb, effective_alpha)

Wherever *either* matte sees a person, the replacement clip wins. The trade is
explicit: ``source_only`` pixels are now drawn from the O1 clip, so O1's
regenerated background can show through there instead of the old person. This
is a cheap POC to judge whether that region is small enough to live with, not
the final architecture; background recovery is the fallback if it is not. The
mattes are used exactly as decoded — no thresholding, dilation, feathering or
inpainting — and :func:`composite_frame` is reused unchanged, so the
``alpha == 0`` / ``alpha == 255`` byte-copy guarantee above holds for the
*effective* alpha.

Source-matte hardening (v3 POC)
-------------------------------
Visual QA of the union composite still showed the old person's hair ghosting
through. The union keeps the *source* matte's partial alpha inside
``source_only``, and that alpha is not near-opaque: a read-only sweep found a
mean residual source contribution of 14.0 % there, spread over a broad tail
(13 % of those pixels at alpha 128-159, 13 % at 160-191, only 20 % at exactly
255). Hardening only near-opaque pixels therefore buys little — thresholds of
208-240 leave 11-13 % residual.

:func:`composite_video_hardened_union` applies one extra rule on top of the
union, with :data:`SOURCE_HARDEN_THRESHOLD` = 160::

    effective = max(source_alpha, replacement_alpha)
    harden = (source_alpha >= 160) & (source_alpha > replacement_alpha)
    effective[harden] = 255

On the real mattes that hardens 87 % of ``source_only`` (residual 14.0 % ->
5.7 %), changes 0.33 % of the frame on average (max 0.89 %), and overrides
about a fifth of the replacement matte's soft edge — nearly all of it where
the source matte is >= 240 anyway. What survives is the source matte's own
1-2 px anti-aliased rim (87 % of the remaining pixels lie within 2 px of the
source silhouette): a thin outline rather than a patch. The replacement matte
is never thresholded, the source matte is hardened only where it is *more
confident than the replacement*, and there is still no dilation, feathering
or inpainting; ``source_only`` pixels still come from the O1 clip.

Source-removal mask (v4 POC)
----------------------------
Visual QA of the hardened union still showed the old person's hair around
the replacement. A read-only morphology sweep explained why: below 128 the
source matte's alpha is not a blending weight worth honouring. 69 % of those
pixels sit at alpha <= 7 and 10-30 px from the silhouette (VP9 alpha
compression haze), while the hair you can see (alpha >= 32) lies within 4 px
(p90) of the ``alpha >= 64`` core — and blending with *any* partial source
alpha keeps most of a strand, because the source pixel *is* the strand.

v4 therefore stops treating the source matte as an alpha at all. It is a
binary support — "the old person must not survive here" — built from
:data:`SOURCE_REMOVAL_THRESHOLD` = 64 and a Euclidean-disk dilation of
:data:`SOURCE_REMOVAL_DILATION_RADIUS` = 4 px::

    source_core = source_alpha >= 64
    removal = dilate_disk(source_core, 4)        # offsets with dy*dy + dx*dx <= 16
    effective = replacement_alpha.copy()
    effective[removal] = 255

Outside the mask the replacement matte behaves exactly as in v1. Inside it
the source clip contributes nothing: where the replacement person is, the O1
person is copied; where only the old person was, O1's background is copied
instead of the old person leaking through. Measured on the real mattes the
mask covers ~96 % of visible source hair, costs 0.208 % of the frame per
frame of true background (the dilation band; 0.694 % removal in total) and
shows no flicker signature. Radius 2 left too much hair; radius 6 replaced
markedly more background for little gain. No ``max()`` and no partial source
alpha anywhere in v4, and still no feathering, temporal smoothing or
inpainting. The dilation is plain NumPy — a shifted OR per integer offset
inside the disk, clipped at the border — so no scipy or OpenCV.
"""

from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import IO

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "ALPHA_DECODERS",
    "ALPHA_PIX_FMTS",
    "SOURCE_HARDEN_THRESHOLD",
    "SOURCE_REMOVAL_DILATION_RADIUS",
    "SOURCE_REMOVAL_THRESHOLD",
    "CompositeError",
    "CompositeReport",
    "HardenedUnionCompositeReport",
    "HardenedUnionStreamStats",
    "SourceRemovalCompositeReport",
    "SourceRemovalStreamStats",
    "UnionCompositeReport",
    "UnionStreamStats",
    "VideoInfo",
    "composite_frame",
    "composite_streams",
    "composite_streams_hardened_union",
    "composite_streams_source_removal",
    "composite_streams_union",
    "composite_video",
    "composite_video_hardened_union",
    "composite_video_source_removal",
    "composite_video_union",
    "dilate_disk",
    "hardened_union_alpha",
    "probe_video",
    "removal_effective_alpha",
    "soft_edge_ratio",
    "source_removal_mask",
    "union_alpha",
]

RgbFrame = NDArray[np.uint8]
AlphaPlane = NDArray[np.uint8]

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"

# Codecs whose alpha channel only appears under a specific decoder. FFmpeg's
# built-in vp8/vp9 decoders report a 3-plane format and drop the alpha.
ALPHA_DECODERS = {"vp9": "libvpx-vp9", "vp8": "libvpx-vp8"}

# Decoded pixel formats that carry an alpha channel. Anything not listed is
# rejected rather than assumed opaque or assumed transparent; extend the set
# deliberately if a new matte format shows up.
ALPHA_PIX_FMTS = frozenset(
    {
        "yuva420p", "yuva422p", "yuva444p",
        "yuva420p9le", "yuva422p9le", "yuva444p9le",
        "yuva420p10le", "yuva422p10le", "yuva444p10le",
        "yuva420p16le", "yuva422p16le", "yuva444p16le",
        "rgba", "bgra", "argb", "abgr", "rgba64le", "bgra64le",
        "gbrap", "gbrap10le", "gbrap12le", "gbrap16le",
        "ya8", "ya16le",
    }
)

# Source-matte hardening threshold for the v3 composite. Chosen from a
# read-only sweep over {160, 192, 208, 224, 240}: the only candidate that
# materially removed the old person's ghosting. See the module docstring.
SOURCE_HARDEN_THRESHOLD = 160

# Source-removal mask for the v4 composite: the source matte becomes a binary
# support (alpha >= threshold, dilated by a Euclidean disk of this radius)
# rather than a blending weight. Chosen from a read-only morphology sweep over
# thresholds {64, 96, 128} x radii {0, 2, 4, 6, 8}. See the module docstring.
SOURCE_REMOVAL_THRESHOLD = 64
SOURCE_REMOVAL_DILATION_RADIUS = 4


class CompositeError(RuntimeError):
    """A composite could not be produced from the given inputs."""


@dataclass(frozen=True)
class VideoInfo:
    """What ``ffprobe`` reports about one video stream."""

    path: Path
    codec_name: str
    width: int
    height: int
    pix_fmt: str
    frame_rate: Fraction
    frame_count: int

    @property
    def has_alpha(self) -> bool:
        return self.pix_fmt in ALPHA_PIX_FMTS

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height

    @property
    def rate_argument(self) -> str:
        """The frame rate as ffmpeg wants it, exactly — no float rounding."""
        return f"{self.frame_rate.numerator}/{self.frame_rate.denominator}"


@dataclass(frozen=True)
class CompositeReport:
    """What one composite run produced."""

    output_path: Path
    frames: int
    width: int
    height: int
    frame_rate: Fraction
    soft_edge_ratio: float
    """Mean fraction of pixels per frame that were blended rather than copied."""


@dataclass(frozen=True)
class UnionStreamStats:
    """Per-clip means gathered while streaming a dual-matte composite."""

    soft_edge_ratio: float
    """Mean fraction of pixels per frame blended under the *effective* alpha."""
    union_lift_ratio: float
    """Mean fraction of pixels per frame where the source matte's alpha exceeded
    the replacement matte's — where the union changed what the single-matte
    path would have done. Those pixels now come from the replacement clip; on
    the real mattes this is the ``source_only`` region plus part of the soft
    edge."""


@dataclass(frozen=True)
class UnionCompositeReport:
    """What one dual-matte composite run produced."""

    output_path: Path
    frames: int
    width: int
    height: int
    frame_rate: Fraction
    soft_edge_ratio: float
    """See :attr:`UnionStreamStats.soft_edge_ratio`."""
    union_lift_ratio: float
    """See :attr:`UnionStreamStats.union_lift_ratio`."""


@dataclass(frozen=True)
class HardenedUnionStreamStats:
    """Per-clip means gathered while streaming a hardened dual-matte composite."""

    soft_edge_ratio: float
    """Mean fraction of pixels per frame blended under the *effective* alpha."""
    union_lift_ratio: float
    """As :attr:`UnionStreamStats.union_lift_ratio`: source alpha above the
    replacement's, before hardening."""
    hardened_ratio: float
    """Mean fraction of full-frame pixels per frame that the hardening rule
    actually changed to 255 — pixels whose plain-union alpha was below 255 and
    met the rule. Pixels already at 255 are not counted."""


@dataclass(frozen=True)
class HardenedUnionCompositeReport:
    """What one hardened dual-matte composite run produced."""

    output_path: Path
    frames: int
    width: int
    height: int
    frame_rate: Fraction
    source_threshold: int
    """The hardening threshold that was applied."""
    soft_edge_ratio: float
    union_lift_ratio: float
    hardened_ratio: float
    """See :class:`HardenedUnionStreamStats` for the three ratios."""


@dataclass(frozen=True)
class SourceRemovalStreamStats:
    """Per-clip means gathered while streaming a source-removal composite."""

    soft_edge_ratio: float
    """Mean fraction of pixels per frame blended under the *effective* alpha —
    the replacement matte's own soft edge outside the removal mask."""
    removal_ratio: float
    """Mean fraction of full-frame pixels per frame inside the removal mask."""
    dilation_only_ratio: float
    """Mean fraction of full-frame pixels per frame that the dilation added
    beyond the ``source_alpha >= threshold`` core."""
    replacement_override_ratio: float
    """Mean fraction of full-frame pixels per frame where the mask forced a
    replacement alpha below 255 up to 255."""


@dataclass(frozen=True)
class SourceRemovalCompositeReport:
    """What one source-removal composite run produced."""

    output_path: Path
    frames: int
    width: int
    height: int
    frame_rate: Fraction
    threshold: int
    dilation_radius: int
    """The removal-mask parameters that were applied."""
    soft_edge_ratio: float
    removal_ratio: float
    dilation_only_ratio: float
    replacement_override_ratio: float
    """See :class:`SourceRemovalStreamStats` for the four ratios."""


# -- the pure part -------------------------------------------------------


def composite_frame(
    source: RgbFrame, replacement: RgbFrame, alpha: AlphaPlane
) -> RgbFrame:
    """Blend one frame. Pure: no I/O, no globals, no ffmpeg.

    Args:
        source: ``(H, W, 3)`` uint8 RGB, the original background.
        replacement: ``(H, W, 3)`` uint8 RGB, the edited clip.
        alpha: ``(H, W)`` uint8 matte, 0 = keep source, 255 = take replacement.

    Returns:
        A new ``(H, W, 3)`` uint8 RGB frame.

    Raises:
        CompositeError: on mismatched shapes or a non-uint8 input.

    Blending is integer-only, on the partial pixels alone::

        out = (alpha * replacement + (255 - alpha) * source + 127) // 255

    Integer rather than float so the result is exact and identical on every
    platform, with no rounding mode to agree on. The formula already returns
    ``source`` at alpha 0 and ``replacement`` at alpha 255, but those pixels
    are copied instead of computed: it makes the preservation guarantee
    structural rather than a property of the arithmetic, and it keeps the
    computed set down to the soft edge.
    """
    _check_frames(source, replacement, alpha)

    # Start from the source. Every alpha == 0 pixel is now already correct and
    # will never be written again.
    out: RgbFrame = source.copy()

    opaque = alpha == 255
    if opaque.any():
        out[opaque] = replacement[opaque]

    soft = (alpha != 0) & ~opaque
    if soft.any():
        a = alpha[soft].astype(np.uint32)[:, None]
        rep = replacement[soft].astype(np.uint32)
        src = source[soft].astype(np.uint32)
        out[soft] = ((a * rep + (255 - a) * src + 127) // 255).astype(np.uint8)

    return out


def soft_edge_ratio(alpha: AlphaPlane) -> float:
    """Fraction of the matte that is partially transparent, in ``[0, 1]``.

    The pixels :func:`composite_frame` actually computes. A ratio of 0 means
    the matte is hard-edged; a ratio near 1 means it is mush.
    """
    return float(np.count_nonzero((alpha != 0) & (alpha != 255))) / float(alpha.size)


def union_alpha(source_alpha: AlphaPlane, replacement_alpha: AlphaPlane) -> AlphaPlane:
    """Pixel-wise maximum of two mattes. Pure: no I/O, no globals.

    Args:
        source_alpha: ``(H, W)`` uint8 matte of the person in the *source* clip.
        replacement_alpha: ``(H, W)`` uint8 matte of the person in the
            *replacement* clip.

    Returns:
        A new ``(H, W)`` uint8 plane, exactly
        ``np.maximum(source_alpha, replacement_alpha)``.

    Raises:
        CompositeError: on mismatched shapes, a non-2-D plane, or a non-uint8
            input.

    Neither input is thresholded, dilated, feathered or otherwise touched: a
    pixel's effective alpha is whichever matte is more confident that there is
    a person there. ``max(0, 0) == 0`` keeps the background copied from the
    source; ``max(255, x) == 255`` copies the replacement wherever *either*
    matte is fully opaque.
    """
    _check_alphas(source_alpha, replacement_alpha)
    out: AlphaPlane = np.maximum(source_alpha, replacement_alpha)
    return out


def hardened_union_alpha(
    source_alpha: AlphaPlane,
    replacement_alpha: AlphaPlane,
    *,
    source_threshold: int = SOURCE_HARDEN_THRESHOLD,
) -> AlphaPlane:
    """Union of two mattes, with confident source pixels forced opaque. Pure.

    Args:
        source_alpha: ``(H, W)`` uint8 matte of the person in the *source* clip.
        replacement_alpha: ``(H, W)`` uint8 matte of the person in the
            *replacement* clip.
        source_threshold: source alpha at or above which a pixel is hardened,
            ``1..255``. Defaults to :data:`SOURCE_HARDEN_THRESHOLD`.

    Returns:
        A new ``(H, W)`` uint8 plane. Exactly::

            effective = np.maximum(source_alpha, replacement_alpha)
            harden = (source_alpha >= source_threshold) & (source_alpha > replacement_alpha)
            effective[harden] = 255

    Raises:
        CompositeError: on mismatched shapes, a non-2-D plane, a non-uint8
            input, or a threshold outside ``1..255``.

    The rule never lowers a pixel below the plain union, and it never fires
    where the replacement matte is at least as confident as the source — so
    the replacement's own soft edge stands wherever it is the stronger
    opinion. The replacement matte is not thresholded, and the source matte is
    not thresholded globally: a source pixel below the threshold keeps its
    partial value through the union exactly as before.
    """
    _check_alphas(source_alpha, replacement_alpha)
    _check_threshold(source_threshold)
    out: AlphaPlane = np.maximum(source_alpha, replacement_alpha)
    out[_harden_mask(source_alpha, replacement_alpha, source_threshold)] = 255
    return out


def _harden_mask(
    source_alpha: AlphaPlane, replacement_alpha: AlphaPlane, source_threshold: int
) -> NDArray[np.bool_]:
    mask: NDArray[np.bool_] = (source_alpha >= source_threshold) & (
        source_alpha > replacement_alpha
    )
    return mask


def dilate_disk(mask: NDArray[np.bool_], radius: int) -> NDArray[np.bool_]:
    """Binary dilation by a Euclidean disk of integer ``radius``. Pure.

    A pixel is set in the result if any set pixel of ``mask`` lies at an
    integer offset ``(dy, dx)`` with ``dy*dy + dx*dx <= radius*radius`` — the
    same disk the morphology analysis used. Radius 4 is therefore a 49-pixel
    disk, not a 9x9 square: the square's corners are up to 5.7 px away and
    are excluded. Implemented as one shifted OR per offset using slices, so
    nothing wraps around the image border. ``radius == 0`` returns a copy.

    Args:
        mask: ``(H, W)`` bool.
        radius: non-negative int.

    Raises:
        CompositeError: on a non-2-D or non-bool mask, or a negative or
            non-int radius.
    """
    _check_mask(mask)
    _check_radius(radius)
    out: NDArray[np.bool_] = mask.copy()
    height, width = mask.shape
    for dy, dx in _disk_offsets(radius):
        if (dy == 0 and dx == 0) or abs(dy) >= height or abs(dx) >= width:
            continue
        rows_to = slice(max(dy, 0), height + min(dy, 0))
        rows_from = slice(max(-dy, 0), height + min(-dy, 0))
        cols_to = slice(max(dx, 0), width + min(dx, 0))
        cols_from = slice(max(-dx, 0), width + min(-dx, 0))
        out[rows_to, cols_to] |= mask[rows_from, cols_from]
    return out


def _disk_offsets(radius: int) -> list[tuple[int, int]]:
    """Integer ``(dy, dx)`` offsets inside a Euclidean disk, in a fixed order."""
    limit = radius * radius
    return [
        (dy, dx)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
        if dy * dy + dx * dx <= limit
    ]


def source_removal_mask(
    source_alpha: AlphaPlane,
    *,
    threshold: int = SOURCE_REMOVAL_THRESHOLD,
    dilation_radius: int = SOURCE_REMOVAL_DILATION_RADIUS,
) -> NDArray[np.bool_]:
    """Where the old person must not survive: a binary support, not an alpha.

    Args:
        source_alpha: ``(H, W)`` uint8 matte of the person in the *source* clip.
        threshold: source alpha at or above which a pixel is in the core,
            ``1..255``. Defaults to :data:`SOURCE_REMOVAL_THRESHOLD`.
        dilation_radius: Euclidean-disk radius in pixels, ``>= 0``. Defaults
            to :data:`SOURCE_REMOVAL_DILATION_RADIUS`.

    Returns:
        ``(H, W)`` bool. Exactly::

            core = source_alpha >= threshold
            removal = dilate_disk(core, dilation_radius)

    Raises:
        CompositeError: on a non-2-D or non-uint8 plane, a threshold outside
            ``1..255``, or a negative or non-int radius.

    The alpha *value* matters only through the threshold: 63 is out, 64 is
    in, and 64 and 255 are treated identically. The dilation deliberately
    reaches pixels whose source alpha is 0 — that is where the matte's own
    fine hair sits.
    """
    _check_alpha_plane("source", source_alpha)
    _check_int_range("threshold", threshold, 1, 255)
    _check_radius(dilation_radius)
    return dilate_disk(_source_core(source_alpha, threshold), dilation_radius)


def removal_effective_alpha(
    source_alpha: AlphaPlane,
    replacement_alpha: AlphaPlane,
    *,
    threshold: int = SOURCE_REMOVAL_THRESHOLD,
    dilation_radius: int = SOURCE_REMOVAL_DILATION_RADIUS,
) -> AlphaPlane:
    """The replacement matte, forced opaque inside the source-removal mask. Pure.

    Args:
        source_alpha: ``(H, W)`` uint8 matte of the person in the *source* clip.
        replacement_alpha: ``(H, W)`` uint8 matte of the person in the
            *replacement* clip.
        threshold: see :func:`source_removal_mask`.
        dilation_radius: see :func:`source_removal_mask`.

    Returns:
        A new ``(H, W)`` uint8 plane. Exactly::

            removal = source_removal_mask(source_alpha, threshold=..., dilation_radius=...)
            effective = replacement_alpha.copy()
            effective[removal] = 255

    Raises:
        CompositeError: as :func:`source_removal_mask`, or on mismatched
            shapes or a non-uint8 replacement plane.

    No ``max()`` and no partial source alpha: outside the mask the effective
    alpha *is* the replacement matte, byte for byte; inside it the source clip
    contributes exactly nothing to :func:`composite_frame`.
    """
    _check_alphas(source_alpha, replacement_alpha)
    removal = source_removal_mask(
        source_alpha, threshold=threshold, dilation_radius=dilation_radius
    )
    return _apply_removal(replacement_alpha, removal)


def _source_core(source_alpha: AlphaPlane, threshold: int) -> NDArray[np.bool_]:
    core: NDArray[np.bool_] = source_alpha >= threshold
    return core


def _apply_removal(replacement_alpha: AlphaPlane, removal: NDArray[np.bool_]) -> AlphaPlane:
    effective: AlphaPlane = replacement_alpha.copy()
    effective[removal] = 255
    return effective


def _check_mask(mask: NDArray[np.bool_]) -> None:
    if mask.ndim != 2:
        raise CompositeError(f"mask must be (H, W), got {mask.shape}")
    if mask.dtype != np.bool_:
        raise CompositeError(f"mask must be bool, got {mask.dtype}")


def _check_radius(radius: int) -> None:
    _check_int_range("dilation_radius", radius, 0, None)


def _check_alpha_plane(name: str, alpha: AlphaPlane) -> None:
    if alpha.ndim != 2:
        raise CompositeError(f"{name} alpha must be (H, W), got {alpha.shape}")
    if alpha.dtype != np.uint8:
        raise CompositeError(f"{name} alpha must be uint8, got {alpha.dtype}")


def _check_threshold(source_threshold: int) -> None:
    _check_int_range("source_threshold", source_threshold, 1, 255)


def _check_int_range(name: str, value: int, low: int, high: int | None) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CompositeError(f"{name} must be an int, got {value!r}")
    if value < low or (high is not None and value > high):
        bounds = f"in {low}..{high}" if high is not None else f">= {low}"
        raise CompositeError(f"{name} must be {bounds}, got {value}")


def _check_alphas(source_alpha: AlphaPlane, replacement_alpha: AlphaPlane) -> None:
    if source_alpha.ndim != 2:
        raise CompositeError(f"source alpha must be (H, W), got {source_alpha.shape}")
    if replacement_alpha.shape != source_alpha.shape:
        raise CompositeError(
            f"replacement alpha shape {replacement_alpha.shape} != "
            f"source alpha shape {source_alpha.shape}"
        )
    for name, array in (("source", source_alpha), ("replacement", replacement_alpha)):
        if array.dtype != np.uint8:
            raise CompositeError(f"{name} alpha must be uint8, got {array.dtype}")


def _check_frames(
    source: RgbFrame, replacement: RgbFrame, alpha: AlphaPlane
) -> None:
    if source.ndim != 3 or source.shape[2] != 3:
        raise CompositeError(f"source must be (H, W, 3), got {source.shape}")
    if replacement.shape != source.shape:
        raise CompositeError(
            f"replacement shape {replacement.shape} != source shape {source.shape}"
        )
    if alpha.shape != source.shape[:2]:
        raise CompositeError(
            f"alpha shape {alpha.shape} != source frame shape {source.shape[:2]}"
        )
    for name, array in (("source", source), ("replacement", replacement), ("alpha", alpha)):
        if array.dtype != np.uint8:
            raise CompositeError(f"{name} must be uint8, got {array.dtype}")


# -- the streaming part (file-like in, file-like out) --------------------


def composite_streams(
    source: IO[bytes],
    replacement: IO[bytes],
    matte: IO[bytes],
    output: IO[bytes],
    *,
    width: int,
    height: int,
    frames: int,
) -> float:
    """Composite ``frames`` frames of raw video, one at a time.

    Reads RGB24 from ``source`` and ``replacement`` and RGBA from ``matte``,
    writes RGB24 to ``output``. Only three frames are ever in memory.

    Returns:
        The mean soft-edge ratio across the clip.

    Raises:
        CompositeError: if any stream ends before ``frames`` frames, or if any
            still has data afterwards.
    """
    soft_total = 0.0

    for index in range(frames):
        source_frame = _read_rgb_frame(source, "source", index, width, height)
        replacement_frame = _read_rgb_frame(replacement, "replacement", index, width, height)
        alpha = _read_alpha_frame(matte, "matte", index, width, height)

        soft_total += soft_edge_ratio(alpha)
        output.write(composite_frame(source_frame, replacement_frame, alpha).tobytes())

    _assert_drained({"source": source, "replacement": replacement, "matte": matte}, frames)

    return soft_total / frames if frames else 0.0


def composite_streams_union(
    source: IO[bytes],
    replacement: IO[bytes],
    source_matte: IO[bytes],
    replacement_matte: IO[bytes],
    output: IO[bytes],
    *,
    width: int,
    height: int,
    frames: int,
) -> UnionStreamStats:
    """Composite ``frames`` frames under the union of two mattes, one at a time.

    Reads RGB24 from ``source`` and ``replacement`` and RGBA from both mattes,
    writes RGB24 to ``output``. Per frame::

        effective_alpha = union_alpha(source_alpha, replacement_alpha)
        output_frame = composite_frame(source_rgb, replacement_rgb, effective_alpha)

    Only four input frames and one output frame are ever in memory.

    Returns:
        :class:`UnionStreamStats` with the clip's mean ratios.

    Raises:
        CompositeError: if any of the four streams ends before ``frames``
            frames, or if any still has data afterwards.
    """
    soft_total = 0.0
    lift_total = 0.0

    for index in range(frames):
        source_frame = _read_rgb_frame(source, "source", index, width, height)
        replacement_frame = _read_rgb_frame(replacement, "replacement", index, width, height)
        source_alpha = _read_alpha_frame(source_matte, "source_matte", index, width, height)
        replacement_alpha = _read_alpha_frame(
            replacement_matte, "replacement_matte", index, width, height
        )

        alpha = union_alpha(source_alpha, replacement_alpha)
        soft_total += soft_edge_ratio(alpha)
        lifted = np.count_nonzero(source_alpha > replacement_alpha)
        lift_total += float(lifted) / float(alpha.size)
        output.write(composite_frame(source_frame, replacement_frame, alpha).tobytes())

    _assert_drained(
        {
            "source": source,
            "replacement": replacement,
            "source_matte": source_matte,
            "replacement_matte": replacement_matte,
        },
        frames,
    )

    if not frames:
        return UnionStreamStats(soft_edge_ratio=0.0, union_lift_ratio=0.0)
    return UnionStreamStats(
        soft_edge_ratio=soft_total / frames, union_lift_ratio=lift_total / frames
    )


def composite_streams_hardened_union(
    source: IO[bytes],
    replacement: IO[bytes],
    source_matte: IO[bytes],
    replacement_matte: IO[bytes],
    output: IO[bytes],
    *,
    width: int,
    height: int,
    frames: int,
    source_threshold: int = SOURCE_HARDEN_THRESHOLD,
) -> HardenedUnionStreamStats:
    """Composite ``frames`` frames under the hardened union, one at a time.

    As :func:`composite_streams_union`, but per frame::

        effective_alpha = hardened_union_alpha(
            source_alpha, replacement_alpha, source_threshold=source_threshold
        )
        output_frame = composite_frame(source_rgb, replacement_rgb, effective_alpha)

    Returns:
        :class:`HardenedUnionStreamStats` with the clip's mean ratios.

    Raises:
        CompositeError: if the threshold is out of range (before anything is
            read), if any of the four streams ends before ``frames`` frames,
            or if any still has data afterwards.
    """
    _check_threshold(source_threshold)
    soft_total = 0.0
    lift_total = 0.0
    hardened_total = 0.0

    for index in range(frames):
        source_frame = _read_rgb_frame(source, "source", index, width, height)
        replacement_frame = _read_rgb_frame(replacement, "replacement", index, width, height)
        source_alpha = _read_alpha_frame(source_matte, "source_matte", index, width, height)
        replacement_alpha = _read_alpha_frame(
            replacement_matte, "replacement_matte", index, width, height
        )

        union = union_alpha(source_alpha, replacement_alpha)
        alpha = hardened_union_alpha(
            source_alpha, replacement_alpha, source_threshold=source_threshold
        )
        size = float(alpha.size)
        soft_total += soft_edge_ratio(alpha)
        lift_total += float(np.count_nonzero(source_alpha > replacement_alpha)) / size
        hardened_total += float(np.count_nonzero(alpha != union)) / size
        output.write(composite_frame(source_frame, replacement_frame, alpha).tobytes())

    _assert_drained(
        {
            "source": source,
            "replacement": replacement,
            "source_matte": source_matte,
            "replacement_matte": replacement_matte,
        },
        frames,
    )

    if not frames:
        return HardenedUnionStreamStats(0.0, 0.0, 0.0)
    return HardenedUnionStreamStats(
        soft_edge_ratio=soft_total / frames,
        union_lift_ratio=lift_total / frames,
        hardened_ratio=hardened_total / frames,
    )


def composite_streams_source_removal(
    source: IO[bytes],
    replacement: IO[bytes],
    source_matte: IO[bytes],
    replacement_matte: IO[bytes],
    output: IO[bytes],
    *,
    width: int,
    height: int,
    frames: int,
    threshold: int = SOURCE_REMOVAL_THRESHOLD,
    dilation_radius: int = SOURCE_REMOVAL_DILATION_RADIUS,
) -> SourceRemovalStreamStats:
    """Composite ``frames`` frames under the source-removal mask, one at a time.

    As :func:`composite_streams_union`, but per frame::

        removal = source_removal_mask(source_alpha, threshold=..., dilation_radius=...)
        effective_alpha = replacement_alpha.copy()
        effective_alpha[removal] = 255
        output_frame = composite_frame(source_rgb, replacement_rgb, effective_alpha)

    Returns:
        :class:`SourceRemovalStreamStats` with the clip's mean ratios.

    Raises:
        CompositeError: if the threshold or radius is invalid (before anything
            is read), if any of the four streams ends before ``frames``
            frames, or if any still has data afterwards.
    """
    _check_int_range("threshold", threshold, 1, 255)
    _check_radius(dilation_radius)
    soft_total = 0.0
    removal_total = 0.0
    dilation_total = 0.0
    override_total = 0.0

    for index in range(frames):
        source_frame = _read_rgb_frame(source, "source", index, width, height)
        replacement_frame = _read_rgb_frame(replacement, "replacement", index, width, height)
        source_alpha = _read_alpha_frame(source_matte, "source_matte", index, width, height)
        replacement_alpha = _read_alpha_frame(
            replacement_matte, "replacement_matte", index, width, height
        )

        removal = source_removal_mask(
            source_alpha, threshold=threshold, dilation_radius=dilation_radius
        )
        alpha = _apply_removal(replacement_alpha, removal)
        size = float(alpha.size)
        soft_total += soft_edge_ratio(alpha)
        removal_total += float(np.count_nonzero(removal)) / size
        core = _source_core(source_alpha, threshold)
        dilation_total += float(np.count_nonzero(removal & ~core)) / size
        overridden = removal & (replacement_alpha != 255)
        override_total += float(np.count_nonzero(overridden)) / size
        output.write(composite_frame(source_frame, replacement_frame, alpha).tobytes())

    _assert_drained(
        {
            "source": source,
            "replacement": replacement,
            "source_matte": source_matte,
            "replacement_matte": replacement_matte,
        },
        frames,
    )

    if not frames:
        return SourceRemovalStreamStats(0.0, 0.0, 0.0, 0.0)
    return SourceRemovalStreamStats(
        soft_edge_ratio=soft_total / frames,
        removal_ratio=removal_total / frames,
        dilation_only_ratio=dilation_total / frames,
        replacement_override_ratio=override_total / frames,
    )


def _read_rgb_frame(
    stream: IO[bytes], name: str, index: int, width: int, height: int
) -> RgbFrame:
    raw = _read_exact(stream, width * height * 3, name, index)
    frame: RgbFrame = np.frombuffer(raw, np.uint8).reshape(height, width, 3)
    return frame


def _read_alpha_frame(
    stream: IO[bytes], name: str, index: int, width: int, height: int
) -> AlphaPlane:
    """Read one RGBA frame and keep only its alpha plane."""
    raw = _read_exact(stream, width * height * 4, name, index)
    alpha: AlphaPlane = np.frombuffer(raw, np.uint8).reshape(height, width, 4)[:, :, 3]
    return alpha


def _assert_drained(streams: Mapping[str, IO[bytes]], frames: int) -> None:
    """Every input must be exhausted once ``frames`` frames have been read."""
    for name, stream in streams.items():
        if stream.read(1):
            raise CompositeError(f"{name} has more than the expected {frames} frames")


def _read_exact(stream: IO[bytes], size: int, name: str, index: int) -> bytes:
    """Read exactly ``size`` bytes; a short read means the stream ran out."""
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            got = size - remaining
            raise CompositeError(
                f"{name} ended during frame {index}: wanted {size} bytes, got {got}"
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


# -- probing -------------------------------------------------------------


def probe_video(path: Path, decoder: str | None = None) -> VideoInfo:
    """Describe a video's first video stream.

    Args:
        path: the file to probe.
        decoder: force this decoder, which changes the reported ``pix_fmt``
            for formats whose alpha only one decoder exposes.

    Raises:
        CompositeError: if the file is missing, ffprobe fails, or the stream
            is missing a field we need.
    """
    if not path.is_file():
        raise CompositeError(f"input not found: {path}")
    command = [FFPROBE, "-v", "error"]
    if decoder is not None:
        command += ["-c:v", decoder]
    command += [
        "-select_streams", "v:0",
        "-count_frames",
        "-show_entries", "stream=codec_name,width,height,pix_fmt,r_frame_rate,nb_read_frames",
        "-of", "default=noprint_wrappers=1:nokey=0",
        str(path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise CompositeError(
            f"ffprobe failed for {path}: {completed.stderr.strip() or completed.returncode}"
        )
    fields = dict(
        line.split("=", 1) for line in completed.stdout.splitlines() if "=" in line
    )
    try:
        return VideoInfo(
            path=path,
            codec_name=fields["codec_name"],
            width=int(fields["width"]),
            height=int(fields["height"]),
            pix_fmt=fields["pix_fmt"],
            frame_rate=Fraction(fields["r_frame_rate"]),
            frame_count=int(fields["nb_read_frames"]),
        )
    except (KeyError, ValueError, ZeroDivisionError) as exc:
        raise CompositeError(f"ffprobe gave no usable stream for {path}: {fields}") from exc


def _probe_matte(path: Path) -> VideoInfo:
    """Probe a matte through whichever decoder actually exposes its alpha."""
    info = probe_video(path)
    decoder = ALPHA_DECODERS.get(info.codec_name)
    return probe_video(path, decoder=decoder) if decoder else info


def _validate(source: VideoInfo, replacement: VideoInfo, matte: VideoInfo) -> None:
    """Fail closed on anything that would make same-index compositing wrong."""
    if not source.size == replacement.size == matte.size:
        raise CompositeError(
            "inputs differ in size: "
            f"source {source.width}x{source.height}, "
            f"replacement {replacement.width}x{replacement.height}, "
            f"matte {matte.width}x{matte.height}"
        )
    if source.frame_rate != replacement.frame_rate:
        raise CompositeError(
            f"source is {source.frame_rate} fps but replacement is "
            f"{replacement.frame_rate} fps"
        )
    counts = {
        "source": source.frame_count,
        "replacement": replacement.frame_count,
        "matte": matte.frame_count,
    }
    if len(set(counts.values())) != 1:
        raise CompositeError(f"inputs differ in frame count: {counts}")
    if source.frame_count == 0:
        raise CompositeError(f"{source.path} has no frames")
    if not matte.has_alpha:
        raise CompositeError(
            f"matte {matte.path} decodes to {matte.pix_fmt}, which has no alpha "
            f"channel (decoder: {ALPHA_DECODERS.get(matte.codec_name, 'default')})"
        )


def _validate_union(
    source: VideoInfo,
    replacement: VideoInfo,
    source_matte: VideoInfo,
    replacement_matte: VideoInfo,
) -> None:
    """Fail closed unless all four streams agree and both mattes carry alpha.

    Each matte is held to exactly the single-matte rules against the same
    source/replacement pair, so every stream must share one size and one frame
    count, the two clips one frame rate, and each matte must have decoded with
    an alpha channel. The offending matte is named in the error.
    """
    mattes = (("source matte", source_matte), ("replacement matte", replacement_matte))
    for name, matte in mattes:
        try:
            _validate(source, replacement, matte)
        except CompositeError as exc:
            raise CompositeError(f"{name}: {exc}") from exc


# -- orchestration -------------------------------------------------------


def composite_video(
    source_path: Path,
    replacement_path: Path,
    matte_path: Path,
    output_path: Path,
    *,
    crf: int = 16,
    preset: str = "slow",
) -> CompositeReport:
    """Composite three clips into one H.264 MP4.

    Args:
        source_path: the original footage, kept wherever the matte is clear.
        replacement_path: the edited footage, taken wherever the matte is solid.
        matte_path: the subject matte; its **alpha channel** is the blend factor.
        output_path: the ``.mp4`` to write.
        crf: x264 quality, 0-51, lower is better. No value makes the decoded
            output match the composited RGB frames byte-for-byte: the
            ``rgb24 -> yuv420p`` conversion in front of x264 is itself lossy.
            See the module docstring.
        preset: x264 speed/efficiency preset.

    Returns:
        A :class:`CompositeReport` describing what was written.

    Raises:
        CompositeError: on any input mismatch, a frame-count disagreement, a
            matte without alpha, or a failing ffmpeg process.

    The alpha == 0 background-preservation guarantee applies to the frames fed
    to the encoder, not to the ``.mp4`` this writes. See the module docstring.
    """
    source = probe_video(source_path)
    replacement = probe_video(replacement_path)
    matte = _probe_matte(matte_path)
    _validate(source, replacement, matte)

    decodes = {
        "source": _Decode(source_path, "rgb24", None),
        "replacement": _Decode(replacement_path, "rgb24", None),
        "matte": _Decode(matte_path, "rgba", ALPHA_DECODERS.get(matte.codec_name)),
    }
    pipeline = _ffmpeg_pipeline(decodes, output_path, source, crf=crf, preset=preset)
    with pipeline as (pipes, encode):
        ratio = composite_streams(
            pipes["source"],
            pipes["replacement"],
            pipes["matte"],
            encode,
            width=source.width,
            height=source.height,
            frames=source.frame_count,
        )

    return CompositeReport(
        output_path=output_path,
        frames=source.frame_count,
        width=source.width,
        height=source.height,
        frame_rate=source.frame_rate,
        soft_edge_ratio=ratio,
    )


def composite_video_union(
    source_path: Path,
    replacement_path: Path,
    source_matte_path: Path,
    replacement_matte_path: Path,
    output_path: Path,
    *,
    crf: int = 16,
    preset: str = "slow",
) -> UnionCompositeReport:
    """Composite four clips into one H.264 MP4 under the union of two mattes.

    Args:
        source_path: the original footage.
        replacement_path: the edited footage (Kling O1).
        source_matte_path: matte of the person in the *source* clip; its
            alpha channel is one input to the union.
        replacement_matte_path: matte of the person in the *replacement* clip;
            the other input to the union.
        output_path: the ``.mp4`` to write.
        crf: x264 quality, as for :func:`composite_video`.
        preset: x264 speed/efficiency preset.

    Returns:
        A :class:`UnionCompositeReport` describing what was written.

    Raises:
        CompositeError: if any of the four streams differs in size or frame
            count, the two clips differ in frame rate, either matte decodes
            without alpha, any stream ends early or late, or any ffmpeg
            process fails.

    Per frame ``effective_alpha = max(source_alpha, replacement_alpha)`` and
    the frame goes through :func:`composite_frame` unchanged, so
    ``effective_alpha == 0`` pixels are the source's exact bytes and
    ``effective_alpha == 255`` pixels the replacement's, pre-encode. See the
    module docstring for what the union trades away.
    """
    source, decodes = _probe_union_inputs(
        source_path, replacement_path, source_matte_path, replacement_matte_path
    )
    pipeline = _ffmpeg_pipeline(decodes, output_path, source, crf=crf, preset=preset)
    with pipeline as (pipes, encode):
        stats = composite_streams_union(
            pipes["source"],
            pipes["replacement"],
            pipes["source_matte"],
            pipes["replacement_matte"],
            encode,
            width=source.width,
            height=source.height,
            frames=source.frame_count,
        )

    return UnionCompositeReport(
        output_path=output_path,
        frames=source.frame_count,
        width=source.width,
        height=source.height,
        frame_rate=source.frame_rate,
        soft_edge_ratio=stats.soft_edge_ratio,
        union_lift_ratio=stats.union_lift_ratio,
    )


def composite_video_hardened_union(
    source_path: Path,
    replacement_path: Path,
    source_matte_path: Path,
    replacement_matte_path: Path,
    output_path: Path,
    *,
    source_threshold: int = SOURCE_HARDEN_THRESHOLD,
    crf: int = 16,
    preset: str = "slow",
) -> HardenedUnionCompositeReport:
    """Composite four clips into one H.264 MP4 under the hardened union.

    As :func:`composite_video_union` — same four inputs, same fail-closed
    validation, same ``libvpx-vp9`` alpha decoding, same streaming — but the
    effective alpha is :func:`hardened_union_alpha` with ``source_threshold``.

    Args:
        source_path: the original footage.
        replacement_path: the edited footage (Kling O1).
        source_matte_path: matte of the person in the *source* clip.
        replacement_matte_path: matte of the person in the *replacement* clip.
        output_path: the ``.mp4`` to write.
        source_threshold: see :func:`hardened_union_alpha`. Defaults to
            :data:`SOURCE_HARDEN_THRESHOLD`.
        crf: x264 quality, as for :func:`composite_video`.
        preset: x264 speed/efficiency preset.

    Returns:
        A :class:`HardenedUnionCompositeReport`; its ``hardened_ratio`` is the
        mean fraction of the frame the rule actually changed.

    Raises:
        CompositeError: as :func:`composite_video_union`, or if the threshold
            is outside ``1..255`` (checked before anything is probed).
    """
    _check_threshold(source_threshold)
    source, decodes = _probe_union_inputs(
        source_path, replacement_path, source_matte_path, replacement_matte_path
    )
    pipeline = _ffmpeg_pipeline(decodes, output_path, source, crf=crf, preset=preset)
    with pipeline as (pipes, encode):
        stats = composite_streams_hardened_union(
            pipes["source"],
            pipes["replacement"],
            pipes["source_matte"],
            pipes["replacement_matte"],
            encode,
            width=source.width,
            height=source.height,
            frames=source.frame_count,
            source_threshold=source_threshold,
        )

    return HardenedUnionCompositeReport(
        output_path=output_path,
        frames=source.frame_count,
        width=source.width,
        height=source.height,
        frame_rate=source.frame_rate,
        source_threshold=source_threshold,
        soft_edge_ratio=stats.soft_edge_ratio,
        union_lift_ratio=stats.union_lift_ratio,
        hardened_ratio=stats.hardened_ratio,
    )


def composite_video_source_removal(
    source_path: Path,
    replacement_path: Path,
    source_matte_path: Path,
    replacement_matte_path: Path,
    output_path: Path,
    *,
    threshold: int = SOURCE_REMOVAL_THRESHOLD,
    dilation_radius: int = SOURCE_REMOVAL_DILATION_RADIUS,
    crf: int = 16,
    preset: str = "slow",
) -> SourceRemovalCompositeReport:
    """Composite four clips into one H.264 MP4 under the source-removal mask.

    As :func:`composite_video_union` — same four inputs, same fail-closed
    validation, same ``libvpx-vp9`` alpha decoding, same streaming — but the
    source matte is used only as the binary support of
    :func:`source_removal_mask`, and the effective alpha is
    :func:`removal_effective_alpha`.

    Args:
        source_path: the original footage.
        replacement_path: the edited footage (Kling O1).
        source_matte_path: matte of the person in the *source* clip.
        replacement_matte_path: matte of the person in the *replacement* clip.
        output_path: the ``.mp4`` to write.
        threshold: see :func:`source_removal_mask`. Defaults to
            :data:`SOURCE_REMOVAL_THRESHOLD`.
        dilation_radius: see :func:`source_removal_mask`. Defaults to
            :data:`SOURCE_REMOVAL_DILATION_RADIUS`.
        crf: x264 quality, as for :func:`composite_video`.
        preset: x264 speed/efficiency preset.

    Returns:
        A :class:`SourceRemovalCompositeReport` with the four mean ratios.

    Raises:
        CompositeError: as :func:`composite_video_union`, or if the threshold
            or radius is invalid (checked before anything is probed).
    """
    _check_int_range("threshold", threshold, 1, 255)
    _check_radius(dilation_radius)
    source, decodes = _probe_union_inputs(
        source_path, replacement_path, source_matte_path, replacement_matte_path
    )
    pipeline = _ffmpeg_pipeline(decodes, output_path, source, crf=crf, preset=preset)
    with pipeline as (pipes, encode):
        stats = composite_streams_source_removal(
            pipes["source"],
            pipes["replacement"],
            pipes["source_matte"],
            pipes["replacement_matte"],
            encode,
            width=source.width,
            height=source.height,
            frames=source.frame_count,
            threshold=threshold,
            dilation_radius=dilation_radius,
        )

    return SourceRemovalCompositeReport(
        output_path=output_path,
        frames=source.frame_count,
        width=source.width,
        height=source.height,
        frame_rate=source.frame_rate,
        threshold=threshold,
        dilation_radius=dilation_radius,
        soft_edge_ratio=stats.soft_edge_ratio,
        removal_ratio=stats.removal_ratio,
        dilation_only_ratio=stats.dilation_only_ratio,
        replacement_override_ratio=stats.replacement_override_ratio,
    )


def _probe_union_inputs(
    source_path: Path,
    replacement_path: Path,
    source_matte_path: Path,
    replacement_matte_path: Path,
) -> tuple[VideoInfo, dict[str, _Decode]]:
    """Probe and validate the four dual-matte inputs; say how to decode each.

    Shared by every dual-matte composite (union, hardened union, source
    removal) so all of them fail closed on exactly the same conditions,
    before any ffmpeg process is spawned.
    """
    source = probe_video(source_path)
    replacement = probe_video(replacement_path)
    source_matte = _probe_matte(source_matte_path)
    replacement_matte = _probe_matte(replacement_matte_path)
    _validate_union(source, replacement, source_matte, replacement_matte)

    decodes = {
        "source": _Decode(source_path, "rgb24", None),
        "replacement": _Decode(replacement_path, "rgb24", None),
        "source_matte": _Decode(
            source_matte_path, "rgba", ALPHA_DECODERS.get(source_matte.codec_name)
        ),
        "replacement_matte": _Decode(
            replacement_matte_path, "rgba", ALPHA_DECODERS.get(replacement_matte.codec_name)
        ),
    }
    return source, decodes


@dataclass(frozen=True)
class _Decode:
    """One ffmpeg decode to spawn: which file, to which raw format, under which decoder."""

    path: Path
    pix_fmt: str
    decoder: str | None


@contextmanager
def _ffmpeg_pipeline(
    decodes: Mapping[str, _Decode],
    output_path: Path,
    encode_as: VideoInfo,
    *,
    crf: int,
    preset: str,
) -> Iterator[tuple[Mapping[str, IO[bytes]], IO[bytes]]]:
    """Spawn one decoder per input plus the encoder; hand back their pipes.

    Yields ``(decoder stdouts by name, encoder stdin)``. On a clean exit from
    the block the encoder's input is closed and every process must exit 0,
    otherwise :class:`CompositeError` carries its stderr. If the block raises,
    every still-running process is killed and the exception propagates.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logs = {name: tempfile.TemporaryFile() for name in (*decodes, "encode")}
    processes: dict[str, subprocess.Popen[bytes]] = {}
    try:
        for name, decode in decodes.items():
            processes[name] = _decoder(decode.path, decode.pix_fmt, decode.decoder, logs[name])
        processes["encode"] = _encoder(
            output_path, encode_as, crf=crf, preset=preset, log=logs["encode"]
        )
        pipes = {name: _pipe(processes[name].stdout, name) for name in decodes}
        yield pipes, _pipe(processes["encode"].stdin, "encode")
        _finish(processes, logs)
    finally:
        for process in processes.values():
            if process.poll() is None:
                process.kill()
        for log in logs.values():
            log.close()


def _decoder(
    path: Path, pix_fmt: str, decoder: str | None, log: IO[bytes]
) -> subprocess.Popen[bytes]:
    command = [FFMPEG, "-v", "error", "-nostdin"]
    if decoder is not None:
        command += ["-c:v", decoder]
    command += ["-i", str(path), "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", pix_fmt, "-"]
    return subprocess.Popen(command, stdout=subprocess.PIPE, stderr=log)


def _encoder(
    path: Path, source: VideoInfo, *, crf: int, preset: str, log: IO[bytes]
) -> subprocess.Popen[bytes]:
    command = [
        FFMPEG, "-v", "error", "-nostdin", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{source.width}x{source.height}",
        "-r", source.rate_argument,
        "-i", "-",
        "-an",
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(path),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)


def _pipe(stream: IO[bytes] | None, name: str) -> IO[bytes]:
    if stream is None:
        raise CompositeError(f"ffmpeg {name} pipe was not opened")
    return stream


def _finish(
    processes: Mapping[str, subprocess.Popen[bytes]], logs: Mapping[str, IO[bytes]]
) -> None:
    """Close the encoder's input, then insist every process exited cleanly."""
    encode = processes["encode"]
    if encode.stdin is not None:
        encode.stdin.close()
    for name, process in processes.items():
        if process.stdout is not None:
            process.stdout.close()
        _raise_for_failure(name, process.wait(), _read_log(logs[name]))


def _read_log(log: IO[bytes]) -> str:
    log.seek(0)
    return log.read().decode("utf-8", errors="replace").strip()


def _raise_for_failure(name: str, returncode: int, stderr: str) -> None:
    if returncode != 0:
        detail = f": {stderr}" if stderr else ""
        raise CompositeError(f"ffmpeg {name} exited {returncode}{detail}")
