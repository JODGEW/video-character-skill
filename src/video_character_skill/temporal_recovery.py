"""Temporal real-background recovery — the v5 composite.

Why
---
The v4 source-removal composite gets rid of the old person, but it fills the
whole removal mask with O1 pixels. Wherever the replacement person is absent
that is O1's *regenerated* background, and visual QA sees it as a
background-coloured outline around the subject. A read-only feasibility
analysis showed the fix does not need geometry: the camera is static (every
6-frame background pair aligns at (0, 0) +/- 1 px, and whole-clip pairs agree
within 2-3 luma levels once a brightness offset is removed), so the real
background behind the old person can be borrowed from other frames at the
*same* coordinates. Within +/-24 frames 81.5 % of the recovery region has at
least one usable observation (92.9 % over the whole clip), the nearest one is
a median 2-3 frames away, and 19.8 % of the region is already real background
in its own frame. Frames ~183-222 carry a white-balance/exposure shift, so
donors are corrected with a per-channel additive offset fitted on the
background the two frames share.

Algorithm, per target frame ``i``
---------------------------------
::

    removal           = source_removal_mask(source_alpha_i, 64, 4)   # v4, unchanged
    recovery_region   = removal & (replacement_alpha_i < 128)
    own_background    = recovery_region & (source_alpha_i < 32)
    needs_temporal    = recovery_region & ~own_background
    force_replacement = removal & (replacement_alpha_i >= 128)

    for j in i-1, i+1, i-2, i+2, ..., i-24, i+24 (clipped to the clip):
        usable_j  = source_alpha_j < 32
        shared_bg = (source_alpha_i < 32) & usable_j
        offset    = clamp(median over shared_bg of (source_rgb_i - source_rgb_j), -64..64)
        corrected = clip(source_rgb_j + offset, 0, 255)
        every needs_temporal pixel with usable_j takes corrected[pixel], until it holds 5

    recovered  = per-pixel, per-channel median of its observations
    background = source_rgb_i.copy()
    background[recovered pixels]   = recovered
    background[unrecovered pixels] = replacement_rgb_i         # O1 fallback, never the old person
    effective_alpha = replacement_alpha_i.copy()
    effective_alpha[force_replacement] = 255
    output = composite_frame(background, replacement_rgb_i, effective_alpha)

Precedence inside the recovery region: own-frame real background, then
temporally borrowed real background, then O1 background. Outside the removal
mask nothing differs from plain replacement-matte compositing. There is no
registration, optical flow, gain fitting, spatial inpainting or feathering.

Temporal cache
--------------
The source clip and source matte are read ahead of the replacement streams by
``radius`` frames and held in a window of at most ``2 * radius + 1`` frames
(49 at the default radius). A cached frame is its RGB bytes (``W*H*3``) plus
a compact copy of its alpha plane (``W*H``): about 8.3 MB per 1080x1920 frame,
so the window peaks near 410 MB. Nothing else is retained across frames.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import IO

import numpy as np
from numpy.typing import NDArray

from video_character_skill.compositor import (
    SOURCE_REMOVAL_DILATION_RADIUS,
    SOURCE_REMOVAL_THRESHOLD,
    AlphaPlane,
    CompositeError,
    RgbFrame,
    _assert_drained,
    _check_alphas,
    _check_int_range,
    _check_radius,
    _ffmpeg_pipeline,
    _probe_union_inputs,
    _read_alpha_frame,
    _read_rgb_frame,
    composite_frame,
    soft_edge_ratio,
    source_removal_mask,
)

__all__ = [
    "MAX_TEMPORAL_OBSERVATIONS",
    "PHOTOMETRIC_MIN_SAMPLES",
    "PHOTOMETRIC_OFFSET_LIMIT",
    "PHOTOMETRIC_SUBSAMPLE_STRIDE",
    "REPLACEMENT_FOREGROUND_THRESHOLD",
    "SOURCE_BACKGROUND_THRESHOLD",
    "TEMPORAL_RECOVERY_RADIUS",
    "PhotometricOffset",
    "RecoveryRegions",
    "TemporalRecovery",
    "TemporalRecoveryCompositeReport",
    "TemporalRecoveryStreamStats",
    "aggregate_observations",
    "apply_offset",
    "composite_streams_temporal_recovery",
    "composite_video_temporal_recovery",
    "donor_frames",
    "donor_offsets",
    "photometric_offset",
    "recover_pixels",
    "recovered_background",
    "recovery_effective_alpha",
    "recovery_regions",
]

BoolMask = NDArray[np.bool_]
Donor = tuple[int, RgbFrame, AlphaPlane]
"""One donor frame: its index, its source RGB and its source alpha plane."""

# Source alpha below this is real background: usable as a donor observation,
# and kept as-is when it is the target frame's own pixel.
SOURCE_BACKGROUND_THRESHOLD = 32
# Replacement alpha at or above this is the solid replacement person: inside
# the removal mask it is forced opaque exactly as v4 does; below it the pixel
# needs real background behind whatever soft edge the replacement has.
REPLACEMENT_FOREGROUND_THRESHOLD = 128
# Donor frames are searched within +/- this many frames of the target.
TEMPORAL_RECOVERY_RADIUS = 24
# At most this many nearest usable observations are aggregated per pixel.
MAX_TEMPORAL_OBSERVATIONS = 5
# Photometric compensation: additive per-channel offset, fitted as the median
# difference over a strided subsample of the background both frames share.
PHOTOMETRIC_OFFSET_LIMIT = 64
PHOTOMETRIC_SUBSAMPLE_STRIDE = 8
PHOTOMETRIC_MIN_SAMPLES = 256  # subsampled pixels; ~16k full-res pixels at stride 8


# -- results and reports -------------------------------------------------


@dataclass(frozen=True)
class RecoveryRegions:
    """The per-frame partition of the v4 removal mask. All ``(H, W)`` bool."""

    removal: BoolMask
    """v4 source-removal mask: the old person must not survive here."""
    recovery_region: BoolMask
    """``removal`` where the replacement is not solid person: needs real background."""
    own_background: BoolMask
    """``recovery_region`` that is real background in the target frame itself."""
    needs_temporal: BoolMask
    """``recovery_region`` still hiding the old person: borrow from other frames."""
    force_replacement: BoolMask
    """``removal`` where the replacement is solid person: forced opaque, as in v4."""


@dataclass(frozen=True)
class PhotometricOffset:
    """Additive per-channel RGB correction that maps a donor onto the target."""

    offset: tuple[int, int, int]
    """Already clamped. ``(0, 0, 0)`` when not fitted."""
    samples: int
    """Shared-background samples the fit had (after subsampling)."""
    fitted: bool
    """False when ``samples`` was below the minimum and the zero offset was used."""


@dataclass(frozen=True)
class TemporalRecovery:
    """What temporal borrowing produced for one frame's ``needs_temporal`` pixels."""

    indices: NDArray[np.intp]
    """Flat pixel indices of the ``needs_temporal`` region, in raster order."""
    rgb: NDArray[np.uint8]
    """``(N, 3)`` recovered values; meaningful only where ``counts > 0``."""
    counts: NDArray[np.int64]
    """``(N,)`` observations aggregated per pixel; 0 means unrecovered."""
    distances: NDArray[np.int64]
    """``|donor - target|`` in frames, one entry per observation used."""
    donor_fits: int
    """Photometric fits performed (one per donor that contributed)."""
    zero_offset_fallbacks: int
    """Of those, how many fell back to the zero offset."""


@dataclass(frozen=True)
class TemporalRecoveryStreamStats:
    """Clip-level diagnostics of a temporal-recovery composite.

    Every ``*_ratio`` except ``o1_fallback_ratio`` is a mean over frames of a
    fraction of the *full frame*. ``o1_fallback_ratio`` is the mean over
    frames of the fraction of the *recovery region* that ended up as O1
    background (frames with an empty region are skipped).
    """

    soft_edge_ratio: float
    recovery_region_ratio: float
    own_background_ratio: float
    temporal_recovered_ratio: float
    temporal_unrecovered_ratio: float
    o1_fallback_ratio: float
    median_donor_distance: float
    """Median ``|donor - target|`` over all observations used, in frames; NaN if none."""
    p90_donor_distance: float
    mean_observations_per_recovered_pixel: float
    """NaN if no pixel was recovered."""
    donor_fits: int
    zero_offset_fallbacks: int
    """Donor photometric fits that had too little shared background and used zero."""
    peak_cached_frames: int
    """Largest number of source frames held at once; at most ``2 * radius + 1``."""


@dataclass(frozen=True)
class TemporalRecoveryCompositeReport:
    """What one temporal-recovery composite run produced."""

    output_path: Path
    frames: int
    width: int
    height: int
    frame_rate: Fraction
    removal_threshold: int
    dilation_radius: int
    background_threshold: int
    foreground_threshold: int
    radius: int
    max_observations: int
    stats: TemporalRecoveryStreamStats


# -- pure helpers --------------------------------------------------------


def donor_offsets(radius: int) -> list[int]:
    """Frame offsets to try, nearest first: ``[-1, 1, -2, 2, ..., -radius, radius]``."""
    _check_int_range("radius", radius, 0, None)
    return [sign * distance for distance in range(1, radius + 1) for sign in (-1, 1)]


def donor_frames(index: int, frame_count: int, radius: int) -> list[int]:
    """Donor frame indices for target ``index``, nearest first, clipped to the clip.

    The target itself is never a donor. At the clip ends the window is
    one-sided, so the list simply has fewer entries.
    """
    _check_int_range("frame_count", frame_count, 1, None)
    _check_int_range("index", index, 0, frame_count - 1)
    return [index + d for d in donor_offsets(radius) if 0 <= index + d < frame_count]


def recovery_regions(
    source_alpha: AlphaPlane,
    replacement_alpha: AlphaPlane,
    *,
    removal_threshold: int = SOURCE_REMOVAL_THRESHOLD,
    dilation_radius: int = SOURCE_REMOVAL_DILATION_RADIUS,
    background_threshold: int = SOURCE_BACKGROUND_THRESHOLD,
    foreground_threshold: int = REPLACEMENT_FOREGROUND_THRESHOLD,
) -> RecoveryRegions:
    """Partition the frame. Pure.

    ``removal`` is exactly v4's :func:`source_removal_mask`. Inside it, the
    replacement matte decides: ``>= foreground_threshold`` is solid
    replacement person (``force_replacement``); below that the pixel needs real
    background (``recovery_region``), which the target frame supplies itself
    where its own source alpha is ``< background_threshold``
    (``own_background``) and which must otherwise be borrowed
    (``needs_temporal``).
    """
    _check_alphas(source_alpha, replacement_alpha)
    _check_int_range("background_threshold", background_threshold, 1, 255)
    _check_int_range("foreground_threshold", foreground_threshold, 1, 255)
    removal = source_removal_mask(
        source_alpha, threshold=removal_threshold, dilation_radius=dilation_radius
    )
    solid: BoolMask = replacement_alpha >= foreground_threshold
    recovery: BoolMask = removal & ~solid
    own: BoolMask = recovery & (source_alpha < background_threshold)
    return RecoveryRegions(
        removal=removal,
        recovery_region=recovery,
        own_background=own,
        needs_temporal=recovery & ~own,
        force_replacement=removal & solid,
    )


def photometric_offset(
    target_rgb: RgbFrame,
    donor_rgb: RgbFrame,
    target_background: BoolMask,
    donor_background: BoolMask,
    *,
    stride: int = PHOTOMETRIC_SUBSAMPLE_STRIDE,
    min_samples: int = PHOTOMETRIC_MIN_SAMPLES,
    limit: int = PHOTOMETRIC_OFFSET_LIMIT,
) -> PhotometricOffset:
    """Additive per-channel offset that maps ``donor_rgb`` onto ``target_rgb``. Pure.

    Over the background both frames share, subsampled deterministically on a
    ``stride x stride`` grid starting at ``(0, 0)``::

        offset[c] = clamp(round(median(target[c] - donor[c])), -limit, limit)

    Differences are taken in int16, so nothing wraps. With fewer than
    ``min_samples`` shared samples the offset is ``(0, 0, 0)`` and
    ``fitted`` is False. An even sample count averages the two middle values
    before rounding (half to even).

    Raises:
        CompositeError: on mismatched shapes/dtypes or bad parameters.
    """
    _check_rgb_pair(target_rgb, donor_rgb)
    _check_masks(target_background, donor_background, target_rgb.shape[:2])
    _check_int_range("stride", stride, 1, None)
    _check_int_range("min_samples", min_samples, 1, None)
    _check_int_range("limit", limit, 0, 255)
    shared = (target_background & donor_background)[::stride, ::stride]
    samples = int(np.count_nonzero(shared))
    if samples < min_samples:
        return PhotometricOffset(offset=(0, 0, 0), samples=samples, fitted=False)
    target = target_rgb[::stride, ::stride][shared].astype(np.int16)
    donor = donor_rgb[::stride, ::stride][shared].astype(np.int16)
    medians = np.median(target - donor, axis=0)
    clamped = [int(min(max(int(np.rint(m)), -limit), limit)) for m in medians]
    return PhotometricOffset(
        offset=(clamped[0], clamped[1], clamped[2]), samples=samples, fitted=True
    )


def apply_offset(rgb: NDArray[np.uint8], offset: tuple[int, int, int]) -> NDArray[np.uint8]:
    """``clip(rgb + offset, 0, 255)`` on any ``(..., 3)`` uint8 array, in int16. Pure."""
    if rgb.ndim < 1 or rgb.shape[-1] != 3:
        raise CompositeError(f"rgb must be (..., 3), got {rgb.shape}")
    if rgb.dtype != np.uint8:
        raise CompositeError(f"rgb must be uint8, got {rgb.dtype}")
    shifted = rgb.astype(np.int16) + np.asarray(offset, dtype=np.int16)
    corrected: NDArray[np.uint8] = np.clip(shifted, 0, 255).astype(np.uint8)
    return corrected


def aggregate_observations(
    observations: NDArray[np.uint8], valid: NDArray[np.bool_]
) -> tuple[NDArray[np.uint8], NDArray[np.int64]]:
    """Per-pixel, per-channel median over the valid observations. Pure.

    Args:
        observations: ``(K, N, 3)`` uint8 — up to ``K`` observations of ``N`` pixels.
        valid: ``(K, N)`` bool — which slots hold an observation.

    Returns:
        ``(N, 3)`` uint8 medians (0 where a pixel has no observation) and the
        ``(N,)`` observation counts. An even count averages the two middle
        values and rounds half to even.
    """
    if observations.ndim != 3 or observations.shape[2] != 3:
        raise CompositeError(f"observations must be (K, N, 3), got {observations.shape}")
    if valid.shape != observations.shape[:2]:
        raise CompositeError(
            f"valid shape {valid.shape} != observations slots {observations.shape[:2]}"
        )
    if observations.dtype != np.uint8 or valid.dtype != np.bool_:
        raise CompositeError("observations must be uint8 and valid must be bool")
    counts: NDArray[np.int64] = valid.sum(axis=0, dtype=np.int64)
    values = observations.astype(np.float32)
    values[~valid] = np.nan
    medians = np.zeros(observations.shape[1:], dtype=np.float32)
    has = counts > 0
    if has.any():
        medians[has] = np.nanmedian(values[:, has, :], axis=0)
    rgb: NDArray[np.uint8] = np.clip(np.rint(medians), 0, 255).astype(np.uint8)
    return rgb, counts


def recover_pixels(
    target_rgb: RgbFrame,
    target_alpha: AlphaPlane,
    needs_temporal: BoolMask,
    donors: Sequence[Donor],
    *,
    target_index: int,
    background_threshold: int = SOURCE_BACKGROUND_THRESHOLD,
    max_observations: int = MAX_TEMPORAL_OBSERVATIONS,
    offset_stride: int = PHOTOMETRIC_SUBSAMPLE_STRIDE,
    offset_min_samples: int = PHOTOMETRIC_MIN_SAMPLES,
    offset_limit: int = PHOTOMETRIC_OFFSET_LIMIT,
) -> TemporalRecovery:
    """Borrow real background for ``needs_temporal`` pixels from ``donors``. Pure.

    ``donors`` must already be in the order they should be tried (nearest
    first — see :func:`donor_frames`); each is ``(index, rgb, alpha)``. A donor
    pixel is usable when its source alpha is below ``background_threshold``.
    Each target pixel keeps its first ``max_observations`` usable donor values,
    each corrected by that donor's :func:`photometric_offset`, and the search
    stops as soon as every pixel is full. The result is the per-channel median
    of what each pixel collected (:func:`aggregate_observations`).
    """
    _check_rgb_alpha(target_rgb, target_alpha)
    _check_masks(needs_temporal, needs_temporal, target_rgb.shape[:2])
    _check_int_range("background_threshold", background_threshold, 1, 255)
    _check_int_range("max_observations", max_observations, 1, None)
    indices = np.flatnonzero(needs_temporal.ravel())
    pixel_count = int(indices.size)
    observations = np.zeros((max_observations, pixel_count, 3), dtype=np.uint8)
    valid = np.zeros((max_observations, pixel_count), dtype=np.bool_)
    filled = np.zeros(pixel_count, dtype=np.int64)
    distances: list[NDArray[np.int64]] = []
    fits = 0
    fallbacks = 0
    target_background: BoolMask = target_alpha < background_threshold

    for donor_index, donor_rgb, donor_alpha in donors:
        if pixel_count == 0 or bool((filled >= max_observations).all()):
            break
        if donor_index == target_index:
            raise CompositeError(f"frame {target_index} offered as its own donor")
        _check_rgb_alpha(donor_rgb, donor_alpha)
        if donor_rgb.shape != target_rgb.shape:
            raise CompositeError(
                f"donor {donor_index} shape {donor_rgb.shape} != target shape {target_rgb.shape}"
            )
        donor_background: BoolMask = donor_alpha < background_threshold
        take: BoolMask = donor_background.ravel()[indices] & (filled < max_observations)
        if not take.any():
            continue
        offset = photometric_offset(
            target_rgb,
            donor_rgb,
            target_background,
            donor_background,
            stride=offset_stride,
            min_samples=offset_min_samples,
            limit=offset_limit,
        )
        fits += 1
        fallbacks += 0 if offset.fitted else 1
        taken = np.flatnonzero(take)
        samples = apply_offset(donor_rgb.reshape(-1, 3)[indices[taken]], offset.offset)
        slots = filled[taken]
        observations[slots, taken, :] = samples
        valid[slots, taken] = True
        filled[taken] += 1
        distances.append(np.full(taken.size, abs(donor_index - target_index), dtype=np.int64))

    rgb, counts = aggregate_observations(observations, valid)
    return TemporalRecovery(
        indices=indices,
        rgb=rgb,
        counts=counts,
        distances=(
            np.concatenate(distances) if distances else np.zeros(0, dtype=np.int64)
        ),
        donor_fits=fits,
        zero_offset_fallbacks=fallbacks,
    )


def recovered_background(
    source_rgb: RgbFrame, replacement_rgb: RgbFrame, recovery: TemporalRecovery
) -> RgbFrame:
    """The background plate for one frame. Pure.

    A copy of the source frame in which every recovered ``needs_temporal``
    pixel holds its borrowed real background and every unrecovered one holds
    the replacement (O1) pixel — the explicit fallback that guarantees the old
    person never survives. Own-frame background pixels are simply untouched
    source pixels.
    """
    _check_rgb_pair(source_rgb, replacement_rgb)
    background: RgbFrame = source_rgb.copy()
    flat = background.reshape(-1, 3)
    got = recovery.counts > 0
    flat[recovery.indices[got]] = recovery.rgb[got]
    flat[recovery.indices[~got]] = replacement_rgb.reshape(-1, 3)[recovery.indices[~got]]
    return background


def recovery_effective_alpha(
    replacement_alpha: AlphaPlane, force_replacement: BoolMask
) -> AlphaPlane:
    """The replacement matte, forced opaque where v4 would force it. Pure."""
    if force_replacement.shape != replacement_alpha.shape or force_replacement.dtype != np.bool_:
        raise CompositeError("force_replacement must be a bool mask shaped like the alpha")
    effective: AlphaPlane = replacement_alpha.copy()
    effective[force_replacement] = 255
    return effective


def _check_rgb_pair(a: RgbFrame, b: RgbFrame) -> None:
    for name, array in (("target", a), ("donor", b)):
        if array.ndim != 3 or array.shape[2] != 3:
            raise CompositeError(f"{name} rgb must be (H, W, 3), got {array.shape}")
        if array.dtype != np.uint8:
            raise CompositeError(f"{name} rgb must be uint8, got {array.dtype}")
    if a.shape != b.shape:
        raise CompositeError(f"rgb shapes differ: {a.shape} vs {b.shape}")


def _check_rgb_alpha(rgb: RgbFrame, alpha: AlphaPlane) -> None:
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise CompositeError(f"rgb must be (H, W, 3) uint8, got {rgb.shape} {rgb.dtype}")
    if alpha.shape != rgb.shape[:2] or alpha.dtype != np.uint8:
        raise CompositeError(
            f"alpha must be (H, W) uint8 matching rgb, got {alpha.shape} {alpha.dtype}"
        )


def _check_masks(a: BoolMask, b: BoolMask, shape: tuple[int, ...]) -> None:
    for mask in (a, b):
        if mask.shape != shape or mask.dtype != np.bool_:
            raise CompositeError(f"mask must be {shape} bool, got {mask.shape} {mask.dtype}")


# -- the bounded temporal cache -----------------------------------------


class _SourceWindow:
    """Source RGB + alpha for frames ``[target - radius, target + radius]``.

    Reads ahead from the two source streams, evicts what the next target can
    no longer reach, and remembers its peak size so the bound can be checked.
    """

    def __init__(
        self,
        source: IO[bytes],
        source_matte: IO[bytes],
        *,
        width: int,
        height: int,
        frames: int,
        radius: int,
    ) -> None:
        self._source = source
        self._matte = source_matte
        self._width = width
        self._height = height
        self._frames = frames
        self._radius = radius
        self._next = 0
        self._cache: dict[int, tuple[RgbFrame, AlphaPlane]] = {}
        self.peak = 0

    def advance(self, target: int) -> None:
        """Make every frame within ``radius`` of ``target`` available."""
        for stale in [j for j in self._cache if j < target - self._radius]:
            del self._cache[stale]
        last = min(target + self._radius, self._frames - 1)
        while self._next <= last:
            rgb = _read_rgb_frame(self._source, "source", self._next, self._width, self._height)
            alpha = _read_alpha_frame(
                self._matte, "source_matte", self._next, self._width, self._height
            ).copy()  # drop the RGBA buffer the plane was a view into
            self._cache[self._next] = (rgb, alpha)
            self._next += 1
        self.peak = max(self.peak, len(self._cache))

    def frame(self, index: int) -> tuple[RgbFrame, AlphaPlane]:
        return self._cache[index]

    def __len__(self) -> int:
        return len(self._cache)


# -- streaming -----------------------------------------------------------


def composite_streams_temporal_recovery(
    source: IO[bytes],
    replacement: IO[bytes],
    source_matte: IO[bytes],
    replacement_matte: IO[bytes],
    output: IO[bytes],
    *,
    width: int,
    height: int,
    frames: int,
    removal_threshold: int = SOURCE_REMOVAL_THRESHOLD,
    dilation_radius: int = SOURCE_REMOVAL_DILATION_RADIUS,
    background_threshold: int = SOURCE_BACKGROUND_THRESHOLD,
    foreground_threshold: int = REPLACEMENT_FOREGROUND_THRESHOLD,
    radius: int = TEMPORAL_RECOVERY_RADIUS,
    max_observations: int = MAX_TEMPORAL_OBSERVATIONS,
    offset_stride: int = PHOTOMETRIC_SUBSAMPLE_STRIDE,
    offset_min_samples: int = PHOTOMETRIC_MIN_SAMPLES,
    offset_limit: int = PHOTOMETRIC_OFFSET_LIMIT,
) -> TemporalRecoveryStreamStats:
    """Composite ``frames`` frames with temporal background recovery.

    Reads RGB24 from ``source`` and ``replacement`` and RGBA from both mattes,
    writes RGB24 to ``output``. The source streams are read ``radius`` frames
    ahead of the replacement streams (see the module docstring for the cache
    bound); the replacement streams are consumed one frame at a time.

    Raises:
        CompositeError: on an invalid parameter (before anything is read), if
            any of the four streams ends before ``frames`` frames, or if any
            still has data afterwards.
    """
    _check_int_range("removal_threshold", removal_threshold, 1, 255)
    _check_radius(dilation_radius)
    _check_int_range("background_threshold", background_threshold, 1, 255)
    _check_int_range("foreground_threshold", foreground_threshold, 1, 255)
    _check_int_range("radius", radius, 0, None)
    _check_int_range("max_observations", max_observations, 1, None)
    _check_int_range("offset_stride", offset_stride, 1, None)
    _check_int_range("offset_min_samples", offset_min_samples, 1, None)
    _check_int_range("offset_limit", offset_limit, 0, 255)

    window = _SourceWindow(
        source, source_matte, width=width, height=height, frames=frames, radius=radius
    )
    size = float(width * height)
    soft_total = region_total = own_total = recovered_total = unrecovered_total = 0.0
    fallback_total = 0.0
    frames_with_region = 0
    distance_hist = np.zeros(radius + 1, dtype=np.int64)
    recovered_pixels = 0
    observations_used = 0
    fits = 0
    fallbacks = 0

    for index in range(frames):
        window.advance(index)
        replacement_frame = _read_rgb_frame(replacement, "replacement", index, width, height)
        replacement_alpha = _read_alpha_frame(
            replacement_matte, "replacement_matte", index, width, height
        )
        source_frame, source_alpha = window.frame(index)

        regions = recovery_regions(
            source_alpha,
            replacement_alpha,
            removal_threshold=removal_threshold,
            dilation_radius=dilation_radius,
            background_threshold=background_threshold,
            foreground_threshold=foreground_threshold,
        )
        donors: list[Donor] = [
            (j, *window.frame(j)) for j in donor_frames(index, frames, radius)
        ]
        recovery = recover_pixels(
            source_frame,
            source_alpha,
            regions.needs_temporal,
            donors,
            target_index=index,
            background_threshold=background_threshold,
            max_observations=max_observations,
            offset_stride=offset_stride,
            offset_min_samples=offset_min_samples,
            offset_limit=offset_limit,
        )
        background = recovered_background(source_frame, replacement_frame, recovery)
        alpha = recovery_effective_alpha(replacement_alpha, regions.force_replacement)
        output.write(composite_frame(background, replacement_frame, alpha).tobytes())

        region_count = int(np.count_nonzero(regions.recovery_region))
        got = int(np.count_nonzero(recovery.counts > 0))
        missing = int(recovery.counts.size) - got
        soft_total += soft_edge_ratio(alpha)
        region_total += region_count / size
        own_total += float(np.count_nonzero(regions.own_background)) / size
        recovered_total += got / size
        unrecovered_total += missing / size
        if region_count:
            frames_with_region += 1
            fallback_total += missing / region_count
        if recovery.distances.size:
            distance_hist += np.bincount(recovery.distances, minlength=radius + 1)[: radius + 1]
        recovered_pixels += got
        observations_used += int(recovery.counts.sum())
        fits += recovery.donor_fits
        fallbacks += recovery.zero_offset_fallbacks

    _assert_drained(
        {
            "source": source,
            "replacement": replacement,
            "source_matte": source_matte,
            "replacement_matte": replacement_matte,
        },
        frames,
    )

    nan = float("nan")
    return TemporalRecoveryStreamStats(
        soft_edge_ratio=soft_total / frames if frames else 0.0,
        recovery_region_ratio=region_total / frames if frames else 0.0,
        own_background_ratio=own_total / frames if frames else 0.0,
        temporal_recovered_ratio=recovered_total / frames if frames else 0.0,
        temporal_unrecovered_ratio=unrecovered_total / frames if frames else 0.0,
        o1_fallback_ratio=fallback_total / frames_with_region if frames_with_region else 0.0,
        median_donor_distance=_hist_percentile(distance_hist, 50),
        p90_donor_distance=_hist_percentile(distance_hist, 90),
        mean_observations_per_recovered_pixel=(
            observations_used / recovered_pixels if recovered_pixels else nan
        ),
        donor_fits=fits,
        zero_offset_fallbacks=fallbacks,
        peak_cached_frames=window.peak,
    )


def _hist_percentile(hist: NDArray[np.int64], percent: float) -> float:
    """Smallest bin index at which the cumulative count reaches ``percent`` %."""
    total = int(hist.sum())
    if total == 0:
        return float("nan")
    cumulative = np.cumsum(hist)
    return float(int(np.searchsorted(cumulative, percent / 100.0 * total)))


# -- orchestration -------------------------------------------------------


def composite_video_temporal_recovery(
    source_path: Path,
    replacement_path: Path,
    source_matte_path: Path,
    replacement_matte_path: Path,
    output_path: Path,
    *,
    removal_threshold: int = SOURCE_REMOVAL_THRESHOLD,
    dilation_radius: int = SOURCE_REMOVAL_DILATION_RADIUS,
    background_threshold: int = SOURCE_BACKGROUND_THRESHOLD,
    foreground_threshold: int = REPLACEMENT_FOREGROUND_THRESHOLD,
    radius: int = TEMPORAL_RECOVERY_RADIUS,
    max_observations: int = MAX_TEMPORAL_OBSERVATIONS,
    crf: int = 16,
    preset: str = "slow",
) -> TemporalRecoveryCompositeReport:
    """Composite four clips into one H.264 MP4 with temporal background recovery.

    Same four inputs, fail-closed validation, ``libvpx-vp9`` alpha decoding
    and ffmpeg lifecycle as :func:`video_character_skill.compositor.composite_video_union`;
    the per-frame algorithm is the module docstring's.

    Raises:
        CompositeError: on an invalid parameter (checked before anything is
            probed), any input mismatch, a matte without alpha, a stream that
            ends early or late, or a failing ffmpeg process.
    """
    _check_int_range("removal_threshold", removal_threshold, 1, 255)
    _check_radius(dilation_radius)
    _check_int_range("background_threshold", background_threshold, 1, 255)
    _check_int_range("foreground_threshold", foreground_threshold, 1, 255)
    _check_int_range("radius", radius, 0, None)
    _check_int_range("max_observations", max_observations, 1, None)
    source, decodes = _probe_union_inputs(
        source_path, replacement_path, source_matte_path, replacement_matte_path
    )
    pipeline = _ffmpeg_pipeline(decodes, output_path, source, crf=crf, preset=preset)
    with pipeline as (pipes, encode):
        stats = composite_streams_temporal_recovery(
            pipes["source"],
            pipes["replacement"],
            pipes["source_matte"],
            pipes["replacement_matte"],
            encode,
            width=source.width,
            height=source.height,
            frames=source.frame_count,
            removal_threshold=removal_threshold,
            dilation_radius=dilation_radius,
            background_threshold=background_threshold,
            foreground_threshold=foreground_threshold,
            radius=radius,
            max_observations=max_observations,
        )

    return TemporalRecoveryCompositeReport(
        output_path=output_path,
        frames=source.frame_count,
        width=source.width,
        height=source.height,
        frame_rate=source.frame_rate,
        removal_threshold=removal_threshold,
        dilation_radius=dilation_radius,
        background_threshold=background_threshold,
        foreground_threshold=foreground_threshold,
        radius=radius,
        max_observations=max_observations,
        stats=stats,
    )
