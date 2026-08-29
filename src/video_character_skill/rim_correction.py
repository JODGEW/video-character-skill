"""Local rim-tone correction of O1's foreground edge — the v9 composite.

Why
---
After v7 every compositing-side cause of the thin contour around the head
had been excluded (source hair, soft alpha, the alpha 128-254 ring, plate
colour and texture, static wall shading, codec ringing). What remains is
baked into Kling O1's own RGB: inside ``replacement_alpha >= 250`` the outer
4-6 px are rendered brighter than the interior (head region: +15 luma at the
outermost ring, decaying over ~6 px), about 1.6x stronger and wider than the
real person's natural edge. The silhouette is three very different
materials (dark, mid, bright interiors in equal thirds), so a global
per-ring offset over-corrects the dark ones and leaves the bright ones; a
local box-window offset flattens all three.

Algorithm, per frame (before the unchanged v7 compositing path)
---------------------------------------------------------------
::

    core       = replacement_alpha >= 250
    band_k     = erode_disk(core, lo_k) & ~erode_disk(core, hi_k)   for (lo, hi) in
                 (0,1) (1,2) (2,3) (3,4) (4,6)                       # the target: depth 0-6
    reference  = erode_disk(core, 8) & ~erode_disk(core, 12)         # the interior: depth 8-12

    for each band, per channel, at every band pixel p:
        mean_ref  = mean of reference RGB inside the 32x32 box around p (clipped at the frame)
        mean_band = mean of band_k RGB inside the same box
        valid     = (band pixels in box >= 8) & (reference pixels in box >= 16)
        offset    = mean_ref - mean_band                              # 0 where not valid
        rgb[p]    = clip(round_half_even(rgb[p] + 0.5 * offset), 0, 255)

Box means are computed with float64 integral images; the window is
``[y - 16, y + 16] x [x - 16, x + 16]``, clipped to the frame. Only band
pixels are ever written; the alpha plane, the source-removal mask, the
temporal/spatial recovery and the v7 alpha semantics (soft alpha kept,
``force_replacement`` at 128, removal 32/4) are untouched: the corrected
frame simply replaces the decoded replacement frame via
:data:`video_character_skill.spatial_recovery.ReplacementFilter`.

v11 adds the optional ``residual_threshold``, forwarded unchanged to the v6
loop's tier-2 residual pass (see ``spatial_recovery``); the rim correction
itself, the alpha semantics and the removal geometry are untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import IO

import numpy as np
from numpy.typing import NDArray

from video_character_skill.compositor import (
    SOURCE_REMOVAL_DILATION_RADIUS,
    AlphaPlane,
    CompositeError,
    RgbFrame,
    _check_int_range,
    _check_radius,
    _ffmpeg_pipeline,
    _probe_union_inputs,
)
from video_character_skill.hard_inset_recovery import erode_disk
from video_character_skill.spatial_recovery import (
    SpatialRecoveryStreamStats,
    composite_streams_spatial_recovery,
)
from video_character_skill.temporal_recovery import (
    MAX_TEMPORAL_OBSERVATIONS,
    REPLACEMENT_FOREGROUND_THRESHOLD,
    SOURCE_BACKGROUND_THRESHOLD,
    TEMPORAL_RECOVERY_RADIUS,
    BoolMask,
)

__all__ = [
    "RIM_BANDS",
    "RIM_CORE_THRESHOLD",
    "RIM_MIN_BAND_SAMPLES",
    "RIM_MIN_REFERENCE_SAMPLES",
    "RIM_REFERENCE_DEPTHS",
    "RIM_STRENGTH",
    "RIM_WINDOW",
    "V7_REMOVAL_THRESHOLD",
    "RimBands",
    "RimCorrectedCompositeReport",
    "RimCorrection",
    "RimFilter",
    "RimStreamStats",
    "composite_streams_rim_corrected",
    "composite_video_rim_corrected",
    "correct_rim",
    "local_rim_offsets",
    "rim_bands",
]

Depths = tuple[int, int]

# Replacement alpha at or above this is the opaque core whose edge is corrected.
RIM_CORE_THRESHOLD = 250
# Inward depth bands (erosion radii) that are corrected: 0-6 px.
RIM_BANDS: tuple[Depths, ...] = ((0, 1), (1, 2), (2, 3), (3, 4), (4, 6))
# The interior reference band the edge is matched to.
RIM_REFERENCE_DEPTHS: Depths = (8, 12)
# Side of the square box window and the fraction of the estimated offset applied.
RIM_WINDOW = 32
RIM_STRENGTH = 0.5
# A window is usable only with this many band / reference samples.
RIM_MIN_BAND_SAMPLES = 8
RIM_MIN_REFERENCE_SAMPLES = 16
# Source removal as in the successful v7 POC.
V7_REMOVAL_THRESHOLD = 32


# -- results and reports -------------------------------------------------


@dataclass(frozen=True)
class RimBands:
    """Depth partition of the replacement core. All ``(H, W)`` bool."""

    core: BoolMask
    bands: tuple[BoolMask, ...]
    """One mask per entry of ``RIM_BANDS``; together they are ``target``."""
    reference: BoolMask
    target: BoolMask


@dataclass(frozen=True)
class RimCorrection:
    """What :func:`correct_rim` produced for one frame."""

    rgb: RgbFrame
    """The replacement frame with corrected target pixels; all others unchanged."""
    corrected: BoolMask
    """Target pixels that had a valid window (the only pixels that may differ)."""
    target_pixels: int
    corrected_pixels: int
    clipped_pixels: int
    """Corrected pixels where a channel had to be clipped to 0..255."""
    sum_abs_offset: float
    """Sum over corrected pixels of the max-channel |applied offset| (before rounding)."""
    max_abs_offset: float


@dataclass(frozen=True)
class RimStreamStats:
    """Clip-level diagnostics of the rim correction."""

    target_ratio: float
    """Mean over frames of the target share of the full frame."""
    corrected_ratio: float
    """Mean over frames of the corrected share of the full frame."""
    valid_ratio: float
    """Corrected / target pixels, pixel-weighted over the clip (0 if no target)."""
    mean_abs_offset: float
    """Mean applied max-channel |offset| over corrected pixels (0 if none)."""
    max_abs_offset: float
    clipped_ratio: float
    """Clipped / corrected pixels, pixel-weighted (0 if none)."""


@dataclass(frozen=True)
class RimCorrectedCompositeReport:
    """What one rim-corrected composite run produced."""

    output_path: Path
    frames: int
    width: int
    height: int
    frame_rate: Fraction
    removal_threshold: int
    dilation_radius: int
    background_threshold: int
    residual_threshold: int | None
    foreground_threshold: int
    radius: int
    max_observations: int
    window: int
    strength: float
    stats: SpatialRecoveryStreamStats
    rim: RimStreamStats


# -- pure helpers --------------------------------------------------------


def rim_bands(
    replacement_alpha: AlphaPlane,
    *,
    core_threshold: int = RIM_CORE_THRESHOLD,
    bands: tuple[Depths, ...] = RIM_BANDS,
    reference: Depths = RIM_REFERENCE_DEPTHS,
) -> RimBands:
    """Depth bands of the replacement core by repeated Euclidean-disk erosion. Pure.

    Band ``(lo, hi)`` is ``erode_disk(core, lo) & ~erode_disk(core, hi)``
    with ``erode_disk(core, 0) == core``; the same integer-disk convention
    as :func:`video_character_skill.compositor.dilate_disk`.
    """
    if replacement_alpha.ndim != 2 or replacement_alpha.dtype != np.uint8:
        raise CompositeError("replacement alpha must be (H, W) uint8")
    _check_int_range("core_threshold", core_threshold, 1, 255)
    for lo, hi in (*bands, reference):
        _check_radius(lo)
        _check_radius(hi)
        if hi <= lo:
            raise CompositeError(f"band ({lo}, {hi}) must have lo < hi")
    core: BoolMask = replacement_alpha >= core_threshold
    eroded: dict[int, BoolMask] = {0: core}
    for r in sorted({r for pair in (*bands, reference) for r in pair}):
        if r not in eroded:
            eroded[r] = erode_disk(core, r)
    band_masks = tuple(eroded[lo] & ~eroded[hi] for lo, hi in bands)
    target = np.zeros(core.shape, dtype=np.bool_)
    for m in band_masks:
        target |= m
    return RimBands(
        core=core,
        bands=band_masks,
        reference=eroded[reference[0]] & ~eroded[reference[1]],
        target=target,
    )


def local_rim_offsets(
    rgb: RgbFrame,
    band: BoolMask,
    reference: BoolMask,
    *,
    window: int = RIM_WINDOW,
    min_band_samples: int = RIM_MIN_BAND_SAMPLES,
    min_reference_samples: int = RIM_MIN_REFERENCE_SAMPLES,
) -> tuple[NDArray[np.float64], BoolMask]:
    """Per-pixel additive offset for one band, and where it is valid. Pure.

    For every pixel the ``window x window`` box around it (``half = window
    // 2`` on each side, clipped at the frame) yields the per-channel mean
    of ``reference`` pixels and of ``band`` pixels; the offset is their
    difference. A pixel is valid iff the box holds at least
    ``min_band_samples`` band pixels and ``min_reference_samples`` reference
    pixels; elsewhere the offset is 0. Returns ``(offset (H, W, 3) float64,
    valid (H, W) bool)`` — both defined everywhere, but only ``band & valid``
    is meaningful to apply.
    """
    _check_rgb(rgb)
    for name, m in (("band", band), ("reference", reference)):
        if m.shape != rgb.shape[:2] or m.dtype != np.bool_:
            raise CompositeError(f"{name} must be a bool mask shaped like the frame")
    _check_int_range("window", window, 1, None)
    _check_int_range("min_band_samples", min_band_samples, 1, None)
    _check_int_range("min_reference_samples", min_reference_samples, 1, None)
    half = window // 2
    values = rgb.astype(np.float64)
    count_band = _box_sum(_integral(band.astype(np.float64)), half)
    count_ref = _box_sum(_integral(reference.astype(np.float64)), half)
    valid: BoolMask = (count_band >= min_band_samples) & (count_ref >= min_reference_samples)
    offset = np.zeros(rgb.shape, dtype=np.float64)
    for ch in range(3):
        mean_band = _box_sum(_integral(values[..., ch] * band), half) / np.maximum(count_band, 1)
        mean_ref = _box_sum(_integral(values[..., ch] * reference), half) / np.maximum(count_ref, 1)
        offset[..., ch] = np.where(valid, mean_ref - mean_band, 0.0)
    return offset, valid


def correct_rim(
    rgb: RgbFrame,
    replacement_alpha: AlphaPlane,
    *,
    window: int = RIM_WINDOW,
    strength: float = RIM_STRENGTH,
    core_threshold: int = RIM_CORE_THRESHOLD,
    bands: tuple[Depths, ...] = RIM_BANDS,
    reference: Depths = RIM_REFERENCE_DEPTHS,
    min_band_samples: int = RIM_MIN_BAND_SAMPLES,
    min_reference_samples: int = RIM_MIN_REFERENCE_SAMPLES,
) -> RimCorrection:
    """Apply the local rim-tone correction to one replacement frame. Pure.

    ``rgb[p] = clip(rint(rgb[p] + strength * offset[p]), 0, 255)`` for every
    valid pixel of every band (:func:`local_rim_offsets`); every other pixel
    — and the alpha plane — is returned unchanged. ``strength == 0`` is the
    identity.
    """
    _check_rgb(rgb)
    if replacement_alpha.shape != rgb.shape[:2]:
        raise CompositeError("replacement alpha must be shaped like the frame")
    if not 0.0 <= strength <= 1.0:
        raise CompositeError(f"strength must be within 0..1, got {strength}")
    partition = rim_bands(
        replacement_alpha, core_threshold=core_threshold, bands=bands, reference=reference
    )
    out: RgbFrame = rgb.copy()
    corrected = np.zeros(rgb.shape[:2], dtype=np.bool_)
    clipped = 0
    sum_abs = 0.0
    max_abs = 0.0
    for band in partition.bands:
        if not band.any():
            continue
        offset, valid = local_rim_offsets(
            rgb,
            band,
            partition.reference,
            window=window,
            min_band_samples=min_band_samples,
            min_reference_samples=min_reference_samples,
        )
        apply: BoolMask = band & valid
        if not apply.any():
            continue
        shifted = rgb[apply].astype(np.float64) + strength * offset[apply]
        rounded = np.rint(shifted)
        clipped += int(((rounded < 0) | (rounded > 255)).any(axis=1).sum())
        out[apply] = np.clip(rounded, 0, 255).astype(np.uint8)
        magnitude = np.abs(strength * offset[apply]).max(axis=1)
        sum_abs += float(magnitude.sum())
        max_abs = max(max_abs, float(magnitude.max()))
        corrected |= apply
    return RimCorrection(
        rgb=out,
        corrected=corrected,
        target_pixels=int(np.count_nonzero(partition.target)),
        corrected_pixels=int(np.count_nonzero(corrected)),
        clipped_pixels=clipped,
        sum_abs_offset=sum_abs,
        max_abs_offset=max_abs,
    )


class RimFilter:
    """A :data:`ReplacementFilter` that applies :func:`correct_rim` and keeps totals."""

    def __init__(self, *, window: int = RIM_WINDOW, strength: float = RIM_STRENGTH) -> None:
        _check_int_range("window", window, 1, None)
        if not 0.0 <= strength <= 1.0:
            raise CompositeError(f"strength must be within 0..1, got {strength}")
        self.window = window
        self.strength = strength
        self.frames = 0
        self.target_total = 0.0
        self.corrected_total = 0.0
        self.target_pixels = 0
        self.corrected_pixels = 0
        self.clipped_pixels = 0
        self.sum_abs_offset = 0.0
        self.max_abs_offset = 0.0

    def __call__(self, rgb: RgbFrame, replacement_alpha: AlphaPlane) -> RgbFrame:
        result = correct_rim(
            rgb, replacement_alpha, window=self.window, strength=self.strength
        )
        size = float(rgb.shape[0] * rgb.shape[1])
        self.frames += 1
        self.target_total += result.target_pixels / size
        self.corrected_total += result.corrected_pixels / size
        self.target_pixels += result.target_pixels
        self.corrected_pixels += result.corrected_pixels
        self.clipped_pixels += result.clipped_pixels
        self.sum_abs_offset += result.sum_abs_offset
        self.max_abs_offset = max(self.max_abs_offset, result.max_abs_offset)
        return result.rgb

    def stats(self) -> RimStreamStats:
        return RimStreamStats(
            target_ratio=self.target_total / self.frames if self.frames else 0.0,
            corrected_ratio=self.corrected_total / self.frames if self.frames else 0.0,
            valid_ratio=(
                self.corrected_pixels / self.target_pixels if self.target_pixels else 0.0
            ),
            mean_abs_offset=(
                self.sum_abs_offset / self.corrected_pixels if self.corrected_pixels else 0.0
            ),
            max_abs_offset=self.max_abs_offset,
            clipped_ratio=(
                self.clipped_pixels / self.corrected_pixels if self.corrected_pixels else 0.0
            ),
        )


def _integral(a: NDArray[np.float64]) -> NDArray[np.float64]:
    out = np.zeros((a.shape[0] + 1, a.shape[1] + 1), dtype=np.float64)
    out[1:, 1:] = np.cumsum(np.cumsum(a, axis=0), axis=1)
    return out


def _box_sum(integral: NDArray[np.float64], half: int) -> NDArray[np.float64]:
    """Sum over ``[y - half, y + half] x [x - half, x + half]``, clipped to the frame."""
    height, width = integral.shape[0] - 1, integral.shape[1] - 1
    y = np.arange(height)
    x = np.arange(width)
    y0 = np.clip(y - half, 0, height)
    y1 = np.clip(y + half + 1, 0, height)
    x0 = np.clip(x - half, 0, width)
    x1 = np.clip(x + half + 1, 0, width)
    total: NDArray[np.float64] = (
        integral[y1][:, x1] - integral[y0][:, x1] - integral[y1][:, x0] + integral[y0][:, x0]
    )
    return total


def _check_rgb(rgb: NDArray[np.uint8]) -> None:
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise CompositeError(f"rgb must be (H, W, 3) uint8, got {rgb.shape} {rgb.dtype}")


# -- streaming and orchestration ------------------------------------------


def composite_streams_rim_corrected(
    source: IO[bytes],
    replacement: IO[bytes],
    source_matte: IO[bytes],
    replacement_matte: IO[bytes],
    output: IO[bytes],
    *,
    width: int,
    height: int,
    frames: int,
    window: int = RIM_WINDOW,
    strength: float = RIM_STRENGTH,
    removal_threshold: int = V7_REMOVAL_THRESHOLD,
    dilation_radius: int = SOURCE_REMOVAL_DILATION_RADIUS,
    background_threshold: int = SOURCE_BACKGROUND_THRESHOLD,
    foreground_threshold: int = REPLACEMENT_FOREGROUND_THRESHOLD,
    radius: int = TEMPORAL_RECOVERY_RADIUS,
    max_observations: int = MAX_TEMPORAL_OBSERVATIONS,
    residual_threshold: int | None = None,
) -> tuple[SpatialRecoveryStreamStats, RimStreamStats]:
    """v7 streaming composite with the rim correction applied to each replacement frame.

    Everything is :func:`composite_streams_spatial_recovery` (v6/v7) with a
    :class:`RimFilter` as ``replacement_filter``; ``residual_threshold`` is
    forwarded unchanged (v11 tier-2 residual pass, ``None`` keeps v7/v9).
    Returns the v6 stats and the rim stats.
    """
    rim_filter = RimFilter(window=window, strength=strength)
    stats = composite_streams_spatial_recovery(
        source,
        replacement,
        source_matte,
        replacement_matte,
        output,
        width=width,
        height=height,
        frames=frames,
        removal_threshold=removal_threshold,
        dilation_radius=dilation_radius,
        background_threshold=background_threshold,
        foreground_threshold=foreground_threshold,
        radius=radius,
        max_observations=max_observations,
        residual_threshold=residual_threshold,
        replacement_filter=rim_filter,
    )
    return stats, rim_filter.stats()


def composite_video_rim_corrected(
    source_path: Path,
    replacement_path: Path,
    source_matte_path: Path,
    replacement_matte_path: Path,
    output_path: Path,
    *,
    window: int = RIM_WINDOW,
    strength: float = RIM_STRENGTH,
    removal_threshold: int = V7_REMOVAL_THRESHOLD,
    dilation_radius: int = SOURCE_REMOVAL_DILATION_RADIUS,
    background_threshold: int = SOURCE_BACKGROUND_THRESHOLD,
    foreground_threshold: int = REPLACEMENT_FOREGROUND_THRESHOLD,
    radius: int = TEMPORAL_RECOVERY_RADIUS,
    max_observations: int = MAX_TEMPORAL_OBSERVATIONS,
    residual_threshold: int | None = None,
    crf: int = 16,
    preset: str = "slow",
) -> RimCorrectedCompositeReport:
    """Composite four clips into one H.264 MP4: v7 semantics + local rim correction.

    Same four inputs, fail-closed validation, ``libvpx-vp9`` alpha decoding
    and ffmpeg lifecycle as the v5-v8 orchestrators.

    Raises:
        CompositeError: on an invalid parameter (checked before anything is
            probed), any input mismatch, a matte without alpha, a stream that
            ends early or late, or a failing ffmpeg process.
    """
    _check_int_range("window", window, 1, None)
    if not 0.0 <= strength <= 1.0:
        raise CompositeError(f"strength must be within 0..1, got {strength}")
    _check_int_range("removal_threshold", removal_threshold, 1, 255)
    _check_radius(dilation_radius)
    _check_int_range("background_threshold", background_threshold, 1, 255)
    _check_int_range("foreground_threshold", foreground_threshold, 1, 255)
    _check_int_range("radius", radius, 0, None)
    _check_int_range("max_observations", max_observations, 1, None)
    if residual_threshold is not None:
        _check_int_range("residual_threshold", residual_threshold, 1, 255)
    source, decodes = _probe_union_inputs(
        source_path, replacement_path, source_matte_path, replacement_matte_path
    )
    pipeline = _ffmpeg_pipeline(decodes, output_path, source, crf=crf, preset=preset)
    with pipeline as (pipes, encode):
        stats, rim = composite_streams_rim_corrected(
            pipes["source"],
            pipes["replacement"],
            pipes["source_matte"],
            pipes["replacement_matte"],
            encode,
            width=source.width,
            height=source.height,
            frames=source.frame_count,
            window=window,
            strength=strength,
            removal_threshold=removal_threshold,
            dilation_radius=dilation_radius,
            background_threshold=background_threshold,
            foreground_threshold=foreground_threshold,
            radius=radius,
            max_observations=max_observations,
            residual_threshold=residual_threshold,
        )
    return RimCorrectedCompositeReport(
        output_path=output_path,
        frames=source.frame_count,
        width=source.width,
        height=source.height,
        frame_rate=source.frame_rate,
        removal_threshold=removal_threshold,
        dilation_radius=dilation_radius,
        background_threshold=background_threshold,
        residual_threshold=residual_threshold,
        foreground_threshold=foreground_threshold,
        radius=radius,
        max_observations=max_observations,
        window=window,
        strength=strength,
        stats=stats,
        rim=rim,
    )
