"""Hard-inset replacement foreground — the v8 diagnostic composite.

Why
---
After v7 (source removal 32/4 on top of v6's temporal + spatial background
recovery) the source person no longer leaks, the soft replacement alpha is not
a meaningful seam, and no alpha-forcing counterfactual removes the thin
contour that remains around the head. What does explain it is Kling O1's own
anti-aliased edge: O1 rendered the new person over *its* regenerated
background, so every partially covered edge pixel of O1's RGB already carries
O1 background, and VEED's matte gives those pixels alpha 128-254. Composited
at any opacity, that colour shows as a hair-thin fringe.

This POC is deliberately diagnostic: it discards O1's whole edge and keeps
only an inset opaque core. If the contour disappears, edge-RGB contamination
is proven to be the remaining problem.

Algorithm, per target frame ``i``
---------------------------------
::

    removal          = source_removal_mask(source_alpha_i, 32, 4)     # v7 parameters
    replacement_core = replacement_alpha_i >= 250
    hard_foreground  = erode_disk(replacement_core, 2)                 # outside frame = background

    recovery_region  = removal & ~hard_foreground   # old-person pixels the core will not cover
    own_background   = recovery_region & (source_alpha_i < 32)
    needs_temporal   = recovery_region & ~own_background

    temporal recovery (v5) on needs_temporal, spatial fill (v6) on what it misses,
    O1 only for what the spatial fill cannot reach -> reconstructed background

    effective_alpha  = 255 where hard_foreground, 0 everywhere else    # binary
    output           = composite_frame(reconstructed_background, replacement_i, effective_alpha)

Compared with v6/v7 the recovery region is no longer ``removal &
(replacement_alpha < 128)``: the old person under the discarded alpha
128-249 ring inside the removal mask must be rebuilt from real background,
otherwise dropping the ring would expose it. There is no ``force_replacement``,
no soft alpha, no feathering. The silhouette is a little harder and inset by
the erosion radius on purpose.
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
    _assert_drained,
    _check_alphas,
    _check_int_range,
    _check_radius,
    _ffmpeg_pipeline,
    _probe_union_inputs,
    _read_alpha_frame,
    _read_rgb_frame,
    composite_frame,
    dilate_disk,
    source_removal_mask,
)
from video_character_skill.spatial_recovery import FrameBackground, recover_frame_background
from video_character_skill.temporal_recovery import (
    MAX_TEMPORAL_OBSERVATIONS,
    PHOTOMETRIC_MIN_SAMPLES,
    PHOTOMETRIC_OFFSET_LIMIT,
    PHOTOMETRIC_SUBSAMPLE_STRIDE,
    SOURCE_BACKGROUND_THRESHOLD,
    TEMPORAL_RECOVERY_RADIUS,
    BoolMask,
    Donor,
    _hist_percentile,
    _SourceWindow,
    donor_frames,
    recover_pixels,
)

__all__ = [
    "HARD_FOREGROUND_EROSION_RADIUS",
    "HARD_FOREGROUND_THRESHOLD",
    "HARD_INSET_REMOVAL_THRESHOLD",
    "HardInsetCompositeReport",
    "HardInsetRegions",
    "HardInsetStreamStats",
    "composite_streams_hard_inset_recovery",
    "composite_video_hard_inset_recovery",
    "erode_disk",
    "hard_effective_alpha",
    "hard_foreground_mask",
    "hard_inset_regions",
]

# Replacement alpha at or above this is the opaque core of the new person.
HARD_FOREGROUND_THRESHOLD = 250
# The core is inset by this Euclidean-disk radius so O1's contaminated edge is dropped.
HARD_FOREGROUND_EROSION_RADIUS = 2
# Source removal as in the successful v7 POC (v5/v6 keep their own default of 64).
HARD_INSET_REMOVAL_THRESHOLD = 32


# -- results and reports -------------------------------------------------


@dataclass(frozen=True)
class HardInsetRegions:
    """The per-frame partition for v8. All ``(H, W)`` bool."""

    removal: BoolMask
    """Source-removal mask: the old person must not survive here."""
    hard_foreground: BoolMask
    """Eroded replacement core: the only pixels the replacement is composited on."""
    recovery_region: BoolMask
    """``removal & ~hard_foreground``: needs real background."""
    own_background: BoolMask
    """``recovery_region`` that is real background in the target frame itself."""
    needs_temporal: BoolMask
    """``recovery_region`` still hiding the old person."""
    dropped_ring: BoolMask
    """``removal & ~hard_foreground & (replacement_alpha >= 128)``: v6 would force this opaque."""


@dataclass(frozen=True)
class HardInsetStreamStats:
    """Clip-level diagnostics of a hard-inset composite.

    Every ``*_ratio`` except the fallback and ring ratios is a mean over
    frames of a fraction of the *full frame*. ``o1_fallback_ratio`` is the
    mean over frames of the O1 share of the recovery region (frames with an
    empty region skipped). The ``ring_*`` ratios are pixel-weighted over the
    clip: which fallback resolved the dropped alpha 128-254 ring inside the
    removal mask (they sum to 1 when the ring is non-empty, else all 0).
    """

    hard_foreground_ratio: float
    dropped_replacement_ratio: float
    """Replacement alpha > 0 but not hard foreground."""
    dropped_128_249_ratio: float
    """Replacement alpha in 128..249 (never hard foreground)."""
    recovery_region_ratio: float
    own_background_ratio: float
    temporal_recovered_ratio: float
    temporal_unrecovered_ratio: float
    spatial_recovered_ratio: float
    spatial_unrecovered_ratio: float
    o1_fallback_ratio: float
    dropped_ring_ratio: float
    recovered_dropped_ring_ratio: float
    """Share of the dropped ring resolved by real background (own, temporal or spatial)."""
    ring_own_ratio: float
    ring_temporal_ratio: float
    ring_spatial_ratio: float
    ring_o1_ratio: float
    median_donor_distance: float
    p90_donor_distance: float
    mean_observations_per_recovered_pixel: float
    donor_fits: int
    zero_offset_fallbacks: int
    spatial_components: int
    components_without_seed: int
    median_propagation_depth: float
    p90_propagation_depth: float
    max_propagation_depth: int
    peak_cached_frames: int


@dataclass(frozen=True)
class HardInsetCompositeReport:
    """What one hard-inset composite run produced."""

    output_path: Path
    frames: int
    width: int
    height: int
    frame_rate: Fraction
    removal_threshold: int
    dilation_radius: int
    background_threshold: int
    foreground_threshold: int
    erosion_radius: int
    radius: int
    max_observations: int
    stats: HardInsetStreamStats


# -- pure helpers --------------------------------------------------------


def erode_disk(mask: NDArray[np.bool_], radius: int) -> NDArray[np.bool_]:
    """Euclidean-disk erosion with the same disk as :func:`dilate_disk`. Pure.

    A pixel survives iff every offset ``(dy, dx)`` with
    ``dy*dy + dx*dx <= radius*radius`` lands on a True pixel *inside the
    image*: pixels outside the image count as background, so foreground
    touching the border erodes normally. Implemented as
    ``NOT dilate_disk(NOT mask)`` on a copy padded with ``radius`` pixels of
    background, so nothing wraps around. Radius 0 is the identity.
    """
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise CompositeError(f"mask must be a 2-D bool array, got {mask.shape} {mask.dtype}")
    _check_radius(radius)
    if radius == 0:
        return mask.copy()
    height, width = mask.shape
    padded = np.zeros((height + 2 * radius, width + 2 * radius), dtype=np.bool_)
    padded[radius : radius + height, radius : radius + width] = mask
    grown = dilate_disk(~padded, radius)
    eroded: NDArray[np.bool_] = ~grown[radius : radius + height, radius : radius + width]
    return eroded


def hard_foreground_mask(
    replacement_alpha: AlphaPlane,
    *,
    threshold: int = HARD_FOREGROUND_THRESHOLD,
    radius: int = HARD_FOREGROUND_EROSION_RADIUS,
) -> BoolMask:
    """``erode_disk(replacement_alpha >= threshold, radius)``. Pure, binary."""
    if replacement_alpha.ndim != 2 or replacement_alpha.dtype != np.uint8:
        raise CompositeError(
            f"replacement alpha must be (H, W) uint8, got "
            f"{replacement_alpha.shape} {replacement_alpha.dtype}"
        )
    _check_int_range("threshold", threshold, 1, 255)
    _check_radius(radius)
    core: BoolMask = replacement_alpha >= threshold
    return erode_disk(core, radius)


def hard_inset_regions(
    source_alpha: AlphaPlane,
    replacement_alpha: AlphaPlane,
    *,
    removal_threshold: int = HARD_INSET_REMOVAL_THRESHOLD,
    dilation_radius: int = SOURCE_REMOVAL_DILATION_RADIUS,
    background_threshold: int = SOURCE_BACKGROUND_THRESHOLD,
    foreground_threshold: int = HARD_FOREGROUND_THRESHOLD,
    erosion_radius: int = HARD_FOREGROUND_EROSION_RADIUS,
) -> HardInsetRegions:
    """Partition the frame for v8. Pure.

    ``removal`` is v4's :func:`source_removal_mask`; ``hard_foreground`` is
    :func:`hard_foreground_mask`. Everything in ``removal`` that the hard
    foreground will not cover is ``recovery_region``, split into
    ``own_background`` (source alpha below ``background_threshold``) and
    ``needs_temporal`` exactly as v5/v6 do. ``dropped_ring`` is the part of
    the recovery region v6 would have forced opaque (replacement alpha >= 128).
    """
    _check_alphas(source_alpha, replacement_alpha)
    _check_int_range("background_threshold", background_threshold, 1, 255)
    removal = source_removal_mask(
        source_alpha, threshold=removal_threshold, dilation_radius=dilation_radius
    )
    hard = hard_foreground_mask(
        replacement_alpha, threshold=foreground_threshold, radius=erosion_radius
    )
    recovery: BoolMask = removal & ~hard
    own: BoolMask = recovery & (source_alpha < background_threshold)
    return HardInsetRegions(
        removal=removal,
        hard_foreground=hard,
        recovery_region=recovery,
        own_background=own,
        needs_temporal=recovery & ~own,
        dropped_ring=recovery & (replacement_alpha >= 128),
    )


def hard_effective_alpha(hard_foreground: BoolMask) -> AlphaPlane:
    """``255`` on the hard foreground, ``0`` everywhere else. Pure, binary."""
    if hard_foreground.ndim != 2 or hard_foreground.dtype != np.bool_:
        raise CompositeError("hard_foreground must be a 2-D bool mask")
    alpha: AlphaPlane = np.where(hard_foreground, 255, 0).astype(np.uint8)
    return alpha


# -- streaming -----------------------------------------------------------


def composite_streams_hard_inset_recovery(
    source: IO[bytes],
    replacement: IO[bytes],
    source_matte: IO[bytes],
    replacement_matte: IO[bytes],
    output: IO[bytes],
    *,
    width: int,
    height: int,
    frames: int,
    removal_threshold: int = HARD_INSET_REMOVAL_THRESHOLD,
    dilation_radius: int = SOURCE_REMOVAL_DILATION_RADIUS,
    background_threshold: int = SOURCE_BACKGROUND_THRESHOLD,
    foreground_threshold: int = HARD_FOREGROUND_THRESHOLD,
    erosion_radius: int = HARD_FOREGROUND_EROSION_RADIUS,
    radius: int = TEMPORAL_RECOVERY_RADIUS,
    max_observations: int = MAX_TEMPORAL_OBSERVATIONS,
    offset_stride: int = PHOTOMETRIC_SUBSAMPLE_STRIDE,
    offset_min_samples: int = PHOTOMETRIC_MIN_SAMPLES,
    offset_limit: int = PHOTOMETRIC_OFFSET_LIMIT,
) -> HardInsetStreamStats:
    """Composite ``frames`` frames with the hard-inset foreground.

    Same streams, read-ahead window, parameters and failure modes as the v5/v6
    streaming functions; the per-frame algorithm is the module docstring's.

    Raises:
        CompositeError: on an invalid parameter (before anything is read), if
            any of the four streams ends before ``frames`` frames, or if any
            still has data afterwards.
    """
    _check_int_range("removal_threshold", removal_threshold, 1, 255)
    _check_radius(dilation_radius)
    _check_int_range("background_threshold", background_threshold, 1, 255)
    _check_int_range("foreground_threshold", foreground_threshold, 1, 255)
    _check_radius(erosion_radius)
    _check_int_range("radius", radius, 0, None)
    _check_int_range("max_observations", max_observations, 1, None)
    _check_int_range("offset_stride", offset_stride, 1, None)
    _check_int_range("offset_min_samples", offset_min_samples, 1, None)
    _check_int_range("offset_limit", offset_limit, 0, 255)

    window = _SourceWindow(
        source, source_matte, width=width, height=height, frames=frames, radius=radius
    )
    size = float(width * height)
    totals = {
        key: 0.0
        for key in (
            "hard", "dropped", "dropped_128_249", "region", "own", "recovered", "unrecovered",
            "spatial", "residual", "fallback", "ring",
        )
    }
    ring = {"total": 0, "own": 0, "temporal": 0, "spatial": 0, "o1": 0}
    frames_with_region = 0
    distance_hist = np.zeros(radius + 1, dtype=np.int64)
    depth_hist: NDArray[np.int64] = np.zeros(1, dtype=np.int64)
    recovered_pixels = observations_used = fits = fallbacks = components = without_seed = 0

    for index in range(frames):
        window.advance(index)
        replacement_frame = _read_rgb_frame(replacement, "replacement", index, width, height)
        replacement_alpha = _read_alpha_frame(
            replacement_matte, "replacement_matte", index, width, height
        )
        source_frame, source_alpha = window.frame(index)

        regions = hard_inset_regions(
            source_alpha,
            replacement_alpha,
            removal_threshold=removal_threshold,
            dilation_radius=dilation_radius,
            background_threshold=background_threshold,
            foreground_threshold=foreground_threshold,
            erosion_radius=erosion_radius,
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
        result: FrameBackground = recover_frame_background(
            source_frame,
            replacement_frame,
            source_alpha,
            recovery,
            background_threshold=background_threshold,
        )
        alpha = hard_effective_alpha(regions.hard_foreground)
        output.write(composite_frame(result.background, replacement_frame, alpha).tobytes())

        region_count = int(np.count_nonzero(regions.recovery_region))
        got = int(np.count_nonzero(result.temporal_recovered))
        missing = int(np.count_nonzero(result.temporal_unrecovered))
        spatial = int(np.count_nonzero(result.fill.filled))
        residual = int(np.count_nonzero(result.residual))
        totals["hard"] += float(np.count_nonzero(regions.hard_foreground)) / size
        totals["dropped"] += (
            float(np.count_nonzero((replacement_alpha > 0) & ~regions.hard_foreground)) / size
        )
        totals["dropped_128_249"] += (
            float(np.count_nonzero((replacement_alpha >= 128) & (replacement_alpha <= 249)))
            / size
        )
        totals["region"] += region_count / size
        totals["own"] += float(np.count_nonzero(regions.own_background)) / size
        totals["recovered"] += got / size
        totals["unrecovered"] += missing / size
        totals["spatial"] += spatial / size
        totals["residual"] += residual / size
        totals["ring"] += float(np.count_nonzero(regions.dropped_ring)) / size
        if region_count:
            frames_with_region += 1
            totals["fallback"] += residual / region_count
        ring["total"] += int(np.count_nonzero(regions.dropped_ring))
        ring["own"] += int(np.count_nonzero(regions.dropped_ring & regions.own_background))
        ring["temporal"] += int(np.count_nonzero(regions.dropped_ring & result.temporal_recovered))
        ring["spatial"] += int(np.count_nonzero(regions.dropped_ring & result.fill.filled))
        ring["o1"] += int(np.count_nonzero(regions.dropped_ring & result.residual))
        if recovery.distances.size:
            distance_hist += np.bincount(recovery.distances, minlength=radius + 1)[: radius + 1]
        if spatial:
            depths = np.bincount(result.fill.depth[result.fill.filled])
            if depths.size > depth_hist.size:
                depth_hist = np.concatenate(
                    [depth_hist, np.zeros(depths.size - depth_hist.size, dtype=np.int64)]
                )
            depth_hist[: depths.size] += depths
        recovered_pixels += got
        observations_used += int(recovery.counts.sum())
        fits += recovery.donor_fits
        fallbacks += recovery.zero_offset_fallbacks
        components += result.fill.components
        without_seed += result.fill.components_without_seed

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

    def mean(key: str) -> float:
        return totals[key] / frames if frames else 0.0

    def ring_share(key: str) -> float:
        return ring[key] / ring["total"] if ring["total"] else 0.0

    return HardInsetStreamStats(
        hard_foreground_ratio=mean("hard"),
        dropped_replacement_ratio=mean("dropped"),
        dropped_128_249_ratio=mean("dropped_128_249"),
        recovery_region_ratio=mean("region"),
        own_background_ratio=mean("own"),
        temporal_recovered_ratio=mean("recovered"),
        temporal_unrecovered_ratio=mean("unrecovered"),
        spatial_recovered_ratio=mean("spatial"),
        spatial_unrecovered_ratio=mean("residual"),
        o1_fallback_ratio=totals["fallback"] / frames_with_region if frames_with_region else 0.0,
        dropped_ring_ratio=mean("ring"),
        recovered_dropped_ring_ratio=(
            ring_share("own") + ring_share("temporal") + ring_share("spatial")
        ),
        ring_own_ratio=ring_share("own"),
        ring_temporal_ratio=ring_share("temporal"),
        ring_spatial_ratio=ring_share("spatial"),
        ring_o1_ratio=ring_share("o1"),
        median_donor_distance=_hist_percentile(distance_hist, 50),
        p90_donor_distance=_hist_percentile(distance_hist, 90),
        mean_observations_per_recovered_pixel=(
            observations_used / recovered_pixels if recovered_pixels else nan
        ),
        donor_fits=fits,
        zero_offset_fallbacks=fallbacks,
        spatial_components=components,
        components_without_seed=without_seed,
        median_propagation_depth=_hist_percentile(depth_hist, 50),
        p90_propagation_depth=_hist_percentile(depth_hist, 90),
        max_propagation_depth=int(np.flatnonzero(depth_hist).max()) if depth_hist.any() else 0,
        peak_cached_frames=window.peak,
    )


# -- orchestration -------------------------------------------------------


def composite_video_hard_inset_recovery(
    source_path: Path,
    replacement_path: Path,
    source_matte_path: Path,
    replacement_matte_path: Path,
    output_path: Path,
    *,
    removal_threshold: int = HARD_INSET_REMOVAL_THRESHOLD,
    dilation_radius: int = SOURCE_REMOVAL_DILATION_RADIUS,
    background_threshold: int = SOURCE_BACKGROUND_THRESHOLD,
    foreground_threshold: int = HARD_FOREGROUND_THRESHOLD,
    erosion_radius: int = HARD_FOREGROUND_EROSION_RADIUS,
    radius: int = TEMPORAL_RECOVERY_RADIUS,
    max_observations: int = MAX_TEMPORAL_OBSERVATIONS,
    crf: int = 16,
    preset: str = "slow",
) -> HardInsetCompositeReport:
    """Composite four clips into one H.264 MP4 with the hard-inset foreground.

    Same four inputs, fail-closed validation, ``libvpx-vp9`` alpha decoding
    and ffmpeg lifecycle as the v5/v6 orchestrators; the per-frame algorithm
    is the module docstring's.

    Raises:
        CompositeError: on an invalid parameter (checked before anything is
            probed), any input mismatch, a matte without alpha, a stream that
            ends early or late, or a failing ffmpeg process.
    """
    _check_int_range("removal_threshold", removal_threshold, 1, 255)
    _check_radius(dilation_radius)
    _check_int_range("background_threshold", background_threshold, 1, 255)
    _check_int_range("foreground_threshold", foreground_threshold, 1, 255)
    _check_radius(erosion_radius)
    _check_int_range("radius", radius, 0, None)
    _check_int_range("max_observations", max_observations, 1, None)
    source, decodes = _probe_union_inputs(
        source_path, replacement_path, source_matte_path, replacement_matte_path
    )
    pipeline = _ffmpeg_pipeline(decodes, output_path, source, crf=crf, preset=preset)
    with pipeline as (pipes, encode):
        stats = composite_streams_hard_inset_recovery(
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
            erosion_radius=erosion_radius,
            radius=radius,
            max_observations=max_observations,
        )

    return HardInsetCompositeReport(
        output_path=output_path,
        frames=source.frame_count,
        width=source.width,
        height=source.height,
        frame_rate=source.frame_rate,
        removal_threshold=removal_threshold,
        dilation_radius=dilation_radius,
        background_threshold=background_threshold,
        foreground_threshold=foreground_threshold,
        erosion_radius=erosion_radius,
        radius=radius,
        max_observations=max_observations,
        stats=stats,
    )
