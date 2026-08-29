"""Spatial real-background fill — the v6 composite.

Why
---
Read-only attribution of the v5 output (``composite_v5_recovery.mp4``)
showed that temporal recovery works: the recovered pixels sit against the
surrounding real background as well as real background does (seam p90 11.8
vs 10.6 for the control). The residual light patches are the pixels v5 could
not recover at all — ``temporal_unrecovered``, 0.12 % of the frame, ~18 % of
the recovery region — which fell back to O1's regenerated background (seam
p90 103.8, systematically lighter than the room). A wider temporal search
cannot fix them: 47 % of those pixels are never real background at that
coordinate in any frame, because the old person's silhouette never moves off
them. They must be filled *spatially*, from real background next to them.

Algorithm, per target frame ``i``
---------------------------------
Everything through temporal recovery is v5, unchanged (same masks, donors,
photometric correction, cache). Then::

    plate      = source_rgb_i with the recovered pixels written in     # no O1 yet
    trusted    = (source_alpha_i < 32) | temporal_recovered            # real background only
    target     = temporal_unrecovered                                  # the only pixels touched

    for each 8-connected component of target, inside its bbox + 1 px:
        resolved = trusted pixels 8-adjacent to the component           # seeds
        repeat (synchronous waves):
            for every unresolved component pixel with >= 1 resolved 8-neighbour:
                rgb[c] = round_half_up(mean of the resolved neighbours' rgb[c])
            mark those pixels resolved                                  # all at once
        until no unresolved pixel has a resolved neighbour

    background = plate;  background[filled]   = wave values
                         background[residual] = replacement_i          # O1 only here
    effective_alpha = replacement_alpha_i;  effective_alpha[force_replacement] = 255
    output = composite_frame(background, replacement_i, effective_alpha)

Each wave reads only the resolved state left by the previous wave, so the
result is independent of traversal order. Replacement-person pixels, the old
person's own pixels and O1 pixels are never resolved, so nothing propagates
through or from them; a component with no trusted neighbour at all stays
unfilled and keeps the O1 fallback. Precedence inside the recovery region is
therefore: own-frame real background, temporally borrowed real background,
spatially propagated real background, and only then O1. The old person is
never a fallback. No optical flow, registration, gain fitting, feathering,
external inpainting or new dependency.

Tier-2 residual recovery (v11)
------------------------------
With ``residual_threshold`` above ``background_threshold`` (v11: 1 and 32),
the tier-1 residual — the components the fill above left for O1 — gets one
more, lower-confidence pass before the O1 fallback. Nothing above changes::

    residual_1 = temporal_unrecovered & ~filled         # whole seedless components
    low        = background_threshold <= source_alpha_i < residual_threshold
    S_inside   = residual_1 & low       # own-frame soft pixels: source RGB copied in
    T          = residual_1 & ~low      # old-person pixels: filled by the same waves
    seeds      = trusted | filled | low # tier-1 resolved, S_inside and the low ring
                                        # outside residual_1 (source RGB)
    fill T from seeds exactly as above
    residual_2 = T & ~filled_2          # only this falls back to O1

Tier-1 own, temporal and spatial pixels are never rewritten; only
``residual_1`` pixels can differ from the single-tier result. A residual
component no seed touches is left whole to O1. ``None`` (the default) or a
threshold not above ``background_threshold`` disables the pass and reproduces
v6 byte for byte.
"""

from __future__ import annotations

import math
from collections.abc import Callable
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
    _check_int_range,
    _check_radius,
    _ffmpeg_pipeline,
    _probe_union_inputs,
    _read_alpha_frame,
    _read_rgb_frame,
    composite_frame,
    soft_edge_ratio,
)
from video_character_skill.temporal_recovery import (
    MAX_TEMPORAL_OBSERVATIONS,
    PHOTOMETRIC_MIN_SAMPLES,
    PHOTOMETRIC_OFFSET_LIMIT,
    PHOTOMETRIC_SUBSAMPLE_STRIDE,
    REPLACEMENT_FOREGROUND_THRESHOLD,
    SOURCE_BACKGROUND_THRESHOLD,
    TEMPORAL_RECOVERY_RADIUS,
    BoolMask,
    Donor,
    TemporalRecovery,
    _hist_percentile,
    _SourceWindow,
    donor_frames,
    recover_pixels,
    recovery_effective_alpha,
    recovery_regions,
)

__all__ = [
    "SPATIAL_FILL_MARGIN",
    "ComponentFill",
    "FrameBackground",
    "ReplacementFilter",
    "ResidualFill",
    "SpatialFill",
    "SpatialRecoveryCompositeReport",
    "SpatialRecoveryStreamStats",
    "composite_streams_spatial_recovery",
    "composite_video_spatial_recovery",
    "label_components",
    "recover_frame_background",
    "recover_residual",
    "spatial_fill_component",
    "spatial_fill_components",
    "temporal_plate",
    "trusted_background",
]

# Pixels of context kept around a component's bounding box: one ring, which
# is exactly the reach of an 8-neighbourhood.
SPATIAL_FILL_MARGIN = 1

ReplacementFilter = Callable[[RgbFrame, AlphaPlane], RgbFrame]
"""Optional per-frame hook: ``(replacement_rgb, replacement_alpha) -> replacement_rgb``.

Applied to the decoded replacement frame before anything else sees it. It
may only return a new RGB frame; the alpha plane, every mask and the
compositing path are untouched. ``None`` means v6 behaviour, byte for byte.
"""

# The eight (dy, dx) neighbour offsets, in a fixed order (the order is
# irrelevant to the result: each wave only sums over them).
_NEIGHBOURS = tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0))
_DY = np.array([dy for dy, _ in _NEIGHBOURS], dtype=np.int64)
_DX = np.array([dx for _, dx in _NEIGHBOURS], dtype=np.int64)


# -- results and reports -------------------------------------------------


@dataclass(frozen=True)
class ComponentFill:
    """What the wavefront produced for one component, on its crop."""

    rgb: NDArray[np.uint8]
    """Crop of the plate with the component's filled pixels written; else unchanged."""
    filled: BoolMask
    """Component pixels that were resolved by some wave."""
    depth: NDArray[np.int32]
    """Wave index (1-based) at which each filled pixel resolved; 0 elsewhere."""
    seeds: int
    """Trusted pixels 8-adjacent to the component that started the propagation."""


@dataclass(frozen=True)
class SpatialFill:
    """What the spatial fill produced for one frame."""

    rgb: RgbFrame
    """The plate with every filled target pixel written; all other pixels unchanged."""
    filled: BoolMask
    """Target pixels that were filled."""
    depth: NDArray[np.int32]
    """Wave index (1-based) at which each filled pixel resolved; 0 elsewhere."""
    components: int
    """8-connected components of the target."""
    components_without_seed: int
    """Components with no trusted 8-neighbour; their pixels are left unfilled."""


@dataclass(frozen=True)
class ResidualFill:
    """What the tier-2 pass produced for one frame's tier-1 residual."""

    rgb: RgbFrame
    """The tier-1 plate with ``accepted`` and ``filled`` written; all else unchanged."""
    accepted: BoolMask
    """``S_inside``: residual pixels in the low band, holding their own source RGB."""
    filled: BoolMask
    """``T`` pixels the waves resolved."""
    residual: BoolMask
    """``residual_2``: ``T`` pixels no wave reached — the only O1 fallback left."""
    depth: NDArray[np.int32]
    """Wave index (1-based) at which each filled pixel resolved; 0 elsewhere."""
    components: int
    """8-connected components of the tier-1 residual."""
    components_without_seed: int
    """Of those, the ones no seed touched: left whole to O1."""


@dataclass(frozen=True)
class FrameBackground:
    """The v6 background plate for one frame and how it was assembled."""

    background: RgbFrame
    temporal_recovered: BoolMask
    """Pixels that hold v5's temporally borrowed real background."""
    temporal_unrecovered: BoolMask
    """Pixels v5 could not recover: the spatial target."""
    fill: SpatialFill
    tier1_residual: BoolMask
    """``temporal_unrecovered`` pixels the tier-1 fill could not reach."""
    tier2: ResidualFill | None
    """The tier-2 pass over ``tier1_residual``; ``None`` when it is disabled."""
    residual: BoolMask
    """Pixels that end as O1 fallback: ``tier1_residual`` with tier 2 off, else its ``residual``."""


@dataclass(frozen=True)
class SpatialRecoveryStreamStats:
    """Clip-level diagnostics of a spatial-recovery composite.

    Every ``*_ratio`` except ``o1_fallback_ratio`` is a mean over frames of a
    fraction of the *full frame*. ``o1_fallback_ratio`` is the mean over
    frames of the fraction of the *recovery region* that ended up as O1
    background (frames with an empty region are skipped). The propagation
    depths are over all spatially filled pixels of the clip, in waves
    (1 = adjacent to a trusted seed); NaN / 0 when nothing was filled.

    ``spatial_unrecovered_ratio`` and ``o1_fallback_ratio`` count the pixels
    that actually end as O1 fallback: ``residual_2`` when the tier-2 pass is
    on, the tier-1 residual otherwise. ``tier1_residual_ratio`` and the
    ``tier2_*`` fields report that pass; its depths are ``math.nan`` (one
    shared object, so stats stay comparable) / 0 when nothing was filled or
    the pass is off. ``tier2_residual_ratio`` equals ``spatial_unrecovered_ratio``.
    """

    soft_edge_ratio: float
    recovery_region_ratio: float
    own_background_ratio: float
    temporal_recovered_ratio: float
    temporal_unrecovered_ratio: float
    spatial_recovered_ratio: float
    spatial_unrecovered_ratio: float
    o1_fallback_ratio: float
    spatial_components: int
    components_without_seed: int
    median_propagation_depth: float
    p90_propagation_depth: float
    max_propagation_depth: int
    median_donor_distance: float
    p90_donor_distance: float
    mean_observations_per_recovered_pixel: float
    donor_fits: int
    zero_offset_fallbacks: int
    peak_cached_frames: int
    tier1_residual_ratio: float
    """Mean over frames of the tier-1 residual share of the full frame (before tier 2)."""
    tier2_accepted_ratio: float
    """Mean over frames of the ``S_inside`` share of the full frame."""
    tier2_filled_ratio: float
    """Mean over frames of the tier-2 filled ``T`` share of the full frame."""
    tier2_residual_ratio: float
    """Mean over frames of the final O1-fallback share; equals ``spatial_unrecovered_ratio``."""
    tier2_components: int
    """Tier-1 residual components the tier-2 pass processed."""
    tier2_components_without_seed: int
    """Of those, the ones no tier-2 seed touched: left whole to O1."""
    median_tier2_depth: float
    p90_tier2_depth: float
    max_tier2_depth: int


@dataclass(frozen=True)
class SpatialRecoveryCompositeReport:
    """What one spatial-recovery composite run produced."""

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
    stats: SpatialRecoveryStreamStats


# -- pure helpers --------------------------------------------------------


def label_components(mask: BoolMask) -> tuple[NDArray[np.int32], int]:
    """8-connected components of a bool mask. Pure and deterministic.

    Returns ``(labels, count)``: ``labels`` is ``(H, W)`` int32 with 0 off
    the mask and ``1..count`` on it, numbered in raster order of each
    component's first pixel.
    """
    _check_mask("mask", mask)
    labels = np.zeros(mask.shape, dtype=np.int32)
    ys, xs = np.nonzero(mask)
    n = int(ys.size)
    if n == 0:
        return labels, 0
    height, width = mask.shape
    ids = np.full(mask.shape, -1, dtype=np.int64)
    ids[ys, xs] = np.arange(n)
    heads: list[NDArray[np.int64]] = []
    tails: list[NDArray[np.int64]] = []
    for dy, dx in ((0, 1), (1, -1), (1, 0), (1, 1)):  # each pair once
        y2, x2 = ys + dy, xs + dx
        inside = (y2 < height) & (x2 >= 0) & (x2 < width)
        neighbour = ids[y2[inside], x2[inside]]
        linked = neighbour >= 0
        heads.append(np.flatnonzero(inside)[linked])
        tails.append(neighbour[linked])
    a = np.concatenate(heads)
    b = np.concatenate(tails)
    root = np.arange(n)
    while True:  # hook every edge to the smaller root, then compress; repeat until stable
        low = np.minimum(root[a], root[b])
        hooked = root.copy()
        np.minimum.at(hooked, root[a], low)
        np.minimum.at(hooked, root[b], low)
        while True:
            jumped = hooked[hooked]
            if np.array_equal(jumped, hooked):
                break
            hooked = jumped
        if np.array_equal(hooked, root):
            break
        root = hooked
    roots, inverse = np.unique(root, return_inverse=True)
    labels[ys, xs] = inverse.astype(np.int32) + 1
    return labels, int(roots.size)


def spatial_fill_component(
    rgb: NDArray[np.uint8], trusted: BoolMask, component: BoolMask
) -> ComponentFill:
    """Fill one component from its trusted 8-neighbours by synchronous waves. Pure.

    Args:
        rgb: ``(h, w, 3)`` uint8 crop of the plate.
        trusted: ``(h, w)`` bool — real-background pixels whose values may be read.
        component: ``(h, w)`` bool — the pixels to fill; disjoint from ``trusted``.

    Wave ``k`` computes, for every still-unresolved component pixel with at
    least one resolved 8-neighbour, the per-channel mean of those resolved
    neighbours rounded half up, ``(2 * sum + count) // (2 * count)``, from
    the resolved state at the end of wave ``k - 1``; then all of them become
    resolved together. Resolved pixels are the seeds (trusted pixels
    8-adjacent to the component) plus what earlier waves filled — never
    other pixels of the crop, so nothing is read through replacement-person,
    old-person or O1 pixels. Stops when no unresolved pixel has a resolved
    neighbour; a component with no seed fills nothing.

    The waves are evaluated on a sparse frontier: the pixels that can become
    ready in wave ``k`` are exactly the unresolved component pixels
    8-adjacent to the pixels resolved in wave ``k - 1`` (any pixel adjacent
    to an older resolved pixel was already ready in an earlier wave), so
    only those candidates are found (:func:`_wave_candidates`) and only
    their own 8-neighbourhoods are summed (:func:`_fill_candidates`), against
    the resolved mask as it stood before the wave. The work is proportional
    to the component's pixels, not to its bounding box times its depth.
    """
    _check_rgb("rgb", rgb)
    _check_mask("trusted", trusted, rgb.shape[:2])
    _check_mask("component", component, rgb.shape[:2])
    if bool((trusted & component).any()):
        raise CompositeError("trusted and component overlap")
    resolved: BoolMask = trusted & _adjacent(component)
    seeds = int(np.count_nonzero(resolved))
    width = component.shape[1]
    values = rgb.astype(np.int64)
    filled = np.zeros(component.shape, dtype=np.bool_)
    depth = np.zeros(component.shape, dtype=np.int32)
    unresolved = component.copy()
    frontier: NDArray[np.int64] = np.flatnonzero(resolved).astype(np.int64)  # wave 0: the seeds
    wave = 0
    while frontier.size:
        candidates = _wave_candidates(frontier, unresolved)
        if candidates.size == 0:
            break
        wave += 1
        new_values = _fill_candidates(values, resolved, candidates)  # previous wave's state
        cy, cx = np.divmod(candidates, width)
        values[cy, cx] = new_values  # committed together, after every candidate was computed
        resolved[cy, cx] = True
        unresolved[cy, cx] = False
        filled[cy, cx] = True
        depth[cy, cx] = wave
        frontier = candidates
    out = rgb.copy()
    out[filled] = values[filled].astype(np.uint8)
    return ComponentFill(rgb=out, filled=filled, depth=depth, seeds=seeds)


def spatial_fill_components(rgb: RgbFrame, trusted: BoolMask, target: BoolMask) -> SpatialFill:
    """Fill every 8-connected component of ``target``, each on its own crop. Pure.

    Only ``target`` pixels are ever written. Each component is processed on
    its bounding box plus :data:`SPATIAL_FILL_MARGIN` and sees only
    ``trusted`` pixels as seeds, so components never contaminate each other
    even when they share a crop. Raises :class:`CompositeError` if
    ``trusted`` and ``target`` overlap.
    """
    _check_rgb("rgb", rgb)
    _check_mask("trusted", trusted, rgb.shape[:2])
    _check_mask("target", target, rgb.shape[:2])
    if bool((trusted & target).any()):
        raise CompositeError("trusted and target overlap")
    labels, count = label_components(target)
    out: RgbFrame = rgb.copy()
    filled = np.zeros(target.shape, dtype=np.bool_)
    depth = np.zeros(target.shape, dtype=np.int32)
    without_seed = 0
    for label, (rows, cols) in enumerate(_component_windows(labels, count), start=1):
        component: BoolMask = labels[rows, cols] == label
        result = spatial_fill_component(rgb[rows, cols], trusted[rows, cols], component)
        if result.seeds == 0:
            without_seed += 1
        window_out = out[rows, cols]
        window_out[result.filled] = result.rgb[result.filled]
        filled[rows, cols] |= result.filled
        window_depth = depth[rows, cols]
        window_depth[result.filled] = result.depth[result.filled]
    return SpatialFill(
        rgb=out,
        filled=filled,
        depth=depth,
        components=count,
        components_without_seed=without_seed,
    )


def trusted_background(
    source_alpha: AlphaPlane,
    temporal_recovered: BoolMask,
    *,
    background_threshold: int = SOURCE_BACKGROUND_THRESHOLD,
) -> BoolMask:
    """Real background the spatial fill may read from. Pure.

    ``(source_alpha < background_threshold) | temporal_recovered``: the
    frame's own real background plus what v5 borrowed from other frames.
    Replacement-person pixels are never trusted.
    """
    _check_alpha("source_alpha", source_alpha)
    _check_mask("temporal_recovered", temporal_recovered, source_alpha.shape)
    _check_int_range("background_threshold", background_threshold, 1, 255)
    trusted: BoolMask = (source_alpha < background_threshold) | temporal_recovered
    return trusted


def temporal_plate(
    source_rgb: RgbFrame, recovery: TemporalRecovery
) -> tuple[RgbFrame, BoolMask, BoolMask]:
    """The source frame with v5's recovered pixels written in, and its two masks. Pure.

    Returns ``(plate, temporal_recovered, temporal_unrecovered)``. Unlike
    :func:`video_character_skill.temporal_recovery.recovered_background` no
    O1 pixel is written: unrecovered pixels still hold the old person here
    and are exactly the spatial target.
    """
    _check_rgb("source_rgb", source_rgb)
    plate: RgbFrame = source_rgb.copy()
    got = recovery.counts > 0
    flat = plate.reshape(-1, 3)
    flat[recovery.indices[got]] = recovery.rgb[got]
    recovered = np.zeros(source_rgb.shape[:2], dtype=np.bool_).reshape(-1)
    recovered[recovery.indices[got]] = True
    unrecovered = np.zeros(source_rgb.shape[:2], dtype=np.bool_).reshape(-1)
    unrecovered[recovery.indices[~got]] = True
    return plate, recovered.reshape(source_rgb.shape[:2]), unrecovered.reshape(source_rgb.shape[:2])


def recover_residual(
    plate: RgbFrame,
    source_rgb: RgbFrame,
    source_alpha: AlphaPlane,
    resolved: BoolMask,
    residual: BoolMask,
    *,
    background_threshold: int,
    residual_threshold: int,
) -> ResidualFill:
    """The tier-2 pass over the tier-1 residual. Pure.

    ``plate`` is the tier-1 plate (own background, temporal recovery and the
    tier-1 fill written in; ``residual`` pixels still hold the source frame),
    ``resolved`` everything tier 1 resolved (trusted plus filled) and
    ``residual`` the tier-1 residual; the two must not overlap. With
    ``low = background_threshold <= source_alpha < residual_threshold``::

        accepted = residual & low      # S_inside: its own source RGB is copied in
        target   = residual & ~low     # T
        seeds    = resolved | low      # S_inside, the low ring outside the residual
                                       # and whatever tier 1 resolved

    ``target`` is filled by :func:`spatial_fill_components` from ``seeds``,
    read from the plate — which is the source frame at every low pixel that
    tier 1 did not resolve (:func:`temporal_plate` and the tier-1 fill write
    nothing else), so every new seed contributes source RGB. Only
    ``accepted`` and filled pixels are written. A residual component no seed
    touches stays unfilled: the returned ``residual`` (``residual_2``) is
    ``target`` minus the filled pixels. With ``residual_threshold <=
    background_threshold`` the low band is empty and nothing is accepted or
    filled.
    """
    _check_rgb("plate", plate)
    _check_rgb("source_rgb", source_rgb)
    if source_rgb.shape != plate.shape:
        raise CompositeError(f"source shape {source_rgb.shape} != plate shape {plate.shape}")
    _check_alpha("source_alpha", source_alpha)
    if source_alpha.shape != plate.shape[:2]:
        raise CompositeError(f"source_alpha shape {source_alpha.shape} != {plate.shape[:2]}")
    _check_mask("resolved", resolved, plate.shape[:2])
    _check_mask("residual", residual, plate.shape[:2])
    if bool((resolved & residual).any()):
        raise CompositeError("resolved and residual overlap")
    _check_int_range("background_threshold", background_threshold, 1, 255)
    _check_int_range("residual_threshold", residual_threshold, 1, 255)
    low: BoolMask = (source_alpha >= background_threshold) & (source_alpha < residual_threshold)
    accepted: BoolMask = residual & low
    target: BoolMask = residual & ~low
    seeds: BoolMask = resolved | low
    plate2: RgbFrame = plate.copy()
    plate2[accepted] = source_rgb[accepted]
    fill = spatial_fill_components(plate2, seeds, target)
    labels, components = label_components(residual)
    touched = np.zeros(components + 1, dtype=np.bool_)
    touched[labels[accepted | fill.filled]] = True
    return ResidualFill(
        rgb=fill.rgb,
        accepted=accepted,
        filled=fill.filled,
        residual=target & ~fill.filled,
        depth=fill.depth,
        components=components,
        components_without_seed=components - int(np.count_nonzero(touched[1:])),
    )


def recover_frame_background(
    source_rgb: RgbFrame,
    replacement_rgb: RgbFrame,
    source_alpha: AlphaPlane,
    recovery: TemporalRecovery,
    *,
    background_threshold: int = SOURCE_BACKGROUND_THRESHOLD,
    residual_threshold: int | None = None,
) -> FrameBackground:
    """The v6 background plate for one frame. Pure.

    Own-frame background and temporally recovered pixels are v5's, byte for
    byte; ``temporal_unrecovered`` pixels are spatially filled from trusted
    real background where a component has a trusted neighbour, and hold the
    replacement (O1) pixel otherwise. The old person never survives.

    ``residual_threshold`` above ``background_threshold`` runs the tier-2
    pass (:func:`recover_residual`) over what the fill left, so that only
    its ``residual_2`` holds O1. ``None`` (the default) or a value not above
    ``background_threshold`` leaves tier 2 off, and the result is v6's, byte
    for byte.
    """
    _check_rgb("replacement_rgb", replacement_rgb)
    if replacement_rgb.shape != source_rgb.shape:
        raise CompositeError(
            f"replacement shape {replacement_rgb.shape} != source shape {source_rgb.shape}"
        )
    if residual_threshold is not None:
        _check_int_range("residual_threshold", residual_threshold, 1, 255)
    plate, recovered, unrecovered = temporal_plate(source_rgb, recovery)
    trusted = trusted_background(
        source_alpha, recovered, background_threshold=background_threshold
    )
    fill = spatial_fill_components(plate, trusted, unrecovered)
    tier1_residual: BoolMask = unrecovered & ~fill.filled
    tier2: ResidualFill | None = None
    if residual_threshold is not None and residual_threshold > background_threshold:
        tier2 = recover_residual(
            fill.rgb,
            source_rgb,
            source_alpha,
            trusted | fill.filled,
            tier1_residual,
            background_threshold=background_threshold,
            residual_threshold=residual_threshold,
        )
        background: RgbFrame = tier2.rgb.copy()
        residual: BoolMask = tier2.residual
    else:
        background = fill.rgb.copy()
        residual = tier1_residual
    background[residual] = replacement_rgb[residual]
    return FrameBackground(
        background=background,
        temporal_recovered=recovered,
        temporal_unrecovered=unrecovered,
        fill=fill,
        tier1_residual=tier1_residual,
        tier2=tier2,
        residual=residual,
    )


def _adjacent(mask: BoolMask) -> BoolMask:
    """Pixels with at least one 8-neighbour in ``mask`` (the mask itself included)."""
    out = mask.copy()
    height, width = mask.shape
    for dy, dx in _NEIGHBOURS:
        to, frm = _shifted(dy, dx, height, width)
        out[to] |= mask[frm]
    return out


def _wave_candidates(frontier: NDArray[np.int64], unresolved: BoolMask) -> NDArray[np.int64]:
    """Flat indices of unresolved pixels 8-adjacent to ``frontier``, unique and sorted.

    ``frontier`` holds flat indices into ``unresolved``'s shape. Neighbours
    outside the image are dropped. Duplicates (a pixel next to several
    frontier pixels) are collapsed, so each candidate is evaluated once, in
    a fixed order.
    """
    height, width = unresolved.shape
    fy, fx = np.divmod(frontier, width)
    ny = (fy[:, None] + _DY[None, :]).ravel()
    nx = (fx[:, None] + _DX[None, :]).ravel()
    inside = (ny >= 0) & (ny < height) & (nx >= 0) & (nx < width)
    flat = ny[inside] * width + nx[inside]
    flat = flat[unresolved.reshape(-1)[flat]]
    unique: NDArray[np.int64] = np.unique(flat).astype(np.int64)
    return unique


def _fill_candidates(
    values: NDArray[np.int64], resolved: BoolMask, candidates: NDArray[np.int64]
) -> NDArray[np.int64]:
    """``(n, 3)`` rounded-half-up means of each candidate's resolved 8-neighbours.

    Reads ``resolved`` and ``values`` as they are — the caller commits the
    result afterwards, so every candidate of one wave sees the same state.
    Neighbours outside the image never contribute.
    """
    height, width = resolved.shape
    cy, cx = np.divmod(candidates, width)
    ny = cy[:, None] + _DY[None, :]
    nx = cx[:, None] + _DX[None, :]
    inside = (ny >= 0) & (ny < height) & (nx >= 0) & (nx < width)
    ny_c = np.clip(ny, 0, height - 1)
    nx_c = np.clip(nx, 0, width - 1)
    contributes = inside & resolved[ny_c, nx_c]
    counts = contributes.sum(axis=1, dtype=np.int64)
    if bool((counts == 0).any()):
        raise CompositeError("candidate without a resolved neighbour")
    sums = (values[ny_c, nx_c] * contributes[:, :, None]).sum(axis=1, dtype=np.int64)
    means: NDArray[np.int64] = (2 * sums + counts[:, None]) // (2 * counts[:, None])
    return means


def _shifted(
    dy: int, dx: int, height: int, width: int
) -> tuple[tuple[slice, slice], tuple[slice, slice]]:
    """Index pairs so that ``out[to] op= src[frm]`` reads the neighbour at ``(dy, dx)``."""
    rows_to = slice(max(-dy, 0), height - max(dy, 0))
    rows_from = slice(max(dy, 0), height - max(-dy, 0))
    cols_to = slice(max(-dx, 0), width - max(dx, 0))
    cols_from = slice(max(dx, 0), width - max(-dx, 0))
    return (rows_to, cols_to), (rows_from, cols_from)


def _component_windows(labels: NDArray[np.int32], count: int) -> list[tuple[slice, slice]]:
    """Bounding box + margin of each label ``1..count``, clipped to the frame."""
    if count == 0:
        return []
    height, width = labels.shape
    ys, xs = np.nonzero(labels)
    index = labels[ys, xs] - 1
    y0 = np.full(count, height, dtype=np.int64)
    y1 = np.full(count, -1, dtype=np.int64)
    x0 = np.full(count, width, dtype=np.int64)
    x1 = np.full(count, -1, dtype=np.int64)
    np.minimum.at(y0, index, ys)
    np.maximum.at(y1, index, ys)
    np.minimum.at(x0, index, xs)
    np.maximum.at(x1, index, xs)
    m = SPATIAL_FILL_MARGIN
    return [
        (
            slice(max(int(y0[k]) - m, 0), min(int(y1[k]) + m + 1, height)),
            slice(max(int(x0[k]) - m, 0), min(int(x1[k]) + m + 1, width)),
        )
        for k in range(count)
    ]


def _check_rgb(name: str, rgb: NDArray[np.uint8]) -> None:
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise CompositeError(f"{name} must be (H, W, 3) uint8, got {rgb.shape} {rgb.dtype}")


def _check_alpha(name: str, alpha: AlphaPlane) -> None:
    if alpha.ndim != 2 or alpha.dtype != np.uint8:
        raise CompositeError(f"{name} must be (H, W) uint8, got {alpha.shape} {alpha.dtype}")


def _check_mask(name: str, mask: BoolMask, shape: tuple[int, ...] | None = None) -> None:
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise CompositeError(f"{name} must be a 2-D bool mask, got {mask.shape} {mask.dtype}")
    if shape is not None and mask.shape != tuple(shape):
        raise CompositeError(f"{name} shape {mask.shape} != {tuple(shape)}")


def _add_depths(hist: NDArray[np.int64], depths: NDArray[np.int32]) -> NDArray[np.int64]:
    """``hist`` with a bincount of ``depths`` added, grown to fit."""
    counts = np.bincount(depths)
    if counts.size > hist.size:
        hist = np.concatenate([hist, np.zeros(counts.size - hist.size, dtype=np.int64)])
    hist[: counts.size] += counts
    return hist


# -- streaming -----------------------------------------------------------


def composite_streams_spatial_recovery(
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
    residual_threshold: int | None = None,
    replacement_filter: ReplacementFilter | None = None,
) -> SpatialRecoveryStreamStats:
    """Composite ``frames`` frames with temporal then spatial background recovery.

    Same streams, read-ahead window, parameters and failure modes as
    :func:`video_character_skill.temporal_recovery.composite_streams_temporal_recovery`;
    the per-frame algorithm is the module docstring's. ``residual_threshold``
    above ``background_threshold`` enables the tier-2 residual pass
    (:func:`recover_residual`); ``None`` keeps v6 byte for byte.
    ``replacement_filter``, when given, rewrites each replacement RGB frame
    (never its alpha) before it is used — see :data:`ReplacementFilter`.

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
    if residual_threshold is not None:
        _check_int_range("residual_threshold", residual_threshold, 1, 255)

    window = _SourceWindow(
        source, source_matte, width=width, height=height, frames=frames, radius=radius
    )
    size = float(width * height)
    soft_total = region_total = own_total = recovered_total = unrecovered_total = 0.0
    spatial_total = residual_total = fallback_total = 0.0
    frames_with_region = 0
    distance_hist = np.zeros(radius + 1, dtype=np.int64)
    depth_hist: NDArray[np.int64] = np.zeros(1, dtype=np.int64)
    recovered_pixels = 0
    observations_used = 0
    fits = 0
    fallbacks = 0
    components = 0
    without_seed = 0
    tier1_residual_total = accepted_total = filled2_total = 0.0
    depth2_hist: NDArray[np.int64] = np.zeros(1, dtype=np.int64)
    tier2_components = 0
    tier2_without_seed = 0

    for index in range(frames):
        window.advance(index)
        replacement_frame = _read_rgb_frame(replacement, "replacement", index, width, height)
        replacement_alpha = _read_alpha_frame(
            replacement_matte, "replacement_matte", index, width, height
        )
        if replacement_filter is not None:
            replacement_frame = replacement_filter(replacement_frame, replacement_alpha)
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
        result = recover_frame_background(
            source_frame,
            replacement_frame,
            source_alpha,
            recovery,
            background_threshold=background_threshold,
            residual_threshold=residual_threshold,
        )
        alpha = recovery_effective_alpha(replacement_alpha, regions.force_replacement)
        output.write(composite_frame(result.background, replacement_frame, alpha).tobytes())

        region_count = int(np.count_nonzero(regions.recovery_region))
        got = int(np.count_nonzero(result.temporal_recovered))
        missing = int(np.count_nonzero(result.temporal_unrecovered))
        spatial = int(np.count_nonzero(result.fill.filled))
        residual = int(np.count_nonzero(result.residual))
        soft_total += soft_edge_ratio(alpha)
        region_total += region_count / size
        own_total += float(np.count_nonzero(regions.own_background)) / size
        recovered_total += got / size
        unrecovered_total += missing / size
        spatial_total += spatial / size
        residual_total += residual / size
        if region_count:
            frames_with_region += 1
            fallback_total += residual / region_count
        if recovery.distances.size:
            distance_hist += np.bincount(recovery.distances, minlength=radius + 1)[: radius + 1]
        if spatial:
            depth_hist = _add_depths(depth_hist, result.fill.depth[result.fill.filled])
        tier1_residual_total += float(np.count_nonzero(result.tier1_residual)) / size
        if result.tier2 is not None:
            accepted_total += float(np.count_nonzero(result.tier2.accepted)) / size
            filled2 = int(np.count_nonzero(result.tier2.filled))
            filled2_total += filled2 / size
            tier2_components += result.tier2.components
            tier2_without_seed += result.tier2.components_without_seed
            if filled2:
                depth2_hist = _add_depths(depth2_hist, result.tier2.depth[result.tier2.filled])
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
    return SpatialRecoveryStreamStats(
        soft_edge_ratio=soft_total / frames if frames else 0.0,
        recovery_region_ratio=region_total / frames if frames else 0.0,
        own_background_ratio=own_total / frames if frames else 0.0,
        temporal_recovered_ratio=recovered_total / frames if frames else 0.0,
        temporal_unrecovered_ratio=unrecovered_total / frames if frames else 0.0,
        spatial_recovered_ratio=spatial_total / frames if frames else 0.0,
        spatial_unrecovered_ratio=residual_total / frames if frames else 0.0,
        o1_fallback_ratio=fallback_total / frames_with_region if frames_with_region else 0.0,
        spatial_components=components,
        components_without_seed=without_seed,
        median_propagation_depth=_hist_percentile(depth_hist, 50),
        p90_propagation_depth=_hist_percentile(depth_hist, 90),
        max_propagation_depth=int(np.flatnonzero(depth_hist).max()) if depth_hist.any() else 0,
        median_donor_distance=_hist_percentile(distance_hist, 50),
        p90_donor_distance=_hist_percentile(distance_hist, 90),
        mean_observations_per_recovered_pixel=(
            observations_used / recovered_pixels if recovered_pixels else nan
        ),
        donor_fits=fits,
        zero_offset_fallbacks=fallbacks,
        peak_cached_frames=window.peak,
        tier1_residual_ratio=tier1_residual_total / frames if frames else 0.0,
        tier2_accepted_ratio=accepted_total / frames if frames else 0.0,
        tier2_filled_ratio=filled2_total / frames if frames else 0.0,
        tier2_residual_ratio=residual_total / frames if frames else 0.0,
        tier2_components=tier2_components,
        tier2_components_without_seed=tier2_without_seed,
        median_tier2_depth=_hist_percentile(depth2_hist, 50) if depth2_hist.any() else math.nan,
        p90_tier2_depth=_hist_percentile(depth2_hist, 90) if depth2_hist.any() else math.nan,
        max_tier2_depth=int(np.flatnonzero(depth2_hist).max()) if depth2_hist.any() else 0,
    )


# -- orchestration -------------------------------------------------------


def composite_video_spatial_recovery(
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
) -> SpatialRecoveryCompositeReport:
    """Composite four clips into one H.264 MP4 with temporal + spatial recovery.

    Same four inputs, fail-closed validation, ``libvpx-vp9`` alpha decoding
    and ffmpeg lifecycle as
    :func:`video_character_skill.temporal_recovery.composite_video_temporal_recovery`;
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
        stats = composite_streams_spatial_recovery(
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

    return SpatialRecoveryCompositeReport(
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
