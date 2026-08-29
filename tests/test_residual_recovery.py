"""Unit tests for the v11 two-tier residual recovery.

Tier 1 is v10 (``background_threshold=1``: only ``source_alpha == 0`` is
trusted); the tier-2 pass reworks only the components tier 1 left for O1.
Pure helpers run on hand-built frames, the streaming loop on file-like
objects, the orchestrator against the fake ``Popen`` shared with the v2-v9
tests. Nothing here decodes, encodes or writes a video.
"""

from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from test_compositor_union import FakeFfmpeg, H, W, default_outputs, fake_probe, plane, wire
from test_rim_correction import TinyRimFilter, assert_stats_equal
from test_temporal_recovery import (
    BG,
    FULL_FIT,
    O1_BG,
    O1_PERSON,
    PERSON,
    Frame,
    P,
    clip_streams,
    pixel,
    replacement_frame,
    solid,
    source_frame,
)
from video_character_skill.compositor import CompositeError, VideoInfo
from video_character_skill.rim_correction import (
    RimCorrectedCompositeReport,
    composite_streams_rim_corrected,
    composite_video_rim_corrected,
)
from video_character_skill.spatial_recovery import (
    FrameBackground,
    ResidualFill,
    SpatialRecoveryStreamStats,
    composite_streams_spatial_recovery,
    recover_frame_background,
    recover_residual,
)
from video_character_skill.temporal_recovery import TemporalRecovery, recovery_regions

Rgb = tuple[int, int, int]
Bool = NDArray[np.bool_]

RING: Rgb = (60, 60, 60)  # source colour of the low-alpha pixels around the old person
SOFT: Rgb = (40, 40, 40)  # source colour of a low-alpha pixel inside the component
LOW = 9  # a source alpha inside the low band 1..31
BLOCK = 40  # a source alpha that is neither trusted (>= 1) nor low (>= 32)
V10: dict[str, Any] = {"background_threshold": 1}
V11: dict[str, Any] = {"background_threshold": 1, "residual_threshold": 32}
TIER1_FIELDS = (
    "soft_edge_ratio", "recovery_region_ratio", "own_background_ratio",
    "temporal_recovered_ratio", "temporal_unrecovered_ratio", "spatial_recovered_ratio",
    "spatial_components", "components_without_seed", "median_propagation_depth",
    "p90_propagation_depth", "max_propagation_depth", "median_donor_distance",
    "p90_donor_distance", "mean_observations_per_recovered_pixel", "donor_fits",
    "zero_offset_fallbacks", "peak_cached_frames",
)


# -- pure-frame builders (6x7) ------------------------------------------------------------


INSIDE = [(2, 2), (2, 3), (2, 4)]  # the old-person pixels tier 1 cannot recover, raster order
SHAPE = (6, 7)


def unrecovered(shape: tuple[int, int], pixels: list[tuple[int, int]]) -> TemporalRecovery:
    """A temporal recovery that observed nothing for ``pixels``."""
    n = len(pixels)
    return TemporalRecovery(
        indices=np.array([y * shape[1] + x for y, x in pixels], dtype=np.intp),
        rgb=np.zeros((n, 3), dtype=np.uint8),
        counts=np.zeros(n, dtype=np.int64),
        distances=np.zeros(0, dtype=np.int64),
        donor_fits=0,
        zero_offset_fallbacks=0,
    )


def scene(
    alphas: tuple[int, int, int], *, ring_alpha: int = LOW
) -> tuple[NDArray[np.uint8], NDArray[np.uint8], TemporalRecovery]:
    """Source frame: real background (alpha 0) around a 3x5 block of alpha ``ring_alpha``
    whose middle row holds the three unrecovered pixels with the given alphas."""
    rgb = np.full((*SHAPE, 3), BG, dtype=np.uint8)
    alpha = np.zeros(SHAPE, dtype=np.uint8)
    alpha[1:4, 1:6] = ring_alpha
    rgb[1:4, 1:6] = RING
    for (y, x), a in zip(INSIDE, alphas, strict=True):
        alpha[y, x] = a
        rgb[y, x] = SOFT if a < 32 else PERSON
    return rgb, alpha, unrecovered(SHAPE, INSIDE)


def recover(
    rgb: NDArray[np.uint8], alpha: NDArray[np.uint8], recovery: TemporalRecovery, **params: Any
) -> FrameBackground:
    o1 = np.full(rgb.shape, O1_BG, dtype=np.uint8)
    return recover_frame_background(rgb, o1, alpha, recovery, **params)


def where(mask: Bool) -> set[tuple[int, int]]:
    return {(int(y), int(x)) for y, x in zip(*np.nonzero(mask), strict=True)}


def rgb_at(frame: NDArray[np.uint8], at: tuple[int, int]) -> Rgb:
    r, g, b = (int(v) for v in frame[at])
    return (r, g, b)


# -- 1, 2: disabled and no-op paths are byte-identical ---------------------------------------------


def test_none_is_the_default_and_reproduces_v10_byte_for_byte() -> None:
    rgb, alpha, rec = scene((20, 50, 50))
    v10 = recover(rgb, alpha, rec, **V10)
    explicit = recover(rgb, alpha, rec, background_threshold=1, residual_threshold=None)
    np.testing.assert_array_equal(explicit.background, v10.background)
    np.testing.assert_array_equal(explicit.residual, v10.residual)
    assert v10.tier2 is None and explicit.tier2 is None
    np.testing.assert_array_equal(v10.tier1_residual, v10.residual)
    assert where(v10.residual) == set(INSIDE)  # v10: the whole component is O1
    for at in INSIDE:
        assert rgb_at(v10.background, at) == O1_BG
    defaults = recover_frame_background.__kwdefaults__
    assert defaults is not None and defaults["residual_threshold"] is None


@pytest.mark.parametrize(("background", "residual"), [(1, 1), (32, 32), (32, 16), (32, 1)])
def test_a_threshold_not_above_background_is_a_byte_identical_no_op(
    background: int, residual: int
) -> None:
    rgb, alpha, rec = scene((50, 50, 50), ring_alpha=BLOCK)  # a residual at every threshold
    off = recover(rgb, alpha, rec, background_threshold=background)
    assert where(off.tier1_residual) == set(INSIDE)
    on = recover(rgb, alpha, rec, background_threshold=background, residual_threshold=residual)
    np.testing.assert_array_equal(on.background, off.background)
    np.testing.assert_array_equal(on.residual, off.residual)
    assert on.tier2 is None
    # the pure pass itself: an empty low band accepts and fills nothing
    resolved = alpha == 0
    pure = recover_residual(
        rgb, rgb, alpha, resolved, off.tier1_residual,
        background_threshold=background, residual_threshold=residual,
    )
    np.testing.assert_array_equal(pure.rgb, rgb)
    assert not pure.accepted.any() and not pure.filled.any()
    np.testing.assert_array_equal(pure.residual, off.tier1_residual)
    assert pure.components == 1 and pure.components_without_seed == 1


# -- 3, 4: tier 1 is untouched; only residual_1 pixels may change  ---------------------------------


def test_tier1_masks_and_rgb_are_unchanged_and_only_residual_1_pixels_differ() -> None:
    rgb, alpha, rec = scene((20, 50, 50))
    # a second, tier-1-fillable component: an old-person pixel next to real background
    alpha[5, 6] = 200
    rgb[5, 6] = PERSON
    rec = unrecovered(SHAPE, [*INSIDE, (5, 6)])
    off = recover(rgb, alpha, rec, **V10)
    on = recover(rgb, alpha, rec, **V11)

    for name in ("temporal_recovered", "temporal_unrecovered", "tier1_residual"):
        np.testing.assert_array_equal(getattr(on, name), getattr(off, name))
    np.testing.assert_array_equal(on.fill.filled, off.fill.filled)
    np.testing.assert_array_equal(on.fill.rgb, off.fill.rgb)
    np.testing.assert_array_equal(on.fill.depth, off.fill.depth)
    assert off.fill.filled[5, 6] and rgb_at(on.background, (5, 6)) == BG  # tier-1 fill kept
    changed = np.any(on.background != off.background, axis=2)
    assert changed.any() and (changed <= off.tier1_residual).all()
    assert not on.residual.any()
    assert isinstance(on.tier2, ResidualFill)
    outside = ~off.tier1_residual
    np.testing.assert_array_equal(on.tier2.rgb[outside], off.fill.rgb[outside])


# -- 5, 6: S_inside is copied from the source and never filled; inside seeds fill all T ----------


def test_inside_seeds_are_accepted_source_pixels_and_fill_every_reachable_t() -> None:
    rgb, alpha, rec = scene((20, 50, 50))
    on = recover(rgb, alpha, rec, **V11)
    t2 = on.tier2
    assert t2 is not None
    assert where(t2.accepted) == {(2, 2)} and not t2.filled[2, 2]
    assert rgb_at(on.background, (2, 2)) == SOFT == rgb_at(rgb, (2, 2))
    assert where(t2.filled) == {(2, 3), (2, 4)}
    # (2,3): six ring neighbours at 60 and the accepted pixel at 40 -> 400/7 -> 57 (half up)
    assert rgb_at(on.background, (2, 3)) == (57, 57, 57)
    # (2,4): seven ring neighbours at 60; (2,3) is not yet resolved in wave 1
    assert rgb_at(on.background, (2, 4)) == RING
    assert t2.depth[2, 3] == 1 and t2.depth[2, 4] == 1 and t2.depth[2, 2] == 0
    assert not t2.residual.any() and not on.residual.any()
    assert t2.components == 1 and t2.components_without_seed == 0
    # SOFT is no wave value, so the accepted pixel was copied, not filled
    assert SOFT not in ((57, 57, 57), RING)


# -- 7: external ring seeds alone ------------------------------------------------------------------


def test_ring_seeds_alone_fill_t_with_their_source_rgb() -> None:
    rgb, alpha, rec = scene((50, 50, 50))
    on = recover(rgb, alpha, rec, **V11)
    t2 = on.tier2
    assert t2 is not None
    assert not t2.accepted.any() and where(t2.filled) == set(INSIDE)
    for at in INSIDE:
        assert rgb_at(on.background, at) == RING and t2.depth[at] == 1
    assert not on.residual.any()


def test_a_single_ring_seed_propagates_wave_by_wave() -> None:
    rgb, alpha, rec = scene((50, 50, 50), ring_alpha=BLOCK)
    alpha[2, 1] = LOW  # the only low pixel: left of the row
    on = recover(rgb, alpha, rec, **V11)
    t2 = on.tier2
    assert t2 is not None
    assert where(t2.filled) == set(INSIDE) and not on.residual.any()
    assert [int(t2.depth[at]) for at in INSIDE] == [1, 2, 3]
    for at in INSIDE:
        assert rgb_at(on.background, at) == RING


# -- 8: several T sub-components around one inside seed -------------------------------------------


def test_every_t_subcomponent_next_to_an_inside_seed_is_filled() -> None:
    rgb, alpha, rec = scene((50, 20, 50), ring_alpha=BLOCK)  # T, S, T with no ring seed
    on = recover(rgb, alpha, rec, **V11)
    t2 = on.tier2
    assert t2 is not None
    assert where(t2.accepted) == {(2, 3)} and where(t2.filled) == {(2, 2), (2, 4)}
    assert rgb_at(on.background, (2, 2)) == SOFT and rgb_at(on.background, (2, 4)) == SOFT
    assert rgb_at(on.background, (2, 3)) == SOFT
    assert not on.residual.any() and t2.components == 1 and t2.components_without_seed == 0


# -- 9: no seed at all -> O1  ----------------------------------------------------------------------


def test_a_component_without_any_tier2_seed_stays_o1() -> None:
    rgb, alpha, rec = scene((50, 50, 50), ring_alpha=BLOCK)
    off = recover(rgb, alpha, rec, **V10)
    on = recover(rgb, alpha, rec, **V11)
    t2 = on.tier2
    assert t2 is not None
    assert not t2.accepted.any() and not t2.filled.any()
    assert where(t2.residual) == set(INSIDE) and where(on.residual) == set(INSIDE)
    assert t2.components == 1 and t2.components_without_seed == 1
    np.testing.assert_array_equal(on.background, off.background)
    for at in INSIDE:
        assert rgb_at(on.background, at) == O1_BG


def test_residual_2_is_exactly_the_unseeded_components() -> None:
    shape = (6, 11)
    rgb = np.full((*shape, 3), BG, dtype=np.uint8)
    alpha = np.zeros(shape, dtype=np.uint8)
    alpha[1:4, 1:6] = LOW  # left block: low ring
    alpha[1:4, 6:11] = BLOCK  # right block: blocking ring
    rgb[1:4, 1:11] = RING
    left = [(2, 2), (2, 3), (2, 4)]
    right = [(2, 7), (2, 8), (2, 9)]
    for at in (*left, *right):
        alpha[at] = 200
        rgb[at] = PERSON
    rec = unrecovered(shape, [*left, *right])
    o1 = np.full(rgb.shape, O1_BG, dtype=np.uint8)
    on = recover_frame_background(rgb, o1, alpha, rec, **V11)
    t2 = on.tier2
    assert t2 is not None
    assert where(on.tier1_residual) == set(left) | set(right)
    assert t2.components == 2 and t2.components_without_seed == 1
    assert where(t2.filled) == set(left) and where(t2.residual) == set(right)
    np.testing.assert_array_equal(on.residual, t2.residual)
    for at in left:
        assert rgb_at(on.background, at) == RING
    for at in right:
        assert rgb_at(on.background, at) == O1_BG


def test_recover_residual_checks_its_inputs() -> None:
    rgb, alpha, rec = scene((20, 50, 50))
    off = recover(rgb, alpha, rec, **V10)
    resolved = alpha == 0
    good = dict(background_threshold=1, residual_threshold=32)
    with pytest.raises(CompositeError, match="overlap"):
        recover_residual(rgb, rgb, alpha, resolved | off.tier1_residual, off.tier1_residual, **good)
    bad_thresholds: list[dict[str, int]] = [
        {"background_threshold": 0}, {"residual_threshold": 0}, {"residual_threshold": 256},
    ]
    for bad in bad_thresholds:
        with pytest.raises(CompositeError):
            recover_residual(rgb, rgb, alpha, resolved, off.tier1_residual, **{**good, **bad})
    with pytest.raises(CompositeError):
        recover_residual(rgb, rgb[:, :1], alpha, resolved, off.tier1_residual, **good)
    with pytest.raises(CompositeError):
        recover_residual(rgb, rgb, alpha[:1], resolved, off.tier1_residual, **good)
    for bad_threshold in (0, 256):
        with pytest.raises(CompositeError):
            recover(rgb, alpha, rec, background_threshold=1, residual_threshold=bad_threshold)


# -- streaming scenes (4x5) ------------------------------------------------------------------------


def run(
    sources: list[Frame], replacements: list[Frame], **overrides: Any
) -> tuple[NDArray[np.uint8], SpatialRecoveryStreamStats]:
    streams = clip_streams(sources, replacements)
    out = io.BytesIO()
    stats = composite_streams_spatial_recovery(
        streams["source"], streams["replacement"], streams["source_matte"],
        streams["replacement_matte"], out, width=W, height=H, frames=len(sources),
        **{**FULL_FIT, **overrides},
    )
    return np.frombuffer(out.getvalue(), np.uint8).reshape(len(sources), H, W, 3), stats


def ring_scene(ring_alpha: int = LOW) -> tuple[list[Frame], list[Frame]]:
    """The old person at P, its eight neighbours at ``ring_alpha``, real background elsewhere.
    Removal (threshold 64, no dilation) is P alone; no donor ever sees background there."""
    frames = []
    for _ in range(3):
        rgb, a = source_frame(P)
        a[0:3, 0:3] = ring_alpha
        rgb[0:3, 0:3] = RING
        a[P] = 200
        rgb[P] = PERSON
        frames.append((rgb, a))
    return frames, [replacement_frame()] * 3


def split_scene(ring_alpha: int = LOW, soft_alpha: int = 20) -> tuple[list[Frame], list[Frame]]:
    """Removal (threshold 64, dilation 1) is the plus around P. Inside it P is the old person
    (T) and (1, 2) a soft pixel (S); the other three plus pixels are solid replacement person
    (force_replacement). Every other 8-neighbour of the pair sits outside the removal mask at
    ``ring_alpha``, so tier 1 has no seed."""
    sources, replacements = [], []
    for _ in range(3):
        rgb, a = source_frame(P)
        a[0:3, 0:4] = ring_alpha
        rgb[0:3, 0:4] = RING
        a[P] = 200
        rgb[P] = PERSON
        a[1, 2] = soft_alpha
        rgb[1, 2] = SOFT
        sources.append((rgb, a))
        o1, oa = replacement_frame()
        for at in ((0, 1), (1, 0), (2, 1)):
            oa[at] = 255
            o1[at] = O1_PERSON
        replacements.append((o1, oa))
    return sources, replacements


def row_scene() -> tuple[list[Frame], list[Frame]]:
    """Three old-person pixels in a row, walled by blocking alpha except one low pixel at
    the left end: tier 2 must propagate wave by wave."""
    frames = []
    for _ in range(3):
        rgb, a = source_frame()
        a[0:3, 0:5] = BLOCK
        rgb[0:3, 0:5] = RING
        a[1, 0] = LOW
        for x in (1, 2, 3):
            a[1, x] = 200
            rgb[1, x] = PERSON
        frames.append((rgb, a))
    return frames, [replacement_frame()] * 3


def rim_run(
    sources: list[Frame], replacements: list[Frame], **overrides: Any
) -> tuple[NDArray[np.uint8], SpatialRecoveryStreamStats, Any]:
    streams = clip_streams(sources, replacements)
    out = io.BytesIO()
    stats, rim = composite_streams_rim_corrected(
        streams["source"], streams["replacement"], streams["source_matte"],
        streams["replacement_matte"], out, width=W, height=H, frames=len(sources), **overrides,
    )
    return np.frombuffer(out.getvalue(), np.uint8).reshape(len(sources), H, W, 3), stats, rim


# -- 11: v6/v7/v9/v10 are unchanged with tier 2 off; 2: exact no-op below the background threshold


@pytest.mark.parametrize(
    "params", [{}, {"removal_threshold": 32}, V10, {"removal_threshold": 32, **V10}],
    ids=["v6", "v7", "v10", "v7+v10"],
)
def test_streams_with_tier2_disabled_are_byte_identical(params: dict[str, Any]) -> None:
    sources, replacements = ring_scene()
    frames_a, stats_a = run(sources, replacements, **params)
    frames_b, stats_b = run(sources, replacements, residual_threshold=None, **params)
    np.testing.assert_array_equal(frames_a, frames_b)
    assert_stats_equal(stats_a, stats_b)
    assert stats_a.tier2_components == 0 and stats_a.tier2_filled_ratio == 0.0
    assert stats_a.tier1_residual_ratio == stats_a.spatial_unrecovered_ratio
    assert stats_a.tier2_residual_ratio == stats_a.spatial_unrecovered_ratio
    assert math.isnan(stats_a.median_tier2_depth) and stats_a.max_tier2_depth == 0


def test_v9_rim_path_is_byte_identical_with_tier2_disabled() -> None:
    sources, replacements = ring_scene()
    frames_a, stats_a, rim_a = rim_run(sources, replacements)
    frames_b, stats_b, rim_b = rim_run(sources, replacements, residual_threshold=None)
    np.testing.assert_array_equal(frames_a, frames_b)
    assert_stats_equal(stats_a, stats_b)
    assert rim_a == rim_b
    for function in (
        recover_frame_background, composite_streams_spatial_recovery,
        composite_streams_rim_corrected, composite_video_rim_corrected,
    ):
        defaults = function.__kwdefaults__
        assert defaults is not None and defaults["residual_threshold"] is None


@pytest.mark.parametrize(("background", "residual"), [(1, 1), (32, 32), (32, 16)])
def test_streams_are_a_byte_identical_no_op_below_the_background_threshold(
    background: int, residual: int
) -> None:
    sources, replacements = ring_scene()
    frames_a, stats_a = run(sources, replacements, background_threshold=background)
    frames_b, stats_b = run(
        sources, replacements, background_threshold=background, residual_threshold=residual
    )
    np.testing.assert_array_equal(frames_a, frames_b)
    assert_stats_equal(stats_a, stats_b)


# -- 7, 10, 3: ring seeds through the stream; the three residual reports agree  --------------------


def test_stream_ring_seeds_fill_the_residual_and_the_stats_describe_it() -> None:
    sources, replacements = ring_scene()
    off, off_stats = run(sources, replacements, **V10)
    on, on_stats = run(sources, replacements, **V11)

    expected = solid(BG)
    expected[0:3, 0:3] = RING  # the ring is source, outside the removal mask
    for i in range(3):
        np.testing.assert_array_equal(on[i], expected)
        assert pixel(off, i) == O1_BG  # v10: O1 fallback
    changed = np.any(on != off, axis=3)
    assert all(where(changed[i]) == {P} for i in range(3))

    assert off_stats.spatial_unrecovered_ratio == pytest.approx(1 / (H * W))
    assert off_stats.o1_fallback_ratio == 1.0
    assert on_stats.tier1_residual_ratio == pytest.approx(1 / (H * W))
    assert on_stats.tier2_filled_ratio == pytest.approx(1 / (H * W))
    assert on_stats.tier2_accepted_ratio == 0.0
    assert on_stats.tier2_residual_ratio == on_stats.spatial_unrecovered_ratio == 0.0
    assert on_stats.o1_fallback_ratio == 0.0
    assert on_stats.tier2_components == 3 and on_stats.tier2_components_without_seed == 0
    assert (on_stats.median_tier2_depth, on_stats.p90_tier2_depth) == (1.0, 1.0)
    assert on_stats.max_tier2_depth == 1
    for name in TIER1_FIELDS:  # tier 1 reported identically
        a, b = getattr(on_stats, name), getattr(off_stats, name)
        assert (math.isnan(a) and math.isnan(b)) or a == b, name
    assert on_stats.tier1_residual_ratio == off_stats.spatial_unrecovered_ratio


def test_stream_accepts_inside_seeds_and_fills_t() -> None:
    sources, replacements = split_scene()
    off, off_stats = run(sources, replacements, dilation_radius=1, **V10)
    on, on_stats = run(sources, replacements, dilation_radius=1, **V11)

    expected = solid(BG)
    expected[0:3, 0:4] = RING
    for at in ((0, 1), (1, 0), (2, 1)):
        expected[at] = O1_PERSON  # force_replacement: alpha semantics unchanged
    expected[1, 2] = SOFT  # accepted S: own-frame source RGB
    expected[P] = (58, 58, 58)  # seven ring seeds at 60 and the accepted 40: 460/8 -> 58
    for i in range(3):
        np.testing.assert_array_equal(on[i], expected)
        assert pixel(off, i) == O1_BG and pixel(off, i, (1, 2)) == O1_BG
    changed = np.any(on != off, axis=3)
    assert all(where(changed[i]) == {P, (1, 2)} for i in range(3))
    assert on_stats.tier1_residual_ratio == pytest.approx(2 / (H * W))
    assert on_stats.tier2_accepted_ratio == pytest.approx(1 / (H * W))
    assert on_stats.tier2_filled_ratio == pytest.approx(1 / (H * W))
    assert on_stats.spatial_unrecovered_ratio == 0.0 and on_stats.o1_fallback_ratio == 0.0
    assert on_stats.soft_edge_ratio == off_stats.soft_edge_ratio


def test_stream_seedless_component_is_o1_and_the_three_residual_reports_agree() -> None:
    sources, replacements = ring_scene(ring_alpha=BLOCK)
    off, off_stats = run(sources, replacements, **V10)
    on, on_stats = run(sources, replacements, **V11)
    np.testing.assert_array_equal(on, off)
    assert all(pixel(on, i) == O1_BG for i in range(3))
    assert on_stats.tier1_residual_ratio == pytest.approx(1 / (H * W))
    assert on_stats.tier2_residual_ratio == on_stats.spatial_unrecovered_ratio
    assert on_stats.spatial_unrecovered_ratio == pytest.approx(1 / (H * W))
    assert on_stats.o1_fallback_ratio == 1.0
    assert on_stats.tier2_components == 3 and on_stats.tier2_components_without_seed == 3
    assert on_stats.tier2_accepted_ratio == 0.0 and on_stats.tier2_filled_ratio == 0.0
    assert math.isnan(on_stats.median_tier2_depth) and on_stats.max_tier2_depth == 0


def test_stream_tier2_depth_diagnostics() -> None:
    sources, replacements = row_scene()
    on, stats = run(sources, replacements, **V11)
    for i in range(3):
        for x in (1, 2, 3):
            assert pixel(on, i, (1, x)) == RING
    assert (stats.median_tier2_depth, stats.p90_tier2_depth, stats.max_tier2_depth) == (2.0, 3.0, 3)
    assert stats.tier2_filled_ratio == pytest.approx(3 / (H * W))
    assert stats.spatial_unrecovered_ratio == 0.0


# -- 12: wrappers forward and validate ------------------------------------------------------------


def test_stream_wrappers_validate_the_threshold_before_reading() -> None:
    sources, replacements = ring_scene()
    for bad in (0, 256, -1):
        streams = clip_streams(sources, replacements)
        with pytest.raises(CompositeError, match="residual_threshold"):
            composite_streams_spatial_recovery(
                streams["source"], streams["replacement"], streams["source_matte"],
                streams["replacement_matte"], io.BytesIO(), width=W, height=H, frames=3,
                residual_threshold=bad, **FULL_FIT,
            )
        assert all(s.tell() == 0 for s in streams.values())
        streams = clip_streams(sources, replacements)
        with pytest.raises(CompositeError, match="residual_threshold"):
            composite_streams_rim_corrected(
                streams["source"], streams["replacement"], streams["source_matte"],
                streams["replacement_matte"], io.BytesIO(), width=W, height=H, frames=3,
                residual_threshold=bad,
            )
        assert all(s.tell() == 0 for s in streams.values())


def test_rim_stream_forwards_the_threshold() -> None:
    sources, replacements = ring_scene()
    params: dict[str, Any] = {"dilation_radius": 0, **V11}
    on, on_stats, rim = rim_run(sources, replacements, **params)
    plain, plain_stats = run(sources, replacements, removal_threshold=32, **params)
    # the 4x5 frame holds no rim reference, so the rim path is the plain path
    np.testing.assert_array_equal(on, plain)
    assert rim.corrected_ratio == 0.0
    assert on_stats.tier2_filled_ratio == pytest.approx(1 / (H * W))
    assert all(pixel(on, i) == RING for i in range(3))
    off, _, _ = rim_run(sources, replacements, dilation_radius=0, **V10)
    assert all(pixel(off, i) == O1_BG for i in range(3))


def run_video(tmp_path: Path, **kwargs: Any) -> RimCorrectedCompositeReport:
    return composite_video_rim_corrected(
        tmp_path / "src.mp4", tmp_path / "rep.mp4", tmp_path / "src.webm", tmp_path / "rep.webm",
        tmp_path / "v11.mp4", **kwargs,
    )


def test_video_wrapper_validates_before_probing_and_reports_the_threshold(
    tmp_path: Path, monkeypatch: Any
) -> None:
    ffmpeg = FakeFfmpeg(default_outputs(2))

    def never(path: Path, decoder: str | None = None) -> VideoInfo:
        raise AssertionError(f"probed {path}")

    wire(monkeypatch, ffmpeg, never)
    for bad in (0, 256):
        with pytest.raises(CompositeError, match="residual_threshold"):
            run_video(tmp_path, residual_threshold=bad)
    assert ffmpeg.commands == []

    ffmpeg = FakeFfmpeg(default_outputs(2))
    wire(monkeypatch, ffmpeg, fake_probe(2))
    report = run_video(tmp_path, **V11)
    assert (report.background_threshold, report.residual_threshold) == (1, 32)
    assert isinstance(report.stats, SpatialRecoveryStreamStats)
    assert not any(p.killed for p in ffmpeg.processes)
    ffmpeg = FakeFfmpeg(default_outputs(2))
    wire(monkeypatch, ffmpeg, fake_probe(2))
    assert run_video(tmp_path).residual_threshold is None


# -- 13: alpha, removal geometry and the rim correction are unchanged  -----------------------------


def test_alpha_removal_geometry_and_rim_correction_are_unchanged() -> None:
    assert "residual_threshold" not in (recovery_regions.__kwdefaults__ or {})
    sources, replacements = ring_scene()
    for o1, oa in replacements:  # a 3x3 solid replacement block: a rim core with a reference
        oa[1:4, 2:5] = 255
        o1[1:4, 2:5] = np.arange(27, dtype=np.uint8).reshape(3, 3, 3) * 9
    rim_off = TinyRimFilter()
    rim_on = TinyRimFilter()
    off, off_stats = run(sources, replacements, replacement_filter=rim_off, **V10)
    on, on_stats = run(sources, replacements, replacement_filter=rim_on, **V11)
    assert rim_off.stats() == rim_on.stats() and rim_on.corrected_pixels > 0
    changed = np.any(on != off, axis=3)
    assert all(where(changed[i]) == {P} for i in range(3))  # the rim block is identical
    assert on_stats.soft_edge_ratio == off_stats.soft_edge_ratio
    assert on_stats.recovery_region_ratio == off_stats.recovery_region_ratio
    assert plane(0).shape == (H, W)
