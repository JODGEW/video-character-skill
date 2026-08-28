"""Unit tests for the spatial real-background fill compositor (v6).

Pure helpers are exercised on hand-built crops; the streaming loop runs on
file-like objects; the ffmpeg pipeline runs against the fake ``Popen`` shared
with the v2-v5 tests. Nothing here decodes, encodes or writes a video. The
v1-v5 test modules are untouched; the last section pins down that v5's
output is unchanged wherever v6 does not spatially fill.
"""

from __future__ import annotations

import io
import math
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

import video_character_skill
from test_compositor_union import (
    FILES,
    STREAMS,
    FakeFfmpeg,
    H,
    W,
    default_outputs,
    fake_probe,
    frames_bytes,
    plane,
    wire,
)
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
    scene,
    scene_outputs,
    solid,
    source_frame,
)
from test_temporal_recovery import run_clip as run_clip_v5
from video_character_skill.compositor import CompositeError, VideoInfo, composite_frame
from video_character_skill.spatial_recovery import (
    SPATIAL_FILL_MARGIN,
    ComponentFill,
    FrameBackground,
    SpatialFill,
    SpatialRecoveryCompositeReport,
    SpatialRecoveryStreamStats,
    composite_streams_spatial_recovery,
    composite_video_spatial_recovery,
    label_components,
    recover_frame_background,
    spatial_fill_component,
    spatial_fill_components,
    temporal_plate,
    trusted_background,
)
from video_character_skill.temporal_recovery import (
    TemporalRecovery,
    recover_pixels,
    recovery_regions,
)

Rgb = tuple[int, int, int]
Bool = NDArray[np.bool_]


# -- crop builders --------------------------------------------------------


def crop(
    rows: list[str], values: dict[str, Rgb] | None = None
) -> tuple[NDArray[np.uint8], Bool, Bool]:
    """Build ``(rgb, trusted, target)`` from a picture.

    ``T`` trusted (value ``values['T']``, default 100), ``t`` target, ``x``
    neither (an untrusted non-target pixel, value ``values['x']``, default
    a loud 255). Any other letter is trusted with its own colour from
    ``values``.
    """
    colours: dict[str, Rgb] = {"T": (100, 100, 100), "x": (255, 255, 255), "t": (0, 0, 0)}
    colours.update(values or {})
    h, w = len(rows), len(rows[0])
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    trusted = np.zeros((h, w), dtype=np.bool_)
    target = np.zeros((h, w), dtype=np.bool_)
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            rgb[y, x] = colours[ch]
            trusted[y, x] = ch not in ("t", "x")
            target[y, x] = ch == "t"
    return rgb, trusted, target


def fill(rows: list[str], values: dict[str, Rgb] | None = None) -> SpatialFill:
    rgb, trusted, target = crop(rows, values)
    return spatial_fill_components(rgb, trusted, target)


def grey(result: SpatialFill | ComponentFill) -> NDArray[np.int64]:
    assert (result.rgb[..., 0] == result.rgb[..., 1]).all()
    return result.rgb[..., 0].astype(np.int64)


# -- 1-2: holes -----------------------------------------------------------


def test_single_pixel_hole_takes_the_mean_of_its_eight_trusted_neighbours() -> None:
    rgb, trusted, target = crop(["TTT", "TtT", "TTT"])
    rgb[0, 0] = (10, 20, 30)
    rgb[2, 2] = (90, 80, 70)
    # channel means over the 8 neighbours: 6 x 100 + the two odd ones
    expected = tuple((2 * (600 + a + b) + 8) // 16 for a, b in ((10, 90), (20, 80), (30, 70)))

    result = spatial_fill_components(rgb, trusted, target)

    assert tuple(int(v) for v in result.rgb[1, 1]) == expected
    assert result.filled.sum() == 1 and result.filled[1, 1]
    assert result.depth[1, 1] == 1
    assert (result.components, result.components_without_seed) == (1, 0)


def test_multi_pixel_component_fills_inward_wave_by_wave() -> None:
    result = fill(["TTTTT", "TtttT", "TtttT", "TtttT", "TTTTT"])

    assert (grey(result)[1:4, 1:4] == 100).all()
    np.testing.assert_array_equal(result.depth[1:4, 1:4], [[1, 1, 1], [1, 2, 1], [1, 1, 1]])
    assert result.filled.sum() == 9


def test_a_five_wide_hole_needs_three_waves_and_blends_its_two_sides() -> None:
    result = fill(["TTTTTTT", "TtttttT", "TTTTTTT"], {"T": (100, 100, 100)})
    np.testing.assert_array_equal(result.depth[1, 1:6], [1, 1, 1, 1, 1])  # row above/below seed all

    # a corridor: only the two ends are trusted, so waves must march inward
    result = fill(["xxxxxxx", "AtttttB", "xxxxxxx"], {"A": (0, 0, 0), "B": (200, 200, 200)})
    np.testing.assert_array_equal(result.depth[1, 1:6], [1, 2, 3, 2, 1])
    np.testing.assert_array_equal(grey(result)[1, 1:6], [0, 0, 100, 200, 200])


# -- 3: synchronous waves -------------------------------------------------


def test_waves_are_synchronous_so_traversal_order_cannot_matter() -> None:
    rows = ["xxxxxx", "AttttB", "xxxxxx"]
    values = {"A": (0, 0, 0), "B": (200, 200, 200)}
    forward = fill(rows, values)
    mirrored = fill([r[::-1] for r in rows], values)

    # a raster-order (Gauss-Seidel) fill would give 0, 0, 0, 100 here
    np.testing.assert_array_equal(grey(forward)[1, 1:5], [0, 0, 200, 200])
    np.testing.assert_array_equal(forward.depth[1, 1:5], [1, 2, 2, 1])
    np.testing.assert_array_equal(grey(mirrored)[1, ::-1], grey(forward)[1])
    np.testing.assert_array_equal(mirrored.depth[1, ::-1], forward.depth[1])


def test_a_wave_reads_only_the_previous_resolved_state() -> None:
    # pixel (1,2) has a trusted neighbour only diagonally; (1,1) is filled in
    # the same wave and must not feed it early
    result = fill(["Txxx", "xttx", "xxxx"], {"T": (40, 40, 40)})
    assert result.depth[1, 1] == 1 and result.depth[1, 2] == 2
    assert grey(result)[1, 1] == 40 and grey(result)[1, 2] == 40


def test_mean_is_rounded_half_up_in_integer_arithmetic() -> None:
    rgb, trusted, target = crop(["xxx", "AtB", "xxx"], {"A": (100, 0, 255), "B": (101, 1, 255)})
    result = spatial_fill_components(rgb, trusted, target)
    assert tuple(int(v) for v in result.rgb[1, 1]) == (101, 1, 255)  # 100.5 -> 101, no overflow


# -- 4-5: connectivity and isolation --------------------------------------


def test_eight_connectivity_lets_a_diagonal_seed_fill_a_pixel() -> None:
    result = fill(["Tx", "xt"], {"T": (7, 7, 7)})
    assert result.filled[1, 1] and grey(result)[1, 1] == 7 and result.depth[1, 1] == 1


def test_diagonally_touching_target_pixels_are_one_component() -> None:
    labels, count = label_components(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=bool))
    assert count == 1 and labels[0, 0] == labels[1, 1] == labels[2, 2] == 1


def test_components_are_labelled_deterministically_in_raster_order() -> None:
    mask = np.zeros((5, 6), dtype=np.bool_)
    mask[0, 5] = mask[1, 4] = True  # diagonal pair, first pixel (0,5)
    mask[3, 0] = True  # lone pixel
    mask[4, 3:6] = True  # run
    labels, count = label_components(mask)
    assert count == 3
    assert labels[0, 5] == labels[1, 4] == 1
    assert labels[3, 0] == 2
    assert (labels[4, 3:6] == 3).all()
    assert (labels[~mask] == 0).all()


def test_label_components_handles_an_empty_mask() -> None:
    labels, count = label_components(np.zeros((3, 4), dtype=np.bool_))
    assert count == 0 and not labels.any()


def test_two_components_in_one_crop_do_not_contaminate_each_other() -> None:
    # A is L-shaped so its bounding box (and crop) contains B; B has no
    # trusted neighbour and must stay unfilled even though A was filled.
    rows = ["TTTTT", "TtttT", "Ttxxx", "Ttxtx", "Txxxx"]
    result = fill(rows, {"T": (50, 50, 50)})

    assert result.components == 2
    assert result.components_without_seed == 1
    assert result.filled[1, 1:4].all() and result.filled[2, 1] and result.filled[3, 1]
    assert not result.filled[3, 3]
    assert (grey(result)[result.filled] == 50).all()
    assert tuple(int(v) for v in result.rgb[3, 3]) == (0, 0, 0)  # untouched target value


def test_separate_components_fill_from_their_own_seeds_only() -> None:
    rows = ["xxxxx", "AtxtB", "xxxxx"]
    result = fill(rows, {"A": (0, 0, 0), "B": (200, 200, 200)})
    assert grey(result)[1, 1] == 0 and grey(result)[1, 3] == 200
    assert result.components == 2 and result.components_without_seed == 0


# -- 6-10: what counts as a seed ------------------------------------------


def test_untrusted_neighbours_are_never_read() -> None:
    # (1,1)'s only trusted neighbour is A; the seven 'x' pixels are loud 255s
    result = fill(["Axx", "xtx", "xxx"], {"A": (30, 30, 30)})
    assert grey(result)[1, 1] == 30


def test_source_person_pixels_are_never_spatial_seeds() -> None:
    src = solid(BG)
    src[1, 2] = PERSON
    alpha = plane(0)
    alpha[1, 1] = 255  # the target: never recovered
    alpha[1, 2] = 40  # old-person pixel that is neither target nor background
    recovery = unrecovered_at([(1, 1)])

    result = recover_frame_background(src, solid(O1_BG), alpha, recovery)

    assert not result.fill.filled[1, 2] and not result.temporal_unrecovered[1, 2]
    assert tuple(int(v) for v in result.background[1, 1]) == BG  # mean of the 7 BG neighbours
    assert tuple(int(v) for v in result.background[1, 2]) == PERSON  # not a target: untouched here


def test_o1_pixels_are_never_spatial_seeds() -> None:
    src = solid(BG)
    alpha = plane(255)  # nothing trusted at all
    alpha[1, 1] = 255
    recovery = unrecovered_at([(1, 1)])
    o1 = solid(O1_BG)

    result = recover_frame_background(src, o1, alpha, recovery)

    assert result.fill.components_without_seed == 1 and not result.fill.filled.any()
    assert tuple(int(v) for v in result.background[1, 1]) == O1_BG  # residual fallback only
    # the O1 value at (1,1) did not seed anything: alter it and the fill stays empty
    o1[1, 1] = (1, 2, 3)
    again = recover_frame_background(src, o1, alpha, recovery)
    assert not again.fill.filled.any()
    assert tuple(int(v) for v in again.background[1, 1]) == (1, 2, 3)


def test_temporally_recovered_pixels_are_trusted_seeds() -> None:
    src = solid(PERSON)
    alpha = plane(255)  # no own-frame background anywhere
    borrowed: Rgb = (60, 70, 80)
    recovery = TemporalRecovery(
        indices=np.array([flat(1, 1), flat(1, 2)]),
        rgb=np.array([[0, 0, 0], borrowed], dtype=np.uint8),
        counts=np.array([0, 1]),
        distances=np.array([1]),
        donor_fits=1,
        zero_offset_fallbacks=0,
    )

    result = recover_frame_background(src, solid(O1_BG), alpha, recovery)

    assert result.temporal_recovered[1, 2] and result.temporal_unrecovered[1, 1]
    assert result.fill.filled[1, 1]
    assert tuple(int(v) for v in result.background[1, 1]) == borrowed
    assert tuple(int(v) for v in result.background[1, 2]) == borrowed  # byte-identical v5 value


@pytest.mark.parametrize(("alpha_value", "seeded"), [(31, True), (32, False)])
def test_source_alpha_31_is_a_trusted_seed_and_32_is_not(alpha_value: int, seeded: bool) -> None:
    src = solid(BG)
    src[1, 2] = (11, 22, 33)
    alpha = plane(255)
    alpha[1, 2] = alpha_value
    recovery = unrecovered_at([(1, 1)])

    result = recover_frame_background(src, solid(O1_BG), alpha, recovery)

    assert bool(result.fill.filled[1, 1]) is seeded
    expected = (11, 22, 33) if seeded else O1_BG
    assert tuple(int(v) for v in result.background[1, 1]) == expected


def test_trusted_background_is_own_background_or_recovered() -> None:
    alpha = plane(255)
    alpha[0, 0] = 31
    alpha[0, 1] = 32
    recovered = np.zeros((H, W), dtype=np.bool_)
    recovered[3, 4] = True
    trusted = trusted_background(alpha, recovered)
    assert trusted[0, 0] and not trusted[0, 1] and trusted[3, 4] and trusted.sum() == 2
    with pytest.raises(CompositeError):
        trusted_background(alpha, recovered, background_threshold=0)


# -- 11: no seed ----------------------------------------------------------


def test_a_component_without_trusted_boundary_stays_unrecovered() -> None:
    result = fill(["xxxx", "xttx", "xxxx"])
    assert not result.filled.any()
    assert (result.components, result.components_without_seed) == (1, 1)
    assert not result.depth.any()


def test_no_seed_is_not_reached_through_a_replacement_person_wall() -> None:
    # the trusted pixel is two steps away behind an untrusted wall: unreachable
    result = fill(["Txxx", "xxxx", "xxtx"])
    assert not result.filled.any() and result.components_without_seed == 1


# -- 12-14: what the fill may touch ----------------------------------------


def test_spatial_fill_writes_only_target_pixels() -> None:
    rgb, trusted, target = crop(["TTTT", "TttT", "TxtT", "TTTT"])
    before = rgb.copy()
    result = spatial_fill_components(rgb, trusted, target)
    np.testing.assert_array_equal(rgb, before)  # input untouched
    np.testing.assert_array_equal(result.rgb[~target], before[~target])
    assert (result.filled <= target).all()


def test_overlapping_trusted_and_target_are_refused() -> None:
    rgb, trusted, target = crop(["Tt"])
    with pytest.raises(CompositeError, match="overlap"):
        spatial_fill_components(rgb, trusted | target, target)
    with pytest.raises(CompositeError, match="overlap"):
        spatial_fill_component(rgb, trusted | target, target)


def test_bad_shapes_and_dtypes_are_refused() -> None:
    rgb, trusted, target = crop(["Tt"])
    with pytest.raises(CompositeError):
        spatial_fill_components(rgb.astype(np.int16), trusted, target)
    with pytest.raises(CompositeError):
        spatial_fill_components(rgb, trusted[:, :1], target)
    with pytest.raises(CompositeError):
        spatial_fill_components(rgb, trusted, target.astype(np.uint8))
    with pytest.raises(CompositeError):
        label_components(target.astype(np.uint8))


def test_own_frame_background_and_recovered_pixels_are_byte_identical_to_v5() -> None:
    sources = [source_frame(P), source_frame(P), source_frame(P)]  # P never recoverable
    sources[1][0][2, 3] = PERSON  # (2,3) hidden only in frame 1: temporally recovered
    sources[1][1][2, 3] = 200
    replacements = [replacement_frame()] * 3
    v5, _ = run_clip_v5(sources, replacements)
    v6, stats = run_clip(sources, replacements)

    changed = np.any(v5 != v6, axis=3)
    for i in range(3):
        assert changed[i].sum() == 1 and changed[i][P]  # only the spatial target differs
    assert pixel(v5, 1) == O1_BG and pixel(v6, 1) == BG
    assert pixel(v6, 1, (2, 3)) == BG
    assert stats.temporal_recovered_ratio == pytest.approx(1 / (H * W) / 3)
    assert stats.spatial_recovered_ratio == pytest.approx(1 / (H * W))


# -- 15-18: composite semantics ---------------------------------------------


def test_replacement_foreground_and_partial_alpha_behave_exactly_as_v5() -> None:
    sources = [source_frame(P, alpha=255), source_frame(P), source_frame(P)]
    replacements = [replacement_frame(), replacement_frame((2, 3), alpha=255), replacement_frame()]
    replacements[1][0][0, 0] = O1_PERSON
    replacements[1][1][0, 0] = 200  # partial replacement alpha over own background
    replacements[2][1][P] = 200  # partial replacement alpha over the target: forced opaque

    v5, _ = run_clip_v5(sources, replacements)
    v6, _ = run_clip(sources, replacements)

    assert pixel(v6, 1, (2, 3)) == pixel(v5, 1, (2, 3)) == O1_PERSON
    assert pixel(v6, 1, (0, 0)) == pixel(v5, 1, (0, 0))
    assert pixel(v6, 2) == pixel(v5, 2) == O1_BG  # force_replacement: replacement wins
    np.testing.assert_array_equal(v6[2], v5[2])  # P is force_replacement here: nothing to fill
    differs = np.any(v6[1] != v5[1], axis=2)
    assert differs.sum() == 1 and differs[P]  # only the spatially filled target


def test_outside_the_removal_mask_output_equals_plain_replacement_compositing() -> None:
    sources = [source_frame(P, alpha=90)] * 3
    replacements = [replacement_frame((3, 4), alpha=128)] * 3
    frames, _ = run_clip(sources, replacements)
    removal = recovery_regions(sources[0][1], replacements[0][1], dilation_radius=0).removal
    plain = composite_frame(sources[0][0], replacements[0][0], replacements[0][1])
    for i in range(3):
        np.testing.assert_array_equal(frames[i][~removal], plain[~removal])


def test_the_old_source_person_is_never_a_fallback() -> None:
    sources = [source_frame(P)] * 3
    sources_walled = [source_frame(P)] * 3
    for src in sources_walled:  # every neighbour of P is old person too: no seed anywhere
        src[1][:] = 255
        src[0][:] = PERSON
    seeded, _ = run_clip(sources, [replacement_frame()] * 3)
    walled, stats = run_clip(sources_walled, [replacement_frame()] * 3)

    assert pixel(seeded, 1) == BG
    assert (walled == O1_BG).all(axis=3).all()  # whole frame is residual: O1, never PERSON
    assert not (walled == PERSON).all(axis=3).any()
    assert stats.o1_fallback_ratio == 1.0 and stats.components_without_seed == 3


def test_o1_is_used_only_for_the_spatially_unrecoverable_residual() -> None:
    sources = [source_frame(P)] * 3
    for src in sources:
        src[1][3, 4] = 255  # a second, isolated old-person pixel in the far corner...
        src[0][3, 4] = PERSON
        for wall in ((2, 3), (2, 4), (3, 3)):  # ...walled off by non-background old-person alpha
            src[1][wall] = 40
            src[0][wall] = PERSON
    frames, stats = run_clip(sources, [replacement_frame()] * 3)

    assert pixel(frames, 1, (3, 4)) == O1_BG  # residual
    assert pixel(frames, 1) == BG  # P's other neighbours are real background
    assert stats.spatial_components == 6 and stats.components_without_seed == 3
    assert stats.spatial_unrecovered_ratio == pytest.approx(1 / (H * W))
    assert stats.o1_fallback_ratio == pytest.approx(1 / 2)


# -- 19: diagnostics ----------------------------------------------------------


def test_propagation_depth_diagnostics() -> None:
    sources = [source_frame()] * 3
    for src in sources:  # a 3x3 block of old person, never background
        src[1][0:3, 1:4] = 200
        src[0][0:3, 1:4] = PERSON
    frames, stats = run_clip(sources, [replacement_frame()] * 3)

    assert (frames == BG).all(axis=3).all()
    assert isinstance(stats, SpatialRecoveryStreamStats)
    assert stats.spatial_components == 3 and stats.components_without_seed == 0
    assert (stats.median_propagation_depth, stats.p90_propagation_depth) == (1.0, 2.0)
    assert stats.max_propagation_depth == 2
    assert stats.spatial_recovered_ratio == pytest.approx(9 / (H * W))
    assert stats.spatial_unrecovered_ratio == 0.0
    assert stats.temporal_unrecovered_ratio == pytest.approx(9 / (H * W))
    assert stats.o1_fallback_ratio == 0.0


def test_v5_diagnostics_are_preserved() -> None:
    sources = [source_frame(), source_frame(P), source_frame()]
    _, stats = run_clip(sources, [replacement_frame()] * 3)
    assert (stats.median_donor_distance, stats.p90_donor_distance) == (1.0, 1.0)
    assert stats.mean_observations_per_recovered_pixel == 2.0
    assert (stats.donor_fits, stats.zero_offset_fallbacks) == (2, 0)
    assert stats.peak_cached_frames == 3
    assert stats.recovery_region_ratio == pytest.approx(1 / (H * W) / 3)
    assert stats.own_background_ratio == 0.0
    assert stats.spatial_components == 0 and math.isnan(stats.median_propagation_depth)
    assert stats.max_propagation_depth == 0


def test_zero_frames_writes_nothing() -> None:
    out = io.BytesIO()
    stats = run_streams({name: io.BytesIO() for name in STREAMS}, 0, out)
    assert out.getvalue() == b""
    assert stats.spatial_recovered_ratio == 0.0 and stats.peak_cached_frames == 0
    assert math.isnan(stats.median_propagation_depth) and math.isnan(stats.median_donor_distance)


# -- 20-21: awkward component shapes -----------------------------------------


def test_border_touching_components_fill_from_inside_the_frame() -> None:
    result = fill(["tT", "TT"], {"T": (9, 9, 9)})
    assert result.filled[0, 0] and grey(result)[0, 0] == 9

    result = fill(["tttt", "TTTT"], {"T": (9, 9, 9)})
    assert result.filled[0].all() and (result.depth[0] == 1).all()

    result = fill(["tttt", "xxxT"], {"T": (9, 9, 9)})  # the seed is diagonal to (0,2) too
    np.testing.assert_array_equal(result.depth[0], [3, 2, 1, 1])
    assert (grey(result)[0] == 9).all()


def test_one_pixel_wide_irregular_components_propagate_along_their_length() -> None:
    rows = ["Txxxx", "xtxxx", "xttxx", "xxxtx", "xxxtt"]
    result = fill(rows, {"T": (33, 33, 33)})
    assert result.components == 1
    assert result.filled.sum() == 6
    assert (grey(result)[result.filled] == 33).all()
    assert result.depth[1, 1] == 1 and result.depth[2, 1] == 2 and result.depth[2, 2] == 2
    assert result.depth[3, 3] == 3 and result.depth[4, 3] == 4 and result.depth[4, 4] == 4
    assert result.depth.max() == 4


def test_components_touching_the_margin_use_a_clipped_crop() -> None:
    assert SPATIAL_FILL_MARGIN == 1
    rgb, trusted, target = crop(["tt", "tt"], {"T": (1, 1, 1)})
    result = spatial_fill_components(rgb, trusted, target)  # whole frame is target
    assert result.components == 1 and result.components_without_seed == 1


def test_temporal_plate_holds_recovered_values_and_no_o1() -> None:
    src = solid(PERSON)
    recovery = TemporalRecovery(
        indices=np.array([flat(0, 0), flat(0, 1)]),
        rgb=np.array([[5, 6, 7], [0, 0, 0]], dtype=np.uint8),
        counts=np.array([1, 0]),
        distances=np.array([1]),
        donor_fits=1,
        zero_offset_fallbacks=0,
    )
    plate, recovered, unrecovered = temporal_plate(src, recovery)
    assert tuple(int(v) for v in plate[0, 0]) == (5, 6, 7)
    assert tuple(int(v) for v in plate[0, 1]) == PERSON  # still the old person: the target
    assert recovered.sum() == 1 and recovered[0, 0]
    assert unrecovered.sum() == 1 and unrecovered[0, 1]


def test_recover_frame_background_reports_its_masks() -> None:
    src = solid(BG)
    alpha = plane(0)
    alpha[1, 1] = 255
    result = recover_frame_background(src, solid(O1_BG), alpha, unrecovered_at([(1, 1)]))
    assert isinstance(result, FrameBackground)
    assert result.temporal_unrecovered.sum() == 1 and result.fill.filled[1, 1]
    assert not result.residual.any()
    with pytest.raises(CompositeError):
        recover_frame_background(src, solid(O1_BG)[:, :1], alpha, unrecovered_at([(1, 1)]))


# -- streaming: errors ------------------------------------------------------


def test_parameters_are_checked_before_any_stream_is_read() -> None:
    streams = clip_streams([source_frame()] * 2, [replacement_frame()] * 2)
    bad_params: list[dict[str, Any]] = [
        {"radius": -1},
        {"max_observations": 0},
        {"background_threshold": 0},
        {"foreground_threshold": 256},
        {"offset_limit": 300},
        {"offset_stride": 0},
        {"dilation_radius": -1},
    ]
    for bad in bad_params:
        with pytest.raises(CompositeError):
            run_streams(streams, 2, **bad)
    assert all(stream.tell() == 0 for stream in streams.values())


@pytest.mark.parametrize("short", STREAMS)
def test_premature_eof_on_any_of_the_four_streams_names_it(short: str) -> None:
    streams = clip_streams([source_frame()] * 3, [replacement_frame()] * 3)
    streams[short] = io.BytesIO(streams[short].getvalue()[: -H * W])
    with pytest.raises(CompositeError, match=f"{short} ended during frame 2"):
        run_streams(streams, 3)


@pytest.mark.parametrize("long", STREAMS)
def test_extra_data_on_any_of_the_four_streams_names_it(long: str) -> None:
    streams = clip_streams([source_frame()] * 2, [replacement_frame()] * 2)
    streams[long] = io.BytesIO(streams[long].getvalue() + b"\0")
    with pytest.raises(CompositeError, match=f"{long} has more than the expected 2"):
        run_streams(streams, 2)


# -- orchestration, against the fake ffmpeg ---------------------------------


def run_recovery(tmp_path: Path, out: Path, **kwargs: Any) -> SpatialRecoveryCompositeReport:
    return composite_video_spatial_recovery(
        tmp_path / "src.mp4",
        tmp_path / "rep.mp4",
        tmp_path / "src.webm",
        tmp_path / "rep.webm",
        out,
        **kwargs,
    )


def test_composite_video_spatial_recovery_matches_the_streaming_function(
    tmp_path: Path, monkeypatch: Any
) -> None:
    sources = [source_frame(P), source_frame(P), source_frame(P)]  # P never background: filled
    sources[1][0][3, 0] = PERSON  # (3,0) hidden only in frame 1: temporally recovered
    sources[1][1][3, 0] = 200
    replacements = [replacement_frame(), replacement_frame((2, 3), alpha=255), replacement_frame()]
    ffmpeg = FakeFfmpeg(scene_outputs(sources, replacements))
    wire(monkeypatch, ffmpeg, fake_probe(3))
    out = tmp_path / "nested" / "v6.mp4"

    report = run_recovery(tmp_path, out)

    expected, expected_stats = run_clip(
        sources, replacements, dilation_radius=4, offset_stride=8, offset_min_samples=256
    )
    assert ffmpeg.encoded == expected.tobytes()
    assert report.stats == expected_stats
    written = np.frombuffer(ffmpeg.encoded, np.uint8).reshape(3, H, W, 3)
    assert pixel(written, 1) == BG and pixel(written, 1, (2, 3)) == O1_PERSON
    assert pixel(written, 1, (3, 0)) == BG
    assert not math.isnan(report.stats.median_donor_distance)
    assert report.output_path == out
    assert (report.frames, report.width, report.height) == (3, W, H)
    assert report.frame_rate == Fraction(24, 1)
    assert (report.removal_threshold, report.dilation_radius) == (64, 4)
    assert (report.background_threshold, report.foreground_threshold) == (32, 128)
    assert (report.radius, report.max_observations) == (24, 5)
    assert report.stats.peak_cached_frames == 3
    assert out.parent.is_dir()
    assert not any(p.killed for p in ffmpeg.processes)


def test_spatial_decoders_force_libvpx_for_both_mattes_only(
    tmp_path: Path, monkeypatch: Any
) -> None:
    ffmpeg = FakeFfmpeg(default_outputs(2))
    wire(monkeypatch, ffmpeg, fake_probe(2))
    out = tmp_path / "v6.mp4"

    run_recovery(tmp_path, out)

    assert ffmpeg.input_names() == ["src.mp4", "rep.mp4", "src.webm", "rep.webm", "-"]
    src, rep, source_matte, replacement_matte, encode = ffmpeg.commands
    for clip in (src, rep):
        assert "-c:v" not in clip and clip[clip.index("-pix_fmt") + 1] == "rgb24"
    for matte in (source_matte, replacement_matte):
        assert matte[matte.index("-c:v") + 1] == "libvpx-vp9"
        assert matte[matte.index("-pix_fmt") + 1] == "rgba"
    assert encode[encode.index("-s") + 1] == f"{W}x{H}"
    assert encode[encode.index("-r") + 1] == "24/1"
    assert encode[-1] == str(out)


def test_parameters_pass_through_orchestration(tmp_path: Path, monkeypatch: Any) -> None:
    sources, replacements = scene()
    ffmpeg = FakeFfmpeg(scene_outputs(sources, replacements))
    wire(monkeypatch, ffmpeg, fake_probe(3))

    report = run_recovery(tmp_path, tmp_path / "v6.mp4", radius=0, dilation_radius=0)

    assert (report.radius, report.dilation_radius) == (0, 0)
    assert report.stats.peak_cached_frames == 1
    # no donors at radius 0, but the spatial fill reaches P from its neighbours
    assert report.stats.temporal_unrecovered_ratio == pytest.approx(1 / (H * W) / 3)
    assert report.stats.spatial_recovered_ratio == pytest.approx(1 / (H * W) / 3)
    assert report.stats.o1_fallback_ratio == 0.0
    written = np.frombuffer(ffmpeg.encoded, np.uint8).reshape(3, H, W, 3)
    assert pixel(written, 1) == BG


def test_invalid_parameters_are_refused_before_anything_is_probed(
    tmp_path: Path, monkeypatch: Any
) -> None:
    ffmpeg = FakeFfmpeg(default_outputs(1))

    def never(path: Path, decoder: str | None = None) -> VideoInfo:
        raise AssertionError(f"probed {path}")

    wire(monkeypatch, ffmpeg, never)

    for bad in ({"radius": -1}, {"max_observations": 0}, {"foreground_threshold": 0}):
        with pytest.raises(CompositeError):
            run_recovery(tmp_path, tmp_path / "v6.mp4", **bad)
    assert ffmpeg.commands == []


def test_a_matte_without_alpha_stops_the_run_before_any_process_spawns(
    tmp_path: Path, monkeypatch: Any
) -> None:
    ffmpeg = FakeFfmpeg(default_outputs(2))
    probe = fake_probe(2)

    def alpha_lost(path: Path, decoder: str | None = None) -> VideoInfo:
        result = probe(path, decoder)
        return replace(result, pix_fmt="yuv420p") if path.name == "src.webm" else result

    wire(monkeypatch, ffmpeg, alpha_lost)

    with pytest.raises(CompositeError, match="source matte: matte .* has no alpha"):
        run_recovery(tmp_path, tmp_path / "v6.mp4")
    assert ffmpeg.commands == []


def test_a_missing_input_is_rejected_before_ffprobe_runs(tmp_path: Path) -> None:
    with pytest.raises(CompositeError, match="input not found"):
        run_recovery(tmp_path, tmp_path / "v6.mp4")


@pytest.mark.parametrize("name", STREAMS)
def test_a_failing_decoder_is_reported_with_its_name_and_stderr(
    name: str, tmp_path: Path, monkeypatch: Any
) -> None:
    ffmpeg = FakeFfmpeg(default_outputs(2), failing=FILES[name], stderr="could not decode")
    wire(monkeypatch, ffmpeg, fake_probe(2))

    with pytest.raises(CompositeError, match=f"ffmpeg {name} exited 1: could not decode"):
        run_recovery(tmp_path, tmp_path / "v6.mp4")


def test_a_failing_encoder_is_reported(tmp_path: Path, monkeypatch: Any) -> None:
    ffmpeg = FakeFfmpeg(default_outputs(2), failing="encode", stderr="broken pipe")
    wire(monkeypatch, ffmpeg, fake_probe(2))

    with pytest.raises(CompositeError, match="ffmpeg encode exited 1: broken pipe"):
        run_recovery(tmp_path, tmp_path / "v6.mp4")


def test_a_decoder_that_dies_mid_stream_aborts_and_kills_the_rest(
    tmp_path: Path, monkeypatch: Any
) -> None:
    outputs = default_outputs(3)
    outputs["src.mp4"] = frames_bytes([1])  # one frame, then silence
    ffmpeg = FakeFfmpeg(outputs, failing="src.mp4", stderr="decode error")
    wire(monkeypatch, ffmpeg, fake_probe(3))

    with pytest.raises(CompositeError, match="source ended during frame 1"):
        run_recovery(tmp_path, tmp_path / "v6.mp4")
    assert all(p.killed for p in ffmpeg.processes)


def test_spatial_api_is_exported_from_the_package() -> None:
    package: Any = video_character_skill
    assert package.composite_video_spatial_recovery is composite_video_spatial_recovery
    assert package.SpatialRecoveryCompositeReport is SpatialRecoveryCompositeReport
    assert package.SPATIAL_FILL_MARGIN == 1
    assert "composite_video_spatial_recovery" in video_character_skill.__all__


# -- v5 is not affected by v6 -----------------------------------------------


def test_v5_still_falls_back_to_o1_where_v6_fills_spatially() -> None:
    sources = [source_frame(P)] * 3
    replacements = [replacement_frame()] * 3
    v5, v5_stats = run_clip_v5(sources, replacements)
    v6, v6_stats = run_clip(sources, replacements)
    assert pixel(v5, 1) == O1_BG and pixel(v6, 1) == BG
    assert v5_stats.o1_fallback_ratio == 1.0 and v6_stats.o1_fallback_ratio == 0.0
    assert v5_stats.temporal_unrecovered_ratio == v6_stats.temporal_unrecovered_ratio


# -- helpers ------------------------------------------------------------------


def flat(y: int, x: int) -> int:
    return y * W + x


def unrecovered_at(pixels: list[tuple[int, int]]) -> TemporalRecovery:
    n = len(pixels)
    return TemporalRecovery(
        indices=np.array([flat(y, x) for y, x in pixels]),
        rgb=np.zeros((n, 3), dtype=np.uint8),
        counts=np.zeros(n, dtype=np.int64),
        distances=np.zeros(0, dtype=np.int64),
        donor_fits=0,
        zero_offset_fallbacks=0,
    )


def run_streams(
    streams: dict[str, io.BytesIO], frames: int, out: io.BytesIO | None = None, **kwargs: Any
) -> SpatialRecoveryStreamStats:
    return composite_streams_spatial_recovery(
        streams["source"],
        streams["replacement"],
        streams["source_matte"],
        streams["replacement_matte"],
        out if out is not None else io.BytesIO(),
        width=W,
        height=H,
        frames=frames,
        **kwargs,
    )


def run_clip(
    sources: list[Frame], replacements: list[Frame], **overrides: Any
) -> tuple[NDArray[np.uint8], SpatialRecoveryStreamStats]:
    out = io.BytesIO()
    params = {**FULL_FIT, **overrides}
    stats = run_streams(clip_streams(sources, replacements), len(sources), out, **params)
    frames = np.frombuffer(out.getvalue(), np.uint8).reshape(len(sources), H, W, 3)
    return frames, stats


def test_recover_pixels_is_reused_unchanged_for_the_temporal_step() -> None:
    """The spatial step starts from exactly what v5's recover_pixels produced."""
    sources = [source_frame(), source_frame(P), source_frame()]
    regions = recovery_regions(sources[1][1], plane(0), dilation_radius=0)
    donors = [(0, *sources[0]), (2, *sources[2])]
    recovery = recover_pixels(
        sources[1][0], sources[1][1], regions.needs_temporal, donors, target_index=1,
        offset_stride=1, offset_min_samples=1,
    )
    result = recover_frame_background(sources[1][0], solid(O1_BG), sources[1][1], recovery)
    assert result.temporal_recovered[P] and not result.temporal_unrecovered.any()
    assert result.fill.components == 0
    np.testing.assert_array_equal(result.background, solid(BG))
