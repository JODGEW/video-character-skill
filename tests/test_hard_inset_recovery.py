"""Unit tests for the hard-inset replacement foreground compositor (v8).

Erosion geometry is checked against a brute-force reference; the streaming
loop runs on file-like objects; the ffmpeg pipeline runs against the fake
``Popen`` shared with the v2-v6 tests. Nothing here decodes, encodes or
writes a video. The v1-v7 modules and tests are untouched; the last section
pins down that v6 keeps its own semantics.
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
from test_spatial_recovery import run_clip as run_clip_v6
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
    scene_outputs,
    source_frame,
)
from video_character_skill.compositor import (
    CompositeError,
    VideoInfo,
    _disk_offsets,
    source_removal_mask,
)
from video_character_skill.hard_inset_recovery import (
    HARD_FOREGROUND_EROSION_RADIUS,
    HARD_FOREGROUND_THRESHOLD,
    HARD_INSET_REMOVAL_THRESHOLD,
    HardInsetCompositeReport,
    HardInsetRegions,
    HardInsetStreamStats,
    composite_streams_hard_inset_recovery,
    composite_video_hard_inset_recovery,
    erode_disk,
    hard_effective_alpha,
    hard_foreground_mask,
    hard_inset_regions,
)

Bool = NDArray[np.bool_]

# The 4x5 test frames cannot hold a core eroded by 2 px, so the stream-level
# tests run with radius 0 (or 1) and the geometry is tested on larger arrays.
V8_TEST: dict[str, Any] = {**FULL_FIT, "erosion_radius": 0}


def erode_reference(mask: Bool, radius: int) -> Bool:
    """Brute force: a pixel survives iff every disk offset lands inside on True."""
    h, w = mask.shape
    out = np.zeros_like(mask)
    for y in range(h):
        for x in range(w):
            out[y, x] = all(
                0 <= y + dy < h and 0 <= x + dx < w and mask[y + dy, x + dx]
                for dy, dx in _disk_offsets(radius)
            )
    return out


def picture(rows: list[str]) -> Bool:
    return np.array([[c == "#" for c in row] for row in rows], dtype=np.bool_)


# -- 1-5: erosion geometry ---------------------------------------------------


def test_the_constants_are_the_agreed_poc_values() -> None:
    assert HARD_FOREGROUND_THRESHOLD == 250
    assert HARD_FOREGROUND_EROSION_RADIUS == 2
    assert HARD_INSET_REMOVAL_THRESHOLD == 32


def test_erosion_radius_zero_is_the_identity_on_a_copy() -> None:
    mask = picture(["#.#", ".#.", "#.."])
    out = erode_disk(mask, 0)
    np.testing.assert_array_equal(out, mask)
    out[0, 0] = False
    assert mask[0, 0]  # a copy, not a view


def test_erosion_radius_one_uses_the_euclidean_plus_not_the_square() -> None:
    plus = picture([".#.", "###", ".#."])
    assert erode_disk(plus, 1)[1, 1]  # the four axis neighbours suffice ...
    assert erode_disk(plus, 1).sum() == 1
    block = picture(["###", "###", "###"])
    np.testing.assert_array_equal(erode_disk(block, 1), picture(["...", ".#.", "..."]))
    missing_axis = picture(["...", "###", ".#."])  # (0,1) missing: centre must die
    assert not erode_disk(missing_axis, 1).any()


def test_erosion_radius_two_is_the_exact_integer_disk() -> None:
    assert len(_disk_offsets(2)) == 13  # the disk r=2 has 13 offsets, corners excluded
    block = np.ones((5, 5), dtype=np.bool_)
    centre_only = picture([".....", ".....", "..#..", ".....", "....."])
    np.testing.assert_array_equal(erode_disk(block, 2), centre_only)
    no_corners = block.copy()
    no_corners[[0, 0, 4, 4], [0, 4, 0, 4]] = False  # (±2, ±2) are outside the disk
    assert erode_disk(no_corners, 2)[2, 2]
    no_axis = block.copy()
    no_axis[0, 2] = False  # (-2, 0) is inside the disk
    assert not erode_disk(no_axis, 2).any()
    no_diag = block.copy()
    no_diag[1, 1] = False  # (-1, -1) is inside the disk
    assert not erode_disk(no_diag, 2).any()


def test_erosion_at_the_image_border_treats_outside_as_background() -> None:
    full = np.ones((5, 5), dtype=np.bool_)
    np.testing.assert_array_equal(erode_disk(full, 1)[1:4, 1:4], np.ones((3, 3), dtype=np.bool_))
    assert erode_disk(full, 1).sum() == 9  # the border ring erodes
    assert erode_disk(full, 2).sum() == 1 and erode_disk(full, 2)[2, 2]
    corner = picture(["##.", "##.", "..."])
    assert not erode_disk(corner, 1).any()  # (0,0) touches the frame edge on two sides


def test_erosion_does_not_wrap_around() -> None:
    stripe = np.zeros((5, 7), dtype=np.bool_)
    stripe[1:4, :] = True
    expected = np.zeros((5, 7), dtype=np.bool_)
    expected[2, 1:6] = True  # a cyclic implementation would also keep (2,0) and (2,6)
    np.testing.assert_array_equal(erode_disk(stripe, 1), expected)


@pytest.mark.parametrize("radius", [0, 1, 2, 3])
def test_erosion_matches_the_brute_force_reference_on_random_masks(radius: int) -> None:
    rng = np.random.default_rng(radius)
    for _ in range(4):
        mask = rng.random((9, 11)) < 0.8
        np.testing.assert_array_equal(erode_disk(mask, radius), erode_reference(mask, radius))


def test_erosion_refuses_bad_input() -> None:
    with pytest.raises(CompositeError):
        erode_disk(np.ones((3, 3), dtype=np.uint8), 1)
    with pytest.raises(CompositeError):
        erode_disk(np.ones((3, 3), dtype=np.bool_), -1)


# -- 6-9: the hard foreground ------------------------------------------------


@pytest.mark.parametrize(("value", "inside"), [(249, False), (250, True), (255, True)])
def test_threshold_boundary_249_out_250_in(value: int, inside: bool) -> None:
    alpha = plane(0)
    alpha[1, 1] = value
    assert bool(hard_foreground_mask(alpha, radius=0)[1, 1]) is inside


def test_hard_foreground_is_strictly_binary() -> None:
    alpha = np.arange(256, dtype=np.uint8).reshape(16, 16)
    hard = hard_foreground_mask(alpha, radius=0)
    assert hard.dtype == np.bool_ and hard.sum() == 6  # 250..255
    eff = hard_effective_alpha(hard)
    assert eff.dtype == np.uint8 and set(np.unique(eff).tolist()) <= {0, 255}
    assert (eff[hard] == 255).all() and (eff[~hard] == 0).all()


def test_original_alpha_128_to_249_is_never_foreground() -> None:
    alpha = np.full((7, 7), 249, dtype=np.uint8)
    alpha[3, 3] = 128
    assert not hard_foreground_mask(alpha, radius=0).any()
    assert not hard_foreground_mask(alpha).any()


def test_alpha_250_plus_survives_only_if_it_survives_erosion() -> None:
    alpha = np.zeros((7, 7), dtype=np.uint8)
    alpha[1:6, 1:6] = 255
    hard = hard_foreground_mask(alpha)  # default radius 2
    assert hard.sum() == 1 and hard[3, 3]
    alpha[1, 3] = 249  # one edge pixel drops below the threshold: the centre loses its disk
    assert not hard_foreground_mask(alpha).any()
    alpha[1, 3] = 250
    assert hard_foreground_mask(alpha)[3, 3]


def test_hard_foreground_refuses_bad_input() -> None:
    with pytest.raises(CompositeError):
        hard_foreground_mask(plane(0).astype(np.int16))
    with pytest.raises(CompositeError):
        hard_foreground_mask(plane(0), threshold=0)
    not_a_mask: Any = plane(0)
    with pytest.raises(CompositeError):
        hard_effective_alpha(not_a_mask)


# -- 10: the region math -------------------------------------------------------


def test_recovery_region_is_removal_and_not_hard_foreground_exactly() -> None:
    source_alpha = plane(0)
    source_alpha[1, 1] = 200  # old person
    replacement_alpha = plane(0)
    replacement_alpha[0:3, 1:4] = 255  # 3x3 core: eroded by 1 -> (1,2)
    replacement_alpha[3, 0] = 200  # a soft pixel outside the core

    regions = hard_inset_regions(
        source_alpha, replacement_alpha, dilation_radius=1, erosion_radius=1
    )

    removal = source_removal_mask(source_alpha, threshold=32, dilation_radius=1)
    assert isinstance(regions, HardInsetRegions)
    np.testing.assert_array_equal(regions.removal, removal)
    assert regions.hard_foreground.sum() == 1 and regions.hard_foreground[1, 2]
    np.testing.assert_array_equal(regions.recovery_region, removal & ~regions.hard_foreground)
    recovery = regions.recovery_region
    np.testing.assert_array_equal(regions.own_background, recovery & (source_alpha < 32))
    np.testing.assert_array_equal(regions.needs_temporal, recovery & (source_alpha >= 32))
    np.testing.assert_array_equal(regions.dropped_ring, recovery & (replacement_alpha >= 128))
    assert regions.dropped_ring[1, 1] and regions.dropped_ring[0, 1]
    assert not regions.dropped_ring[1, 2]
    assert not regions.dropped_ring[3, 0]  # outside removal (radius 1 from (1,1))
    with pytest.raises(CompositeError):
        hard_inset_regions(source_alpha, replacement_alpha, background_threshold=0)


def test_region_defaults_are_v7_removal_and_v8_foreground() -> None:
    regions = hard_inset_regions(plane(40), plane(255))
    assert regions.removal.all()  # alpha 40 >= 32 is core at the v7 threshold
    assert not regions.hard_foreground.any()  # 4x5 cannot survive a 2 px erosion
    assert regions.recovery_region.all() and regions.needs_temporal.all()


# -- 11-16: precedence inside the recovery region -------------------------------


def test_dropped_128_249_pixels_inside_removal_are_background_recovered() -> None:
    # old person at P forever; replacement alpha 200 at P (v6 would force it opaque)
    sources = [source_frame(P)] * 3
    replacements = [replacement_frame(P, alpha=200)] * 3
    frames, stats = run_clip(sources, replacements)

    assert pixel(frames, 1) == BG  # spatially filled real background
    assert pixel(frames, 1) not in (PERSON, O1_PERSON, O1_BG)
    assert stats.dropped_ring_ratio == pytest.approx(1 / (H * W))
    assert stats.dropped_128_249_ratio == pytest.approx(1 / (H * W))
    assert stats.recovered_dropped_ring_ratio == 1.0
    assert stats.ring_spatial_ratio == 1.0 and stats.ring_o1_ratio == 0.0


def test_the_dropped_ring_never_exposes_the_old_source_rgb() -> None:
    sources = [source_frame(P)] * 3
    for src in sources:  # a solid 2x2 of old person, replacement alpha 200 over all of it
        src[1][1:3, 1:3] = 220
        src[0][1:3, 1:3] = PERSON
    replacements = [replacement_frame()] * 3
    for rep in replacements:
        rep[1][1:3, 1:3] = 200
        rep[0][1:3, 1:3] = O1_PERSON
    frames, stats = run_clip(sources, replacements)

    assert not (frames == PERSON).all(axis=3).any()
    assert not (frames == O1_PERSON).all(axis=3).any()
    assert (frames == BG).all(axis=3).all()
    assert stats.recovered_dropped_ring_ratio == 1.0


def test_own_frame_background_has_precedence_in_the_ring() -> None:
    # (2,3) is real background in the source (alpha 0) but inside removal via dilation,
    # and the replacement has alpha 200 there: the source pixel must be kept byte for byte
    sources = [source_frame(P)] * 3
    for src in sources:
        src[0][2, 3] = (1, 2, 3)
    replacements = [replacement_frame((2, 3), alpha=200)] * 3
    frames, stats = run_clip(sources, replacements, dilation_radius=4)

    assert pixel(frames, 1, (2, 3)) == (1, 2, 3)
    assert stats.ring_own_ratio == 1.0 and stats.recovered_dropped_ring_ratio == 1.0


def test_temporal_recovery_has_precedence_over_spatial_in_the_ring() -> None:
    sources = [source_frame(), source_frame(P), source_frame()]
    sources[0][0][P] = (7, 8, 9)  # the real background at P, visible in frames 0 and 2
    sources[2][0][P] = (7, 8, 9)
    replacements = [replacement_frame(P, alpha=200)] * 3
    frames, stats = run_clip(sources, replacements)

    assert pixel(frames, 1) == (7, 8, 9)  # borrowed, not the (BG) spatial mean
    assert stats.ring_temporal_ratio == 1.0
    assert stats.temporal_recovered_ratio == pytest.approx(1 / (H * W) / 3)


def test_spatial_fill_has_precedence_over_o1_in_the_ring() -> None:
    sources = [source_frame(P)] * 3
    replacements = [replacement_frame(P, alpha=200)] * 3
    frames, stats = run_clip(sources, replacements)
    assert pixel(frames, 1) == BG and stats.ring_spatial_ratio == 1.0
    assert stats.spatial_recovered_ratio == pytest.approx(1 / (H * W))


def test_o1_is_only_the_final_residual_fallback() -> None:
    sources = [source_frame(P)] * 3
    for src in sources:  # whole frame is old person: no trusted seed anywhere
        src[1][:] = 255
        src[0][:] = PERSON
    replacements = [replacement_frame()] * 3
    frames, stats = run_clip(sources, replacements)

    assert (frames == O1_BG).all(axis=3).all()
    assert not (frames == PERSON).all(axis=3).any()
    assert stats.o1_fallback_ratio == 1.0 and stats.spatial_unrecovered_ratio == 1.0
    assert stats.components_without_seed == 3


# -- 17-20: what the final composite is made of -------------------------------------


def test_outside_removal_and_outside_hard_foreground_preserves_the_source() -> None:
    sources = [source_frame(P, alpha=200)] * 3
    replacements = [replacement_frame((3, 4), alpha=100)] * 3  # soft, outside removal
    frames, _ = run_clip(sources, replacements, dilation_radius=0)
    removal = source_removal_mask(sources[0][1], threshold=32, dilation_radius=0)
    for i in range(3):
        np.testing.assert_array_equal(frames[i][~removal], sources[i][0][~removal])
    assert pixel(frames, 1, (3, 4)) == BG  # the soft alpha did nothing


def test_inside_hard_foreground_the_replacement_is_copied_exactly() -> None:
    sources = [source_frame(P)] * 3
    replacements = [replacement_frame()] * 3
    for rep in replacements:
        rep[1][0:3, 0:3] = 255
        rep[0][0:3, 0:3] = (250, 251, 252)
    frames, stats = run_clip(sources, replacements, erosion_radius=1)

    assert pixel(frames, 1) == (250, 251, 252)  # (1,1) is the eroded core
    assert stats.hard_foreground_ratio == pytest.approx(1 / (H * W))
    assert pixel(frames, 1, (0, 0)) == BG  # the rest of the 3x3 was dropped and rebuilt


def test_no_replacement_soft_alpha_survives() -> None:
    sources = [source_frame(P)] * 3
    replacements = [replacement_frame()] * 3
    for rep in replacements:
        rep[1][:] = 249  # everything just below the threshold
        rep[0][:] = O1_PERSON
    frames, stats = run_clip(sources, replacements)
    assert (frames == BG).all(axis=3).all()
    assert stats.hard_foreground_ratio == 0.0
    assert stats.dropped_replacement_ratio == 1.0 and stats.dropped_128_249_ratio == 1.0


def test_v8_has_no_force_replacement_semantics_unlike_v6() -> None:
    sources = [source_frame(P)] * 3
    replacements = [replacement_frame(P, alpha=200)] * 3
    v6, _ = run_clip_v6(sources, replacements)
    v8, _ = run_clip(sources, replacements)
    assert pixel(v6, 1) == O1_PERSON  # v6 forces alpha 200 inside removal to 255
    assert pixel(v8, 1) == BG  # v8 drops it and rebuilds real background


# -- diagnostics ------------------------------------------------------------------


def test_ring_shares_partition_the_dropped_ring() -> None:
    sources = [source_frame(), source_frame(P), source_frame()]
    for src in sources:
        src[0][2, 3] = (1, 2, 3)  # own background inside the ring
    replacements = [replacement_frame()] * 3
    for rep in replacements:
        rep[1][P] = 200
        rep[1][2, 3] = 200
    _, stats = run_clip(sources, replacements, dilation_radius=4)
    assert isinstance(stats, HardInsetStreamStats)
    shares = (
        stats.ring_own_ratio,
        stats.ring_temporal_ratio,
        stats.ring_spatial_ratio,
        stats.ring_o1_ratio,
    )
    assert sum(shares) == pytest.approx(1.0)
    # frames 0 and 2 have no old person, so no removal mask and no ring; in
    # frame 1 the ring is {P, (2,3)}: P borrowed temporally, (2,3) own background
    assert stats.ring_own_ratio == pytest.approx(1 / 2)
    assert stats.ring_temporal_ratio == pytest.approx(1 / 2)
    assert stats.dropped_ring_ratio == pytest.approx(2 / (H * W) / 3)
    assert stats.recovered_dropped_ring_ratio == pytest.approx(1.0)


def test_v5_v6_diagnostics_are_preserved() -> None:
    sources = [source_frame(), source_frame(P), source_frame()]
    _, stats = run_clip(sources, [replacement_frame()] * 3)
    assert (stats.median_donor_distance, stats.p90_donor_distance) == (1.0, 1.0)
    assert stats.mean_observations_per_recovered_pixel == 2.0
    assert (stats.donor_fits, stats.zero_offset_fallbacks) == (2, 0)
    assert stats.peak_cached_frames == 3
    assert stats.spatial_components == 0 and math.isnan(stats.median_propagation_depth)
    assert stats.max_propagation_depth == 0


def test_propagation_depth_diagnostics() -> None:
    sources = [source_frame()] * 3
    for src in sources:
        src[1][0:3, 1:4] = 200
        src[0][0:3, 1:4] = PERSON
    _, stats = run_clip(sources, [replacement_frame()] * 3)
    assert stats.spatial_components == 3
    assert (stats.median_propagation_depth, stats.p90_propagation_depth) == (1.0, 2.0)
    assert stats.max_propagation_depth == 2


def test_zero_frames_writes_nothing() -> None:
    out = io.BytesIO()
    stats = run_streams({name: io.BytesIO() for name in STREAMS}, 0, out)
    assert out.getvalue() == b""
    assert stats.hard_foreground_ratio == 0.0 and stats.peak_cached_frames == 0
    assert stats.recovered_dropped_ring_ratio == 0.0
    assert math.isnan(stats.median_donor_distance)


# -- streaming: errors ---------------------------------------------------------------


def test_parameters_are_checked_before_any_stream_is_read() -> None:
    streams = clip_streams([source_frame()] * 2, [replacement_frame()] * 2)
    bad_params: list[dict[str, Any]] = [
        {"radius": -1},
        {"max_observations": 0},
        {"background_threshold": 0},
        {"foreground_threshold": 256},
        {"erosion_radius": -1},
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


# -- orchestration, against the fake ffmpeg ------------------------------------------


def run_recovery(tmp_path: Path, out: Path, **kwargs: Any) -> HardInsetCompositeReport:
    return composite_video_hard_inset_recovery(
        tmp_path / "src.mp4",
        tmp_path / "rep.mp4",
        tmp_path / "src.webm",
        tmp_path / "rep.webm",
        out,
        **kwargs,
    )


def test_composite_video_hard_inset_recovery_matches_the_streaming_function(
    tmp_path: Path, monkeypatch: Any
) -> None:
    sources = [source_frame(P), source_frame(P), source_frame(P)]
    sources[1][0][3, 0] = PERSON  # (3,0) hidden only in frame 1: temporally recovered
    sources[1][1][3, 0] = 200
    replacements = [replacement_frame(), replacement_frame(P, alpha=200), replacement_frame()]
    ffmpeg = FakeFfmpeg(scene_outputs(sources, replacements))
    wire(monkeypatch, ffmpeg, fake_probe(3))
    out = tmp_path / "nested" / "v8.mp4"

    report = run_recovery(tmp_path, out, erosion_radius=0)

    expected, expected_stats = run_clip(
        sources, replacements, dilation_radius=4, offset_stride=8, offset_min_samples=256
    )
    assert ffmpeg.encoded == expected.tobytes()
    assert report.stats == expected_stats
    written = np.frombuffer(ffmpeg.encoded, np.uint8).reshape(3, H, W, 3)
    assert pixel(written, 1) == BG and pixel(written, 1, (3, 0)) == BG
    assert not math.isnan(report.stats.median_donor_distance)
    assert report.output_path == out
    assert (report.frames, report.width, report.height) == (3, W, H)
    assert report.frame_rate == Fraction(24, 1)
    assert (report.removal_threshold, report.dilation_radius) == (32, 4)
    assert (report.background_threshold, report.foreground_threshold) == (32, 250)
    assert report.erosion_radius == 0
    assert (report.radius, report.max_observations) == (24, 5)
    assert out.parent.is_dir()
    assert not any(p.killed for p in ffmpeg.processes)


def test_orchestration_defaults_are_the_v8_poc_parameters(tmp_path: Path, monkeypatch: Any) -> None:
    ffmpeg = FakeFfmpeg(default_outputs(2))
    wire(monkeypatch, ffmpeg, fake_probe(2))
    report = run_recovery(tmp_path, tmp_path / "v8.mp4")
    assert (report.removal_threshold, report.dilation_radius) == (32, 4)
    assert (report.foreground_threshold, report.erosion_radius) == (250, 2)
    assert report.stats.hard_foreground_ratio == 0.0  # 4x5 cannot survive a 2 px erosion


def test_hard_inset_decoders_force_libvpx_for_both_mattes_only(
    tmp_path: Path, monkeypatch: Any
) -> None:
    ffmpeg = FakeFfmpeg(default_outputs(2))
    wire(monkeypatch, ffmpeg, fake_probe(2))
    out = tmp_path / "v8.mp4"

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


def test_invalid_parameters_are_refused_before_anything_is_probed(
    tmp_path: Path, monkeypatch: Any
) -> None:
    ffmpeg = FakeFfmpeg(default_outputs(1))

    def never(path: Path, decoder: str | None = None) -> VideoInfo:
        raise AssertionError(f"probed {path}")

    wire(monkeypatch, ffmpeg, never)

    for bad in ({"radius": -1}, {"erosion_radius": -1}, {"foreground_threshold": 0}):
        with pytest.raises(CompositeError):
            run_recovery(tmp_path, tmp_path / "v8.mp4", **bad)
    assert ffmpeg.commands == []


def test_a_matte_without_alpha_stops_the_run_before_any_process_spawns(
    tmp_path: Path, monkeypatch: Any
) -> None:
    ffmpeg = FakeFfmpeg(default_outputs(2))
    probe = fake_probe(2)

    def alpha_lost(path: Path, decoder: str | None = None) -> VideoInfo:
        result = probe(path, decoder)
        return replace(result, pix_fmt="yuv420p") if path.name == "rep.webm" else result

    wire(monkeypatch, ffmpeg, alpha_lost)

    with pytest.raises(CompositeError, match="replacement matte: matte .* has no alpha"):
        run_recovery(tmp_path, tmp_path / "v8.mp4")
    assert ffmpeg.commands == []


def test_a_missing_input_is_rejected_before_ffprobe_runs(tmp_path: Path) -> None:
    with pytest.raises(CompositeError, match="input not found"):
        run_recovery(tmp_path, tmp_path / "v8.mp4")


@pytest.mark.parametrize("name", STREAMS)
def test_a_failing_decoder_is_reported_with_its_name_and_stderr(
    name: str, tmp_path: Path, monkeypatch: Any
) -> None:
    ffmpeg = FakeFfmpeg(default_outputs(2), failing=FILES[name], stderr="could not decode")
    wire(monkeypatch, ffmpeg, fake_probe(2))

    with pytest.raises(CompositeError, match=f"ffmpeg {name} exited 1: could not decode"):
        run_recovery(tmp_path, tmp_path / "v8.mp4")


def test_a_failing_encoder_is_reported(tmp_path: Path, monkeypatch: Any) -> None:
    ffmpeg = FakeFfmpeg(default_outputs(2), failing="encode", stderr="broken pipe")
    wire(monkeypatch, ffmpeg, fake_probe(2))

    with pytest.raises(CompositeError, match="ffmpeg encode exited 1: broken pipe"):
        run_recovery(tmp_path, tmp_path / "v8.mp4")


def test_a_decoder_that_dies_mid_stream_aborts_and_kills_the_rest(
    tmp_path: Path, monkeypatch: Any
) -> None:
    outputs = default_outputs(3)
    outputs["src.mp4"] = frames_bytes([1])
    ffmpeg = FakeFfmpeg(outputs, failing="src.mp4", stderr="decode error")
    wire(monkeypatch, ffmpeg, fake_probe(3))

    with pytest.raises(CompositeError, match="source ended during frame 1"):
        run_recovery(tmp_path, tmp_path / "v8.mp4")
    assert all(p.killed for p in ffmpeg.processes)


def test_hard_inset_api_is_exported_from_the_package() -> None:
    package: Any = video_character_skill
    assert package.composite_video_hard_inset_recovery is composite_video_hard_inset_recovery
    assert package.HardInsetCompositeReport is HardInsetCompositeReport
    assert package.HARD_FOREGROUND_THRESHOLD == 250
    assert package.HARD_FOREGROUND_EROSION_RADIUS == 2
    assert package.HARD_INSET_REMOVAL_THRESHOLD == 32
    assert "composite_video_hard_inset_recovery" in video_character_skill.__all__


# -- v6 keeps its semantics ---------------------------------------------------------


def test_v6_defaults_are_untouched_by_v8() -> None:
    from video_character_skill.spatial_recovery import composite_streams_spatial_recovery
    from video_character_skill.temporal_recovery import recovery_regions

    spatial_defaults = composite_streams_spatial_recovery.__kwdefaults__
    region_defaults = recovery_regions.__kwdefaults__
    assert spatial_defaults is not None and region_defaults is not None
    assert spatial_defaults["removal_threshold"] == 64
    assert region_defaults["foreground_threshold"] == 128
    assert video_character_skill.SOURCE_REMOVAL_THRESHOLD == 64


# -- helpers --------------------------------------------------------------------------


def run_streams(
    streams: dict[str, io.BytesIO], frames: int, out: io.BytesIO | None = None, **kwargs: Any
) -> HardInsetStreamStats:
    return composite_streams_hard_inset_recovery(
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
) -> tuple[NDArray[np.uint8], HardInsetStreamStats]:
    out = io.BytesIO()
    params = {**V8_TEST, **overrides}
    stats = run_streams(clip_streams(sources, replacements), len(sources), out, **params)
    frames = np.frombuffer(out.getvalue(), np.uint8).reshape(len(sources), H, W, 3)
    return frames, stats
