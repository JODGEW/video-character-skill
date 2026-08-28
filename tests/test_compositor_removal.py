"""Unit tests for the source-removal compositor (v4).

The removal mask and the effective alpha are pure functions over arrays, the
streaming loop runs on file-like objects, and the ffmpeg pipeline runs
against the fake ``subprocess.Popen`` shared with the v2/v3 tests. Nothing
here decodes, encodes or writes a video. The v1/v2/v3 test modules are
untouched; the last section pins down that v3 is not affected by v4.
"""

from __future__ import annotations

import io
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
    matte_bytes,
    plane,
    rgb,
    wire,
)
from video_character_skill.compositor import (
    SOURCE_REMOVAL_DILATION_RADIUS,
    SOURCE_REMOVAL_THRESHOLD,
    CompositeError,
    SourceRemovalCompositeReport,
    SourceRemovalStreamStats,
    VideoInfo,
    composite_frame,
    composite_streams_source_removal,
    composite_video_hardened_union,
    composite_video_source_removal,
    dilate_disk,
    hardened_union_alpha,
    removal_effective_alpha,
    source_removal_mask,
)

T = SOURCE_REMOVAL_THRESHOLD
R = SOURCE_REMOVAL_DILATION_RADIUS


def blend(a: int, rep: int, src: int) -> int:
    return (a * rep + (255 - a) * src + 127) // 255


def point(shape: tuple[int, int], y: int, x: int) -> NDArray[np.bool_]:
    mask: NDArray[np.bool_] = np.zeros(shape, dtype=np.bool_)
    mask[y, x] = True
    return mask


def disk_offsets(radius: int) -> set[tuple[int, int]]:
    """The reference definition of the disk, written independently of the code."""
    return {
        (dy, dx)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
        if dy * dy + dx * dx <= radius * radius
    }


def set_pixels(mask: NDArray[np.bool_], origin: tuple[int, int] = (0, 0)) -> set[tuple[int, int]]:
    ys, xs = np.nonzero(mask)
    return {(int(y) - origin[0], int(x) - origin[1]) for y, x in zip(ys, xs, strict=True)}


def test_the_constants_are_exactly_64_and_4() -> None:
    assert SOURCE_REMOVAL_THRESHOLD == 64
    assert SOURCE_REMOVAL_DILATION_RADIUS == 4
    assert (T, R) == (64, 4)


# -- dilate_disk: geometry ----------------------------------------------


def test_radius_0_is_the_identity_as_a_copy() -> None:
    rng = np.random.default_rng(0)
    mask = rng.random((H, W)) < 0.3

    out = dilate_disk(mask, 0)

    np.testing.assert_array_equal(out, mask)
    assert out is not mask
    assert not np.shares_memory(out, mask)


def test_radius_1_is_a_plus_not_a_3x3_square() -> None:
    out = dilate_disk(point((5, 5), 2, 2), 1)

    expected = np.zeros((5, 5), dtype=np.bool_)
    expected[2, 1:4] = True
    expected[1:4, 2] = True
    np.testing.assert_array_equal(out, expected)
    assert int(out.sum()) == 5
    assert not (out[1, 1] or out[1, 3] or out[3, 1] or out[3, 3])


def test_radius_4_is_the_49_pixel_euclidean_disk() -> None:
    out = dilate_disk(point((11, 11), 5, 5), 4)

    got = set_pixels(out, origin=(5, 5))
    assert got == disk_offsets(4)
    assert len(got) == 49
    # on the axis at exactly radius 4, and interior points at distance sqrt(13), sqrt(10)
    assert out[9, 5] and out[5, 1] and out[8, 7] and out[7, 8] and out[6, 8]


def test_radius_4_excludes_square_kernel_corners_outside_the_radius() -> None:
    out = dilate_disk(point((11, 11), 5, 5), 4)

    for dy, dx in ((4, 4), (3, 3), (4, 3), (4, 2), (2, 4), (3, 4), (4, 1), (1, 4)):
        assert dy * dy + dx * dx > 16
        for sy in (1, -1):
            for sx in (1, -1):
                assert not out[5 + sy * dy, 5 + sx * dx]
    assert int(out.sum()) == 49 < 81  # a 9x9 square would be 81


@pytest.mark.parametrize("radius", [1, 2, 3, 4, 6])
def test_dilation_matches_the_offset_definition_on_random_masks(radius: int) -> None:
    rng = np.random.default_rng(radius)
    mask = rng.random((24, 30)) < 0.05
    expected = np.zeros_like(mask)
    ys, xs = np.nonzero(mask)
    for y, x in zip(ys, xs, strict=True):
        for dy, dx in disk_offsets(radius):
            yy, xx = int(y) + dy, int(x) + dx
            if 0 <= yy < 24 and 0 <= xx < 30:
                expected[yy, xx] = True

    np.testing.assert_array_equal(dilate_disk(mask, radius), expected)


def test_dilation_does_not_wrap_around_the_image_border() -> None:
    top_left = dilate_disk(point((12, 12), 0, 0), 4)
    assert top_left[0, 0] and top_left[4, 0] and top_left[0, 4] and not top_left[5, 0]
    assert not top_left[-4:, :].any()
    assert not top_left[:, -4:].any()

    bottom_right = dilate_disk(point((12, 12), 11, 11), 4)
    assert not bottom_right[:4, :].any()
    assert not bottom_right[:, :4].any()
    quadrant = {(dy, dx) for dy, dx in disk_offsets(4) if dy <= 0 and dx <= 0}
    assert int(bottom_right.sum()) == len(quadrant) == 17


def test_dilation_near_the_border_is_clipped_not_shifted() -> None:
    """A pixel one column from the edge still gets the full disk inward."""
    out = dilate_disk(point((12, 12), 5, 1), 4)

    assert out[5, 0] and out[5, 5] and not out[5, 6]
    assert out[1, 1] and out[9, 1]
    assert int(out.sum()) == len({(dy, dx) for dy, dx in disk_offsets(4) if dx >= -1})


def test_dilation_on_a_frame_smaller_than_the_radius_is_safe() -> None:
    out = dilate_disk(point((2, 3), 0, 1), 4)
    assert out.all()


def test_dilation_of_an_empty_mask_is_empty() -> None:
    assert not dilate_disk(np.zeros((H, W), dtype=np.bool_), 4).any()


def test_dilation_does_not_mutate_its_input() -> None:
    mask = point((9, 9), 4, 4)
    before = mask.copy()

    dilate_disk(mask, 3)

    np.testing.assert_array_equal(mask, before)


def test_dilate_disk_refuses_bad_inputs() -> None:
    with pytest.raises(CompositeError, match="mask must be bool"):
        dilate_disk(np.zeros((H, W), np.uint8), 1)
    with pytest.raises(CompositeError, match=r"mask must be \(H, W\)"):
        dilate_disk(np.zeros((H, W, 1), np.bool_), 1)
    with pytest.raises(CompositeError, match="dilation_radius must be >= 0"):
        dilate_disk(np.zeros((H, W), np.bool_), -1)
    bad: Any = 1.0
    with pytest.raises(CompositeError, match="dilation_radius must be an int"):
        dilate_disk(np.zeros((H, W), np.bool_), bad)


# -- source_removal_mask ------------------------------------------------


def test_threshold_boundary_63_vs_64_with_no_dilation() -> None:
    source = plane(0)
    source[0, 0], source[0, 1] = 63, 64

    out = source_removal_mask(source, dilation_radius=0)

    assert not out[0, 0]
    assert out[0, 1]
    assert int(out.sum()) == 1


def test_threshold_boundary_63_vs_64_with_the_default_dilation() -> None:
    source = plane(0)
    source[1, 1] = 63
    assert not source_removal_mask(source).any()

    source[1, 1] = 64
    assert source_removal_mask(source)[1, 1]


def test_source_alpha_1_to_63_alone_never_enters_the_core() -> None:
    rng = np.random.default_rng(2)
    source = rng.integers(1, 64, (H, W), dtype=np.uint8)
    source[0, 0] = 63

    assert not source_removal_mask(source).any()
    assert not source_removal_mask(source, dilation_radius=8).any()


def test_default_mask_is_the_core_dilated_by_a_radius_4_disk() -> None:
    source = np.zeros((11, 11), dtype=np.uint8)
    source[5, 5] = 64

    out = source_removal_mask(source)

    np.testing.assert_array_equal(out, dilate_disk(source >= 64, 4))
    assert set_pixels(out, origin=(5, 5)) == disk_offsets(4)


def test_dilation_captures_pixels_whose_source_alpha_is_0() -> None:
    source = np.zeros((11, 11), dtype=np.uint8)
    source[5, 5] = 64

    out = source_removal_mask(source)

    assert out[5, 9] and out[1, 5] and out[8, 7]
    assert source[5, 9] == 0 and source[1, 5] == 0 and source[8, 7] == 0
    assert int((out & (source == 0)).sum()) == 48


def test_the_alpha_value_above_the_threshold_is_irrelevant() -> None:
    """Binary support: 64 and 255 yield the identical mask."""
    low = np.zeros((11, 11), dtype=np.uint8)
    low[5, 5] = 64
    high = low.copy()
    high[5, 5] = 255

    np.testing.assert_array_equal(source_removal_mask(low), source_removal_mask(high))


def test_custom_threshold_and_radius_are_honoured() -> None:
    source = plane(0)
    source[1, 1] = 100

    assert not source_removal_mask(source, threshold=101, dilation_radius=0).any()
    assert int(source_removal_mask(source, threshold=100, dilation_radius=0).sum()) == 1
    assert int(source_removal_mask(source, threshold=100, dilation_radius=1).sum()) == 5


def test_mask_accepts_read_only_planes() -> None:
    source = np.frombuffer(bytes([64] * (H * W)), np.uint8).reshape(H, W)
    assert source_removal_mask(source).all()


def test_source_removal_mask_refuses_bad_inputs() -> None:
    with pytest.raises(CompositeError, match="source alpha must be uint8"):
        source_removal_mask(np.zeros((H, W), np.float32))
    with pytest.raises(CompositeError, match=r"source alpha must be \(H, W\)"):
        source_removal_mask(np.zeros((H, W, 1), np.uint8))
    for threshold in (0, 256, -5):
        with pytest.raises(CompositeError, match="threshold must be in 1..255"):
            source_removal_mask(plane(0), threshold=threshold)
    for bad_threshold in (64.0, True, "64"):
        bad: Any = bad_threshold
        with pytest.raises(CompositeError, match="^threshold must be an int"):
            source_removal_mask(plane(0), threshold=bad)
    with pytest.raises(CompositeError, match="dilation_radius must be >= 0"):
        source_removal_mask(plane(0), dilation_radius=-1)
    for bad_radius in (4.0, True, None):
        bad = bad_radius
        with pytest.raises(CompositeError, match="dilation_radius must be an int"):
            source_removal_mask(plane(0), dilation_radius=bad)


# -- removal_effective_alpha --------------------------------------------


def test_inside_removal_effective_alpha_is_exactly_255() -> None:
    rng = np.random.default_rng(3)
    replacement = rng.integers(0, 255, (H, W), dtype=np.uint8)  # never 255 on its own
    source = plane(0)
    source[:, :2] = 64

    out = removal_effective_alpha(source, replacement, dilation_radius=0)

    np.testing.assert_array_equal(out[:, :2], 255)
    assert out.dtype == np.uint8


def test_outside_removal_replacement_alpha_is_unchanged_byte_for_byte() -> None:
    rng = np.random.default_rng(4)
    replacement = rng.integers(0, 256, (H, W), dtype=np.uint8)
    source = rng.integers(0, 64, (H, W), dtype=np.uint8)  # sub-threshold noise everywhere
    source[:, 0] = 200

    out = removal_effective_alpha(source, replacement, dilation_radius=0)

    np.testing.assert_array_equal(out[:, 1:], replacement[:, 1:])
    np.testing.assert_array_equal(out[:, 0], 255)


def test_source_matte_is_only_a_binary_support() -> None:
    """Changing source alpha inside the core, or below the threshold, changes nothing."""
    rng = np.random.default_rng(5)
    replacement = rng.integers(0, 256, (H, W), dtype=np.uint8)
    a = plane(0)
    a[:, :2] = 64
    b = a.copy()
    b[:, :2] = 255  # same core, different values
    b[:, 3] = 63  # below threshold: still not a core pixel

    np.testing.assert_array_equal(
        removal_effective_alpha(a, replacement), removal_effective_alpha(b, replacement)
    )


def test_partial_source_alpha_is_never_carried_into_the_effective_alpha() -> None:
    """No max(): s=50 is below the threshold, so r=20 stays 20 (v3 gives 50)."""
    assert np.unique(removal_effective_alpha(plane(50), plane(20))).tolist() == [20]
    assert np.unique(hardened_union_alpha(plane(50), plane(20))).tolist() == [50]
    # In the core the answer is 255, not the source's 200 and not max(200, 20).
    assert np.unique(removal_effective_alpha(plane(200), plane(20))).tolist() == [255]


def test_matches_the_documented_formula_exactly() -> None:
    rng = np.random.default_rng(6)
    source = rng.integers(0, 256, (H * 4, W * 4), dtype=np.uint8)
    replacement = rng.integers(0, 256, (H * 4, W * 4), dtype=np.uint8)
    expected = replacement.copy()
    expected[dilate_disk(source >= 64, 4)] = 255

    np.testing.assert_array_equal(removal_effective_alpha(source, replacement), expected)


def test_dilated_pixels_with_source_alpha_0_become_255() -> None:
    source = np.zeros((11, 11), dtype=np.uint8)
    source[5, 5] = 64
    replacement = np.zeros((11, 11), dtype=np.uint8)

    out = removal_effective_alpha(source, replacement)

    disk = dilate_disk(source >= 64, 4)
    np.testing.assert_array_equal(out[disk], 255)
    np.testing.assert_array_equal(out[~disk], 0)
    assert int((out == 255).sum()) == 49


def test_alpha_zero_true_background_remains_exact_source_bytes() -> None:
    rng = np.random.default_rng(7)
    source_rgb = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    replacement_rgb = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    source_alpha = plane(0)
    source_alpha[:, 0] = 64  # old person in column 0; radius 1 reaches column 1
    replacement_alpha = plane(0)  # no replacement person at all

    effective = removal_effective_alpha(source_alpha, replacement_alpha, dilation_radius=1)
    out = composite_frame(source_rgb, replacement_rgb, effective)

    np.testing.assert_array_equal(effective[:, :2], 255)
    np.testing.assert_array_equal(effective[:, 2:], 0)
    np.testing.assert_array_equal(out[:, :2], replacement_rgb[:, :2])  # O1 background
    np.testing.assert_array_equal(out[:, 2:], source_rgb[:, 2:])  # untouched source


def test_inputs_are_not_mutated_and_the_output_is_new() -> None:
    source, replacement = plane(64), plane(20)
    s0, r0 = source.copy(), replacement.copy()

    out = removal_effective_alpha(source, replacement)

    np.testing.assert_array_equal(source, s0)
    np.testing.assert_array_equal(replacement, r0)
    assert not np.shares_memory(out, source)
    assert not np.shares_memory(out, replacement)


def test_effective_alpha_refuses_mismatched_planes() -> None:
    with pytest.raises(CompositeError, match="replacement alpha shape"):
        removal_effective_alpha(plane(0), np.zeros((H, W + 1), np.uint8))
    with pytest.raises(CompositeError, match="replacement alpha must be uint8"):
        removal_effective_alpha(plane(0), np.zeros((H, W), np.float32))
    with pytest.raises(CompositeError, match="threshold must be in 1..255"):
        removal_effective_alpha(plane(0), plane(0), threshold=0)
    with pytest.raises(CompositeError, match="dilation_radius must be >= 0"):
        removal_effective_alpha(plane(0), plane(0), dilation_radius=-2)


# -- streaming ----------------------------------------------------------


def stream_set(n: int) -> dict[str, io.BytesIO]:
    return {
        "source": io.BytesIO(frames_bytes(list(range(1, n + 1)))),
        "replacement": io.BytesIO(frames_bytes(list(range(101, 101 + n)))),
        "source_matte": io.BytesIO(matte_bytes([64] * n)),
        "replacement_matte": io.BytesIO(matte_bytes([0] * n)),
    }


def run_streams(
    streams: dict[str, io.BytesIO],
    frames: int,
    out: io.BytesIO | None = None,
    **kwargs: Any,
) -> SourceRemovalStreamStats:
    return composite_streams_source_removal(
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


def matte_plane_bytes(alpha: NDArray[np.uint8]) -> bytes:
    frame = np.zeros((H, W, 4), dtype=np.uint8)
    frame[:, :, :3] = 77
    frame[:, :, 3] = alpha
    return frame.tobytes()


def test_streaming_composites_under_the_removal_mask_frame_by_frame() -> None:
    out = io.BytesIO()

    stats = composite_streams_source_removal(
        io.BytesIO(frames_bytes([10, 20, 30, 40, 50])),
        io.BytesIO(frames_bytes([200, 210, 220, 230, 240])),
        io.BytesIO(matte_bytes([0, 64, 63, 0, 255])),  # source matte
        io.BytesIO(matte_bytes([0, 0, 0, 128, 255])),  # replacement matte
        out,
        width=W,
        height=H,
        frames=5,
    )

    written = np.frombuffer(out.getvalue(), np.uint8).reshape(5, H, W, 3)
    assert np.unique(written[0]).tolist() == [10]  # nothing anywhere -> source
    assert np.unique(written[1]).tolist() == [210]  # in core -> O1, even with r=0
    assert np.unique(written[2]).tolist() == [30]  # 63 is not core -> source
    assert np.unique(written[3]).tolist() == [blend(128, 230, 40)]  # replacement's own edge
    assert np.unique(written[4]).tolist() == [240]  # core, replacement already 255
    assert stats.soft_edge_ratio == pytest.approx(1 / 5)  # frame 3
    assert stats.removal_ratio == pytest.approx(2 / 5)  # frames 1, 4
    assert stats.dilation_only_ratio == 0.0  # uniform frames: nothing to grow into
    assert stats.replacement_override_ratio == pytest.approx(1 / 5)  # frame 1 only


def test_streaming_default_radius_4_reaches_pixels_with_source_alpha_0() -> None:
    source_alpha = plane(0)
    source_alpha[0, 0] = 64
    out = io.BytesIO()

    stats = composite_streams_source_removal(
        io.BytesIO(rgb(0).tobytes()),
        io.BytesIO(rgb(255).tobytes()),
        io.BytesIO(matte_plane_bytes(source_alpha)),
        io.BytesIO(matte_plane_bytes(plane(0))),
        out,
        width=W,
        height=H,
        frames=1,
    )

    disk = {(y, x) for y in range(H) for x in range(W) if y * y + x * x <= 16}
    assert len(disk) == 16
    written = np.frombuffer(out.getvalue(), np.uint8).reshape(H, W, 3)
    for y in range(H):
        for x in range(W):
            assert int(written[y, x, 0]) == (255 if (y, x) in disk else 0)
    assert stats.removal_ratio == pytest.approx(16 / (H * W))
    assert stats.dilation_only_ratio == pytest.approx(15 / (H * W))
    assert stats.replacement_override_ratio == pytest.approx(16 / (H * W))
    assert stats.soft_edge_ratio == 0.0


def test_streaming_reports_dilation_only_and_override_with_radius_1() -> None:
    source_alpha = plane(0)
    source_alpha[1, 1] = 200
    replacement_alpha = plane(0)
    replacement_alpha[1, 2] = 255  # already opaque: not an override
    replacement_alpha[3, 4] = 100  # outside the mask: the replacement's own edge

    stats = run_streams(
        {
            "source": io.BytesIO(rgb(0).tobytes()),
            "replacement": io.BytesIO(rgb(255).tobytes()),
            "source_matte": io.BytesIO(matte_plane_bytes(source_alpha)),
            "replacement_matte": io.BytesIO(matte_plane_bytes(replacement_alpha)),
        },
        1,
        dilation_radius=1,
    )

    assert stats.removal_ratio == pytest.approx(5 / (H * W))  # the plus
    assert stats.dilation_only_ratio == pytest.approx(4 / (H * W))  # plus minus its centre
    assert stats.replacement_override_ratio == pytest.approx(4 / (H * W))  # (1, 2) excluded
    assert stats.soft_edge_ratio == pytest.approx(1 / (H * W))  # (3, 4)


def test_parameters_are_checked_before_any_stream_is_read() -> None:
    streams = stream_set(2)
    with pytest.raises(CompositeError, match="threshold must be in 1..255"):
        run_streams(streams, 2, threshold=0)
    with pytest.raises(CompositeError, match="dilation_radius must be >= 0"):
        run_streams(streams, 2, dilation_radius=-1)
    assert all(stream.tell() == 0 for stream in streams.values())


def test_custom_parameters_flow_through_the_stream() -> None:
    out = io.BytesIO()

    stats = run_streams(stream_set(1), 1, out, threshold=65, dilation_radius=0)

    assert np.unique(np.frombuffer(out.getvalue(), np.uint8)).tolist() == [1]  # 64 < 65
    assert stats.removal_ratio == 0.0


@pytest.mark.parametrize("short", STREAMS)
def test_premature_eof_on_any_of_the_four_streams_names_it(short: str) -> None:
    streams = stream_set(3)
    streams[short] = io.BytesIO(streams[short].getvalue()[: -H * W])

    with pytest.raises(CompositeError, match=f"{short} ended during frame 2"):
        run_streams(streams, 3)


@pytest.mark.parametrize("short", STREAMS)
def test_a_completely_empty_stream_fails_on_frame_0(short: str) -> None:
    streams = stream_set(2)
    streams[short] = io.BytesIO(b"")

    with pytest.raises(CompositeError, match=f"{short} ended during frame 0"):
        run_streams(streams, 2)


def test_no_partial_frame_is_written_when_a_stream_runs_dry() -> None:
    streams = stream_set(2)
    streams["replacement_matte"] = io.BytesIO(streams["replacement_matte"].getvalue()[:-1])
    out = io.BytesIO()

    with pytest.raises(CompositeError, match="replacement_matte ended during frame 1"):
        run_streams(streams, 2, out)

    assert out.getvalue() == frames_bytes([101])  # frame 0 only: removed -> replacement


@pytest.mark.parametrize("long", STREAMS)
def test_extra_data_on_any_of_the_four_streams_names_it(long: str) -> None:
    streams = stream_set(2)
    streams[long] = io.BytesIO(streams[long].getvalue() + b"\0")

    with pytest.raises(CompositeError, match=f"{long} has more than the expected 2"):
        run_streams(streams, 2)


def test_zero_frames_writes_nothing() -> None:
    out = io.BytesIO()

    stats = run_streams({name: io.BytesIO() for name in STREAMS}, 0, out)

    assert stats == SourceRemovalStreamStats(0.0, 0.0, 0.0, 0.0)
    assert out.getvalue() == b""


# -- orchestration, against the fake ffmpeg -----------------------------


def run_removal(tmp_path: Path, out: Path, **kwargs: Any) -> SourceRemovalCompositeReport:
    return composite_video_source_removal(
        tmp_path / "src.mp4",
        tmp_path / "rep.mp4",
        tmp_path / "src.webm",
        tmp_path / "rep.webm",
        out,
        **kwargs,
    )


def test_composite_video_source_removal_streams_four_inputs_into_one_encoder(
    tmp_path: Path, monkeypatch: Any
) -> None:
    ffmpeg = FakeFfmpeg(
        {
            "src.mp4": frames_bytes([10, 20, 30]),
            "rep.mp4": frames_bytes([200, 210, 220]),
            "src.webm": matte_bytes([64, 0, 63]),
            "rep.webm": matte_bytes([0, 128, 0]),
        }
    )
    wire(monkeypatch, ffmpeg, fake_probe(3))
    out = tmp_path / "nested" / "v4.mp4"

    report = run_removal(tmp_path, out)

    assert report.output_path == out
    assert (report.frames, report.width, report.height) == (3, W, H)
    assert report.frame_rate == Fraction(24, 1)
    assert (report.threshold, report.dilation_radius) == (64, 4)
    assert report.soft_edge_ratio == pytest.approx(1 / 3)  # frame 1
    assert report.removal_ratio == pytest.approx(1 / 3)  # frame 0
    assert report.dilation_only_ratio == 0.0
    assert report.replacement_override_ratio == pytest.approx(1 / 3)  # frame 0
    assert ffmpeg.encoded == frames_bytes([200, blend(128, 210, 20), 30])
    assert out.parent.is_dir()
    assert not any(p.killed for p in ffmpeg.processes)


def test_removal_decoders_force_libvpx_for_both_mattes_only(
    tmp_path: Path, monkeypatch: Any
) -> None:
    ffmpeg = FakeFfmpeg(default_outputs(2))
    wire(monkeypatch, ffmpeg, fake_probe(2))
    out = tmp_path / "v4.mp4"

    run_removal(tmp_path, out)

    assert ffmpeg.input_names() == ["src.mp4", "rep.mp4", "src.webm", "rep.webm", "-"]
    src, rep, source_matte, replacement_matte, encode = ffmpeg.commands
    for clip in (src, rep):
        assert "-c:v" not in clip
        assert clip[clip.index("-pix_fmt") + 1] == "rgb24"
    for matte in (source_matte, replacement_matte):
        assert matte[matte.index("-c:v") + 1] == "libvpx-vp9"
        assert matte[matte.index("-pix_fmt") + 1] == "rgba"
    assert encode[encode.index("-s") + 1] == f"{W}x{H}"
    assert encode[encode.index("-r") + 1] == "24/1"
    assert encode[encode.index("-c:v") + 1] == "libx264"
    assert encode[-1] == str(out)


def test_the_parameters_pass_through_orchestration(tmp_path: Path, monkeypatch: Any) -> None:
    ffmpeg = FakeFfmpeg(
        {
            "src.mp4": frames_bytes([10]),
            "rep.mp4": frames_bytes([200]),
            "src.webm": matte_bytes([64]),
            "rep.webm": matte_bytes([40]),
        }
    )
    wire(monkeypatch, ffmpeg, fake_probe(1))

    report = run_removal(tmp_path, tmp_path / "v4.mp4", threshold=65, dilation_radius=0)

    assert (report.threshold, report.dilation_radius) == (65, 0)
    assert report.removal_ratio == 0.0
    assert ffmpeg.encoded == frames_bytes([blend(40, 200, 10)])


def test_invalid_parameters_are_refused_before_anything_is_probed(
    tmp_path: Path, monkeypatch: Any
) -> None:
    ffmpeg = FakeFfmpeg(default_outputs(1))

    def never(path: Path, decoder: str | None = None) -> VideoInfo:
        raise AssertionError(f"probed {path}")

    wire(monkeypatch, ffmpeg, never)

    with pytest.raises(CompositeError, match="threshold must be in 1..255"):
        run_removal(tmp_path, tmp_path / "v4.mp4", threshold=256)
    with pytest.raises(CompositeError, match="dilation_radius must be >= 0"):
        run_removal(tmp_path, tmp_path / "v4.mp4", dilation_radius=-1)
    assert ffmpeg.commands == []


def test_removal_api_is_exported_from_the_package() -> None:
    assert video_character_skill.composite_video_source_removal is composite_video_source_removal
    assert video_character_skill.source_removal_mask is source_removal_mask
    assert video_character_skill.removal_effective_alpha is removal_effective_alpha
    assert video_character_skill.SourceRemovalCompositeReport is SourceRemovalCompositeReport
    assert video_character_skill.SOURCE_REMOVAL_THRESHOLD == 64
    assert video_character_skill.SOURCE_REMOVAL_DILATION_RADIUS == 4
    exported = set(video_character_skill.__all__)
    assert {
        "composite_video_source_removal",
        "source_removal_mask",
        "removal_effective_alpha",
        "SourceRemovalCompositeReport",
        "SOURCE_REMOVAL_THRESHOLD",
        "SOURCE_REMOVAL_DILATION_RADIUS",
    } <= exported


# -- validation failures at the orchestration level ---------------------


def test_a_matte_without_alpha_stops_the_run_before_any_process_spawns(
    tmp_path: Path, monkeypatch: Any
) -> None:
    ffmpeg = FakeFfmpeg(default_outputs(2))
    probe = fake_probe(2)

    def alpha_lost_on_one_matte(path: Path, decoder: str | None = None) -> VideoInfo:
        result = probe(path, decoder)
        return replace(result, pix_fmt="yuv420p") if path.name == "src.webm" else result

    wire(monkeypatch, ffmpeg, alpha_lost_on_one_matte)

    with pytest.raises(CompositeError, match="source matte: matte .* has no alpha"):
        run_removal(tmp_path, tmp_path / "v4.mp4")

    assert ffmpeg.commands == []
    assert not (tmp_path / "v4.mp4").exists()


def test_a_frame_count_mismatch_stops_the_run_before_any_process_spawns(
    tmp_path: Path, monkeypatch: Any
) -> None:
    ffmpeg = FakeFfmpeg(default_outputs(2))
    probe = fake_probe(2)

    def one_frame_long(path: Path, decoder: str | None = None) -> VideoInfo:
        result = probe(path, decoder)
        return replace(result, frame_count=3) if path.name == "rep.webm" else result

    wire(monkeypatch, ffmpeg, one_frame_long)

    with pytest.raises(CompositeError, match="replacement matte: inputs differ in frame count"):
        run_removal(tmp_path, tmp_path / "v4.mp4")

    assert ffmpeg.commands == []


def test_a_missing_input_is_rejected_before_ffprobe_runs(tmp_path: Path) -> None:
    with pytest.raises(CompositeError, match="input not found"):
        run_removal(tmp_path, tmp_path / "v4.mp4")


# -- ffmpeg failure propagation -----------------------------------------


@pytest.mark.parametrize("name", STREAMS)
def test_a_failing_decoder_is_reported_with_its_name_and_stderr(
    name: str, tmp_path: Path, monkeypatch: Any
) -> None:
    ffmpeg = FakeFfmpeg(default_outputs(2), failing=FILES[name], stderr="could not decode")
    wire(monkeypatch, ffmpeg, fake_probe(2))

    with pytest.raises(CompositeError, match=f"ffmpeg {name} exited 1: could not decode"):
        run_removal(tmp_path, tmp_path / "v4.mp4")


def test_a_failing_encoder_is_reported(tmp_path: Path, monkeypatch: Any) -> None:
    ffmpeg = FakeFfmpeg(default_outputs(2), failing="encode", stderr="broken pipe")
    wire(monkeypatch, ffmpeg, fake_probe(2))

    with pytest.raises(CompositeError, match="ffmpeg encode exited 1: broken pipe"):
        run_removal(tmp_path, tmp_path / "v4.mp4")


def test_a_decoder_that_dies_mid_stream_aborts_and_kills_the_rest(
    tmp_path: Path, monkeypatch: Any
) -> None:
    outputs = default_outputs(3)
    outputs["src.webm"] = matte_bytes([0])  # one frame, then silence
    ffmpeg = FakeFfmpeg(outputs, failing="src.webm", stderr="decode error")
    wire(monkeypatch, ffmpeg, fake_probe(3))

    with pytest.raises(CompositeError, match="source_matte ended during frame 1"):
        run_removal(tmp_path, tmp_path / "v4.mp4")

    assert all(p.killed for p in ffmpeg.processes)


# -- v3 is not affected by v4 -------------------------------------------


def test_the_hardened_union_composite_is_unchanged_by_v4(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Same inputs through v3 and v4: v3 still blends with max(), v4 does not."""
    outputs = {
        "src.mp4": frames_bytes([10, 10]),
        "rep.mp4": frames_bytes([200, 200]),
        "src.webm": matte_bytes([50, 100]),
        "rep.webm": matte_bytes([20, 0]),
    }
    ffmpeg_v3, ffmpeg_v4 = FakeFfmpeg(dict(outputs)), FakeFfmpeg(dict(outputs))

    wire(monkeypatch, ffmpeg_v3, fake_probe(2))
    v3 = composite_video_hardened_union(
        tmp_path / "src.mp4",
        tmp_path / "rep.mp4",
        tmp_path / "src.webm",
        tmp_path / "rep.webm",
        tmp_path / "v3.mp4",
    )
    wire(monkeypatch, ffmpeg_v4, fake_probe(2))
    v4 = run_removal(tmp_path, tmp_path / "v4.mp4")

    # frame 0: s=50/r=20 -> v3 max()=50 blend, v4 keeps r=20 blend
    # frame 1: s=100/r=0 -> v3 max()=100 blend (below 160), v4 core -> O1 copy
    assert ffmpeg_v3.encoded == frames_bytes([blend(50, 200, 10), blend(100, 200, 10)])
    assert ffmpeg_v4.encoded == frames_bytes([blend(20, 200, 10), 200])
    assert v3.source_threshold == 160
    assert not hasattr(v3, "removal_ratio")
    assert v4.removal_ratio == pytest.approx(1 / 2)
