"""Unit tests for the source-hardened dual-matte compositor (v3).

The hardening rule is a pure function over arrays, the streaming loop runs on
file-like objects, and the ffmpeg pipeline runs against the same fake
``subprocess.Popen`` the union tests use. Nothing here decodes, encodes or
writes a video. The v1 and v2 test modules are untouched; the last section
pins down that v2's output is not affected by the new rule.
"""

from __future__ import annotations

import io
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest

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
from video_character_skill import compositor
from video_character_skill.compositor import (
    SOURCE_HARDEN_THRESHOLD,
    CompositeError,
    HardenedUnionCompositeReport,
    HardenedUnionStreamStats,
    VideoInfo,
    composite_frame,
    composite_streams_hardened_union,
    composite_video_hardened_union,
    composite_video_union,
    hardened_union_alpha,
    union_alpha,
)

T = SOURCE_HARDEN_THRESHOLD


def blend(a: int, rep: int, src: int) -> int:
    return (a * rep + (255 - a) * src + 127) // 255


def test_the_default_threshold_is_exactly_160() -> None:
    assert SOURCE_HARDEN_THRESHOLD == 160
    assert T == 160


# -- hardened_union_alpha: the rule -------------------------------------


@pytest.mark.parametrize(
    ("src", "rep", "expected"),
    [
        (160, 17, 255),  # at threshold, source more confident -> hardened
        (219, 17, 255),  # the real source_only mean alphas -> hardened
        (159, 17, 159),  # one below threshold -> plain union keeps the partial value
        (200, 220, 220),  # replacement more confident -> replacement wins, untouched
        (200, 200, 200),  # equal: source is not *more* confident -> untouched
        (161, 160, 255),  # strictly greater by one -> hardened
        (128, 0, 128),  # foreground by the analysis definition, but below T
        (160, 255, 255),  # replacement already opaque
        (240, 239, 255),
        (239, 240, 240),
        (255, 0, 255),
        (0, 255, 255),
        (0, 0, 0),
    ],
)
def test_rule_on_each_value_pair(src: int, rep: int, expected: int) -> None:
    assert int(hardened_union_alpha(plane(src), plane(rep))[0, 0]) == expected


def test_threshold_boundary_159_vs_160() -> None:
    source = plane(0)
    source[0, 0], source[0, 1] = 159, 160
    replacement = plane(17)

    out = hardened_union_alpha(source, replacement)

    assert int(out[0, 0]) == 159
    assert int(out[0, 1]) == 255


def test_source_must_be_strictly_more_confident_than_the_replacement() -> None:
    """The ``source > replacement`` clause: equal or lower source never hardens."""
    source = plane(200)
    replacement = plane(0)
    replacement[0, :] = 199  # strictly below -> harden
    replacement[1, :] = 200  # equal -> plain union
    replacement[2, :] = 201  # replacement more confident -> replacement wins
    replacement[3, :] = 255

    out = hardened_union_alpha(source, replacement)

    assert np.unique(out[0]).tolist() == [255]
    assert np.unique(out[1]).tolist() == [200]
    assert np.unique(out[2]).tolist() == [201]
    assert np.unique(out[3]).tolist() == [255]


def test_replacement_matte_wins_wherever_it_is_at_least_as_confident() -> None:
    """The replacement matte is never thresholded: r stays exactly r where r >= s."""
    rng = np.random.default_rng(4)
    replacement = rng.integers(0, 256, (H, W), dtype=np.uint8)
    source = np.minimum(replacement, rng.integers(0, 256, (H, W), dtype=np.uint8))

    np.testing.assert_array_equal(hardened_union_alpha(source, replacement), replacement)


def test_source_below_threshold_is_never_touched() -> None:
    """No global thresholding: s < 160 goes through the plain union unchanged."""
    rng = np.random.default_rng(5)
    source = rng.integers(0, T, (H, W), dtype=np.uint8)
    replacement = rng.integers(0, 256, (H, W), dtype=np.uint8)

    np.testing.assert_array_equal(
        hardened_union_alpha(source, replacement), union_alpha(source, replacement)
    )


def test_rule_only_raises_alpha_and_only_where_it_fires() -> None:
    rng = np.random.default_rng(6)
    shape = (H * 8, W * 8)
    source = rng.integers(0, 256, shape, dtype=np.uint8)
    replacement = rng.integers(0, 256, shape, dtype=np.uint8)
    union = union_alpha(source, replacement)
    fires = (source >= T) & (source > replacement)
    assert fires.any() and not fires.all()

    out = hardened_union_alpha(source, replacement)

    np.testing.assert_array_equal(out[fires], 255)
    np.testing.assert_array_equal(out[~fires], union[~fires])
    assert bool((out >= union).all())
    assert out.dtype == np.uint8
    assert out.shape == shape


def test_matches_the_documented_numpy_formula_exactly() -> None:
    rng = np.random.default_rng(7)
    source = rng.integers(0, 256, (H, W), dtype=np.uint8)
    replacement = rng.integers(0, 256, (H, W), dtype=np.uint8)
    expected = np.maximum(source, replacement)
    expected[(source >= 160) & (source > replacement)] = 255

    np.testing.assert_array_equal(hardened_union_alpha(source, replacement), expected)


def test_mixed_values_within_one_frame() -> None:
    source = plane(0)
    replacement = plane(0)
    source[0, 0], replacement[0, 0] = 219, 17  # hardened
    source[0, 1], replacement[0, 1] = 159, 17  # partial, kept
    source[0, 2], replacement[0, 2] = 200, 220  # replacement wins
    source[0, 3], replacement[0, 3] = 0, 90  # replacement partial, kept
    source[0, 4], replacement[0, 4] = 160, 159  # hardened
    source[1, :], replacement[1, :] = 255, 255
    source[2, :], replacement[2, :] = 130, 0

    out = hardened_union_alpha(source, replacement)

    assert out[0].tolist() == [255, 159, 220, 90, 255]
    assert np.unique(out[1]).tolist() == [255]
    assert np.unique(out[2]).tolist() == [130]
    assert np.unique(out[3]).tolist() == [0]


def test_custom_threshold_is_honoured() -> None:
    source = plane(0)
    source[0, :], source[1, :] = 190, 200
    replacement = plane(17)

    out = hardened_union_alpha(source, replacement, source_threshold=200)

    assert np.unique(out[0]).tolist() == [190]
    assert np.unique(out[1]).tolist() == [255]
    # 255 can only "harden" pixels that are already opaque: identical to the union.
    np.testing.assert_array_equal(
        hardened_union_alpha(source, replacement, source_threshold=255),
        union_alpha(source, replacement),
    )


def test_inputs_are_not_mutated_and_the_output_is_new() -> None:
    source, replacement = plane(219), plane(17)
    s0, r0 = source.copy(), replacement.copy()

    out = hardened_union_alpha(source, replacement)

    np.testing.assert_array_equal(source, s0)
    np.testing.assert_array_equal(replacement, r0)
    assert not np.shares_memory(out, source)
    assert not np.shares_memory(out, replacement)


def test_read_only_planes_are_accepted() -> None:
    """The streams hand over ``np.frombuffer`` views, which are read-only."""
    source = np.frombuffer(bytes([219] * (H * W)), np.uint8).reshape(H, W)
    replacement = np.frombuffer(bytes([17] * (H * W)), np.uint8).reshape(H, W)

    assert np.unique(hardened_union_alpha(source, replacement)).tolist() == [255]


# -- the invariant under the hardened alpha -----------------------------


def test_alpha_zero_pixels_still_copy_the_source_bytes_exactly() -> None:
    rng = np.random.default_rng(8)
    source_rgb = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    replacement_rgb = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    source_alpha, replacement_alpha = plane(0), plane(0)
    source_alpha[:, :2] = 219  # old person on the left, partial alpha
    replacement_alpha[:, :1] = 17

    effective = hardened_union_alpha(source_alpha, replacement_alpha)
    out = composite_frame(source_rgb, replacement_rgb, effective)

    np.testing.assert_array_equal(effective[:, 2:], 0)
    np.testing.assert_array_equal(out[:, 2:], source_rgb[:, 2:])
    np.testing.assert_array_equal(effective[:, :2], 255)
    np.testing.assert_array_equal(out[:, :2], replacement_rgb[:, :2])


def test_source_only_interior_pixels_are_now_replacement_bytes_not_a_blend() -> None:
    """The v3 point: 219 / 17 was a ~14 % source blend under v2; now it is a copy."""
    rng = np.random.default_rng(9)
    source_rgb = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    replacement_rgb = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)

    v2 = composite_frame(source_rgb, replacement_rgb, union_alpha(plane(219), plane(17)))
    v3 = composite_frame(
        source_rgb, replacement_rgb, hardened_union_alpha(plane(219), plane(17))
    )

    np.testing.assert_array_equal(v3, replacement_rgb)
    assert not np.array_equal(v2, replacement_rgb)


# -- malformed inputs ---------------------------------------------------


def test_shape_mismatch_is_refused() -> None:
    with pytest.raises(CompositeError, match="replacement alpha shape"):
        hardened_union_alpha(plane(0), np.zeros((H, W + 1), np.uint8))


def test_a_channel_axis_is_refused() -> None:
    with pytest.raises(CompositeError, match=r"source alpha must be \(H, W\)"):
        hardened_union_alpha(np.zeros((H, W, 1), np.uint8), np.zeros((H, W, 1), np.uint8))


@pytest.mark.parametrize("dtype", [np.float32, np.uint16, np.int8, np.bool_])
def test_non_uint8_source_is_refused(dtype: Any) -> None:
    with pytest.raises(CompositeError, match="source alpha must be uint8"):
        hardened_union_alpha(np.zeros((H, W), dtype), plane(0))


@pytest.mark.parametrize("dtype", [np.float32, np.uint16, np.int8, np.bool_])
def test_non_uint8_replacement_is_refused(dtype: Any) -> None:
    with pytest.raises(CompositeError, match="replacement alpha must be uint8"):
        hardened_union_alpha(plane(0), np.zeros((H, W), dtype))


@pytest.mark.parametrize("threshold", [0, -1, 256, 1000])
def test_threshold_outside_1_to_255_is_refused(threshold: int) -> None:
    with pytest.raises(CompositeError, match="source_threshold must be in 1..255"):
        hardened_union_alpha(plane(0), plane(0), source_threshold=threshold)


@pytest.mark.parametrize("threshold", [160.0, "160", True, None])
def test_non_int_threshold_is_refused(threshold: Any) -> None:
    with pytest.raises(CompositeError, match="source_threshold must be an int"):
        hardened_union_alpha(plane(0), plane(0), source_threshold=threshold)


# -- streaming ----------------------------------------------------------


def stream_set(n: int) -> dict[str, io.BytesIO]:
    return {
        "source": io.BytesIO(frames_bytes(list(range(1, n + 1)))),
        "replacement": io.BytesIO(frames_bytes(list(range(101, 101 + n)))),
        "source_matte": io.BytesIO(matte_bytes([219] * n)),
        "replacement_matte": io.BytesIO(matte_bytes([17] * n)),
    }


def run_streams(
    streams: dict[str, io.BytesIO],
    frames: int,
    out: io.BytesIO | None = None,
    **kwargs: Any,
) -> HardenedUnionStreamStats:
    return composite_streams_hardened_union(
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


def test_streaming_composites_under_the_hardened_union_frame_by_frame() -> None:
    out = io.BytesIO()

    stats = composite_streams_hardened_union(
        io.BytesIO(frames_bytes([10, 20, 30, 40, 50])),
        io.BytesIO(frames_bytes([200, 210, 220, 230, 240])),
        io.BytesIO(matte_bytes([0, 219, 159, 200, 255])),  # source matte
        io.BytesIO(matte_bytes([0, 17, 17, 220, 0])),  # replacement matte
        out,
        width=W,
        height=H,
        frames=5,
    )

    written = np.frombuffer(out.getvalue(), np.uint8).reshape(5, H, W, 3)
    assert np.unique(written[0]).tolist() == [10]  # 0/0 -> source
    assert np.unique(written[1]).tolist() == [210]  # 219/17 -> hardened -> replacement
    assert np.unique(written[2]).tolist() == [blend(159, 220, 30)]  # below T -> blend
    assert np.unique(written[3]).tolist() == [blend(220, 230, 40)]  # replacement wins
    assert np.unique(written[4]).tolist() == [240]  # already opaque -> replacement
    assert stats.soft_edge_ratio == pytest.approx(2 / 5)  # frames 2 and 3
    assert stats.union_lift_ratio == pytest.approx(3 / 5)  # frames 1, 2, 4 (s > r)
    assert stats.hardened_ratio == pytest.approx(1 / 5)  # frame 1 only


def test_hardened_ratio_counts_only_pixels_the_rule_actually_changed() -> None:
    source_matte = np.zeros((H, W, 4), dtype=np.uint8)
    replacement_matte = np.zeros((H, W, 4), dtype=np.uint8)
    source_matte[:, 0, 3], replacement_matte[:, 0, 3] = 255, 0  # already 255: not counted
    source_matte[:, 1, 3], replacement_matte[:, 1, 3] = 200, 0  # changed: counted
    source_matte[:, 2, 3], replacement_matte[:, 2, 3] = 200, 220  # replacement wins
    source_matte[:, 3, 3], replacement_matte[:, 3, 3] = 159, 0  # below threshold
    source_matte[:, 4, 3], replacement_matte[:, 4, 3] = 160, 160  # equal: no fire
    out = io.BytesIO()

    stats = composite_streams_hardened_union(
        io.BytesIO(rgb(0).tobytes()),
        io.BytesIO(rgb(255).tobytes()),
        io.BytesIO(source_matte.tobytes()),
        io.BytesIO(replacement_matte.tobytes()),
        out,
        width=W,
        height=H,
        frames=1,
    )

    assert stats.hardened_ratio == pytest.approx(1 / W)  # column 1 only
    assert stats.union_lift_ratio == pytest.approx(3 / W)  # columns 0, 1, 3
    assert stats.soft_edge_ratio == pytest.approx(3 / W)  # columns 2 (220), 3 (159), 4 (160)
    written = np.frombuffer(out.getvalue(), np.uint8).reshape(H, W, 3)
    assert np.unique(written[:, :2]).tolist() == [255]
    assert np.unique(written[:, 2]).tolist() == [blend(220, 255, 0)]
    assert np.unique(written[:, 3]).tolist() == [blend(159, 255, 0)]
    assert np.unique(written[:, 4]).tolist() == [blend(160, 255, 0)]


def test_a_custom_threshold_flows_through_the_stream() -> None:
    out = io.BytesIO()

    stats = run_streams(stream_set(1), 1, out, source_threshold=220)

    written = np.frombuffer(out.getvalue(), np.uint8).reshape(H, W, 3)
    assert np.unique(written).tolist() == [blend(219, 101, 1)]  # 219 < 220: not hardened
    assert stats.hardened_ratio == 0.0


def test_an_invalid_threshold_is_refused_before_any_stream_is_read() -> None:
    streams = stream_set(2)

    with pytest.raises(CompositeError, match="source_threshold must be in 1..255"):
        run_streams(streams, 2, source_threshold=0)

    assert all(stream.tell() == 0 for stream in streams.values())


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
    streams["source_matte"] = io.BytesIO(streams["source_matte"].getvalue()[:-1])
    out = io.BytesIO()

    with pytest.raises(CompositeError, match="source_matte ended during frame 1"):
        run_streams(streams, 2, out)

    assert out.getvalue() == frames_bytes([101])  # frame 0 only: 219/17 -> replacement


@pytest.mark.parametrize("long", STREAMS)
def test_extra_data_on_any_of_the_four_streams_names_it(long: str) -> None:
    streams = stream_set(2)
    streams[long] = io.BytesIO(streams[long].getvalue() + b"\0")

    with pytest.raises(CompositeError, match=f"{long} has more than the expected 2"):
        run_streams(streams, 2)


def test_zero_frames_writes_nothing() -> None:
    out = io.BytesIO()

    stats = run_streams({name: io.BytesIO() for name in STREAMS}, 0, out)

    assert stats == HardenedUnionStreamStats(0.0, 0.0, 0.0)
    assert out.getvalue() == b""


# -- orchestration, against the fake ffmpeg -----------------------------


def run_hardened(tmp_path: Path, out: Path, **kwargs: Any) -> HardenedUnionCompositeReport:
    return composite_video_hardened_union(
        tmp_path / "src.mp4",
        tmp_path / "rep.mp4",
        tmp_path / "src.webm",
        tmp_path / "rep.webm",
        out,
        **kwargs,
    )


def test_composite_video_hardened_union_streams_four_inputs_into_one_encoder(
    tmp_path: Path, monkeypatch: Any
) -> None:
    ffmpeg = FakeFfmpeg(
        {
            "src.mp4": frames_bytes([10, 20, 30]),
            "rep.mp4": frames_bytes([200, 210, 220]),
            "src.webm": matte_bytes([219, 0, 159]),
            "rep.webm": matte_bytes([17, 0, 17]),
        }
    )
    wire(monkeypatch, ffmpeg, fake_probe(3))
    out = tmp_path / "nested" / "v3.mp4"

    report = run_hardened(tmp_path, out)

    assert report.output_path == out
    assert (report.frames, report.width, report.height) == (3, W, H)
    assert report.frame_rate == Fraction(24, 1)
    assert report.source_threshold == 160
    assert report.soft_edge_ratio == pytest.approx(1 / 3)  # frame 2 (159)
    assert report.union_lift_ratio == pytest.approx(2 / 3)  # frames 0 and 2
    assert report.hardened_ratio == pytest.approx(1 / 3)  # frame 0 only
    assert ffmpeg.encoded == frames_bytes([200, 20, blend(159, 220, 30)])
    assert out.parent.is_dir()
    assert not any(p.killed for p in ffmpeg.processes)


def test_hardened_decoders_force_libvpx_for_both_mattes_only(
    tmp_path: Path, monkeypatch: Any
) -> None:
    ffmpeg = FakeFfmpeg(default_outputs(2))
    wire(monkeypatch, ffmpeg, fake_probe(2))
    out = tmp_path / "v3.mp4"

    run_hardened(tmp_path, out)

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


def test_the_threshold_passes_through_orchestration(tmp_path: Path, monkeypatch: Any) -> None:
    ffmpeg = FakeFfmpeg(
        {
            "src.mp4": frames_bytes([10]),
            "rep.mp4": frames_bytes([200]),
            "src.webm": matte_bytes([219]),
            "rep.webm": matte_bytes([17]),
        }
    )
    wire(monkeypatch, ffmpeg, fake_probe(1))

    report = run_hardened(tmp_path, tmp_path / "v3.mp4", source_threshold=250)

    assert report.source_threshold == 250
    assert report.hardened_ratio == 0.0
    assert ffmpeg.encoded == frames_bytes([blend(219, 200, 10)])


def test_an_invalid_threshold_is_refused_before_anything_is_probed(
    tmp_path: Path, monkeypatch: Any
) -> None:
    ffmpeg = FakeFfmpeg(default_outputs(1))

    def never(path: Path, decoder: str | None = None) -> VideoInfo:
        raise AssertionError(f"probed {path}")

    wire(monkeypatch, ffmpeg, never)

    with pytest.raises(CompositeError, match="source_threshold must be in 1..255"):
        run_hardened(tmp_path, tmp_path / "v3.mp4", source_threshold=256)

    assert ffmpeg.commands == []


def test_union_api_is_exported_from_the_package() -> None:
    assert video_character_skill.composite_video_hardened_union is composite_video_hardened_union
    assert video_character_skill.hardened_union_alpha is hardened_union_alpha
    assert video_character_skill.HardenedUnionCompositeReport is HardenedUnionCompositeReport
    assert video_character_skill.SOURCE_HARDEN_THRESHOLD == 160
    exported = set(video_character_skill.__all__)
    assert {
        "composite_video_hardened_union",
        "hardened_union_alpha",
        "HardenedUnionCompositeReport",
        "SOURCE_HARDEN_THRESHOLD",
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
        run_hardened(tmp_path, tmp_path / "v3.mp4")

    assert ffmpeg.commands == []
    assert not (tmp_path / "v3.mp4").exists()


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
        run_hardened(tmp_path, tmp_path / "v3.mp4")

    assert ffmpeg.commands == []


def test_a_size_mismatch_stops_the_run_before_any_process_spawns(
    tmp_path: Path, monkeypatch: Any
) -> None:
    ffmpeg = FakeFfmpeg(default_outputs(2))
    probe = fake_probe(2)

    def narrower_replacement(path: Path, decoder: str | None = None) -> VideoInfo:
        result = probe(path, decoder)
        return replace(result, width=W - 1) if path.name == "rep.mp4" else result

    wire(monkeypatch, ffmpeg, narrower_replacement)

    with pytest.raises(CompositeError, match="differ in size"):
        run_hardened(tmp_path, tmp_path / "v3.mp4")

    assert ffmpeg.commands == []


def test_a_missing_input_is_rejected_before_ffprobe_runs(tmp_path: Path) -> None:
    with pytest.raises(CompositeError, match="input not found"):
        run_hardened(tmp_path, tmp_path / "v3.mp4")


# -- ffmpeg failure propagation -----------------------------------------


@pytest.mark.parametrize("name", STREAMS)
def test_a_failing_decoder_is_reported_with_its_name_and_stderr(
    name: str, tmp_path: Path, monkeypatch: Any
) -> None:
    ffmpeg = FakeFfmpeg(default_outputs(2), failing=FILES[name], stderr="could not decode")
    wire(monkeypatch, ffmpeg, fake_probe(2))

    with pytest.raises(CompositeError, match=f"ffmpeg {name} exited 1: could not decode"):
        run_hardened(tmp_path, tmp_path / "v3.mp4")


def test_a_failing_encoder_is_reported(tmp_path: Path, monkeypatch: Any) -> None:
    ffmpeg = FakeFfmpeg(default_outputs(2), failing="encode", stderr="broken pipe")
    wire(monkeypatch, ffmpeg, fake_probe(2))

    with pytest.raises(CompositeError, match="ffmpeg encode exited 1: broken pipe"):
        run_hardened(tmp_path, tmp_path / "v3.mp4")


def test_a_decoder_that_dies_mid_stream_aborts_and_kills_the_rest(
    tmp_path: Path, monkeypatch: Any
) -> None:
    outputs = default_outputs(3)
    outputs["rep.webm"] = matte_bytes([0])  # one frame, then silence
    ffmpeg = FakeFfmpeg(outputs, failing="rep.webm", stderr="decode error")
    wire(monkeypatch, ffmpeg, fake_probe(3))

    with pytest.raises(CompositeError, match="replacement_matte ended during frame 1"):
        run_hardened(tmp_path, tmp_path / "v3.mp4")

    assert all(p.killed for p in ffmpeg.processes)


# -- v2 is not affected by the new rule ---------------------------------


def test_the_plain_union_composite_is_unchanged_by_v3(tmp_path: Path, monkeypatch: Any) -> None:
    """Same inputs through v2 and v3: v2 still blends the 219 / 17 pixel."""
    outputs = {
        "src.mp4": frames_bytes([10]),
        "rep.mp4": frames_bytes([200]),
        "src.webm": matte_bytes([219]),
        "rep.webm": matte_bytes([17]),
    }
    ffmpeg_v2, ffmpeg_v3 = FakeFfmpeg(dict(outputs)), FakeFfmpeg(dict(outputs))

    wire(monkeypatch, ffmpeg_v2, fake_probe(1))
    v2 = composite_video_union(
        tmp_path / "src.mp4",
        tmp_path / "rep.mp4",
        tmp_path / "src.webm",
        tmp_path / "rep.webm",
        tmp_path / "v2.mp4",
    )
    wire(monkeypatch, ffmpeg_v3, fake_probe(1))
    v3 = run_hardened(tmp_path, tmp_path / "v3.mp4")

    assert ffmpeg_v2.encoded == frames_bytes([blend(219, 200, 10)])
    assert ffmpeg_v3.encoded == frames_bytes([200])
    assert v2.union_lift_ratio == v3.union_lift_ratio == 1.0
    assert not hasattr(v2, "hardened_ratio")
    assert v3.hardened_ratio == 1.0
    assert compositor.composite_video_union is composite_video_union
