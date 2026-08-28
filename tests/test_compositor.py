"""Unit tests for the compositor.

The blend is a pure function over arrays and the streaming loop runs on
file-like objects, so almost everything here is exercised without ffmpeg and
without writing a single video. Only the orchestration seams are faked.
"""

from __future__ import annotations

import io
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from video_character_skill.compositor import (
    ALPHA_DECODERS,
    CompositeError,
    VideoInfo,
    _raise_for_failure,
    _validate,
    composite_frame,
    composite_streams,
    composite_video,
    probe_video,
    soft_edge_ratio,
)

H, W = 4, 5


def rgb(value: int) -> NDArray[np.uint8]:
    return np.full((H, W, 3), value, dtype=np.uint8)


def info(**overrides: Any) -> VideoInfo:
    fields: dict[str, Any] = {
        "path": Path("clip.mp4"),
        "codec_name": "h264",
        "width": 1080,
        "height": 1920,
        "pix_fmt": "yuv420p",
        "frame_rate": Fraction(24, 1),
        "frame_count": 225,
    }
    fields.update(overrides)
    return VideoInfo(**fields)


# -- the invariant ------------------------------------------------------


def test_alpha_zero_copies_the_source_exactly() -> None:
    source, replacement = rgb(37), rgb(211)
    alpha = np.zeros((H, W), dtype=np.uint8)

    out = composite_frame(source, replacement, alpha)

    np.testing.assert_array_equal(out, source)


def test_alpha_zero_preserves_every_distinct_source_byte() -> None:
    """Not just a flat colour: each channel of each pixel must survive."""
    rng = np.random.default_rng(0)
    source = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    replacement = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    alpha = np.zeros((H, W), dtype=np.uint8)

    np.testing.assert_array_equal(composite_frame(source, replacement, alpha), source)


def test_alpha_255_copies_the_replacement_exactly() -> None:
    rng = np.random.default_rng(1)
    source = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    replacement = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    alpha = np.full((H, W), 255, dtype=np.uint8)

    np.testing.assert_array_equal(
        composite_frame(source, replacement, alpha), replacement
    )


def test_alpha_128_blends_deterministically() -> None:
    source, replacement = rgb(0), rgb(255)
    alpha = np.full((H, W), 128, dtype=np.uint8)

    out = composite_frame(source, replacement, alpha)

    # (128 * 255 + 127 * 0 + 127) // 255 == 128
    assert np.unique(out).tolist() == [128]


@pytest.mark.parametrize(
    ("a", "src", "rep", "expected"),
    [
        (1, 0, 255, 1),
        (64, 0, 255, 64),
        (128, 10, 200, 105),
        (200, 100, 100, 100),
        (254, 0, 255, 254),
        (128, 255, 255, 255),
    ],
)
def test_blend_matches_the_documented_integer_formula(
    a: int, src: int, rep: int, expected: int
) -> None:
    out = composite_frame(rgb(src), rgb(rep), np.full((H, W), a, dtype=np.uint8))

    assert int(out[0, 0, 0]) == expected == (a * rep + (255 - a) * src + 127) // 255


def test_mixed_alpha_within_one_frame_routes_each_pixel_correctly() -> None:
    source, replacement = rgb(0), rgb(255)
    alpha = np.zeros((H, W), dtype=np.uint8)
    alpha[0, 0] = 0      # keep source
    alpha[0, 1] = 255    # take replacement
    alpha[0, 2] = 128    # blend
    alpha[1, :] = 255
    alpha[2, :] = 64

    out = composite_frame(source, replacement, alpha)

    assert int(out[0, 0, 0]) == 0
    assert int(out[0, 1, 0]) == 255
    assert int(out[0, 2, 0]) == 128
    np.testing.assert_array_equal(out[1], np.full((W, 3), 255, dtype=np.uint8))
    np.testing.assert_array_equal(out[2], np.full((W, 3), 64, dtype=np.uint8))
    np.testing.assert_array_equal(out[3], np.zeros((W, 3), dtype=np.uint8))


def test_blend_never_overflows_at_the_extremes() -> None:
    source, replacement = rgb(255), rgb(255)
    alpha = np.arange(H * W, dtype=np.uint8).reshape(H, W)

    assert np.unique(composite_frame(source, replacement, alpha)).tolist() == [255]


def test_the_source_array_is_not_mutated() -> None:
    source, replacement = rgb(10), rgb(250)
    before = source.copy()

    composite_frame(source, replacement, np.full((H, W), 200, dtype=np.uint8))

    np.testing.assert_array_equal(source, before)


# -- malformed frames ---------------------------------------------------


def test_dimension_mismatch_between_source_and_replacement_is_refused() -> None:
    with pytest.raises(CompositeError, match="replacement shape"):
        composite_frame(
            rgb(0), np.zeros((H + 1, W, 3), np.uint8), np.zeros((H, W), np.uint8)
        )


def test_alpha_of_the_wrong_size_is_refused() -> None:
    with pytest.raises(CompositeError, match="alpha shape"):
        composite_frame(rgb(0), rgb(1), np.zeros((H, W + 2), np.uint8))


def test_alpha_with_a_channel_axis_is_refused() -> None:
    with pytest.raises(CompositeError, match="alpha shape"):
        composite_frame(rgb(0), rgb(1), np.zeros((H, W, 1), np.uint8))


def test_a_non_rgb_source_is_refused() -> None:
    with pytest.raises(CompositeError, match=r"source must be \(H, W, 3\)"):
        composite_frame(
            np.zeros((H, W, 4), np.uint8), np.zeros((H, W, 4), np.uint8),
            np.zeros((H, W), np.uint8),
        )


@pytest.mark.parametrize("dtype", [np.float32, np.uint16, np.int8])
def test_non_uint8_alpha_is_refused(dtype: Any) -> None:
    with pytest.raises(CompositeError, match="alpha must be uint8"):
        composite_frame(rgb(0), rgb(1), np.zeros((H, W), dtype))


def test_float_source_is_refused() -> None:
    with pytest.raises(CompositeError, match="source must be uint8"):
        composite_frame(
            np.zeros((H, W, 3), np.float32), rgb(1), np.zeros((H, W), np.uint8)
        )


# -- soft edge ratio ----------------------------------------------------


def test_soft_edge_ratio_counts_only_partial_pixels() -> None:
    alpha = np.zeros((10, 10), dtype=np.uint8)
    alpha[0, :5] = 255
    alpha[1, :2] = 128

    assert soft_edge_ratio(alpha) == pytest.approx(0.02)
    assert soft_edge_ratio(np.zeros((4, 4), np.uint8)) == 0.0
    assert soft_edge_ratio(np.full((4, 4), 255, np.uint8)) == 0.0


# -- streaming ----------------------------------------------------------


def frames_bytes(values: list[int], channels: int = 3) -> bytes:
    return b"".join(
        np.full((H, W, channels), v, dtype=np.uint8).tobytes() for v in values
    )


def matte_bytes(alphas: list[int]) -> bytes:
    out = []
    for a in alphas:
        frame = np.zeros((H, W, 4), dtype=np.uint8)
        frame[:, :, 3] = a
        out.append(frame.tobytes())
    return b"".join(out)


def test_streaming_composites_each_frame_in_order() -> None:
    out = io.BytesIO()

    ratio = composite_streams(
        io.BytesIO(frames_bytes([10, 20, 30])),
        io.BytesIO(frames_bytes([200, 210, 220])),
        io.BytesIO(matte_bytes([0, 255, 128])),
        out,
        width=W,
        height=H,
        frames=3,
    )

    written = np.frombuffer(out.getvalue(), np.uint8).reshape(3, H, W, 3)
    assert np.unique(written[0]).tolist() == [10]     # alpha 0  -> source
    assert np.unique(written[1]).tolist() == [210]    # alpha 255 -> replacement
    assert np.unique(written[2]).tolist() == [125]    # (128*220 + 127*30 + 127)//255
    assert ratio == pytest.approx(1 / 3)


def test_streaming_holds_only_one_frame_at_a_time() -> None:
    """A reader that refuses to hand over two frames at once still works."""

    class OneFrameAtATime(io.BytesIO):
        def read(self, size: int | None = -1) -> bytes:
            cap = H * W * 3
            return super().read(min(size, cap) if size and size > 0 else cap)

    out = io.BytesIO()
    composite_streams(
        OneFrameAtATime(frames_bytes([1, 2])),
        OneFrameAtATime(frames_bytes([3, 4])),
        io.BytesIO(matte_bytes([0, 0])),
        out,
        width=W,
        height=H,
        frames=2,
    )
    assert len(out.getvalue()) == 2 * H * W * 3


@pytest.mark.parametrize("short", ["source", "replacement", "matte"])
def test_premature_eof_is_reported_with_the_stream_and_frame(short: str) -> None:
    streams = {
        "source": io.BytesIO(frames_bytes([1, 2, 3])),
        "replacement": io.BytesIO(frames_bytes([4, 5, 6])),
        "matte": io.BytesIO(matte_bytes([0, 0, 0])),
    }
    truncated = streams[short].getvalue()[: -H * W]
    streams[short] = io.BytesIO(truncated)

    with pytest.raises(CompositeError, match=f"{short} ended during frame 2"):
        composite_streams(
            streams["source"], streams["replacement"], streams["matte"],
            io.BytesIO(), width=W, height=H, frames=3,
        )


@pytest.mark.parametrize("long", ["source", "replacement", "matte"])
def test_extra_frames_beyond_the_expected_count_are_reported(long: str) -> None:
    streams = {
        "source": io.BytesIO(frames_bytes([1, 2])),
        "replacement": io.BytesIO(frames_bytes([3, 4])),
        "matte": io.BytesIO(matte_bytes([0, 0])),
    }
    extra = frames_bytes([9, 9]) if long != "matte" else matte_bytes([9, 9])
    streams[long] = io.BytesIO(streams[long].getvalue() + extra[: H * W * 3])

    with pytest.raises(CompositeError, match=f"{long} has more than the expected 2"):
        composite_streams(
            streams["source"], streams["replacement"], streams["matte"],
            io.BytesIO(), width=W, height=H, frames=2,
        )


def test_zero_frames_writes_nothing() -> None:
    out = io.BytesIO()
    assert composite_streams(
        io.BytesIO(), io.BytesIO(), io.BytesIO(), out, width=W, height=H, frames=0
    ) == 0.0
    assert out.getvalue() == b""


# -- validation ---------------------------------------------------------


def test_matching_inputs_validate() -> None:
    matte = info(codec_name="vp9", pix_fmt="yuva420p")
    _validate(info(), info(), matte)


def test_size_mismatch_is_refused() -> None:
    with pytest.raises(CompositeError, match="differ in size"):
        _validate(info(), info(width=720), info(codec_name="vp9", pix_fmt="yuva420p"))


def test_fps_mismatch_between_source_and_replacement_is_refused() -> None:
    with pytest.raises(CompositeError, match="30 fps"):
        _validate(
            info(),
            info(frame_rate=Fraction(30, 1)),
            info(codec_name="vp9", pix_fmt="yuva420p"),
        )


def test_equivalent_frame_rates_written_differently_still_match() -> None:
    _validate(
        info(frame_rate=Fraction(24, 1)),
        info(frame_rate=Fraction(48, 2)),
        info(codec_name="vp9", pix_fmt="yuva420p"),
    )


def test_frame_count_mismatch_is_refused() -> None:
    with pytest.raises(CompositeError, match="differ in frame count"):
        _validate(
            info(), info(frame_count=224), info(codec_name="vp9", pix_fmt="yuva420p")
        )


def test_empty_clip_is_refused() -> None:
    with pytest.raises(CompositeError, match="has no frames"):
        _validate(
            info(frame_count=0),
            info(frame_count=0),
            info(codec_name="vp9", pix_fmt="yuva420p", frame_count=0),
        )


def test_a_matte_without_alpha_is_refused() -> None:
    """The exact trap the native vp9 decoder sets: yuv420p, alpha gone."""
    with pytest.raises(CompositeError, match="has no alpha channel"):
        _validate(info(), info(), info(codec_name="vp9", pix_fmt="yuv420p"))


def test_alpha_bearing_formats_are_recognised() -> None:
    assert info(pix_fmt="yuva420p").has_alpha
    assert info(pix_fmt="rgba").has_alpha
    assert not info(pix_fmt="yuv420p").has_alpha
    assert not info(pix_fmt="gbrp").has_alpha


def test_vp9_and_vp8_get_the_alpha_capable_decoder() -> None:
    assert ALPHA_DECODERS["vp9"] == "libvpx-vp9"
    assert ALPHA_DECODERS["vp8"] == "libvpx-vp8"
    assert "h264" not in ALPHA_DECODERS


def test_rate_argument_avoids_float_rounding() -> None:
    assert info(frame_rate=Fraction(24, 1)).rate_argument == "24/1"
    assert info(frame_rate=Fraction(30000, 1001)).rate_argument == "30000/1001"


# -- ffmpeg failure propagation -----------------------------------------


def test_ffprobe_failure_is_propagated(tmp_path: Path, monkeypatch: Any) -> None:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"not a video")

    def fail(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 1, "", "Invalid data found")

    monkeypatch.setattr(subprocess, "run", fail)

    with pytest.raises(CompositeError, match="ffprobe failed.*Invalid data found"):
        probe_video(clip)


def test_ffprobe_output_missing_fields_is_reported(tmp_path: Path, monkeypatch: Any) -> None:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")

    def partial(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 0, "codec_name=h264\nwidth=10\n", "")

    monkeypatch.setattr(subprocess, "run", partial)

    with pytest.raises(CompositeError, match="no usable stream"):
        probe_video(clip)


def test_a_missing_input_file_is_reported_before_ffprobe_runs(tmp_path: Path) -> None:
    with pytest.raises(CompositeError, match="input not found"):
        probe_video(tmp_path / "absent.mp4")


def test_a_nonzero_ffmpeg_exit_is_raised_with_its_stderr() -> None:
    with pytest.raises(CompositeError, match=r"ffmpeg matte exited 1: could not decode"):
        _raise_for_failure("matte", 1, "could not decode")


def test_a_nonzero_ffmpeg_exit_without_stderr_still_raises() -> None:
    with pytest.raises(CompositeError, match="ffmpeg encode exited 137"):
        _raise_for_failure("encode", 137, "")


def test_a_clean_ffmpeg_exit_raises_nothing() -> None:
    _raise_for_failure("source", 0, "")  # must not raise


def test_composite_video_rejects_a_missing_source(tmp_path: Path) -> None:
    """Validation happens before any ffmpeg process is spawned."""
    with pytest.raises(CompositeError, match="input not found"):
        composite_video(
            tmp_path / "absent.mp4",
            tmp_path / "absent.mp4",
            tmp_path / "absent.webm",
            tmp_path / "out.mp4",
        )
