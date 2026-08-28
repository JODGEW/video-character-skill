"""Unit tests for the dual-matte union compositor.

Same approach as ``test_compositor.py``: the union is a pure function over
arrays, the streaming loop runs on file-like objects, and the ffmpeg pipeline
is exercised against a fake ``subprocess.Popen`` — nothing here decodes,
encodes or writes a video. Because the pipeline helper is shared with the
single-matte path, that path is wired up against the same fake too.
"""

from __future__ import annotations

import io
import subprocess
from collections.abc import Callable
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import IO, Any

import numpy as np
import pytest
from numpy.typing import NDArray

import video_character_skill
from video_character_skill import compositor
from video_character_skill.compositor import (
    CompositeError,
    CompositeReport,
    UnionCompositeReport,
    UnionStreamStats,
    VideoInfo,
    _validate_union,
    composite_frame,
    composite_streams_union,
    composite_video,
    composite_video_union,
    union_alpha,
)

H, W = 4, 5
STREAMS = ("source", "replacement", "source_matte", "replacement_matte")
MATTES = ("source matte", "replacement matte")


def rgb(value: int) -> NDArray[np.uint8]:
    return np.full((H, W, 3), value, dtype=np.uint8)


def plane(value: int) -> NDArray[np.uint8]:
    return np.full((H, W), value, dtype=np.uint8)


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


def matte_info(**overrides: Any) -> VideoInfo:
    fields: dict[str, Any] = {
        "path": Path("matte.webm"),
        "codec_name": "vp9",
        "pix_fmt": "yuva420p",
    }
    fields.update(overrides)
    return info(**fields)


# -- union_alpha: exact max semantics -----------------------------------


def test_union_is_exactly_the_elementwise_maximum() -> None:
    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, (H, W), dtype=np.uint8)
    b = rng.integers(0, 256, (H, W), dtype=np.uint8)

    out = union_alpha(a, b)

    np.testing.assert_array_equal(out, np.maximum(a, b))
    assert out.dtype == np.uint8
    assert out.shape == (H, W)


def test_union_is_symmetric() -> None:
    rng = np.random.default_rng(1)
    a = rng.integers(0, 256, (H, W), dtype=np.uint8)
    b = rng.integers(0, 256, (H, W), dtype=np.uint8)

    np.testing.assert_array_equal(union_alpha(a, b), union_alpha(b, a))


def test_source_matte_larger_than_replacement_matte() -> None:
    """The old silhouette pokes out: those pixels keep the source's alpha."""
    source = plane(0)
    source[:, :3] = 255  # old person covers columns 0-2
    replacement = plane(0)
    replacement[:, :1] = 255  # new person covers column 0 only

    out = union_alpha(source, replacement)

    np.testing.assert_array_equal(out[:, :3], 255)
    np.testing.assert_array_equal(out[:, 3:], 0)


def test_replacement_matte_larger_than_source_matte() -> None:
    source = plane(0)
    source[:2, :] = 255
    replacement = plane(0)
    replacement[:3, :] = 255

    out = union_alpha(source, replacement)

    np.testing.assert_array_equal(out[:3], 255)
    np.testing.assert_array_equal(out[3:], 0)


def test_equal_mattes_come_back_unchanged() -> None:
    rng = np.random.default_rng(2)
    a = rng.integers(0, 256, (H, W), dtype=np.uint8)

    np.testing.assert_array_equal(union_alpha(a, a.copy()), a)


@pytest.mark.parametrize(
    ("src", "rep", "expected"),
    [
        (0, 0, 0),
        (255, 0, 255),
        (0, 255, 255),
        (255, 255, 255),
        (100, 50, 100),
        (50, 100, 100),
        (128, 128, 128),
        (1, 254, 254),
        (219, 17, 219),  # the real source_only region's mean alphas
    ],
)
def test_union_of_each_value_pair(src: int, rep: int, expected: int) -> None:
    assert int(union_alpha(plane(src), plane(rep))[0, 0]) == expected


def test_mixed_values_within_one_frame() -> None:
    source = plane(0)
    replacement = plane(0)
    source[0, 1] = 255
    replacement[0, 2] = 255
    source[0, 3], replacement[0, 3] = 200, 40
    source[0, 4], replacement[0, 4] = 40, 200
    source[1, :], replacement[1, :] = 128, 127
    source[2, :], replacement[2, :] = 255, 255

    out = union_alpha(source, replacement)

    assert out[0].tolist() == [0, 255, 255, 200, 200]
    assert np.unique(out[1]).tolist() == [128]
    assert np.unique(out[2]).tolist() == [255]
    assert np.unique(out[3]).tolist() == [0]


def test_union_does_not_mutate_its_inputs_and_returns_a_new_array() -> None:
    a, b = plane(10), plane(200)
    a0, b0 = a.copy(), b.copy()

    out = union_alpha(a, b)

    np.testing.assert_array_equal(a, a0)
    np.testing.assert_array_equal(b, b0)
    assert not np.shares_memory(out, a)
    assert not np.shares_memory(out, b)


def test_union_accepts_read_only_planes() -> None:
    """The streams hand over ``np.frombuffer`` views, which are read-only."""
    a = np.frombuffer(bytes([200] * (H * W)), np.uint8).reshape(H, W)
    b = np.frombuffer(bytes([100] * (H * W)), np.uint8).reshape(H, W)

    assert np.unique(union_alpha(a, b)).tolist() == [200]


# -- union_alpha: malformed inputs --------------------------------------


def test_union_shape_mismatch_is_refused() -> None:
    with pytest.raises(CompositeError, match="replacement alpha shape"):
        union_alpha(plane(0), np.zeros((H, W + 1), np.uint8))


def test_union_with_a_channel_axis_is_refused() -> None:
    with pytest.raises(CompositeError, match=r"source alpha must be \(H, W\)"):
        union_alpha(np.zeros((H, W, 1), np.uint8), np.zeros((H, W, 1), np.uint8))


@pytest.mark.parametrize("dtype", [np.float32, np.uint16, np.int8, np.bool_])
def test_union_non_uint8_source_is_refused(dtype: Any) -> None:
    with pytest.raises(CompositeError, match="source alpha must be uint8"):
        union_alpha(np.zeros((H, W), dtype), plane(0))


@pytest.mark.parametrize("dtype", [np.float32, np.uint16, np.int8, np.bool_])
def test_union_non_uint8_replacement_is_refused(dtype: Any) -> None:
    with pytest.raises(CompositeError, match="replacement alpha must be uint8"):
        union_alpha(plane(0), np.zeros((H, W), dtype))


# -- the point of the union: old-person pixels route to the replacement --


def test_source_only_region_is_taken_from_the_replacement_clip() -> None:
    """Where the old silhouette pokes out from under the new one, the single
    matte shows the old person; the union must show the O1 clip's bytes."""
    rng = np.random.default_rng(3)
    source_rgb = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    replacement_rgb = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)

    source_alpha = plane(0)
    source_alpha[:, :3] = 255  # old person: columns 0-2
    replacement_alpha = plane(0)
    replacement_alpha[:, :2] = 255  # new person: columns 0-1
    source_only = (source_alpha == 255) & (replacement_alpha == 0)  # column 2
    overlap = (source_alpha == 255) & (replacement_alpha == 255)
    background = (source_alpha == 0) & (replacement_alpha == 0)
    assert source_only.any() and overlap.any() and background.any()

    v1 = composite_frame(source_rgb, replacement_rgb, replacement_alpha)
    v2 = composite_frame(
        source_rgb, replacement_rgb, union_alpha(source_alpha, replacement_alpha)
    )

    # v1 leaks the old person there; v2 routes the very same pixels to the
    # replacement, byte-exact, because their effective alpha is 255.
    np.testing.assert_array_equal(v1[source_only], source_rgb[source_only])
    np.testing.assert_array_equal(v2[source_only], replacement_rgb[source_only])
    # The invariant still holds at both ends of the effective alpha.
    np.testing.assert_array_equal(v2[background], source_rgb[background])
    np.testing.assert_array_equal(v2[overlap], replacement_rgb[overlap])


def test_partially_opaque_source_only_pixels_blend_toward_the_replacement() -> None:
    """The real source_only region averages source alpha 219 vs replacement 17."""
    effective = union_alpha(plane(219), plane(17))

    out = composite_frame(rgb(0), rgb(255), effective)

    assert np.unique(effective).tolist() == [219]
    expected = (219 * 255 + (255 - 219) * 0 + 127) // 255
    assert np.unique(out).tolist() == [expected]
    assert expected > 127  # nearer the replacement than the source


# -- streaming ----------------------------------------------------------


def frames_bytes(values: list[int]) -> bytes:
    return b"".join(np.full((H, W, 3), v, dtype=np.uint8).tobytes() for v in values)


def matte_bytes(alphas: list[int]) -> bytes:
    out = []
    for a in alphas:
        frame = np.zeros((H, W, 4), dtype=np.uint8)
        frame[:, :, :3] = 77  # the matte's RGB is irrelevant and must be ignored
        frame[:, :, 3] = a
        out.append(frame.tobytes())
    return b"".join(out)


def stream_set(n: int) -> dict[str, io.BytesIO]:
    return {
        "source": io.BytesIO(frames_bytes(list(range(1, n + 1)))),
        "replacement": io.BytesIO(frames_bytes(list(range(101, 101 + n)))),
        "source_matte": io.BytesIO(matte_bytes([0] * n)),
        "replacement_matte": io.BytesIO(matte_bytes([0] * n)),
    }


def run_streams(
    streams: dict[str, io.BytesIO], frames: int, out: io.BytesIO | None = None
) -> UnionStreamStats:
    return composite_streams_union(
        streams["source"],
        streams["replacement"],
        streams["source_matte"],
        streams["replacement_matte"],
        out if out is not None else io.BytesIO(),
        width=W,
        height=H,
        frames=frames,
    )


def test_streaming_composites_under_the_union_frame_by_frame() -> None:
    out = io.BytesIO()

    stats = composite_streams_union(
        io.BytesIO(frames_bytes([10, 20, 30, 40])),
        io.BytesIO(frames_bytes([200, 210, 220, 230])),
        io.BytesIO(matte_bytes([0, 255, 0, 128])),  # source matte
        io.BytesIO(matte_bytes([0, 0, 255, 64])),  # replacement matte
        out,
        width=W,
        height=H,
        frames=4,
    )

    written = np.frombuffer(out.getvalue(), np.uint8).reshape(4, H, W, 3)
    assert np.unique(written[0]).tolist() == [10]  # max(0, 0) = 0 -> source
    assert np.unique(written[1]).tolist() == [210]  # max(255, 0) = 255 -> replacement
    assert np.unique(written[2]).tolist() == [220]  # max(0, 255) = 255 -> replacement
    blended = (128 * 230 + (255 - 128) * 40 + 127) // 255  # max(128, 64) = 128
    assert np.unique(written[3]).tolist() == [blended]
    assert stats == UnionStreamStats(soft_edge_ratio=0.25, union_lift_ratio=0.5)


def test_streaming_takes_alpha_from_each_matte_independently() -> None:
    """A per-pixel mix inside one frame, through the raw-bytes path."""
    source_matte = np.zeros((H, W, 4), dtype=np.uint8)
    replacement_matte = np.zeros((H, W, 4), dtype=np.uint8)
    source_matte[:, 0, 3] = 255  # column 0: source only
    replacement_matte[:, 1, 3] = 255  # column 1: replacement only
    source_matte[:, 2, 3] = 255  # column 2: both
    replacement_matte[:, 2, 3] = 255
    out = io.BytesIO()

    composite_streams_union(
        io.BytesIO(rgb(0).tobytes()),
        io.BytesIO(rgb(255).tobytes()),
        io.BytesIO(source_matte.tobytes()),
        io.BytesIO(replacement_matte.tobytes()),
        out,
        width=W,
        height=H,
        frames=1,
    )

    written = np.frombuffer(out.getvalue(), np.uint8).reshape(H, W, 3)
    assert np.unique(written[:, :3]).tolist() == [255]
    assert np.unique(written[:, 3:]).tolist() == [0]


def test_streaming_holds_only_one_frame_at_a_time() -> None:
    """A reader that refuses to hand over two frames at once still works."""

    class OneFrameAtATime(io.BytesIO):
        def read(self, size: int | None = -1) -> bytes:
            cap = H * W * 3
            return super().read(min(size, cap) if size and size > 0 else cap)

    out = io.BytesIO()
    composite_streams_union(
        OneFrameAtATime(frames_bytes([1, 2])),
        OneFrameAtATime(frames_bytes([3, 4])),
        OneFrameAtATime(matte_bytes([0, 0])),
        OneFrameAtATime(matte_bytes([0, 0])),
        out,
        width=W,
        height=H,
        frames=2,
    )
    assert out.getvalue() == frames_bytes([1, 2])


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

    assert out.getvalue() == frames_bytes([1])  # frame 0 only, intact


@pytest.mark.parametrize("long", STREAMS)
def test_extra_data_on_any_of_the_four_streams_names_it(long: str) -> None:
    streams = stream_set(2)
    streams[long] = io.BytesIO(streams[long].getvalue() + b"\0")

    with pytest.raises(CompositeError, match=f"{long} has more than the expected 2"):
        run_streams(streams, 2)


def test_zero_frames_writes_nothing() -> None:
    out = io.BytesIO()

    stats = run_streams(
        {name: io.BytesIO() for name in STREAMS}, 0, out
    )

    assert stats == UnionStreamStats(soft_edge_ratio=0.0, union_lift_ratio=0.0)
    assert out.getvalue() == b""


# -- validation ---------------------------------------------------------


def test_four_matching_streams_validate() -> None:
    _validate_union(info(), info(), matte_info(), matte_info())


def validate_with(which: str, matte: VideoInfo) -> None:
    mattes = {"source matte": matte_info(), "replacement matte": matte_info()}
    mattes[which] = matte
    _validate_union(info(), info(), mattes["source matte"], mattes["replacement matte"])


@pytest.mark.parametrize("which", MATTES)
def test_frame_count_mismatch_on_either_matte_is_refused_and_named(which: str) -> None:
    with pytest.raises(CompositeError, match=f"{which}: inputs differ in frame count"):
        validate_with(which, matte_info(frame_count=224))


def test_frame_count_mismatch_between_the_clips_is_refused() -> None:
    with pytest.raises(CompositeError, match="differ in frame count"):
        _validate_union(info(), info(frame_count=224), matte_info(), matte_info())


@pytest.mark.parametrize("which", MATTES)
def test_size_mismatch_on_either_matte_is_refused_and_named(which: str) -> None:
    with pytest.raises(CompositeError, match=f"{which}: inputs differ in size"):
        validate_with(which, matte_info(width=720, height=1280))


def test_size_mismatch_between_the_clips_is_refused() -> None:
    with pytest.raises(CompositeError, match="differ in size"):
        _validate_union(info(), info(width=720), matte_info(), matte_info())


def test_fps_mismatch_between_the_clips_is_refused() -> None:
    with pytest.raises(CompositeError, match="30 fps"):
        _validate_union(info(), info(frame_rate=Fraction(30, 1)), matte_info(), matte_info())


def test_equivalent_frame_rates_written_differently_still_match() -> None:
    _validate_union(
        info(frame_rate=Fraction(24, 1)),
        info(frame_rate=Fraction(48, 2)),
        matte_info(),
        matte_info(),
    )


@pytest.mark.parametrize("which", MATTES)
def test_a_matte_without_alpha_is_refused_and_named(which: str) -> None:
    """The native vp9 decoder trap, on either matte: yuv420p, alpha gone."""
    with pytest.raises(CompositeError, match=f"{which}: matte .* has no alpha channel"):
        validate_with(which, matte_info(pix_fmt="yuv420p"))


def test_empty_clips_are_refused() -> None:
    with pytest.raises(CompositeError, match="has no frames"):
        _validate_union(
            info(frame_count=0),
            info(frame_count=0),
            matte_info(frame_count=0),
            matte_info(frame_count=0),
        )


# -- the ffmpeg pipeline, against a fake Popen --------------------------


class Sink(io.BytesIO):
    """An encoder stdin that keeps what was written to it after it is closed."""

    def __init__(self) -> None:
        super().__init__()
        self.captured = b""

    def close(self) -> None:
        self.captured = self.getvalue()
        super().close()


class FakeProcess:
    """The slice of ``subprocess.Popen`` the compositor touches."""

    def __init__(self, *, stdout: bytes | None, returncode: int) -> None:
        self.stdout: IO[bytes] | None = None if stdout is None else io.BytesIO(stdout)
        self.stdin: Sink | None = Sink() if stdout is None else None
        self._returncode = returncode
        self._exited = False
        self.killed = False

    def poll(self) -> int | None:
        return self._returncode if self._exited else None

    def wait(self) -> int:
        self._exited = True
        return self._returncode

    def kill(self) -> None:
        self.killed = True
        self._exited = True


class FakeFfmpeg:
    """Stands in for ``subprocess.Popen``.

    Decoders emit the bytes registered for their input file's basename; the
    encoder (``-i -``) swallows whatever it is fed. One process can be made to
    exit non-zero, with text on its stderr.
    """

    def __init__(
        self, outputs: dict[str, bytes], *, failing: str | None = None, stderr: str = ""
    ) -> None:
        self.outputs = outputs
        self.failing = failing
        self.stderr = stderr
        self.commands: list[list[str]] = []
        self.processes: list[FakeProcess] = []

    def __call__(self, command: list[str], **kwargs: Any) -> FakeProcess:
        self.commands.append(command)
        target = command[command.index("-i") + 1]
        name = "encode" if target == "-" else Path(target).name
        fails = name == self.failing
        if fails and self.stderr:
            kwargs["stderr"].write(self.stderr.encode())
        process = FakeProcess(
            stdout=None if name == "encode" else self.outputs[name],
            returncode=1 if fails else 0,
        )
        self.processes.append(process)
        return process

    def input_names(self) -> list[str]:
        return [Path(c[c.index("-i") + 1]).name for c in self.commands]

    @property
    def encoded(self) -> bytes:
        sink = self.processes[-1].stdin
        assert sink is not None
        return sink.captured


Probe = Callable[[Path, "str | None"], VideoInfo]


def fake_probe(frames: int) -> Probe:
    """``probe_video`` for files that need not exist: ``.mp4`` is h264,
    ``.webm`` is vp9 whose alpha only shows under libvpx-vp9 — the real trap."""

    def probe(path: Path, decoder: str | None = None) -> VideoInfo:
        if path.suffix == ".webm":
            pix_fmt = "yuva420p" if decoder == "libvpx-vp9" else "yuv420p"
            return info(
                path=path, codec_name="vp9", pix_fmt=pix_fmt, width=W, height=H,
                frame_count=frames,
            )
        return info(path=path, width=W, height=H, frame_count=frames)

    return probe


def wire(monkeypatch: Any, ffmpeg: FakeFfmpeg, probe: Probe) -> None:
    monkeypatch.setattr(compositor, "probe_video", probe)
    monkeypatch.setattr(subprocess, "Popen", ffmpeg)


FILES = {
    "source": "src.mp4",
    "replacement": "rep.mp4",
    "source_matte": "src.webm",
    "replacement_matte": "rep.webm",
}


def default_outputs(n: int) -> dict[str, bytes]:
    return {
        "src.mp4": frames_bytes([1] * n),
        "rep.mp4": frames_bytes([2] * n),
        "src.webm": matte_bytes([0] * n),
        "rep.webm": matte_bytes([0] * n),
    }


def run_union(tmp_path: Path, out: Path) -> UnionCompositeReport:
    return composite_video_union(
        tmp_path / "src.mp4",
        tmp_path / "rep.mp4",
        tmp_path / "src.webm",
        tmp_path / "rep.webm",
        out,
    )


def test_composite_video_union_streams_four_inputs_into_one_encoder(
    tmp_path: Path, monkeypatch: Any
) -> None:
    ffmpeg = FakeFfmpeg(
        {
            "src.mp4": frames_bytes([10, 20, 30]),
            "rep.mp4": frames_bytes([200, 210, 220]),
            "src.webm": matte_bytes([255, 0, 128]),
            "rep.webm": matte_bytes([0, 0, 64]),
        }
    )
    wire(monkeypatch, ffmpeg, fake_probe(3))
    out = tmp_path / "nested" / "v2.mp4"

    report = run_union(tmp_path, out)

    assert report.output_path == out
    assert (report.frames, report.width, report.height) == (3, W, H)
    assert report.frame_rate == Fraction(24, 1)
    assert report.soft_edge_ratio == pytest.approx(1 / 3)  # frame 2 only
    assert report.union_lift_ratio == pytest.approx(2 / 3)  # frames 0 and 2
    blended = (128 * 220 + (255 - 128) * 30 + 127) // 255
    assert ffmpeg.encoded == frames_bytes([200, 20, blended])
    assert out.parent.is_dir()
    assert not any(p.killed for p in ffmpeg.processes)


def test_union_decoders_force_libvpx_for_both_mattes_only(
    tmp_path: Path, monkeypatch: Any
) -> None:
    ffmpeg = FakeFfmpeg(default_outputs(2))
    wire(monkeypatch, ffmpeg, fake_probe(2))
    out = tmp_path / "v2.mp4"

    run_union(tmp_path, out)

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


def test_union_api_is_exported_from_the_package() -> None:
    assert video_character_skill.composite_video_union is composite_video_union
    assert video_character_skill.union_alpha is union_alpha
    assert video_character_skill.UnionCompositeReport is UnionCompositeReport
    exported = set(video_character_skill.__all__)
    assert {"composite_video_union", "union_alpha", "UnionCompositeReport"} <= exported


# -- ffmpeg failure propagation -----------------------------------------


@pytest.mark.parametrize("name", STREAMS)
def test_a_failing_decoder_is_reported_with_its_name_and_stderr(
    name: str, tmp_path: Path, monkeypatch: Any
) -> None:
    ffmpeg = FakeFfmpeg(default_outputs(2), failing=FILES[name], stderr="could not decode")
    wire(monkeypatch, ffmpeg, fake_probe(2))

    with pytest.raises(CompositeError, match=f"ffmpeg {name} exited 1: could not decode"):
        run_union(tmp_path, tmp_path / "v2.mp4")


def test_a_failing_encoder_is_reported(tmp_path: Path, monkeypatch: Any) -> None:
    ffmpeg = FakeFfmpeg(default_outputs(2), failing="encode", stderr="broken pipe")
    wire(monkeypatch, ffmpeg, fake_probe(2))

    with pytest.raises(CompositeError, match="ffmpeg encode exited 1: broken pipe"):
        run_union(tmp_path, tmp_path / "v2.mp4")


def test_a_decoder_that_dies_mid_stream_aborts_and_kills_the_rest(
    tmp_path: Path, monkeypatch: Any
) -> None:
    outputs = default_outputs(3)
    outputs["src.webm"] = matte_bytes([0])  # one frame, then silence
    ffmpeg = FakeFfmpeg(outputs, failing="src.webm", stderr="decode error")
    wire(monkeypatch, ffmpeg, fake_probe(3))

    with pytest.raises(CompositeError, match="source_matte ended during frame 1"):
        run_union(tmp_path, tmp_path / "v2.mp4")

    assert all(p.killed for p in ffmpeg.processes)


def test_a_matte_without_alpha_stops_the_run_before_any_process_spawns(
    tmp_path: Path, monkeypatch: Any
) -> None:
    ffmpeg = FakeFfmpeg(default_outputs(2))
    probe = fake_probe(2)

    def alpha_lost_on_one_matte(path: Path, decoder: str | None = None) -> VideoInfo:
        result = probe(path, decoder)
        return replace(result, pix_fmt="yuv420p") if path.name == "rep.webm" else result

    wire(monkeypatch, ffmpeg, alpha_lost_on_one_matte)

    with pytest.raises(CompositeError, match="replacement matte: matte .* has no alpha"):
        run_union(tmp_path, tmp_path / "v2.mp4")

    assert ffmpeg.commands == []
    assert not (tmp_path / "v2.mp4").exists()


def test_a_frame_count_mismatch_stops_the_run_before_any_process_spawns(
    tmp_path: Path, monkeypatch: Any
) -> None:
    ffmpeg = FakeFfmpeg(default_outputs(2))
    probe = fake_probe(2)

    def one_frame_short(path: Path, decoder: str | None = None) -> VideoInfo:
        result = probe(path, decoder)
        return replace(result, frame_count=1) if path.name == "src.webm" else result

    wire(monkeypatch, ffmpeg, one_frame_short)

    with pytest.raises(CompositeError, match="source matte: inputs differ in frame count"):
        run_union(tmp_path, tmp_path / "v2.mp4")

    assert ffmpeg.commands == []


def test_composite_video_union_rejects_a_missing_input_before_ffprobe_runs(
    tmp_path: Path,
) -> None:
    with pytest.raises(CompositeError, match="input not found"):
        run_union(tmp_path, tmp_path / "out.mp4")


# -- the single-matte path is unchanged by the shared pipeline ----------


def test_single_matte_composite_video_still_wires_three_decoders(
    tmp_path: Path, monkeypatch: Any
) -> None:
    ffmpeg = FakeFfmpeg(
        {
            "src.mp4": frames_bytes([10, 20]),
            "rep.mp4": frames_bytes([200, 210]),
            "m.webm": matte_bytes([0, 255]),
        }
    )
    wire(monkeypatch, ffmpeg, fake_probe(2))
    out = tmp_path / "v1.mp4"

    report = composite_video(tmp_path / "src.mp4", tmp_path / "rep.mp4", tmp_path / "m.webm", out)

    assert report == CompositeReport(
        output_path=out,
        frames=2,
        width=W,
        height=H,
        frame_rate=Fraction(24, 1),
        soft_edge_ratio=0.0,
    )
    assert ffmpeg.input_names() == ["src.mp4", "rep.mp4", "m.webm", "-"]
    matte = ffmpeg.commands[2]
    assert matte[matte.index("-c:v") + 1] == "libvpx-vp9"
    assert matte[matte.index("-pix_fmt") + 1] == "rgba"
    assert ffmpeg.encoded == frames_bytes([10, 210])
    assert not any(p.killed for p in ffmpeg.processes)


def test_single_matte_failure_propagation_is_unchanged(
    tmp_path: Path, monkeypatch: Any
) -> None:
    ffmpeg = FakeFfmpeg(
        {
            "src.mp4": frames_bytes([1]),
            "rep.mp4": frames_bytes([2]),
            "m.webm": matte_bytes([0]),
        },
        failing="m.webm",
        stderr="could not decode",
    )
    wire(monkeypatch, ffmpeg, fake_probe(1))

    with pytest.raises(CompositeError, match="ffmpeg matte exited 1: could not decode"):
        composite_video(
            tmp_path / "src.mp4", tmp_path / "rep.mp4", tmp_path / "m.webm", tmp_path / "v1.mp4"
        )
