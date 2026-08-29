"""Tests for the v11 front door: pinned parameters, explicit audio, atomic output.

The engine is exercised two ways: through a recording stand-in for
``composite_video_rim_corrected`` (parameter pinning, audio and publication
semantics) and through the fake ``Popen`` shared with the v2-v11 tests
(byte-identity with the v11 streaming function). ``subprocess.run`` is
replaced by a fake that answers ffprobe and performs the mux on disk.
Nothing here needs a real ffmpeg.
"""

from __future__ import annotations

import inspect
import io
import os
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import fields
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

import video_character_skill
from test_compositor_union import FakeFfmpeg, H, W, default_outputs, fake_probe, wire
from test_rim_correction import assert_stats_equal
from test_temporal_recovery import P, replacement_frame, scene_outputs, source_frame
from video_character_skill import pipeline
from video_character_skill.compositor import CompositeError
from video_character_skill.pipeline import (
    V11CompositeReport,
    composite_video_v11,
    mux_audio_command,
    probe_audio_stream,
)
from video_character_skill.rim_correction import (
    RimCorrectedCompositeReport,
    RimStreamStats,
    composite_streams_rim_corrected,
)
from video_character_skill.spatial_recovery import SpatialRecoveryStreamStats

# The accepted v11 configuration, spelled out independently of the module's constants.
PINNED: dict[str, Any] = {
    "window": 32,
    "strength": 0.5,
    "removal_threshold": 32,
    "dilation_radius": 4,
    "background_threshold": 1,
    "foreground_threshold": 128,
    "radius": 24,
    "max_observations": 5,
    "residual_threshold": 32,
}
VIDEO_BYTES = b"rendered-video"
AUDIO_TAG = b"+audio"


# -- stand-ins ----------------------------------------------------------------------------


def zero(cls: Any) -> Any:
    """A dataclass instance with every field 0 — stats whose values do not matter here."""
    return cls(**{f.name: 0 for f in fields(cls)})


class FakeComposite:
    """Records the call the wrapper makes; writes the rendered file or fails."""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[tuple[Path, ...], Path, dict[str, Any]]] = []

    def __call__(
        self, source: Path, replacement: Path, source_matte: Path, replacement_matte: Path,
        output: Path, **kwargs: Any,
    ) -> RimCorrectedCompositeReport:
        self.calls.append(((source, replacement, source_matte, replacement_matte), output, kwargs))
        if self.fail:
            raise CompositeError("ffmpeg encode exited 1: boom")
        output.write_bytes(VIDEO_BYTES)
        return RimCorrectedCompositeReport(
            output_path=output, frames=2, width=W, height=H, frame_rate=Fraction(24, 1),
            removal_threshold=kwargs["removal_threshold"],
            dilation_radius=kwargs["dilation_radius"],
            background_threshold=kwargs["background_threshold"],
            residual_threshold=kwargs["residual_threshold"],
            foreground_threshold=kwargs["foreground_threshold"],
            radius=kwargs["radius"], max_observations=kwargs["max_observations"],
            window=kwargs["window"], strength=kwargs["strength"],
            stats=zero(SpatialRecoveryStreamStats), rim=zero(RimStreamStats),
        )


class FakeRun:
    """Stands in for ``subprocess.run``: answers the ffprobe audio query, performs the mux."""

    def __init__(
        self, *, audio_codec: str | None = "aac", mux_fails: bool = False, stderr: str = ""
    ) -> None:
        self.audio_codec = audio_codec
        self.mux_fails = mux_fails
        self.stderr = stderr
        self.commands: list[list[str]] = []
        self.kwargs: list[dict[str, Any]] = []

    def __call__(self, command: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert isinstance(command, list), "arguments must be list-form, never a shell string"
        self.commands.append(list(command))
        self.kwargs.append(kwargs)
        tool = Path(command[0]).name
        if tool == "ffprobe":
            stdout = f"index=0\ncodec_name={self.audio_codec}\n" if self.audio_codec else ""
            return subprocess.CompletedProcess(list(command), 0, stdout, "")
        assert tool == "ffmpeg"
        if self.mux_fails:
            return subprocess.CompletedProcess(list(command), 1, "", self.stderr)
        video = Path(command[command.index("-i") + 1])
        Path(command[-1]).write_bytes(video.read_bytes() + AUDIO_TAG)
        return subprocess.CompletedProcess(list(command), 0, "", "")


def tools_present(monkeypatch: Any, *, missing: str | None = None) -> None:
    def which(tool: str) -> str | None:
        return None if tool == missing else f"/opt/bin/{tool}"

    monkeypatch.setattr(shutil, "which", which)


def inputs(directory: Path) -> tuple[Path, Path, Path, Path]:
    paths = (
        directory / "src.mp4", directory / "rep.mp4", directory / "src.webm", directory / "rep.webm"
    )
    for path in paths:
        path.write_bytes(b"x")
    return paths


def audio_file(directory: Path, name: str = "audio.mp4") -> Path:
    path = directory / name
    path.write_bytes(b"a")
    return path


def setup(
    monkeypatch: Any, tmp_path: Path, *, composite: FakeComposite | None = None,
    run: FakeRun | None = None,
) -> tuple[FakeComposite, FakeRun, tuple[Path, Path, Path, Path], Path]:
    composite = composite or FakeComposite()
    run = run or FakeRun()
    tools_present(monkeypatch)
    monkeypatch.setattr(pipeline, "composite_video_rim_corrected", composite)
    monkeypatch.setattr(subprocess, "run", run)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    return composite, run, inputs(tmp_path / "in"), out_dir / "final.mp4"


def leftovers(output: Path, *keep: Path) -> list[Path]:
    return sorted(p for p in output.parent.iterdir() if p not in {output, *keep})


@pytest.fixture(autouse=True)
def make_input_dir(tmp_path: Path) -> None:
    (tmp_path / "in").mkdir()


# -- 1: every accepted value is pinned and forwarded exactly --------------------------------


def test_every_v11_parameter_is_pinned_and_forwarded_exactly(
    tmp_path: Path, monkeypatch: Any
) -> None:
    composite, _, clips, out = setup(monkeypatch, tmp_path)

    report = composite_video_v11(*clips, out, audio_source_path=None, crf=20, preset="fast")

    assert len(composite.calls) == 1
    passed, rendered, kwargs = composite.calls[0]
    assert passed == clips
    assert kwargs == {**PINNED, "crf": 20, "preset": "fast"}
    assert rendered != out and rendered.parent == out.parent
    assert rendered.name.startswith(".final.video.") and rendered.name.endswith(".part.mp4")
    assert (report.composite.background_threshold, report.composite.residual_threshold) == (1, 32)
    assert (report.composite.removal_threshold, report.composite.dilation_radius) == (32, 4)
    assert (report.composite.window, report.composite.strength) == (32, 0.5)
    assert (pipeline.V11_REMOVAL_THRESHOLD, pipeline.V11_DILATION_RADIUS) == (32, 4)
    assert (pipeline.V11_BACKGROUND_THRESHOLD, pipeline.V11_RESIDUAL_THRESHOLD) == (1, 32)
    assert (pipeline.V11_FOREGROUND_THRESHOLD, pipeline.V11_TEMPORAL_RADIUS) == (128, 24)
    assert (pipeline.V11_MAX_OBSERVATIONS, pipeline.V11_RIM_WINDOW) == (5, 32)
    assert pipeline.V11_RIM_STRENGTH == 0.5


def test_the_pinned_values_are_not_parameters_of_the_wrapper() -> None:
    parameters = inspect.signature(composite_video_v11).parameters
    assert not set(PINNED) & set(parameters)
    keyword_only = [n for n, p in parameters.items() if p.kind is p.KEYWORD_ONLY]
    assert keyword_only == ["audio_source_path", "crf", "preset"]
    assert parameters["audio_source_path"].default is inspect.Parameter.empty
    assert parameters["crf"].default == 16 and parameters["preset"].default == "slow"


# -- 2: the engine is untouched: byte-identical to the v11 streaming function ---------------


def test_wrapper_is_byte_identical_to_the_v11_stream_and_adds_no_processing(
    tmp_path: Path, monkeypatch: Any
) -> None:
    sources = [source_frame(), source_frame(P), source_frame()]
    replacements = [replacement_frame(), replacement_frame((2, 3)), replacement_frame()]
    outputs = scene_outputs(sources, replacements)
    ffmpeg = FakeFfmpeg(outputs)
    wire(monkeypatch, ffmpeg, fake_probe(3))
    tools_present(monkeypatch)
    clips = inputs(tmp_path / "in")
    out = tmp_path / "out" / "v11.mp4"

    report = composite_video_v11(*clips, out, audio_source_path=None)

    expected_stats, expected_rim = composite_streams_rim_corrected(
        io.BytesIO(outputs["src.mp4"]), io.BytesIO(outputs["rep.mp4"]),
        io.BytesIO(outputs["src.webm"]), io.BytesIO(outputs["rep.webm"]), io.BytesIO(),
        width=W, height=H, frames=3, **PINNED,
    )
    assert ffmpeg.encoded == _v11_stream_bytes(outputs)
    assert_stats_equal(report.composite.stats, expected_stats)
    assert report.composite.rim == expected_rim
    # exactly the engine's five processes: four decodes and one encode, nothing extra
    assert ffmpeg.input_names() == ["src.mp4", "rep.mp4", "src.webm", "rep.webm", "-"]
    for matte in ffmpeg.commands[2:4]:
        assert matte[matte.index("-c:v") + 1] == "libvpx-vp9"
    encode = ffmpeg.commands[4]
    assert Path(encode[-1]) != out and Path(encode[-1]).parent == out.parent
    assert encode[encode.index("-crf") + 1] == "16"
    assert encode[encode.index("-preset") + 1] == "slow"
    assert report.output_path == out and report.composite.output_path == out
    assert not any(p.killed for p in ffmpeg.processes)
    # the front door holds no pixel code of its own
    assert "np" not in vars(pipeline) and "numpy" not in vars(pipeline)


def _v11_stream_bytes(outputs: dict[str, bytes]) -> bytes:
    sink = io.BytesIO()
    composite_streams_rim_corrected(
        io.BytesIO(outputs["src.mp4"]), io.BytesIO(outputs["rep.mp4"]),
        io.BytesIO(outputs["src.webm"]), io.BytesIO(outputs["rep.webm"]), sink,
        width=W, height=H, frames=3, **PINNED,
    )
    return sink.getvalue()


# -- 3: None is an intentionally silent output ----------------------------------------------


def test_audio_none_publishes_the_rendered_video_without_a_mux(
    tmp_path: Path, monkeypatch: Any
) -> None:
    _, run, clips, out = setup(monkeypatch, tmp_path)

    report = composite_video_v11(*clips, out, audio_source_path=None)

    assert run.commands == []  # neither probed nor muxed
    assert out.read_bytes() == VIDEO_BYTES
    assert report == V11CompositeReport(
        output_path=out, composite=report.composite, audio_requested=False, audio_muxed=False,
        audio_source_path=None,
    )
    assert leftovers(out) == []


def test_audio_source_path_is_required_keyword_only(tmp_path: Path, monkeypatch: Any) -> None:
    _, _, clips, out = setup(monkeypatch, tmp_path)
    with pytest.raises(TypeError):
        composite_video_v11(*clips, out)  # type: ignore[call-arg]


# -- 4, 5: a valid audio source gives the expected mapping, copy and AAC ---------------------


def test_a_valid_audio_source_produces_the_expected_mux_command(
    tmp_path: Path, monkeypatch: Any
) -> None:
    composite, run, clips, out = setup(monkeypatch, tmp_path)
    audio = audio_file(tmp_path)

    report = composite_video_v11(*clips, out, audio_source_path=audio)

    probe, mux = run.commands
    assert probe[0] == "ffprobe" and probe[-1] == str(audio)
    assert probe[probe.index("-select_streams") + 1] == "a:0"
    rendered = composite.calls[0][1]
    mux_part = Path(mux[-1])
    assert mux == mux_audio_command(rendered, audio, mux_part)
    assert mux == [
        "ffmpeg", "-v", "error", "-nostdin", "-y",
        "-i", str(rendered), "-i", str(audio),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac",
        "-af", "apad", "-shortest",
        "-movflags", "+faststart",
        str(mux_part),
    ]
    assert mux_part != out and mux_part.parent == out.parent
    assert mux_part.name.startswith(".final.mux.") and mux_part.name.endswith(".part.mp4")
    assert out.read_bytes() == VIDEO_BYTES + AUDIO_TAG
    assert (report.audio_requested, report.audio_muxed, report.audio_source_path) == (
        True, True, audio,
    )
    assert report.output_path == out and report.composite.output_path == out
    assert leftovers(out) == []


def test_video_is_stream_copied_and_audio_is_encoded_to_aac(tmp_path: Path) -> None:
    command = mux_audio_command(tmp_path / "v.mp4", tmp_path / "a.wav", tmp_path / "o.mp4")
    assert command[command.index("-c:v") + 1] == "copy"
    assert command[command.index("-c:a") + 1] == "aac"
    assert "-shortest" in command
    assert command[command.index("-movflags") + 1] == "+faststart"
    assert command.count("-map") == 2
    assert command[command.index("-map") + 1] == "0:v:0"
    assert command[command.index("-map", command.index("-map") + 1) + 1] == "1:a:0"


def test_shorter_audio_is_padded_and_longer_audio_is_cut_at_the_video_end(tmp_path: Path) -> None:
    """The published file always has the full video duration.

    ``-af apad`` pads a short audio stream with silence for as long as the
    video runs; ``-shortest`` then ends the file with the video, cutting a
    long audio stream there. With the video stream-copied, nothing can
    shorten it.
    """
    command = mux_audio_command(tmp_path / "v.mp4", tmp_path / "a.wav", tmp_path / "o.mp4")
    assert command[command.index("-af") + 1] == "apad"
    assert command.count("-af") == 1 and command.count("-shortest") == 1
    assert command.index("-af") < command.index("-shortest")  # pad first, then bound by the video
    assert command[command.index("-c:v") + 1] == "copy"  # the video stream is never re-timed
    assert "-t" not in command and "-to" not in command and "-ss" not in command
    doc = " ".join((mux_audio_command.__doc__ or "").split())  # unwrap the docstring lines
    assert "full video duration" in doc and "silence" in doc and "never shortened" in doc


# -- 6: a missing or audio-less source, or a missing tool, fails before compositing ----------


def test_a_missing_audio_source_fails_before_compositing(tmp_path: Path, monkeypatch: Any) -> None:
    composite, run, clips, out = setup(monkeypatch, tmp_path)
    with pytest.raises(CompositeError, match="audio source not found"):
        composite_video_v11(*clips, out, audio_source_path=tmp_path / "absent.mp4")
    assert composite.calls == [] and run.commands == []
    assert not out.exists() and leftovers(out) == []


def test_an_audio_less_source_fails_before_compositing_with_the_silent_alternative(
    tmp_path: Path, monkeypatch: Any
) -> None:
    composite, run, clips, out = setup(monkeypatch, tmp_path, run=FakeRun(audio_codec=None))
    audio = audio_file(tmp_path)
    with pytest.raises(CompositeError, match="has no audio stream.*audio_source_path=None"):
        composite_video_v11(*clips, out, audio_source_path=audio)
    assert composite.calls == []
    assert [c[0] for c in run.commands] == ["ffprobe"]
    assert not out.exists() and leftovers(out) == []


def test_ffprobe_failure_on_the_audio_source_is_reported(tmp_path: Path, monkeypatch: Any) -> None:
    tools_present(monkeypatch)
    audio = audio_file(tmp_path)

    def fail(command: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(list(command), 1, "", "Invalid data found")

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(CompositeError, match="ffprobe failed for audio source.*Invalid data"):
        probe_audio_stream(audio)


def test_probe_audio_stream_returns_the_codec(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(subprocess, "run", FakeRun(audio_codec="pcm_s16le"))
    assert probe_audio_stream(audio_file(tmp_path)) == "pcm_s16le"


@pytest.mark.parametrize("tool", ["ffmpeg", "ffprobe"])
def test_a_missing_tool_fails_before_compositing_with_an_install_hint(
    tool: str, tmp_path: Path, monkeypatch: Any
) -> None:
    composite, run, clips, out = setup(monkeypatch, tmp_path)
    tools_present(monkeypatch, missing=tool)
    with pytest.raises(CompositeError, match=f"{tool} was not found on PATH.*install"):
        composite_video_v11(*clips, out, audio_source_path=audio_file(tmp_path))
    assert composite.calls == [] and run.commands == []


# -- 7: a composite failure preserves the output and cleans up -------------------------------


def test_composite_failure_preserves_an_existing_output_and_leaves_no_temporaries(
    tmp_path: Path, monkeypatch: Any
) -> None:
    composite, run, clips, out = setup(monkeypatch, tmp_path, composite=FakeComposite(fail=True))
    out.write_bytes(b"previous good render")
    audio = audio_file(tmp_path)
    with pytest.raises(CompositeError, match="ffmpeg encode exited 1: boom"):
        composite_video_v11(*clips, out, audio_source_path=audio)
    assert out.read_bytes() == b"previous good render"
    assert leftovers(out) == []
    assert [c[0] for c in run.commands] == ["ffprobe"]  # no mux was attempted


def test_a_failing_encoder_in_the_real_pipeline_preserves_the_output(
    tmp_path: Path, monkeypatch: Any
) -> None:
    ffmpeg = FakeFfmpeg(default_outputs(2), failing="encode", stderr="disk full")
    wire(monkeypatch, ffmpeg, fake_probe(2))
    tools_present(monkeypatch)
    clips = inputs(tmp_path / "in")
    out = tmp_path / "out" / "v11.mp4"
    out.parent.mkdir()
    out.write_bytes(b"previous good render")
    with pytest.raises(CompositeError, match="ffmpeg encode exited 1: disk full"):
        composite_video_v11(*clips, out, audio_source_path=None)
    assert out.read_bytes() == b"previous good render"
    assert leftovers(out) == []


# -- 8: a mux failure preserves the output and cleans up -------------------------------------


def test_mux_failure_preserves_an_existing_output_and_leaves_no_temporaries(
    tmp_path: Path, monkeypatch: Any
) -> None:
    run = FakeRun(mux_fails=True, stderr="Invalid audio stream")
    composite, run, clips, out = setup(monkeypatch, tmp_path, run=run)
    out.write_bytes(b"previous good render")
    audio = audio_file(tmp_path)
    with pytest.raises(CompositeError, match="ffmpeg mux exited 1: Invalid audio stream"):
        composite_video_v11(*clips, out, audio_source_path=audio)
    assert out.read_bytes() == b"previous good render"
    assert leftovers(out) == []
    assert len(composite.calls) == 1 and [c[0] for c in run.commands] == ["ffprobe", "ffmpeg"]


# -- 9: publication is one atomic os.replace, never a write to the final path ----------------


@pytest.mark.parametrize("with_audio", [False, True])
def test_publication_is_a_single_atomic_replace(
    with_audio: bool, tmp_path: Path, monkeypatch: Any
) -> None:
    composite, run, clips, out = setup(monkeypatch, tmp_path)
    audio = audio_file(tmp_path) if with_audio else None
    out.write_bytes(b"previous good render")
    replaces: list[tuple[Path, Path]] = []
    real_replace = os.replace

    expected = VIDEO_BYTES + AUDIO_TAG if with_audio else VIDEO_BYTES

    def recording(src: Any, dst: Any, **kwargs: Any) -> None:
        assert Path(src).read_bytes() == expected  # the temporary is complete before publication
        replaces.append((Path(src), Path(dst)))
        real_replace(src, dst, **kwargs)

    monkeypatch.setattr(os, "replace", recording)

    composite_video_v11(*clips, out, audio_source_path=audio)

    assert len(replaces) == 1 and replaces[0][1] == out
    assert replaces[0][0].parent == out.parent and replaces[0][0].suffix == ".mp4"
    # nothing ever targeted the final path directly
    assert composite.calls[0][1] != out
    assert all(Path(c[-1]) != out for c in run.commands)
    assert out.read_bytes() == (VIDEO_BYTES + AUDIO_TAG if with_audio else VIDEO_BYTES)
    assert leftovers(out) == []


def test_temporaries_are_unique_per_run(tmp_path: Path, monkeypatch: Any) -> None:
    composite, _, clips, out = setup(monkeypatch, tmp_path)
    composite_video_v11(*clips, out, audio_source_path=None)
    composite_video_v11(*clips, out, audio_source_path=None)
    first, second = (call[1] for call in composite.calls)
    assert first != second


# -- 10: paths with spaces travel as single list arguments ------------------------------------


def test_paths_with_spaces_are_passed_as_single_arguments(tmp_path: Path, monkeypatch: Any) -> None:
    composite, run, _, _ = setup(monkeypatch, tmp_path)
    spaced = tmp_path / "my clips dir"
    spaced.mkdir()
    clips = inputs(spaced)
    audio = audio_file(spaced, "my audio track.mp4")
    out = tmp_path / "final cut dir" / "final cut.mp4"

    composite_video_v11(*clips, out, audio_source_path=audio)

    assert composite.calls[0][0] == clips
    probe, mux = run.commands
    assert probe[-1] == str(audio) and str(audio) in mux
    assert str(composite.calls[0][1]) in mux
    assert all(" " not in arg or arg in {str(audio), str(composite.calls[0][1]), mux[-1]}
               for arg in mux)
    assert not any(kw.get("shell") for kw in run.kwargs)
    assert out.read_bytes() == VIDEO_BYTES + AUDIO_TAG


# -- 11: the API is public ---------------------------------------------------------------------


def test_the_v11_api_is_exported_from_the_package() -> None:
    package: Any = video_character_skill
    assert package.composite_video_v11 is composite_video_v11
    assert package.V11CompositeReport is V11CompositeReport
    assert "composite_video_v11" in video_character_skill.__all__
    assert "V11CompositeReport" in video_character_skill.__all__


# -- 12: invalid crf / preset / paths fail with actionable errors, before anything runs --------


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"crf": -1}, "crf must be in 0..51"),
        ({"crf": 52}, "crf must be in 0..51"),
        ({"crf": "16"}, "crf must be an int"),
        ({"crf": True}, "crf must be an int"),
        ({"preset": "fastest"}, "preset must be one of"),
        ({"preset": ""}, "preset must be one of"),
    ],
)
def test_invalid_crf_and_preset_are_refused(
    kwargs: dict[str, Any], message: str, tmp_path: Path, monkeypatch: Any
) -> None:
    composite, run, clips, out = setup(monkeypatch, tmp_path)
    with pytest.raises(CompositeError, match=message):
        composite_video_v11(*clips, out, audio_source_path=None, **kwargs)
    assert composite.calls == [] and run.commands == [] and leftovers(out) == []


def test_invalid_paths_are_refused_by_name(tmp_path: Path, monkeypatch: Any) -> None:
    composite, run, clips, out = setup(monkeypatch, tmp_path)
    src, rep, sm, rm = clips
    cases: list[tuple[tuple[Any, ...], dict[str, Any], str]] = [
        ((tmp_path / "in" / "nope.mp4", rep, sm, rm, out), {}, "source video not found"),
        ((src, tmp_path / "in" / "nope.mp4", sm, rm, out), {}, "replacement video not found"),
        ((src, rep, tmp_path / "in" / "nope.webm", rm, out), {}, "source matte not found"),
        ((src, rep, sm, tmp_path / "in" / "nope.webm", out), {}, "replacement matte not found"),
        ((str(src), rep, sm, rm, out), {}, "source video must be a pathlib.Path, got str"),
        ((src, rep, sm, rm, str(out)), {}, "output_path must be a pathlib.Path"),
        ((src, rep, sm, rm, out.with_suffix(".mov")), {}, r"output_path must end in \.mp4"),
        ((src, rep, sm, rm, out.parent), {}, "output_path must end in"),
        ((src, rep, sm, rm, out), {"audio_source_path": "audio.mp4"},
         "audio_source_path must be a pathlib.Path"),
    ]
    directory_output = out.parent / "dir.mp4"
    directory_output.mkdir()
    cases.append(((src, rep, sm, rm, directory_output), {}, "output_path is a directory"))
    for args, kwargs, message in cases:
        with pytest.raises(CompositeError, match=message):
            composite_video_v11(*args, **{"audio_source_path": None, **kwargs})
    assert composite.calls == [] and run.commands == []
    assert leftovers(out, directory_output) == []


# -- 13: the output may not alias any input ---------------------------------------------------

PROTECTED = (
    "source_path", "replacement_path", "source_matte_path", "replacement_matte_path",
    "audio_source_path",
)


def protected_inputs(tmp_path: Path) -> dict[str, Path]:
    src, rep, sm, rm = inputs(tmp_path / "in")
    return {
        "source_path": src, "replacement_path": rep, "source_matte_path": sm,
        "replacement_matte_path": rm, "audio_source_path": audio_file(tmp_path / "in"),
    }


def call_with_output(protected: dict[str, Path], output: Path) -> None:
    composite_video_v11(
        protected["source_path"], protected["replacement_path"],
        protected["source_matte_path"], protected["replacement_matte_path"], output,
        audio_source_path=protected["audio_source_path"],
    )


def guard_everything(monkeypatch: Any) -> tuple[FakeComposite, FakeRun]:
    """Any tool probe, subprocess or composite after the alias check is a failure."""
    composite, run = FakeComposite(), FakeRun()
    monkeypatch.setattr(pipeline, "composite_video_rim_corrected", composite)
    monkeypatch.setattr(subprocess, "run", run)

    def never(tool: str) -> str | None:
        raise AssertionError(f"tools were probed ({tool}) before the alias check")

    monkeypatch.setattr(shutil, "which", never)
    return composite, run


def try_link(link: Any, target: Path, path: Path) -> None:
    try:
        link(target, path)
    except (OSError, NotImplementedError) as exc:  # e.g. no privilege on Windows
        pytest.skip(f"{link.__name__} unsupported here: {exc}")


@pytest.mark.parametrize("which", PROTECTED)
def test_output_may_not_be_an_input_under_any_spelling(
    which: str, tmp_path: Path, monkeypatch: Any
) -> None:
    protected = protected_inputs(tmp_path)
    target = protected[which]
    before = {name: path.read_bytes() for name, path in protected.items()}
    composite, run = guard_everything(monkeypatch)
    monkeypatch.chdir(tmp_path)
    aliases: dict[str, Path] = {
        "identical": target,
        "relative": Path(os.path.relpath(target, tmp_path)),
        "dot-dot": tmp_path / "in" / ".." / "in" / target.name,
        "absolute-of-relative": Path(os.path.relpath(target, tmp_path)).absolute(),
    }
    for label, alias in aliases.items():
        with pytest.raises(CompositeError, match=f"output_path and {which} refer to the same"):
            call_with_output(protected, alias)
        assert composite.calls == [] and run.commands == [], label
    for name, path in protected.items():
        assert path.read_bytes() == before[name]  # nothing was touched, let alone removed


@pytest.mark.parametrize("which", PROTECTED)
def test_output_may_not_be_a_symlink_to_an_input_or_vice_versa(
    which: str, tmp_path: Path, monkeypatch: Any
) -> None:
    protected = protected_inputs(tmp_path)
    target = protected[which]
    composite, run = guard_everything(monkeypatch)
    links = tmp_path / "links"
    links.mkdir()
    link = links / "output.mp4"
    try_link(os.symlink, target, link)
    with pytest.raises(CompositeError, match=f"output_path and {which} refer to the same"):
        call_with_output(protected, link)  # output is a symlink to the input
    # the input itself given through a symlink, the output naming the real file
    through_link = dict(protected, **{which: link})
    with pytest.raises(CompositeError, match=f"output_path and {which} refer to the same"):
        call_with_output(through_link, target)
    # a symlinked directory component
    linked_dir = links / "in"
    try_link(os.symlink, tmp_path / "in", linked_dir)
    with pytest.raises(CompositeError, match=f"output_path and {which} refer to the same"):
        call_with_output(protected, linked_dir / target.name)
    assert composite.calls == [] and run.commands == []
    assert link.is_symlink() and target.is_file()  # neither side was modified or removed


@pytest.mark.parametrize("which", PROTECTED)
def test_output_may_not_be_a_hard_link_of_an_input(
    which: str, tmp_path: Path, monkeypatch: Any
) -> None:
    protected = protected_inputs(tmp_path)
    target = protected[which]
    composite, run = guard_everything(monkeypatch)
    hard = tmp_path / "elsewhere" / "output.mp4"
    hard.parent.mkdir()
    try_link(os.link, target, hard)
    assert hard.resolve() != target.resolve()  # not detectable by path normalization
    with pytest.raises(CompositeError, match=f"output_path and {which} refer to the same"):
        call_with_output(protected, hard)
    assert composite.calls == [] and run.commands == []
    assert hard.read_bytes() == target.read_bytes() and os.stat(hard).st_nlink == 2


def test_an_unrelated_output_next_to_the_inputs_is_accepted(
    tmp_path: Path, monkeypatch: Any
) -> None:
    protected = protected_inputs(tmp_path)
    tools_present(monkeypatch)
    composite = FakeComposite()
    monkeypatch.setattr(pipeline, "composite_video_rim_corrected", composite)
    monkeypatch.setattr(subprocess, "run", FakeRun())
    out = tmp_path / "in" / "final.mp4"  # same directory, different file: fine
    call_with_output(protected, out)
    assert out.read_bytes() == VIDEO_BYTES + AUDIO_TAG and len(composite.calls) == 1


def test_the_output_directory_is_created(tmp_path: Path, monkeypatch: Any) -> None:
    _, _, clips, _ = setup(monkeypatch, tmp_path)
    out = tmp_path / "deep" / "nested" / "final.mp4"
    report = composite_video_v11(*clips, out, audio_source_path=None)
    assert report.output_path == out and out.read_bytes() == VIDEO_BYTES
    assert leftovers(out) == []
