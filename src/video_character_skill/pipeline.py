"""The accepted v11 composite as one call: pinned parameters, audio, atomic output.

Why
---
:func:`video_character_skill.rim_correction.composite_video_rim_corrected`
is the v11 engine, but its *defaults* reproduce v9 (``background_threshold``
32, no tier-2 pass): the accepted configuration lived only in a README
snippet. The encoder it drives also writes video only (``-an``) straight to
the requested path, so a failed run could leave a truncated file where a
good one used to be. This module is the safe front door:

* every accepted v11 value is pinned here and passed explicitly — none of
  them is a parameter of :func:`composite_video_v11`;
* ``ffmpeg``/``ffprobe`` and the audio source are checked *before* the
  expensive composite starts;
* audio is an explicit decision: a :class:`~pathlib.Path` muxes that file's
  first audio stream, ``None`` means an intentionally silent output, and a
  requested source that cannot be used is an error, never a silent fallback;
* all intermediate files are unique temporaries in the output directory and
  the result is published with :func:`os.replace`, so an existing output is
  byte-identical to before unless the whole run succeeded.

The audio source must already be time-aligned to the source video
The wrapper only maps its first audio stream onto the rendered frames: audio
shorter than the video is padded with silence (``-af apad``), audio longer
than the video is cut at the video's end (``-shortest``), and the video
stream itself is never shortened by the audio's duration. It does not
retime, trim, offset or resample anything, and it does not check duration.
Use the same clip the frames came from (or an extract of it made with the
same trim), not the provider's output.

``output_path`` may not alias an input: the same path, a relative/absolute
spelling of it, a symlink to it or a hard link of it is refused before any
tool is probed, because publication would overwrite that input.

Not in scope here: provider polling or downloading, frame-rate / frame-count
alignment of the four inputs, and any paid call. The four inputs must
already exist locally and pass the compositor's own validation (one size,
one frame count, one frame rate, both mattes with alpha).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from video_character_skill.compositor import FFMPEG, FFPROBE, CompositeError, _check_int_range
from video_character_skill.rim_correction import (
    RimCorrectedCompositeReport,
    composite_video_rim_corrected,
)

__all__ = [
    "V11_BACKGROUND_THRESHOLD",
    "V11_DILATION_RADIUS",
    "V11_FOREGROUND_THRESHOLD",
    "V11_MAX_OBSERVATIONS",
    "V11_REMOVAL_THRESHOLD",
    "V11_RESIDUAL_THRESHOLD",
    "V11_RIM_STRENGTH",
    "V11_RIM_WINDOW",
    "V11_TEMPORAL_RADIUS",
    "X264_PRESETS",
    "V11CompositeReport",
    "composite_video_v11",
    "mux_audio_command",
    "probe_audio_stream",
    "require_ffmpeg_tools",
]

# The accepted v11 configuration (README, "Two-tier residual recovery").
# Pinned here on purpose: composite_video_v11 does not expose any of them.
V11_REMOVAL_THRESHOLD = 32
V11_DILATION_RADIUS = 4
V11_BACKGROUND_THRESHOLD = 1
V11_RESIDUAL_THRESHOLD = 32
V11_FOREGROUND_THRESHOLD = 128
V11_TEMPORAL_RADIUS = 24
V11_MAX_OBSERVATIONS = 5
V11_RIM_WINDOW = 32
V11_RIM_STRENGTH = 0.5

# The x264 presets ffmpeg accepts; anything else fails inside the encoder
# only after every input was decoded, so it is refused up front.
X264_PRESETS = frozenset(
    {
        "ultrafast", "superfast", "veryfast", "faster", "fast",
        "medium", "slow", "slower", "veryslow", "placebo",
    }
)


@dataclass(frozen=True)
class V11CompositeReport:
    """What one :func:`composite_video_v11` run produced."""

    output_path: Path
    """The published file."""
    composite: RimCorrectedCompositeReport
    """The engine's report; its ``output_path`` is rewritten to the published file."""
    audio_requested: bool
    """False iff ``audio_source_path`` was ``None`` (an intentionally silent output)."""
    audio_muxed: bool
    """True iff an audio stream was muxed into the published file."""
    audio_source_path: Path | None


# -- pre-flight checks ---------------------------------------------------


def require_ffmpeg_tools() -> None:
    """Fail with an install hint if ``ffmpeg`` or ``ffprobe`` is not on PATH."""
    for tool in (FFMPEG, FFPROBE):
        if shutil.which(tool) is None:
            raise CompositeError(
                f"{tool} was not found on PATH; the compositor decodes and encodes "
                "through FFmpeg. Install it (for example `brew install ffmpeg` or "
                f"`apt install ffmpeg`) and make sure `{tool}` is on PATH"
            )


def probe_audio_stream(path: Path) -> str:
    """Return the codec name of ``path``'s first audio stream.

    Raises:
        CompositeError: if the file is missing, ffprobe fails, or the file
            has no audio stream at all.
    """
    if not path.is_file():
        raise CompositeError(f"audio source not found: {path}")
    command = [
        FFPROBE, "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=index,codec_name",
        "-of", "default=noprint_wrappers=1:nokey=0",
        str(path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise CompositeError(
            f"ffprobe failed for audio source {path}: "
            f"{completed.stderr.strip() or completed.returncode}"
        )
    fields = dict(
        line.split("=", 1) for line in completed.stdout.splitlines() if "=" in line
    )
    codec = fields.get("codec_name", "").strip()
    if not codec:
        raise CompositeError(
            f"audio source {path} has no audio stream; pass audio_source_path=None "
            "for an intentionally silent output"
        )
    return codec


def mux_audio_command(video_path: Path, audio_path: Path, output_path: Path) -> list[str]:
    """The ffmpeg argument list that muxes ``audio_path``'s first audio stream onto ``video_path``.

    Video is stream-copied and audio is encoded as AAC for MP4 compatibility.
    The output always has the full video duration: ``-af apad`` pads audio
    that is shorter than the video with silence, and ``-shortest`` then cuts
    audio that is longer at the video's end — so the video stream is never
    shortened by the audio's duration. The moov atom is moved to the front.
    List form: paths are never shell-parsed.
    """
    return [
        FFMPEG, "-v", "error", "-nostdin", "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-af", "apad",
        "-shortest",
        "-movflags", "+faststart",
        str(output_path),
    ]


def _mux_audio(video_path: Path, audio_path: Path, output_path: Path) -> None:
    command = mux_audio_command(video_path, audio_path, output_path)
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        detail = f": {completed.stderr.strip()}" if completed.stderr.strip() else ""
        raise CompositeError(f"ffmpeg mux exited {completed.returncode}{detail}")


def _check_path(name: str, path: object) -> Path:
    if not isinstance(path, Path):
        raise CompositeError(f"{name} must be a pathlib.Path, got {type(path).__name__}")
    return path


def _check_input(name: str, path: object) -> Path:
    checked = _check_path(name, path)
    if not checked.is_file():
        raise CompositeError(f"{name} not found: {checked}")
    return checked


def _check_output_target(output: Path) -> None:
    if output.suffix.lower() != ".mp4":
        raise CompositeError(
            f"output_path must end in .mp4 (the output is an H.264 MP4), got {output}"
        )
    if output.is_dir():
        raise CompositeError(f"output_path is a directory: {output}")


def _same_destination(a: Path, b: Path) -> bool:
    """True if publishing to ``a`` would overwrite ``b``.

    Existing files are compared by identity (device and inode, so hard links
    and symlinks count); otherwise by the resolved, case-normalized path, so
    relative/absolute spellings and symlinked components count too.
    """
    if a.exists() and b.exists():
        try:
            if a.samefile(b):
                return True
        except OSError:
            pass
    return os.path.normcase(os.fspath(a.resolve())) == os.path.normcase(os.fspath(b.resolve()))


def _reject_output_aliases(output: Path, inputs: Sequence[tuple[str, Path]]) -> None:
    """Refuse an output that is one of the inputs under any spelling. Touches no file."""
    for name, path in inputs:
        if _same_destination(output, path):
            raise CompositeError(
                f"output_path and {name} refer to the same file ({output} vs {path}); "
                "publishing would overwrite that input, so choose a different output_path"
            )


def _temporary_path(output_path: Path, stage: str) -> Path:
    """A unique, already-created ``.part.mp4`` next to the output (same filesystem)."""
    fd, name = tempfile.mkstemp(
        dir=output_path.parent, prefix=f".{output_path.stem}.{stage}.", suffix=".part.mp4"
    )
    os.close(fd)
    return Path(name)


# -- the entry point -----------------------------------------------------


def composite_video_v11(
    source_path: Path,
    replacement_path: Path,
    source_matte_path: Path,
    replacement_matte_path: Path,
    output_path: Path,
    *,
    audio_source_path: Path | None,
    crf: int = 16,
    preset: str = "slow",
) -> V11CompositeReport:
    """Render the accepted v11 composite to ``output_path``, atomically, with explicit audio.

    Args:
        source_path: the original footage (its background is preserved).
        replacement_path: the replacement-person clip (Kling O1 output).
        source_matte_path: VEED matte of the person in the source clip.
        replacement_matte_path: VEED matte of the person in the replacement clip.
        output_path: the ``.mp4`` to publish. Created only on success; an
            existing file is left byte-identical if anything fails.
        audio_source_path: **keyword-only and required.** A path muxes that
            file's first audio stream into the output — it must already be
            time-aligned to ``source_path`` (see the module docstring); this
            wrapper does not retime or check duration. ``None`` produces an
            intentionally silent output. A path that is missing or has no
            audio stream is an error before compositing starts, never a
            silent fallback.
        crf: x264 quality, 0-51 (see :func:`video_character_skill.compositor.composite_video`).
        preset: an x264 preset name (:data:`X264_PRESETS`).

    Every accepted v11 value (removal 32/4, background threshold 1, residual
    threshold 32, replacement foreground 128, temporal radius 24, at most 5
    observations, rim window 32 at strength 0.5) is pinned and passed
    explicitly to :func:`composite_video_rim_corrected`; none is exposed.

    Raises:
        CompositeError: on an invalid argument, a missing input, an
            ``output_path`` that aliases any input (same path, relative or
            absolute spelling, symlink or hard link), a missing
            ``ffmpeg``/``ffprobe``, an unusable audio source (all before any
            decoding), or any failure of the composite or the mux.
    """
    _check_int_range("crf", crf, 0, 51)
    if preset not in X264_PRESETS:
        raise CompositeError(
            f"preset must be one of {sorted(X264_PRESETS)}, got {preset!r}"
        )
    source = _check_input("source video", source_path)
    replacement = _check_input("replacement video", replacement_path)
    source_matte = _check_input("source matte", source_matte_path)
    replacement_matte = _check_input("replacement matte", replacement_matte_path)
    output = _check_path("output_path", output_path)
    if audio_source_path is not None:
        _check_path("audio_source_path", audio_source_path)
    protected: list[tuple[str, Path]] = [
        ("source_path", source),
        ("replacement_path", replacement),
        ("source_matte_path", source_matte),
        ("replacement_matte_path", replacement_matte),
    ]
    if audio_source_path is not None:
        protected.append(("audio_source_path", audio_source_path))
    _reject_output_aliases(output, protected)
    _check_output_target(output)

    require_ffmpeg_tools()
    if audio_source_path is not None:
        probe_audio_stream(audio_source_path)

    output.parent.mkdir(parents=True, exist_ok=True)
    video_part = _temporary_path(output, "video")
    mux_part: Path | None = None
    try:
        report = composite_video_rim_corrected(
            source,
            replacement,
            source_matte,
            replacement_matte,
            video_part,
            window=V11_RIM_WINDOW,
            strength=V11_RIM_STRENGTH,
            removal_threshold=V11_REMOVAL_THRESHOLD,
            dilation_radius=V11_DILATION_RADIUS,
            background_threshold=V11_BACKGROUND_THRESHOLD,
            foreground_threshold=V11_FOREGROUND_THRESHOLD,
            radius=V11_TEMPORAL_RADIUS,
            max_observations=V11_MAX_OBSERVATIONS,
            residual_threshold=V11_RESIDUAL_THRESHOLD,
            crf=crf,
            preset=preset,
        )
        if audio_source_path is None:
            os.replace(video_part, output)
            muxed = False
        else:
            mux_part = _temporary_path(output, "mux")
            _mux_audio(video_part, audio_source_path, mux_part)
            os.replace(mux_part, output)
            muxed = True
    finally:
        video_part.unlink(missing_ok=True)
        if mux_part is not None:
            mux_part.unlink(missing_ok=True)

    return V11CompositeReport(
        output_path=output,
        composite=replace(report, output_path=output),
        audio_requested=audio_source_path is not None,
        audio_muxed=muxed,
        audio_source_path=audio_source_path,
    )
