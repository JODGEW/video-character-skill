"""Composite a replacement subject back onto the original background.

The pipeline's last step. Kling O1 gives us a video in which the person has
been replaced but the room was regenerated; VEED gives us that same clip's
person as an alpha matte. This module puts the original background back::

    output = alpha * replacement + (1 - alpha) * source

The point of the whole masked architecture is that background pixels survive
untouched, so the blend is not applied uniformly. Pixels are partitioned by
their matte alpha:

``alpha == 0``
    Copied byte-for-byte from the *source* frame. No arithmetic runs on them
    at all — the output buffer starts life as a copy of the source, so a
    transparent pixel is preserved by construction rather than by a blend that
    happens to round back to the same value.

``alpha == 255``
    Copied byte-for-byte from the *replacement* frame.

``0 < alpha < 255``
    The only pixels that are computed, using integer arithmetic (see
    :func:`composite_frame`). On the real matte this is 1.5-2.35 % of each
    frame.

What that guarantee does and does not cover
-------------------------------------------
The invariant is **pre-encode**, and only pre-encode. In the RGB24 frame handed
to the encoder:

* every ``alpha == 0`` pixel holds the source frame's exact RGB24 bytes;
* every ``alpha == 255`` pixel holds the replacement frame's exact RGB24 bytes;
* only ``0 < alpha < 255`` pixels have been computed at all.

Nothing is claimed about the encoded file. This pipeline is
``RGB24 -> libx264 -> yuv420p``, and the RGB->YUV 4:2:0 conversion happens
*before* x264 ever runs: it rounds every pixel through a colour-space matrix
and throws away three quarters of the chroma resolution. Decoding back to RGB
cannot undo either step. So decoded background pixels of the output ``.mp4``
are **not** guaranteed to match the source byte-for-byte, and lowering the
quantiser does not change that. ``crf=0`` makes x264 mathematically lossless
*with respect to the yuv420p frames it was given* — which are already not the
composited RGB — so it does not buy bit-exact RGB either.

Verifying the invariant therefore means checking the composited frames, not the
encoded file: compare what :func:`composite_frame` returns against the source
frame, or compare raw RGB24 piped out of the compositor. A lossless delivery
path is a separate concern and is not implemented here.

Decoding VP9 alpha
------------------
FFmpeg's *native* ``vp9`` decoder silently drops alpha: ``ffprobe`` reports
``pix_fmt=yuv420p`` for ``out/o1_matte.webm`` even though the file carries
``ALPHA_MODE=1``. Only ``-c:v libvpx-vp9`` surfaces it, as ``yuva420p``. A
matte decoded the wrong way would come back fully opaque and the compositor
would emit the replacement clip unchanged — a silent, total loss of the
background. :data:`ALPHA_DECODERS` forces the right decoder, and validation
fails closed if the decoded pixel format has no alpha channel.
"""

from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import IO

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "ALPHA_DECODERS",
    "ALPHA_PIX_FMTS",
    "CompositeError",
    "CompositeReport",
    "VideoInfo",
    "composite_frame",
    "composite_streams",
    "composite_video",
    "probe_video",
    "soft_edge_ratio",
]

RgbFrame = NDArray[np.uint8]
AlphaPlane = NDArray[np.uint8]

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"

# Codecs whose alpha channel only appears under a specific decoder. FFmpeg's
# built-in vp8/vp9 decoders report a 3-plane format and drop the alpha.
ALPHA_DECODERS = {"vp9": "libvpx-vp9", "vp8": "libvpx-vp8"}

# Decoded pixel formats that carry an alpha channel. Anything not listed is
# rejected rather than assumed opaque or assumed transparent; extend the set
# deliberately if a new matte format shows up.
ALPHA_PIX_FMTS = frozenset(
    {
        "yuva420p", "yuva422p", "yuva444p",
        "yuva420p9le", "yuva422p9le", "yuva444p9le",
        "yuva420p10le", "yuva422p10le", "yuva444p10le",
        "yuva420p16le", "yuva422p16le", "yuva444p16le",
        "rgba", "bgra", "argb", "abgr", "rgba64le", "bgra64le",
        "gbrap", "gbrap10le", "gbrap12le", "gbrap16le",
        "ya8", "ya16le",
    }
)


class CompositeError(RuntimeError):
    """A composite could not be produced from the given inputs."""


@dataclass(frozen=True)
class VideoInfo:
    """What ``ffprobe`` reports about one video stream."""

    path: Path
    codec_name: str
    width: int
    height: int
    pix_fmt: str
    frame_rate: Fraction
    frame_count: int

    @property
    def has_alpha(self) -> bool:
        return self.pix_fmt in ALPHA_PIX_FMTS

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height

    @property
    def rate_argument(self) -> str:
        """The frame rate as ffmpeg wants it, exactly — no float rounding."""
        return f"{self.frame_rate.numerator}/{self.frame_rate.denominator}"


@dataclass(frozen=True)
class CompositeReport:
    """What one composite run produced."""

    output_path: Path
    frames: int
    width: int
    height: int
    frame_rate: Fraction
    soft_edge_ratio: float
    """Mean fraction of pixels per frame that were blended rather than copied."""


# -- the pure part -------------------------------------------------------


def composite_frame(
    source: RgbFrame, replacement: RgbFrame, alpha: AlphaPlane
) -> RgbFrame:
    """Blend one frame. Pure: no I/O, no globals, no ffmpeg.

    Args:
        source: ``(H, W, 3)`` uint8 RGB, the original background.
        replacement: ``(H, W, 3)`` uint8 RGB, the edited clip.
        alpha: ``(H, W)`` uint8 matte, 0 = keep source, 255 = take replacement.

    Returns:
        A new ``(H, W, 3)`` uint8 RGB frame.

    Raises:
        CompositeError: on mismatched shapes or a non-uint8 input.

    Blending is integer-only, on the partial pixels alone::

        out = (alpha * replacement + (255 - alpha) * source + 127) // 255

    Integer rather than float so the result is exact and identical on every
    platform, with no rounding mode to agree on. The formula already returns
    ``source`` at alpha 0 and ``replacement`` at alpha 255, but those pixels
    are copied instead of computed: it makes the preservation guarantee
    structural rather than a property of the arithmetic, and it keeps the
    computed set down to the soft edge.
    """
    _check_frames(source, replacement, alpha)

    # Start from the source. Every alpha == 0 pixel is now already correct and
    # will never be written again.
    out: RgbFrame = source.copy()

    opaque = alpha == 255
    if opaque.any():
        out[opaque] = replacement[opaque]

    soft = (alpha != 0) & ~opaque
    if soft.any():
        a = alpha[soft].astype(np.uint32)[:, None]
        rep = replacement[soft].astype(np.uint32)
        src = source[soft].astype(np.uint32)
        out[soft] = ((a * rep + (255 - a) * src + 127) // 255).astype(np.uint8)

    return out


def soft_edge_ratio(alpha: AlphaPlane) -> float:
    """Fraction of the matte that is partially transparent, in ``[0, 1]``.

    The pixels :func:`composite_frame` actually computes. A ratio of 0 means
    the matte is hard-edged; a ratio near 1 means it is mush.
    """
    return float(np.count_nonzero((alpha != 0) & (alpha != 255))) / float(alpha.size)


def _check_frames(
    source: RgbFrame, replacement: RgbFrame, alpha: AlphaPlane
) -> None:
    if source.ndim != 3 or source.shape[2] != 3:
        raise CompositeError(f"source must be (H, W, 3), got {source.shape}")
    if replacement.shape != source.shape:
        raise CompositeError(
            f"replacement shape {replacement.shape} != source shape {source.shape}"
        )
    if alpha.shape != source.shape[:2]:
        raise CompositeError(
            f"alpha shape {alpha.shape} != source frame shape {source.shape[:2]}"
        )
    for name, array in (("source", source), ("replacement", replacement), ("alpha", alpha)):
        if array.dtype != np.uint8:
            raise CompositeError(f"{name} must be uint8, got {array.dtype}")


# -- the streaming part (file-like in, file-like out) --------------------


def composite_streams(
    source: IO[bytes],
    replacement: IO[bytes],
    matte: IO[bytes],
    output: IO[bytes],
    *,
    width: int,
    height: int,
    frames: int,
) -> float:
    """Composite ``frames`` frames of raw video, one at a time.

    Reads RGB24 from ``source`` and ``replacement`` and RGBA from ``matte``,
    writes RGB24 to ``output``. Only three frames are ever in memory.

    Returns:
        The mean soft-edge ratio across the clip.

    Raises:
        CompositeError: if any stream ends before ``frames`` frames, or if any
            still has data afterwards.
    """
    rgb_size = width * height * 3
    rgba_size = width * height * 4
    soft_total = 0.0

    for index in range(frames):
        raw_source = _read_exact(source, rgb_size, "source", index)
        raw_replacement = _read_exact(replacement, rgb_size, "replacement", index)
        raw_matte = _read_exact(matte, rgba_size, "matte", index)

        source_frame = np.frombuffer(raw_source, np.uint8).reshape(height, width, 3)
        replacement_frame = np.frombuffer(raw_replacement, np.uint8).reshape(
            height, width, 3
        )
        alpha = np.frombuffer(raw_matte, np.uint8).reshape(height, width, 4)[:, :, 3]

        soft_total += soft_edge_ratio(alpha)
        output.write(composite_frame(source_frame, replacement_frame, alpha).tobytes())

    for name, stream in (("source", source), ("replacement", replacement), ("matte", matte)):
        if stream.read(1):
            raise CompositeError(
                f"{name} has more than the expected {frames} frames"
            )

    return soft_total / frames if frames else 0.0


def _read_exact(stream: IO[bytes], size: int, name: str, index: int) -> bytes:
    """Read exactly ``size`` bytes; a short read means the stream ran out."""
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            got = size - remaining
            raise CompositeError(
                f"{name} ended during frame {index}: wanted {size} bytes, got {got}"
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


# -- probing -------------------------------------------------------------


def probe_video(path: Path, decoder: str | None = None) -> VideoInfo:
    """Describe a video's first video stream.

    Args:
        path: the file to probe.
        decoder: force this decoder, which changes the reported ``pix_fmt``
            for formats whose alpha only one decoder exposes.

    Raises:
        CompositeError: if the file is missing, ffprobe fails, or the stream
            is missing a field we need.
    """
    if not path.is_file():
        raise CompositeError(f"input not found: {path}")
    command = [FFPROBE, "-v", "error"]
    if decoder is not None:
        command += ["-c:v", decoder]
    command += [
        "-select_streams", "v:0",
        "-count_frames",
        "-show_entries", "stream=codec_name,width,height,pix_fmt,r_frame_rate,nb_read_frames",
        "-of", "default=noprint_wrappers=1:nokey=0",
        str(path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise CompositeError(
            f"ffprobe failed for {path}: {completed.stderr.strip() or completed.returncode}"
        )
    fields = dict(
        line.split("=", 1) for line in completed.stdout.splitlines() if "=" in line
    )
    try:
        return VideoInfo(
            path=path,
            codec_name=fields["codec_name"],
            width=int(fields["width"]),
            height=int(fields["height"]),
            pix_fmt=fields["pix_fmt"],
            frame_rate=Fraction(fields["r_frame_rate"]),
            frame_count=int(fields["nb_read_frames"]),
        )
    except (KeyError, ValueError, ZeroDivisionError) as exc:
        raise CompositeError(f"ffprobe gave no usable stream for {path}: {fields}") from exc


def _probe_matte(path: Path) -> VideoInfo:
    """Probe a matte through whichever decoder actually exposes its alpha."""
    info = probe_video(path)
    decoder = ALPHA_DECODERS.get(info.codec_name)
    return probe_video(path, decoder=decoder) if decoder else info


def _validate(source: VideoInfo, replacement: VideoInfo, matte: VideoInfo) -> None:
    """Fail closed on anything that would make same-index compositing wrong."""
    if not source.size == replacement.size == matte.size:
        raise CompositeError(
            "inputs differ in size: "
            f"source {source.width}x{source.height}, "
            f"replacement {replacement.width}x{replacement.height}, "
            f"matte {matte.width}x{matte.height}"
        )
    if source.frame_rate != replacement.frame_rate:
        raise CompositeError(
            f"source is {source.frame_rate} fps but replacement is "
            f"{replacement.frame_rate} fps"
        )
    counts = {
        "source": source.frame_count,
        "replacement": replacement.frame_count,
        "matte": matte.frame_count,
    }
    if len(set(counts.values())) != 1:
        raise CompositeError(f"inputs differ in frame count: {counts}")
    if source.frame_count == 0:
        raise CompositeError(f"{source.path} has no frames")
    if not matte.has_alpha:
        raise CompositeError(
            f"matte {matte.path} decodes to {matte.pix_fmt}, which has no alpha "
            f"channel (decoder: {ALPHA_DECODERS.get(matte.codec_name, 'default')})"
        )


# -- orchestration -------------------------------------------------------


def composite_video(
    source_path: Path,
    replacement_path: Path,
    matte_path: Path,
    output_path: Path,
    *,
    crf: int = 16,
    preset: str = "slow",
) -> CompositeReport:
    """Composite three clips into one H.264 MP4.

    Args:
        source_path: the original footage, kept wherever the matte is clear.
        replacement_path: the edited footage, taken wherever the matte is solid.
        matte_path: the subject matte; its **alpha channel** is the blend factor.
        output_path: the ``.mp4`` to write.
        crf: x264 quality, 0-51, lower is better. No value makes the decoded
            output match the composited RGB frames byte-for-byte: the
            ``rgb24 -> yuv420p`` conversion in front of x264 is itself lossy.
            See the module docstring.
        preset: x264 speed/efficiency preset.

    Returns:
        A :class:`CompositeReport` describing what was written.

    Raises:
        CompositeError: on any input mismatch, a frame-count disagreement, a
            matte without alpha, or a failing ffmpeg process.

    The alpha == 0 background-preservation guarantee applies to the frames fed
    to the encoder, not to the ``.mp4`` this writes. See the module docstring.
    """
    source = probe_video(source_path)
    replacement = probe_video(replacement_path)
    matte = _probe_matte(matte_path)
    _validate(source, replacement, matte)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logs = {name: tempfile.TemporaryFile() for name in ("source", "replacement", "matte", "encode")}
    processes: dict[str, subprocess.Popen[bytes]] = {}
    try:
        processes["source"] = _decoder(source_path, "rgb24", None, logs["source"])
        processes["replacement"] = _decoder(replacement_path, "rgb24", None, logs["replacement"])
        processes["matte"] = _decoder(
            matte_path, "rgba", ALPHA_DECODERS.get(matte.codec_name), logs["matte"]
        )
        processes["encode"] = _encoder(
            output_path, source, crf=crf, preset=preset, log=logs["encode"]
        )

        ratio = composite_streams(
            _pipe(processes["source"].stdout, "source"),
            _pipe(processes["replacement"].stdout, "replacement"),
            _pipe(processes["matte"].stdout, "matte"),
            _pipe(processes["encode"].stdin, "encode"),
            width=source.width,
            height=source.height,
            frames=source.frame_count,
        )
        _finish(processes, logs)
    finally:
        for process in processes.values():
            if process.poll() is None:
                process.kill()
        for log in logs.values():
            log.close()

    return CompositeReport(
        output_path=output_path,
        frames=source.frame_count,
        width=source.width,
        height=source.height,
        frame_rate=source.frame_rate,
        soft_edge_ratio=ratio,
    )


def _decoder(
    path: Path, pix_fmt: str, decoder: str | None, log: IO[bytes]
) -> subprocess.Popen[bytes]:
    command = [FFMPEG, "-v", "error", "-nostdin"]
    if decoder is not None:
        command += ["-c:v", decoder]
    command += ["-i", str(path), "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", pix_fmt, "-"]
    return subprocess.Popen(command, stdout=subprocess.PIPE, stderr=log)


def _encoder(
    path: Path, source: VideoInfo, *, crf: int, preset: str, log: IO[bytes]
) -> subprocess.Popen[bytes]:
    command = [
        FFMPEG, "-v", "error", "-nostdin", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{source.width}x{source.height}",
        "-r", source.rate_argument,
        "-i", "-",
        "-an",
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(path),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)


def _pipe(stream: IO[bytes] | None, name: str) -> IO[bytes]:
    if stream is None:
        raise CompositeError(f"ffmpeg {name} pipe was not opened")
    return stream


def _finish(
    processes: Mapping[str, subprocess.Popen[bytes]], logs: Mapping[str, IO[bytes]]
) -> None:
    """Close the encoder's input, then insist every process exited cleanly."""
    encode = processes["encode"]
    if encode.stdin is not None:
        encode.stdin.close()
    for name, process in processes.items():
        if process.stdout is not None:
            process.stdout.close()
        _raise_for_failure(name, process.wait(), _read_log(logs[name]))


def _read_log(log: IO[bytes]) -> str:
    log.seek(0)
    return log.read().decode("utf-8", errors="replace").strip()


def _raise_for_failure(name: str, returncode: int, stderr: str) -> None:
    if returncode != 0:
        detail = f": {stderr}" if stderr else ""
        raise CompositeError(f"ffmpeg {name} exited {returncode}{detail}")
