"""Unit tests for the v9 local rim-tone correction.

The estimator is checked against a literal per-pixel reference of the
analysed arithmetic; the compositing path is checked to be v7 byte for byte
except on the corrected rim pixels. Nothing here decodes or encodes video.
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
    STREAMS,
    FakeFfmpeg,
    H,
    W,
    default_outputs,
    fake_probe,
    plane,
    wire,
)
from test_temporal_recovery import (
    FULL_FIT,
    Frame,
    P,
    clip_streams,
    replacement_frame,
    scene_outputs,
    solid,
    source_frame,
)
from video_character_skill.compositor import CompositeError, VideoInfo
from video_character_skill.hard_inset_recovery import erode_disk
from video_character_skill.rim_correction import (
    RIM_BANDS,
    RIM_CORE_THRESHOLD,
    RIM_MIN_BAND_SAMPLES,
    RIM_MIN_REFERENCE_SAMPLES,
    RIM_REFERENCE_DEPTHS,
    RIM_STRENGTH,
    RIM_WINDOW,
    V7_REMOVAL_THRESHOLD,
    RimCorrectedCompositeReport,
    RimFilter,
    RimStreamStats,
    composite_streams_rim_corrected,
    composite_video_rim_corrected,
    correct_rim,
    local_rim_offsets,
    rim_bands,
)
from video_character_skill.spatial_recovery import (
    SpatialRecoveryStreamStats,
    composite_streams_spatial_recovery,
)
from video_character_skill.temporal_recovery import recovery_regions

Bool = NDArray[np.bool_]


def assert_stats_equal(a: Any, b: Any) -> None:
    """Dataclass equality that treats NaN == NaN (no donors -> NaN distances)."""
    assert type(a) is type(b)
    for name, va in vars(a).items():
        vb = getattr(b, name)
        if isinstance(va, float) and math.isnan(va):
            assert math.isnan(vb), name
        else:
            assert va == vb, name

# Tiny-frame parameters: the 4x5 test frames cannot hold depth 8-12, so the
# stream tests use band (0,1) and reference (1,2) with 1-sample minimums.
TINY: dict[str, Any] = {"bands": ((0, 1),), "reference": (1, 2), "min_band_samples": 1,
                        "min_reference_samples": 1}


# -- reference implementation of the analysed arithmetic ------------------------


def reference_correct(
    rgb: NDArray[np.uint8], alpha: NDArray[np.uint8], *, window: int, strength: float,
    core_threshold: int = RIM_CORE_THRESHOLD, bands: tuple[tuple[int, int], ...] = RIM_BANDS,
    reference: tuple[int, int] = RIM_REFERENCE_DEPTHS, min_band: int = RIM_MIN_BAND_SAMPLES,
    min_ref: int = RIM_MIN_REFERENCE_SAMPLES,
) -> tuple[NDArray[np.uint8], Bool]:
    """Literal per-pixel version: clipped box window, means, offset, rint, clip."""
    h, w = alpha.shape
    core = alpha >= core_threshold
    er = {0: core}
    for r in sorted({r for pair in (*bands, reference) for r in pair}):
        er.setdefault(r, erode_disk(core, r))
    ref = er[reference[0]] & ~er[reference[1]]
    out = rgb.copy()
    corrected = np.zeros((h, w), dtype=np.bool_)
    half = window // 2
    for lo, hi in bands:
        band = er[lo] & ~er[hi]
        for y, x in zip(*np.nonzero(band), strict=True):
            ys = slice(max(y - half, 0), min(y + half + 1, h))
            xs = slice(max(x - half, 0), min(x + half + 1, w))
            in_band = band[ys, xs]
            in_ref = ref[ys, xs]
            if in_band.sum() < min_band or in_ref.sum() < min_ref:
                continue
            win = rgb[ys, xs].astype(np.float64)
            offset = win[in_ref].mean(axis=0) - win[in_band].mean(axis=0)
            value = np.rint(rgb[y, x].astype(np.float64) + strength * offset)
            out[y, x] = np.clip(value, 0, 255).astype(np.uint8)
            corrected[y, x] = True
    return out, corrected


def blob_alpha(
    h: int = 40, w: int = 44, rng: np.random.Generator | None = None
) -> NDArray[np.uint8]:
    """An opaque ellipse with a soft skirt, plus a few sub-250 pixels inside."""
    yy, xx = np.mgrid[:h, :w]
    inside = ((yy - h / 2) / (h * 0.42)) ** 2 + ((xx - w / 2) / (w * 0.44)) ** 2 <= 1
    alpha = np.zeros((h, w), dtype=np.uint8)
    alpha[inside] = 255
    ring = ((yy - h / 2) / (h * 0.47)) ** 2 + ((xx - w / 2) / (w * 0.49)) ** 2 <= 1
    alpha[ring & ~inside] = 150
    if rng is not None:
        for _ in range(3):
            alpha[int(rng.integers(h)), int(rng.integers(w))] = 249
    return alpha


def rim_image(alpha: NDArray[np.uint8], rng: np.random.Generator) -> NDArray[np.uint8]:
    """Dark interior with a brighter rim inside the core, random texture everywhere."""
    h, w = alpha.shape
    rgb = rng.integers(20, 60, (h, w, 3)).astype(np.float64)
    core = alpha >= 250
    for r, boost in ((0, 30), (1, 24), (2, 18), (3, 12), (4, 6)):
        band = erode_disk(core, r) & ~erode_disk(core, r + 1)
        rgb[band] += boost
    rgb[~core] = rng.integers(150, 200, (int((~core).sum()), 3))
    return np.clip(rgb, 0, 255).astype(np.uint8)


# -- 1: only the target band changes -------------------------------------------


def test_pixels_outside_the_target_band_are_identical_to_the_input() -> None:
    rng = np.random.default_rng(0)
    alpha = blob_alpha(rng=rng)
    rgb = rim_image(alpha, rng)
    before = rgb.copy()

    result = correct_rim(rgb, alpha)

    partition = rim_bands(alpha)
    np.testing.assert_array_equal(rgb, before)  # input untouched
    np.testing.assert_array_equal(result.rgb[~partition.target], rgb[~partition.target])
    assert (result.corrected <= partition.target).all()
    assert result.corrected_pixels > 0
    changed = np.any(result.rgb != rgb, axis=2)
    assert (changed <= result.corrected).all()
    assert result.target_pixels == int(partition.target.sum())


def test_target_is_depth_0_to_6_and_reference_is_8_to_12_of_the_250_core() -> None:
    alpha = blob_alpha(60, 64)
    partition = rim_bands(alpha)
    core = alpha >= 250
    np.testing.assert_array_equal(partition.core, core)
    np.testing.assert_array_equal(partition.target, core & ~erode_disk(core, 6))
    np.testing.assert_array_equal(partition.reference, erode_disk(core, 8) & ~erode_disk(core, 12))
    for (lo, hi), band in zip(RIM_BANDS, partition.bands, strict=True):
        np.testing.assert_array_equal(band, erode_disk(core, lo) & ~erode_disk(core, hi))
    assert RIM_BANDS == ((0, 1), (1, 2), (2, 3), (3, 4), (4, 6))
    assert RIM_REFERENCE_DEPTHS == (8, 12) and RIM_CORE_THRESHOLD == 250
    assert (RIM_WINDOW, RIM_STRENGTH) == (32, 0.5)
    assert (RIM_MIN_BAND_SAMPLES, RIM_MIN_REFERENCE_SAMPLES) == (8, 16)
    assert V7_REMOVAL_THRESHOLD == 32


def test_alpha_249_is_outside_the_core_and_soft_skirt_is_never_corrected() -> None:
    rng = np.random.default_rng(1)
    alpha = blob_alpha(rng=rng)
    rgb = rim_image(alpha, rng)
    result = correct_rim(rgb, alpha)
    assert not result.corrected[alpha < 250].any()
    np.testing.assert_array_equal(result.rgb[alpha < 250], rgb[alpha < 250])


# -- 2: alpha and the v7 masks are untouched ------------------------------------------


def test_correction_never_touches_alpha_and_v7_masks_are_unchanged() -> None:
    rng = np.random.default_rng(2)
    alpha = blob_alpha(rng=rng)
    alpha_before = alpha.copy()
    rgb = rim_image(alpha, rng)
    source_alpha = rng.integers(0, 256, alpha.shape, dtype=np.uint8)

    result = correct_rim(rgb, alpha)

    np.testing.assert_array_equal(alpha, alpha_before)
    before = recovery_regions(source_alpha, alpha, removal_threshold=32)
    after = recovery_regions(source_alpha, alpha_before, removal_threshold=32)
    for name in ("removal", "recovery_region", "own_background", "needs_temporal"):
        np.testing.assert_array_equal(getattr(before, name), getattr(after, name))
    np.testing.assert_array_equal(before.force_replacement, after.force_replacement)
    assert result.rgb.dtype == np.uint8 and result.rgb.shape == rgb.shape


def test_stream_stats_and_masks_are_identical_to_v7_and_only_rim_pixels_differ() -> None:
    # a solid replacement person covering the whole 4x5 frame: band (0,1) is the
    # border ring, reference (1,2) the 2x3 interior. Old person at P.
    sources = [source_frame(P)] * 3
    replacements = []
    rng = np.random.default_rng(3)
    for _ in range(3):
        rgb = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
        replacements.append((rgb, plane(255)))
    v7_out, v7_stats = run_v7(sources, replacements)
    v9_out, v9_stats, rim = run_v9(sources, replacements)

    assert_stats_equal(v9_stats, v7_stats)  # same masks, same recovery, same alpha
    partition = rim_bands(plane(255), **{k: TINY[k] for k in ("bands", "reference")})
    for i in range(3):
        np.testing.assert_array_equal(v9_out[i][~partition.target], v7_out[i][~partition.target])
    assert np.any(v9_out != v7_out)
    assert isinstance(rim, RimStreamStats)
    assert rim.valid_ratio == 1.0 and rim.target_ratio == pytest.approx(14 / (H * W))


# -- 3: strength 0 is the identity -------------------------------------------------------


def test_strength_zero_is_byte_identical() -> None:
    rng = np.random.default_rng(4)
    alpha = blob_alpha(rng=rng)
    rgb = rim_image(alpha, rng)
    result = correct_rim(rgb, alpha, strength=0.0)
    np.testing.assert_array_equal(result.rgb, rgb)
    assert result.corrected_pixels > 0 and result.max_abs_offset == 0.0
    sources = [source_frame(P)] * 3
    replacements = [replacement_frame((2, 3), alpha=255)] * 3
    v7_out, v7_stats = run_v7(sources, replacements)
    v9_out, v9_stats, rim = run_v9(sources, replacements, strength=0.0)
    np.testing.assert_array_equal(v9_out, v7_out)
    assert_stats_equal(v9_stats, v7_stats)
    assert rim.mean_abs_offset == 0.0


def test_the_hook_defaults_to_v6_behaviour_when_absent() -> None:
    sources = [source_frame(P)] * 2
    replacements = [replacement_frame((2, 3), alpha=255)] * 2
    a, sa = run_v7(sources, replacements)
    streams = clip_streams(sources, replacements)
    out = io.BytesIO()
    sb = composite_streams_spatial_recovery(
        streams["source"], streams["replacement"], streams["source_matte"],
        streams["replacement_matte"], out, width=W, height=H, frames=2,
        removal_threshold=32, replacement_filter=None, **FULL_FIT,
    )
    assert out.getvalue() == a.tobytes()
    assert_stats_equal(sa, sb)


# -- 4: exact arithmetic --------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(4))
def test_correction_matches_the_literal_reference(seed: int) -> None:
    rng = np.random.default_rng(10 + seed)
    alpha = blob_alpha(rng=rng)
    rgb = rim_image(alpha, rng)
    got = correct_rim(rgb, alpha)
    want, corrected = reference_correct(rgb, alpha, window=RIM_WINDOW, strength=RIM_STRENGTH)
    np.testing.assert_array_equal(got.rgb, want)
    np.testing.assert_array_equal(got.corrected, corrected)


def test_offset_is_mean_reference_minus_mean_band_over_the_clipped_box() -> None:
    rgb = np.zeros((7, 9, 3), dtype=np.uint8)
    band = np.zeros((7, 9), dtype=np.bool_)
    reference = np.zeros((7, 9), dtype=np.bool_)
    band[3, 1] = band[3, 2] = True
    reference[0, 0] = reference[6, 8] = reference[3, 8] = True
    rgb[3, 1] = (10, 20, 30)
    rgb[3, 2] = (30, 20, 10)
    rgb[0, 0] = (100, 0, 0)
    rgb[6, 8] = (200, 0, 60)
    rgb[3, 8] = (0, 90, 0)
    offset, valid = local_rim_offsets(rgb, band, reference, window=32, min_band_samples=2,
                                      min_reference_samples=3)
    # the 32 px box around (3,1) covers the whole 7x9 frame
    expected = np.array([100, 30, 20]) - np.array([20, 20, 20])
    np.testing.assert_allclose(offset[3, 1], expected)
    assert valid[3, 1]
    with_small_window, valid_small = local_rim_offsets(rgb, band, reference, window=4,
                                                       min_band_samples=2, min_reference_samples=3)
    assert not valid_small[3, 1] and (with_small_window[3, 1] == 0).all()


def test_half_strength_rounds_half_to_even_and_clips() -> None:
    rgb = np.zeros((3, 3, 3), dtype=np.uint8)
    rgb[1, 1] = (250, 3, 128)
    band = np.zeros((3, 3), dtype=np.bool_)
    band[1, 1] = True
    reference = np.zeros((3, 3), dtype=np.bool_)
    reference[0, 0] = True
    rgb[0, 0] = (255, 0, 129)  # offsets +5, -3, +1  -> half: +2.5, -1.5, +0.5
    alpha = np.zeros((3, 3), dtype=np.uint8)
    result = correct_rim(
        rgb, alpha, strength=0.5, bands=((0, 1),), reference=(1, 2),
        min_band_samples=1, min_reference_samples=1, core_threshold=1,
    )
    assert not result.corrected.any()  # alpha 0: no core at all
    # drive the same arithmetic through local_rim_offsets + the documented formula
    offset, valid = local_rim_offsets(rgb, band, reference, window=32, min_band_samples=1,
                                      min_reference_samples=1)
    value = np.clip(np.rint(rgb[1, 1] + 0.5 * offset[1, 1]), 0, 255)
    assert tuple(int(v) for v in value) == (252, 2, 128)  # 252.5->252, 1.5->2, 128.5->128


def test_insufficient_samples_leave_pixels_uncorrected() -> None:
    rng = np.random.default_rng(5)
    alpha = blob_alpha(rng=rng)
    rgb = rim_image(alpha, rng)
    result = correct_rim(rgb, alpha, min_reference_samples=10_000)
    np.testing.assert_array_equal(result.rgb, rgb)
    assert result.corrected_pixels == 0 and result.target_pixels > 0


# -- 5: boundary handling and determinism ------------------------------------------------


def test_windows_clipped_at_the_frame_border_match_the_reference() -> None:
    rng = np.random.default_rng(6)
    alpha = np.full((30, 34), 255, dtype=np.uint8)  # core touches every border
    rgb = rim_image(alpha, rng)
    got = correct_rim(rgb, alpha, window=16, min_band_samples=4, min_reference_samples=4,
                      bands=((0, 1), (1, 2)), reference=(3, 5))
    want, corrected = reference_correct(rgb, alpha, window=16, strength=RIM_STRENGTH,
                                        bands=((0, 1), (1, 2)), reference=(3, 5), min_band=4,
                                        min_ref=4)
    np.testing.assert_array_equal(got.rgb, want)
    np.testing.assert_array_equal(got.corrected, corrected)
    assert got.corrected[0, 0] or got.corrected[0, 1]  # border pixels are corrected too


def test_correction_is_deterministic_and_mirror_equivariant() -> None:
    rng = np.random.default_rng(7)
    alpha = blob_alpha(rng=rng)
    rgb = rim_image(alpha, rng)
    a = correct_rim(rgb, alpha)
    b = correct_rim(rgb.copy(), alpha.copy())
    np.testing.assert_array_equal(a.rgb, b.rgb)
    for flip in (np.fliplr, np.flipud):
        flipped = correct_rim(np.ascontiguousarray(flip(rgb)), np.ascontiguousarray(flip(alpha)))
        np.testing.assert_array_equal(flipped.rgb, np.ascontiguousarray(flip(a.rgb)))
    transposed = correct_rim(
        np.ascontiguousarray(rgb.transpose(1, 0, 2)), np.ascontiguousarray(alpha.T)
    )
    np.testing.assert_array_equal(transposed.rgb, np.ascontiguousarray(a.rgb.transpose(1, 0, 2)))


def test_bad_parameters_are_refused() -> None:
    alpha = blob_alpha()
    rgb = np.zeros((*alpha.shape, 3), dtype=np.uint8)
    with pytest.raises(CompositeError):
        correct_rim(rgb, alpha, strength=1.5)
    with pytest.raises(CompositeError):
        correct_rim(rgb, alpha, window=0)
    with pytest.raises(CompositeError):
        correct_rim(rgb.astype(np.int16), alpha)
    with pytest.raises(CompositeError):
        rim_bands(alpha, bands=((2, 1),))
    with pytest.raises(CompositeError):
        RimFilter(strength=-0.1)


def test_rim_filter_accumulates_stats() -> None:
    rng = np.random.default_rng(8)
    alpha = blob_alpha(rng=rng)
    rgb = rim_image(alpha, rng)
    f = RimFilter()
    out = f(rgb, alpha)
    np.testing.assert_array_equal(out, correct_rim(rgb, alpha).rgb)
    s = f.stats()
    assert s.target_ratio > 0 and 0 < s.valid_ratio <= 1 and s.mean_abs_offset > 0
    assert f.frames == 1 and 0 <= s.clipped_ratio <= 1
    assert RimFilter().stats() == RimStreamStats(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


# -- orchestration, against the fake ffmpeg -------------------------------------------------


def run_recovery(tmp_path: Path, out: Path, **kwargs: Any) -> RimCorrectedCompositeReport:
    return composite_video_rim_corrected(
        tmp_path / "src.mp4", tmp_path / "rep.mp4", tmp_path / "src.webm", tmp_path / "rep.webm",
        out, **kwargs,
    )


def test_composite_video_rim_corrected_matches_the_streaming_function(
    tmp_path: Path, monkeypatch: Any
) -> None:
    sources = [source_frame(P)] * 3
    rng = np.random.default_rng(9)
    replacements: list[Frame] = [
        (rng.integers(0, 256, (H, W, 3), dtype=np.uint8), plane(255)) for _ in range(3)
    ]
    ffmpeg = FakeFfmpeg(scene_outputs(sources, replacements))
    wire(monkeypatch, ffmpeg, fake_probe(3))
    out = tmp_path / "nested" / "v9.mp4"

    report = run_recovery(tmp_path, out)

    # the 4x5 frame holds no 8-12 px reference, so the default geometry corrects nothing
    outputs = scene_outputs(sources, replacements)
    expected_stats, expected_rim = composite_streams_rim_corrected(
        io.BytesIO(outputs["src.mp4"]),
        io.BytesIO(outputs["rep.mp4"]),
        io.BytesIO(outputs["src.webm"]),
        io.BytesIO(outputs["rep.webm"]),
        io.BytesIO(),
        width=W,
        height=H,
        frames=3,
    )
    assert_stats_equal(report.stats, expected_stats)
    assert report.rim == expected_rim
    assert report.rim.corrected_ratio == 0.0 and report.rim.target_ratio > 0
    assert (report.removal_threshold, report.dilation_radius) == (32, 4)
    assert (report.foreground_threshold, report.background_threshold) == (128, 32)
    assert (report.window, report.strength) == (32, 0.5)
    assert report.frame_rate == Fraction(24, 1) and report.output_path == out
    assert isinstance(report.stats, SpatialRecoveryStreamStats)
    assert not any(p.killed for p in ffmpeg.processes)


def test_rim_decoders_force_libvpx_for_both_mattes_only(tmp_path: Path, monkeypatch: Any) -> None:
    ffmpeg = FakeFfmpeg(default_outputs(2))
    wire(monkeypatch, ffmpeg, fake_probe(2))
    run_recovery(tmp_path, tmp_path / "v9.mp4")
    assert ffmpeg.input_names() == ["src.mp4", "rep.mp4", "src.webm", "rep.webm", "-"]
    for matte in ffmpeg.commands[2:4]:
        assert matte[matte.index("-c:v") + 1] == "libvpx-vp9"


def test_invalid_parameters_are_refused_before_anything_is_probed(
    tmp_path: Path, monkeypatch: Any
) -> None:
    ffmpeg = FakeFfmpeg(default_outputs(1))

    def never(path: Path, decoder: str | None = None) -> VideoInfo:
        raise AssertionError(f"probed {path}")

    wire(monkeypatch, ffmpeg, never)
    for bad in ({"strength": 2.0}, {"window": 0}, {"radius": -1}):
        with pytest.raises(CompositeError):
            run_recovery(tmp_path, tmp_path / "v9.mp4", **bad)
    assert ffmpeg.commands == []


def test_a_matte_without_alpha_stops_the_run(tmp_path: Path, monkeypatch: Any) -> None:
    ffmpeg = FakeFfmpeg(default_outputs(2))
    probe = fake_probe(2)

    def alpha_lost(path: Path, decoder: str | None = None) -> VideoInfo:
        result = probe(path, decoder)
        return replace(result, pix_fmt="yuv420p") if path.name == "rep.webm" else result

    wire(monkeypatch, ffmpeg, alpha_lost)
    with pytest.raises(CompositeError, match="replacement matte: matte .* has no alpha"):
        run_recovery(tmp_path, tmp_path / "v9.mp4")
    assert ffmpeg.commands == []


@pytest.mark.parametrize("short", STREAMS)
def test_premature_eof_is_reported(short: str) -> None:
    streams = clip_streams([source_frame()] * 3, [replacement_frame()] * 3)
    streams[short] = io.BytesIO(streams[short].getvalue()[: -H * W])
    with pytest.raises(CompositeError, match=f"{short} ended during frame 2"):
        composite_streams_rim_corrected(
            streams["source"], streams["replacement"], streams["source_matte"],
            streams["replacement_matte"], io.BytesIO(), width=W, height=H, frames=3,
        )


def test_rim_api_is_exported_from_the_package() -> None:
    package: Any = video_character_skill
    assert package.composite_video_rim_corrected is composite_video_rim_corrected
    assert package.RimCorrectedCompositeReport is RimCorrectedCompositeReport
    assert package.RIM_WINDOW == 32 and package.RIM_STRENGTH == 0.5
    assert "composite_video_rim_corrected" in video_character_skill.__all__


# -- helpers ---------------------------------------------------------------------------------


def run_v7(
    sources: list[Frame], replacements: list[Frame]
) -> tuple[NDArray[np.uint8], SpatialRecoveryStreamStats]:
    streams = clip_streams(sources, replacements)
    out = io.BytesIO()
    stats = composite_streams_spatial_recovery(
        streams["source"], streams["replacement"], streams["source_matte"],
        streams["replacement_matte"], out, width=W, height=H, frames=len(sources),
        removal_threshold=32, **FULL_FIT,
    )
    return np.frombuffer(out.getvalue(), np.uint8).reshape(len(sources), H, W, 3), stats


def run_v9(
    sources: list[Frame], replacements: list[Frame], *, strength: float = RIM_STRENGTH
) -> tuple[NDArray[np.uint8], SpatialRecoveryStreamStats, RimStreamStats]:
    streams = clip_streams(sources, replacements)
    out = io.BytesIO()
    rim_filter = TinyRimFilter(strength=strength)
    stats = composite_streams_spatial_recovery(
        streams["source"], streams["replacement"], streams["source_matte"],
        streams["replacement_matte"], out, width=W, height=H, frames=len(sources),
        removal_threshold=32, replacement_filter=rim_filter, **FULL_FIT,
    )
    frames = np.frombuffer(out.getvalue(), np.uint8).reshape(len(sources), H, W, 3)
    return frames, stats, rim_filter.stats()


class TinyRimFilter(RimFilter):
    """RimFilter with the tiny-frame band geometry."""

    def __call__(
        self, rgb: NDArray[np.uint8], replacement_alpha: NDArray[np.uint8]
    ) -> NDArray[np.uint8]:
        result = correct_rim(
            rgb, replacement_alpha, window=self.window, strength=self.strength, **TINY
        )
        size = float(rgb.shape[0] * rgb.shape[1])
        self.frames += 1
        self.target_total += result.target_pixels / size
        self.corrected_total += result.corrected_pixels / size
        self.target_pixels += result.target_pixels
        self.corrected_pixels += result.corrected_pixels
        self.clipped_pixels += result.clipped_pixels
        self.sum_abs_offset += result.sum_abs_offset
        self.max_abs_offset = max(self.max_abs_offset, result.max_abs_offset)
        return result.rgb


def test_tiny_geometry_sanity() -> None:
    partition = rim_bands(plane(255), bands=TINY["bands"], reference=TINY["reference"])
    assert partition.target.sum() == 14  # the border ring of the 4x5 frame
    assert partition.reference.sum() == 6  # the 2x3 interior
    assert solid((1, 2, 3)).shape == (H, W, 3)
