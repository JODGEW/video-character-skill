"""Unit tests for the temporal background-recovery compositor (v5).

Everything is exercised on tiny synthetic clips: the helpers are pure
functions over arrays, the streaming loop runs on file-like objects, and the
ffmpeg pipeline runs against the fake ``subprocess.Popen`` shared with the
v2-v4 tests. Nothing here decodes, encodes or writes a video. The v1-v4 test
modules are untouched; the last section pins down that v4 is not affected.
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
    matte_bytes,
    plane,
    wire,
)
from video_character_skill.compositor import (
    CompositeError,
    VideoInfo,
    composite_frame,
    composite_video_source_removal,
    source_removal_mask,
)
from video_character_skill.temporal_recovery import (
    MAX_TEMPORAL_OBSERVATIONS,
    PHOTOMETRIC_MIN_SAMPLES,
    PHOTOMETRIC_OFFSET_LIMIT,
    PHOTOMETRIC_SUBSAMPLE_STRIDE,
    REPLACEMENT_FOREGROUND_THRESHOLD,
    SOURCE_BACKGROUND_THRESHOLD,
    TEMPORAL_RECOVERY_RADIUS,
    PhotometricOffset,
    TemporalRecoveryCompositeReport,
    TemporalRecoveryStreamStats,
    aggregate_observations,
    apply_offset,
    composite_streams_temporal_recovery,
    composite_video_temporal_recovery,
    donor_frames,
    donor_offsets,
    photometric_offset,
    recover_pixels,
    recovered_background,
    recovery_effective_alpha,
    recovery_regions,
)

Rgb = tuple[int, int, int]
Frame = tuple[NDArray[np.uint8], NDArray[np.uint8]]  # (rgb, alpha)

BG: Rgb = (100, 120, 140)
PERSON: Rgb = (10, 10, 10)
O1_BG: Rgb = (200, 50, 50)
O1_PERSON: Rgb = (250, 250, 250)
P = (1, 1)  # the pixel most tests put the old person on

# The 4x5 test frames subsample to a single pixel at stride 8, so the tests
# that care about the photometric fit run it at full resolution.
FULL_FIT: dict[str, Any] = {"dilation_radius": 0, "offset_stride": 1, "offset_min_samples": 1}


def blend(a: int, rep: int, src: int) -> int:
    return (a * rep + (255 - a) * src + 127) // 255


def solid(color: Rgb) -> NDArray[np.uint8]:
    return np.full((H, W, 3), color, dtype=np.uint8)


def source_frame(
    person: tuple[int, int] | None = None, *, alpha: int = 200, background: Rgb = BG
) -> Frame:
    rgb, a = solid(background), plane(0)
    if person is not None:
        rgb[person] = PERSON
        a[person] = alpha
    return rgb, a


def replacement_frame(person: tuple[int, int] | None = None, *, alpha: int = 255) -> Frame:
    rgb, a = solid(O1_BG), plane(0)
    if person is not None:
        rgb[person] = O1_PERSON
        a[person] = alpha
    return rgb, a


def matte_plane_bytes(alpha: NDArray[np.uint8]) -> bytes:
    frame = np.zeros((H, W, 4), dtype=np.uint8)
    frame[:, :, :3] = 77
    frame[:, :, 3] = alpha
    return frame.tobytes()


def clip_streams(sources: list[Frame], replacements: list[Frame]) -> dict[str, io.BytesIO]:
    return {
        "source": io.BytesIO(b"".join(s[0].tobytes() for s in sources)),
        "replacement": io.BytesIO(b"".join(r[0].tobytes() for r in replacements)),
        "source_matte": io.BytesIO(b"".join(matte_plane_bytes(s[1]) for s in sources)),
        "replacement_matte": io.BytesIO(b"".join(matte_plane_bytes(r[1]) for r in replacements)),
    }


def run_streams(
    streams: dict[str, io.BytesIO], frames: int, out: io.BytesIO | None = None, **kwargs: Any
) -> TemporalRecoveryStreamStats:
    return composite_streams_temporal_recovery(
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
) -> tuple[NDArray[np.uint8], TemporalRecoveryStreamStats]:
    """Composite a synthetic clip; returns ``(frames (n, H, W, 3), stats)``."""
    out = io.BytesIO()
    params = {**FULL_FIT, **overrides}
    stats = run_streams(clip_streams(sources, replacements), len(sources), out, **params)
    frames = np.frombuffer(out.getvalue(), np.uint8).reshape(len(sources), H, W, 3)
    return frames, stats


def pixel(frames: NDArray[np.uint8], index: int, at: tuple[int, int] = P) -> Rgb:
    r, g, b = (int(v) for v in frames[index][at])
    return (r, g, b)


def test_the_constants_are_the_agreed_poc_values() -> None:
    assert SOURCE_BACKGROUND_THRESHOLD == 32
    assert REPLACEMENT_FOREGROUND_THRESHOLD == 128
    assert TEMPORAL_RECOVERY_RADIUS == 24
    assert MAX_TEMPORAL_OBSERVATIONS == 5
    assert PHOTOMETRIC_OFFSET_LIMIT == 64
    assert (PHOTOMETRIC_SUBSAMPLE_STRIDE, PHOTOMETRIC_MIN_SAMPLES) == (8, 256)


# -- donor ordering -------------------------------------------------------


def test_donor_offsets_are_nearest_first_and_alternate_sides() -> None:
    assert donor_offsets(3) == [-1, 1, -2, 2, -3, 3]
    assert donor_offsets(1) == [-1, 1]
    assert donor_offsets(0) == []
    assert donor_offsets(24)[:4] == [-1, 1, -2, 2]
    assert donor_offsets(24)[-2:] == [-24, 24]
    assert len(donor_offsets(24)) == 48


def test_donor_frames_exclude_the_target_and_clip_at_both_ends() -> None:
    assert donor_frames(2, 5, 1) == [1, 3]
    assert donor_frames(2, 5, 24) == [1, 3, 0, 4]
    assert donor_frames(0, 5, 3) == [1, 2, 3]  # clip start: one-sided
    assert donor_frames(4, 5, 3) == [3, 2, 1]  # clip end: one-sided
    assert donor_frames(0, 1, 24) == []
    assert 2 not in donor_frames(2, 225, 24)
    assert len(donor_frames(100, 225, 24)) == 48


def test_donor_frames_refuse_a_target_outside_the_clip() -> None:
    with pytest.raises(CompositeError, match="index must be in 0..4"):
        donor_frames(5, 5, 2)
    with pytest.raises(CompositeError, match="radius must be >= 0"):
        donor_offsets(-1)


# -- recovery_regions -----------------------------------------------------


def test_regions_partition_the_removal_mask() -> None:
    source = plane(0)
    source[1, 1] = 200  # core
    source[1, 2] = 31  # in the r=1 band: own background
    source[1, 0] = 32  # in the r=1 band: still hides the old person
    replacement = plane(0)
    replacement[0, 1] = 128  # in the band, solid replacement person
    replacement[2, 1] = 127  # in the band, not solid

    regions = recovery_regions(source, replacement, dilation_radius=1)

    np.testing.assert_array_equal(regions.removal, source_removal_mask(source, dilation_radius=1))
    assert regions.force_replacement[0, 1] and not regions.recovery_region[0, 1]
    assert regions.recovery_region[2, 1] and not regions.force_replacement[2, 1]
    assert regions.own_background[1, 2] and not regions.needs_temporal[1, 2]
    assert regions.needs_temporal[1, 0] and not regions.own_background[1, 0]
    assert regions.needs_temporal[1, 1]
    # the three inner masks partition the removal mask exactly
    inner = regions.own_background | regions.needs_temporal | regions.force_replacement
    np.testing.assert_array_equal(inner, regions.removal)
    assert not (regions.own_background & regions.needs_temporal).any()
    assert not (regions.recovery_region & regions.force_replacement).any()
    assert not regions.removal[3, 4]


def test_regions_thresholds_are_validated() -> None:
    with pytest.raises(CompositeError, match="background_threshold must be in 1..255"):
        recovery_regions(plane(0), plane(0), background_threshold=0)
    with pytest.raises(CompositeError, match="foreground_threshold must be in 1..255"):
        recovery_regions(plane(0), plane(0), foreground_threshold=256)
    with pytest.raises(CompositeError, match="replacement alpha shape"):
        recovery_regions(plane(0), np.zeros((H, W + 1), np.uint8))


# -- photometric offset ---------------------------------------------------


def test_offset_is_the_per_channel_median_difference_over_shared_background() -> None:
    target = solid((100, 120, 140))
    donor = solid((90, 130, 140))
    shared = np.ones((H, W), dtype=np.bool_)

    fit = photometric_offset(target, donor, shared, shared, stride=1, min_samples=1)

    assert fit == PhotometricOffset(offset=(10, -10, 0), samples=H * W, fitted=True)


def test_offset_uses_only_the_background_both_frames_share() -> None:
    target, donor = solid((100, 100, 100)), solid((100, 100, 100))
    donor[0, :] = (0, 0, 0)  # wildly different, but not background in the target
    target_bg = np.ones((H, W), dtype=np.bool_)
    target_bg[0, :] = False
    donor_bg = np.ones((H, W), dtype=np.bool_)

    fit = photometric_offset(target, donor, target_bg, donor_bg, stride=1, min_samples=1)

    assert fit.offset == (0, 0, 0)
    assert fit.samples == (H - 1) * W


def test_offset_is_clamped_to_the_documented_range() -> None:
    shared = np.ones((H, W), dtype=np.bool_)
    up = photometric_offset(
        solid((200, 0, 0)), solid((0, 200, 0)), shared, shared, stride=1, min_samples=1
    )
    assert up.offset == (64, -64, 0)
    custom = photometric_offset(
        solid((200, 0, 0)), solid((0, 200, 0)), shared, shared, stride=1, min_samples=1, limit=10
    )
    assert custom.offset == (10, -10, 0)


def test_offset_falls_back_to_zero_with_too_little_shared_background() -> None:
    target, donor = solid((100, 120, 140)), solid((90, 130, 140))
    shared = np.zeros((H, W), dtype=np.bool_)
    shared[0, :3] = True

    fit = photometric_offset(target, donor, shared, shared, stride=1, min_samples=4)

    assert fit == PhotometricOffset(offset=(0, 0, 0), samples=3, fitted=False)
    assert photometric_offset(target, donor, shared, shared, stride=1, min_samples=3).fitted


def test_offset_subsamples_on_a_deterministic_stride_grid() -> None:
    target, donor = solid((50, 50, 50)), solid((40, 40, 40))
    donor[::2, ::2] = (45, 45, 45)  # only the grid points differ by 5
    shared = np.ones((H, W), dtype=np.bool_)

    fit = photometric_offset(target, donor, shared, shared, stride=2, min_samples=1)

    assert fit.samples == 2 * 3  # rows 0, 2 x cols 0, 2, 4 of a 4x5 frame
    assert fit.offset == (5, 5, 5)


def test_offset_rounds_an_even_sample_count_half_to_even() -> None:
    target, donor = solid((0, 0, 0)), solid((0, 0, 0))
    donor[0, 0] = (10, 10, 10)
    donor[0, 1] = (20, 20, 20)
    shared = np.zeros((H, W), dtype=np.bool_)
    shared[0, :2] = True

    fit = photometric_offset(target, donor, shared, shared, stride=1, min_samples=1)

    assert fit.offset == (-15, -15, -15)


def test_offset_refuses_bad_inputs() -> None:
    shared = np.ones((H, W), dtype=np.bool_)
    with pytest.raises(CompositeError, match="rgb shapes differ"):
        photometric_offset(solid(BG), np.zeros((H, W + 1, 3), np.uint8), shared, shared)
    with pytest.raises(CompositeError, match="target rgb must be uint8"):
        photometric_offset(np.zeros((H, W, 3), np.float32), solid(BG), shared, shared)
    with pytest.raises(CompositeError, match="mask must be"):
        photometric_offset(solid(BG), solid(BG), shared, np.ones((H, W), np.uint8))
    with pytest.raises(CompositeError, match="stride must be >= 1"):
        photometric_offset(solid(BG), solid(BG), shared, shared, stride=0)
    with pytest.raises(CompositeError, match="limit must be in 0..255"):
        photometric_offset(solid(BG), solid(BG), shared, shared, limit=256)


def test_apply_offset_is_integer_safe_and_clipped() -> None:
    samples = np.array([[250, 5, 100], [0, 255, 128]], dtype=np.uint8)

    out = apply_offset(samples, (10, -10, 64))

    assert out.tolist() == [[255, 0, 164], [10, 245, 192]]
    assert out.dtype == np.uint8
    assert apply_offset(solid(BG), (-64, 0, 64)).shape == (H, W, 3)
    with pytest.raises(CompositeError, match=r"rgb must be \(..., 3\)"):
        apply_offset(np.zeros((H, W), np.uint8), (0, 0, 0))


# -- aggregation ----------------------------------------------------------


def test_aggregation_is_a_per_channel_median_over_valid_slots() -> None:
    observations = np.zeros((5, 2, 3), dtype=np.uint8)
    valid = np.zeros((5, 2), dtype=np.bool_)
    observations[:3, 0] = [(10, 200, 30), (200, 30, 10), (30, 10, 200)]
    valid[:3, 0] = True
    observations[:2, 1] = [(10, 10, 10), (20, 20, 20)]
    valid[:2, 1] = True

    rgb, counts = aggregate_observations(observations, valid)

    assert rgb.tolist() == [[30, 30, 30], [15, 15, 15]]
    assert counts.tolist() == [3, 2]


def test_aggregation_ignores_values_in_invalid_slots_and_reports_zero_for_none() -> None:
    observations = np.full((5, 3, 3), 255, dtype=np.uint8)
    valid = np.zeros((5, 3), dtype=np.bool_)
    observations[0, 0] = (7, 8, 9)
    valid[0, 0] = True
    observations[[1, 3], 1] = [(40, 40, 40), (44, 44, 44)]
    valid[[1, 3], 1] = True

    rgb, counts = aggregate_observations(observations, valid)

    assert rgb.tolist() == [[7, 8, 9], [42, 42, 42], [0, 0, 0]]
    assert counts.tolist() == [1, 2, 0]


def test_aggregation_refuses_bad_shapes() -> None:
    with pytest.raises(CompositeError, match=r"observations must be \(K, N, 3\)"):
        aggregate_observations(np.zeros((5, 2), np.uint8), np.zeros((5, 2), np.bool_))
    with pytest.raises(CompositeError, match="valid shape"):
        aggregate_observations(np.zeros((5, 2, 3), np.uint8), np.zeros((5, 3), np.bool_))


# -- recover_pixels -------------------------------------------------------


def needs_at(at: tuple[int, int] = P) -> NDArray[np.bool_]:
    mask = np.zeros((H, W), dtype=np.bool_)
    mask[at] = True
    return mask


DonorTuple = tuple[int, NDArray[np.uint8], NDArray[np.uint8]]


def donor(index: int, value: Rgb, alpha: int = 0) -> DonorTuple:
    rgb, a = source_frame()
    rgb[P] = value
    a[P] = alpha
    return index, rgb, a


def test_recover_pixels_takes_the_first_usable_donor_in_the_given_order() -> None:
    target_rgb, target_alpha = source_frame(P)
    donors = [donor(4, (1, 1, 1), alpha=200), donor(6, (2, 2, 2)), donor(3, (3, 3, 3))]

    result = recover_pixels(
        target_rgb, target_alpha, needs_at(), donors, target_index=5, max_observations=1,
        offset_stride=1, offset_min_samples=1,
    )

    assert result.rgb.tolist() == [[2, 2, 2]]
    assert result.counts.tolist() == [1]
    assert result.distances.tolist() == [1]
    assert result.donor_fits == 1 and result.zero_offset_fallbacks == 0


def test_recover_pixels_keeps_at_most_max_observations_nearest_first() -> None:
    target_rgb, target_alpha = source_frame(P)
    values = [10, 20, 30, 40, 50, 200, 210, 220]
    donors = [donor(10 + k, (v, v, v)) for k, v in enumerate(values)]

    result = recover_pixels(
        target_rgb, target_alpha, needs_at(), donors, target_index=0,
        offset_stride=1, offset_min_samples=1,
    )

    assert result.counts.tolist() == [5]
    assert result.rgb.tolist() == [[30, 30, 30]]  # median of the first five, not of all eight
    assert result.distances.tolist() == [10, 11, 12, 13, 14]
    assert result.donor_fits == 5  # the search stopped once the pixel was full


def test_recover_pixels_usability_boundary_is_31_in_32_out() -> None:
    target_rgb, target_alpha = source_frame(P)
    donors = [donor(1, (5, 5, 5), alpha=32), donor(2, (6, 6, 6), alpha=31)]

    result = recover_pixels(
        target_rgb, target_alpha, needs_at(), donors, target_index=0,
        offset_stride=1, offset_min_samples=1,
    )

    assert result.rgb.tolist() == [[6, 6, 6]]
    assert result.distances.tolist() == [2]


def test_recover_pixels_leaves_unrecoverable_pixels_at_count_zero() -> None:
    target_rgb, target_alpha = source_frame(P)

    result = recover_pixels(target_rgb, target_alpha, needs_at(), [], target_index=0)

    assert result.counts.tolist() == [0]
    assert result.distances.size == 0
    assert result.donor_fits == 0


def test_recover_pixels_refuses_the_target_as_its_own_donor() -> None:
    target_rgb, target_alpha = source_frame(P)
    with pytest.raises(CompositeError, match="frame 3 offered as its own donor"):
        recover_pixels(target_rgb, target_alpha, needs_at(), [donor(3, BG)], target_index=3)


def test_recover_pixels_counts_zero_offset_fallbacks() -> None:
    target_rgb, target_alpha = source_frame(P)
    donors = [donor(1, (50, 60, 70)), donor(2, (50, 60, 70))]

    result = recover_pixels(
        target_rgb, target_alpha, needs_at(), donors, target_index=0,
        offset_stride=1, offset_min_samples=1000,
    )

    assert result.donor_fits == 2 and result.zero_offset_fallbacks == 2
    assert result.rgb.tolist() == [[50, 60, 70]]


# -- background plate and effective alpha --------------------------------


def test_recovered_background_layers_recovered_then_o1_over_the_source() -> None:
    source_rgb, source_alpha = source_frame(P)
    replacement_rgb, _ = replacement_frame()
    needs = needs_at()
    needs[2, 3] = True
    result = recover_pixels(
        source_rgb, source_alpha, needs, [donor(1, (1, 2, 3))], target_index=0,
        offset_stride=1, offset_min_samples=1,
    )
    # pixel (1,1) recovered from the donor; pixel (2,3) had nothing to borrow
    assert result.counts.tolist() == [1, 1]  # (2,3) is background in the donor too
    result = replace(result, counts=np.array([1, 0]))

    background = recovered_background(source_rgb, replacement_rgb, result)

    assert tuple(background[P]) == (1, 2, 3)
    assert tuple(background[2, 3]) == O1_BG
    assert tuple(background[0, 0]) == BG
    assert tuple(source_rgb[P]) == PERSON  # input untouched


def test_effective_alpha_forces_only_the_v4_force_replacement_pixels() -> None:
    replacement = plane(0)
    replacement[0, :] = 200
    replacement[1, :] = 127
    force = np.zeros((H, W), dtype=np.bool_)
    force[0, 0] = True

    out = recovery_effective_alpha(replacement, force)

    assert int(out[0, 0]) == 255
    assert out[0, 1:].tolist() == [200] * (W - 1)
    assert np.unique(out[1]).tolist() == [127]
    with pytest.raises(CompositeError, match="force_replacement must be a bool mask"):
        recovery_effective_alpha(replacement, np.zeros((H, W), np.uint8))


# -- streaming: the invariants ------------------------------------------


def test_same_coordinate_donor_recovery_replaces_the_old_person_with_real_background() -> None:
    sources = [source_frame(), source_frame(P), source_frame()]
    replacements = [replacement_frame()] * 3

    frames, stats = run_clip(sources, replacements)

    assert pixel(frames, 1) == BG  # borrowed from frames 0 and 2
    assert pixel(frames, 1) not in (PERSON, O1_BG)
    assert pixel(frames, 1, (0, 0)) == BG  # outside the removal mask: plain source
    assert pixel(frames, 0) == BG and pixel(frames, 2) == BG
    assert stats.temporal_recovered_ratio == pytest.approx(1 / (H * W) / 3)
    assert stats.temporal_unrecovered_ratio == 0.0
    assert stats.mean_observations_per_recovered_pixel == 2.0


def test_own_frame_background_is_never_replaced() -> None:
    target_rgb, target_alpha = source_frame(P)
    target_rgb[1, 2] = (7, 7, 7)
    target_alpha[1, 2] = 31  # inside the r=1 band, real background in its own frame
    donor_rgb, donor_alpha = source_frame()
    donor_rgb[1, 2] = (99, 99, 99)
    sources = [(donor_rgb, donor_alpha), (target_rgb, target_alpha), (donor_rgb, donor_alpha)]
    replacements = [replacement_frame()] * 3

    frames, stats = run_clip(sources, replacements, dilation_radius=1)

    assert pixel(frames, 1, (1, 2)) == (7, 7, 7)
    assert pixel(frames, 1) == BG  # the core pixel itself was borrowed
    assert stats.own_background_ratio > 0


def test_own_background_boundary_31_keeps_source_32_borrows() -> None:
    def clip_with_band_alpha(alpha: int) -> Rgb:
        target_rgb, target_alpha = source_frame(P)
        target_rgb[1, 2] = (7, 7, 7)
        target_alpha[1, 2] = alpha
        donor_rgb, donor_alpha = source_frame()
        donor_rgb[1, 2] = (99, 99, 99)
        frames, _ = run_clip(
            [(donor_rgb, donor_alpha), (target_rgb, target_alpha)],
            [replacement_frame()] * 2,
            dilation_radius=1,
        )
        return pixel(frames, 1, (1, 2))

    assert clip_with_band_alpha(31) == (7, 7, 7)
    assert clip_with_band_alpha(32) == (99, 99, 99)


def test_clip_start_window_is_one_sided_and_still_recovers() -> None:
    late = source_frame()
    late[0][P] = (1, 2, 3)
    sources = [source_frame(P), source_frame(P), late]

    frames, stats = run_clip(sources, [replacement_frame()] * 3)

    assert pixel(frames, 0) == (1, 2, 3)  # frame 1 unusable, frame 2 usable
    assert pixel(frames, 1) == (1, 2, 3)
    assert stats.median_donor_distance == 1.0 and stats.p90_donor_distance == 2.0


def test_clip_end_window_is_one_sided_and_still_recovers() -> None:
    early = source_frame()
    early[0][P] = (4, 5, 6)
    sources = [early, source_frame(P), source_frame(P)]

    frames, _ = run_clip(sources, [replacement_frame()] * 3)

    assert pixel(frames, 2) == (4, 5, 6)  # frame 1 unusable, frame 0 usable
    assert pixel(frames, 1) == (4, 5, 6)


def test_at_most_five_observations_are_used_through_the_stream() -> None:
    by_frame = {4: 10, 6: 20, 3: 30, 7: 40, 2: 50, 8: 200, 1: 210, 9: 220, 0: 230, 10: 240}
    sources = []
    for i in range(11):
        if i == 5:
            sources.append(source_frame(P))
        else:
            frame = source_frame()
            frame[0][P] = (by_frame[i],) * 3
            sources.append(frame)

    frames, stats = run_clip(sources, [replacement_frame()] * 11)

    assert pixel(frames, 5) == (30, 30, 30)  # median of frames 4, 6, 3, 7, 2
    assert stats.mean_observations_per_recovered_pixel == 5.0
    assert stats.p90_donor_distance == 3.0


def test_recovered_value_is_the_per_channel_median() -> None:
    values = [(10, 200, 30), (200, 30, 10), (30, 10, 200)]
    sources = [source_frame(P)]
    for value in values:
        frame = source_frame()
        frame[0][P] = value
        sources.append(frame)

    frames, _ = run_clip(sources, [replacement_frame()] * 4)

    assert pixel(frames, 0) == (30, 30, 30)


def test_donor_usability_boundary_31_usable_32_not() -> None:
    def clip_with_donor_alpha(alpha: int) -> Rgb:
        donor_rgb, donor_alpha = source_frame()
        donor_rgb[P] = (1, 2, 3)
        donor_alpha[P] = alpha
        frames, _ = run_clip([source_frame(P), (donor_rgb, donor_alpha)], [replacement_frame()] * 2)
        return pixel(frames, 0)

    assert clip_with_donor_alpha(31) == (1, 2, 3)
    assert clip_with_donor_alpha(32) == O1_BG  # unusable -> O1 fallback


def test_replacement_alpha_127_recovers_128_forces_replacement() -> None:
    def output_at(alpha: int) -> Rgb:
        frames, _ = run_clip(
            [source_frame(), source_frame(P), source_frame()],
            [replacement_frame(), replacement_frame(P, alpha=alpha), replacement_frame()],
        )
        return pixel(frames, 1)

    expected_blend = tuple(blend(127, o, b) for o, b in zip(O1_PERSON, BG, strict=True))
    assert output_at(127) == expected_blend  # O1 soft edge over recovered real background
    assert output_at(128) == O1_PERSON  # forced opaque, exactly as v4


def test_v4_force_replacement_is_retained_for_partial_replacement_alpha() -> None:
    frames, stats = run_clip(
        [source_frame(P)], [replacement_frame(P, alpha=200)]
    )
    assert pixel(frames, 0) == O1_PERSON
    assert stats.recovery_region_ratio == 0.0


def test_photometric_offset_corrects_a_darker_donor_per_channel() -> None:
    donor_rgb, donor_alpha = source_frame(background=(90, 130, 140))
    donor_rgb[P] = (50, 60, 70)

    frames, stats = run_clip([source_frame(P), (donor_rgb, donor_alpha)], [replacement_frame()] * 2)

    assert pixel(frames, 0) == (60, 50, 70)
    assert stats.donor_fits == 1 and stats.zero_offset_fallbacks == 0


def test_photometric_offset_is_clipped_at_plus_minus_64() -> None:
    donor_rgb, donor_alpha = source_frame(background=(0, 120, 255))
    donor_rgb[P] = (50, 60, 70)

    frames, _ = run_clip([source_frame(P), (donor_rgb, donor_alpha)], [replacement_frame()] * 2)

    assert pixel(frames, 0) == (114, 60, 6)  # +100 -> +64, -115 -> -64


def test_insufficient_shared_background_uses_zero_offset_and_is_counted() -> None:
    donor_rgb, donor_alpha = source_frame(background=(90, 130, 140))
    donor_rgb[P] = (50, 60, 70)

    frames, stats = run_clip(
        [source_frame(P), (donor_rgb, donor_alpha)],
        [replacement_frame()] * 2,
        offset_min_samples=1000,
    )

    assert pixel(frames, 0) == (50, 60, 70)  # uncorrected
    assert stats.donor_fits == 1 and stats.zero_offset_fallbacks == 1


def test_unrecoverable_pixel_falls_back_to_o1_never_the_old_person() -> None:
    frames, stats = run_clip([source_frame(P)], [replacement_frame()])
    assert pixel(frames, 0) == O1_BG
    assert stats.temporal_unrecovered_ratio == pytest.approx(1 / (H * W))
    assert stats.o1_fallback_ratio == 1.0
    assert math.isnan(stats.median_donor_distance)

    # a partial replacement alpha there still yields the replacement pixel exactly
    frames, _ = run_clip([source_frame(P)], [replacement_frame(P, alpha=50)])
    assert pixel(frames, 0) == O1_PERSON


def test_outside_the_removal_mask_output_equals_plain_replacement_compositing() -> None:
    rng = np.random.default_rng(11)
    sources: list[Frame] = []
    replacements: list[Frame] = []
    def random_frame() -> Frame:
        return (
            rng.integers(0, 256, (H, W, 3), dtype=np.uint8),
            rng.integers(0, 256, (H, W), dtype=np.uint8),
        )

    for _ in range(3):
        sources.append(random_frame())
        replacements.append(random_frame())

    frames, _ = run_clip(sources, replacements)

    for i in range(3):
        outside = ~source_removal_mask(sources[i][1], dilation_radius=0)
        assert outside.any()
        plain = composite_frame(sources[i][0], replacements[i][0], replacements[i][1])
        np.testing.assert_array_equal(frames[i][outside], plain[outside])


def test_stats_are_exact_on_a_small_clip() -> None:
    sources = [source_frame(), source_frame(P), source_frame()]

    _, stats = run_clip(sources, [replacement_frame()] * 3)

    one = 1 / (H * W)
    assert isinstance(stats, TemporalRecoveryStreamStats)
    assert stats.soft_edge_ratio == 0.0
    assert stats.recovery_region_ratio == pytest.approx(one / 3)
    assert stats.own_background_ratio == 0.0
    assert stats.temporal_recovered_ratio == pytest.approx(one / 3)
    assert stats.temporal_unrecovered_ratio == 0.0
    assert stats.o1_fallback_ratio == 0.0
    assert (stats.median_donor_distance, stats.p90_donor_distance) == (1.0, 1.0)
    assert stats.mean_observations_per_recovered_pixel == 2.0
    assert (stats.donor_fits, stats.zero_offset_fallbacks) == (2, 0)
    assert stats.peak_cached_frames == 3


# -- the bounded cache ----------------------------------------------------


@pytest.mark.parametrize(
    ("frames", "radius", "expected"), [(60, 24, 49), (10, 2, 5), (3, 24, 3), (4, 0, 1)]
)
def test_the_source_window_never_exceeds_two_radius_plus_one(
    frames: int, radius: int, expected: int
) -> None:
    sources = [source_frame()] * frames
    _, stats = run_clip(sources, [replacement_frame()] * frames, radius=radius)
    assert stats.peak_cached_frames == expected


def test_a_radius_of_zero_has_no_donors() -> None:
    frames, stats = run_clip(
        [source_frame(), source_frame(P), source_frame()], [replacement_frame()] * 3, radius=0
    )
    assert pixel(frames, 1) == O1_BG
    assert stats.o1_fallback_ratio == 1.0


# -- streaming: errors ----------------------------------------------------


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


def test_the_source_streams_are_read_ahead_by_the_radius() -> None:
    """A source truncated at frame 2 fails while frame 0 is being produced."""
    streams = clip_streams([source_frame()] * 3, [replacement_frame()] * 3)
    streams["source"] = io.BytesIO(streams["source"].getvalue()[: -H * W])
    out = io.BytesIO()

    with pytest.raises(CompositeError, match="source ended during frame 2"):
        run_streams(streams, 3, out)

    assert out.getvalue() == b""  # nothing was written: frame 0 needed frame 2 first
    assert streams["replacement"].tell() == 0


@pytest.mark.parametrize("long", STREAMS)
def test_extra_data_on_any_of_the_four_streams_names_it(long: str) -> None:
    streams = clip_streams([source_frame()] * 2, [replacement_frame()] * 2)
    streams[long] = io.BytesIO(streams[long].getvalue() + b"\0")

    with pytest.raises(CompositeError, match=f"{long} has more than the expected 2"):
        run_streams(streams, 2)


def test_zero_frames_writes_nothing() -> None:
    out = io.BytesIO()
    stats = run_streams({name: io.BytesIO() for name in STREAMS}, 0, out)
    assert out.getvalue() == b""
    assert stats.recovery_region_ratio == 0.0 and stats.peak_cached_frames == 0
    assert math.isnan(stats.median_donor_distance)


# -- orchestration, against the fake ffmpeg -----------------------------


def scene() -> tuple[list[Frame], list[Frame]]:
    sources = [source_frame(), source_frame(P), source_frame()]
    replacements = [replacement_frame(), replacement_frame((2, 3), alpha=255), replacement_frame()]
    return sources, replacements


def scene_outputs(sources: list[Frame], replacements: list[Frame]) -> dict[str, bytes]:
    streams = clip_streams(sources, replacements)
    return {
        "src.mp4": streams["source"].getvalue(),
        "rep.mp4": streams["replacement"].getvalue(),
        "src.webm": streams["source_matte"].getvalue(),
        "rep.webm": streams["replacement_matte"].getvalue(),
    }


def run_recovery(tmp_path: Path, out: Path, **kwargs: Any) -> TemporalRecoveryCompositeReport:
    return composite_video_temporal_recovery(
        tmp_path / "src.mp4",
        tmp_path / "rep.mp4",
        tmp_path / "src.webm",
        tmp_path / "rep.webm",
        out,
        **kwargs,
    )


def test_composite_video_temporal_recovery_matches_the_streaming_function(
    tmp_path: Path, monkeypatch: Any
) -> None:
    sources, replacements = scene()
    ffmpeg = FakeFfmpeg(scene_outputs(sources, replacements))
    wire(monkeypatch, ffmpeg, fake_probe(3))
    out = tmp_path / "nested" / "v5.mp4"

    report = run_recovery(tmp_path, out)

    expected, expected_stats = run_clip(
        sources, replacements, dilation_radius=4, offset_stride=8, offset_min_samples=256
    )
    assert ffmpeg.encoded == expected.tobytes()
    assert report.stats == expected_stats
    assert report.output_path == out
    assert (report.frames, report.width, report.height) == (3, W, H)
    assert report.frame_rate == Fraction(24, 1)
    assert (report.removal_threshold, report.dilation_radius) == (64, 4)
    assert (report.background_threshold, report.foreground_threshold) == (32, 128)
    assert (report.radius, report.max_observations) == (24, 5)
    assert report.stats.peak_cached_frames == 3
    assert out.parent.is_dir()
    assert not any(p.killed for p in ffmpeg.processes)


def test_recovery_decoders_force_libvpx_for_both_mattes_only(
    tmp_path: Path, monkeypatch: Any
) -> None:
    ffmpeg = FakeFfmpeg(default_outputs(2))
    wire(monkeypatch, ffmpeg, fake_probe(2))
    out = tmp_path / "v5.mp4"

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

    report = run_recovery(tmp_path, tmp_path / "v5.mp4", radius=0, dilation_radius=0)

    assert (report.radius, report.dilation_radius) == (0, 0)
    assert report.stats.peak_cached_frames == 1
    assert report.stats.o1_fallback_ratio == 1.0  # no donors at radius 0
    written = np.frombuffer(ffmpeg.encoded, np.uint8).reshape(3, H, W, 3)
    assert pixel(written, 1) == O1_BG


def test_invalid_parameters_are_refused_before_anything_is_probed(
    tmp_path: Path, monkeypatch: Any
) -> None:
    ffmpeg = FakeFfmpeg(default_outputs(1))

    def never(path: Path, decoder: str | None = None) -> VideoInfo:
        raise AssertionError(f"probed {path}")

    wire(monkeypatch, ffmpeg, never)

    for bad in ({"radius": -1}, {"max_observations": 0}, {"foreground_threshold": 0}):
        with pytest.raises(CompositeError):
            run_recovery(tmp_path, tmp_path / "v5.mp4", **bad)
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
        run_recovery(tmp_path, tmp_path / "v5.mp4")
    assert ffmpeg.commands == []


def test_a_missing_input_is_rejected_before_ffprobe_runs(tmp_path: Path) -> None:
    with pytest.raises(CompositeError, match="input not found"):
        run_recovery(tmp_path, tmp_path / "v5.mp4")


@pytest.mark.parametrize("name", STREAMS)
def test_a_failing_decoder_is_reported_with_its_name_and_stderr(
    name: str, tmp_path: Path, monkeypatch: Any
) -> None:
    ffmpeg = FakeFfmpeg(default_outputs(2), failing=FILES[name], stderr="could not decode")
    wire(monkeypatch, ffmpeg, fake_probe(2))

    with pytest.raises(CompositeError, match=f"ffmpeg {name} exited 1: could not decode"):
        run_recovery(tmp_path, tmp_path / "v5.mp4")


def test_a_failing_encoder_is_reported(tmp_path: Path, monkeypatch: Any) -> None:
    ffmpeg = FakeFfmpeg(default_outputs(2), failing="encode", stderr="broken pipe")
    wire(monkeypatch, ffmpeg, fake_probe(2))

    with pytest.raises(CompositeError, match="ffmpeg encode exited 1: broken pipe"):
        run_recovery(tmp_path, tmp_path / "v5.mp4")


def test_a_decoder_that_dies_mid_stream_aborts_and_kills_the_rest(
    tmp_path: Path, monkeypatch: Any
) -> None:
    outputs = default_outputs(3)
    outputs["src.mp4"] = frames_bytes([1])  # one frame, then silence
    ffmpeg = FakeFfmpeg(outputs, failing="src.mp4", stderr="decode error")
    wire(monkeypatch, ffmpeg, fake_probe(3))

    with pytest.raises(CompositeError, match="source ended during frame 1"):
        run_recovery(tmp_path, tmp_path / "v5.mp4")
    assert all(p.killed for p in ffmpeg.processes)


def test_recovery_api_is_exported_from_the_package() -> None:
    package: Any = video_character_skill
    assert package.composite_video_temporal_recovery is composite_video_temporal_recovery
    assert package.TemporalRecoveryCompositeReport is TemporalRecoveryCompositeReport
    assert video_character_skill.TEMPORAL_RECOVERY_RADIUS == 24
    assert video_character_skill.MAX_TEMPORAL_OBSERVATIONS == 5
    assert video_character_skill.SOURCE_BACKGROUND_THRESHOLD == 32
    assert video_character_skill.REPLACEMENT_FOREGROUND_THRESHOLD == 128
    assert "composite_video_temporal_recovery" in video_character_skill.__all__


# -- v4 is not affected by v5 -------------------------------------------


def test_the_v4_composite_still_uses_o1_where_v5_recovers_real_background(
    tmp_path: Path, monkeypatch: Any
) -> None:
    sources, replacements = scene()
    outputs = scene_outputs(sources, replacements)
    ffmpeg_v4, ffmpeg_v5 = FakeFfmpeg(dict(outputs)), FakeFfmpeg(dict(outputs))

    wire(monkeypatch, ffmpeg_v4, fake_probe(3))
    composite_video_source_removal(
        tmp_path / "src.mp4", tmp_path / "rep.mp4", tmp_path / "src.webm", tmp_path / "rep.webm",
        tmp_path / "v4.mp4",
    )
    wire(monkeypatch, ffmpeg_v5, fake_probe(3))
    run_recovery(tmp_path, tmp_path / "v5.mp4")

    v4 = np.frombuffer(ffmpeg_v4.encoded, np.uint8).reshape(3, H, W, 3)
    v5 = np.frombuffer(ffmpeg_v5.encoded, np.uint8).reshape(3, H, W, 3)
    assert pixel(v4, 1) == O1_BG
    assert pixel(v5, 1) == BG
    assert pixel(v4, 1, (2, 3)) == pixel(v5, 1, (2, 3)) == O1_PERSON
    assert matte_bytes([0]) != b""  # harness sanity: the shared helpers are importable
