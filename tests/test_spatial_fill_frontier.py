"""Equivalence and complexity tests for the frontier-based spatial wave fill.

``reference_fill_component`` below is the original full-bounding-box
synchronous implementation, kept here only as the oracle: every wave it
recomputes neighbour sums over the whole crop. The production
``spatial_fill_component`` must produce byte-identical results while doing
work proportional to the component, not to bbox area x depth.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from video_character_skill import spatial_recovery
from video_character_skill.compositor import CompositeError
from video_character_skill.spatial_recovery import (
    ComponentFill,
    label_components,
    spatial_fill_component,
    spatial_fill_components,
)

Bool = NDArray[np.bool_]
NEIGHBOURS = tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0))


# -- the oracle: the old full-bbox synchronous algorithm -----------------------


Window = tuple[tuple[slice, slice], tuple[slice, slice]]


def _shifted(dy: int, dx: int, h: int, w: int) -> Window:
    rows_to = slice(max(-dy, 0), h - max(dy, 0))
    rows_from = slice(max(dy, 0), h - max(-dy, 0))
    cols_to = slice(max(-dx, 0), w - max(dx, 0))
    cols_from = slice(max(dx, 0), w - max(-dx, 0))
    return (rows_to, cols_to), (rows_from, cols_from)


def _adjacent(mask: Bool) -> Bool:
    out = mask.copy()
    h, w = mask.shape
    for dy, dx in NEIGHBOURS:
        to, frm = _shifted(dy, dx, h, w)
        out[to] |= mask[frm]
    return out


def _neighbour_sums(
    values: NDArray[np.int64], resolved: Bool
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    h, w = resolved.shape
    sums = np.zeros((h, w, 3), dtype=np.int64)
    counts = np.zeros((h, w), dtype=np.int64)
    masked = values * resolved[:, :, None]
    for dy, dx in NEIGHBOURS:
        to, frm = _shifted(dy, dx, h, w)
        sums[to] += masked[frm]
        counts[to] += resolved[frm]
    return sums, counts


def reference_fill_component(
    rgb: NDArray[np.uint8], trusted: Bool, component: Bool
) -> ComponentFill:
    """The original implementation, verbatim in behaviour."""
    resolved = trusted & _adjacent(component)
    seeds = int(np.count_nonzero(resolved))
    values = rgb.astype(np.int64)
    filled = np.zeros(component.shape, dtype=np.bool_)
    depth = np.zeros(component.shape, dtype=np.int32)
    unresolved = component.copy()
    wave = 0
    while unresolved.any():
        wave += 1
        sums, counts = _neighbour_sums(values, resolved)
        ready = unresolved & (counts > 0)
        if not ready.any():
            break
        count = counts[ready][:, None]
        values[ready] = (2 * sums[ready] + count) // (2 * count)
        filled |= ready
        depth[ready] = wave
        resolved = resolved | ready
        unresolved &= ~ready
    out = rgb.copy()
    out[filled] = values[filled].astype(np.uint8)
    return ComponentFill(rgb=out, filled=filled, depth=depth, seeds=seeds)


def assert_equivalent(rgb: NDArray[np.uint8], trusted: Bool, target: Bool) -> None:
    """Optimized == reference, per component, for rgb, filled, depth and seeds."""
    labels, count = label_components(target)
    assert count >= 1
    for k in range(1, count + 1):
        component = labels == k
        got = spatial_fill_component(rgb, trusted, component)
        want = reference_fill_component(rgb, trusted, component)
        np.testing.assert_array_equal(got.rgb, want.rgb)
        np.testing.assert_array_equal(got.filled, want.filled)
        np.testing.assert_array_equal(got.depth, want.depth)
        assert got.seeds == want.seeds


def picture(rows: list[str], rng: np.random.Generator) -> tuple[NDArray[np.uint8], Bool, Bool]:
    """``#`` target, ``T`` trusted, ``.`` neither; random RGB everywhere."""
    h, w = len(rows), len(rows[0])
    rgb = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    trusted = np.array([[c == "T" for c in r] for r in rows], dtype=np.bool_)
    target = np.array([[c == "#" for c in r] for r in rows], dtype=np.bool_)
    return rgb, trusted, target


# -- 1-3: random masks, random rgb, disconnected shapes ---------------------------


@pytest.mark.parametrize("seed", range(12))
def test_random_masks_and_rgb_match_the_reference(seed: int) -> None:
    rng = np.random.default_rng(seed)
    h, w = int(rng.integers(3, 14)), int(rng.integers(3, 14))
    target = rng.random((h, w)) < rng.uniform(0.15, 0.6)
    trusted = (rng.random((h, w)) < rng.uniform(0.2, 0.8)) & ~target
    rgb = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    if not target.any():
        target[h // 2, w // 2] = True
    assert_equivalent(rgb, trusted, target)


def test_multiple_disconnected_shapes_match_the_reference() -> None:
    rng = np.random.default_rng(1)
    rows = [
        "TTTT.TTTT.T",
        "T##T.T##T.#",
        "T##T.T##T.#",
        "TTTT.TTTT.T",
        "...........",
        "T#T.#..###T",
        "..T.#..#T#.",
        "....#..###.",
    ]
    assert_equivalent(*picture(rows, rng))


# -- 4-6: corridors, diagonals, L / U / snake -------------------------------------


def test_long_one_pixel_corridor_matches_the_reference() -> None:
    rng = np.random.default_rng(2)
    rows = ["T" + "#" * 40 + "T", "." * 42]
    assert_equivalent(*picture(rows, rng))
    rows = ["." * 42, "T" + "#" * 40 + "."]  # seeded at one end only
    assert_equivalent(*picture(rows, rng))


def test_diagonal_paths_match_the_reference() -> None:
    rng = np.random.default_rng(3)
    n = 16
    target = np.zeros((n, n), dtype=np.bool_)
    trusted = np.zeros((n, n), dtype=np.bool_)
    for i in range(1, n):
        target[i, i] = True
    trusted[0, 0] = True
    rgb = rng.integers(0, 256, (n, n, 3), dtype=np.uint8)
    assert_equivalent(rgb, trusted, target)
    anti = np.fliplr(target).copy()
    assert_equivalent(rgb, np.fliplr(trusted).copy(), anti)


def test_l_u_and_snake_shapes_match_the_reference() -> None:
    rng = np.random.default_rng(4)
    shapes = [
        ["T#....", ".#....", ".#....", ".####.", "......"],
        ["T#..#.", ".#..#.", ".#..#.", ".####.", "......"],
        ["T######", "......#", "#######", "#......", "#######", "......#"],
        ["..T....", ".#####.", ".....#.", ".####..", ".#.....", ".#####."],
    ]
    for rows in shapes:
        assert_equivalent(*picture(rows, rng))


# -- 7-9: borders, multi-valued seeds, mixed-depth means ---------------------------


def test_border_touching_components_match_the_reference() -> None:
    rng = np.random.default_rng(5)
    rows = ["####", "T...", "...#", "..##"]
    assert_equivalent(*picture(rows, rng))
    rows = ["#T", "##"]
    assert_equivalent(*picture(rows, rng))
    full = np.ones((5, 6), dtype=np.bool_)
    trusted = np.zeros((5, 6), dtype=np.bool_)
    full[2, 3] = False
    trusted[2, 3] = True
    rgb = rng.integers(0, 256, (5, 6, 3), dtype=np.uint8)
    assert_equivalent(rgb, trusted, full)


def test_multiple_seeds_with_different_rgb_match_the_reference() -> None:
    rgb = np.zeros((3, 9, 3), dtype=np.uint8)
    rgb[1, 0] = (0, 10, 200)
    rgb[1, 8] = (255, 20, 100)
    rgb[0, 4] = (128, 128, 128)
    trusted = np.zeros((3, 9), dtype=np.bool_)
    trusted[1, 0] = trusted[1, 8] = trusted[0, 4] = True
    target = np.zeros((3, 9), dtype=np.bool_)
    target[1, 1:8] = True
    assert_equivalent(rgb, trusted, target)


def test_means_over_neighbours_resolved_at_different_depths_match() -> None:
    # A 3-wide block seeded from the left only: the middle column's pixels
    # average wave-1 pixels and (diagonally) seeds; the right column averages
    # wave-1 and wave-2 pixels. Distinct values make any ordering slip visible.
    rng = np.random.default_rng(6)
    rows = [".....", "T###.", "T###.", "T###.", "....."]
    rgb, trusted, target = picture(rows, rng)
    assert_equivalent(rgb, trusted, target)
    got = spatial_fill_component(rgb, trusted, target)
    assert got.depth[2, 1] == 1 and got.depth[2, 2] == 2 and got.depth[2, 3] == 3


# -- 10: mirrored / transposed ------------------------------------------------------


@pytest.mark.parametrize("seed", range(4))
def test_mirrored_and_transposed_inputs_match_the_reference_and_each_other(seed: int) -> None:
    rng = np.random.default_rng(100 + seed)
    h, w = 9, 13
    target = rng.random((h, w)) < 0.4
    trusted = (rng.random((h, w)) < 0.5) & ~target
    rgb = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    base = spatial_fill_components(rgb, trusted, target)
    for transform in (np.fliplr, np.flipud, lambda a: a.transpose(1, 0, *range(2, a.ndim))):
        t_rgb = np.ascontiguousarray(transform(rgb))
        t_trusted = np.ascontiguousarray(transform(trusted))
        t_target = np.ascontiguousarray(transform(target))
        assert_equivalent(t_rgb, t_trusted, t_target)
        got = spatial_fill_components(t_rgb, t_trusted, t_target)
        np.testing.assert_array_equal(got.rgb, np.ascontiguousarray(transform(base.rgb)))
        np.testing.assert_array_equal(got.depth, np.ascontiguousarray(transform(base.depth)))


# -- the pathological topology --------------------------------------------------------


def snake(h: int, w: int, thickness: int) -> tuple[Bool, Bool, int]:
    """A boustrophedon band through an ``h x w`` box, seeded only at its start.

    Returns ``(target, trusted, path_length)``; the wave depth is about the
    path length while the bbox is the whole box.
    """
    target = np.zeros((h, w), dtype=np.bool_)
    trusted = np.zeros((h, w), dtype=np.bool_)
    pitch = thickness + 2
    y = 1
    length = 0
    direction = 1
    while y + thickness <= h - 1:
        target[y : y + thickness, 1 : w - 1] = True
        length += w - 2
        nxt = y + pitch
        if nxt + thickness <= h - 1:
            col = slice(w - 1 - thickness, w - 1) if direction == 1 else slice(1, 1 + thickness)
            target[y + thickness : nxt, col] = True
            length += pitch - thickness
        y = nxt
        direction = -direction
    trusted[0, 1 : 1 + thickness] = True  # seeds above the very first pixels only
    return target, trusted, length


def test_long_thin_component_in_a_large_sparse_bbox_matches_the_reference() -> None:
    target, trusted, length = snake(60, 80, 2)
    rgb = np.random.default_rng(7).integers(0, 256, (60, 80, 3), dtype=np.uint8)
    got = spatial_fill_component(rgb, trusted, target)
    want = reference_fill_component(rgb, trusted, target)
    np.testing.assert_array_equal(got.rgb, want.rgb)
    np.testing.assert_array_equal(got.filled, want.filled)
    np.testing.assert_array_equal(got.depth, want.depth)
    assert got.seeds == want.seeds == 2
    assert got.filled.all() == target.all() and got.filled.sum() == target.sum()
    assert got.depth.max() > 20 * 2  # depth far larger than the band's width
    assert target.sum() / target.size < 0.5


def test_work_scales_with_component_pixels_not_bbox_times_depth(monkeypatch: Any) -> None:
    target, trusted, _ = snake(90, 120, 1)
    rgb = np.random.default_rng(8).integers(0, 256, (90, 120, 3), dtype=np.uint8)
    evaluations = {"candidates": 0, "waves": 0}
    original = spatial_recovery._fill_candidates

    def counting(values: Any, resolved: Any, candidates: Any) -> Any:
        evaluations["candidates"] += int(candidates.size)
        evaluations["waves"] += 1
        return original(values, resolved, candidates)

    monkeypatch.setattr(spatial_recovery, "_fill_candidates", counting)
    result = spatial_fill_component(rgb, trusted, target)

    pixels = int(target.sum())
    depth = int(result.depth.max())
    assert result.filled.sum() == pixels
    assert evaluations["candidates"] == pixels  # every pixel evaluated exactly once
    assert evaluations["waves"] == depth
    assert depth > 500  # a genuinely deep, thin component ...
    assert pixels < target.size / 3  # ... in a sparse box
    # far below even one full-bbox scan, let alone `depth` of them
    assert evaluations["candidates"] < target.size


def test_candidates_are_deduplicated_and_never_include_resolved_pixels() -> None:
    unresolved = np.ones((3, 3), dtype=np.bool_)
    unresolved[1, 1] = False
    frontier = np.array([4, 4, 4], dtype=np.int64)  # the centre, three times
    candidates = spatial_recovery._wave_candidates(frontier, unresolved)
    np.testing.assert_array_equal(candidates, [0, 1, 2, 3, 5, 6, 7, 8])
    corner = np.array([0], dtype=np.int64)
    np.testing.assert_array_equal(spatial_recovery._wave_candidates(corner, unresolved), [1, 3])


def test_fill_candidates_refuses_a_candidate_without_resolved_neighbour() -> None:
    values = np.zeros((2, 2, 3), dtype=np.int64)
    resolved = np.zeros((2, 2), dtype=np.bool_)
    with pytest.raises(CompositeError):
        spatial_recovery._fill_candidates(values, resolved, np.array([0], dtype=np.int64))
