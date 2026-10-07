import math

import numpy as np
import pytest

from region.geometry import RegionGeometry, ecef_to_cell
from region.grid import RegionHexGrid


def _grid(region_id: int = 20, n_cols: int = 8) -> RegionHexGrid:
    return RegionHexGrid(RegionGeometry(region_id, np.pi), n_cols)


def test_hexagons_cover_the_whole_cell():
    """Every point of the cell square falls in some hexagon — no uncovered slivers."""
    grid = _grid()
    assert len(grid) > 0
    rng = np.random.default_rng(0)
    pts = rng.random((4000, 2))
    d = pts[:, None, :] - grid.centers[None, :, :]
    x, y = np.abs(d[..., 0]), np.abs(d[..., 1])
    sqrt3, R = math.sqrt(3.0), grid.radius
    inside = (
        (y <= sqrt3 * 0.5 * R + 1e-12)
        & (sqrt3 * x + y <= sqrt3 * R + 1e-12)
        & (sqrt3 * x - y <= sqrt3 * R + 1e-12)
    ).any(axis=1)
    assert inside.all()


def test_overhang_beyond_the_cell_stays_bounded():
    """Coverage costs at most one hexagon of spill on each side."""
    grid = _grid()
    assert grid.centers.min() >= -grid.radius - 1e-9
    assert grid.centers.max() <= 1.0 + grid.radius + 1e-9


def test_hex_count_scales_with_columns():
    assert len(_grid(n_cols=16)) > len(_grid(n_cols=8))


def test_invalid_arguments():
    with pytest.raises(ValueError):
        _grid(n_cols=0)
    with pytest.raises(ValueError):
        _grid().texture_local(0)


def test_texture_rows_run_north_to_south():
    """Row 0 must be the top (largest t) so textures blit without a flip."""
    local = _grid().texture_local(8)
    assert np.all(local[:, 0, :, 1] > local[:, -1, :, 1])
    assert np.all(local[:, :, 0, 0] < local[:, :, -1, 0])


def test_texture_positions_on_unit_sphere():
    grid = _grid()
    pos = grid.texture_positions(8)
    assert pos.shape == (len(grid), 8, 8, 3)
    assert np.allclose(np.linalg.norm(pos, axis=-1), 1.0, atol=1e-9)


def test_centers_map_into_their_own_region():
    """Interior hexagon centres lie in their region (edge centres may resolve either way)."""
    for region_id in (0, 13, 20, 53):
        grid = RegionHexGrid(RegionGeometry(region_id, np.pi), 6)
        interior = np.all((grid.centers > 1e-6) & (grid.centers < 1.0 - 1e-6), axis=1)
        assert interior.sum() > 0
        cells = ecef_to_cell(grid.centers_ecef()[interior], np.pi)
        assert np.all(cells == region_id)


def test_in_hexagon_mask_matches_area_ratio():
    """Hexagon / bounding-square area = 3*sqrt(3)/8 ~= 0.6495."""
    mask = _grid(n_cols=8).texture_in_hexagon(64)
    assert mask.mean() == pytest.approx(3.0 * math.sqrt(3.0) / 8.0, abs=0.01)


def test_in_hexagon_mask_is_centred():
    """The centre texel is always inside; the extreme corners never are."""
    mask = _grid().texture_in_hexagon(9)
    assert np.all(mask[:, 4, 4])
    assert not np.any(mask[:, 0, 0])
    assert not np.any(mask[:, -1, -1])


def test_texel_spacing_shrinks_with_resolution():
    grid = _grid()
    coarse = grid.texel_spacing_rad(16)
    fine = grid.texel_spacing_rad(32)
    assert fine == pytest.approx(coarse * 0.5, rel=0.05)


def test_texel_at_finds_the_in_hexagon_texel_at_its_own_centre():
    grid = _grid(n_cols=5)
    tex = 12
    local = grid.texture_local(tex)
    in_hex = grid.texture_in_hexagon(tex)
    for i, row, col in zip(*np.nonzero(in_hex)):
        s, t = local[i, row, col]
        if 0.0 <= s <= 1.0 and 0.0 <= t <= 1.0:
            assert grid.texel_at(s, t, tex) == (i, row, col)


def test_texel_at_is_none_outside_the_cell():
    grid = _grid(n_cols=5)
    assert grid.texel_at(-0.01, 0.5, 8) is None
    assert grid.texel_at(0.5, 1.01, 8) is None


def test_texel_neighbours_are_symmetric_and_close():
    """Drawn texels link to drawn texels about one texel away, across hexagon edges too."""
    grid, t = _grid(n_cols=4), 8
    nb = grid.texel_neighbours(t)
    drawn = nb[:, 0] != -2
    assert np.array_equal(drawn, grid.texture_in_hexagon(t).reshape(-1))
    local = grid.texture_local(t).reshape(-1, 2)
    step = 2.0 * grid.radius / t
    src = np.repeat(np.arange(len(nb)), 4)[nb.reshape(-1) >= 0]
    dst = nb.reshape(-1)[nb.reshape(-1) >= 0]
    assert drawn[dst].all()
    # Hexagon textures don't share a lattice: across an edge a link may reach ~2 steps.
    assert np.linalg.norm(local[src] - local[dst], axis=1).max() < 2.5 * step
    # …and links are only near-symmetric.
    pairs = set(zip(src.tolist(), dst.tolist()))
    assert np.mean([(b, a) in pairs for a, b in pairs]) > 0.9
    # Some links cross hexagon edges; the map's edge is marked -1.
    hexes = np.repeat(np.arange(len(grid)), t * t)
    assert (hexes[src] != hexes[dst]).any()
    assert (nb[drawn] == -1).any()


def test_texel_neighbours_respect_the_domain():
    grid, t = _grid(n_cols=4), 8
    domain = grid.texture_local(t)[..., 0] < 0.5
    nb = grid.texel_neighbours(t, domain)
    drawn = nb[:, 0] != -2
    assert not drawn[~domain.reshape(-1)].any()
    # Texels on the cut line see the edge.
    s = grid.texture_local(t)[..., 0].reshape(-1)
    near_cut = drawn & (s > 0.5 - 2.0 * grid.radius / t)
    assert (nb[near_cut] == -1).any(axis=1).all()
