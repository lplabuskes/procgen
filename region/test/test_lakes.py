"""Small lakes: basins in a tier's refined relief, too small for its parent."""
import numpy as np
import pytest

from region.corridor import raster_frame, sample_raster
from region.grid import RegionHexGrid
from region.geometry import RegionGeometry
from region.lakes import basin_lake_levels, budyko_runoff_ratio, drain_nodes, glacial_breach_scale, smooth_land
from region.upsample import (
    SMALL_LAKE_KEY, RegionUpsampleParams, build_region_fields, conservation_report, reshape_to_target,
)

REGION_ID = 51   # mostly land, and thaws (conftest's lake region 40 is ice cap: no small lakes)
N = 21


def _bowl(depth: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """An ``N × N`` plane sloping up to the east with a round pit, and its west-edge outlet."""
    rows, cols = np.mgrid[0:N, 0:N]
    elevation = 0.001 * cols - depth * np.exp(-((rows - 10) ** 2 + (cols - 10) ** 2) / 8.0)
    outlet = np.zeros((N, N), dtype=bool)
    outlet[:, 0] = True
    return elevation.ravel(), outlet.ravel()


def test_a_pit_fills_to_its_spill_level():
    elevation, outlet = _bowl()
    level = basin_lake_levels(elevation, outlet, N)
    lake = np.isfinite(level)
    assert lake.reshape(N, N)[10, 10]
    assert np.all(level[lake] > elevation[lake])
    # One flat surface, at the lowest point of its rim.
    assert np.ptp(level[lake]) == 0.0
    assert not lake[outlet].any()


def test_shallow_or_steep_basins_are_breached():
    elevation, outlet = _bowl(depth=0.05)
    level = basin_lake_levels(elevation, outlet, N)
    lake = np.isfinite(level)
    deepest = float(np.max(level[lake] - elevation[lake]))
    assert not np.isfinite(basin_lake_levels(elevation, outlet, N, min_depth=deepest * 1.01)).any()
    assert not np.isfinite(basin_lake_levels(elevation, outlet, N, breach=np.full(N * N, deepest * 1.01))).any()
    assert np.isfinite(basin_lake_levels(elevation, outlet, N, breach=np.full(N * N, deepest * 0.5))).any()


def test_too_small_basins_are_dropped():
    elevation, outlet = _bowl()
    size = int(np.isfinite(basin_lake_levels(elevation, outlet, N)).sum())
    assert not np.isfinite(basin_lake_levels(elevation, outlet, N, min_nodes=size + 1)).any()


def test_too_big_basins_stay_dry():
    elevation, outlet = _bowl()
    size = int(np.isfinite(basin_lake_levels(elevation, outlet, N)).sum())
    assert not np.isfinite(basin_lake_levels(elevation, outlet, N, max_nodes=size - 1)).any()
    assert np.isfinite(basin_lake_levels(elevation, outlet, N, max_nodes=size)).sum() == size


def test_dry_basins_shrink_and_vetoed_ones_vanish():
    elevation, outlet = _bowl()
    full = np.isfinite(basin_lake_levels(elevation, outlet, N))
    wet = basin_lake_levels(elevation, outlet, N, runoff=np.full(N * N, 1.0), evaporation=np.full(N * N, 1.0))
    assert np.array_equal(np.isfinite(wet), full)
    # Little inflow: a smaller lake, lower, still inside the full one.
    dry = basin_lake_levels(elevation, outlet, N, runoff=np.full(N * N, 0.01), evaporation=np.full(N * N, 1.0))
    shrunk = np.isfinite(dry)
    assert 0 < shrunk.sum() < full.sum()
    assert not (shrunk & ~full).any()
    assert dry[shrunk].max() < basin_lake_levels(elevation, outlet, N)[full].max()
    assert not np.isfinite(basin_lake_levels(elevation, outlet, N, allowed=np.zeros(N * N, dtype=bool))).any()


def test_smoothing_averages_land_only():
    rng = np.random.default_rng(0)
    elevation = rng.normal(0.0, 0.01, (N, N))
    water = np.zeros((N, N), dtype=bool)
    water[:, :3] = True
    elevation[water] = -1.0
    out = smooth_land(elevation, water, 2.0)
    assert np.array_equal(out[water], elevation[water])
    # The sea floor doesn't drag the shore down; speckle is damped.
    assert out[~water].min() > -0.05
    assert out[~water].std() < 0.5 * elevation[~water].std()
    assert np.array_equal(smooth_land(elevation, water, 0.0), elevation)


def test_a_lake_covering_its_anchor_shallows_its_bed():
    """Bed texels above the surface are pushed under; with nothing dry to raise,
    the deeper ones rise towards the surface instead, keeping the mean."""
    elevation = np.array([0.003, 0.002, -0.004, -0.008])     # one anchor; surface 0, mean −0.00175
    anchors, weights = np.zeros(4, dtype=np.int64), np.ones(4)
    target = np.full(4, -0.002)
    one = reshape_to_target(elevation, target, anchors, weights, 1, level=0.0, min_offset=0.001)
    assert np.array_equal(one, elevation)                    # one-sided: nothing to absorb it
    two = reshape_to_target(elevation, target, anchors, weights, 1, level=0.0, min_offset=0.001, two_sided=True)
    assert np.all(two < 0.0)
    assert two.mean() == pytest.approx(elevation.mean())


def test_budyko_runoff_falls_with_aridity():
    pet = np.array([0.0, 500.0, 1000.0, 3000.0, 10000.0])
    ratio = budyko_runoff_ratio(pet, np.full(5, 1000.0))
    assert ratio[0] == pytest.approx(1.0)
    assert np.all(np.diff(ratio) < 0) and ratio[-1] < 0.01
    assert budyko_runoff_ratio(np.array([500.0]), np.array([0.0]))[0] == 0.0


def test_glacial_scale_ramps_from_the_factor_to_one():
    scale = glacial_breach_scale(np.array([-20.0, 0.0, 5.0, 10.0, 30.0]), 0.3)
    assert scale == pytest.approx([0.3, 0.3, 0.65, 1.0, 1.0])


def test_drain_nodes_follow_their_segment():
    grid = RegionHexGrid(RegionGeometry(REGION_ID, float(np.pi)), 6)
    geometry = grid.geometry
    segment = geometry.local_to_ecef(np.array([0.2, 0.8]), np.array([0.5, 0.5]))[None]
    n, lo, step = 41, 0.0, 1.0 / 40
    hit = drain_nodes(grid, segment, lo, step, n).reshape(n, n)
    assert hit[20, 8:33].all()          # t = 0.5, s from 0.2 to 0.8
    assert hit.sum() == 25
    assert not drain_nodes(grid, np.zeros((0, 2, 3)), lo, step, n).any()


@pytest.fixture(scope="module")
def fields(synthetic_globe):
    # Every gate open: the analytic globe is smooth and its test region coarse.
    return build_region_fields(synthetic_globe, REGION_ID, RegionUpsampleParams(
        n_cols=6, tex_size=16, lake_breach_parent_texels=0.0, lake_min_depth_km=0.0, lake_min_texels=1,
        lake_min_aridity_index=0.0))


def test_the_region_finds_small_lakes_on_its_land(fields, synthetic_globe):
    tx = fields.textures
    small = tx[SMALL_LAKE_KEY][..., 0] > 0.5
    assert small.any()
    assert not (small & ~(tx["lake_mask"][..., 0] > 0.5)).any()
    assert not (small & (tx["ocean_mask"][..., 0] > 0.5)).any()
    # Water stands on them: their surface is above the ground.
    surface = tx["filled_height"][..., 0][small]
    assert np.all(surface > tx["elevation"][..., 0][small])


def test_small_lakes_keep_every_field_conserved(fields, synthetic_globe):
    for name, stats in conservation_report(fields, synthetic_globe).items():
        assert stats["max"] < (1e-3 if name == "precipitation" else 1e-5), name


def test_switching_them_off_leaves_none(synthetic_globe):
    off = build_region_fields(synthetic_globe, REGION_ID, RegionUpsampleParams(
        n_cols=6, tex_size=16, small_lakes=False))
    assert not (off.textures[SMALL_LAKE_KEY] > 0.5).any()


def test_channels_end_in_the_globes_water_but_run_through_small_lakes(fields, synthetic_globe):
    lo, step, n = raster_frame(fields, 1.0, 0.0)
    raster = sample_raster(fields, synthetic_globe, lo, step, n, np.ones(n * n, dtype=bool))
    assert raster.small_lake.any()
    assert not (raster.sinks & raster.small_lake).any()
    assert np.array_equal(raster.sinks | raster.small_lake, raster.water)


def test_no_small_lakes_on_ice_cap(synthetic_globe):
    """Region 40 never thaws (warmest season < 0 °C): even with every other gate open."""
    ice = build_region_fields(synthetic_globe, 40, RegionUpsampleParams(
        n_cols=6, tex_size=16, lake_breach_parent_texels=0.0, lake_min_depth_km=0.0, lake_min_texels=1,
        lake_min_aridity_index=0.0))
    assert ice.textures["temperature"].max(axis=-1).max() < 0.0
    assert not (ice.textures[SMALL_LAKE_KEY] > 0.5).any()
