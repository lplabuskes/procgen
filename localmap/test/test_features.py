"""Local rivers and roads: the region's re-routed again over the local terrain."""
import numpy as np
import pytest

from localmap.rivers import build_local_rivers
from localmap.roads import build_local_roads
from localmap.test.conftest import local_params
from localmap.upsample import build_local_fields
from region.rivers import RegionRivers
from region.roads import RegionRoads


@pytest.fixture(scope="module")
def local(globe, region, river_hex):
    return build_local_fields(globe, region, river_hex, local_params())


@pytest.fixture(scope="module")
def rivers(local, region, region_rivers, globe):
    return build_local_rivers(local, region, region_rivers, globe)


def _dist_to(points: np.ndarray, line: np.ndarray) -> np.ndarray:
    """Chord distance from each point to the nearest vertex of a densified line."""
    line = np.asarray(line, dtype=np.float64)
    frac = np.linspace(0.0, 1.0, 20)[None, :, None]
    dense = (line[:-1, None, :] + frac * (line[1:, None, :] - line[:-1, None, :])).reshape(-1, 3)
    return np.linalg.norm(points[:, None, :] - dense[None, :, :], axis=2).min(axis=1)


def test_the_regions_rivers_are_drawn(rivers):
    assert len(rivers.paths) > 0
    for positions, flows, in_water in rivers.paths:
        assert len(positions) >= 2 and len(flows) == len(positions) - 1 == len(in_water)


def test_local_rivers_stay_near_the_regions(rivers, region_rivers, region):
    spacing = region.grid.texel_spacing_rad(region.tex_size)
    region_lines = [np.asarray(p, dtype=np.float64) for p, _, _ in region_rivers.paths]
    for positions, _, _ in rivers.paths:
        pts = np.asarray(positions, dtype=np.float64)
        d = np.min([_dist_to(pts, line) for line in region_lines], axis=0)
        # One corridor, plus the widened mouth search and the smoothing.
        assert np.percentile(d, 95) <= 3.0 * spacing


def test_flow_is_the_regions_and_grows_downstream(rivers, region_rivers):
    top = max(float(np.max(f)) for _, f, _ in region_rivers.paths)
    for _, flows, _ in rivers.paths:
        assert np.all(np.diff(flows) >= 0.0)
        assert float(flows.max()) <= top + 1e-3


def test_local_rivers_are_deterministic(local, region, region_rivers, globe, rivers):
    again = build_local_rivers(local, region, region_rivers, globe)
    assert len(again.paths) == len(rivers.paths)
    for (a, _, _), (b, _, _) in zip(rivers.paths, again.paths):
        assert np.array_equal(a, b)


def test_no_region_rivers_no_local_rivers(local, region, globe):
    empty = RegionRivers(region_id=region.region_id, paths=[])
    assert build_local_rivers(local, region, empty, globe).paths == []


@pytest.fixture(scope="module")
def road_input(local):
    """A regional road straight across the local map, over dry land, and a sea
    leg far outside it."""
    water = (local.textures["ocean_mask"][..., 0] > 0.5) | (local.textures["lake_mask"][..., 0] > 0.5)
    dry = local.positions[local.domain & ~water]
    s, t, _ = local.geometry.ecef_to_local(dry)
    a = dry[np.argmin(s)]
    b = dry[np.argmax(s)]
    frac = np.linspace(0.0, 1.0, 12)[:, None]
    line = a + frac * (b - a)
    line /= np.linalg.norm(line, axis=1, keepdims=True)
    far = -local.geometry.center_ecef()
    sea = [(far.astype(np.float32), (far + 0.01).astype(np.float32), 5.0)]
    return RegionRoads(
        region_id=local.region_id, segments=[], polylines=[line], polyline_flows=[np.full(11, 3.0)], sea=sea,
    )


def test_roads_are_rerouted_between_the_same_ends(local, region, road_input, globe, rivers):
    roads = build_local_roads(local, region, road_input, globe, rivers)
    assert roads.polylines
    line = road_input.polylines[0]
    path = roads.polylines[0]
    assert np.allclose(path[0], line[0]) and np.allclose(path[-1], line[-1])
    assert len(path) > len(line)                       # refined, not copied
    assert np.all(roads.polyline_flows[0] == 3.0)
    assert len(roads.segments) == sum(len(p) - 1 for p in roads.polylines)


def test_far_sea_legs_are_dropped(local, region, road_input, globe):
    roads = build_local_roads(local, region, road_input, globe)
    assert roads.sea == []
