"""Regional rivers: the world's rivers re-routed over the refined relief, plus
sub-threshold tributaries."""
import numpy as np
import pytest

from region.rivers import RegionRiverParams, build_region_rivers
from region.upsample import RegionUpsampleParams, build_region_fields

REGION_ID = 40   # coast, a lake and a peak (see conftest)


@pytest.fixture(scope="module")
def fields(synthetic_globe):
    return build_region_fields(synthetic_globe, REGION_ID, RegionUpsampleParams(n_cols=6, tex_size=16))


def _local_line(fields, a: np.ndarray, b: np.ndarray, k: int = 40) -> np.ndarray:
    """ECEF polyline of ``k`` points along the straight local segment a → b."""
    st = a[None, :] + np.linspace(0.0, 1.0, k)[:, None] * (b - a)[None, :]
    return fields.geometry.local_to_ecef(st[:, 0], st[:, 1]).astype(np.float32)


@pytest.fixture(scope="module")
def world_rivers(fields, synthetic_globe):
    """A stem from high ground to the nearest refined sea, and a tributary
    joining it half way; installed as the world's "rivers" feature."""
    local = fields.grid.texture_local(fields.tex_size).reshape(-1, 2)
    elev = fields.textures["elevation"].reshape(-1)
    ocean = fields.textures["ocean_mask"].reshape(-1) > 0.5
    lake = fields.textures["lake_mask"].reshape(-1) > 0.5
    in_cell = np.all((local > 0.1) & (local < 0.9), axis=1)
    land = in_cell & ~ocean & ~lake
    source = local[np.flatnonzero(land)[np.argmax(elev[land])]]
    sea = np.flatnonzero(in_cell & ocean)
    mouth = local[sea[np.argmin(np.linalg.norm(local[sea] - source, axis=1))]]
    mid = 0.5 * (source + mouth)
    side = np.array([-(mouth - source)[1], (mouth - source)[0]])
    side *= 0.25 / np.linalg.norm(side)

    stem = _local_line(fields, source, mouth)
    trib = _local_line(fields, mid + side, mid, 20)
    rivers = [
        (stem, np.linspace(600.0, 2000.0, len(stem) - 1).astype(np.float32), np.zeros(len(stem) - 1, bool)),
        (trib, np.linspace(500.0, 700.0, len(trib) - 1).astype(np.float32), np.zeros(len(trib) - 1, bool)),
    ]
    synthetic_globe.add_feature("rivers", rivers)
    yield rivers
    synthetic_globe.remove_feature("rivers")


@pytest.fixture(scope="module")
def region_rivers(fields, synthetic_globe, world_rivers):
    return build_region_rivers(fields, synthetic_globe)


def _chord_to_polyline(points: np.ndarray, line: np.ndarray) -> np.ndarray:
    """Chord distance from each point to the nearest vertex of a dense polyline."""
    return np.linalg.norm(points[:, None, :] - line[None, :, :], axis=-1).min(axis=1)


def _densify(line: np.ndarray, k: int = 20) -> np.ndarray:
    f = np.linspace(0.0, 1.0, k, endpoint=False)[:, None, None]
    return (line[:-1][None] + f * (line[1:] - line[:-1])[None]).transpose(1, 0, 2).reshape(-1, 3)


def _global_texel_chord(globe) -> float:
    gw = globe.mesh.face_textures["elevation"].shape[2]
    return float(np.sqrt(np.mean(globe.mesh.face_areas))) / (gw - 1)


def test_every_world_river_is_rerouted(region_rivers):
    assert len(region_rivers.rivers) == 2


def test_rerouted_rivers_stay_in_their_corridor(region_rivers, world_rivers, synthetic_globe):
    """A re-routed river follows the refined valleys but never leaves the band
    the global grid leaves open around its world path."""
    corridor = RegionRiverParams().corridor_global_texels * _global_texel_chord(synthetic_globe)
    world = [_densify(np.asarray(r[0], dtype=np.float64)) for r in world_rivers]
    for path in region_rivers.rivers:
        pts = np.asarray(path[0], dtype=np.float64)
        dists = min((_chord_to_polyline(pts, w) for w in world), key=np.median)
        assert np.median(dists) < corridor
        # Only the mouth may stray further: it is allowed to reach a stem up to
        # two corridors off the world confluence.
        assert dists.max() < 2.5 * corridor


def test_rerouted_rivers_actually_move(region_rivers, world_rivers):
    """Refinement follows the refined relief rather than copying the world line."""
    stem = max(region_rivers.rivers, key=lambda p: p[1][-1])
    world = _densify(np.asarray(world_rivers[0][0], dtype=np.float64))
    assert _chord_to_polyline(np.asarray(stem[0], dtype=np.float64), world).max() > 1e-4


def test_the_world_tributary_still_meets_its_stem(region_rivers, fields):
    stem, trib = sorted(region_rivers.rivers, key=lambda p: -p[1][-1])
    end = np.asarray(trib[0][-1:], dtype=np.float64)
    spacing = fields.grid.texel_spacing_rad(fields.tex_size)
    assert _chord_to_polyline(end, _densify(np.asarray(stem[0], dtype=np.float64)))[0] < 3.0 * spacing


def test_the_stem_reaches_the_refined_sea(region_rivers, fields):
    stem = max(region_rivers.rivers, key=lambda p: p[1][-1])
    mouth = np.asarray(stem[0][-1], dtype=np.float64)
    ocean_texels = fields.positions[fields.textures["ocean_mask"][..., 0] > 0.5]
    spacing = fields.grid.texel_spacing_rad(fields.tex_size)
    assert np.linalg.norm(ocean_texels - mouth, axis=1).min() < 3.0 * spacing


def test_rerouted_flow_comes_from_the_world_river(region_rivers, world_rivers):
    stem = max(region_rivers.rivers, key=lambda p: p[1][-1])
    flows = stem[1]
    assert np.all(np.diff(flows) >= 0.0)
    assert flows.min() >= world_rivers[0][1].min() - 1e-3
    assert flows.max() <= world_rivers[0][1].max() + 1e-3


def test_tributaries_are_added_below_the_world_threshold(region_rivers):
    params = RegionRiverParams()
    assert len(region_rivers.tributaries) > 0
    for positions, flows, in_water in region_rivers.tributaries:
        assert len(positions) == len(flows) + 1 == len(in_water) + 1
        assert len(positions) >= params.tributary_min_nodes
        assert flows[0] >= params.tributary_min_flow
        assert np.all(np.diff(flows) >= 0.0)


def test_lower_threshold_gives_more_tributaries(fields, synthetic_globe, world_rivers):
    few = build_region_rivers(fields, synthetic_globe, RegionRiverParams(tributary_min_flow=80.0))
    many = build_region_rivers(fields, synthetic_globe, RegionRiverParams(tributary_min_flow=10.0))
    assert len(many.tributaries) > len(few.tributaries)


def test_generation_is_deterministic(fields, synthetic_globe, region_rivers):
    again = build_region_rivers(fields, synthetic_globe)
    assert len(again.paths) == len(region_rivers.paths)
    for a, b in zip(again.paths, region_rivers.paths):
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])


def test_tributaries_only_without_world_rivers(fields, synthetic_globe, world_rivers):
    synthetic_globe.remove_feature("rivers")
    try:
        result = build_region_rivers(fields, synthetic_globe)
    finally:
        synthetic_globe.add_feature("rivers", world_rivers)
    assert result.rivers == [] and len(result.tributaries) > 0
