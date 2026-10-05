"""Refining the world's towns onto a region's terrain and streams."""
import numpy as np
import pytest

from realmgen.test.conftest import _settlement
from realmgen.towns.builder import TownParams
from realmgen.towns.placement import SPACING_EXPONENT
from realmgen.towns.types import TOWN_MARKET, TOWN_RIVER, Towns
from region.geometry import ecef_to_cell
from region.rivers import build_region_rivers
from region.towns import RegionSettlementParams, _river_points, refine_region_towns
from region.upsample import RegionUpsampleParams, build_region_fields
from worldgen.cost.surfaces import EARTH_RADIUS_KM

REGION_ID = 40


@pytest.fixture(scope="module")
def fields(synthetic_globe):
    return build_region_fields(synthetic_globe, REGION_ID, RegionUpsampleParams(n_cols=6, tex_size=16))


@pytest.fixture(scope="module")
def rivers(fields, synthetic_globe):
    return build_region_rivers(fields, synthetic_globe)


def _unit_centroids(globe) -> np.ndarray:
    c = np.asarray(globe.mesh.face_centroids, dtype=np.float64)
    return c / np.linalg.norm(c, axis=1, keepdims=True)


@pytest.fixture
def world(fields, synthetic_globe):
    """A fresh town at every land face centroid the region touches (half of them
    river towns), and a city on one of them; installed for one test."""
    centroids = _unit_centroids(synthetic_globe)
    faces = np.unique(fields.faces[fields.in_hex])
    tex = synthetic_globe.mesh.face_textures
    faces = faces[(tex["ocean_mask"][faces, 3, 3, 0] < 0.5) & (tex["lake_mask"][faces, 3, 3, 0] < 0.5)]
    towns = Towns(
        positions=centroids[faces].copy(),
        population=np.linspace(6_000.0, 1_000.0, len(faces)),
        face_id=faces.astype(np.int32),
        parent=np.full(len(faces), -1, dtype=np.int32),
        kind=np.where(np.arange(len(faces)) % 2 == 0, TOWN_RIVER, TOWN_MARKET).astype(np.int8),
    )
    owned = ecef_to_cell(towns.site_positions, synthetic_globe.antimeridian) == REGION_ID
    # On top of a town's site, so that town has to make room.
    city_face = int(faces[np.flatnonzero(owned)[0]])
    city = _settlement(face_id=city_face, pop=50_000.0, position=centroids[city_face])
    synthetic_globe.add_feature("towns", towns)
    synthetic_globe.add_feature("settlements", [city])
    yield towns, city, owned
    synthetic_globe.remove_feature("towns")
    synthetic_globe.remove_feature("settlements")


def _reach(globe) -> float:
    gw = globe.mesh.face_textures["elevation"].shape[2]
    texel = float(np.sqrt(np.mean(globe.mesh.face_areas))) / (gw - 1)
    return RegionSettlementParams().refine_radius_global_texels * texel


def _radii(towns: Towns) -> np.ndarray:
    p = TownParams()
    return p.town_spacing_km / EARTH_RADIUS_KM * (towns.population / p.town_min) ** SPACING_EXPONENT


def test_only_this_regions_towns_are_refined(fields, rivers, synthetic_globe, world):
    towns, _, owned = world
    assert owned.any() and (~owned).any()
    assert refine_region_towns(fields, rivers, synthetic_globe) == owned.sum()
    np.testing.assert_array_equal(towns.refined, owned)
    np.testing.assert_array_equal(towns.positions[~owned], towns.site_positions[~owned])
    assert np.all(np.any(towns.positions[owned] != towns.site_positions[owned], axis=1))


def test_refined_towns_stay_near_their_site_in_their_face_on_land(fields, rivers, synthetic_globe, world):
    towns, _, owned = world
    refine_region_towns(fields, rivers, synthetic_globe)
    moved = np.linalg.norm(towns.positions - towns.site_positions, axis=1)
    assert moved.max() <= _reach(synthetic_globe) + 1e-9
    owner = np.argmax(towns.positions @ _unit_centroids(synthetic_globe).T, axis=1)
    np.testing.assert_array_equal(owner[owned], towns.face_id[owned])
    water = (fields.textures["ocean_mask"][..., 0] > 0.5) | (fields.textures["lake_mask"][..., 0] > 0.5)
    dry = fields.positions[fields.in_hex & ~water]
    d = np.linalg.norm(dry[None, :, :] - towns.positions[owned][:, None, :], axis=-1).min(axis=1)
    assert d.max() < 1e-9


def test_the_towns_themselves_are_unchanged(fields, rivers, synthetic_globe, world):
    towns, _, _ = world
    before = (towns.population.copy(), towns.face_id.copy(), towns.kind.copy(), towns.site_positions.copy())
    refine_region_towns(fields, rivers, synthetic_globe)
    for a, b in zip(before, (towns.population, towns.face_id, towns.kind, towns.site_positions)):
        np.testing.assert_array_equal(a, b)


def test_towns_keep_clear_of_each_other_and_the_city(fields, rivers, synthetic_globe, world):
    towns, city, owned = world
    refine_region_towns(fields, rivers, synthetic_globe)
    radii = _radii(towns)
    for i in np.flatnonzero(owned):
        others = np.arange(len(towns)) != i
        d = np.linalg.norm(towns.positions[others] - towns.positions[i], axis=1)
        assert np.all(d >= np.maximum(radii[others], radii[i]) - 1e-9)
    p = TownParams()
    city_radius = p.city_clearance_km / EARTH_RADIUS_KM * (city.population / 7_000.0) ** SPACING_EXPONENT
    assert np.linalg.norm(towns.positions[owned] - city.position, axis=1).min() >= city_radius - 1e-9


def test_river_towns_land_on_the_drawn_streams(fields, rivers, synthetic_globe):
    """A river town sited on a stream texel ends up on the stream the region draws."""
    streams = synthetic_globe.get_feature("streams")
    pos = np.asarray(streams.positions, dtype=np.float64)
    in_region = streams.channel & (ecef_to_cell(pos, synthetic_globe.antimeridian) == REGION_ID)
    faces = np.argmax(pos @ _unit_centroids(synthetic_globe).T, axis=1)
    # One town per face, on its first stream texel there.
    _, first = np.unique(faces[in_region], return_index=True)
    idx = np.flatnonzero(in_region)[first]
    towns = Towns(
        positions=pos[idx].copy(),
        population=np.full(idx.size, 1_000.0),
        face_id=faces[idx].astype(np.int32),
        parent=np.full(idx.size, -1, dtype=np.int32),
        kind=np.full(idx.size, TOWN_RIVER, dtype=np.int8),
    )
    synthetic_globe.add_feature("towns", towns)
    try:
        refine_region_towns(fields, rivers, synthetic_globe)
    finally:
        synthetic_globe.remove_feature("towns")
    spacing = fields.grid.texel_spacing_rad(fields.tex_size)
    stream = _river_points(rivers, 0.5 * spacing)
    assert stream is not None and idx.size >= 5
    on = np.linalg.norm(stream[None, :, :] - towns.positions[:, None, :], axis=-1).min(axis=1) < 1.5 * spacing
    at_site = np.linalg.norm(stream[None, :, :] - towns.site_positions[:, None, :], axis=-1).min(axis=1) < 1.5 * spacing
    assert on.mean() >= 0.8 and on.mean() > at_site.mean()


def test_refinement_happens_once(fields, rivers, synthetic_globe, world):
    towns, _, _ = world
    refine_region_towns(fields, rivers, synthetic_globe)
    after = towns.positions.copy()
    assert refine_region_towns(fields, rivers, synthetic_globe) == 0
    np.testing.assert_array_equal(towns.positions, after)


def test_reset_puts_towns_back_and_refinement_is_deterministic(fields, rivers, synthetic_globe, world):
    towns, _, _ = world
    refine_region_towns(fields, rivers, synthetic_globe)
    first = towns.positions.copy()
    towns.reset_refinement()
    np.testing.assert_array_equal(towns.positions, towns.site_positions)
    assert not towns.refined.any()
    refine_region_towns(fields, rivers, synthetic_globe)
    np.testing.assert_array_equal(towns.positions, first)


def test_refinement_round_trips_through_pack(fields, rivers, synthetic_globe, world):
    towns, _, _ = world
    refine_region_towns(fields, rivers, synthetic_globe)
    back = Towns.unpack(towns.pack())
    np.testing.assert_allclose(back.positions, towns.positions, atol=1e-6)
    np.testing.assert_allclose(back.site_positions, towns.site_positions, atol=1e-6)
    np.testing.assert_array_equal(back.refined, towns.refined)


def test_no_towns_nothing_to_refine(fields, rivers, synthetic_globe):
    assert not synthetic_globe.has_feature("towns")
    assert refine_region_towns(fields, rivers, synthetic_globe) == 0


def test_the_route_flag_reads_the_regional_roads():
    """Re-routed roads, sea legs and the undrawn world hops all count as routes."""
    from region.roads import RegionRoads
    from region.towns import _route_points
    a, b = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    c = np.array([0.0, 0.0, 1.0])
    roads = RegionRoads(
        region_id=0, segments=[], polylines=[np.stack([a, b])], sea=[(b, c, 1.0)], undrawn=[(c, a, 1.0)],
    )
    pts = _route_points(roads, 0.5)
    assert pts is not None
    for p in (a, b, c, (a + b) / 2):
        assert np.linalg.norm(pts - p, axis=1).min() < 0.3
