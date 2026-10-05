"""Refining the world's cities onto a region's coast and rivers."""
import numpy as np
import pytest

from realmgen.placement.types import SITE_KIND_LAKE_COAST, SITE_KIND_OCEAN_COAST
from realmgen.test.conftest import _settlement
from region.cities import refine_region_cities
from region.geometry import ecef_to_cell
from region.rivers import build_region_rivers
from region.roads import move_points
from region.towns import global_texel_rad, tier_sites
from region.upsample import RegionUpsampleParams, build_region_fields

REGION_ID = 40


@pytest.fixture(scope="module")
def fields(synthetic_globe):
    return build_region_fields(synthetic_globe, REGION_ID, RegionUpsampleParams(n_cols=6, tex_size=16))


@pytest.fixture(scope="module")
def rivers(fields, synthetic_globe):
    return build_region_rivers(fields, synthetic_globe)


@pytest.fixture(scope="module")
def sites(fields, rivers, synthetic_globe):
    return tier_sites(fields, rivers, synthetic_globe, None, 1.5)


@pytest.fixture
def world(fields, sites, synthetic_globe):
    """A harbour city a little inland of each refined-coast face, plus one in another region."""
    reach = global_texel_rad(synthetic_globe)
    cities, coast_of = [], []
    for face in np.unique(sites.face[sites.coast]):
        in_face = np.flatnonzero(sites.face == face)
        coast = in_face[sites.coast[in_face]]
        inland = in_face[~sites.coast[in_face]]
        if not coast.size or not inland.size:
            continue
        d = np.linalg.norm(sites.positions[inland][:, None] - sites.positions[coast][None], axis=2).min(axis=1)
        ok = inland[(d > 0.3 * reach) & (d < 0.8 * reach)]
        if not ok.size or ecef_to_cell(sites.positions[ok[:1]], synthetic_globe.antimeridian)[0] != REGION_ID:
            continue
        i = int(ok[0])
        nearest = coast[np.argmin(np.linalg.norm(sites.positions[coast] - sites.positions[i], axis=1))]
        kind = SITE_KIND_OCEAN_COAST if sites.coast_ocean[nearest] else SITE_KIND_LAKE_COAST
        city = _settlement(face_id=int(face), pop=10_000.0 * (1 + len(cities)), position=sites.positions[i].copy())
        city.site_kind = kind
        cities.append(city)
        coast_of.append(kind)
    far = -sites.positions[0]
    assert ecef_to_cell(far[None], synthetic_globe.antimeridian)[0] != REGION_ID
    cities.append(_settlement(face_id=0, pop=5_000.0, position=far))
    synthetic_globe.add_feature("settlements", cities)
    yield cities, np.array(coast_of)
    synthetic_globe.remove_feature("settlements")


def test_harbour_cities_land_on_the_refined_coast(fields, rivers, sites, synthetic_globe, world):
    cities, kinds = world
    assert len(kinds) >= 2
    refine_region_cities(fields, rivers, synthetic_globe)
    for city, kind in zip(cities, kinds):
        at = np.flatnonzero(np.all(sites.positions == city.position, axis=1))
        assert at.size == 1
        coast = sites.coast_ocean if kind == SITE_KIND_OCEAN_COAST else sites.coast_lake
        assert coast[at[0]]


def test_only_this_regions_cities_move_once_near_their_site_in_their_face(
        fields, rivers, sites, synthetic_globe, world):
    cities, kinds = world
    assert refine_region_cities(fields, rivers, synthetic_globe) == len(kinds)
    reach = global_texel_rad(synthetic_globe)
    centroids = np.asarray(synthetic_globe.mesh.face_centroids, dtype=np.float64)
    for city in cities[:-1]:
        assert city.refined and not city.local_refined
        np.testing.assert_array_equal(city.region_position, city.position)
        assert np.linalg.norm(city.position - city.site_position) <= reach + 1e-9
        assert int(np.argmax(centroids @ city.position)) == city.face_id
    far = cities[-1]
    assert not far.refined
    np.testing.assert_array_equal(far.position, far.site_position)
    after = [c.position.copy() for c in cities]
    assert refine_region_cities(fields, rivers, synthetic_globe) == 0
    for city, p in zip(cities, after):
        np.testing.assert_array_equal(city.position, p)


def test_reset_puts_cities_back_and_refinement_is_deterministic(fields, rivers, synthetic_globe, world):
    cities, _ = world
    refine_region_cities(fields, rivers, synthetic_globe)
    first = [c.position.copy() for c in cities]
    for city in cities:
        city.reset_refinement()
        np.testing.assert_array_equal(city.position, city.site_position)
        assert not city.refined
    refine_region_cities(fields, rivers, synthetic_globe)
    for city, p in zip(cities, first):
        np.testing.assert_array_equal(city.position, p)


def test_an_inland_city_does_not_take_up_a_harbour(fields, rivers, sites, synthetic_globe, world):
    cities, _ = world
    for city in cities:
        city.site_kind = 0
    refine_region_cities(fields, rivers, synthetic_globe)
    on_coast = [bool(sites.coast[np.all(sites.positions == c.position, axis=1)].any()) for c in cities[:-1]]
    # Fertility and the pull home decide; a shore texel only by chance.
    assert np.mean(on_coast) < 0.5


def test_route_ends_follow_their_city():
    old = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    new = np.array([[0.9, 0.1, 0.0], [0.1, 0.9, 0.0]])
    pts = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0 + 1e-8]])
    out = move_points(pts, (old, new))
    np.testing.assert_array_equal(out, [new[0], pts[1], new[1]])
    np.testing.assert_array_equal(move_points(pts, None), pts)
