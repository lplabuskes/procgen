"""Refining towns and cities onto a local map: which map owns them, and where they land."""
import numpy as np
import pytest

from localmap.builder import build_local_map
from localmap.params import LocalParams
from localmap.rivers import build_local_rivers
from localmap.settlements import LocalSettlementParams, local_owned, refine_local_cities, refine_local_towns
from localmap.test.conftest import local_params
from localmap.upsample import build_local_fields
from realmgen.placement.types import SITE_KIND_LAKE_COAST
from realmgen.test.conftest import _settlement
from realmgen.towns.types import TOWN_PORT, Towns
from region.geometry import ecef_to_cell
from region.towns import tier_sites


@pytest.fixture(scope="module")
def local(globe, region, lake_hex):
    return build_local_fields(globe, region, lake_hex, local_params())


@pytest.fixture(scope="module")
def rivers(local, region, region_rivers, globe):
    return build_local_rivers(local, region, region_rivers, globe)


@pytest.fixture(scope="module")
def sites(local, rivers, globe):
    return tier_sites(local, rivers, globe, None, 1.5)


def _reach(region) -> float:
    return LocalSettlementParams().refine_radius_parent_texels * region.grid.texel_spacing_rad(region.tex_size)


@pytest.fixture(scope="module")
def inland(sites, region, lake_hex, globe):
    """Dry texels of the lake hex a little way from the local lake shore, owned by it."""
    reach = _reach(region)
    coast = sites.positions[sites.coast_lake]
    d = np.linalg.norm(sites.positions[:, None] - coast[None], axis=2).min(axis=1)
    s, t, _ = region.geometry.ecef_to_local(sites.positions)
    c = region.grid.centers
    nearest = np.argmin((s[:, None] - c[None, :, 0]) ** 2 + (t[:, None] - c[None, :, 1]) ** 2, axis=1)
    ok = np.flatnonzero((d > 0.3 * reach) & (d < 0.7 * reach) & (nearest == lake_hex) & ~sites.coast)
    ok = ok[ecef_to_cell(sites.positions[ok], globe.antimeridian) == region.region_id]
    assert ok.size >= 2
    return ok[np.linspace(0, ok.size - 1, 2).astype(int)]


def _cities(sites, idx, refined=True):
    out = []
    for n, i in enumerate(idx):
        c = _settlement(face_id=int(sites.face[i]), pop=20_000.0 * (n + 1), position=sites.positions[i].copy())
        c.site_kind = SITE_KIND_LAKE_COAST
        c.refined = refined
        out.append(c)
    return out


@pytest.fixture
def with_cities(globe, sites, inland):
    cities = _cities(sites, inland)
    globe.add_feature("settlements", cities)
    yield cities
    globe.remove_feature("settlements")


def test_a_settlement_belongs_to_the_local_map_of_its_nearest_hex(region, lake_hex, globe):
    centers = region.geometry.local_to_ecef(region.grid.centers[:, 0], region.grid.centers[:, 1])
    here = ecef_to_cell(centers, globe.antimeridian) == region.region_id
    n = len(centers)
    yes, no = np.ones(n, dtype=bool), np.zeros(n, dtype=bool)
    owned = local_owned(region, lake_hex, centers, centers, yes, no, globe.antimeridian)
    np.testing.assert_array_equal(np.flatnonzero(owned), [lake_hex] if here[lake_hex] else [])
    # Not yet refined by the region, or already by a local map: not this map's.
    assert not local_owned(region, lake_hex, centers, centers, no, no, globe.antimeridian).any()
    assert not local_owned(region, lake_hex, centers, centers, yes, yes, globe.antimeridian).any()


def test_harbour_cities_land_on_the_local_shore(local, region, lake_hex, rivers, sites, globe, with_cities):
    old, new = refine_local_cities(local, region, lake_hex, rivers, globe)
    assert len(old) == len(with_cities)
    for city in with_cities:
        assert city.refined and city.local_refined
        at = np.flatnonzero(np.all(sites.positions == city.position, axis=1))
        assert at.size == 1 and sites.coast_lake[at[0]]
        assert np.linalg.norm(city.position - city.region_position) <= _reach(region) + 1e-9
        # The region's refinement is kept, for the roads and for a reset.
        assert np.any(city.region_position != city.position)
    np.testing.assert_array_equal(old, [c.region_position for c in with_cities])
    np.testing.assert_array_equal(new, [c.position for c in with_cities])
    # Once only: a second pass leaves them where they are.
    first = [c.position.copy() for c in with_cities]
    refine_local_cities(local, region, lake_hex, rivers, globe, LocalSettlementParams(refine_radius_parent_texels=0.0))
    for city, p in zip(with_cities, first):
        np.testing.assert_array_equal(city.position, p)
    for city in with_cities:
        city.reset_local_refinement()
        np.testing.assert_array_equal(city.position, city.region_position)
        assert city.refined and not city.local_refined


def test_cities_the_region_has_not_refined_stay_put(local, region, lake_hex, rivers, sites, globe, inland):
    cities = _cities(sites, inland, refined=False)
    globe.add_feature("settlements", cities)
    try:
        old, _ = refine_local_cities(local, region, lake_hex, rivers, globe)
    finally:
        globe.remove_feature("settlements")
    assert len(old) == 0 and not any(c.local_refined for c in cities)


def test_port_towns_land_on_the_local_shore(local, region, lake_hex, rivers, sites, globe, inland):
    pos = sites.positions[inland]
    towns = Towns(
        positions=pos.copy(), population=np.array([3_000.0, 1_000.0]), face_id=sites.face[inland].astype(np.int32),
        parent=np.full(2, -1, dtype=np.int32), kind=np.full(2, TOWN_PORT, dtype=np.int8),
        refined=np.ones(2, dtype=bool),
    )
    globe.add_feature("towns", towns)
    try:
        assert refine_local_towns(local, region, lake_hex, rivers, globe) == 2
    finally:
        globe.remove_feature("towns")
    assert towns.local_refined.all()
    np.testing.assert_array_equal(towns.region_positions, pos)
    moved = np.flatnonzero(np.any(towns.positions != pos, axis=1))
    assert moved.size >= 1
    for i in moved:
        at = np.flatnonzero(np.all(sites.positions == towns.positions[i], axis=1))
        assert at.size == 1 and sites.coast[at[0]]


def test_the_local_map_draws_its_cities_where_they_now_stand(region, region_rivers, lake_hex, globe, with_cities):
    lm = build_local_map(globe, region, lake_hex, LocalParams(upsample=local_params()), region_rivers=region_rivers)
    assert all(c.local_refined for c in with_cities)
    for k, city in enumerate(with_cities):
        j = list(lm.city_index).index(k)
        np.testing.assert_array_equal(lm.city_positions[j], city.position)
