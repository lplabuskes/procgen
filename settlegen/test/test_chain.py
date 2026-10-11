"""The chain from a settlement to its local maps: built only when the store lacks
them, the owning map first, and every map the frame reaches."""
import numpy as np
import pytest

from localmap.test.test_store import REGION_ID, small_store
from region.geometry import ecef_to_cell
from region.test.test_builder import land_towns
from region.test.test_resources import globe  # noqa: F401  (fixture)
from settlegen.chain import SettlementRef, ensure_local_maps, frame_keys, region_of


@pytest.fixture
def towns(globe):  # noqa: F811
    towns = land_towns(globe)
    globe.add_feature("towns", towns)
    yield towns
    globe.remove_feature("towns")


def _a_town(globe, towns) -> SettlementRef:  # noqa: F811
    here = np.flatnonzero(ecef_to_cell(towns.site_positions, globe.antimeridian) == REGION_ID)
    return SettlementRef("town", int(here[0]))


def test_a_settlement_builds_its_region_and_local_map_once(globe, towns):  # noqa: F811
    store = small_store(globe)
    ref = _a_town(globe, towns)
    first = ensure_local_maps(store, ref)
    assert first.built[0] == ("region", REGION_ID) and first.built[1][0] == "local"
    assert first.keys[0] == first.built[1][1] and len(first.built) == 1 + len(first.keys)
    assert towns.local_refined[ref.index]
    position = towns.positions[ref.index].copy()
    again = ensure_local_maps(store, ref)
    assert again.built == [] and again.maps[0] is first.maps[0]
    np.testing.assert_array_equal(towns.positions[ref.index], position)


def test_maps_the_viewer_built_are_not_rebuilt(globe, towns):  # noqa: F811
    store = small_store(globe)
    ref = _a_town(globe, towns)
    probe = ensure_local_maps(small_store(globe), ref)       # which maps it needs, in another store
    towns.reset_refinement()
    store.region(REGION_ID)                                  # as the viewer would
    for key in probe.keys:
        store.local(*key)
    assert ensure_local_maps(store, ref).built == []


def test_the_frame_reaches_exactly_the_maps_whose_ground_it_covers(globe, towns):  # noqa: F811
    store = small_store(globe)
    region = store.region(REGION_ID)
    grid = region.fields.grid
    # The hex nearest the cell's middle: well inside the region.
    h = int(np.argmin(np.hypot(*(grid.centers - 0.5).T)))
    owner = store.local(REGION_ID, h)
    n_cols = int(store.region_params.upsample.n_cols)

    def at(s, t):
        return region.fields.geometry.local_to_ecef(np.array([s]), np.array([t]))[0]

    c = grid.centers[h]
    assert frame_keys(globe, owner, at(*c), 1.0, n_cols) == [(REGION_ID, h)]
    # On the edge the hex shares with a neighbour: both maps.
    d = np.hypot(*(grid.centers - c).T)
    d[h] = np.inf
    n = int(np.argmin(d))
    mid = 0.5 * (c + grid.centers[n])
    assert sorted(frame_keys(globe, owner, at(*mid), 1.0, n_cols)) == sorted([(REGION_ID, h), (REGION_ID, n)])
    # On the region's own edge: a map of the neighbouring region too.
    keys = frame_keys(globe, owner, at(1.0, c[1]), 1.0, n_cols)
    assert {k[0] for k in keys} - {REGION_ID}


def test_the_region_is_the_site_cell(globe, towns):  # noqa: F811
    ref = _a_town(globe, towns)
    assert region_of(globe, ref) == REGION_ID
