"""MapStore: each region and local map built once, while held; params reset as the viewer did."""
import dataclasses

import numpy as np
import pytest

from localmap.params import LocalParams
from localmap.store import MapStore
from localmap.test.conftest import local_params
from localmap.villages import VillageParams
from region.params import RegionParams
from region.test.test_builder import land_towns
from region.test.test_resources import globe  # noqa: F401  (fixture)

REGION_ID = 40
SMALL_REGION = dataclasses.replace(
    RegionParams(), upsample=dataclasses.replace(RegionParams().upsample, n_cols=6, tex_size=16))
SMALL_LOCAL = LocalParams(
    upsample=local_params(), villages=VillageParams(village_min=20_000.0, village_max=100_000.0, spacing_km=15.0))


def small_store(globe, **kwargs) -> MapStore:  # noqa: F811
    store = MapStore(globe, **kwargs)
    store.set_region_params(SMALL_REGION)
    store.set_local_params(SMALL_LOCAL)
    return store


@pytest.fixture
def towns(globe):  # noqa: F811
    towns = land_towns(globe)
    globe.add_feature("towns", towns)
    yield towns
    globe.remove_feature("towns")


def test_maps_are_built_once_while_held(globe, towns):  # noqa: F811
    store = small_store(globe)
    region = store.region(REGION_ID)
    assert store.region(REGION_ID) is region
    local = store.local(REGION_ID, 0)
    assert store.local(REGION_ID, 0) is local
    assert store.builds == [("region", REGION_ID), ("local", (REGION_ID, 0))]


def test_least_recently_used_maps_go_first(globe, towns):  # noqa: F811
    store = small_store(globe, local_capacity=2)
    store.local(REGION_ID, 0)
    store.local(REGION_ID, 1)
    store.local(REGION_ID, 0)                   # 1 is now the oldest
    store.local(REGION_ID, 2)
    assert store.has_local(REGION_ID, 0) and store.has_local(REGION_ID, 2) and not store.has_local(REGION_ID, 1)


def test_local_params_drop_local_maps_and_local_refinement(globe, towns):  # noqa: F811
    store = small_store(globe)
    store.region(REGION_ID)
    store.local(REGION_ID, 0)
    towns.local_refined[:] = True
    new = dataclasses.replace(SMALL_LOCAL, villages=dataclasses.replace(SMALL_LOCAL.villages, spacing_km=10.0))
    assert store.set_local_params(new) and not store.set_local_params(new)
    assert store.has_region(REGION_ID) and not store.has_local(REGION_ID, 0)
    assert not towns.local_refined.any()
    np.testing.assert_array_equal(towns.positions, towns.region_positions)


def test_region_params_keep_fields_unless_the_detail_changes(globe, towns):  # noqa: F811
    store = small_store(globe)
    fields = store.region(REGION_ID).fields
    store.local(REGION_ID, 0)
    store.set_region_params(dataclasses.replace(
        SMALL_REGION, rivers=dataclasses.replace(SMALL_REGION.rivers, valley_seeking=1.0)))
    assert not store.has_region(REGION_ID) and not store.has_local(REGION_ID, 0)
    assert not towns.refined.any()
    again = store.region(REGION_ID)
    assert again.fields is fields                  # features re-run on the kept fields
    assert towns.refined.any()
    store.set_region_params(dataclasses.replace(
        SMALL_REGION, upsample=dataclasses.replace(SMALL_REGION.upsample, n_cols=5)))
    assert store.region(REGION_ID).fields is not fields


def test_a_new_globe_drops_everything(globe, towns):  # noqa: F811
    store = small_store(globe)
    store.region(REGION_ID)
    store.set_globe(globe)
    assert store.has_region(REGION_ID)
    store.set_globe(None)
    assert not store.has_region(REGION_ID)
