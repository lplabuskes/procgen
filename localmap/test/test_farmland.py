"""Farmland: quotas by settlement kind, on dry unbuilt ground, preferring good land."""
import dataclasses

import numpy as np
import pytest

from localmap.cities import city_footprints
from localmap.farmland import FARM_CITY, FARM_TOWN, FARM_VILLAGE, FarmlandParams, farmland_rgba, place_farmland
from localmap.test.conftest import local_params
from localmap.upsample import build_local_fields
from region.upsample import EARTH_RADIUS_KM


@pytest.fixture(scope="module")
def local(globe, region, lake_hex):
    return build_local_fields(globe, region, lake_hex, local_params())


def _water(local) -> np.ndarray:
    return (local.textures["ocean_mask"][..., 0] > 0.5) | (local.textures["lake_mask"][..., 0] > 0.5)


@pytest.fixture(scope="module")
def sites(local):
    """Three dry texels with plenty of dry ground around them, well apart."""
    dry = local.positions[local.in_hex & ~_water(local)]
    spacing = local.grid.texel_spacing_rad(local.tex_size)
    d = np.linalg.norm(dry[:, None, :] - dry[None, :, :], axis=2)
    roomy = (d < 6.0 * spacing).sum(axis=1)
    first = int(np.argmax(roomy))
    second = int(np.argmin(np.abs(d[first] - 8.0 * spacing) - 0.01 * roomy))
    third = int(np.argmax(np.minimum(d[first], d[second]) * (roomy > 0.5 * roomy.max())))
    return dry[[first, second, third]]


def _people(local, n_texels, ha=FarmlandParams().ha_per_farmer) -> np.ndarray:
    """Farmers who need exactly ``n_texels`` texels of fields each."""
    return np.asarray(n_texels, dtype=np.float64) * local.texel_area_km2 / (ha * 0.01)


def test_villages_and_towns_get_their_quota(local, sites):
    v_pop = _people(local, [6, 9])
    t_pop = _people(local, [8]) / FarmlandParams().town_farming_share
    fm = place_farmland(local, sites[:2], v_pop, sites[2:], t_pop, None, seed=1)
    assert (fm.kind == FARM_VILLAGE).sum() == 15
    assert (fm.kind == FARM_TOWN).sum() == 8
    np.testing.assert_array_equal(np.bincount(fm.owner[fm.kind == FARM_VILLAGE], minlength=2), [6, 9])
    np.testing.assert_allclose(fm.area_km2[FARM_VILLAGE], fm.target_km2[FARM_VILLAGE])
    np.testing.assert_allclose(fm.area_km2[FARM_TOWN], [8 * local.texel_area_km2])


def test_fields_stay_off_water_and_out_of_the_city(local, sites):
    fp = city_footprints(local, sites[:1], np.array([60_000.0]), seed=2)
    fm = place_farmland(local, sites[1:], _people(local, [40, 40]), np.zeros((0, 3)), np.zeros(0), fp, seed=2)
    farmed = fm.kind >= 0
    assert farmed.any()
    assert not (farmed & _water(local)).any()
    assert not (farmed & (fp.owner >= 0)).any()
    assert not (farmed & ~local.in_hex).any()


def test_gardens_ring_the_city(local, sites):
    fp = city_footprints(local, sites[:1], np.array([60_000.0]), seed=3)
    # The test map's texels are coarse: a ring two and a half texels wide.
    ring_km = 2.5 * local.grid.texel_spacing_rad(local.tex_size) * EARTH_RADIUS_KM
    params = FarmlandParams(city_garden_ring_km=ring_km, edge_roughness=0.0,
                            fertility_weight=0.0, slope_weight=0.0, forest_weight=0.0)
    fm = place_farmland(local, np.zeros((0, 3)), np.zeros(0), np.zeros((0, 3)), np.zeros(0), fp, 3, params)
    gardens = local.positions[fm.kind == FARM_CITY]
    assert len(gardens) > (fp.owner >= 0).sum()
    built = local.positions[fp.owner >= 0]
    gap = np.linalg.norm(gardens[:, None] - built[None], axis=2).min(axis=1) * EARTH_RADIUS_KM
    assert gap.max() <= ring_km + 1e-6
    assert fm.area_km2[FARM_CITY][0] == pytest.approx(len(gardens) * local.texel_area_km2)
    # Without the ring a city has no farmland of its own.
    none = place_farmland(local, np.zeros((0, 3)), np.zeros(0), np.zeros((0, 3)), np.zeros(0), fp, 3,
                          dataclasses.replace(params, city_garden_ring_km=0.0))
    assert not (none.kind >= 0).any()


def test_fields_prefer_fertile_ground(local, sites):
    """Poison the soil on one side of the village: its fields go to the other."""
    s, _, _ = local.geometry.ecef_to_local(local.positions.reshape(-1, 3))
    s0, _, _ = local.geometry.ecef_to_local(sites[:1])
    east = (s > s0[0]).reshape(local.in_hex.shape)
    fert = np.where(east, 0.0, 1.0)[..., None].astype(np.float32)
    poor = dataclasses.replace(local, textures={**local.textures, "fertility": fert})
    params = FarmlandParams(fertility_weight=3.0, edge_roughness=0.0)
    fm = place_farmland(poor, sites[:1], _people(local, [30]), np.zeros((0, 3)), np.zeros(0), None, 4, params)
    fields = fm.kind == FARM_VILLAGE
    assert fields.sum() == 30
    assert (fields & ~east).sum() > 2 * (fields & east).sum()


def test_neighbours_share_the_ground_between_them(local, sites):
    """Two villages side by side: neither is starved, whichever comes first."""
    spacing = local.grid.texel_spacing_rad(local.tex_size)
    d = np.linalg.norm(local.positions[local.in_hex & ~_water(local)] - sites[0], axis=1)
    near = local.positions[local.in_hex & ~_water(local)][np.argmin(np.abs(d - 3.0 * spacing))]
    pair = np.stack([sites[0], near])
    for order in (pair, pair[::-1]):
        fm = place_farmland(local, order, _people(local, [25, 25]), np.zeros((0, 3)), np.zeros(0), None, 5)
        np.testing.assert_array_equal(np.bincount(fm.owner[fm.kind == FARM_VILLAGE], minlength=2), [25, 25])


def test_deterministic(local, sites):
    args = (local, sites[:2], _people(local, [12, 7]), sites[2:], _people(local, [9]), None, 6)
    a, b = place_farmland(*args), place_farmland(*args)
    np.testing.assert_array_equal(a.kind, b.kind)
    np.testing.assert_array_equal(a.owner, b.owner)


def test_colours_follow_the_overlays(local, sites):
    fp = city_footprints(local, sites[:1], np.array([60_000.0]), seed=7)
    fm = place_farmland(local, sites[1:2], _people(local, [10]), sites[2:], _people(local, [10]), fp, 7)
    for show in ((True, False, False), (False, True, False), (False, False, True), (True, True, True)):
        alpha = farmland_rgba(fm, show)[..., 3] > 0
        expected = np.zeros_like(alpha)
        for k, on in enumerate(show):
            if on:
                expected |= fm.kind == k
        np.testing.assert_array_equal(alpha, expected)


def test_mining_settlements_farm_only_their_farming_share(local, sites):
    share = FarmlandParams().mining_farming_share
    v_pop = _people(local, [20, 20])
    t_pop = _people(local, [10]) / FarmlandParams().town_farming_share
    fm = place_farmland(local, sites[:2], v_pop, sites[2:], t_pop, None, seed=8,
                        village_mining=np.array([True, False]), town_mining=np.array([True]))
    v_area = np.bincount(fm.owner[fm.kind == FARM_VILLAGE], minlength=2)
    assert v_area[0] == round(20 * share)
    assert v_area[1] == 20
    np.testing.assert_allclose(fm.target_km2[FARM_TOWN], t_pop * share * FarmlandParams().ha_per_farmer * 0.01)


def test_only_a_villages_farmers_need_fields(local, sites):
    """Fishers and miners are part of a village's people: only the farming share is fed by fields."""
    v_pop = _people(local, [20, 20])
    fm = place_farmland(local, sites[:2], v_pop, np.zeros((0, 3)), np.zeros(0), None, seed=9,
                        village_mining=np.array([True, False]), village_farming=np.array([0.25, 0.5]))
    v_area = np.bincount(fm.owner[fm.kind == FARM_VILLAGE], minlength=2)
    assert list(v_area) == [5, 10]
