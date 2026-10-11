"""Villages: sized per global texel ∩ map, sited on the local terrain, carved out
of the rural density."""
import numpy as np
import pytest

from localmap.builder import build_local_map
from localmap.params import LocalParams
from localmap.test.conftest import local_params
from localmap.upsample import build_local_fields
from localmap.villages import VillageParams, place_villages, village_budget_report
from region.upsample import EARTH_RADIUS_KM

# The test grid is coarse (local texels ~9 km), so villages here are scaled up to
# match: a few per budget, each claiming a few texels.
PARAMS = VillageParams(village_min=20_000.0, village_max=100_000.0, spacing_km=15.0, settlement_clearance_km=20.0)


@pytest.fixture(scope="module")
def local(globe, region, river_hex):
    return build_local_fields(globe, region, river_hex, local_params())


@pytest.fixture(scope="module")
def villages(local):
    return place_villages(local, None, None, np.zeros((0, 3)), np.zeros(0), seed=11, params=PARAMS)


def _water(local) -> np.ndarray:
    return (local.textures["ocean_mask"][..., 0] > 0.5) | (local.textures["lake_mask"][..., 0] > 0.5)


def test_people_are_conserved_per_budget(local, villages):
    report = village_budget_report(local, villages)
    assert len(villages) > 0
    assert report["max_relative_residual"] < 1e-5


def test_villages_take_about_the_village_fraction(local, villages):
    report = village_budget_report(local, villages)
    assert 0.5 * PARAMS.village_fraction < report["village_share"] <= PARAMS.village_fraction + 0.05


def test_sizes_are_in_range(villages):
    assert villages.population.min() >= PARAMS.village_min - 1e-6
    assert villages.population.max() <= PARAMS.village_max + 1e-6


def test_villages_stand_on_dry_land_in_the_map(local, villages):
    dry = local.positions[local.domain & ~_water(local)]
    d = np.linalg.norm(villages.positions[:, None, :] - dry[None, :, :], axis=2).min(axis=1)
    assert d.max() < 1e-9


def test_villages_keep_their_spacing(villages):
    r = PARAMS.spacing_km / EARTH_RADIUS_KM * (villages.population / PARAMS.village_min) ** 0.25
    d = np.linalg.norm(villages.positions[:, None, :] - villages.positions[None, :, :], axis=2)
    need = np.maximum(r[:, None], r[None, :])
    np.fill_diagonal(d, np.inf)
    assert np.all(d >= need * (1.0 - 1e-9))


def test_villages_keep_clear_of_towns(local, villages):
    town = villages.positions[:1]
    again = place_villages(local, None, None, town, np.array([1_000.0]), seed=11, params=PARAMS)
    clearance = PARAMS.settlement_clearance_km / EARTH_RADIUS_KM
    assert np.linalg.norm(again.positions - town, axis=1).min() >= clearance


def test_villages_are_deterministic(local, villages):
    again = place_villages(local, None, None, np.zeros((0, 3)), np.zeros(0), seed=11, params=PARAMS)
    assert np.array_equal(again.positions, villages.positions)
    assert np.array_equal(again.population, villages.population)


def test_no_villages_when_the_fraction_is_zero(local):
    none = place_villages(local, None, None, np.zeros((0, 3)), np.zeros(0), seed=11,
                          params=VillageParams(village_fraction=0.0))
    assert len(none) == 0
    assert np.array_equal(none.rural_density[..., 0], local.textures["rural_density"][..., 0])


def test_the_whole_map_builds(globe, region, region_rivers, river_hex):
    params = LocalParams(upsample=local_params(), villages=PARAMS)
    lm = build_local_map(globe, region, river_hex, params, region_rivers=region_rivers)
    assert lm.hex_index == river_hex
    assert lm.rivers.paths and len(lm.villages) > 0
    assert village_budget_report(lm.fields, lm.villages)["max_relative_residual"] < 1e-5


def test_a_villages_farming_share_is_what_its_site_does_not_fish_or_mine(local, villages):
    share = villages.farming_share
    assert len(share) == len(villages) and ((share >= 0.0) & (share <= 1.0)).all()
    rho = local.textures["rural_density"][..., 0].astype(np.float64)
    others = sum(local.textures[k][..., 0].astype(np.float64) for k in ("fishing_density", "mining_density")
                 if k in local.textures)
    s, t, _ = local.geometry.ecef_to_local(villages.positions)
    hexes, rows, cols, _ = local.grid.texels_at(s, t, local.tex_size)
    expect = np.where(rho[hexes, rows, cols] > 0,
                      np.clip(1.0 - np.broadcast_to(others, rho.shape)[hexes, rows, cols]
                              / np.maximum(rho[hexes, rows, cols], 1e-300), 0.0, 1.0), 1.0)
    np.testing.assert_allclose(share, expect, atol=1e-6)
