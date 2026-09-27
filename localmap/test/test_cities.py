"""City footprints: area from population and density, on dry ground, one city per texel."""
import numpy as np
import pytest

from localmap.cities import CityParams, city_footprints
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
    """The dry texel with the most dry ground around it, and a second a few texels off."""
    dry = local.positions[local.in_hex & ~_water(local)]
    spacing = local.grid.texel_spacing_rad(local.tex_size)
    d = np.linalg.norm(dry[:, None, :] - dry[None, :, :], axis=2)
    first = int(np.argmax((d < 4.0 * spacing).sum(axis=1)))
    second = int(np.argmin(np.abs(d[first] - 4.0 * spacing)))
    return np.stack([dry[first], dry[second]])


def _pops(local, n_texels, params=CityParams()) -> np.ndarray:
    """Populations that fill exactly ``n_texels`` texels each."""
    return np.asarray(n_texels, dtype=np.float64) * local.texel_area_km2 * params.density_per_ha * 100.0


def test_area_follows_population(local, sites):
    fp = city_footprints(local, sites[:1], _pops(local, [7]), seed=1)
    assert (fp.owner == 0).sum() == 7
    assert fp.area_km2[0] == pytest.approx(7 * local.texel_area_km2)
    assert fp.corners.shape == (7, 4, 3)


def test_a_tiny_city_still_claims_one_texel(local, sites):
    fp = city_footprints(local, sites[:1], np.array([1.0]), seed=1)
    assert (fp.owner == 0).sum() == 1


def test_footprints_stay_off_water(local, sites):
    fp = city_footprints(local, sites, _pops(local, [40, 25]), seed=3)
    assert not (_water(local) & (fp.owner >= 0)).any()


def test_cities_do_not_share_texels_and_bigger_claims_first(local, sites):
    fp = city_footprints(local, sites, _pops(local, [10, 30]), seed=3)
    assert (fp.owner == 1).sum() == 30               # the larger city is never short-changed
    assert set(np.unique(fp.owner)) <= {-1, 0, 1}
    assert len(fp.city) == (fp.owner >= 0).sum()


def test_footprint_is_compact_around_its_site(local, sites):
    n = 20
    fp = city_footprints(local, sites[:1], _pops(local, [n]), seed=5, params=CityParams(edge_roughness=0.0))
    d = np.linalg.norm(local.positions[fp.owner == 0] - sites[0], axis=1) * EARTH_RADIUS_KM
    # A disc of n texels has radius sqrt(n/π) texel spacings; allow slack for
    # the odd water texel pushing the edge out.
    spacing_km = local.grid.texel_spacing_rad(local.tex_size) * EARTH_RADIUS_KM
    assert d.max() < (np.sqrt(n / np.pi) + 2.0) * spacing_km
