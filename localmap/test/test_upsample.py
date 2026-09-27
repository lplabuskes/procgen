"""The local tier's critical invariant: a local map refines its region's texels."""
import numpy as np
import pytest

from localmap.test.conftest import local_params
from localmap.upsample import build_local_fields, local_conservation_report


@pytest.fixture(scope="module")
def local(globe, region, lake_hex):
    return build_local_fields(globe, region, lake_hex, local_params())


def test_region_exports_its_lake_surface(region):
    lake = region.textures["lake_mask"][..., 0] > 0.5
    assert lake.any()
    filled = region.textures["filled_height"][..., 0]
    elev = region.textures["elevation"][..., 0]
    assert np.all(filled[lake] >= elev[lake])
    assert np.array_equal(filled[~lake], elev[~lake])


def test_local_map_has_the_region_layers(local, region):
    m, t = local.n_hexes, local.tex_size
    for key, tex in region.textures.items():
        assert local.textures[key].shape == (m, t, t, tex.shape[3]), key
    assert local.parent_anchors is not None and local.parent_anchors.shape == (m, t, t)
    assert local.detail_octaves > 0


def test_downsampling_recovers_the_region(local, region):
    report = local_conservation_report(local, region)
    assert {"elevation", "temperature", "precipitation", "rural_density", "fertility"} <= set(report)
    scale = {"elevation": 1.0, "temperature": 30.0, "precipitation": 500.0, "rural_density": 20.0}
    for key, r in report.items():
        assert r["n_anchors"] > 0, key
        assert r["max"] <= 1e-5 * scale.get(key, 1.0), (key, r)


def test_rural_people_are_the_parent_hexs(local, region, lake_hex):
    people = float((local.textures["rural_density"][..., 0] * local.domain).sum() * local.texel_area_km2)
    parent = float(
        (region.textures["rural_density"][lake_hex, ..., 0] * region.in_hex[lake_hex]).sum() * region.texel_area_km2
    )
    assert parent > 0.0
    # Not exact: each tier counts the regional texels on the parent hexagon's
    # rim differently (a texel is in or out of the region's hex by its centre,
    # in or out of the local map by its children's).  That is a ~1/T effect —
    # a few % at this test's T=16, ~0.03% at the real T=64.
    assert people == pytest.approx(parent, rel=0.05)


def test_the_parent_lake_carries_down(local):
    lake = local.textures["lake_mask"][..., 0] > 0.5
    assert (lake & local.domain).any()
    assert np.all(local.textures["filled_height"][..., 0][lake] >= local.textures["elevation"][..., 0][lake])


def test_local_maps_regenerate_identically(globe, region, lake_hex, local):
    again = build_local_fields(globe, region, lake_hex, local_params())
    for key, tex in local.textures.items():
        assert np.array_equal(tex, again.textures[key]), key


def test_global_indices_stay_global(local, globe):
    n, h, w, _ = globe.mesh.face_textures["elevation"].shape
    assert local.anchors.max() < n * h * w
    assert local.faces.max() < n
