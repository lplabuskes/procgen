"""The settlement tier's critical invariant: a settlement map refines the local
texels' elevation it covers — on one local map, or across the seam between two —
and holds every other field constant at the settlement's local value."""
import numpy as np
import pytest

from region.upsample import SMALL_LAKE_KEY
from settlegen.source import LocalMapsSource
from settlegen.test.conftest import FRAME_TEXELS, settlement_params
from settlegen.upsample import build_settlement_fields, clear_thin_land, settlement_conservation_report


@pytest.fixture(scope="module")
def inner(globe, lake_local, inner_center, side_km):
    return build_settlement_fields(globe, [lake_local], inner_center, side_km, settlement_params(side_km))


@pytest.fixture(scope="module")
def seam(globe, lake_local, neighbour_local, seam_center, side_km):
    return build_settlement_fields(
        globe, [lake_local, neighbour_local], seam_center, side_km, settlement_params(side_km))


def _assert_conserved(report):
    assert set(report) == {"elevation"}
    r = report["elevation"]
    assert r["n_anchors"] > 0
    assert r["max"] <= 1e-5, r


def test_settlement_map_has_the_local_layers(inner, lake_local):
    t = inner.tex_size
    assert t == FRAME_TEXELS
    for key, tex in lake_local.textures.items():
        assert inner.textures[key].shape == (1, t, t, tex.shape[3]), key
    assert inner.detail_octaves > 0


def test_downsampling_recovers_the_local_elevation(inner, lake_local, globe):
    _assert_conserved(settlement_conservation_report(inner, [lake_local], globe.antimeridian))


def test_the_rest_is_the_settlements_local_value(inner, lake_local, inner_center, globe):
    at = LocalMapsSource([lake_local], globe.antimeridian)
    loc = at.locate(inner_center[None])
    refined = {"elevation", "ocean_mask", "lake_mask", "filled_height", "coast_smoothness", "delta", SMALL_LAKE_KEY}
    constant = set(lake_local.textures) - refined
    assert {"temperature", "precipitation", "biome", "fertility", "danger"} <= constant
    for key in constant:
        tex = inner.textures[key]
        expected = at.anchor_values(key, loc)[0].astype(tex.dtype)
        assert (tex == expected).all(), key


def test_a_frame_across_the_seam_reads_both_maps(seam, lake_local, neighbour_local, globe):
    which = LocalMapsSource([lake_local, neighbour_local], globe.antimeridian).map_of(
        seam.positions.reshape(-1, 3).astype(np.float64))
    assert set(np.unique(which)) == {0, 1}
    _assert_conserved(settlement_conservation_report(seam, [lake_local, neighbour_local], globe.antimeridian))


def test_a_frame_past_the_given_maps_fails_loudly(globe, lake_local, seam_center, side_km):
    with pytest.raises(ValueError, match="outside every given local map"):
        build_settlement_fields(globe, [lake_local], seam_center, side_km, settlement_params(side_km))


def test_settlement_maps_regenerate_identically(globe, lake_local, inner_center, side_km, inner):
    again = build_settlement_fields(globe, [lake_local], inner_center, side_km, settlement_params(side_km))
    for key, tex in inner.textures.items():
        assert np.array_equal(tex, again.textures[key]), key


def test_every_anchor_is_a_conserved_local_texel(seam, lake_local, neighbour_local, globe):
    """Never a map's over-hang past its parent hexagon: that ground is the neighbour's."""
    maps = [lake_local, neighbour_local]
    domain = np.concatenate([m.domain.reshape(-1) for m in maps])
    assert seam.parent_anchors is not None
    assert domain[seam.parent_anchors.reshape(-1)].all()
    owners = LocalMapsSource(maps, globe.antimeridian).owners()
    assert domain[owners[domain]].all()


def test_thin_land_between_waters_goes_to_the_nearest_water():
    sea = np.zeros((40, 40), dtype=np.uint8)
    sea[:, :20] = 1
    sea[5:35, 25:35] = 1                        # a bay, cut off from the sea by a 5-texel strip
    sea[10, 10] = 0                             # a one-texel islet
    lake = np.zeros((40, 40), dtype=np.uint8)
    textures = {"ocean_mask": sea[None, ..., None], "lake_mask": lake[None, ..., None]}
    clear_thin_land(textures, 15.0, 75.0)       # five texels: the strip stays
    out = textures["ocean_mask"][0, ..., 0]
    assert out[10, 10] == 1 and not out[20, 22]
    clear_thin_land(textures, 15.0, 120.0)      # eight: it goes
    assert textures["ocean_mask"][0, ..., 0][20, 22] == 1
    # Wide land, even at the frame's edge, stays.
    assert not textures["ocean_mask"][0, ..., 0][:, 36:].any()
