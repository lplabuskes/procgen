"""River valleys carved into a refined map: beds never rise, water and parent means are kept."""
import copy
import dataclasses

import numpy as np
import pytest

from region.carve import RiverCarveParams, carve_rivers, decreasing_fit
from region.rivers import build_region_rivers
from region.source import GlobalSource
from region.upsample import RegionUpsampleParams, build_region_fields, conservation_report

REGION_ID = 40   # coast, a lake and a peak (see conftest)


def test_decreasing_fit_never_rises():
    z = np.array([5.0, 3.0, 4.0, 6.0, 2.0, 2.5, 1.0])
    for q in (0.0, 0.25, 0.5):
        fit = decreasing_fit(z, q)
        assert fit.shape == z.shape
        assert np.all(np.diff(fit) <= 0.0)
    assert np.allclose(decreasing_fit(z, 0.0), np.minimum.accumulate(z))      # cut only
    falling = np.array([4.0, 3.0, 3.0, 1.0])
    assert np.allclose(decreasing_fit(falling, 0.5), falling)                 # nothing to fix


def test_decreasing_fit_median_cuts_and_fills():
    fit = decreasing_fit(np.array([1.0, 3.0]), 0.5)                         # one rise, pooled
    assert np.allclose(fit, [2.0, 2.0])


@pytest.fixture(scope="module")
def region(synthetic_globe):
    fields = build_region_fields(synthetic_globe, REGION_ID, RegionUpsampleParams(n_cols=6, tex_size=16))
    return fields, build_region_rivers(fields, synthetic_globe)


def _climb(fields, paths) -> float:
    """Total rise (km) between successive drawn vertices, at each vertex's texel."""
    elev = fields.textures["elevation"][..., 0].astype(np.float64)
    total = 0.0
    for positions, _, in_water in paths:
        s, t, valid = fields.geometry.ecef_to_local(np.asarray(positions, dtype=np.float64))
        h, r, c, covered = fields.grid.nearest_texels(s, t, fields.tex_size)
        ok = valid & covered
        drawn = ok[1:] & ok[:-1] & ~np.asarray(in_water, dtype=bool)
        total += float(np.where(drawn, np.maximum(np.diff(elev[h, r, c]), 0.0), 0.0).sum())
    return total


def _carved(region, globe, **overrides):
    fields, rivers = region
    out = copy.deepcopy(fields)
    params = dataclasses.replace(RiverCarveParams(), **overrides)
    carve_rivers(out, GlobalSource(globe), rivers.paths, params, out.in_hex)
    return out


def test_carving_stops_the_climbs_and_keeps_the_means(region, synthetic_globe):
    fields, rivers = region
    assert _climb(fields, rivers.paths) > 0.0
    carved = _carved(region, synthetic_globe)
    assert _climb(carved, rivers.paths) < 0.25 * _climb(fields, rivers.paths)
    assert conservation_report(carved, synthetic_globe)["elevation"]["max"] < 1e-5
    assert carved.uncarved is not None
    assert np.array_equal(carved.uncarved, fields.textures["elevation"])
    assert not np.array_equal(carved.textures["elevation"], fields.textures["elevation"])


def test_water_is_left_alone(region, synthetic_globe):
    fields, _ = region
    carved = _carved(region, synthetic_globe)
    water = (fields.textures["ocean_mask"] > 0.5) | (fields.textures["lake_mask"] > 0.5)
    assert np.array_equal(carved.textures["elevation"][water], fields.textures["elevation"][water])
    # No dry texel cut below sea level that wasn't already.
    sea = fields.textures["ocean_mask"] > 0.5
    before, after = fields.textures["elevation"], carved.textures["elevation"]
    assert not np.any(~sea & (after < 0.0) & (before >= 0.0))
    # filled_height follows the ground off the lakes.
    lake = fields.textures["lake_mask"] > 0.5
    assert np.array_equal(carved.textures["filled_height"][~lake], after[~lake])


def test_recarving_starts_from_the_uncarved_relief(region, synthetic_globe):
    fields, rivers = region
    carved = _carved(region, synthetic_globe)
    once = carved.textures["elevation"].copy()
    carve_rivers(carved, GlobalSource(synthetic_globe), rivers.paths, RiverCarveParams(), carved.in_hex)
    assert np.array_equal(carved.textures["elevation"], once)
    carve_rivers(carved, GlobalSource(synthetic_globe), rivers.paths, RiverCarveParams(carve=False), carved.in_hex)
    assert np.array_equal(carved.textures["elevation"], fields.textures["elevation"])


def test_a_lake_above_the_stretch_never_lifts_it(region):
    """A stretch running into water higher than ground it crosses climbs in rather than ride a causeway."""
    from region.carve import line_beds
    fields, rivers = region
    elev = fields.textures["elevation"][..., 0].astype(np.float64)
    positions, flows, in_water = max(rivers.paths, key=lambda p: len(p[0]))
    dry = (np.asarray(positions), flows, np.zeros_like(np.asarray(in_water, dtype=bool)))
    s, t, _ = fields.geometry.ecef_to_local(np.asarray(positions, dtype=np.float64)[-1:])
    h, r, c, _ = fields.grid.nearest_texels(s, t, fields.tex_size)
    water = np.zeros(elev.shape, dtype=bool)
    water[h, r, c] = True                                  # the line ends in a lake...
    surface = np.where(water, elev.max() + 1.0, np.inf)     # ...far above everything
    _, beds, _, _ = line_beds(fields, elev, surface, water, [dry], RiverCarveParams())
    assert np.all(np.diff(beds) <= 0.0)
    assert beds.max() <= elev.max()
