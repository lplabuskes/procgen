"""Region/local wetland refinement: relief swing, riparian marsh, idempotence."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from region.upsample import RegionUpsampleParams
from region.wetlands import add_riparian_wetland, refine_wetland
from worldgen.climate.classification import WETLAND_BIOME


def test_refine_keeps_parent_wetland_and_follows_relief():
    p = RegionUpsampleParams()
    coarse = np.array([0.5, 0.5, 0.3, 0.3])
    base = np.zeros(4)
    elev = np.array([0.0, 1.0, -1.0, 0.0])          # km: flat, a big rise, a big hollow, flat
    out = refine_wetland(coarse, elev, base, p)
    assert np.isclose(out[0], 0.5) and np.isclose(out[3], 0.3)
    assert np.isclose(out[1], 0.5 - p.wetland_relief_gain)
    assert np.isclose(out[2], 0.3 + p.wetland_relief_gain)
    assert np.all(out[:2] >= p.wetland_threshold)    # parent wetland stays wetland


def _fields(n: int = 9):
    """One flat 'hex' of n×n texels along the equator, ~1 km apart."""
    step = 1.0 / 6371.0
    lon = (np.arange(n) - n // 2) * step
    lat = (np.arange(n) - n // 2) * step
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    pos = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1)[None]
    tex = {
        "elevation": np.zeros((1, n, n, 1), dtype=np.float32),
        "wetland": np.full((1, n, n, 1), 0.2, dtype=np.float32),
        "biome": np.full((1, n, n, 1), 3, dtype=np.uint8),
        "temperature": np.full((1, n, n, 4), 15.0, dtype=np.float32),
    }
    return SimpleNamespace(textures=tex, positions=pos, texel_area_km2=1.0)


def test_riparian_marsh_along_river_and_idempotent():
    p = RegionUpsampleParams()
    f = _fields()
    row = f.positions[0, 4]                               # a river along the middle row
    rivers = SimpleNamespace(paths=[(row, np.full(len(row) - 1, 1000.0), np.zeros(len(row) - 1, bool))],
                             streams=[])
    add_riparian_wetland(f, rivers, p)                    # type: ignore[arg-type]
    biome = f.textures["biome"][0, ..., 0]
    assert np.all(biome[4] == WETLAND_BIOME)              # on the river: 0.2 + gain ≥ threshold
    assert np.all(biome[0] == 3)                          # 4 km off, past the width: untouched
    first = f.textures["wetland"].copy()
    add_riparian_wetland(f, rivers, p)                    # type: ignore[arg-type]
    assert np.array_equal(first, f.textures["wetland"])
