"""City footprints: the built-up ground each city covers on a local map.

A city of ``P`` people claims ``P / density_per_ha`` of dry texels (~200 people/ha,
walled medieval), nearest first with jittered distance (``edge_roughness``) for a
ragged edge.  Water is never claimed; rivers are vectors, so a city spans its
river.  Larger cities claim first; no texel has two owners.

Display-only (no rural density is carved out).  Qt-free and deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial import KDTree

from core.params import param_field as _f, paramclass
from region.upsample import EARTH_RADIUS_KM, RegionFields

_KM2_PER_HA = 0.01
_CITY_STREAM = 13   # RNG stream tag for the edge jitter


@paramclass
class CityParams:
    density_per_ha: float = _f(
        200.0, label="City built-up density (people/ha)", min=20.0, max=1_000.0, step=10.0)
    edge_roughness: float = _f(
        0.25, label="Footprint edge roughness (× distance)", min=0.0, max=1.0, step=0.05)


@dataclass
class CityFootprints:
    """The texels each of a local map's cities covers; ``owner`` is in ``LocalMap.city_index`` order."""

    owner: np.ndarray        # (M, T, T) int32, -1 where no city stands
    city: np.ndarray         # (K,) int — owning city of each claimed texel
    corners: np.ndarray      # (K, 4, 3) unit-sphere texel corners, in ring order
    area_km2: np.ndarray     # (C,) built-up area each city claimed

    def __len__(self) -> int:
        return len(self.area_km2)


def _texel_corners(local: RegionFields, texels: np.ndarray) -> np.ndarray:
    """``(K, 4, 3)`` sphere corners of the flat ``(hex, row, col)`` texel indices."""
    t = local.tex_size
    hexes, rows, cols = np.unravel_index(texels, (local.n_hexes, t, t))
    width = 2.0 * local.grid.radius / t              # texel side in local units
    centre = local.grid.texture_local(t)[hexes, rows, cols]   # (K, 2)
    half = 0.5 * width
    offsets = np.array([[-half, -half], [half, -half], [half, half], [-half, half]])
    ring = centre[:, None, :] + offsets[None, :, :]  # (K, 4, 2)
    flat = ring.reshape(-1, 2)
    return local.geometry.local_to_ecef(flat[:, 0], flat[:, 1]).reshape(-1, 4, 3)


def city_footprints(
    local: RegionFields, positions: np.ndarray, population: np.ndarray, seed: int,
    params: Optional[CityParams] = None,
) -> CityFootprints:
    """Claim each city's built-up texels.  See the module docstring."""
    params = params or CityParams()
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    population = np.asarray(population, dtype=np.float64).reshape(-1)
    n_cities = len(population)
    shape = local.in_hex.shape
    owner = np.full(shape, -1, dtype=np.int32)
    area = np.zeros(n_cities, dtype=np.float64)
    empty = CityFootprints(owner=owner, city=np.zeros(0, dtype=np.int64), corners=np.zeros((0, 4, 3)), area_km2=area)
    if n_cities == 0:
        return empty

    water = np.zeros(shape, dtype=bool)
    for key in ("ocean_mask", "lake_mask"):
        if key in local.textures:
            water |= local.textures[key][..., 0] > 0.5
    dry = np.flatnonzero((local.in_hex & ~water).reshape(-1))
    if dry.size == 0:
        return empty
    dry_pos = local.positions.reshape(-1, 3)[dry]
    tree = KDTree(dry_pos)
    rng = np.random.default_rng([int(seed), _CITY_STREAM])
    jitter = 1.0 + float(params.edge_roughness) * rng.random(dry.size)
    claimed = np.zeros(dry.size, dtype=bool)
    flat_owner = owner.reshape(-1)
    texel_km2 = float(local.texel_area_km2)
    density_km2 = max(float(params.density_per_ha), 1e-9) / _KM2_PER_HA

    for c in np.argsort(-population, kind="stable"):
        n = max(1, int(round(float(population[c]) / density_km2 / texel_km2)))
        # Search well beyond the footprint: water, jitter and neighbours push the edge out.
        r_km = np.sqrt(n * texel_km2 / np.pi)
        reach = (3.0 * r_km + 2.0 * np.sqrt(texel_km2)) * (1.0 + float(params.edge_roughness)) / EARTH_RADIUS_KM
        near = np.asarray(tree.query_ball_point(positions[c], reach), dtype=np.int64)
        near = near[~claimed[near]]
        if near.size == 0:
            continue
        d = np.linalg.norm(dry_pos[near] - positions[c], axis=1) * jitter[near]
        take = near[np.argsort(d, kind="stable")[:n]]
        claimed[take] = True
        flat_owner[dry[take]] = c
        area[c] = take.size * texel_km2

    texels = np.flatnonzero(flat_owner >= 0)
    return CityFootprints(
        owner=owner, city=flat_owner[texels].astype(np.int64), corners=_texel_corners(local, texels), area_km2=area,
    )
