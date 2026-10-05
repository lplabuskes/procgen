"""Refining towns and cities onto a local map's coast, rivers and roads.

``region/towns.py`` / ``region/cities.py``'s rules one tier down, within
``refine_radius_parent_texels`` regional texels of the region's site.  The result
is saved on the world (``local_refined``) so every map shows it in one place.

Ownership: each settlement is refined once, by the local map of the regional hex
nearest its region position, in the region that refined it.  Other maps draw it
wherever it currently stands.

Cities go before the local roads (so roads end on them); towns after (their
route flag reads the local roads).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from core.params import param_field as _f, paramclass
from region.cities import refine_cities
from region.geometry import ecef_to_cell
from region.rivers import RegionRivers
from region.roads import RegionRoads
from region.towns import LOCAL_TIER, refine_towns
from region.upsample import RegionFields

if TYPE_CHECKING:
    from core.world import World
    from realmgen.population.builder import PopulationParams
    from realmgen.towns.builder import TownParams


@paramclass
class LocalSettlementParams:
    """Tuning for refining town and city positions on a local map, and where settlements stand by rivers."""

    # How far a town or city may move from where its region put it, in regional texels.
    refine_radius_parent_texels: float = _f(
        1.5, label="Refinement radius (regional texels)", min=0.0, max=4.0, step=0.25)
    # Towns, cities and villages stand this far past a river's bank, not in its channel.
    bank_clearance_km: float = _f(0.1, label="Clearance from river banks (km)", min=0.0, max=2.0, step=0.05)


def local_owned(
    region: RegionFields, hex_index: int, site_positions: np.ndarray, region_positions: np.ndarray,
    refined: np.ndarray, local_refined: np.ndarray, antimeridian,
) -> np.ndarray:
    """``(K,)`` True for settlements this hex's local map refines: refined by
    this region, not yet locally, and nearest this hex."""
    site_positions = np.asarray(site_positions, dtype=np.float64).reshape(-1, 3)
    if len(site_positions) == 0:
        return np.zeros(0, dtype=bool)
    here = ecef_to_cell(site_positions, antimeridian) == region.region_id
    s, t, valid = region.geometry.ecef_to_local(np.asarray(region_positions, dtype=np.float64).reshape(-1, 3))
    centers = region.grid.centers
    nearest = np.argmin((s[:, None] - centers[None, :, 0]) ** 2 + (t[:, None] - centers[None, :, 1]) ** 2, axis=1)
    return here & valid & (nearest == int(hex_index)) & np.asarray(refined, bool) & ~np.asarray(local_refined, bool)


def _reach(region: RegionFields, params: LocalSettlementParams) -> float:
    return float(params.refine_radius_parent_texels) * region.grid.texel_spacing_rad(region.tex_size)


def refine_local_cities(
    local: RegionFields, region: RegionFields, hex_index: int, rivers: RegionRivers, globe: "World",
    params: Optional[LocalSettlementParams] = None, river_threshold_factor: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Refine the cities this local map owns, in place; returns ``(old, new)``
    positions of every moved city, for the roads."""
    settlements = globe.get_feature("settlements") or []
    if not settlements:
        return np.zeros((0, 3)), np.zeros((0, 3))
    params = params or LocalSettlementParams()
    todo = np.flatnonzero(local_owned(
        region, hex_index,
        np.array([s.site_position for s in settlements]), np.array([s.region_position for s in settlements]),
        np.array([s.refined for s in settlements]), np.array([s.local_refined for s in settlements]),
        globe.antimeridian,
    ))
    refine_cities(local, rivers, globe, todo, LOCAL_TIER, _reach(region, params), river_threshold_factor)
    old = np.array([s.region_position for s in settlements], dtype=np.float64).reshape(-1, 3)
    new = np.array([s.position for s in settlements], dtype=np.float64).reshape(-1, 3)
    moved = np.any(old != new, axis=1)
    return old[moved], new[moved]


def refine_local_towns(
    local: RegionFields, region: RegionFields, hex_index: int, rivers: RegionRivers, globe: "World",
    params: Optional[LocalSettlementParams] = None,
    town_params: Optional["TownParams"] = None,
    population_params: Optional["PopulationParams"] = None,
    river_threshold_factor: float = 1.5,
    roads: Optional[RegionRoads] = None,
) -> int:
    """Refine the towns this local map owns, in place; returns how many."""
    towns = globe.get_feature("towns")
    if towns is None or len(towns) == 0:
        return 0
    params = params or LocalSettlementParams()
    todo = np.flatnonzero(local_owned(
        region, hex_index, towns.site_positions, towns.region_positions, towns.refined, towns.local_refined,
        globe.antimeridian,
    ))
    refine_towns(
        local, rivers, globe, todo, LOCAL_TIER, _reach(region, params),
        town_params, population_params, river_threshold_factor, roads,
    )
    return int(todo.size)
