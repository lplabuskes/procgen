"""One local map, end to end: fields, rivers, roads, settlements and villages.

:func:`build_local_map` runs the local tier's steps in order over one regional
hex, from the region's own refined fields, rivers and roads:

1. the fields (:func:`localmap.upsample.build_local_fields`);
2. the rivers (:func:`localmap.rivers.build_local_rivers`), and the region's
   deltas moved onto the local river mouths;
3. the roads (:func:`localmap.roads.build_local_roads`), fording the local rivers;
4. the towns and cities in and around the map, at *local* positions — moved off
   the local water, a river town onto its local river and a port onto the local
   coast when one is in reach.  Display-only: the world's ``towns`` feature
   keeps the region's refinement;
5. the villages (:func:`localmap.villages.place_villages`), clear of those
   towns and cities.

Qt-free and deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np
from scipy.spatial import KDTree

from localmap.params import LocalParams
from localmap.rivers import build_local_rivers
from localmap.roads import build_local_roads
from localmap.upsample import build_local_fields
from localmap.villages import Villages, place_villages
from realmgen.towns.types import TOWN_PORT, TOWN_RIVER
from region.display import snap_to_land
from region.rivers import RegionRivers, snap_deltas
from region.roads import RegionRoads
from region.towns import _neighbour_any, _river_points
from region.upsample import RegionFields

if TYPE_CHECKING:
    from core.world import World
    from worldgen.cost.builder import LandCostParams
    from worldgen.resources.builder import ResourcesParams

# How far past the local square a town or city still keeps villages away, in
# local units (the map is 1; a town's clearance is a few km, ~0.01–0.05).
_SETTLEMENT_MARGIN = 0.05


@dataclass
class LocalMap:
    """Everything one local map shows."""

    fields: RegionFields
    rivers: RegionRivers
    roads: RegionRoads
    villages: Villages
    # Towns and cities in and around the map: their world index and local position.
    town_index: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    town_positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    city_index: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    city_positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    deltas: Optional[np.ndarray] = None

    @property
    def hex_index(self) -> int:
        return int(getattr(self.fields.geometry, "hex_index"))


def _in_box(local: RegionFields, positions: np.ndarray, margin: float) -> np.ndarray:
    if len(positions) == 0:
        return np.zeros(0, dtype=bool)
    s, t, valid = local.geometry.ecef_to_local(positions)
    return valid & (s >= -margin) & (s <= 1.0 + margin) & (t >= -margin) & (t <= 1.0 + margin)


def _snap_to(points: np.ndarray, targets: Optional[np.ndarray], reach: float, which: np.ndarray) -> np.ndarray:
    """``points[which]`` moved onto their nearest ``targets`` point within ``reach``."""
    out = points.copy()
    if targets is None or len(targets) == 0 or not which.any():
        return out
    dist, near = KDTree(targets).query(points[which], workers=-1)
    near = np.asarray(near, dtype=np.int64)
    moved = out[which]
    ok = dist < reach
    moved[ok] = targets[near[ok]]
    out[which] = moved
    return out


def _local_settlements(
    local: RegionFields, rivers: RegionRivers, globe: "World", land_reach: float, parent_spacing: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(town_index, town_positions, town_pops, city_index, city_positions, city_pops)``
    for the towns and cities in and around the map, at local positions."""
    towns = globe.get_feature("towns")
    t_idx = np.zeros(0, dtype=np.int64)
    t_pos, t_pop = np.zeros((0, 3)), np.zeros(0)
    if towns is not None and len(towns):
        t_idx = np.flatnonzero(_in_box(local, np.asarray(towns.positions, dtype=np.float64), _SETTLEMENT_MARGIN))
        t_pos = snap_to_land(local, np.asarray(towns.positions, dtype=np.float64)[t_idx], land_reach)
        t_pop = np.asarray(towns.population, dtype=np.float64)[t_idx]
        kind = np.asarray(towns.kind)[t_idx]
        # A river town sat within ~1.5 regional texels of its regional river,
        # which the local map moved by up to a corridor: reach both.
        spacing = local.grid.texel_spacing_rad(local.tex_size)
        t_pos = _snap_to(t_pos, _river_points(rivers, 0.5 * spacing), 2.5 * parent_spacing, kind == TOWN_RIVER)
        water = np.zeros(local.in_hex.shape, dtype=bool)
        for key in ("ocean_mask", "lake_mask"):
            if key in local.textures:
                water |= local.textures[key][..., 0] > 0.5
        coast = local.positions[local.in_hex & ~water & _neighbour_any(water)]
        t_pos = _snap_to(t_pos, coast, 2.0 * parent_spacing, kind == TOWN_PORT)
    settlements = globe.get_feature("settlements") or []
    c_idx = np.zeros(0, dtype=np.int64)
    c_pos, c_pop = np.zeros((0, 3)), np.zeros(0)
    if settlements:
        all_pos = np.array([np.asarray(s.position, dtype=np.float64) for s in settlements]).reshape(-1, 3)
        c_idx = np.flatnonzero(_in_box(local, all_pos, _SETTLEMENT_MARGIN))
        c_pos = snap_to_land(local, all_pos[c_idx], land_reach)
        c_pop = np.array([float(settlements[i].population) for i in c_idx])
    return t_idx, t_pos, t_pop, c_idx, c_pos, c_pop


def build_local_map(
    globe: "World",
    region: RegionFields,
    hex_index: int,
    params: Optional[LocalParams] = None,
    *,
    region_rivers: Optional[RegionRivers] = None,
    region_roads: Optional[RegionRoads] = None,
    region_deltas: Optional[np.ndarray] = None,
    resources: Optional["ResourcesParams"] = None,
    fertility_exponent: float = 2.0,
    land: Optional["LandCostParams"] = None,
    danger_scale: float = 2.0,
) -> LocalMap:
    """Build the local map of regional hex ``hex_index``.  See the module docstring.

    ``region_rivers`` / ``region_roads`` / ``region_deltas`` are the region's
    (as the region view holds them); without them the map has none.  The rest
    is the world's and realm's tuning, as for the region.
    """
    params = params or LocalParams()
    local = build_local_fields(
        globe, region, hex_index, params.upsample, resources=resources, fertility_exponent=fertility_exponent,
    )
    spacing = local.grid.texel_spacing_rad(local.tex_size)
    parent_spacing = region.grid.texel_spacing_rad(region.tex_size)
    # How far a settlement on the local water moves onto land: as far as the
    # refinement lets water spread, plus a couple of texels.
    land_reach = (int(params.upsample.water_spread_texels) + 2) * spacing

    rivers = (build_local_rivers(local, region, region_rivers, globe, params.rivers)
              if region_rivers is not None else RegionRivers(region_id=local.region_id, paths=[]))
    deltas = None
    if region_deltas is not None and len(region_deltas):
        inside = _in_box(local, np.asarray(region_deltas, dtype=np.float64), _SETTLEMENT_MARGIN)
        reach = 2.0 * float(params.rivers.corridor_parent_texels) * parent_spacing + 2.0 * land_reach
        deltas = snap_deltas(np.asarray(region_deltas, dtype=np.float64)[inside], rivers, reach)
    roads = (build_local_roads(local, region, region_roads, globe, rivers, params.roads,
                               land=land, danger_scale=danger_scale, land_reach=land_reach)
             if region_roads is not None else RegionRoads(region_id=local.region_id, segments=[]))

    t_idx, t_pos, t_pop, c_idx, c_pos, c_pop = _local_settlements(local, rivers, globe, land_reach, parent_spacing)
    # One village stream per (world, region, hex); non-negative for SeedSequence.
    seed = (int(globe.seed or 0) & 0xFFFFFFFF) * 1_000_003 + int(region.region_id) * 4_099 + int(hex_index)
    villages = place_villages(
        local, rivers, roads, np.concatenate([t_pos, c_pos]), np.concatenate([t_pop, c_pop]), seed, params.villages,
    )
    return LocalMap(
        fields=local, rivers=rivers, roads=roads, villages=villages,
        town_index=t_idx, town_positions=t_pos, city_index=c_idx, city_positions=c_pos, deltas=deltas,
    )
