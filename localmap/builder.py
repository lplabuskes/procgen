"""One local map, end to end: :func:`build_local_map` runs, over one regional hex:

1. fields (:func:`localmap.upsample.build_local_fields`);
2. rivers and the region's streams (:func:`localmap.rivers.build_local_rivers`),
   with the region's deltas snapped onto the local mouths, and riparian wetland along them
   (:func:`region.wetlands.add_riparian_wetland`);
3. this map's cities re-sited (:func:`localmap.settlements.refine_local_cities`, saved on the world);
4. roads (:func:`localmap.roads.build_local_roads`), ending on the re-sited cities;
5. this map's towns re-sited (:func:`localmap.settlements.refine_local_towns`, saved on the world);
   other maps' towns and cities are drawn where they stand, moved off the local water;
6. villages (:func:`localmap.villages.place_villages`);
7. city footprints (:func:`localmap.cities.city_footprints`);
8. farmland (:func:`localmap.farmland.place_farmland`).

Rivers are drawn at their physical width (:mod:`localmap.banks`): towns, cities
and villages are shown just off the channel (display only: the world keeps its
positions; roads follow their cities), and steps 6–8 skip the texels a channel covers.

Qt-free.  Deterministic, given which settlements earlier local maps re-sited.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

from localmap.banks import covered_texels, onto_banks, river_channels
from localmap.cities import CityFootprints, city_footprints
from localmap.farmland import Farmland, place_farmland
from localmap.params import LocalParams
from localmap.rivers import build_local_rivers
from localmap.roads import build_local_roads
from localmap.settlements import refine_local_cities, refine_local_towns
from localmap.upsample import build_local_fields, path_segments
from localmap.villages import VILLAGE_MINING, Villages, place_villages
from realmgen.towns.types import TOWN_MINING
from region.display import snap_to_land
from region.rivers import RegionRivers, RiverPath, snap_deltas
from region.wetlands import add_riparian_wetland
from region.roads import RegionRoads, move_points
from region.upsample import RegionFields
from worldgen.hydrology.discharge import discharge_per_flow_unit

if TYPE_CHECKING:
    from core.world import World
    from realmgen.population.builder import PopulationParams
    from realmgen.towns.builder import TownParams
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
    city_footprints: Optional[CityFootprints] = None
    farmland: Optional[Farmland] = None

    @property
    def hex_index(self) -> int:
        return int(getattr(self.fields.geometry, "hex_index"))


def _in_box(local: RegionFields, positions: np.ndarray, margin: float) -> np.ndarray:
    if len(positions) == 0:
        return np.zeros(0, dtype=bool)
    s, t, valid = local.geometry.ecef_to_local(positions)
    return valid & (s >= -margin) & (s <= 1.0 + margin) & (t >= -margin) & (t <= 1.0 + margin)


def _local_settlements(
    local: RegionFields, globe: "World", land_reach: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(town_index, town_positions, town_pops, city_index, city_positions, city_pops)``
    in and around the map, moved off the local water (another map's may not be refined yet)."""
    towns = globe.get_feature("towns")
    t_idx = np.zeros(0, dtype=np.int64)
    t_pos, t_pop = np.zeros((0, 3)), np.zeros(0)
    if towns is not None and len(towns):
        t_idx = np.flatnonzero(_in_box(local, np.asarray(towns.positions, dtype=np.float64), _SETTLEMENT_MARGIN))
        t_pos = snap_to_land(local, np.asarray(towns.positions, dtype=np.float64)[t_idx], land_reach)
        t_pop = np.asarray(towns.population, dtype=np.float64)[t_idx]
    settlements = globe.get_feature("settlements") or []
    c_idx = np.zeros(0, dtype=np.int64)
    c_pos, c_pop = np.zeros((0, 3)), np.zeros(0)
    if settlements:
        all_pos = np.array([np.asarray(s.position, dtype=np.float64) for s in settlements]).reshape(-1, 3)
        c_idx = np.flatnonzero(_in_box(local, all_pos, _SETTLEMENT_MARGIN))
        c_pos = snap_to_land(local, all_pos[c_idx], land_reach)
        c_pop = np.array([float(settlements[i].population) for i in c_idx])
    return t_idx, t_pos, t_pop, c_idx, c_pos, c_pop


def _move_road_ends(roads: RegionRoads, moved: tuple[np.ndarray, np.ndarray]) -> RegionRoads:
    """``roads`` with ends at an old position of ``moved = (old, new)`` at its new one."""
    def ends(a, b):
        pts = move_points(np.array([a, b], dtype=np.float64), moved).astype(np.float32)
        return pts[0], pts[1]

    sea = [(*ends(a, b), f) for a, b, f in roads.sea]
    segments = list(sea)
    polylines = []
    for verts, flows in zip(roads.polylines, roads.polyline_flows):
        verts = np.asarray(verts, dtype=np.float64).copy()
        verts[[0, -1]] = move_points(verts[[0, -1]], moved)
        polylines.append(verts)
        segments.extend((verts[k].astype(np.float32), verts[k + 1].astype(np.float32), float(flows[k]))
                        for k in range(len(verts) - 1))
    return dataclasses.replace(roads, segments=segments, polylines=polylines, sea=sea)


def _town_kinds(globe: "World", t_idx: np.ndarray) -> np.ndarray:
    towns = globe.get_feature("towns")
    return np.asarray(towns.kind)[t_idx] if towns is not None and len(towns) else np.zeros(0, dtype=np.int8)


def build_local_map(
    globe: "World",
    region: RegionFields,
    hex_index: int,
    params: Optional[LocalParams] = None,
    *,
    region_rivers: Optional[RegionRivers] = None,
    region_roads: Optional[RegionRoads] = None,
    region_deltas: Optional[np.ndarray] = None,
    region_streams: Optional[list[RiverPath]] = None,
    resources: Optional["ResourcesParams"] = None,
    fertility_exponent: float = 2.0,
    land: Optional["LandCostParams"] = None,
    danger_scale: float = 2.0,
    town_params: Optional["TownParams"] = None,
    population_params: Optional["PopulationParams"] = None,
    river_threshold_factor: float = 1.5,
) -> LocalMap:
    """Build the local map of regional hex ``hex_index``.  See the module docstring.

    ``region_rivers`` / ``region_roads`` / ``region_deltas`` / ``region_streams``
    are the region's (streams from :func:`region.streams.build_region_streams`);
    without them the map has none.  Re-sites this map's towns and cities in
    ``globe``'s features, in place.
    """
    params = params or LocalParams()
    local = build_local_fields(
        globe, region, hex_index, params.upsample, resources=resources, fertility_exponent=fertility_exponent,
        drains=(path_segments(list(region_rivers.paths) + list(region_streams or []))
                if region_rivers is not None else None),
    )
    spacing = local.grid.texel_spacing_rad(local.tex_size)
    parent_spacing = region.grid.texel_spacing_rad(region.tex_size)
    # How far a settlement on local water moves onto land: the water spread plus two texels.
    land_reach = (int(params.upsample.water_spread_texels) + 2) * spacing

    rivers = (build_local_rivers(local, region, region_rivers, globe, params.rivers, region_streams)
              if region_rivers is not None else RegionRivers(region_id=local.region_id, paths=[]))
    add_riparian_wetland(local, rivers, params.upsample)
    deltas = None
    if region_deltas is not None and len(region_deltas):
        inside = _in_box(local, np.asarray(region_deltas, dtype=np.float64), _SETTLEMENT_MARGIN)
        reach = 2.0 * float(params.rivers.corridor_parent_texels) * parent_spacing + 2.0 * land_reach
        deltas = snap_deltas(np.asarray(region_deltas, dtype=np.float64)[inside], rivers, reach)
    moved = refine_local_cities(
        local, region, hex_index, rivers, globe, params.settlements, river_threshold_factor)
    roads = (build_local_roads(local, region, region_roads, globe, rivers, params.roads,
                               land=land, danger_scale=danger_scale, land_reach=land_reach, moved=moved)
             if region_roads is not None else RegionRoads(region_id=local.region_id, segments=[]))
    refine_local_towns(
        local, region, hex_index, rivers, globe, params.settlements,
        town_params, population_params, river_threshold_factor, roads,
    )

    t_idx, t_pos, t_pop, c_idx, c_pos, c_pop = _local_settlements(local, globe, land_reach)
    channels = river_channels(rivers, discharge_per_flow_unit(globe.mesh))
    covered = covered_texels(local, channels)
    clearance = float(params.settlements.bank_clearance_km)
    t_pos = onto_banks(channels, t_pos, clearance)
    c_banked = onto_banks(channels, c_pos, clearance)
    shifted = np.any(c_banked != c_pos, axis=1)
    if shifted.any():
        roads = _move_road_ends(roads, (c_pos[shifted], c_banked[shifted]))
    c_pos = c_banked
    # One village stream per (world, region, hex); non-negative for SeedSequence.
    seed = (int(globe.seed or 0) & 0xFFFFFFFF) * 1_000_003 + int(region.region_id) * 4_099 + int(hex_index)
    villages = place_villages(
        local, rivers, roads, np.concatenate([t_pos, c_pos]), np.concatenate([t_pop, c_pop]), seed, params.villages,
        river_covered=covered,
    )
    villages.positions = onto_banks(channels, villages.positions, clearance)
    footprints = city_footprints(local, c_pos, c_pop, seed, params.cities, river_covered=covered)
    farmland = place_farmland(
        local, villages.positions, villages.population, t_pos, t_pop, footprints, seed, params.farmland,
        village_mining=villages.kind == VILLAGE_MINING,
        town_mining=_town_kinds(globe, t_idx) == TOWN_MINING,
        channels=channels, river_covered=covered,
    )
    return LocalMap(
        fields=local, rivers=rivers, roads=roads, villages=villages,
        town_index=t_idx, town_positions=t_pos, city_index=c_idx, city_positions=c_pos, deltas=deltas,
        city_footprints=footprints, farmland=farmland,
    )
