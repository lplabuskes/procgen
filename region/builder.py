"""One region map, end to end: :func:`build_region_map` runs, over one region cell:

1. fields (:func:`region.upsample.build_region_fields`);
2. rivers (:func:`region.rivers.build_region_rivers`), and riparian wetland
   along them (:func:`region.wetlands.add_riparian_wetland`);
3. this region's cities re-sited (:func:`region.cities.refine_region_cities`, saved on the world);
4. roads (:func:`region.roads.build_region_roads`), ending on the re-sited cities;
5. the world's deltas snapped onto the re-routed mouths (:func:`region.rivers.snap_deltas`);
6. this region's towns re-sited (:func:`region.towns.refine_region_towns`, saved on the world).

Steps 2–6 are :func:`refine_region_features`, which re-runs on existing fields
when only feature params change.  The region's streams, drawn only on its local
maps, are built on demand (:meth:`RegionMap.streams`).

Qt-free.  Deterministic, given which settlements earlier regions re-sited.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

from region.cities import refine_region_cities
from region.params import RegionParams
from region.rivers import RegionRivers, RiverPath, build_region_rivers, snap_deltas
from region.roads import RegionRoads, build_region_roads
from region.streams import RegionStreamParams, build_region_streams
from region.towns import refine_region_towns
from region.upsample import RegionFields, build_region_fields
from region.wetlands import add_riparian_wetland

if TYPE_CHECKING:
    from core.world import World
    from realmgen.population.builder import PopulationParams
    from realmgen.towns.builder import TownParams
    from worldgen.cost.builder import LandCostParams
    from worldgen.resources.builder import ResourcesParams


@dataclass
class RegionMap:
    """Everything one region map holds."""

    fields: RegionFields
    rivers: RegionRivers
    roads: RegionRoads
    # World deltas moved onto the re-routed rivers' mouths.
    deltas: Optional[np.ndarray] = None
    # (params, streams) below the world's network; built lazily, for local maps only.
    _streams: Optional[tuple[RegionStreamParams, list[RiverPath]]] = field(default=None, repr=False)

    @property
    def region_id(self) -> int:
        return self.fields.region_id

    def streams(self, globe: "World", params: RegionStreamParams) -> list[RiverPath]:
        """This region's streams for its local maps, cached until ``params`` change."""
        if self._streams is None or self._streams[0] != params:
            built = build_region_streams(self.fields, globe, self.rivers, params)
            self._streams = (dataclasses.replace(params), built)
        return self._streams[1]


def land_reach(fields: RegionFields, water_spread_texels: int) -> float:
    """How far (chord) a settlement on refined water is moved onto land: the
    refinement's water spread plus two texels of slack."""
    return (int(water_spread_texels) + 2) * fields.grid.texel_spacing_rad(fields.tex_size)


def refine_region_features(
    globe: "World",
    fields: RegionFields,
    params: Optional[RegionParams] = None,
    *,
    land: Optional["LandCostParams"] = None,
    danger_scale: float = 2.0,
    town_params: Optional["TownParams"] = None,
    population_params: Optional["PopulationParams"] = None,
    river_threshold_factor: float = 1.5,
) -> RegionMap:
    """Steps 2–6 of :func:`build_region_map` over existing ``fields``.

    Re-sites this region's towns and cities in ``globe``'s features, in place,
    and adds riparian wetland to ``fields``.
    """
    params = params or RegionParams()
    rivers = build_region_rivers(fields, globe, params.rivers)
    add_riparian_wetland(fields, rivers, params.upsample)
    refine_region_cities(fields, rivers, globe, river_threshold_factor, params.settlements)
    cities = globe.get_feature("settlements") or []
    moved = (np.array([c.site_position for c in cities]).reshape(-1, 3),
             np.array([c.position for c in cities]).reshape(-1, 3))
    reach = land_reach(fields, params.upsample.water_spread_texels)
    roads = build_region_roads(
        fields, globe, rivers, params.roads, land=land, danger_scale=danger_scale, land_reach=reach, moved=moved,
    )
    deltas = globe.get_feature("deltas")
    snapped = None
    if deltas is not None and len(deltas):
        # A re-routed stem ends up to two corridors from its world mouth.
        texel = float(np.sqrt(np.mean(globe.mesh.face_areas))) / max(
            globe.mesh.face_textures["elevation"].shape[2] - 1, 1)
        snapped = snap_deltas(
            np.asarray(deltas), rivers, 2.0 * float(params.rivers.corridor_global_texels) * texel + 2.0 * reach)
    refine_region_towns(
        fields, rivers, globe, town_params, population_params, river_threshold_factor, params.settlements,
        roads=roads,
    )
    return RegionMap(fields=fields, rivers=rivers, roads=roads, deltas=snapped)


def build_region_map(
    globe: "World",
    region_id: int,
    params: Optional[RegionParams] = None,
    *,
    resources: Optional["ResourcesParams"] = None,
    fertility_exponent: float = 2.0,
    land: Optional["LandCostParams"] = None,
    danger_scale: float = 2.0,
    town_params: Optional["TownParams"] = None,
    population_params: Optional["PopulationParams"] = None,
    river_threshold_factor: float = 1.5,
) -> RegionMap:
    """Build the region map of cell ``region_id``.  See the module docstring.

    Re-sites this region's towns and cities in ``globe``'s features, in place.
    """
    params = params or RegionParams()
    fields = build_region_fields(
        globe, region_id, params.upsample, resources=resources, fertility_exponent=fertility_exponent,
    )
    return refine_region_features(
        globe, fields, params, land=land, danger_scale=danger_scale, town_params=town_params,
        population_params=population_params, river_threshold_factor=river_threshold_factor,
    )
