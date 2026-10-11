"""A settlement's site: what the tiers above give it at street scale, as vectors.

:func:`build_site` derives it from the settlement map's terrain and local maps:
the frame (:mod:`settlegen.frame`), the water (:mod:`settlegen.water`), the
footprint and buildable ground (:mod:`settlegen.footprint`), and the roads with
their gates, landings and crossings (:mod:`settlegen.roads`).  Anchor
candidates and the hinterland follow (plan phase 4).  Qt-free.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from shapely.geometry import MultiPolygon

from region.upsample import RegionFields
from settlegen.footprint import SettlementFootprintParams, build_footprint, buildable_ground
from settlegen.frame import Frame
from settlegen.geometry import SettlementGeometry
from settlegen.roads import SettlementRoadParams, SiteRoads, build_site_roads
from settlegen.water import SettlementRiverParams, SiteWater, build_site_water

if TYPE_CHECKING:
    from core.world import World
    from settlegen.chain import SettlementMaps
    from worldgen.cost.builder import LandCostParams


@dataclass
class SettlementSite:
    """The site, in frame metres."""

    frame: Frame
    water: SiteWater
    footprint: MultiPolygon
    buildable: MultiPolygon
    roads: SiteRoads


def build_site(
    globe: "World", fields: RegionFields, chain: "SettlementMaps", rivers: Optional[SettlementRiverParams] = None,
    roads: Optional[SettlementRoadParams] = None, footprint: Optional[SettlementFootprintParams] = None, *,
    local_road_raster_texels: float = 2.0, land: Optional["LandCostParams"] = None, danger_scale: float = 2.0,
) -> SettlementSite:
    """The site of ``chain``'s settlement map, with terrain ``fields``."""
    geometry = fields.geometry
    assert isinstance(geometry, SettlementGeometry)
    frame = Frame(geometry, float(chain.side_km) * 1000.0)
    maps = chain.maps
    water = build_site_water(globe, fields, frame, maps, rivers)
    built = build_footprint(
        fields, frame, maps, water, chain.ref, math.pi * float(chain.radius_km) ** 2, int(globe.seed or 0), footprint,
    )
    site_roads = build_site_roads(
        globe, fields, frame, maps, water, built.polygon, roads, always_bridge=chain.ref.kind == "city",
        local_road_raster_texels=local_road_raster_texels, land=land, danger_scale=danger_scale,
    )
    return SettlementSite(
        frame=frame, water=water, footprint=built.polygon, buildable=buildable_ground(fields, frame, water, footprint),
        roads=site_roads,
    )
