"""A settlement's site: what the tiers above give it at street scale, as vectors.

:func:`build_site` derives it from the settlement map's terrain and local maps:
the frame (:mod:`settlegen.frame`), the water (:mod:`settlegen.water`), the
footprint, and the roads with their gates, landings and crossings
(:mod:`settlegen.roads`).  The footprint is for now a disc of the local
footprint's area (towns and villages: their people at the built-up density) on
dry ground; the buildable ground, anchor candidates and the hinterland follow
(plan phase 4).  Qt-free.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence

from shapely.geometry import MultiPolygon, Point

from localmap.builder import LocalMap
from region.upsample import RegionFields
from settlegen.frame import Frame
from settlegen.geometry import SettlementGeometry
from settlegen.roads import SettlementRoadParams, SiteRoads, build_site_roads
from settlegen.water import SettlementRiverParams, SiteWater, _as_multipolygon, build_site_water

if TYPE_CHECKING:
    from core.world import World
    from worldgen.cost.builder import LandCostParams


@dataclass
class SettlementSite:
    """The site, in frame metres."""

    frame: Frame
    water: SiteWater
    footprint: MultiPolygon
    roads: SiteRoads


def disc_footprint(radius_m: float, water: SiteWater, frame: Frame) -> MultiPolygon:
    """A disc of ``radius_m`` at the frame's centre, less the still water (a city spans its rivers)."""
    disc = Point(0.0, 0.0).buffer(float(radius_m), quad_segs=32).intersection(frame.box())
    still = water.sea.union(water.lakes)
    return _as_multipolygon(disc.difference(still) if not still.is_empty else disc)


def build_site(
    globe: "World", fields: RegionFields, side_km: float, radius_km: float, maps: Sequence[LocalMap],
    rivers: Optional[SettlementRiverParams] = None, roads: Optional[SettlementRoadParams] = None, *,
    city: bool = False, local_road_raster_texels: float = 2.0, land: Optional["LandCostParams"] = None,
    danger_scale: float = 2.0,
) -> SettlementSite:
    """The site of the settlement map with terrain ``fields`` over ``maps`` (owning map first)."""
    geometry = fields.geometry
    assert isinstance(geometry, SettlementGeometry)
    frame = Frame(geometry, float(side_km) * 1000.0)
    water = build_site_water(globe, fields, frame, maps, rivers)
    footprint = disc_footprint(float(radius_km) * 1000.0, water, frame)
    site_roads = build_site_roads(
        globe, fields, frame, maps, water, footprint, roads, always_bridge=city,
        local_road_raster_texels=local_road_raster_texels, land=land, danger_scale=danger_scale,
    )
    return SettlementSite(frame=frame, water=water, footprint=footprint, roads=site_roads)
