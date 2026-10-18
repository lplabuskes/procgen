"""What a site holds: its water and roads, in frame metres."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import shapely
from shapely.geometry import MultiLineString, MultiPolygon, Polygon

from region.rivers import RegionRivers
from region.roads import RegionRoads


@dataclass
class RiverLine:
    """One re-traced river or stream on the site, in frame metres."""

    xy: np.ndarray            # (K, 2)
    flows: np.ndarray         # (K - 1,) world flow per edge
    width_m: np.ndarray       # (K - 1,) bankfull width per edge
    in_water: np.ndarray      # (K - 1,) bool: under a lake or the sea (not drawn)
    minor: bool               # a stream, not a world river


@dataclass
class SiteWater:
    """The site's water.  ``rivers`` are the re-traced paths on the sphere (for routing roads);
    ``channels`` reach as far past the frame as they do (:func:`natural_channels`)."""

    sea: MultiPolygon
    lakes: MultiPolygon
    shorelines: MultiLineString
    lines: list[RiverLine]
    channels: MultiPolygon
    rivers: RegionRivers = field(repr=False)

    @property
    def bodies(self) -> MultiPolygon:
        """Sea, lakes and river channels: all the site's water."""
        merged = shapely.union_all([self.sea, self.lakes, self.channels])
        return _as_multipolygon(merged)


def _as_multipolygon(geom) -> MultiPolygon:  # type: ignore[no-untyped-def]
    if isinstance(geom, MultiPolygon):
        return geom
    if isinstance(geom, Polygon):
        return MultiPolygon([geom]) if not geom.is_empty else MultiPolygon()
    return MultiPolygon([g for g in getattr(geom, "geoms", []) if isinstance(g, Polygon) and not g.is_empty])


@dataclass
class RoadLine:
    """One re-traced road on the site, in frame metres."""

    xy: np.ndarray            # (K, 2)
    flows: np.ndarray         # (K - 1,) trade flow per edge


@dataclass
class Gate:
    """Where a road crosses the footprint's edge."""

    xy: np.ndarray            # (2,)
    direction: np.ndarray     # (2,) unit, along the road, outwards
    flow: float
    road: int                 # index into ``SiteRoads.lines``


@dataclass
class Landing:
    """Where a water leg leaves the land: a quay."""

    xy: np.ndarray            # (2,)
    direction: np.ndarray     # (2,) unit, onto the water
    flow: float
    water: Literal["sea", "lake", "river"]


@dataclass
class Crossing:
    """Where a road crosses a river or stream."""

    xy: np.ndarray            # (2,)
    road: int                 # index into ``SiteRoads.lines``
    river: int                # index into ``SiteWater.lines``
    width_m: float            # the channel's bankfull width there
    flow: float               # the road's
    bridge: bool              # else a ford


@dataclass
class SiteRoads:
    """The site's roads.  ``roads`` are the re-traced roads and river and lake legs
    on the sphere, with the sea legs from their landings (for drawing in the world's form)."""

    lines: list[RoadLine]
    water: list[RoadLine]                               # river and lake legs
    sea: list[tuple[np.ndarray, np.ndarray, float]]     # (a xy, b xy, flow): a at the landing if it has one
    landings: list[Landing]
    crossings: list[Crossing]
    gates: list[Gate]
    roads: RegionRoads = field(repr=False)
