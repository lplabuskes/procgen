"""A settlement's site: what the tiers above give it at street scale, as vectors.

:func:`build_site` derives it from the settlement map's terrain and local maps.
So far: the frame (:mod:`settlegen.frame`) and the water (:mod:`settlegen.water`);
roads and gates, the footprint, the buildable ground, anchor candidates and the
hinterland follow (plan phase 4).  Qt-free.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence

from localmap.builder import LocalMap
from region.upsample import RegionFields
from settlegen.frame import Frame
from settlegen.geometry import SettlementGeometry
from settlegen.water import SettlementRiverParams, SiteWater, build_site_water

if TYPE_CHECKING:
    from core.world import World


@dataclass
class SettlementSite:
    """The site, in frame metres."""

    frame: Frame
    water: SiteWater


def build_site(
    globe: "World", fields: RegionFields, side_km: float, maps: Sequence[LocalMap],
    rivers: Optional[SettlementRiverParams] = None,
) -> SettlementSite:
    """The site of the settlement map with terrain ``fields`` over ``maps`` (owning map first)."""
    geometry = fields.geometry
    assert isinstance(geometry, SettlementGeometry)
    frame = Frame(geometry, float(side_km) * 1000.0)
    return SettlementSite(frame=frame, water=build_site_water(globe, fields, frame, maps, rivers))
