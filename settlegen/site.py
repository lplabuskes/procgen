"""A settlement's site: what the tiers above give it at street scale, as vectors.

:func:`build_site` derives it from the settlement map's terrain and local maps:
the frame (:mod:`settlegen.frame`), the water (:mod:`settlegen.water`), the
roads with their landings and crossings (:mod:`settlegen.roads`), the
footprint and buildable ground (:mod:`settlegen.footprint`), the gates where
the roads leave the footprint, the anchor candidates (:mod:`settlegen.anchors`)
and the hinterland (:mod:`settlegen.hinterland`).  Qt-free.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from shapely.geometry import MultiPolygon

from localmap.farmland import FarmlandParams
from localmap.villages import VILLAGE_MINING
from realmgen.towns.types import TOWN_MINING
from region.upsample import RegionFields
from settlegen.anchors import AnchorCandidates, SettlementAnchorParams, build_anchor_candidates
from settlegen.footprint import SettlementFootprintParams, build_footprint, buildable_ground
from settlegen.frame import Frame
from settlegen.geometry import SettlementGeometry
from settlegen.hinterland import Hinterland, build_hinterland, village_farming
from settlegen.roads import SettlementRoadParams, SiteRoads, build_site_roads, set_gates
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
    density_per_ha: float     # people per hectare of densely built footprint
    sparse: MultiPolygon      # the footprint's sparsely built part (its filled holes)
    buildable: MultiPolygon
    roads: SiteRoads
    anchors: AnchorCandidates
    hinterland: Hinterland


def build_site(
    globe: "World", fields: RegionFields, chain: "SettlementMaps", rivers: Optional[SettlementRiverParams] = None,
    roads: Optional[SettlementRoadParams] = None, footprint: Optional[SettlementFootprintParams] = None,
    anchors: Optional[SettlementAnchorParams] = None, *, farmland: Optional[FarmlandParams] = None,
    local_road_raster_texels: float = 2.0, land: Optional["LandCostParams"] = None, danger_scale: float = 2.0,
) -> SettlementSite:
    """The site of ``chain``'s settlement map, with terrain ``fields``: water, then roads,
    then the footprint (it grows along the roads), then the gates on it, the anchor
    candidates and the hinterland (``farmland``: the local tier's rules)."""
    geometry = fields.geometry
    assert isinstance(geometry, SettlementGeometry)
    roads = roads or SettlementRoadParams()
    frame = Frame(geometry, float(chain.side_km) * 1000.0)
    maps = chain.maps
    water = build_site_water(globe, fields, frame, maps, rivers)
    site_roads = build_site_roads(
        globe, fields, frame, maps, water, roads, always_bridge=chain.ref.kind == "city",
        local_road_raster_texels=local_road_raster_texels, land=land, danger_scale=danger_scale,
    )
    built = build_footprint(
        fields, frame, water, chain.ref, chain.population, chain.density_per_ha, int(globe.seed or 0),
        [line.xy for line in site_roads.lines], footprint,
    )
    set_gates(site_roads, built.polygon, roads, frame, fields)
    hinterland = build_hinterland(
        globe, fields, frame, water, built.polygon, maps, chain.keys, chain.ref, chain.population,
        mining=_is_mining(globe, chain), params=farmland, farming_share=_farming_share(chain),
    )
    return SettlementSite(
        frame=frame, water=water, footprint=built.polygon, density_per_ha=built.density_per_ha, sparse=built.sparse,
        buildable=buildable_ground(fields, frame, water, footprint), roads=site_roads,
        anchors=build_anchor_candidates(fields, frame, water, site_roads, built.polygon, anchors),
        hinterland=hinterland,
    )


def _farming_share(chain: "SettlementMaps") -> float:
    """A village's people who farm (1 for anything else: towns have their own rule)."""
    ref = chain.ref
    return float(village_farming(chain.owner.villages)[ref.index]) if ref.kind == "village" else 1.0


def _is_mining(globe: "World", chain: "SettlementMaps") -> bool:
    """A mining town or village (they farm little)."""
    ref = chain.ref
    if ref.kind == "village":
        return int(chain.owner.villages.kind[ref.index]) == VILLAGE_MINING
    if ref.kind == "town":
        return int(globe.get_feature("towns").kind[ref.index]) == TOWN_MINING
    return False
