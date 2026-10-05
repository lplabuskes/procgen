"""All regional-map tuning, grouped for the params dialog."""
from __future__ import annotations

from core.params import field, paramclass
from region.rivers import RegionRiverParams
from region.roads import RegionRoadParams
from region.towns import RegionSettlementParams
from region.upsample import RegionUpsampleParams


@paramclass
class RegionParams:
    """Region-tier params (fields, rivers, roads, settlements); none touches the world build.

    Changing them re-refines the region and resets refined towns and cities to
    their stage sites.
    """

    upsample: RegionUpsampleParams = field(
        default_factory=RegionUpsampleParams, metadata={"label": "Detail refinement"}
    )
    rivers: RegionRiverParams = field(
        default_factory=RegionRiverParams, metadata={"label": "Rivers"}
    )
    roads: RegionRoadParams = field(
        default_factory=RegionRoadParams, metadata={"label": "Roads"}
    )
    settlements: RegionSettlementParams = field(
        default_factory=RegionSettlementParams, metadata={"label": "Towns and cities"}
    )
