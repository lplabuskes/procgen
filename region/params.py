"""All regional-map tuning, grouped for the params dialog."""
from __future__ import annotations

from core.params import field, paramclass
from region.rivers import RegionRiverParams
from region.towns import RegionTownParams
from region.upsample import RegionUpsampleParams


@paramclass
class RegionParams:
    """Region-tier params: the field refinement, the regional rivers, and town
    refinement.

    None of it touches the world build. Changing it re-refines the region on
    screen and puts every refined town back on its stage site, to be refined
    again when its region is next visited.
    """

    upsample: RegionUpsampleParams = field(
        default_factory=RegionUpsampleParams, metadata={"label": "Detail refinement"}
    )
    rivers: RegionRiverParams = field(
        default_factory=RegionRiverParams, metadata={"label": "Rivers"}
    )
    towns: RegionTownParams = field(
        default_factory=RegionTownParams, metadata={"label": "Towns"}
    )
