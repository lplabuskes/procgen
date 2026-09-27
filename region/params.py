"""All regional-map tuning, grouped for the params dialog."""
from __future__ import annotations

from core.params import field, paramclass
from region.rivers import RegionRiverParams
from region.upsample import RegionUpsampleParams


@paramclass
class RegionParams:
    """Region-tier params: the field refinement and the regional rivers.

    None of it touches the world build — changing it only re-refines the region
    on screen.
    """

    upsample: RegionUpsampleParams = field(
        default_factory=RegionUpsampleParams, metadata={"label": "Detail refinement"}
    )
    rivers: RegionRiverParams = field(
        default_factory=RegionRiverParams, metadata={"label": "Rivers"}
    )
