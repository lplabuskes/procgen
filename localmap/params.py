"""All local-map tuning, grouped for the params dialog."""
from __future__ import annotations

from core.params import field, paramclass
from localmap.cities import CityParams
from localmap.farmland import FarmlandParams
from localmap.rivers import LocalRiverParams
from localmap.roads import LocalRoadParams
from localmap.settlements import LocalSettlementParams
from localmap.upsample import LocalUpsampleParams
from localmap.villages import VillageParams
from region.streams import RegionStreamParams


@paramclass
class LocalParams:
    """Local-tier params (``streams`` are built per region but drawn only on local maps).

    None of it touches the world build or the region; changing it resets every
    town and city to its region position, to be refined again locally."""

    upsample: LocalUpsampleParams = field(
        default_factory=LocalUpsampleParams, metadata={"label": "Detail refinement"}
    )
    streams: RegionStreamParams = field(
        default_factory=RegionStreamParams, metadata={"label": "Streams (regional network)"})
    rivers: LocalRiverParams = field(default_factory=LocalRiverParams, metadata={"label": "Rivers"})
    roads: LocalRoadParams = field(default_factory=LocalRoadParams, metadata={"label": "Roads"})
    settlements: LocalSettlementParams = field(
        default_factory=LocalSettlementParams, metadata={"label": "Towns and cities"})
    villages: VillageParams = field(default_factory=VillageParams, metadata={"label": "Villages"})
    cities: CityParams = field(default_factory=CityParams, metadata={"label": "Cities"})
    farmland: FarmlandParams = field(default_factory=FarmlandParams, metadata={"label": "Farmland"})
