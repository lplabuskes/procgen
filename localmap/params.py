"""All local-map tuning, grouped for the params dialog."""
from __future__ import annotations

from core.params import field, paramclass
from localmap.cities import CityParams
from localmap.rivers import LocalRiverParams
from localmap.roads import LocalRoadParams
from localmap.settlements import LocalSettlementParams
from localmap.upsample import LocalUpsampleParams
from localmap.villages import VillageParams


@paramclass
class LocalParams:
    """Local-tier params: the field refinement, the local rivers and roads, town
    and city refinement, and the villages and the cities' footprints.  None of it
    touches the world build or the region; changing it puts every town and city
    back where its region put it, to be refined again by its local map."""

    upsample: LocalUpsampleParams = field(
        default_factory=LocalUpsampleParams, metadata={"label": "Detail refinement"}
    )
    rivers: LocalRiverParams = field(default_factory=LocalRiverParams, metadata={"label": "Rivers"})
    roads: LocalRoadParams = field(default_factory=LocalRoadParams, metadata={"label": "Roads"})
    settlements: LocalSettlementParams = field(
        default_factory=LocalSettlementParams, metadata={"label": "Towns and cities"})
    villages: VillageParams = field(default_factory=VillageParams, metadata={"label": "Villages"})
    cities: CityParams = field(default_factory=CityParams, metadata={"label": "Cities"})
