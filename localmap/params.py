"""All local-map tuning, grouped for the params dialog."""
from __future__ import annotations

from core.params import field, paramclass
from localmap.cities import CityParams
from localmap.rivers import LocalRiverParams
from localmap.roads import LocalRoadParams
from localmap.upsample import LocalUpsampleParams
from localmap.villages import VillageParams


@paramclass
class LocalParams:
    """Local-tier params: the field refinement, the local rivers and roads, and
    the villages and the cities' footprints.  None of it touches the world build or the region."""

    upsample: LocalUpsampleParams = field(
        default_factory=LocalUpsampleParams, metadata={"label": "Detail refinement"}
    )
    rivers: LocalRiverParams = field(default_factory=LocalRiverParams, metadata={"label": "Rivers"})
    roads: LocalRoadParams = field(default_factory=LocalRoadParams, metadata={"label": "Roads"})
    villages: VillageParams = field(default_factory=VillageParams, metadata={"label": "Villages"})
    cities: CityParams = field(default_factory=CityParams, metadata={"label": "Cities"})
