"""All settlement-map tuning, grouped for the params dialog (advanced only)."""
from __future__ import annotations

from core.params import field, paramclass
from settlegen.roads import SettlementRoadParams
from settlegen.upsample import SettlementUpsampleParams
from settlegen.water import SettlementRiverParams


@paramclass
class SettlementMapParams:
    """Settlement-map params.  None of it touches the world, the region or the
    local maps; changing it only rebuilds the settlement map on screen."""

    upsample: SettlementUpsampleParams = field(
        default_factory=SettlementUpsampleParams, metadata={"label": "Terrain refinement"})
    rivers: SettlementRiverParams = field(default_factory=SettlementRiverParams, metadata={"label": "Rivers"})
    roads: SettlementRoadParams = field(default_factory=SettlementRoadParams, metadata={"label": "Roads"})
