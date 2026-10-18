"""All settlement-map tuning, grouped for the params dialog (advanced only)."""
from __future__ import annotations

from core.params import field, paramclass
from settlegen.site.params import SettlementAnchorParams
from settlegen.site.params import SettlementFootprintParams
from settlegen.layout.params import SettlementLayoutParams
from settlegen.profile.builder import SettlementProfileParams
from settlegen.site.params import SettlementRoadParams
from settlegen.terrain.builder import SettlementUpsampleParams
from settlegen.site.params import SettlementRiverParams


@paramclass
class SettlementMapParams:
    """Settlement-map params.  None of it touches the world, the region or the
    local maps; changing it only rebuilds the settlement map on screen."""

    upsample: SettlementUpsampleParams = field(
        default_factory=SettlementUpsampleParams, metadata={"label": "Terrain refinement"})
    rivers: SettlementRiverParams = field(default_factory=SettlementRiverParams, metadata={"label": "Rivers"})
    roads: SettlementRoadParams = field(default_factory=SettlementRoadParams, metadata={"label": "Roads"})
    footprint: SettlementFootprintParams = field(
        default_factory=SettlementFootprintParams, metadata={"label": "Footprint"})
    anchors: SettlementAnchorParams = field(
        default_factory=SettlementAnchorParams, metadata={"label": "Anchor candidates"})
    profile: SettlementProfileParams = field(
        default_factory=SettlementProfileParams, metadata={"label": "Profile"})
    layout: SettlementLayoutParams = field(
        default_factory=SettlementLayoutParams, metadata={"label": "Layout"})
