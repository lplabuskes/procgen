"""One settlement map, end to end: :func:`build_settlement_map` runs

1. the chain to its local maps (:func:`settlegen.chain.ensure_local_maps`, through
   the shared :class:`localmap.store.MapStore`: built only if not held);
2. the frame's terrain (:func:`settlegen.upsample.build_settlement_fields`);
3. the site, as vectors (:func:`settlegen.site.build_site`).

Profile and layout (plan phases 5–6) will follow here.  Qt-free.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from region.upsample import RegionFields
from settlegen.chain import SettlementMaps, SettlementRef, ensure_local_maps
from settlegen.params import SettlementMapParams
from settlegen.site import SettlementSite, build_site
from settlegen.upsample import build_settlement_fields

if TYPE_CHECKING:
    from localmap.store import MapStore


@dataclass
class SettlementMap:
    """Everything one settlement map holds (so far: its local maps, terrain and site)."""

    chain: SettlementMaps
    fields: RegionFields
    site: SettlementSite

    @property
    def ref(self) -> SettlementRef:
        return self.chain.ref


def build_settlement_map(
    store: "MapStore", ref: SettlementRef, params: Optional[SettlementMapParams] = None,
) -> SettlementMap:
    """Build ``ref``'s settlement map from ``store``'s local maps.  ``params.upsample``
    is expected to carry the local tier's shared detail (:func:`settlegen.upsample.with_local_detail`)."""
    params = params or SettlementMapParams()
    globe = store.globe
    if globe is None:
        raise ValueError("the store has no globe")
    chain = ensure_local_maps(store, ref, float(params.upsample.margin_km))
    fields = build_settlement_fields(
        globe, [m.fields for m in chain.maps], chain.center, chain.side_km, params.upsample)
    realm, world = store.realm_params, store.world_params
    site = build_site(
        globe, fields, chain.side_km, chain.radius_km, chain.maps, params.rivers, params.roads,
        city=ref.kind == "city", local_road_raster_texels=float(store.local_params.roads.raster_texels),
        land=world.cost.land if world is not None else None,
        danger_scale=float(realm.danger.movement_scale) if realm is not None else 2.0,
    )
    return SettlementMap(chain=chain, fields=fields, site=site)
