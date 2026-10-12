"""One settlement map, end to end: :func:`build_settlement_map` runs

1. the chain to its local maps (:func:`settlegen.chain.ensure_local_maps`, through
   the shared :class:`localmap.store.MapStore`: built only if not held);
2. the frame's terrain (:func:`settlegen.upsample.build_settlement_fields`);
3. the site, as vectors (:func:`settlegen.site.build_site`);
4. the profile (:func:`settlegen.profile.build_profile`);
5. the layout (:func:`settlegen.layout.build_layout`).  Qt-free.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from region.upsample import RegionFields
from settlegen.chain import SettlementMaps, SettlementRef, ensure_local_maps
from settlegen.footprint import site_slope
from settlegen.layout import SettlementLayout, build_layout
from settlegen.params import SettlementMapParams
from settlegen.profile import SettlementProfile, build_profile, settlement_key
from settlegen.site import SettlementSite, build_site
from settlegen.upsample import build_settlement_fields

if TYPE_CHECKING:
    from localmap.store import MapStore


@dataclass
class SettlementMap:
    """Everything one settlement map holds (so far: its local maps, terrain, site, profile and layout)."""

    chain: SettlementMaps
    fields: RegionFields
    site: SettlementSite
    profile: SettlementProfile
    layout: SettlementLayout

    @property
    def ref(self) -> SettlementRef:
        return self.chain.ref


def build_settlement_map(
    store: "MapStore", ref: SettlementRef, params: Optional[SettlementMapParams] = None,
    timings: Optional[dict[str, float]] = None,
) -> SettlementMap:
    """Build ``ref``'s settlement map from ``store``'s local maps.  ``params.upsample``
    is expected to carry the local tier's shared detail (:func:`settlegen.upsample.with_local_detail`).
    ``timings``, if given, gets each step's wall-clock seconds (chain, terrain, site, profile, layout)."""
    params = params or SettlementMapParams()
    globe = store.globe
    if globe is None:
        raise ValueError("the store has no globe")
    clock = [time.perf_counter()]

    def lap(name: str) -> None:
        now = time.perf_counter()
        if timings is not None:
            timings[name] = now - clock[0]
        clock[0] = now

    chain = ensure_local_maps(store, ref, float(params.upsample.margin_km), params.footprint)
    lap("chain")
    fields = build_settlement_fields(
        globe, [m.fields for m in chain.maps], chain.center, chain.side_km, params.upsample)
    lap("terrain")
    realm, world = store.realm_params, store.world_params
    site = build_site(
        globe, fields, chain, params.rivers, params.roads, params.footprint, params.anchors,
        farmland=store.local_params.farmland,
        local_road_raster_texels=float(store.local_params.roads.raster_texels),
        land=world.cost.land if world is not None else None,
        danger_scale=float(realm.danger.movement_scale) if realm is not None else 2.0,
    )
    lap("site")
    profile = build_profile(globe, fields, site.frame, chain, params.profile)
    lap("profile")
    layout = build_layout(site, profile, site_slope(fields, site.frame), settlement_key(globe, ref), params.layout)
    lap("layout")
    return SettlementMap(chain=chain, fields=fields, site=site, profile=profile, layout=layout)
