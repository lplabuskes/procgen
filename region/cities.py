"""Refining the world's cities onto a region's (or a local map's) finer coast and rivers.

Cities are sited on a global texel with a ``Settlement.site_kind`` (ocean/lake
coast, river, confluence; :mod:`realmgen.refine.settlements`).  The refined
shore and rivers can leave them kilometres off, so each tier re-sites them:

- within ``reach`` of the parent position (``refine_radius_global_texels`` /
  ``refine_radius_parent_texels``), on dry land in the same face;
- keeping their site kind where this tier shows it (harbours on the refined
  coast, river cities on a drawn river or confluence);
- a city tagged with water links (``Settlement.water_links``, from its trade)
  only on candidates meeting them — on this tier's drawing of the tagged river,
  or the ocean / lake coast — looking past ``reach`` (to its whole face here)
  when none in reach does;
- scored by the stage's weights for the features the site had, fertility, a
  pull toward the parent and seeded jitter.  Features the site lacked earn
  nothing, so inland cities don't wander onto newly revealed shores.

The result is ``Settlement.position`` (the region's also ``region_position``),
flagged ``refined`` / ``local_refined``.  Each city is refined once by the
region containing its site and once by a local map (``localmap/settlements.py``).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
from scipy.spatial import KDTree

from realmgen.placement.types import (
    SITE_KIND_CONFLUENCE, SITE_KIND_LAKE_COAST, SITE_KIND_OCEAN_COAST, SITE_KIND_RIVER,
)
from realmgen.refine.sites import RiverProximity
from realmgen.refine.settlements import (
    SITE_CENTRE, SITE_CONFLUENCE, SITE_FERTILITY, SITE_JITTER, SITE_LAKE, SITE_OCEAN, SITE_RIVER,
)
from realmgen.trade.water_links import river_flows, water_link_mask
from region.geometry import ecef_to_cell
from region.rivers import RegionRivers
from region.towns import REGION_TIER, RegionSettlementParams, global_texel_rad, tier_sites
from region.upsample import RegionFields

if TYPE_CHECKING:
    from core.world import World

# RNG stream tag for a city's refinement jitter, seeded (seed, city, tag, tier).
_CITY_STREAM = 11


def refine_region_cities(
    fields: RegionFields,
    rivers: RegionRivers,
    globe: "World",
    river_threshold_factor: float = 1.5,
    params: Optional[RegionSettlementParams] = None,
) -> int:
    """Refine this region's not-yet-refined cities in place; returns how many."""
    settlements = globe.get_feature("settlements") or []
    if not settlements:
        return 0
    params = params or RegionSettlementParams()
    sites = np.array([s.site_position for s in settlements], dtype=np.float64).reshape(-1, 3)
    refined = np.array([s.refined for s in settlements], dtype=bool)
    todo = np.flatnonzero((ecef_to_cell(sites, globe.antimeridian) == fields.region_id) & ~refined)
    reach = float(params.refine_radius_global_texels) * global_texel_rad(globe)
    refine_cities(fields, rivers, globe, todo, REGION_TIER, reach, river_threshold_factor)
    return int(todo.size)


def refine_cities(
    fields: RegionFields,
    rivers: RegionRivers,
    globe: "World",
    todo: np.ndarray,
    tier: int,
    reach: float,
    river_threshold_factor: float = 1.5,
) -> None:
    """Re-site cities ``todo`` (indices into ``settlements``) on ``fields`` at
    ``tier``, in place.  See the module docstring."""
    settlements = globe.get_feature("settlements") or []
    if not settlements or len(todo) == 0:
        return
    region = tier == REGION_TIER
    sites = tier_sites(fields, rivers, globe, None, river_threshold_factor)
    tree = KDTree(sites.positions) if len(sites.positions) else None
    seed = int(globe.seed or 0)
    spacing = fields.grid.texel_spacing_rad(fields.tex_size)
    snap = float(river_threshold_factor) * spacing
    proximity = RiverProximity(lambda r: _river_runs(rivers, r), 0.5 * spacing)
    flows = river_flows(globe.get_feature("rivers"))

    for k in sorted((int(k) for k in todo), key=lambda k: (-float(settlements[k].population), k)):
        s = settlements[k]
        if region:
            s.refined = True
        else:
            s.local_refined = True
        if tree is None:
            continue
        parent = s.site_position if region else s.region_position
        cand = np.asarray(tree.query_ball_point(parent, reach), dtype=np.int64)
        cand = cand[sites.face[cand] == int(s.face_id)]
        if s.water_links:
            face = np.flatnonzero(sites.face == int(s.face_id))
            on_water = water_link_mask(
                s.water_links, sites.coast_ocean[face], sites.coast_lake[face],
                lambda r: proximity.near(r, sites.positions[face], snap), flows)
            if on_water is not None:
                ok = face[on_water]
                in_reach = np.intersect1d(ok, cand)
                cand = in_reach if in_reach.size else ok
        if cand.size == 0:
            continue
        # Only the features the site had count.
        kind = int(s.site_kind)
        on_river = bool(kind & (SITE_KIND_RIVER | SITE_KIND_CONFLUENCE))
        none = np.zeros(cand.size, dtype=bool)
        ocean = sites.coast_ocean[cand] if kind & SITE_KIND_OCEAN_COAST else none
        lake = sites.coast_lake[cand] & ~ocean if kind & SITE_KIND_LAKE_COAST else none
        conf = sites.confluence[cand] if on_river else none
        river = sites.river[cand] & ~conf if on_river else none
        d = np.linalg.norm(sites.positions[cand] - parent[None, :], axis=1) / max(reach, 1e-12)
        rng = np.random.default_rng([seed, k, _CITY_STREAM, tier])
        score = (
            SITE_OCEAN * ocean + SITE_LAKE * lake + SITE_CONFLUENCE * conf + SITE_RIVER * river
            + SITE_FERTILITY * sites.fertility[cand] - SITE_CENTRE * d + SITE_JITTER * rng.random(cand.size)
        )
        # Keep the harbour, then the river, where this tier has one in reach.
        keep = np.ones(cand.size, dtype=bool)
        coast = ocean | lake
        if coast.any():
            keep &= coast
        if (keep & (river | conf)).any():
            keep &= river | conf
        best = cand[int(np.argmax(np.where(keep, score, -np.inf)))]
        s.position = sites.positions[best].copy()
        if region:
            s.region_position = s.position.copy()


def _river_runs(rivers: RegionRivers, r: int) -> list[np.ndarray]:
    """This tier's drawing of world river ``r``: its dry stretches, as polylines."""
    runs = []
    for (positions, _, in_water), ids in zip(rivers.paths, rivers.river_ids()):
        pts = np.asarray(positions, dtype=np.float64)
        dry = ~np.asarray(in_water, dtype=bool)                  # per segment
        sel = np.flatnonzero(dry & (np.asarray(ids)[:dry.size] == r))
        for seg in np.split(sel, np.flatnonzero(np.diff(sel) > 1) + 1):
            if seg.size:
                runs.append(pts[seg[0]:seg[-1] + 2])
    return runs
