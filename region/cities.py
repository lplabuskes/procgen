"""Refining the world's cities onto a region's (or a local map's) finer coast and rivers.

The refine stage sites every city on a global texel (~14 km), and remembers what
that texel offered (``Settlement.site_kind``: ocean or lake coast, river,
confluence; :mod:`realmgen.refine.settlements`).  The region re-floods the
shoreline and re-routes the rivers over its finer relief, so a harbour city left
on its global texel can stand kilometres from the refined coast.  Each tier
therefore re-sites the city among its own dry texels near where the tier above
put it:

- a city moves at most ``reach`` from its position at the tier above (region:
  ``refine_radius_global_texels`` global texels; local map:
  ``refine_radius_parent_texels`` regional texels) and stays on dry land in its
  own face;
- it keeps what its site offered wherever this tier can show it: a harbour city
  lands on the refined coast (ocean, or lake for a lake port), a river city on
  the drawn river or a confluence;
- among those, the stage's own weights pick the spot — harbour, confluence and
  river bonuses for the features the site had, local fertility, a pull back
  toward the parent position and a little seeded jitter.  Features the site did
  not have earn nothing, so an inland city does not wander onto a stream or a
  shore the finer tier revealed.

The result is the city's position (``Settlement.position``; the region's also
``region_position``), flagged ``refined`` / ``local_refined`` and saved with the
world.  Each city is refined by exactly one region (the one containing its
site) and one local map (``localmap/settlements.py``), once each.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
from scipy.spatial import KDTree

from realmgen.placement.types import (
    SITE_KIND_CONFLUENCE, SITE_KIND_LAKE_COAST, SITE_KIND_OCEAN_COAST, SITE_KIND_RIVER,
)
from realmgen.refine.settlements import (
    SITE_CENTRE, SITE_CONFLUENCE, SITE_FERTILITY, SITE_JITTER, SITE_LAKE, SITE_OCEAN, SITE_RIVER,
)
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
    """Refine the not-yet-refined cities whose site lies in this region, in place;
    returns how many were refined."""
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
        # Keep the harbour, then the river, wherever this tier has one in reach.
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
