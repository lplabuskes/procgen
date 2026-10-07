"""Refining the world's towns onto a region's (or a local map's) finer terrain and streams.

The town stage (:mod:`realmgen.towns`) sites each town on a global texel; a region
refines where in that texel it stands, saved as ``Towns.positions`` (flagged
``refined``):

- a town moves at most ``refine_radius_global_texels``, staying on dry land in its face;
- the site weight is the stage's own (:func:`site_log_weight_and_kind`) with
  river/confluence, route, coast and rural density from the refined tier;
- a town keeps its kind: river towns and ports land on a drawn river or the
  refined coast when one is in reach;
- spacing to every other town and city is kept against where they stand now.

Each town is refined once, by the region containing its site, so results don't
depend on visit order.  Site noise is seeded per town.  Cities are refined first
(``region/cities.py``).

The local tier refines again from the region position (:func:`refine_towns`
with ``LOCAL_TIER``; ownership in ``localmap/settlements.py``), with spacing
against others' region positions so open order doesn't matter.
:func:`tier_sites` serves both tiers and the cities.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
from scipy.spatial import KDTree

from core.params import param_field as _f, paramclass
from realmgen.population.builder import PopulationParams
from realmgen.refine.sites import densify_polylines
from realmgen.towns.builder import RURAL_DENSITY_TEXTURE, TownParams, town_site_context
from realmgen.towns.placement import SPACING_EXPONENT, site_log_weight_and_kind
from realmgen.towns.types import TOWN_PORT, TOWN_RIVER, Towns
from region.geometry import ecef_to_cell
from region.rivers import RegionRivers
from region.roads import RegionRoads
from region.sampling import GlobalSampler
from region.upsample import SMALL_LAKE_KEY, RegionFields

if TYPE_CHECKING:
    from core.world import World

# RNG stream tag for a town's refinement noise, seeded (seed, town, tag[, tier]).
_REFINE_STREAM = 3

# The tiers below the world that refine settlement positions.
REGION_TIER, LOCAL_TIER = 1, 2


@paramclass
class RegionSettlementParams:
    """Tuning for refining town and city positions in a region."""

    # Max move from the world site, in global texels.
    refine_radius_global_texels: float = _f(
        1.0, label="Refinement radius (global texels)", min=0.0, max=2.0, step=0.25)


def _neighbour_any(mask: np.ndarray) -> np.ndarray:
    """True where any 4-neighbour inside the same hex texture is True, ``(M, T, T)``."""
    out = np.zeros_like(mask)
    out[:, 1:, :] |= mask[:, :-1, :]
    out[:, :-1, :] |= mask[:, 1:, :]
    out[:, :, 1:] |= mask[:, :, :-1]
    out[:, :, :-1] |= mask[:, :, 1:]
    return out


def _near(tree: Optional[KDTree], points: np.ndarray, radius: float) -> np.ndarray:
    if tree is None or points.shape[0] == 0:
        return np.zeros(points.shape[0], dtype=bool)
    return tree.query(points, workers=-1)[0] < radius


def _tree(points: Optional[np.ndarray]) -> Optional[KDTree]:
    return KDTree(points) if points is not None and len(points) else None


def _river_points(rivers: RegionRivers, step: float, streams: bool = False) -> Optional[np.ndarray]:
    """Points along every drawn (not under-water) river stretch, plus ``RegionRivers.streams`` if ``streams``."""
    runs = []
    for positions, _, in_water in [*rivers.paths, *(rivers.streams if streams else [])]:
        pts = np.asarray(positions, dtype=np.float64)
        dry = np.flatnonzero(~np.asarray(in_water, dtype=bool))
        # Split at hidden segments so no run bridges a lake.
        for seg in np.split(dry, np.flatnonzero(np.diff(dry) > 1) + 1):
            if seg.size:
                runs.append(pts[seg[0]:seg[-1] + 2])
    return densify_polylines(runs, step)


@dataclass
class _Sites:
    """The region's candidate sites: dry in-hexagon texels and what they offer."""

    positions: np.ndarray    # (P, 3)
    face: np.ndarray         # (P,)
    rho: np.ndarray          # (P,) rural density
    fertility: np.ndarray    # (P,) 0–1
    coast: np.ndarray        # (P,) bool — ocean or lake
    coast_ocean: np.ndarray  # (P,) bool
    coast_lake: np.ndarray   # (P,) bool
    river: np.ndarray        # (P,) bool — on a drawn river or stream, not a confluence
    confluence: np.ndarray   # (P,) bool
    route: np.ndarray        # (P,) bool
    mine: np.ndarray         # (P,) bool


def _route_points(roads: RegionRoads, step: float) -> Optional[np.ndarray]:
    """Points along the region's roads — re-routed or straight — for the route flag."""
    straight = [np.stack([np.asarray(a), np.asarray(b)]) for a, b, _ in roads.undrawn]
    sea = [np.stack([np.asarray(a), np.asarray(b)]) for a, b, _ in roads.sea]
    return densify_polylines([*roads.polylines, *sea, *straight], step)


def tier_sites(
    fields: RegionFields, rivers: RegionRivers, globe: "World", sc, river_threshold_factor: float,
    roads: Optional[RegionRoads] = None,
) -> _Sites:
    """The dry in-hexagon texels of ``fields`` and what each offers (no route/mine without ``sc``)."""
    ocean = fields.textures.get("ocean_mask")
    lake = fields.textures.get("lake_mask")
    ocean = ocean[..., 0] > 0.5 if ocean is not None else np.zeros(fields.in_hex.shape, dtype=bool)
    lake = lake[..., 0] > 0.5 if lake is not None else np.zeros_like(ocean)
    small = fields.textures.get(SMALL_LAKE_KEY)
    small = small[..., 0] > 0.5 if small is not None else np.zeros_like(ocean)
    water = ocean | lake
    dry = fields.in_hex & ~water
    pos = fields.positions[dry]
    face = fields.faces[dry]
    mesh = globe.mesh
    if RURAL_DENSITY_TEXTURE in fields.textures:
        rho = fields.textures[RURAL_DENSITY_TEXTURE][..., 0][dry].astype(np.float64)
    elif RURAL_DENSITY_TEXTURE in mesh.face_textures:
        uv = fields.face_uv[dry]
        rho = GlobalSampler(mesh).sample(mesh.face_textures[RURAL_DENSITY_TEXTURE], face, uv[:, 0], uv[:, 1])[:, 0]
    else:
        rho = np.ones(len(pos))
    fert = fields.textures.get("fertility")
    fertility = (np.clip(fert[..., 0][dry].astype(np.float64), 0.0, 1.0) if fert is not None
                 else np.zeros(len(pos)))
    spacing = fields.grid.texel_spacing_rad(fields.tex_size)
    snap = float(river_threshold_factor) * spacing
    conf = _near(_tree(rivers.confluences), pos, snap)
    coast_ocean = _neighbour_any(ocean)[dry]
    # Harbours are on the sea and the globe's lakes, not on small lakes.
    coast_lake = _neighbour_any(lake & ~small)[dry]
    no = np.zeros(len(pos), dtype=bool)
    return _Sites(
        positions=pos, face=face, rho=rho, fertility=fertility,
        coast=coast_ocean | coast_lake, coast_ocean=coast_ocean, coast_lake=coast_lake,
        river=_near(_tree(_river_points(rivers, 0.5 * spacing)), pos, snap) & ~conf,
        confluence=conf,
        route=no if sc is None else _near(
            _tree(sc.route_points if roads is None else _route_points(roads, 0.5 * sc.route_radius)),
            pos, sc.route_radius),
        mine=no if sc is None else _near(_tree(sc.mineral_points), pos, sc.mining_radius),
    )


def global_texel_rad(globe: "World") -> float:
    """The side of a global texel, as a chord on the unit sphere."""
    mesh = globe.mesh
    gw = mesh.face_textures["elevation"].shape[2]
    return float(np.sqrt(np.mean(mesh.face_areas))) / max(gw - 1, 1)


def refine_region_towns(
    fields: RegionFields,
    rivers: RegionRivers,
    globe: "World",
    town_params: Optional[TownParams] = None,
    population_params: Optional[PopulationParams] = None,
    river_threshold_factor: float = 1.5,
    params: Optional[RegionSettlementParams] = None,
    roads: Optional[RegionRoads] = None,
) -> int:
    """Refine the not-yet-refined towns whose site lies in this region, in place.

    Returns how many towns were refined.  ``river_threshold_factor`` × the
    regional texel spacing is the on-river distance.  With ``roads`` the route
    flag reads the region's roads instead of the world's straight routes.
    """
    towns: Optional[Towns] = globe.get_feature("towns")
    if towns is None or len(towns) == 0:
        return 0
    params = params or RegionSettlementParams()
    owned = (ecef_to_cell(towns.site_positions, globe.antimeridian) == fields.region_id) & ~towns.refined
    todo = np.flatnonzero(owned)
    reach = float(params.refine_radius_global_texels) * global_texel_rad(globe)
    refine_towns(
        fields, rivers, globe, todo, REGION_TIER, reach,
        town_params, population_params, river_threshold_factor, roads,
    )
    return int(todo.size)


def refine_towns(
    fields: RegionFields,
    rivers: RegionRivers,
    globe: "World",
    todo: np.ndarray,
    tier: int,
    reach: float,
    town_params: Optional[TownParams] = None,
    population_params: Optional[PopulationParams] = None,
    river_threshold_factor: float = 1.5,
    roads: Optional[RegionRoads] = None,
) -> None:
    """Re-site towns ``todo`` on ``fields`` at ``tier`` in place, within ``reach`` (chord) of their parent sites."""
    towns: Optional[Towns] = globe.get_feature("towns")
    if towns is None or len(todo) == 0:
        return
    region = tier == REGION_TIER
    p = town_params or TownParams()
    pp = population_params or PopulationParams()
    seed = int(globe.seed or 0)
    sc = town_site_context(globe, p, pp)
    sites = tier_sites(fields, rivers, globe, sc, river_threshold_factor, roads)
    # Parent positions, and neighbour positions for spacing: current in a region,
    # region positions on a local map (order-independent).
    parents = towns.site_positions if region else towns.region_positions
    neighbours = towns.positions if region else towns.region_positions
    settlements = globe.get_feature("settlements") or []
    city_pos = sc.city_positions if region else np.array(
        [s.region_position for s in settlements], dtype=np.float64).reshape(-1, 3)

    town_radii = sc.town_spacing * (towns.population / float(p.town_min)) ** SPACING_EXPONENT
    max_pair = max(float(town_radii.max()), float(sc.city_radii.max()) if len(sc.city_radii) else 0.0)
    parent_tree = KDTree(parents)
    city_tree = _tree(city_pos)
    candidate_tree = KDTree(sites.positions) if len(sites.positions) else None

    # Largest first, like the stage.
    for i in todo[np.argsort(-towns.population[todo], kind="stable")].tolist():
        if region:
            towns.refined[i] = True
        else:
            towns.local_refined[i] = True
        if candidate_tree is None:
            continue
        cand = np.asarray(candidate_tree.query_ball_point(parents[i], reach), dtype=np.int64)
        cand = cand[sites.face[cand] == towns.face_id[i]]
        if cand.size == 0:
            continue

        conf = sites.confluence[cand]
        river = sites.river[cand]
        coast = sites.coast[cand]
        log_w, _ = site_log_weight_and_kind(
            sites.rho[cand], coast, conf, river, np.zeros_like(river),
            sites.route[cand], sites.mine[cand], p,
        )
        # Keep the site's kind where this tier can show it.
        if towns.kind[i] == TOWN_RIVER and (river | conf).any():
            log_w = np.where(river | conf, log_w, -np.inf)
        elif towns.kind[i] == TOWN_PORT and coast.any():
            log_w = np.where(coast, log_w, -np.inf)

        radius = town_radii[i]
        near = np.asarray(parent_tree.query_ball_point(parents[i], max_pair + 2.0 * reach), dtype=np.int64)
        near = near[near != i]
        disc_pos = [neighbours[near]]
        disc_rad = [np.maximum(town_radii[near], radius)]
        if city_tree is not None:
            c = np.asarray(city_tree.query_ball_point(parents[i], max_pair + reach), dtype=np.int64)
            disc_pos.append(city_pos[c])
            disc_rad.append(np.maximum(sc.city_radii[c], radius))
        dpos, drad = np.concatenate(disc_pos), np.concatenate(disc_rad)
        cpos = sites.positions[cand]
        if len(dpos):
            ok = (cpos @ dpos.T <= 1.0 - 0.5 * drad ** 2).all(axis=1)
            log_w = np.where(ok, log_w, -np.inf)

        stream = [seed, i, _REFINE_STREAM] if region else [seed, i, _REFINE_STREAM, tier]
        score = log_w + np.random.default_rng(stream).gumbel(size=cand.size)
        best = int(np.argmax(score))
        if np.isfinite(score[best]):
            towns.positions[i] = cpos[best]
            if region:
                towns.region_positions[i] = cpos[best]
