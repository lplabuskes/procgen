"""Refining the world's towns onto a region's finer terrain and streams.

The town stage sites every town on a global texel (~14 km), favouring rivers,
the smaller streams of the hydrology stream network, coasts, routes and mines
(:mod:`realmgen.towns`).  That choice is a world fact.  A region refines only what
its finer grid can see — *where in that texel* the town stands — and the result
becomes the town's actual position (``Towns.positions``, flagged ``refined``,
saved with the world):

- a town moves at most ``refine_radius_global_texels`` global texels from its
  site, and stays on dry land in its own face;
- the site weight is the stage's own (:func:`site_log_weight_and_kind`), with the
  river and confluence flags taken from the region's re-routed stream network
  (:mod:`region.rivers`) and the coast from the refined shoreline;
- a town keeps its kind, and a river town (on a river or stream) or port lands on
  the drawn river or the refined coast whenever one is in reach;
- spacing to every other town and city is kept, against wherever they stand now.

**Cities never move.**  Each town is refined by exactly one region — the one
containing its site — and only once, so the result does not depend on which
region was visited first or on the overlap between neighbouring regions.  Site
noise is seeded per town.
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
from region.sampling import GlobalSampler
from region.upsample import RegionFields

if TYPE_CHECKING:
    from core.world import World

# RNG stream tag for a town's refinement noise, seeded (seed, town, tag).
_REFINE_STREAM = 3


@paramclass
class RegionTownParams:
    """Tuning for refining town positions in a region."""

    # How far a town may move from the site the world gave it, in global texels.
    refine_radius_global_texels: float = _f(
        1.0, label="Town refinement radius (global texels)", min=0.0, max=2.0, step=0.25)


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


def _river_points(rivers: RegionRivers, step: float) -> Optional[np.ndarray]:
    """Points along every drawn (not under-water) stretch of the regional rivers."""
    runs = []
    for positions, _, in_water in rivers.paths:
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
    coast: np.ndarray        # (P,) bool
    river: np.ndarray        # (P,) bool — on a drawn river or stream, not a confluence
    confluence: np.ndarray   # (P,) bool
    route: np.ndarray        # (P,) bool
    mine: np.ndarray         # (P,) bool


def _region_sites(
    fields: RegionFields, rivers: RegionRivers, globe: "World", sc, river_threshold_factor: float,
) -> _Sites:
    ocean = fields.textures.get("ocean_mask")
    lake = fields.textures.get("lake_mask")
    ocean = ocean[..., 0] > 0.5 if ocean is not None else np.zeros(fields.in_hex.shape, dtype=bool)
    lake = lake[..., 0] > 0.5 if lake is not None else np.zeros_like(ocean)
    water = ocean | lake
    dry = fields.in_hex & ~water
    pos = fields.positions[dry]
    face = fields.faces[dry]
    mesh = globe.mesh
    if RURAL_DENSITY_TEXTURE in mesh.face_textures:
        uv = fields.face_uv[dry]
        rho = GlobalSampler(mesh).sample(mesh.face_textures[RURAL_DENSITY_TEXTURE], face, uv[:, 0], uv[:, 1])[:, 0]
    else:
        rho = np.ones(len(pos))
    spacing = fields.grid.texel_spacing_rad(fields.tex_size)
    snap = float(river_threshold_factor) * spacing
    conf = _near(_tree(rivers.confluences), pos, snap)
    return _Sites(
        positions=pos, face=face, rho=rho,
        coast=(~water & _neighbour_any(water))[dry],
        river=_near(_tree(_river_points(rivers, 0.5 * spacing)), pos, snap) & ~conf,
        confluence=conf,
        route=_near(_tree(sc.route_points), pos, sc.route_radius),
        mine=_near(_tree(sc.mineral_points), pos, sc.mining_radius),
    )


def refine_region_towns(
    fields: RegionFields,
    rivers: RegionRivers,
    globe: "World",
    town_params: Optional[TownParams] = None,
    population_params: Optional[PopulationParams] = None,
    river_threshold_factor: float = 1.5,
    params: Optional[RegionTownParams] = None,
) -> int:
    """Refine the not-yet-refined towns whose site lies in this region, in place.

    Moves ``globe``'s ``towns`` feature positions and sets their ``refined``
    flag; returns how many towns were refined (0 without towns).
    ``river_threshold_factor`` × the regional texel spacing is how close a site
    must be to a river or confluence to count as on it — the town stage's rule at
    the finer spacing.
    """
    towns: Optional[Towns] = globe.get_feature("towns")
    if towns is None or len(towns) == 0:
        return 0
    params = params or RegionTownParams()
    p = town_params or TownParams()
    pp = population_params or PopulationParams()
    owned = (ecef_to_cell(towns.site_positions, globe.antimeridian) == fields.region_id) & ~towns.refined
    todo = np.flatnonzero(owned)
    if todo.size == 0:
        return 0

    mesh = globe.mesh
    seed = int(globe.seed or 0)
    sc = town_site_context(globe, p, pp)
    sites = _region_sites(fields, rivers, globe, sc, river_threshold_factor)
    gw = mesh.face_textures["elevation"].shape[2]
    reach = float(params.refine_radius_global_texels) * float(np.sqrt(np.mean(mesh.face_areas))) / max(gw - 1, 1)

    town_radii = sc.town_spacing * (towns.population / float(p.town_min)) ** SPACING_EXPONENT
    max_pair = max(float(town_radii.max()), float(sc.city_radii.max()) if len(sc.city_radii) else 0.0)
    site_tree = KDTree(towns.site_positions)
    city_tree = _tree(sc.city_positions)
    candidate_tree = KDTree(sites.positions) if len(sites.positions) else None

    # Largest first, like the stage: the biggest towns take the best spots.
    for i in todo[np.argsort(-towns.population[todo], kind="stable")].tolist():
        towns.refined[i] = True
        if candidate_tree is None:
            continue
        cand = np.asarray(candidate_tree.query_ball_point(towns.site_positions[i], reach), dtype=np.int64)
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
        # Keep what the world chose the site for, where the region can show it.
        if towns.kind[i] == TOWN_RIVER and (river | conf).any():
            log_w = np.where(river | conf, log_w, -np.inf)
        elif towns.kind[i] == TOWN_PORT and coast.any():
            log_w = np.where(coast, log_w, -np.inf)

        # Spacing against every town and city near the site, wherever it stands now.
        radius = town_radii[i]
        near = np.asarray(site_tree.query_ball_point(towns.site_positions[i], max_pair + 2.0 * reach), dtype=np.int64)
        near = near[near != i]
        disc_pos = [towns.positions[near]]
        disc_rad = [np.maximum(town_radii[near], radius)]
        if city_tree is not None:
            c = np.asarray(city_tree.query_ball_point(towns.site_positions[i], max_pair + reach), dtype=np.int64)
            disc_pos.append(sc.city_positions[c])
            disc_rad.append(np.maximum(sc.city_radii[c], radius))
        dpos, drad = np.concatenate(disc_pos), np.concatenate(disc_rad)
        cpos = sites.positions[cand]
        if len(dpos):
            ok = (cpos @ dpos.T <= 1.0 - 0.5 * drad ** 2).all(axis=1)
            log_w = np.where(ok, log_w, -np.inf)

        score = log_w + np.random.default_rng([seed, i, _REFINE_STREAM]).gumbel(size=cand.size)
        best = int(np.argmax(score))
        if np.isfinite(score[best]):
            towns.positions[i] = cpos[best]
    return int(todo.size)
