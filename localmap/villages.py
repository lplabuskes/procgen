"""Villages: the local map's settlements of 200–1000 people, carved out of the rural population.

- Budget: the rural people in each global texel ∩ local map (a regional texel is
  too small to hold a village); ``village_fraction`` of it goes to villages.
- Sizes: the town stage's truncated Pareto (:func:`realmgen.towns.placement.draw_town_sizes`),
  whose stochastic remainder gives thin land the odd village.
- Sites: largest first, Gumbel-max over a log weight among the budget's dry texels
  (not under a river's channel)
  (rural density, river/stream, confluence, coast, road raise it; slope lowers it).
  Exclusion radius ``spacing_km · (pop / village_min)^0.25``; towns and cities
  clear ``settlement_clearance_km`` scaled from 1,000 people.
- Accounting: an unsited village stays rural; each budget's rural density is
  scaled down by exactly what its villages took (:func:`village_budget_report`).

Deterministic: each budget has its own stream (world seed, map, global texel).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial import KDTree

from core.params import param_field as _f, paramclass
from realmgen.refine.sites import densify_polylines
from realmgen.towns.placement import SPACING_EXPONENT, draw_town_sizes
from region.resources import refined_slope
from region.rivers import RegionRivers
from region.roads import RegionRoads
from region.towns import _neighbour_any, _river_points
from region.upsample import EARTH_RADIUS_KM, RegionFields

VILLAGE_FARM, VILLAGE_RIVER, VILLAGE_COAST, VILLAGE_ROAD, VILLAGE_MINING = range(5)
VILLAGE_KIND_NAMES = ("Farming village", "River village", "Fishing village", "Roadside village", "Mining village")

_VILLAGE_STREAM = 7   # RNG stream tag
# Density floor (relative to the budget's mean): barren-but-dry ground keeps a small chance.
_DENSITY_FLOOR = 0.05
_CLEARANCE_REFERENCE_POP = 1_000.0


@paramclass
class VillageParams:
    """Tuning for village sizing and siting."""

    village_fraction: float = _f(
        0.7, label="Share of rural people living in villages", min=0.0, max=1.0, step=0.05)
    village_min: float = _f(200.0, label="Smallest village (people)", min=20.0, max=1_000.0, step=10.0)
    village_max: float = _f(1_000.0, label="Largest village (people)", min=100.0, max=5_000.0, step=50.0)
    # α = 1 is rank-size; mean ≈ 400 people on [200, 1000].
    size_exponent: float = _f(1.0, label="Village size exponent α", min=0.3, max=3.0)
    spacing_km: float = _f(2.0, label="Village exclusion radius at the smallest size (km)", min=0.0, max=10.0)
    settlement_clearance_km: float = _f(
        3.0, label="Town/city clearance at 1,000 people (km)", min=0.0, max=20.0)
    river_bonus: float = _f(2.0, label="Site bonus: river or stream", min=0.0, max=10.0)
    confluence_bonus: float = _f(1.0, label="Site bonus: confluence (extra)", min=0.0, max=10.0)
    coast_bonus: float = _f(1.0, label="Site bonus: coast", min=0.0, max=10.0)
    road_bonus: float = _f(1.0, label="Site bonus: road", min=0.0, max=10.0)
    # How close a site must be to a river or road to count as on it.
    water_reach_km: float = _f(0.6, label="River reach (km)", min=0.1, max=5.0, step=0.1)
    road_reach_km: float = _f(0.6, label="Road reach (km)", min=0.1, max=5.0, step=0.1)
    # Fraction of the site weight lost on saturated slopes.
    slope_penalty: float = _f(0.8, label="Slope penalty", min=0.0, max=1.0, step=0.05)
    slope_saturation: float = _f(
        0.15, label="Slope saturation (km/km)", min=0.01, max=1.0, step=0.01)


@dataclass
class Villages:
    """A local map's villages, and its countryside net of them."""

    positions: np.ndarray        # (V, 3) unit sphere
    population: np.ndarray       # (V,) people
    kind: np.ndarray             # (V,) VILLAGE_* code
    anchor: np.ndarray           # (V,) global texel (the budget it came from)
    rural_density: np.ndarray    # (M, T, T, 1) people/km², net of the villages
    # Per budget (global texel ∩ map): anchor, rural people before villages, people placed.
    budget_anchor: np.ndarray    # (B,)
    budget_people: np.ndarray    # (B,)
    budget_placed: np.ndarray    # (B,)

    def __len__(self) -> int:
        return int(self.positions.shape[0])


def _chord(km: float) -> float:
    return float(km) / EARTH_RADIUS_KM


def place_villages(
    local: RegionFields,
    rivers: Optional[RegionRivers],
    roads: Optional[RegionRoads],
    settlement_positions: np.ndarray,
    settlement_populations: np.ndarray,
    seed: int,
    params: Optional[VillageParams] = None,
    river_covered: Optional[np.ndarray] = None,
) -> Villages:
    """Size and site the villages of local map ``local``.

    ``settlement_positions`` / ``settlement_populations``: towns and cities in and
    around the map, which villages keep clear of.  ``seed`` identifies world and map.
    ``river_covered``: ``(M, T, T)`` texels under a river (:func:`localmap.banks.covered_texels`),
    never a site.
    """
    p = params or VillageParams()
    tex = local.textures
    shape = local.in_hex.shape
    domain = local.domain
    rho = tex["rural_density"][..., 0].astype(np.float64) if "rural_density" in tex else np.zeros(shape)
    water = np.zeros(shape, dtype=bool)
    for key in ("ocean_mask", "lake_mask"):
        if key in tex:
            water |= tex[key][..., 0] > 0.5
    anchors = local.anchors

    # ── budgets: people per global texel ∩ map ───────────────────────────────
    budget_anchor, inverse = np.unique(anchors[domain], return_inverse=True)
    people = np.bincount(inverse, weights=rho[domain] * local.texel_area_km2, minlength=budget_anchor.size)

    # ── candidate sites: dry texels of the domain ────────────────────────────
    dry = domain & ~water
    if river_covered is not None:
        dry &= ~river_covered
    cand_pos = local.positions[dry]
    cand_budget = np.searchsorted(budget_anchor, anchors[dry])
    cand_rho = rho[dry]
    spacing = local.grid.texel_spacing_rad(local.tex_size)
    coast = (dry & _neighbour_any(water))[dry]
    near_river = np.zeros(len(cand_pos), dtype=bool)
    confluence = np.zeros(len(cand_pos), dtype=bool)
    # Rivers and the region's streams alike (see localmap/rivers.py).
    if rivers is not None and len(cand_pos):
        river_pts = _river_points(rivers, 0.5 * spacing, streams=True)
        if river_pts is not None:
            near_river = KDTree(river_pts).query(cand_pos, workers=-1)[0] < _chord(p.water_reach_km)
        if len(rivers.confluences):
            confluence = KDTree(rivers.confluences).query(cand_pos, workers=-1)[0] < _chord(p.water_reach_km)
    near_road = np.zeros(len(cand_pos), dtype=bool)
    if roads is not None and roads.polylines and len(cand_pos):
        road_pts = densify_polylines(roads.polylines, 0.5 * spacing)
        if road_pts is not None:
            near_road = KDTree(road_pts).query(cand_pos, workers=-1)[0] < _chord(p.road_reach_km)
    slope = refined_slope(tex["elevation"][..., 0], np.sqrt(local.texel_area_km2))[dry]
    slope_n = np.clip(slope / max(float(p.slope_saturation), 1e-9), 0.0, 1.0)

    mean_rho = np.bincount(cand_budget, weights=cand_rho, minlength=budget_anchor.size) / np.maximum(
        np.bincount(cand_budget, minlength=budget_anchor.size), 1)
    rho_n = np.where(mean_rho[cand_budget] > 0, cand_rho / np.maximum(mean_rho[cand_budget], 1e-300), 1.0)
    weight = (
        (rho_n + _DENSITY_FLOOR)
        * (1.0 + p.river_bonus * near_river + p.confluence_bonus * confluence
           + p.coast_bonus * coast + p.road_bonus * near_road)
        * np.maximum(1.0 - p.slope_penalty * slope_n, 1e-3)
    )
    log_w = np.log(weight)
    # A fishing village is on the shore, or among people who mostly fish.
    fishers = coast.copy()
    if "fishing_density" in tex:
        fish = tex["fishing_density"][..., 0].astype(np.float64)[dry]
        fishers |= fish > 0.5 * cand_rho
    # A mining village is among people who mostly mine.
    miners = np.zeros_like(fishers)
    if "mining_density" in tex:
        miners = tex["mining_density"][..., 0].astype(np.float64)[dry] > 0.5 * cand_rho
    kind = np.select(
        [miners, fishers, near_river | confluence, near_road],
        [VILLAGE_MINING, VILLAGE_COAST, VILLAGE_RIVER, VILLAGE_ROAD],
        default=VILLAGE_FARM,
    ).astype(np.int8)

    # ── sizes, per budget ────────────────────────────────────────────────────
    rngs = [np.random.default_rng([int(seed), int(a), _VILLAGE_STREAM]) for a in budget_anchor]
    frac = float(np.clip(p.village_fraction, 0.0, 1.0))
    requests = []
    for b in range(budget_anchor.size):
        sizes = draw_town_sizes(
            frac * people[b], rngs[b], town_min=float(p.village_min),
            town_max=max(float(p.village_max), float(p.village_min) + 1.0),
            alpha=float(p.size_exponent), cap=float(people[b]),
        )
        requests.extend((float(size), b, j) for j, size in enumerate(sizes))
    requests.sort(key=lambda r: (-r[0], r[1], r[2]))

    # ── sites, largest first ─────────────────────────────────────────────────
    by_budget = [np.empty(0, dtype=np.int64)] * budget_anchor.size
    if len(cand_pos):
        order = np.argsort(cand_budget, kind="stable")
        bounds = np.searchsorted(cand_budget[order], np.arange(budget_anchor.size + 1))
        by_budget = [order[bounds[b]:bounds[b + 1]] for b in range(budget_anchor.size)]
    blocked = np.zeros(len(cand_pos), dtype=bool)
    tree = KDTree(cand_pos) if len(cand_pos) else None
    if tree is not None and len(settlement_positions):
        radii = _chord(p.settlement_clearance_km) * (
            np.maximum(np.asarray(settlement_populations, dtype=np.float64), 1.0) / _CLEARANCE_REFERENCE_POP
        ) ** SPACING_EXPONENT
        for pos, r in zip(np.asarray(settlement_positions, dtype=np.float64), radii):
            blocked[tree.query_ball_point(pos, r)] = True

    placed_pos, placed_pop, placed_kind, placed_anchor = [], [], [], []
    placed = np.zeros(budget_anchor.size)
    base_radius = _chord(p.spacing_km)
    for size, b, _ in requests:
        cand = by_budget[b]
        if cand.size == 0 or tree is None:
            continue
        score = log_w[cand] + rngs[b].gumbel(size=cand.size)
        score[blocked[cand]] = -np.inf
        best = int(np.argmax(score))
        if not np.isfinite(score[best]):
            continue
        c = int(cand[best])
        # Largest first, so blocking by the placed radius keeps each pair apart by the larger.
        radius = base_radius * (size / float(p.village_min)) ** SPACING_EXPONENT
        blocked[tree.query_ball_point(cand_pos[c], radius)] = True
        placed[b] += size
        placed_pos.append(cand_pos[c])
        placed_pop.append(size)
        placed_kind.append(kind[c])
        placed_anchor.append(budget_anchor[b])

    # ── the countryside net of its villages ──────────────────────────────────
    keep = np.where(people > 0, 1.0 - placed / np.maximum(people, 1e-300), 1.0)
    idx = np.searchsorted(budget_anchor, anchors)
    idx_c = np.minimum(idx, budget_anchor.size - 1)
    mine = (budget_anchor.size > 0) & (budget_anchor[idx_c] == anchors)
    rural = np.where(mine, rho * keep[idx_c], rho)

    return Villages(
        positions=np.array(placed_pos, dtype=np.float64).reshape(-1, 3),
        population=np.asarray(placed_pop, dtype=np.float64),
        kind=np.asarray(placed_kind, dtype=np.int8),
        anchor=np.asarray(placed_anchor, dtype=np.int64),
        rural_density=rural[..., None].astype(np.float32),
        budget_anchor=budget_anchor.astype(np.int64),
        budget_people=people,
        budget_placed=placed,
    )


def village_budget_report(local: RegionFields, villages: Villages) -> dict[str, float]:
    """Conservation check: rural (net) + villages per global texel ∩ map vs the rural people before."""
    domain = local.domain
    anchors = local.anchors[domain]
    idx = np.searchsorted(villages.budget_anchor, anchors)
    rural = np.bincount(
        idx, weights=villages.rural_density[..., 0][domain].astype(np.float64) * local.texel_area_km2,
        minlength=villages.budget_anchor.size,
    )
    total = rural + villages.budget_placed
    residual = np.abs(total - villages.budget_people)
    people = float(villages.budget_people.sum())
    return {
        "budgets": float(villages.budget_anchor.size),
        "people": people,
        "in_villages": float(villages.population.sum()),
        "village_share": float(villages.population.sum()) / people if people > 0 else 0.0,
        "max_residual": float(residual.max()) if residual.size else 0.0,
        "max_relative_residual": float((residual / np.maximum(villages.budget_people, 1e-9)).max())
        if residual.size else 0.0,
    }
