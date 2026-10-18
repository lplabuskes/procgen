"""Farmland: the fields and gardens around a local map's settlements.

Area farmed (medieval rules of thumb):

- villages: ``ha_per_farmer`` (1.5 ha: ~1 ha cropland under three-field rotation
  plus meadow and pasture) per villager who farms (``Villages.farming_share``:
  fishers and miners need no fields);
- towns: ``town_farming_share`` of their people;
- mining towns: only ``mining_farming_share`` (they buy food); so mining villages
  too when their farming share isn't given;
- cities: no fields (fed by trade), just a ring of market gardens outside the
  footprint: its nearest free ground, ``garden_m2_per_person`` for each citizen,
  and all of it within ``city_garden_ring_min_km``; reaching farther over poor ground.

Texels are claimed by distance stretched by ``1 + fertility_weight·(1 − fertility)
+ slope_weight·slope + forest_weight·forest`` (slope relative to
``slope_saturation``) and jittered (``edge_roughness``).  Gardens go first;
then towns and villages claim together, cheapest (settlement, texel) pair first,
so neighbours share ground rather than first-served taking it.  Water, built-up
texels and texels under a river's channel are never claimed, and a village farms
only its own bank of a river at least ``river_barrier_width_m`` wide.

Display-only, Qt-free, deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial import KDTree

from core.params import param_field as _f, paramclass
from localmap.banks import Channels, crossing_pairs
from localmap.cities import CityFootprints
from region.upsample import RegionFields

from utils.jit import njit
from utils.raster import texture_slope
from utils.sphere import EARTH_RADIUS_KM, km_to_rad

FARM_VILLAGE, FARM_TOWN, FARM_CITY = range(3)
FARM_KIND_NAMES = ("Village fields", "Town fields", "Market gardens")
# RGBA per kind: translucent wheat fields, dark olive gardens (distinct from city footprint colours).
FARM_KIND_COLORS: list[tuple[float, float, float, float]] = [
    (0.93, 0.84, 0.46, 0.55),
    (0.93, 0.84, 0.46, 0.55),
    (0.30, 0.46, 0.18, 0.70),
]
# ± fraction of brightness each texel varies by, so fields read as a patchwork.
_PATCHWORK = 0.10
_KM2_PER_HA = 0.01
_FARM_STREAM = 17   # RNG stream tag for the edge jitter


@paramclass
class FarmlandParams:
    ha_per_farmer: float = _f(1.5, label="Farmland per farmer (ha)", min=0.1, max=5.0, step=0.1)
    town_farming_share: float = _f(0.5, label="Townspeople who farm", min=0.0, max=1.0, step=0.05)
    mining_farming_share: float = _f(
        0.1, label="Miners' settlements: people who farm", min=0.0, max=1.0, step=0.05)
    # Market gardens, orchards and dairy (von Thünen's inner ring) per citizen.
    garden_m2_per_person: float = _f(100.0, label="City gardens per person (m²)", min=0.0, max=1000.0, step=5.0)
    city_garden_ring_min_km: float = _f(0.5, label="City garden ring at least (km)", min=0.0, max=5.0, step=0.05)
    fertility_weight: float = _f(1.0, label="Poor-soil stretch", min=0.0, max=5.0, step=0.1)
    slope_weight: float = _f(2.0, label="Steep-slope stretch", min=0.0, max=10.0, step=0.1)
    forest_weight: float = _f(1.0, label="Forest stretch", min=0.0, max=5.0, step=0.1)
    slope_saturation: float = _f(0.15, label="Slope saturation (km/km)", min=0.01, max=1.0, step=0.01)
    edge_roughness: float = _f(0.25, label="Field edge roughness (× distance)", min=0.0, max=1.0, step=0.05)
    # Villages don't farm across a river at least this wide (no ford).
    river_barrier_width_m: float = _f(
        50.0, label="Village fields stop at rivers wider than (m)", min=0.0, max=1_000.0, step=5.0)


@dataclass
class Farmland:
    """A local map's farmland texels; ``owner`` indexes settlements of ``kind`` in the order passed."""

    kind: np.ndarray        # (M, T, T) int8 FARM_* code, -1 where not farmed
    owner: np.ndarray       # (M, T, T) int32 settlement index within its kind, -1 where not farmed
    target_km2: tuple[np.ndarray, np.ndarray, np.ndarray]    # per kind: (N_kind,) wanted
    area_km2: tuple[np.ndarray, np.ndarray, np.ndarray]      # per kind: (N_kind,) claimed


@njit(cache=True)
def _claim(order, who, texel, quota, owner):  # pragma: no cover - numba-compiled
    """Assign free texels over (settlement, texel) pairs cheapest first, up to each quota; ``owner`` in place."""
    claimed = np.zeros(quota.shape[0], dtype=np.int64)
    for k in order:
        s = who[k]
        t = texel[k]
        if owner[t] < 0 and claimed[s] < quota[s]:
            owner[t] = s
            claimed[s] += 1
    return claimed


def _flags(mask: Optional[np.ndarray], n: int) -> np.ndarray:
    return np.zeros(n, dtype=bool) if mask is None else np.asarray(mask, dtype=bool).reshape(-1)


def garden_ring_km(
    population: float, built_km2: float, params: FarmlandParams, perimeter_km: Optional[float] = None,
) -> float:
    """The width of a city's garden ring: ``garden_m2_per_person`` for each of its
    people round ``built_km2`` of footprint (its ``perimeter_km``, else a disc's) —
    ring area ``P·w + π·w²`` — and at least ``city_garden_ring_min_km``."""
    area = max(float(population), 0.0) * float(params.garden_m2_per_person) * 1e-6
    p = float(perimeter_km) if perimeter_km is not None else 2.0 * np.sqrt(np.pi * max(float(built_km2), 0.0))
    w = (-p + np.sqrt(p * p + 4.0 * np.pi * area)) / (2.0 * np.pi)
    return max(float(w), float(params.city_garden_ring_min_km))


def place_farmland(
    local: RegionFields,
    village_positions: np.ndarray, village_population: np.ndarray,
    town_positions: np.ndarray, town_population: np.ndarray,
    footprints: Optional[CityFootprints],
    seed: int,
    params: Optional[FarmlandParams] = None,
    *,
    village_mining: Optional[np.ndarray] = None,
    village_farming: Optional[np.ndarray] = None,
    town_mining: Optional[np.ndarray] = None,
    channels: Optional[Channels] = None,
    river_covered: Optional[np.ndarray] = None,
) -> Farmland:
    """Claim every settlement's farmland.  See the module docstring.

    ``village_farming``: each village's share of people who farm (overrides
    ``village_mining``); ``village_mining`` / ``town_mining``: per-settlement bool mining flags.
    ``channels``: the map's rivers (:func:`localmap.banks.river_channels`), whose
    wide ones bound village fields; ``river_covered``: ``(M, T, T)`` texels under
    them (:func:`localmap.banks.covered_texels`), never farmed.
    """
    p = params or FarmlandParams()
    shape = local.in_hex.shape
    kind = np.full(shape, -1, dtype=np.int8)
    owner = np.full(shape, -1, dtype=np.int32)
    v_pos = np.asarray(village_positions, dtype=np.float64).reshape(-1, 3)
    t_pos = np.asarray(town_positions, dtype=np.float64).reshape(-1, 3)
    v_pop = np.asarray(village_population, dtype=np.float64).reshape(-1)
    t_pop = np.asarray(town_population, dtype=np.float64).reshape(-1)
    n_cities = len(footprints) if footprints is not None else 0
    texel_km2 = float(local.texel_area_km2)
    mining_share = float(p.mining_farming_share)
    v_share = np.where(_flags(village_mining, len(v_pop)), mining_share, 1.0)
    if village_farming is not None and len(village_farming) == len(v_pop):
        v_share = np.clip(np.asarray(village_farming, dtype=np.float64), 0.0, 1.0)
    t_share = np.where(_flags(town_mining, len(t_pop)), mining_share, float(p.town_farming_share))
    v_target = v_pop * v_share * float(p.ha_per_farmer) * _KM2_PER_HA
    t_target = t_pop * t_share * float(p.ha_per_farmer) * _KM2_PER_HA
    areas: list[np.ndarray] = [np.zeros(len(v_pop)), np.zeros(len(t_pop)), np.zeros(n_cities)]

    def result() -> Farmland:
        c_target = (np.asarray(footprints.population) * float(p.garden_m2_per_person) * 1e-6
                    if footprints is not None else np.zeros(0))
        return Farmland(
            kind=kind, owner=owner, target_km2=(v_target, t_target, c_target),
            area_km2=(areas[FARM_VILLAGE], areas[FARM_TOWN], areas[FARM_CITY]),
        )

    water = np.zeros(shape, dtype=bool)
    for key in ("ocean_mask", "lake_mask"):
        if key in local.textures:
            water |= local.textures[key][..., 0] > 0.5
    built = footprints.owner >= 0 if footprints is not None else np.zeros(shape, dtype=bool)
    free = local.in_hex & ~water & ~built
    if river_covered is not None:
        free &= ~river_covered
    cells = np.flatnonzero(free.reshape(-1))
    if cells.size == 0:
        return result()
    pos = local.positions.reshape(-1, 3)[cells]
    tree = KDTree(pos)

    # Distance stretch for poor, steep or wooded ground, jittered.
    tex = local.textures
    stretch = np.ones(cells.size)
    if "fertility" in tex:
        fert = np.clip(tex["fertility"][..., 0].reshape(-1)[cells].astype(np.float64), 0.0, 1.0)
        stretch += float(p.fertility_weight) * (1.0 - fert)
    if "elevation" in tex:
        slope = texture_slope(tex["elevation"][..., 0], np.sqrt(texel_km2)).reshape(-1)[cells]
        stretch += float(p.slope_weight) * np.clip(slope / max(float(p.slope_saturation), 1e-9), 0.0, 1.0)
    if "forest_density" in tex:
        forest = np.clip(tex["forest_density"][..., 0].reshape(-1)[cells].astype(np.float64), 0.0, 1.0)
        stretch += float(p.forest_weight) * forest
    rng = np.random.default_rng([int(seed), _FARM_STREAM])
    stretch *= 1.0 + float(p.edge_roughness) * rng.random(cells.size)
    taken = np.full(cells.size, -1, dtype=np.int64)

    # ── market gardens: each city's cheapest free texels by its footprint, nearest city first ──
    if n_cities and footprints is not None and len(footprints.city):
        rings = np.array([garden_ring_km(footprints.population[c], footprints.target_km2[c], p)
                          for c in range(n_cities)])
        quota = np.round(np.asarray(footprints.population) * float(p.garden_m2_per_person) * 1e-6
                         / texel_km2).astype(np.int64)
        floor = km_to_rad(p.city_garden_ring_min_km)
        fp_pos = local.positions.reshape(-1, 3)[np.flatnonzero(footprints.owner.reshape(-1) >= 0)]
        fp_city = footprints.owner.reshape(-1)[footprints.owner.reshape(-1) >= 0]
        # Distance relative to the median stretch: wider over good land.  Search past the
        # all-land ring: water and poor ground push gardens out.
        rel = stretch / float(np.median(stretch))
        reach = 2.0 * km_to_rad(float(rings.max())) / float(rel.min()) + 1e-12
        d, near = KDTree(fp_pos).query(pos, distance_upper_bound=reach, workers=-1)
        hit = np.isfinite(d)
        city = np.where(hit, fp_city[np.minimum(near, len(fp_city) - 1)], -1)
        cost = np.where(hit, d * rel, np.inf)
        order = np.lexsort((cost, city))
        order = order[hit[order]]
        c_sorted = city[order]
        starts = np.searchsorted(c_sorted, np.arange(n_cities))
        rank = np.arange(order.size) - starts[c_sorted]
        keep = (rank < quota[c_sorted]) | (cost[order] <= floor)
        got = order[keep]
        taken[got] = city[got] + len(v_pop) + len(t_pop)
        areas[FARM_CITY] = np.bincount(city[got], minlength=n_cities) * texel_km2

    # ── towns and villages: cheapest (settlement, texel) pair first ─────────────
    s_pos = np.concatenate([v_pos, t_pos])
    s_quota = np.round(np.concatenate([v_target, t_target]) / texel_km2).astype(np.int64)
    if len(s_pos):
        # Search wide enough for poor ground, jitter and neighbours to push fields out.
        r_disc = np.sqrt(np.maximum(s_quota, 1) * texel_km2 / np.pi)
        reach = (3.0 * r_disc + 2.0 * np.sqrt(texel_km2)) * (1.0 + float(p.edge_roughness)) / EARTH_RADIUS_KM
        lists = tree.query_ball_point(s_pos, reach, workers=-1)
        counts = np.array([len(x) for x in lists], dtype=np.int64)
        texel = (np.concatenate([np.asarray(x, dtype=np.int64) for x in lists]) if counts.sum()
                 else np.zeros(0, np.int64))
        who = np.repeat(np.arange(len(s_pos), dtype=np.int64), counts)
        if channels is not None and len(v_pos):
            # A village's fields stop at a river too wide to ford.
            barriers = channels.at_least(float(p.river_barrier_width_m) * 1e-3)
            village = who < len(v_pos)
            far_bank = np.zeros(who.size, dtype=bool)
            far_bank[village] = crossing_pairs(barriers, v_pos, who[village], pos[texel[village]], reach[:len(v_pos)])
            who, texel = who[~far_bank], texel[~far_bank]
        cost = np.linalg.norm(pos[texel] - s_pos[who], axis=1) * stretch[texel]
        order = np.argsort(cost, kind="stable").astype(np.int64)
        owner_s = np.full(cells.size, -1, dtype=np.int64)
        owner_s[taken >= 0] = len(s_pos)          # gardens are already spoken for
        claimed = _claim(order, who, texel, s_quota, owner_s)
        mine = (owner_s >= 0) & (owner_s < len(s_pos))
        taken[mine] = owner_s[mine]
        areas[FARM_VILLAGE] = claimed[:len(v_pos)] * texel_km2
        areas[FARM_TOWN] = claimed[len(v_pos):] * texel_km2

    k = np.full(cells.size, -1, dtype=np.int8)
    o = np.full(cells.size, -1, dtype=np.int32)
    n_v, n_t = len(v_pos), len(t_pos)
    k[(taken >= 0) & (taken < n_v)] = FARM_VILLAGE
    k[(taken >= n_v) & (taken < n_v + n_t)] = FARM_TOWN
    k[taken >= n_v + n_t] = FARM_CITY
    o[taken >= 0] = taken[taken >= 0] - np.where(
        k[taken >= 0] == FARM_VILLAGE, 0, np.where(k[taken >= 0] == FARM_TOWN, n_v, n_v + n_t))
    kind.reshape(-1)[cells] = k
    owner.reshape(-1)[cells] = o
    return result()


def farmland_rgba(farmland: Farmland, show: tuple[bool, bool, bool] = (True, True, True)) -> np.ndarray:
    """``(M, T, T, 4)`` RGBA of the kinds ``show`` picks (villages, towns, cities),
    with fixed per-texel brightness jitter for a patchwork look."""
    rgba = np.zeros(farmland.kind.shape + (4,), dtype=np.float32)
    shade = 1.0 + _PATCHWORK * (2.0 * np.random.default_rng(_FARM_STREAM).random(farmland.kind.shape) - 1.0)
    for k, on in enumerate(show):
        mask = farmland.kind == k
        if not on or not mask.any():
            continue
        r, g, b, a = FARM_KIND_COLORS[k]
        rgba[mask, 0] = np.clip(r * shade[mask], 0.0, 1.0)
        rgba[mask, 1] = np.clip(g * shade[mask], 0.0, 1.0)
        rgba[mask, 2] = np.clip(b * shade[mask], 0.0, 1.0)
        rgba[mask, 3] = a
    return rgba
