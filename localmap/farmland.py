"""Farmland: the fields and gardens around a local map's settlements.

How much ground each settlement farms (medieval rules of thumb):

- **Villages** farm their own fields: ``ha_per_farmer`` (1.5 ha — about 1 ha of
  cropland per person under three-field rotation, fallow included, plus meadow
  and pasture) for every villager.
- **Towns**: about half the townspeople (``town_farming_share``) work the town's
  own fields; the rest of its food comes from the villages around it.
- **Mining villages and towns** live off the mine and buy their food: only
  ``mining_farming_share`` (0.1) of their people farm, so their fields are a
  narrow band, or none.
- **Cities** are fed by their trade hinterland, not by fields of their own, so
  their farmland is a ring of market gardens and orchards
  ``city_garden_ring_km`` (1.5 km) wide outside their built-up footprint
  (:mod:`localmap.cities`) — on the map's typical ground; the stretch below,
  relative to its median, narrows it over poor ground and widens it over good.

Fields are claimed texel by texel (~0.12 km² each), outward from each settlement
like a city's footprint, but preferring good land: a texel's distance is
stretched by ``1 + fertility_weight·(1 − fertility) + slope_weight·slope +
forest_weight·forest`` (slope as a fraction of ``slope_saturation``), and
lightly jittered (``edge_roughness``), so fields fill valleys and skip steep or
forested ground.  The garden rings go first; then towns and villages claim
together, cheapest (settlement, texel) pair first, each until its quota is met,
so neighbours share the ground between them instead of the first-served taking
it.  Water and built-up city texels are never claimed.

Display-only and Qt-free; deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial import KDTree

from core.params import param_field as _f, paramclass
from localmap.cities import CityFootprints
from region.resources import refined_slope
from region.upsample import EARTH_RADIUS_KM, RegionFields

try:
    from numba import njit
except ImportError:  # pragma: no cover - numba is an optional accelerator
    def njit(func=None, **_):  # type: ignore[misc]
        return func if func is not None else (lambda f: f)

# Who a farmland texel belongs to.
FARM_VILLAGE, FARM_TOWN, FARM_CITY = range(3)
FARM_KIND_NAMES = ("Village fields", "Town fields", "Market gardens")
# RGBA fill per kind: fields a translucent wheat, gardens a dark olive (apart from
# the tier colours a city footprint is drawn in).
FARM_KIND_COLORS: list[tuple[float, float, float, float]] = [
    (0.93, 0.84, 0.46, 0.55),
    (0.93, 0.84, 0.46, 0.55),
    (0.30, 0.46, 0.18, 0.70),
]
# ± fraction of brightness each texel varies by, so fields read as a patchwork.
_PATCHWORK = 0.10
# km² per hectare.
_KM2_PER_HA = 0.01
# RNG stream tag for the edge jitter.
_FARM_STREAM = 17


@paramclass
class FarmlandParams:
    ha_per_farmer: float = _f(1.5, label="Farmland per farmer (ha)", min=0.1, max=5.0, step=0.1)
    town_farming_share: float = _f(0.5, label="Townspeople who farm", min=0.0, max=1.0, step=0.05)
    mining_farming_share: float = _f(
        0.1, label="Miners' settlements: people who farm", min=0.0, max=1.0, step=0.05)
    city_garden_ring_km: float = _f(1.5, label="City garden ring (km)", min=0.0, max=5.0, step=0.25)
    fertility_weight: float = _f(1.0, label="Poor-soil stretch", min=0.0, max=5.0, step=0.1)
    slope_weight: float = _f(2.0, label="Steep-slope stretch", min=0.0, max=10.0, step=0.1)
    forest_weight: float = _f(1.0, label="Forest stretch", min=0.0, max=5.0, step=0.1)
    slope_saturation: float = _f(0.15, label="Slope saturation (km/km)", min=0.01, max=1.0, step=0.01)
    edge_roughness: float = _f(0.25, label="Field edge roughness (× distance)", min=0.0, max=1.0, step=0.05)


@dataclass
class Farmland:
    """The farmland texels of a local map and whose they are.

    ``owner`` indexes the settlements of the kind ``kind`` names, in the order
    they were passed (the map's villages; its ``town_index`` / ``city_index``).
    """

    kind: np.ndarray        # (M, T, T) int8 FARM_* code, -1 where not farmed
    owner: np.ndarray       # (M, T, T) int32 settlement index within its kind, -1 where not farmed
    target_km2: tuple[np.ndarray, np.ndarray, np.ndarray]    # per kind: (N_kind,) wanted
    area_km2: tuple[np.ndarray, np.ndarray, np.ndarray]      # per kind: (N_kind,) claimed


@njit(cache=True)
def _claim(order, who, texel, quota, owner):  # pragma: no cover - numba-compiled
    """Walk (settlement, texel) pairs cheapest first; each free texel goes to the
    pair's settlement while it still wants ground.  ``owner`` is updated in place."""
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


def _chord(km: float) -> float:
    return float(km) / EARTH_RADIUS_KM


def place_farmland(
    local: RegionFields,
    village_positions: np.ndarray, village_population: np.ndarray,
    town_positions: np.ndarray, town_population: np.ndarray,
    footprints: Optional[CityFootprints],
    seed: int,
    params: Optional[FarmlandParams] = None,
    *,
    village_mining: Optional[np.ndarray] = None,
    town_mining: Optional[np.ndarray] = None,
) -> Farmland:
    """Claim every settlement's farmland.  See the module docstring.

    ``village_mining`` / ``town_mining`` (bool, per village / town) flag the
    mining settlements, which farm only ``mining_farming_share`` of their people.
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
    t_share = np.where(_flags(town_mining, len(t_pop)), mining_share, float(p.town_farming_share))
    v_target = v_pop * v_share * float(p.ha_per_farmer) * _KM2_PER_HA
    t_target = t_pop * t_share * float(p.ha_per_farmer) * _KM2_PER_HA
    areas: list[np.ndarray] = [np.zeros(len(v_pop)), np.zeros(len(t_pop)), np.zeros(n_cities)]

    def result() -> Farmland:
        # Gardens have no quota: a city wants whatever its ring holds.
        return Farmland(
            kind=kind, owner=owner, target_km2=(v_target, t_target, areas[FARM_CITY].copy()),
            area_km2=(areas[FARM_VILLAGE], areas[FARM_TOWN], areas[FARM_CITY]),
        )

    water = np.zeros(shape, dtype=bool)
    for key in ("ocean_mask", "lake_mask"):
        if key in local.textures:
            water |= local.textures[key][..., 0] > 0.5
    built = footprints.owner >= 0 if footprints is not None else np.zeros(shape, dtype=bool)
    free = local.in_hex & ~water & ~built
    cells = np.flatnonzero(free.reshape(-1))
    if cells.size == 0:
        return result()
    pos = local.positions.reshape(-1, 3)[cells]
    tree = KDTree(pos)

    # How much farther a texel's poor, steep or wooded ground makes it, jittered.
    tex = local.textures
    stretch = np.ones(cells.size)
    if "fertility" in tex:
        fert = np.clip(tex["fertility"][..., 0].reshape(-1)[cells].astype(np.float64), 0.0, 1.0)
        stretch += float(p.fertility_weight) * (1.0 - fert)
    if "elevation" in tex:
        slope = refined_slope(tex["elevation"][..., 0], np.sqrt(texel_km2)).reshape(-1)[cells]
        stretch += float(p.slope_weight) * np.clip(slope / max(float(p.slope_saturation), 1e-9), 0.0, 1.0)
    if "forest_density" in tex:
        forest = np.clip(tex["forest_density"][..., 0].reshape(-1)[cells].astype(np.float64), 0.0, 1.0)
        stretch += float(p.forest_weight) * forest
    rng = np.random.default_rng([int(seed), _FARM_STREAM])
    stretch *= 1.0 + float(p.edge_roughness) * rng.random(cells.size)
    taken = np.full(cells.size, -1, dtype=np.int64)

    # ── market gardens: a ring round each city's footprint, nearest city first ──
    ring = _chord(p.city_garden_ring_km)
    if n_cities and ring > 0 and footprints is not None and len(footprints.city):
        fp_pos = local.positions.reshape(-1, 3)[np.flatnonzero(footprints.owner.reshape(-1) >= 0)]
        fp_city = footprints.owner.reshape(-1)[footprints.owner.reshape(-1) >= 0]
        # The ring is measured on the map's typical ground (its median stretch):
        # wider over good land, narrower over poor.
        rel = stretch / float(np.median(stretch))
        d, near = KDTree(fp_pos).query(pos, distance_upper_bound=ring / float(rel.min()) + 1e-12, workers=-1)
        hit = np.isfinite(d)
        hit[hit] = d[hit] * rel[hit] <= ring
        city = np.where(hit, fp_city[np.minimum(near, len(fp_city) - 1)], -1)
        taken[hit] = city[hit] + len(v_pop) + len(t_pop)
        areas[FARM_CITY] = np.bincount(city[hit], minlength=n_cities) * texel_km2

    # ── towns and villages: cheapest (settlement, texel) pair first ─────────────
    s_pos = np.concatenate([v_pos, t_pos])
    s_quota = np.round(np.concatenate([v_target, t_target]) / texel_km2).astype(np.int64)
    if len(s_pos):
        # Search wide enough that poor ground, the jitter and neighbours can
        # each push a settlement's fields out.
        r_disc = np.sqrt(np.maximum(s_quota, 1) * texel_km2 / np.pi)
        reach = (3.0 * r_disc + 2.0 * np.sqrt(texel_km2)) * (1.0 + float(p.edge_roughness)) / EARTH_RADIUS_KM
        lists = tree.query_ball_point(s_pos, reach, workers=-1)
        counts = np.array([len(x) for x in lists], dtype=np.int64)
        texel = (np.concatenate([np.asarray(x, dtype=np.int64) for x in lists]) if counts.sum()
                 else np.zeros(0, np.int64))
        who = np.repeat(np.arange(len(s_pos), dtype=np.int64), counts)
        cost = np.linalg.norm(pos[texel] - s_pos[who], axis=1) * stretch[texel]
        order = np.argsort(cost, kind="stable").astype(np.int64)
        owner_s = np.full(cells.size, -1, dtype=np.int64)
        owner_s[taken >= 0] = len(s_pos)          # gardens are already spoken for
        claimed = _claim(order, who, texel, s_quota, owner_s)
        mine = (owner_s >= 0) & (owner_s < len(s_pos))
        taken[mine] = owner_s[mine]
        areas[FARM_VILLAGE] = claimed[:len(v_pos)] * texel_km2
        areas[FARM_TOWN] = claimed[len(v_pos):] * texel_km2

    # Back to (kind, index within kind) on the texture grid.
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
    """``(M, T, T, 4)`` float RGBA of the farmland of the kinds ``show`` picks
    (villages, towns, cities), transparent elsewhere; each texel's brightness
    varies a little (a fixed random draw) so the fields read as a patchwork."""
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
