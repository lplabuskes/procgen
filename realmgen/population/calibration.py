"""Population & Calibration — Stage 4 of the realm layer.

Populations are in real people:

* **Rural population** — per face: ``D_max · fertility^γ · area_km²``, the
  countryside's Malthusian ceiling (the food base).  γ ≈ 2 keeps fertile valleys
  dense while the world total stays medieval-Earth scale (~400 M).
* **Carrying capacity** — per settlement: the food base summed over its
  hinterland shares — the food the region grows, not the city's population.
* **Urban population** — ``s · carrying_cap + β · trade_flow``.  The additive trade
  term lets a thin-hinterland chokepoint grow on imported grain
  (Constantinople/Venice).  Below an absolute floor a settlement is culled.

Hinterlands are soft: each face's food base is split among bidding settlements,
so a small city beside a large one still gets a share.  The population stage
relaxes trade → population → trade for a few passes.
"""
from __future__ import annotations

import dataclasses
from typing import List

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as sp_dijkstra

from realmgen.placement.types import Settlement

EARTH_RADIUS_KM = 6371.0
KM2_PER_STERADIAN = EARTH_RADIUS_KM ** 2  # ≈ 40.6 M km² per steradian


def compute_food_base(
    fertility: np.ndarray,
    face_areas: np.ndarray,
    *,
    density_max: float,
    fertility_exponent: float = 2.0,
) -> np.ndarray:
    """Per-face food base in people: ``D_max · fertility^γ · area_km²``.

    γ > 1 because fertility spans ~1 order of magnitude while real density spans
    ~3; a linear law can't give both a realistic total and a dense core.

    Args:
        fertility: (N,) in [0, 1].
        face_areas: (N,) solid-angle areas (steradians).
        density_max: people/km² at fertility 1.
    """
    fert = np.clip(np.asarray(fertility, dtype=np.float64), 0.0, 1.0)
    area_km2 = np.asarray(face_areas, dtype=np.float64) * KM2_PER_STERADIAN
    return float(density_max) * fert ** float(fertility_exponent) * area_km2


def split_fertility(
    fertility_tex: np.ndarray,
    water_tex: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Each face's farmland and fishing grounds, from its texture.

    Args:
        fertility_tex: (N, H, W); water pixels hold aquatic fertility.
        water_tex: (N, H, W) bool — ocean ∪ lake.

    Returns:
        ``(land_fertility, land_fraction, water_fertility, water_fraction)``, each
        (N,): mean fertility over land (water) pixels and their pixel share.
    """
    fert = np.clip(np.asarray(fertility_tex, dtype=np.float64), 0.0, 1.0)
    water = np.asarray(water_tex, dtype=bool)
    n_px = float(water.shape[1] * water.shape[2])
    n_water = water.sum(axis=(1, 2))
    n_land = n_px - n_water
    land_fert = np.where(water, 0.0, fert).sum(axis=(1, 2)) / np.maximum(n_land, 1)
    water_fert = np.where(water, fert, 0.0).sum(axis=(1, 2)) / np.maximum(n_water, 1)
    return land_fert, n_land / n_px, water_fert, n_water / n_px


def compute_farm_and_fish_food(
    fertility_tex: np.ndarray,
    water_tex: np.ndarray,
    face_areas: np.ndarray,
    *,
    density_max: float,
    fishing_density_max: float,
    fertility_exponent: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-face ``(farm, fish)`` food base in people, each (N,).

    :func:`compute_food_base` over the land pixels at ``density_max`` and the
    water pixels at ``fishing_density_max``.
    """
    land_fert, land_frac, water_fert, water_frac = split_fertility(fertility_tex, water_tex)
    areas = np.asarray(face_areas, dtype=np.float64)
    farm = compute_food_base(
        land_fert, areas * land_frac, density_max=density_max, fertility_exponent=fertility_exponent)
    fish = compute_food_base(
        water_fert, areas * water_frac, density_max=fishing_density_max, fertility_exponent=fertility_exponent)
    return farm, fish


def compute_rural_population(
    food_base: np.ndarray,
    is_water: np.ndarray,
) -> np.ndarray:
    """Per-face rural population: the food base, masked to land (water faces house none)."""
    return np.where(np.asarray(is_water, dtype=bool), 0.0, np.asarray(food_base, dtype=np.float64))


def compute_hinterland_shares(
    n_faces: int,
    settlements: List[Settlement],
    cost_graph: csr_matrix,
    *,
    max_distance: float,
    influence_scale: float = 4.5,
    reach_scale: float = 9.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Soft hinterland assignment — fractional capacity shares + display owner.

    Each settlement *k* bids for every face *i*::

        influence[k]    = log(max_pop / pop[k])          # 0 for top settlement
        reach[k]        = max_distance * exp(-influence[k] / reach_scale)
        proximity[k, i] = relu(1 - dist[k, i] / reach[k])       # in [0, 1]
        pull[k]         = exp(-influence[k] / influence_scale)   # in (0, 1]
        bid[k, i]       = proximity[k, i] * pull[k]

    Each face's capacity is split among bidders in proportion to their bids, so a
    village next to a city still claims a nonzero share (and can survive culling).

    Args:
        cost_graph: (N, N) land-cost CSR, used undirected.  A water-permeable
            variant lets coastal settlements claim fishing grounds.
        max_distance: cost-distance reach of the top settlement.
        influence_scale / reach_scale: e-folding scales of pull / reach; smaller
            widens the gap between large and small settlements.

    Returns:
        ``(shares, owner)``: ``shares`` (S, N), each column summing to 1 where any
        settlement is in range, else 0; ``owner`` (N,) int32 highest bidder, -1 if none.
    """
    if not settlements:
        return (
            np.zeros((0, n_faces), dtype=np.float64),
            np.full(n_faces, -1, dtype=np.int32),
        )

    s_nodes = np.array([s.face_id for s in settlements], dtype=np.int32)
    populations = np.array([s.population for s in settlements], dtype=np.float64)
    max_pop = float(populations.max())
    if max_pop <= 0.0:
        influence = np.zeros_like(populations)
    else:
        min_pop = max_pop * 1e-3
        influence = np.log(max_pop / np.maximum(populations, min_pop))

    # Nothing past max_distance (the widest reach) bids, so the search stops there.
    md = max(float(max_distance), 1e-9)
    dist = sp_dijkstra(
        cost_graph, directed=False, indices=s_nodes, return_predecessors=False, limit=md,
    )  # (S, N)

    scale = max(float(influence_scale), 1e-9)
    reach = max(float(reach_scale), 1e-9)
    # reach_scale=inf → uniform md.
    md_k = md * np.exp(-influence / reach)                  # (S,)
    proximity = np.maximum(1.0 - dist / md_k[:, np.newaxis], 0.0)   # (S, N); inf → 0
    pull = np.exp(-influence / scale)                       # (S,)
    bid = proximity * pull[:, np.newaxis]                   # (S, N)

    total = bid.sum(axis=0)                                 # (N,)
    in_range = total > 0.0
    shares = np.zeros_like(bid)
    np.divide(bid, total, out=shares, where=in_range)       # broadcast over rows

    owner = np.where(in_range, np.argmax(bid, axis=0), -1).astype(np.int32)
    return shares, owner


def compute_carrying_capacity(
    shares: np.ndarray,
    food_base: np.ndarray,
) -> np.ndarray:
    """``cap[k] = Σ_i shares[k, i] · food_base[i]`` — (S,) people the hinterland feeds.

    This is the region's food base, not the settlement's population.
    """
    shares = np.asarray(shares, dtype=np.float64)
    if shares.size == 0:
        return np.zeros(shares.shape[0], dtype=np.float64)
    return shares @ np.asarray(food_base, dtype=np.float64)


def compute_trade_wealth(
    settlements: List[Settlement],
    face_flow: np.ndarray,
) -> np.ndarray:
    """(S,) trade flow at each settlement's face, as a wealth proxy."""
    flow = np.asarray(face_flow, dtype=np.float64)
    return np.array([float(flow[s.face_id]) for s in settlements], dtype=np.float64)


def compute_urban_population(
    carrying_cap: np.ndarray,
    trade_wealth: np.ndarray,
    *,
    surplus_rate: float = 0.15,
    trade_food_weight: float = 0.0,
) -> np.ndarray:
    """Urban headcount ``s · carrying_cap + β · trade_wealth``, (S,) people.

    The trade term is additive (imported grain doesn't multiply the local harvest),
    so a poor-but-strategic chokepoint can outgrow what its land could feed.
    """
    cap = np.maximum(carrying_cap, 0.0)
    wealth = np.maximum(trade_wealth, 0.0)
    return float(surplus_rate) * cap + float(trade_food_weight) * wealth


def compute_miners(
    population: np.ndarray,
    trade_people: np.ndarray,
    own_goods_share: np.ndarray,
    deposit_owner: np.ndarray,
    deposit_value: np.ndarray,
    workable: np.ndarray,
    *,
    mining_share: float,
    city_min: float,
) -> np.ndarray:
    """Miners per deposit, ``(D,)`` people, taken out of the cities that work them.

    Of a city's ``trade_people``, the part fed by its own goods
    (``own_goods_share``) times the ``workable`` deposits' share of its owned value
    comes out of the ground; ``mining_share`` of those live at the deposits instead,
    split by value.  No city is cut below ``city_min``.

    Args:
        population: (S,) city people before the miners leave.
        trade_people: (S,) ``β · trade_wealth``.
        own_goods_share: (S,) in [0, 1].
        deposit_owner: (D,) settlement index, -1 for none.
        deposit_value: (D,) credited abundance × rarity.
        workable: (D,) bool — deposits that employ miners (minerals on land).

    Returns:
        (D,) miners; their sum per owner is what that city loses.
    """
    pop = np.asarray(population, dtype=np.float64)
    n_s = pop.size
    owner = np.asarray(deposit_owner, dtype=np.int64)
    value = np.where(owner >= 0, np.maximum(np.asarray(deposit_value, dtype=np.float64), 0.0), 0.0)
    owned = owner >= 0
    total = np.bincount(owner[owned], weights=value[owned], minlength=n_s)
    mined = np.bincount(owner[owned], weights=(value * workable)[owned], minlength=n_s)
    frac = np.where(total > 0, mined / np.maximum(total, 1e-300), 0.0)
    city_miners = (
        float(mining_share) * np.maximum(np.asarray(trade_people, dtype=np.float64), 0.0)
        * np.clip(np.asarray(own_goods_share, dtype=np.float64), 0.0, 1.0) * frac
    )
    city_miners = np.minimum(city_miners, np.maximum(pop - float(city_min), 0.0))
    per_value = np.where(mined > 0, city_miners / np.maximum(mined, 1e-300), 0.0)
    return np.where(owned & workable, value * per_value[np.maximum(owner, 0)], 0.0)


def cull_settlements(
    settlements: List[Settlement],
    population: np.ndarray,
    *,
    min_population: float = 0.0,
) -> tuple[list[Settlement], np.ndarray, np.ndarray]:
    """Drop settlements with ``population < min_population`` (people).

    Returns:
        ``(settlements, population, kept)`` — filtered list, ``(S',)`` population,
        and the ``(S,)`` keep mask (for filtering other aligned arrays).
    """
    pop = np.asarray(population, dtype=np.float64)
    kept = pop >= float(min_population)
    kept_s = [s for s, k in zip(settlements, kept) if k]
    return kept_s, pop[kept], kept


def apply_population_to_settlements(
    settlements: List[Settlement],
    population: np.ndarray,
) -> list[Settlement]:
    """Return a new list with ``Settlement.population`` set from *population* (S,)."""
    pop = np.asarray(population, dtype=np.float64)
    return [dataclasses.replace(s, population=float(p)) for s, p in zip(settlements, pop)]
