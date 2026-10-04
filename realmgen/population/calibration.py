"""Population & Calibration — Stage 4 of the realm layer.

Populations are in **real people**.  Three distinct quantities:

* **Rural population** — per *face*: ``D_max · fertility^γ · area_km²``, the
  Malthusian ceiling of the countryside (the food base).  γ ≈ 2 restores the
  density contrast a linear law flattens: fertile river valleys run ~13
  people/km², marginal land near zero, while the world total stays anchored
  to a medieval-Earth scale (~400 M).
* **Carrying capacity** — per *settlement*: the food base summed over its
  claimed hinterland shares.  This is the food the region grows, **not** the
  settlement's population.
* **Urban population** — per *settlement*:
  ``s · carrying_cap + β · trade_flow``.  The agricultural surplus rate *s*
  (~0.1–0.2) converts regional food into city mouths; the **additive** trade
  term lets a thin-hinterland chokepoint grow large on imported grain
  (Constantinople/Venice).  A settlement exists as a city only if its urban
  population clears an absolute floor.

Mechanisms:

1. **Hinterland assignment** — soft, fractional.  Each settlement bids for
   every face within ``max_distance`` cost-distance; the bid combines a
   linear proximity falloff with an influence-derived pull, and each face's
   food base is split among bidders in proportion to their bids.  A small
   city beside a large one therefore still captures a share of the
   surrounding land instead of being shut out.  The *displayed* hinterland is
   the single highest bidder per face.  When the caller supplies a
   *water-permeable* cost graph (the population stage does), a coastal
   settlement also bids for a *limited* band of nearby ocean/lake, pulling
   its fishing-grounds productivity (the aquatic component of
   ``fertility_mean``) into its food base; the band stays small because water
   steps are made deliberately expensive.

2. **Iterative relaxation** — carrying capacity and trade flow combine into
   a per-settlement urban population.  Running trade → population → trade for
   a few passes lets hub reinforcement stabilise without solving the loop
   analytically.  After each pass, settlements below the urban-formation
   floor are culled.
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

    The number of people a face's land (or fishing grounds — coastal water
    faces carry aquatic productivity in ``fertility_mean``) can feed at a
    medieval tech level.  ``fertility_exponent`` γ > 1 restores density
    contrast: the fertility field spans ~1 order of magnitude while real
    population density spans ~3, so a linear law cannot give both a realistic
    world total and a dense agricultural core.

    Args:
        fertility: (N,) float in [0, 1] — per-face ``fertility_mean``.
        face_areas: (N,) float64 — per-face solid-angle areas (steradians).
        density_max: people/km² on perfectly fertile land (fertility = 1).
        fertility_exponent: γ — fertility→density contrast (γ ≈ 2).

    Returns:
        (N,) float64 — people each face can feed.
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
        fertility_tex: (N, H, W) fertility; water pixels hold aquatic fertility.
        water_tex: (N, H, W) bool — ocean ∪ lake pixels.

    Returns:
        ``(land_fertility, land_fraction, water_fertility, water_fraction)``,
        each (N,) float64: the mean fertility over the face's land (water)
        pixels and the share of its pixels that are land (water).
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
    """Per-face food base split into farmland and fishing grounds, in people.

    Each is :func:`compute_food_base` over its own part of the face: the land
    pixels at ``density_max``, the water pixels (aquatic fertility) at
    ``fishing_density_max``.  A coastal land face's shallows thus feed fishers
    who live on its shore, and a water face's fishing grounds feed the cities
    whose hinterland reaches them.

    Returns:
        ``(farm, fish)`` — (N,) float64 each.
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
    """Per-face rural population: the food base, masked to land.

    The countryside fills to its Malthusian ceiling, so rural population *is*
    the food base on land faces.  Water faces feed people (fishing grounds)
    but house none.

    Args:
        food_base: (N,) float64 from :func:`compute_food_base`.
        is_water: (N,) bool — ocean ∪ lake mask.

    Returns:
        (N,) float64 — rural people living on each face.
    """
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

    Runs undirected multi-source Dijkstra from all settlement nodes, then has
    each settlement *k* bid for every face *i*::

        influence[k]    = log(max_pop / pop[k])          # 0 for top settlement
        reach[k]        = max_distance * exp(-influence[k] / reach_scale)
        proximity[k, i] = relu(1 - dist[k, i] / reach[k])       # in [0, 1]
        pull[k]         = exp(-influence[k] / influence_scale)   # in (0, 1]
        bid[k, i]       = proximity[k, i] * pull[k]

    ``influence`` is derived from each settlement's current population: the
    highest-population settlement gets influence 0 (widest reach, strongest
    pull) and lower-population settlements get proportionally larger influence.
    This means hinterland priority emerges from population rather than from
    fixed tier labels.

    ``max_distance`` caps the largest market area (the top-population settlement
    at influence 0).  ``reach_scale`` controls how steeply catchment radius
    shrinks with influence; smaller values give more differentiation.
    ``influence_scale`` controls how steeply pull decays; smaller values
    widen the pull gap between large and small settlements.

    Each face's carrying capacity is divided among its bidders in proportion to
    their bids, so a low-pull village next to a high-pull city still claims a
    nonzero share of the nearby land (and can survive culling).

    Args:
        n_faces: number of mesh faces (N).
        settlements: list of :class:`~realmgen.placement.types.Settlement`
            (reads ``face_id`` and ``population``).
        cost_graph: (N, N) directed land-cost CSR matrix — used undirected. Pass
            a water-permeable variant to let coastal settlements claim a limited
            band of ocean/lake for fishing grounds.
        max_distance: hard cap (cost-distance) on the widest hinterland reach.
        influence_scale: e-folding scale for ``pull``; smaller widens the pull
            gap between settlements.
        reach_scale: e-folding scale for catchment radius; smaller gives higher-
            population centres a proportionally larger market area.

    Returns:
        ``(shares, owner)`` where:

        * ``shares`` is ``(S, N)`` float64 — fraction of face *i*'s capacity
          claimed by settlement *k*; each column sums to 1 where any settlement
          is in range, else all zero.
        * ``owner`` is ``(N,)`` int32 — index of the single highest bidder per
          face (for display), or -1 where no settlement is within range.
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

    # dist[k, i] = undirected cost-distance from settlements[k] to face i
    # (+inf where unreachable).
    dist = sp_dijkstra(
        cost_graph, directed=False, indices=s_nodes, return_predecessors=False,
    )  # (S, N)

    md = max(float(max_distance), 1e-9)
    scale = max(float(influence_scale), 1e-9)
    reach = max(float(reach_scale), 1e-9)
    # Per-settlement catchment radius: higher-order centres (lower influence)
    # draw from a proportionally larger market area. reach_scale=inf → uniform md.
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
    """Per-settlement carrying capacity from fractional hinterland shares.

    ``cap[k] = Σ_i shares[k, i] · food_base[i]`` — the people the settlement's
    claimed hinterland can feed.  This is the *food base* of the region, not
    the settlement's population; the urban headcount is the surplus this
    ceiling throws off (see :func:`compute_urban_population`).

    Args:
        shares: (S, N) float64 from :func:`compute_hinterland_shares`.
        food_base: (N,) float64 from :func:`compute_food_base` (includes
            aquatic fishing productivity on coastal ocean/lake faces, so water
            claimed via a permeable hinterland graph contributes here).

    Returns:
        (S,) float64 — people the hinterland feeds.
    """
    shares = np.asarray(shares, dtype=np.float64)
    if shares.size == 0:
        return np.zeros(shares.shape[0], dtype=np.float64)
    return shares @ np.asarray(food_base, dtype=np.float64)


def compute_trade_wealth(
    settlements: List[Settlement],
    face_flow: np.ndarray,
) -> np.ndarray:
    """Extract trade flow at each settlement's face as a wealth proxy.

    Args:
        settlements: list of :class:`~realmgen.placement.types.Settlement`.
        face_flow: (N,) float — cumulative trade flow per face.

    Returns:
        (S,) float64 — per-settlement trade wealth (non-negative).
    """
    flow = np.asarray(face_flow, dtype=np.float64)
    return np.array([float(flow[s.face_id]) for s in settlements], dtype=np.float64)


def compute_urban_population(
    carrying_cap: np.ndarray,
    trade_wealth: np.ndarray,
    *,
    surplus_rate: float = 0.15,
    trade_food_weight: float = 0.0,
) -> np.ndarray:
    """Urban headcount: local agricultural surplus plus imported food.

    Formula::

        urban_pop = s · carrying_cap + β · trade_wealth

    The surplus rate *s* is the urbanization ceiling — the fraction of the
    region's food that feeds city mouths rather than the farmers growing it.
    The trade term is **additive**: imported grain adds carrying capacity, it
    does not multiply the local harvest, so a poor-but-strategic chokepoint
    (Constantinople, Venice) can sustain a large urban core its own land could
    never feed.

    Args:
        carrying_cap: (S,) people from :func:`compute_carrying_capacity`.
        trade_wealth: (S,) trade flow from :func:`compute_trade_wealth`.
        surplus_rate: s — agricultural surplus rate (~0.10–0.20).
        trade_food_weight: β — people fed per unit of trade flow.

    Returns:
        (S,) float64 — urban population in people.
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

    A city's trade-fed people (``trade_people``) live on the flow at its face;
    ``own_goods_share`` of that flow is its own goods (deposit hauls in, exports
    out). The ``workable`` deposits' share of the value it owns
    (``deposit_value`` = credited abundance × rarity) is the part that comes out
    of the ground, and ``mining_share`` of those people dig it instead of
    living in the city: they live at the deposits, in proportion to each one's
    value. No city is cut below ``city_min``.

    Args:
        population: (S,) city people before the miners leave.
        trade_people: (S,) ``β · trade_wealth`` — the trade-fed part of it.
        own_goods_share: (S,) in [0, 1].
        deposit_owner: (D,) settlement index working each deposit, -1 for none.
        deposit_value: (D,) credited value of each deposit to its owner.
        workable: (D,) bool — deposits that employ miners (minerals on land).
        mining_share: share of the mined goods' trade people who are miners.
        city_min: floor no city is cut below.

    Returns:
        (D,) float64 — miners at each deposit; their sum per owner is what that
        city loses.
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
    """Remove settlements below the urban-formation floor.

    Any settlement with ``population < min_population`` (people) is dropped —
    a region without the surplus or trade to feed a genuine city has villages
    but no city, and disappears from the cities layer.  Culling is
    unconditional — display tiers are re-derived from population at render
    time, so there is nothing to downgrade.

    Args:
        settlements: list of :class:`~realmgen.placement.types.Settlement`.
        population: (S,) float64 aligned with *settlements*.
        min_population: absolute urban-population floor (people) to survive.

    Returns:
        ``(settlements, population, kept)`` — filtered list, the aligned
        ``(S',)`` population array, and the ``(S,)`` boolean keep mask over the
        input (for filtering other settlement-aligned arrays).
    """
    pop = np.asarray(population, dtype=np.float64)
    kept = pop >= float(min_population)
    kept_s = [s for s, k in zip(settlements, kept) if k]
    return kept_s, pop[kept], kept


def apply_population_to_settlements(
    settlements: List[Settlement],
    population: np.ndarray,
) -> list[Settlement]:
    """Return a new list with ``Settlement.population`` set from *population*.

    Args:
        settlements: original list.
        population: (S,) float64 aligned with *settlements*.

    Returns:
        New list of :class:`Settlement` objects with ``population`` filled in.
    """
    pop = np.asarray(population, dtype=np.float64)
    return [dataclasses.replace(s, population=float(p)) for s, p in zip(settlements, pop)]
