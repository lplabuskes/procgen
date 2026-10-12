"""Physical base cost surfaces for overland and maritime movement (per face).

- **Land** — biome difficulty + ruggedness (face relief, km), plus an anisotropic river term (cheap along a valley, dear
  to ford).  All open water (ocean ∪ lake) is impassable.
- **Sea** — coastal water cheapest, open ocean dearer, straits discounted, land impassable.  Lakes are
  sailable, and navigable rivers extend the network inland up to the fall line.

Navigable-river faces are the only overlap: passable on both surfaces.  The river term is directional,
so it is stored as per-face flow direction + magnitude and folded into edge costs by
:func:`build_land_edge_costs`, which realmgen reuses after adding danger.

Functions only need ``faces``, ``face_adjacency`` and ``face_centroids`` off the mesh (easy to stub).
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import List, TYPE_CHECKING

import numpy as np

from worldgen.climate.classification import N_BIOMES

if TYPE_CHECKING:
    from worldgen.cost.builder import SeaCostParams, StraitParams

EARTH_RADIUS_KM = 6371.0


# Relative overland difficulty per biome (index = BIOME_NAMES); temperate/steppe = 1.0.
# `biome_cost_scale` tunes its overall weight.
BIOME_TRAVERSAL_COST = np.array(
    [
        4.0,  # 0 Icecap
        2.0,  # 1 Tundra
        1.8,  # 2 Taiga
        1.0,  # 3 Temperate
        1.0,  # 4 Steppe
        1.2,  # 5 Savannah
        2.5,  # 6 Tropical
        1.5,  # 7 Semi-Arid
        3.0,  # 8 Desert
        4.0,  # 9 Wetland (per texel on region/local maps; the globe uses wetland_fraction)
    ],
    dtype=np.float32,
)
assert BIOME_TRAVERSAL_COST.shape == (N_BIOMES,), "biome cost table out of sync with BIOME_NAMES"


@dataclass
class CostSurfaces:
    """Per-face physical base traversal-cost layers.

    ``is_water = ~land_passable`` (ocean ∪ lake); ``navigable_river = land_passable & sea_passable``.

    Attributes:
        land_base_cost:  (N,) float32 — biome + ruggedness, meaningful where ``land_passable``.  The
            directional river term is applied per edge by ``build_land_edge_costs``.
        land_passable:   (N,) bool — False on open water.
        river_flow_dir:  (N, 3) float32 — unit tangent downstream direction; 0 where no river.
        river_flow_mag:  (N,) float32 — max upstream flow through the face; 0 where no river.
        river_navigability: (N,) float32 in [0, 1] — 1 placid, 0 at the rapids limit or no river.
        sea_base_cost:   (N,) float32 — meaningful where ``sea_passable``; 0 on land.
        sea_passable:    (N,) bool — ocean ∪ lake ∪ navigable river.
        strait_mask:     (N,) bool — narrow water chokepoints.
    """

    land_base_cost: np.ndarray
    land_passable: np.ndarray
    river_flow_dir: np.ndarray
    river_flow_mag: np.ndarray
    river_navigability: np.ndarray
    sea_base_cost: np.ndarray
    sea_passable: np.ndarray
    strait_mask: np.ndarray
    # (N, 3) float32 coast-blended mean wind direction; zero near coasts.  None without wind data.
    wind_dir: np.ndarray | None = None


# ── Land ─────────────────────────────────────────────────────────────────────


def compute_land_base_cost(
    face_biome: np.ndarray,
    face_relief_km: np.ndarray,
    is_water: np.ndarray,
    *,
    biome_cost_scale: float = 1.0,
    ruggedness_cost_per_km: float = 1.5,
    face_forest_density: np.ndarray | None = None,
    forest_cost_strength: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Isotropic per-face land cost = biome difficulty + ruggedness + forest penalty.

    Args:
        face_biome: (N,) int — biome index per face.
        face_relief_km: (N,) float — face-scale relief (km) the world's ruggedness
            stands for (:func:`worldgen.geology.ruggedness.face_relief`).
        is_water: (N,) bool — open water (ocean ∪ lake), impassable.
        face_forest_density: (N,) float in [0, 1] — mean forest density per face;
            None disables the forest term.
        forest_cost_strength: additive scale on forest density (0 = no effect).

    Returns:
        (land_base_cost (N,) float32, land_passable (N,) bool). Water faces keep a
        finite (meaningless) cost value; use the mask to gate them.
    """
    biome = np.clip(np.asarray(face_biome).astype(np.int64), 0, N_BIOMES - 1)
    cost = (
        biome_cost_scale * BIOME_TRAVERSAL_COST[biome]
        + ruggedness_cost_per_km * np.maximum(np.asarray(face_relief_km, dtype=np.float64), 0.0)
    ).astype(np.float32)
    if face_forest_density is not None and forest_cost_strength > 0.0:
        forest = np.clip(np.asarray(face_forest_density, dtype=np.float32), 0.0, 1.0)
        cost = cost + float(forest_cost_strength) * forest
    passable = ~np.asarray(is_water, dtype=bool)
    return cost, passable


def compute_river_flow_field(
    mesh,
    rivers,
    *,
    include_in_lake: bool = False,
    exclude_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-face downstream flow direction + magnitude from river geometry.

    Each segment's direction is attributed to its start and end faces; per face we keep the
    flow-weighted mean direction (unit, tangent plane) and the max flow seen.

    Args:
        mesh: object exposing ``faces`` and ``face_centroids``.
        rivers: iterable of ``River`` (worldgen.hydrology.types) or None.
        include_in_lake: if False, skip ``in_lake`` segments (they cross a lake surface, not land).
        exclude_mask: (N,) bool or None — faces left at zero flow (e.g. open water).

    Returns:
        (river_flow_dir (N, 3) float32 unit tangent, river_flow_mag (N,) float32).
    """
    n = len(mesh.faces)
    accum = np.zeros((n, 3), dtype=np.float64)
    mag = np.zeros(n, dtype=np.float64)
    excluded = None if exclude_mask is None else np.asarray(exclude_mask, dtype=bool)

    if rivers:
        for river in rivers:
            for seg in river.segments:
                if getattr(seg, "in_lake", False) and not include_in_lake:
                    continue
                start = np.asarray(seg.start.position, dtype=np.float64)
                end = np.asarray(seg.end.position, dtype=np.float64)
                d = end - start
                norm = float(np.linalg.norm(d))
                if norm < 1e-12:
                    continue
                d_unit = d / norm
                fr = float(seg.flow_rate)
                for fid in {int(seg.start.face_id), int(seg.end.face_id)}:
                    if 0 <= fid < n and (excluded is None or not excluded[fid]):
                        accum[fid] += fr * d_unit
                        if fr > mag[fid]:
                            mag[fid] = fr

    centroids = np.asarray(mesh.face_centroids, dtype=np.float64)
    radial = (accum * centroids).sum(axis=1, keepdims=True) * centroids
    tangent = accum - radial
    tnorm = np.linalg.norm(tangent, axis=1, keepdims=True)
    flow_dir = np.divide(tangent, tnorm, out=np.zeros_like(tangent), where=tnorm > 1e-12)
    return flow_dir.astype(np.float32), mag.astype(np.float32)


def compute_river_navigability(
    river_flow_mag: np.ndarray,
    face_relief_km: np.ndarray,
    *,
    navigable_ruggedness_km: float = 0.6,
) -> np.ndarray:
    """Per-face river navigability in [0, 1] — the fall-line taper.

    Falls linearly from 1 (flat) to 0 at ``navigable_ruggedness_km`` of local relief (rapids).  It tapers
    the land corridor discount (:func:`build_land_edge_costs`) and, via :func:`navigable_river_mask`,
    decides which river faces are sailable.  The fall line is where it reaches 0 along a river.

    Args:
        river_flow_mag: (N,) float — river flow per face (0 = none).
        face_relief_km: (N,) float — face-scale relief (km) from the world's ruggedness.
        navigable_ruggedness_km: local relief (km) at which navigability hits 0.

    Returns:
        (N,) float32; 0 where no river passes.
    """
    flow = np.asarray(river_flow_mag, dtype=np.float64)
    relief = np.maximum(np.asarray(face_relief_km, dtype=np.float64), 0.0)
    scale = max(float(navigable_ruggedness_km), 1e-9)
    flat = np.clip(1.0 - relief / scale, 0.0, 1.0)
    nav = np.where(flow > 0.0, flat, 0.0)
    return nav.astype(np.float32)


def navigable_river_mask(
    river_flow_mag: np.ndarray,
    river_navigability: np.ndarray,
    is_water: np.ndarray,
    *,
    navigable_flow_min: float = 500.0,
) -> np.ndarray:
    """(N,) bool: river faces that join the sailable network.

    Flow ≥ ``navigable_flow_min``, navigability > 0 (below the fall line), and not already water.
    """
    flow = np.asarray(river_flow_mag, dtype=np.float64)
    nav = np.asarray(river_navigability, dtype=np.float64)
    water = np.asarray(is_water, dtype=bool)
    return (flow >= float(navigable_flow_min)) & (nav > 0.0) & ~water


def _flatten_adjacency(mesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten ``face_adjacency`` into directed edges (src, dst) and CSR ``indptr`` (edges leaving face i)."""
    adj = mesh.face_adjacency
    indptr = np.zeros(len(adj) + 1, dtype=np.int64)
    src_list: List[int] = []
    dst_list: List[int] = []
    for i, nbrs in enumerate(adj):
        for j in nbrs:
            src_list.append(i)
            dst_list.append(int(j))
        indptr[i + 1] = indptr[i] + len(nbrs)
    return (
        np.asarray(src_list, dtype=np.int64),
        np.asarray(dst_list, dtype=np.int64),
        indptr,
    )


def build_land_edge_costs(
    mesh,
    land_cost: np.ndarray,
    land_passable: np.ndarray,
    river_flow_dir: np.ndarray,
    river_flow_mag: np.ndarray,
    river_navigability: np.ndarray | None = None,
    *,
    river_ford_strength: float = 2.0,
    river_corridor_strength: float = 0.5,
    river_flow_reference: float = 2000.0,
    river_min_mult: float = 0.4,
    river_max_mult: float = 5.0,
    settlement_mask: np.ndarray | None = None,
) -> List[np.ndarray]:
    """Directional per-edge land cost; ``costs[i][k]`` steps from face i to ``face_adjacency[i][k]``.

    ``cost = mean(land_cost of both faces) × river_multiplier``; ``inf`` if either face is impassable.
    Per river face, stepping along the flow discounts (toward ``river_min_mult``) and across it
    penalises (toward ``river_max_mult``), scaled by flow saturating at ``river_flow_reference``.  The
    discount (not the ford penalty) is tapered by ``river_navigability`` (None = fully navigable).
    ``settlement_mask`` faces have bridges: no ford penalty.  ``land_cost`` is the caller's base
    (worldgen: ``land_base_cost``; realmgen: base + danger).
    """
    src, dst, indptr = _flatten_adjacency(mesh)
    n = len(mesh.faces)
    if src.size == 0:
        return [np.empty(0, dtype=np.float64) for _ in range(n)]

    land_cost = np.asarray(land_cost, dtype=np.float64)
    land_passable = np.asarray(land_passable, dtype=bool)
    river_flow_dir = np.asarray(river_flow_dir, dtype=np.float64)
    river_flow_mag = np.asarray(river_flow_mag, dtype=np.float64)
    if river_navigability is None:
        river_nav = np.ones(n, dtype=np.float64)
    else:
        river_nav = np.asarray(river_navigability, dtype=np.float64)
    if settlement_mask is not None:
        s_mask = np.asarray(settlement_mask, dtype=bool)
        ford_strength_arr = np.where(s_mask, 0.0, float(river_ford_strength))
    else:
        ford_strength_arr = None
    centroids = np.asarray(mesh.face_centroids, dtype=np.float64)

    passable_edge = land_passable[src] & land_passable[dst]
    base = 0.5 * (land_cost[src] + land_cost[dst])

    step = centroids[dst] - centroids[src]              # (E, 3)
    step_hat = step / np.maximum(np.linalg.norm(step, axis=1, keepdims=True), 1e-12)

    ref = max(float(river_flow_reference), 1e-9)

    def _face_factor(face_idx: np.ndarray) -> np.ndarray:
        fdir = river_flow_dir[face_idx]                # (E, 3)
        fmag = river_flow_mag[face_idx]                # (E,)
        nav = np.clip(river_nav[face_idx], 0.0, 1.0)   # (E,) fall-line taper
        align = np.abs((step_hat * fdir).sum(axis=1))  # (E,) in [0, 1]
        w = np.minimum(fmag / ref, 1.0)
        ford_s = ford_strength_arr[face_idx] if ford_strength_arr is not None else river_ford_strength
        factor = 1.0 + w * (
            ford_s * (1.0 - align) - river_corridor_strength * nav * align
        )
        return np.where(fmag > 0.0, factor, 1.0)

    mult = np.clip(_face_factor(src) * _face_factor(dst), river_min_mult, river_max_mult)
    flat_cost = np.where(passable_edge, base * mult, np.inf)

    return [flat_cost[indptr[i]:indptr[i + 1]].copy() for i in range(n)]


# ── Sea ──────────────────────────────────────────────────────────────────────


def detect_straits(
    mesh,
    is_water: np.ndarray,
    *,
    params: "StraitParams",
) -> np.ndarray:
    """(N,) bool: water faces pinched between land on opposite sides (chokepoints).

    Two land neighbours' tangent directions must have dot ≤ ``opposite_dot_threshold``, with at least
    ``min_water_neighbors`` water neighbours forming the channel.
    """
    p = params
    n = len(mesh.faces)
    is_water = np.asarray(is_water, dtype=bool)
    centroids = np.asarray(mesh.face_centroids, dtype=np.float64)
    strait = np.zeros(n, dtype=bool)

    for f in range(n):
        if not is_water[f]:
            continue
        cf = centroids[f]
        land_dirs: List[np.ndarray] = []
        water_count = 0
        for g in mesh.face_adjacency[f]:
            d = centroids[g] - cf
            d = d - (d @ cf) * cf          # tangent-plane projection
            dn = float(np.linalg.norm(d))
            if dn < 1e-12:
                continue
            d /= dn
            if is_water[g]:
                water_count += 1
            else:
                land_dirs.append(d)
        if water_count < p.min_water_neighbors or len(land_dirs) < 2:
            continue
        L = np.array(land_dirs)
        dots = L @ L.T
        iu = np.triu_indices(len(L), k=1)
        if float(dots[iu].min()) <= p.opposite_dot_threshold:
            strait[f] = True
    return strait


def compute_sea_distance_to_coast(mesh, is_water: np.ndarray) -> np.ndarray:
    """(N,) float32 great-circle km from each sailable face to the nearest coast.

    Dijkstra over sailable water seeded at water faces touching land.  The opposite of climate's
    ``distance_to_coast`` (land → ocean).  Lakes and navigable rivers read as coastal (~0).  Land and
    water with no reachable coast return 0.
    """
    n = len(mesh.faces)
    is_water = np.asarray(is_water, dtype=bool)
    centroids = np.asarray(mesh.face_centroids, dtype=np.float64)
    adj = mesh.face_adjacency

    dist = np.full(n, np.inf, dtype=np.float64)
    heap: List[tuple] = []
    for f in range(n):
        if is_water[f] and any(not is_water[g] for g in adj[f]):
            dist[f] = 0.0
            heap.append((0.0, f))
    heapq.heapify(heap)

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        cu = centroids[u]
        for v in adj[u]:
            if not is_water[v]:
                continue
            w = EARTH_RADIUS_KM * float(np.arccos(np.clip(cu @ centroids[v], -1.0, 1.0)))
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, int(v)))

    dist[~np.isfinite(dist)] = 0.0
    return dist.astype(np.float32)


def compute_ice_penalty(
    temperature_tex: np.ndarray,
    *,
    freeze_threshold_celsius: float = 0.0,
    penalty_strength: float = 5.0,
) -> np.ndarray:
    """(N,) float32 ice multiplier ``1 + freezing_fraction × penalty_strength``.

    ``temperature_tex`` is (N, H, W, K) per-season °C; a season freezes when its face mean is below
    ``freeze_threshold_celsius``.
    """
    temp = np.asarray(temperature_tex, dtype=np.float64)
    n_faces = temp.shape[0]
    freeze_thresh = float(freeze_threshold_celsius)
    strength = float(penalty_strength)

    if temp.shape[-1] < 1:
        return np.ones(n_faces, dtype=np.float32)

    season_temps = temp.reshape(n_faces, -1, temp.shape[-1])  # (N, H*W, K)
    mean_season_temps = season_temps.mean(axis=1)  # (N, K)
    freezing_seasons = np.sum(mean_season_temps < freeze_thresh, axis=1)  # (N,)
    n_seasons = mean_season_temps.shape[1]
    freezing_fraction = np.clip(freezing_seasons / max(n_seasons, 1), 0.0, 1.0)
    penalty = 1.0 + freezing_fraction * strength
    return penalty.astype(np.float32)


def compute_wind_sea_modifier(
    winds: np.ndarray,
    sea_distance_to_coast_km: np.ndarray,
    *,
    optimal_speed: float = 7.0,
    speed_width: float = 5.0,
    penalty_strength: float = 1.0,
    coast_blend_km: float = 300.0,
) -> np.ndarray:
    """(N,) float32 sea cost multiplier ≥ 1 from mean wind speed (becalmed or stormy costs more).

    An inverted Gaussian of width ``speed_width`` around ``optimal_speed`` m/s, peaking at
    ``1 + penalty_strength``; blended to 1 within ``coast_blend_km`` of the coast.  ``winds`` is
    (N, K, 3) m/s; distances from :func:`compute_sea_distance_to_coast`.
    """
    winds_f = np.asarray(winds, dtype=np.float64)           # (N, K, 3)
    speed = np.linalg.norm(winds_f, axis=2).mean(axis=1)   # (N,) m/s, mean across seasons
    w = max(float(speed_width), 1e-9)
    bell = 1.0 + float(penalty_strength) * (
        1.0 - np.exp(-0.5 * ((speed - float(optimal_speed)) / w) ** 2)
    )
    coast_dist = np.asarray(sea_distance_to_coast_km, dtype=np.float64)
    coast_factor = np.clip(coast_dist / max(float(coast_blend_km), 1.0), 0.0, 1.0)
    modifier = 1.0 + coast_factor * (bell - 1.0)
    return modifier.astype(np.float32)


def compute_wind_mean_dir(
    winds: np.ndarray,
    sea_distance_to_coast_km: np.ndarray,
    *,
    coast_blend_km: float = 300.0,
) -> np.ndarray:
    """(N, 3) float32 unit mean wind direction scaled by the coast blend (→ 0 near shore).

    Uses the same blend as :func:`compute_wind_sea_modifier` so coastal routes get no wind bias.
    """
    winds_f = np.asarray(winds, dtype=np.float64)
    mean_wind = winds_f.mean(axis=1)                         # (N, 3)
    mag = np.linalg.norm(mean_wind, axis=1, keepdims=True)   # (N, 1)
    unit_dir = np.divide(mean_wind, mag, out=np.zeros_like(mean_wind), where=mag > 1e-12)
    coast_dist = np.asarray(sea_distance_to_coast_km, dtype=np.float64)
    coast_factor = np.clip(coast_dist / max(float(coast_blend_km), 1.0), 0.0, 1.0)
    return (unit_dir * coast_factor[:, None]).astype(np.float32)


def compute_sea_cost(
    is_water: np.ndarray,
    sea_distance_to_coast_km: np.ndarray,
    strait_mask: np.ndarray,
    *,
    params: "SeaCostParams",
    temperature_tex: np.ndarray | None = None,
    winds: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-face sea cost: coastal water cheapest, open ocean dearer, straits discounted.

    Ramps linearly from ``coastal_cost`` at the shore to ``open_ocean_cost`` at ``open_ocean_distance_km``;
    straits × ``strait_discount``; then optional ice and wind-speed multipliers.  The directional wind
    term lives in :func:`build_sea_edge_costs`.  Distances must come from
    :func:`compute_sea_distance_to_coast`, not climate's ``distance_to_coast``.  ``is_water`` is the
    sailable mask (ocean ∪ lake ∪ navigable rivers).

    Returns:
        (sea_base_cost (N,) float32, 0 on land; sea_passable (N,) bool).
    """
    p = params
    sea_passable = np.asarray(is_water, dtype=bool)
    t = np.clip(
        np.asarray(sea_distance_to_coast_km, dtype=np.float64) / max(p.open_ocean_distance_km, 1e-9),
        0.0,
        1.0,
    )
    cost = p.coastal_cost + (p.open_ocean_cost - p.coastal_cost) * t
    cost = np.where(np.asarray(strait_mask, dtype=bool), cost * p.strait_discount, cost)

    if temperature_tex is not None:
        ice_penalty = compute_ice_penalty(
            temperature_tex,
            freeze_threshold_celsius=p.ice_freeze_threshold,
            penalty_strength=p.ice_penalty_strength,
        )
        cost = cost * ice_penalty

    if winds is not None:
        wind_mod = compute_wind_sea_modifier(
            winds,
            sea_distance_to_coast_km,
            optimal_speed=p.wind_optimal_speed,
            speed_width=p.wind_speed_width,
            penalty_strength=p.wind_penalty_strength,
            coast_blend_km=p.wind_coast_blend_km,
        )
        cost = cost * wind_mod

    cost = np.where(sea_passable, cost, 0.0).astype(np.float32)
    return cost, sea_passable


def build_sea_edge_costs(
    mesh,
    sea_cost: np.ndarray,
    sea_passable: np.ndarray,
    *,
    wind_dir: np.ndarray | None = None,
    direction_strength: float = 0.4,
) -> List[np.ndarray]:
    """Per-edge sea cost (mirrors ``mesh.face_adjacency``), with an optional directional wind term.

    Base = mean of endpoint costs (``inf`` if either is land).  ``wind_dir`` (from
    :func:`compute_wind_mean_dir`, ~0 near coasts) scales by ``1 - direction_strength · align`` with the
    endpoints' mean wind: cheaper downwind, dearer upwind.
    """
    src, dst, indptr = _flatten_adjacency(mesh)
    n = len(mesh.faces)
    if src.size == 0:
        return [np.empty(0, dtype=np.float64) for _ in range(n)]
    sea_cost = np.asarray(sea_cost, dtype=np.float64)
    sea_passable = np.asarray(sea_passable, dtype=bool)
    passable_edge = sea_passable[src] & sea_passable[dst]
    base = 0.5 * (sea_cost[src] + sea_cost[dst])

    if wind_dir is not None and direction_strength > 0.0:
        centroids = np.asarray(mesh.face_centroids, dtype=np.float64)
        step = centroids[dst] - centroids[src]
        step_hat = step / np.maximum(np.linalg.norm(step, axis=1, keepdims=True), 1e-12)
        wdir = np.asarray(wind_dir, dtype=np.float64)
        mean_wdir = 0.5 * (wdir[src] + wdir[dst])            # (E, 3)
        align = (step_hat * mean_wdir).sum(axis=1)            # (E,) ∈ [-1, 1]
        dir_mult = np.clip(1.0 - direction_strength * align, 1e-3, None)
        base = base * dir_mult

    flat = np.where(passable_edge, base, np.inf)
    return [flat[indptr[i]:indptr[i + 1]].copy() for i in range(n)]
