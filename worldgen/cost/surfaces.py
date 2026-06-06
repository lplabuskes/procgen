"""Physical base cost surfaces for overland and maritime movement.

Two surfaces, per the realmgen design (docs/settlement_generation_design.md):

  * **Land surface** — an isotropic per-face cost (biome difficulty + ruggedness)
    plus an *anisotropic* river term (cheap to travel *along* a navigable valley,
    expensive to *ford across* it). All open water — ocean **and lakes** — is
    impassable on land.
  * **Sea surface** — water is the cheap medium (coastal water cheapest, open
    ocean dearer with distance from coast), land impassable, narrow straits
    tagged and discounted. Lakes are sailable like the ocean, and **navigable
    rivers extend the sailable network inland** so a boat can travel up a river
    from the sea up to the *fall line* — the furthest navigable point, where
    rising ruggedness turns the river to rapids and cities historically cluster.

A navigable-river face is the one place the two surfaces overlap: it is passable
on land (you can walk its banks) *and* on sea (a boat can sail it).

Everything is per-face — faces, not pixels, for logic (see the design doc). The
river anisotropy is the one directional term, so it cannot collapse to a single
per-face scalar; we store a per-face flow direction + magnitude and fold the
direction into the *edge* cost via :func:`build_land_edge_costs`. realmgen calls
the same function after adding its danger field to the per-face base cost, so the
river model lives in exactly one place.

The functions here take only the small attribute set ``faces`` (for length),
``face_adjacency``, and ``face_centroids`` off the mesh, so they are easy to test
against a lightweight stub.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import List, TYPE_CHECKING

import numpy as np

from worldgen.climate.classification import N_BIOMES

if TYPE_CHECKING:
    from worldgen.cost.builder import SeaCostParams, StraitParams

# Sphere radius for converting unit-sphere arc lengths to kilometres (matches the
# value used across the climate stage).
EARTH_RADIUS_KM = 6371.0


# Relative overland traversal difficulty per biome (index matches
# climate.classification.BIOME_NAMES). Open temperate / steppe grassland is the
# 1.0 baseline; dense rainforest, desert heat, and ice are the hardest. This
# table *defines* the model; `biome_cost_scale` tunes its overall weight.
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
    ],
    dtype=np.float32,
)
assert BIOME_TRAVERSAL_COST.shape == (N_BIOMES,), "biome cost table out of sync with BIOME_NAMES"


@dataclass
class CostSurfaces:
    """Per-face physical base traversal-cost layers (worldgen output).

    ``N`` = number of faces. Fields flagged "(meaningful where passable)" hold a
    computed value on every face but should only be trusted where the matching
    passability mask is True.

    A *navigable-river* face is passable on **both** surfaces — it is the only
    overlap. Recover the constituent masks from the two passability arrays:
    ``is_water = ~land_passable`` (ocean ∪ lake), and
    ``navigable_river = land_passable & sea_passable``.

    Attributes:
        land_base_cost:  (N,) float32 — isotropic land cost (biome + ruggedness),
            meaningful where ``land_passable``. The river term is *not* baked in;
            it is directional and applied per-edge by ``build_land_edge_costs``.
        land_passable:   (N,) bool — False on open water (ocean ∪ lake).
        river_flow_dir:  (N, 3) float32 — unit tangent-plane flow direction
            (downstream) per face; ``(0, 0, 0)`` where no river passes.
        river_flow_mag:  (N,) float32 — representative (max) upstream-flow value
            of the river through the face; 0 where no river.
        river_navigability: (N,) float32 in [0, 1] — fall-line taper: 1 where the
            river is placid/flat, falling to 0 at the navigable-ruggedness limit
            (rapids); 0 where no river passes.
        sea_base_cost:   (N,) float32 — sailable-water cost (coastal cheap → open
            dear, straits discounted), meaningful where ``sea_passable``; 0 on
            non-sailable land.
        sea_passable:    (N,) bool — True on sailable water: ocean ∪ lake ∪
            navigable river.
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


# ── Land ─────────────────────────────────────────────────────────────────────


def compute_land_base_cost(
    face_biome: np.ndarray,
    face_elev_range_km: np.ndarray,
    is_water: np.ndarray,
    *,
    biome_cost_scale: float = 1.0,
    ruggedness_cost_per_km: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Isotropic per-face land cost = biome difficulty + ruggedness.

    Args:
        face_biome: (N,) int — biome index per face.
        face_elev_range_km: (N,) float — per-face pixel elevation span (km), the
            ruggedness proxy (per-pixel slope is noise-dominated; see terrain.py).
        is_water: (N,) bool — open water (ocean ∪ lake) is impassable on land.

    Returns:
        (land_base_cost (N,) float32, land_passable (N,) bool). Water faces keep a
        finite (meaningless) cost value; use the mask to gate them.
    """
    biome = np.clip(np.asarray(face_biome).astype(np.int64), 0, N_BIOMES - 1)
    cost = (
        biome_cost_scale * BIOME_TRAVERSAL_COST[biome]
        + ruggedness_cost_per_km * np.maximum(np.asarray(face_elev_range_km, dtype=np.float64), 0.0)
    ).astype(np.float32)
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

    For every river segment, its direction ``end - start`` is attributed to each
    face the segment touches (its start and end face), weighted by the segment's
    upstream-flow value. Per face we keep the flow-weighted mean direction
    (projected onto the face tangent plane, unit length) and the maximum flow
    value seen (the main stem dominates).

    Args:
        mesh: object exposing ``faces`` and ``face_centroids``.
        rivers: iterable of ``River`` (worldgen.hydrology.types) or None.
        include_in_lake: if False, segments flagged ``in_lake`` are skipped (they
            run across a lake surface, not drainable land) — so a river crossing
            an exorheic lake contributes no land cost in that stretch.
        exclude_mask: (N,) bool or None — faces to leave at zero flow (e.g. open
            water: a river's mouth segment touches an ocean face, but a river term
            on a water face is meaningless since it is impassable on land).

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

    # Project each face's accumulated direction onto its tangent plane and
    # normalize. face_centroids are unit vectors, so the radial component is
    # (accum·c) c.
    centroids = np.asarray(mesh.face_centroids, dtype=np.float64)
    radial = (accum * centroids).sum(axis=1, keepdims=True) * centroids
    tangent = accum - radial
    tnorm = np.linalg.norm(tangent, axis=1, keepdims=True)
    flow_dir = np.divide(tangent, tnorm, out=np.zeros_like(tangent), where=tnorm > 1e-12)
    return flow_dir.astype(np.float32), mag.astype(np.float32)


def compute_river_navigability(
    river_flow_mag: np.ndarray,
    face_elev_range_km: np.ndarray,
    *,
    navigable_ruggedness_km: float = 0.6,
) -> np.ndarray:
    """Per-face river navigability in [0, 1] — the fall-line taper.

    A river is navigable where it is *flat*: navigability falls linearly from 1
    (placid, pixel elevation span 0) to 0 at ``navigable_ruggedness_km`` of local
    relief, beyond which the river is rapids/falls. Faces with no river are 0.

    This drives two things: it tapers the land-surface corridor discount toward
    the fall line (:func:`build_land_edge_costs`), and, once thresholded by
    :func:`navigable_river_mask`, it decides which river faces join the sailable
    sea network. The "fall line" — the furthest navigable point up a river, where
    cities cluster — is where this field drops to 0 along the river's course.

    Args:
        river_flow_mag: (N,) float — representative river flow per face (0 = none).
        face_elev_range_km: (N,) float — per-face pixel elevation span (km), the
            ruggedness proxy (per-pixel slope is noise-dominated; see terrain.py).
        navigable_ruggedness_km: local relief (km) at which navigability hits 0.

    Returns:
        (N,) float32 navigability in [0, 1]; 0 where no river passes.
    """
    flow = np.asarray(river_flow_mag, dtype=np.float64)
    relief = np.maximum(np.asarray(face_elev_range_km, dtype=np.float64), 0.0)
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
    """Boolean mask of river faces that join the sailable sea network.

    A river face is navigable-by-boat when it carries at least
    ``navigable_flow_min`` flow, has positive navigability (flat enough — below
    the fall line), and is not already open water. These faces are *amphibious*:
    passable on the land surface (you can walk the banks) **and** on the sea
    surface (a boat can sail them), the only overlap between the two surfaces.
    They let a river connect to the sea directly, not only through a port.

    Args:
        river_flow_mag: (N,) float — representative river flow per face.
        river_navigability: (N,) float — from :func:`compute_river_navigability`.
        is_water: (N,) bool — ocean ∪ lake (already-sailable faces).
        navigable_flow_min: minimum flow to float a boat.

    Returns:
        (N,) bool.
    """
    flow = np.asarray(river_flow_mag, dtype=np.float64)
    nav = np.asarray(river_navigability, dtype=np.float64)
    water = np.asarray(is_water, dtype=bool)
    return (flow >= float(navigable_flow_min)) & (nav > 0.0) & ~water


def _flatten_adjacency(mesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten ``face_adjacency`` into (src, dst, indptr) for vectorized edge ops.

    ``src[k]``/``dst[k]`` are the k-th directed edge (each undirected edge appears
    twice, once per direction). ``indptr[i]:indptr[i+1]`` is the slice of edges
    leaving face i, so a flat per-edge array can be re-split to mirror
    ``face_adjacency``.
    """
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
) -> List[np.ndarray]:
    """Directional per-edge land traversal cost (mirrors ``mesh.face_adjacency``).

    Returns a list ``costs`` where ``costs[i][k]`` is the cost of stepping from
    face ``i`` to ``mesh.face_adjacency[i][k]``. An edge with either endpoint not
    ``land_passable`` is ``np.inf``.

    ``cost = mean(land_cost over the two faces) × river_multiplier``. The
    multiplier comes from how each river face's flow aligns with the step
    direction:

      * parallel (travel *along* the valley) → discount, toward ``river_min_mult``
      * perpendicular (*ford* across the river) → penalty, toward ``river_max_mult``

    scaled by the river's flow magnitude (saturating at ``river_flow_reference``).
    A face with no river contributes a multiplier of 1.

    The *along-valley discount* is additionally tapered by ``river_navigability``
    (1 on placid flat reaches → full discount; 0 in rugged gorges at/above the
    fall line → no discount, leaving only the ford penalty). The ford penalty is
    not tapered — a rapids-choked river is no easier to cross. ``None`` means
    "fully navigable everywhere" (the discount applies in full).

    ``land_cost`` is whatever per-face base the caller wants the river term applied
    to — worldgen passes ``land_base_cost``; realmgen passes base + danger.
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
        factor = 1.0 + w * (
            river_ford_strength * (1.0 - align) - river_corridor_strength * nav * align
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
    """Tag water faces pinched between land on opposite sides (chokepoints).

    A water face is a strait if two of its land neighbours lie in roughly
    opposite tangent directions (their unit direction vectors have a dot product
    at or below ``opposite_dot_threshold``) while it still has at least
    ``min_water_neighbors`` water neighbours forming the through-channel.

    Args:
        mesh: object exposing ``faces``, ``face_adjacency`` and ``face_centroids``.
        is_water: (N,) bool — open water (ocean ∪ lake).
        params: StraitParams — ``opposite_dot_threshold`` + ``min_water_neighbors``.

    Returns:
        (N,) bool strait mask.
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
    """Great-circle km from each water face to the nearest coast.

    Dijkstra over the *sailable-water* sub-graph (ocean ∪ lake ∪ navigable
    rivers), seeded at coastal water (water faces adjacent to at least one land
    face). This is the open-ocean → coast metric the sea surface needs; it is the
    opposite of the climate stage's ``distance_to_coast`` (which is land → ocean
    and ~0 on the water). Lake and navigable-river faces border land on most
    sides, so they read as coastal (~0) — cheap to sail, as intended.

    Land faces, and any water face with no reachable coast (e.g. a world with no
    land), return 0.

    Args:
        mesh: object exposing ``faces``, ``face_adjacency`` and ``face_centroids``.
        is_water: (N,) bool — the sailable mask.

    Returns:
        (N,) float32 distance in km.
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


def compute_sea_cost(
    is_water: np.ndarray,
    sea_distance_to_coast_km: np.ndarray,
    strait_mask: np.ndarray,
    *,
    params: "SeaCostParams",
) -> tuple[np.ndarray, np.ndarray]:
    """Per-face sea cost: coastal water cheapest, open ocean dearer, straits discounted.

    Cost ramps linearly from ``params.coastal_cost`` at the shore to
    ``params.open_ocean_cost`` at ``params.open_ocean_distance_km`` and beyond;
    strait faces are multiplied by ``params.strait_discount`` (near-free
    crossings). Pass distances from :func:`compute_sea_distance_to_coast`
    (open-ocean → coast), *not* the climate stage's land → ocean
    ``distance_to_coast``.

    ``is_water`` is the sailable mask — ocean ∪ lake ∪ navigable rivers; lakes and
    navigable rivers read as coastal (distance ~0) and so come out cheap.

    Returns:
        (sea_base_cost (N,) float32, sea_passable (N,) bool). Non-sailable land
        faces are impassable; their cost is left at 0 (gate via the mask).
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
    cost = np.where(sea_passable, cost, 0.0).astype(np.float32)
    return cost, sea_passable


def build_sea_edge_costs(
    mesh,
    sea_cost: np.ndarray,
    sea_passable: np.ndarray,
) -> List[np.ndarray]:
    """Per-edge sea cost = mean of endpoint sea costs; ``inf`` if either is land.

    Isotropic (no analogue to the river term), so this is a thin averaging pass
    over the ocean sub-graph. Mirrors ``mesh.face_adjacency`` like
    :func:`build_land_edge_costs`.
    """
    src, dst, indptr = _flatten_adjacency(mesh)
    n = len(mesh.faces)
    if src.size == 0:
        return [np.empty(0, dtype=np.float64) for _ in range(n)]
    sea_cost = np.asarray(sea_cost, dtype=np.float64)
    sea_passable = np.asarray(sea_passable, dtype=bool)
    passable_edge = sea_passable[src] & sea_passable[dst]
    base = 0.5 * (sea_cost[src] + sea_cost[dst])
    flat = np.where(passable_edge, base, np.inf)
    return [flat[indptr[i]:indptr[i + 1]].copy() for i in range(n)]
