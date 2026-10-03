"""Regional streams — the region's own drainage network below the world's.

The world's stream network stops at the hydrology stage's
``stream_flow_threshold`` (≈ 100 global texels, ~20,000 km² of catchment), and a
region draws only that (:mod:`region.rivers`).  A local map is fine enough to
want the brooks between those streams, and only the region's refined relief can
say where they run.  So, once per region:

- Flow is routed over the region's refined elevation on the rivers' raster:
  priority-flood from the outlets — refined sea and lakes, the region's re-routed
  rivers, the raster edge — each node draining to the neighbour it was flooded
  from (its lowest), which fills depressions so every node drains to an outlet.
- Each node contributes its catchment at its annual precipitation over the
  world's mean, in the world's flow units (one global texel at mean rain), so a
  regional stream reads on the world's scale.  Flow is capped at the world's
  stream threshold: anything larger is a world stream the world routed
  elsewhere, and the region does not overrule it.
- Nodes above ``min_catchment_km2`` are channels, cut into main stems with the
  global stage's :func:`worldgen.hydrology.rivers.main_stem_paths`.  A stem ends
  on the river, lake or coast it drains into.

The streams end *on* the region's rivers, so they never cross one, and every
local map of a region re-routes the same network (:mod:`localmap.rivers`), so
neighbouring local maps agree on which brooks there are.  The region view does
not draw them.  They are not conserved against anything: the world has no
channels this small to conserve.
"""
from __future__ import annotations

import heapq
from typing import TYPE_CHECKING, Optional

import numpy as np

from core.params import param_field as _f, paramclass
from region.corridor import centreline, node_spacing_rad, raster_frame, sample_raster
from region.rivers import RegionRivers, RiverPath
from region.upsample import EARTH_RADIUS_KM, RegionFields
from worldgen.hydrology.rivers import main_stem_paths

try:
    from numba import njit
except ImportError:  # pragma: no cover - numba is an optional accelerator
    def njit(func=None, **_):  # type: ignore[misc]
        return func if func is not None else (lambda f: f)

if TYPE_CHECKING:
    from core.world import World


@paramclass
class RegionStreamParams:
    """Tuning for the regional stream network the local maps draw."""

    # Smallest catchment that makes a channel, km².  The raster is a regional
    # texel (~4.5 km) apart, so much below ~100 km² is a handful of nodes.
    min_catchment_km2: float = _f(300.0, label="Min catchment (km²)", min=20.0, max=5000.0, step=10.0)
    # Stems shorter than this many raster nodes are dropped as stubs.
    min_nodes: int = _f(3, label="Min stream length (texels)", min=2, max=20)
    # Raster spacing in regional texels, and margin past the cell; as the rivers'.
    raster_texels: float = _f(1.0, label="Stream raster spacing (texels)", min=0.5, max=4.0, step=0.25)
    margin: float = _f(0.08, label="Stream raster margin", min=0.0, max=0.3, step=0.01)


@njit(cache=True)
def _flood(elevation: np.ndarray, outlet: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Priority-flood an ``n × n`` grid (8-connected) from the ``outlet`` nodes.

    Returns ``(parent, order)``: the node each was flooded from (its lowest
    neighbour on the filled surface; -1 at an outlet) and the flood order, so
    reversed it runs every node before the one it drains into.
    """
    size = n * n
    parent = np.full(size, -1, dtype=np.int64)
    order = np.empty(size, dtype=np.int64)
    visited = outlet.copy()
    heap = [(0.0, np.int64(0)) for _ in range(0)]
    for i in range(size):
        if outlet[i]:
            heap.append((elevation[i], np.int64(i)))
    heapq.heapify(heap)
    k = 0
    while heap:
        level, key = heapq.heappop(heap)
        order[k] = key
        k += 1
        r, c = key // n, key % n
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                rr, cc = r + dr, c + dc
                if (dr == 0 and dc == 0) or rr < 0 or rr >= n or cc < 0 or cc >= n:
                    continue
                nb = rr * n + cc
                if visited[nb]:
                    continue
                visited[nb] = True
                parent[nb] = key
                heapq.heappush(heap, (max(elevation[nb], level), np.int64(nb)))
    return parent, order[:k]


@njit(cache=True)
def _accumulate(parent: np.ndarray, order: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Tree accumulation of ``weight`` down ``parent``, upstream first."""
    acc = weight.copy()
    for i in range(order.shape[0] - 1, -1, -1):
        key = order[i]
        if parent[key] >= 0:
            acc[parent[key]] += acc[key]
    return acc


def _river_nodes(fields: RegionFields, rivers: RegionRivers, lo: float, step: float, n: int) -> np.ndarray:
    """``(n²,)`` mask of the raster nodes the region's rivers run through."""
    hit = np.zeros((n, n), dtype=bool)
    hi = lo + step * (n - 1)
    for positions, _, _ in rivers.paths:
        s, t, valid = fields.geometry.ecef_to_local(np.asarray(positions, dtype=np.float64))
        inside = valid & (s >= lo) & (s <= hi) & (t >= lo) & (t <= hi)
        # One run per visit, so a path leaving and re-entering is not bridged.
        idx = np.flatnonzero(inside)
        for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
            if run.size:
                rows, cols = centreline(lo, step, n, s[run], t[run])
                hit[rows, cols] = True
    return hit.ravel()


def _flow_unit_km2(globe: "World") -> float:
    """Catchment of one world flow unit at mean rain, km²: one global texel."""
    mesh = globe.mesh
    gw = mesh.face_textures["elevation"].shape[2]
    spacing = float(np.sqrt(np.mean(mesh.face_areas))) / max(gw - 1, 1)
    return (spacing * EARTH_RADIUS_KM) ** 2


def build_region_streams(
    fields: RegionFields,
    globe: "World",
    rivers: RegionRivers,
    params: Optional[RegionStreamParams] = None,
) -> list[RiverPath]:
    """The region's streams below the world's network, ending on ``rivers``.

    Paths are raster-node polylines in the world ``"rivers"`` feature's
    ``(positions, flows, in_lake)`` form, headwater → outlet, largest flow
    first.  Empty for a world without a ``"streams"`` feature (no threshold to
    stay below) or without precipitation.  Deterministic.
    """
    params = params or RegionStreamParams()
    streams = globe.get_feature("streams")
    mesh = globe.mesh
    if streams is None or len(streams) == 0 or "precipitation" not in mesh.face_textures:
        return []
    cap = float(streams.flow[streams.channel].min()) if streams.channel.any() else float("inf")

    lo, step, n = raster_frame(fields, params.raster_texels, params.margin)
    raster = sample_raster(fields, globe, lo, step, n, np.ones(n * n, dtype=bool), keys=("precipitation",))
    edge = np.zeros((n, n), dtype=bool)
    edge[[0, -1], :] = edge[:, [0, -1]] = True
    edge = edge.ravel()
    outlet = raster.ocean | raster.lake | edge | _river_nodes(fields, rivers, lo, step, n)

    parent, order = _flood(raster.values["elevation"].astype(np.float64), outlet, n)
    unit = _flow_unit_km2(globe)
    node_km2 = (node_spacing_rad(fields, step) * EARTH_RADIUS_KM) ** 2
    mean_rain = max(float(mesh.face_textures["precipitation"].sum(axis=-1).mean()), 1e-6)
    weight = np.where(outlet, 0.0, raster.values["precipitation"] / mean_rain * node_km2)
    catchment = _accumulate(parent, order, weight)        # km² at mean rain

    channel = (catchment >= float(params.min_catchment_km2)) & ~outlet
    flow = np.minimum(catchment / unit, cap)
    out: list[RiverPath] = []
    for path in main_stem_paths(parent, catchment, channel, outlet, int(params.min_nodes)):
        nodes = np.asarray(path, dtype=np.int64)
        if edge[nodes[-1]] and not (raster.ocean[nodes[-1]] or raster.lake[nodes[-1]]):
            # Drains off the raster: cut there, it ends beyond.
            nodes = nodes[:-1]
            if len(nodes) < 2:
                continue
        flows = np.maximum.accumulate(flow[nodes[:-1]]).astype(np.float32)
        out.append((raster.positions[nodes].astype(np.float32), flows, np.zeros(len(flows), dtype=bool)))
    out.sort(key=lambda p: -float(p[1].max()))
    return out
