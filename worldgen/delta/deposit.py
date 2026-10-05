"""River deltas: the sediment plain a large river lays down where it meets the sea or a lake.

Each delta is a footprint of global texels grown by a priority flood from the river's lower course:

- **Where.**  The fan spreads seaward from an apex upstream of the mouth.  Priority is distance from
  the apex, raised off the river's axis (``lateral_cost``), over deeper water (``depth_scale_km``) and on
  land (``land_cost``).  Hills (``max_land_elevation_km``), deep water (``max_depth_km``), other water
  bodies and other deltas' texels are never taken.
- **How much.**  Area ``area_km2 · (supply / 1000) ** area_exponent``, where supply is mouth flow less
  the silt upstream lakes trapped (:mod:`worldgen.delta.sediment`); in a lake, capped once it has lost
  ``max_lake_share`` of its area.  Lake deltas measure heights from the lake surface.
- **Shape.**  River vs wave power (:mod:`worldgen.delta.shape`) bends the priority toward bird's-foot
  fingers, a rounded fan, or a cuspate front.
- **Result.**  The footprint becomes a plain falling from ``apex_height_km`` to ``front_height_km``; the
  water in front shoals over ``skirt_rings`` rings (the prodelta).
- **Rivers.**  Rivers and streams that ended on the old shore are carried across the plain: the delta
  river along the fan's axis, others to the nearest open water or onto the delta river's new reach.  A
  river that ran on through a lake rejoins its old course past the plain.

The flood runs over in-polygon texels; over-hang texels copy their nearest in-polygon texel so seams agree.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import KDTree

from worldgen.delta.sediment import downstream_order, mouth_node, sediment_load
from worldgen.delta.shape import ARCUATE, DeltaShape, WaveField, finger_tips, make_shape, shape_metric, wave_dominance
from worldgen.hydrology.types import River, RiverNode, RiverSegment, StreamNetwork, laplacian_smooth_sphere

if TYPE_CHECKING:
    from core.mesh import PolygonMesh
    from worldgen.delta.builder import DeltaParams

EARTH_RADIUS_KM = 6371.0
# The flow the area law is anchored at (``area_km2`` is a delta's area at it).
REFERENCE_FLOW = 1000.0
# Texel neighbours lie within this many texel spacings (8-neighbours and the
# shared-edge duplicates across a seam).
_NEIGHBOUR_SPACINGS = 1.6
# Texels closer than this many spacings are one texel counted twice (a shared
# face edge belongs to both faces' polygons).
_DUPLICATE_SPACINGS = 0.3
# Smoothing of a river's new reach over the plain.
_EXTENSION_SMOOTHING = 4
# Water-body labels: the sea, and land (lakes are numbered from 1).
SEA, LAND = 0, -1
# A lake's fetch is marched in steps of this many texel spacings.
_LAKE_FETCH_STEP = 1.0


@dataclass
class DeltaFan:
    """One delta: where it is, which way it opens, how big it is."""

    apex: np.ndarray          # (3,) head of the delta, on the river
    mouth: np.ndarray         # (3,) the river's mouth at the new delta front
    axis: np.ndarray          # (3,) unit tangent at the apex, pointing seaward
    radius_km: float          # apex to the footprint's farthest texel
    flow: float               # the river's mouth flow
    area_km2: float           # footprint area
    # Shape: the river's old mouth (where fingers / the cusp start), 0 (river-) …
    # 1 (wave-dominated), the shape it came out as, and a bird's-foot's finger tips.
    head: Optional[np.ndarray] = None
    wave_dominance: float = 0.5
    shape: str = ARCUATE
    fingers: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float32))
    lake: bool = False        # built into a lake rather than the sea
    sediment: float = 1.0     # share of the flow still carrying silt (lakes upstream trap it)


@dataclass
class DeltaResult:
    # (N, H, W, 1): plain membership.
    delta: np.ndarray
    # (N,) share of each face's in-polygon texels on the plain.
    plain_fraction: np.ndarray
    fans: List[DeltaFan] = field(default_factory=list)


def chord(km: float) -> float:
    """Unit-sphere chord of an arc ``km`` long."""
    return 2.0 * np.sin(0.5 * km / EARTH_RADIUS_KM)


def _arc_km(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Great-circle distance between unit vectors (rows), km."""
    c = np.linalg.norm(np.asarray(a) - np.asarray(b), axis=-1)
    return 2.0 * np.arcsin(np.clip(0.5 * c, 0.0, 1.0)) * EARTH_RADIUS_KM


def _unit(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-15)


def polyline_length_km(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        return 0.0
    return float(_arc_km(pts[:-1], pts[1:]).sum())


def _point_upstream(points: np.ndarray, distance_km: float) -> np.ndarray:
    """The point ``distance_km`` back up a polyline from its last vertex."""
    pts = np.asarray(points, dtype=np.float64)
    left = float(distance_km)
    for i in range(len(pts) - 1, 0, -1):
        seg = float(_arc_km(pts[i], pts[i - 1]))
        if seg >= left and seg > 0:
            f = left / seg
            return _unit(pts[i] + f * (pts[i - 1] - pts[i]))
        left -= seg
    return _unit(pts[0])


def _densify(points: np.ndarray, step_km: float) -> np.ndarray:
    """Polyline resampled to at most ``step_km`` between points."""
    pts = np.asarray(points, dtype=np.float64)
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        k = max(int(np.ceil(float(_arc_km(a, b)) / max(step_km, 1e-9))), 1)
        for j in range(1, k + 1):
            out.append(_unit(a + (b - a) * (j / k)))
    return np.asarray(out)


class _BySlot:
    """Per-texel values over a :class:`TexelIndex`'s faces, read by global flat texel id."""

    def __init__(self, values: np.ndarray, slot: np.ndarray) -> None:
        self._values, self._slot = values, slot

    def __getitem__(self, flat):
        s = self._slot[flat]
        if np.any(np.asarray(s) < 0):
            raise IndexError("texel outside the indexed faces")
        return self._values[s]


class TexelIndex:
    """The global texel grid on some faces, as points on the sphere, for flooding
    across faces.  Texels are named by their global flat id ``face·H·W + row·W + col``."""

    def __init__(self, mesh: "PolygonMesh", h: int, w: int, faces: Optional[np.ndarray] = None) -> None:
        self.h, self.w = h, w
        n = len(mesh.faces)
        self.faces = np.arange(n) if faces is None else np.unique(np.asarray(faces, dtype=np.int64))
        which = self.faces.tolist()
        pos = mesh.face_texture_positions(h, w, faces=which).reshape(-1, 3).astype(np.float64)
        inside = mesh.face_texture_in_polygon(h, w, faces=which).reshape(len(self.faces), h * w)
        # All texels of the indexed faces, and where each global id sits among them.
        self.ids = (self.faces[:, None] * (h * w) + np.arange(h * w)[None, :]).reshape(-1)
        slot = np.full(n * h * w, -1, dtype=np.int64)
        slot[self.ids] = np.arange(self.ids.size)
        self._inside_count = np.maximum(inside.sum(axis=1), 1)
        inside_flat = inside.reshape(-1)
        self.positions = _BySlot(pos, slot)
        self.inside = _BySlot(inside_flat, slot)
        self.inside_ids = self.ids[inside_flat]
        self.tree_inside = KDTree(pos[inside_flat])
        self.tree_all = KDTree(pos)
        face_km2 = np.asarray(mesh.face_areas, dtype=np.float64)[self.faces] * EARTH_RADIUS_KM ** 2
        self.texel_area_km2 = _BySlot(np.repeat(face_km2 / self._inside_count, h * w), slot)
        self._inside_mask = inside
        self.spacing = float(np.sqrt(np.mean(mesh.face_areas))) / max(w - 1, 1)   # rad
        self.spacing_km = self.spacing * EARTH_RADIUS_KM

    def nearest_inside(self, points: np.ndarray) -> np.ndarray:
        """Flat id of the in-polygon texel nearest each point."""
        _, k = self.tree_inside.query(np.asarray(points, dtype=np.float64).reshape(-1, 3), workers=-1)
        return self.inside_ids[np.asarray(k, dtype=np.int64)]

    def near(self, point: np.ndarray, radius_km: float, *, inside_only: bool = False) -> np.ndarray:
        """Flat ids of the texels within ``radius_km`` of ``point``."""
        tree, ids = (self.tree_inside, self.inside_ids) if inside_only else (self.tree_all, self.ids)
        return ids[np.asarray(tree.query_ball_point(point, chord(radius_km)), dtype=np.int64)]

    def face_uv(self, flat: int) -> tuple[int, float, float]:
        face, rem = divmod(int(flat), self.h * self.w)
        row, col = divmod(rem, self.w)
        return face, col / max(self.w - 1, 1), row / max(self.h - 1, 1)

    def face_fractions(self, values: np.ndarray, n_faces: int) -> np.ndarray:
        """Per-face mean of ``(N·H·W,)`` ``values`` over the in-polygon texels (0 off the index)."""
        out = np.zeros(n_faces, dtype=np.float64)
        v = values[self.ids].reshape(len(self.faces), -1) * self._inside_mask
        out[self.faces] = v.sum(axis=1) / self._inside_count
        return out


@dataclass
class _Footprint:
    apex: np.ndarray
    axis: np.ndarray
    ids: np.ndarray            # (K,) in-polygon flat ids of the plain
    skirt_ids: np.ndarray      # (S,) in-polygon flat ids shoaled in front
    skirt_ring: np.ndarray     # (S,) 1..skirt_rings
    radius_km: float
    area_km2: float
    # The candidate graph the flood ran on, for routing rivers over the plain.
    cand: np.ndarray           # (C,) flat ids
    graph: csr_matrix          # (C, C) edge lengths, km
    in_plain: np.ndarray       # (C,) bool
    front: np.ndarray          # (C,) bool: open sea next to the plain
    shape: DeltaShape
    fingers: np.ndarray        # (F, 3) built finger tips
    water_km2: float = 0.0     # of it, what was open water
    # The share of its area budget the lake's cap left unbuilt (0 if it never hit it).
    shortfall: float = 0.0


def target_area_km2(flow: float, params: "DeltaParams") -> float:
    return float(params.area_km2) * (max(flow, 0.0) / REFERENCE_FLOW) ** float(params.area_exponent)


def _footprint(
    idx: TexelIndex,
    elev: np.ndarray,
    body: np.ndarray,
    target: int,
    level: float,
    claimed: np.ndarray,
    river_pts: np.ndarray,
    flow: float,
    params: "DeltaParams",
    dominance: float = 0.5,
    rng: Optional[np.random.Generator] = None,
    water_budget_km2: float = np.inf,
) -> Optional[_Footprint]:
    """Grow one delta's footprint from its river's lower course.

    The river runs into water body ``target`` (labels in ``body``: the sea,
    a lake, or land) whose surface is at ``level`` km.  The flood takes at most
    ``water_budget_km2`` of that water.
    """
    area_target = target_area_km2(flow, params)
    size_km = float(np.sqrt(area_target))
    mouth = _unit(river_pts[-1])
    apex = _point_upstream(river_pts, float(params.apex_distance) * size_km)
    # Seaward axis: apex → mouth, in the tangent plane at the apex.
    to_mouth = mouth - apex
    to_mouth -= apex * float(apex @ to_mouth)
    if np.linalg.norm(to_mouth) < 1e-9:
        to_mouth = river_pts[-1] - river_pts[-2]
        to_mouth -= apex * float(apex @ to_mouth)
    if np.linalg.norm(to_mouth) < 1e-12:
        return None
    axis = _unit(to_mouth)

    reach_km = float(_arc_km(apex, mouth)) + float(params.reach) * size_km
    cand = idx.near(apex, reach_km, inside_only=True)
    if cand.size == 0:
        return None
    pos = idx.positions[cand]
    e = elev[cand] - level
    is_water = body[cand] == target
    dist = _arc_km(pos, apex[None, :])
    forward = (pos - apex[None, :]) @ axis * EARTH_RADIUS_KM
    cos_off = np.clip(forward / np.maximum(dist, 1e-9), -1.0, 1.0)

    allowed = (
        ~claimed[cand] & (forward >= -0.5 * idx.spacing_km)
        & np.where(is_water, e >= -float(params.max_depth_km),
                   (body[cand] == LAND) & (e <= float(params.max_land_elevation_km)))
    )
    # The shape metrics start at the shore: the river's last vertex on land.
    on_land = body[idx.nearest_inside(river_pts)] != target
    head = _unit(river_pts[int(np.flatnonzero(on_land)[-1])]) if on_land.any() else mouth
    shape = make_shape(dominance, head, axis, flow, size_km, params, rng or np.random.default_rng(0))
    arcuate = dist * (1.0 + float(params.lateral_cost) * (1.0 - cos_off))
    key = shape_metric(pos, apex, axis, arcuate, shape, idx.spacing_km, params) * np.where(
        is_water, 1.0 + np.maximum(-e, 0.0) / max(float(params.depth_scale_km), 1e-6), float(params.land_cost),
    )

    tree = KDTree(pos)
    pairs = tree.query_pairs(_NEIGHBOUR_SPACINGS * idx.spacing, output_type="ndarray")
    n = len(cand)
    if pairs.size == 0:
        return None
    lengths = np.maximum(_arc_km(pos[pairs[:, 0]], pos[pairs[:, 1]]), 1e-6)
    graph = csr_matrix(
        (np.concatenate([lengths, lengths]),
         (np.concatenate([pairs[:, 0], pairs[:, 1]]), np.concatenate([pairs[:, 1], pairs[:, 0]]))),
        shape=(n, n),
    )
    multiplicity = np.asarray(
        tree.query_ball_point(pos, _DUPLICATE_SPACINGS * idx.spacing, return_length=True), dtype=np.float64
    )
    weight = idx.texel_area_km2[cand] / np.maximum(multiplicity, 1.0)

    # Seeds: the river from the apex down to its mouth.
    tail: list[np.ndarray] = []
    for p in river_pts[::-1]:
        if tail and float((p - apex) @ axis) < 0:
            break
        tail.append(p)
    seed_line = _densify(np.asarray([apex] + tail[::-1]), 0.5 * idx.spacing_km)
    _, seeds = tree.query(seed_line, workers=-1)
    seeds = np.unique(np.asarray(seeds, dtype=np.int64))
    seeds = seeds[allowed[seeds]]
    if seeds.size == 0:
        return None

    taken = np.zeros(n, dtype=bool)
    queued = np.zeros(n, dtype=bool)
    heap = [(0.0, int(s)) for s in seeds]
    queued[seeds] = True
    heapq.heapify(heap)
    area = water = 0.0
    capped = False
    indptr, indices = graph.indptr, graph.indices
    while heap and area < area_target:
        _, k = heapq.heappop(heap)
        if is_water[k]:
            if water + float(weight[k]) > water_budget_km2:
                capped = True
                break   # the lake is as full as it may get
            water += float(weight[k])
        taken[k] = True
        area += float(weight[k])
        for nb in indices[indptr[k]:indptr[k + 1]]:
            if not queued[nb] and allowed[nb]:
                queued[nb] = True
                heapq.heappush(heap, (float(key[nb]), int(nb)))
    if not taken.any():
        return None

    # The prodelta: rings of open sea in front of the plain, shoaled.
    ring = np.zeros(n, dtype=np.int64)
    frontier = taken.copy()
    for r in range(1, int(params.skirt_rings) + 1):
        nbrs = np.zeros(n, dtype=bool)
        for k in np.flatnonzero(frontier):
            nbrs[indices[indptr[k]:indptr[k + 1]]] = True
        nbrs &= is_water & ~taken & (ring == 0) & ~claimed[cand]
        ring[nbrs] = r
        frontier = nbrs
    front = np.zeros(n, dtype=bool)
    for k in np.flatnonzero(taken):
        front[indices[indptr[k]:indptr[k + 1]]] = True
    front &= is_water & ~taken

    return _Footprint(
        apex=apex, axis=axis, ids=cand[taken], skirt_ids=cand[ring > 0], skirt_ring=ring[ring > 0],
        radius_km=float(dist[taken].max()), area_km2=float(weight[taken].sum()),
        cand=cand, graph=graph, in_plain=taken, front=front,
        shape=shape, fingers=finger_tips(pos, taken, shape, idx.spacing_km, params), water_km2=water,
        shortfall=max(1.0 - area / max(area_target, 1e-9), 0.0) if capped else 0.0,
    )


def _apply(
    idx: TexelIndex,
    fp: _Footprint,
    elev: np.ndarray,
    ocean: np.ndarray,
    lake: Optional[np.ndarray],
    body: np.ndarray,
    target: int,
    level: float,
    filled: Optional[np.ndarray],
    delta: np.ndarray,
    claimed: np.ndarray,
    params: "DeltaParams",
) -> np.ndarray:
    """Write a footprint into the flat textures; returns the flat ids now on the plain.

    The plain stands above ``level`` (the surface of water body ``target``),
    and the water in front of it shoals toward that surface.  Every texel near
    the delta — over-hang included — takes the role of the in-polygon texel
    nearest it, so faces agree across their seams.
    """
    plain_set = np.zeros(len(elev), dtype=bool)
    plain_set[fp.ids] = True
    skirt_ring = np.zeros(len(elev), dtype=np.int64)
    skirt_ring[fp.skirt_ids] = fp.skirt_ring

    around = idx.near(fp.apex, fp.radius_km + 3.0 * idx.spacing_km)
    if around.size == 0:
        return around
    role = idx.nearest_inside(idx.positions[around])
    on_plain = around[plain_set[role]]
    skirt_mask = (skirt_ring[role] > 0) & ~plain_set[role]
    in_skirt = around[skirt_mask]

    s = np.clip(_arc_km(idx.positions[on_plain], fp.apex[None, :]) / max(fp.radius_km, 1e-9), 0.0, 1.0)
    top, toe = float(params.apex_height_km), float(params.front_height_km)
    surface = level + toe + (top - toe) * (1.0 - s)
    elev[on_plain] = surface
    ocean[on_plain] = 0
    if lake is not None:
        lake[on_plain] = 0
    body[on_plain] = LAND
    if filled is not None:
        filled[on_plain] = surface
    delta[on_plain, 0] = 1.0
    claimed[on_plain] = True

    rings = max(int(params.skirt_rings), 1)
    depth = level - float(params.skirt_depth_km) * skirt_ring[role[skirt_mask]] / (rings + 1)
    still_water = body[in_skirt] == target
    elev[in_skirt] = np.where(still_water, np.maximum(elev[in_skirt], depth), elev[in_skirt])
    return on_plain


# ── carrying rivers across the plain ─────────────────────────────────────────


def _route(fp: _Footprint, start: int, targets: np.ndarray) -> Optional[np.ndarray]:
    """Shortest candidate-node path from ``start`` over the plain to the nearest target."""
    passable = fp.in_plain | np.isin(np.arange(len(fp.cand)), targets)
    passable[start] = True
    keep = passable[fp.graph.indices]
    rows = np.repeat(np.arange(len(fp.cand)), np.diff(fp.graph.indptr))
    keep &= passable[rows]
    sub = csr_matrix((fp.graph.data[keep], (rows[keep], fp.graph.indices[keep])), shape=fp.graph.shape)
    dist, pred = dijkstra(sub, indices=int(start), return_predecessors=True)
    targets = targets[np.isfinite(dist[targets])]
    if targets.size == 0:
        return None
    goal = int(targets[np.argmin(dist[targets])])
    path = [goal]
    while path[-1] != start:
        prev = int(pred[path[-1]])
        if prev < 0:
            return None
        path.append(prev)
    return np.asarray(path[::-1], dtype=np.int64)


def _main_target(fp: _Footprint, idx: TexelIndex) -> np.ndarray:
    """The open-sea texel at the tip of the fan, along its axis."""
    front = np.flatnonzero(fp.front)
    if front.size == 0:
        return front
    pos = idx.positions[fp.cand[front]] - fp.apex[None, :]
    along = pos @ fp.axis
    across = np.linalg.norm(pos - along[:, None] * fp.axis[None, :], axis=1)
    return front[[int(np.argmax(along - 0.5 * across))]]


def _extension_points(start_pos: np.ndarray, nodes: np.ndarray, fp: _Footprint, idx: TexelIndex) -> np.ndarray:
    pts = [np.asarray(start_pos, dtype=np.float64)] + [idx.positions[fp.cand[k]] for k in nodes[1:]]
    smooth = laplacian_smooth_sphere(pts, iterations=_EXTENSION_SMOOTHING, alpha=0.5)
    return np.stack(smooth).astype(np.float64)


def _extend_river(river: River, pts: np.ndarray, nodes: np.ndarray, fp: _Footprint,
                  idx: TexelIndex, elev: np.ndarray) -> None:
    """Append the reach ``pts[1:]`` (over candidate ``nodes``) to ``river``."""
    if len(pts) < 2 or not river.segments:
        return
    flow = float(river.segments[-1].flow_rate)
    prev = river.segments[-1].end
    for p, k in zip(pts[1:], nodes[1:]):
        flat = int(fp.cand[k])
        face, u, v = idx.face_uv(flat)
        node = RiverNode(position=p.astype(np.float32), elevation=float(elev[flat]), face_id=face, uv=(u, v))
        river.segments.append(RiverSegment(start=prev, end=node, flow_rate=flow))
        prev = node
    river.mouth = prev


class _StreamExtension:
    """New nodes appended to a ``StreamNetwork`` (and old ones dropped), applied in one go at the end."""

    def __init__(self, streams: StreamNetwork) -> None:
        self.streams = streams
        self.base = len(streams)
        self.nodes: list[int] = []
        self.positions: list[np.ndarray] = []
        self.receiver: list[int] = []
        self.flow: list[float] = []
        self.channel: list[bool] = []
        self.stop: list[bool] = []
        self.relink: dict[int, int] = {}
        self.drop: set[int] = set()

    def add_path(self, start: int, flats: list[int], positions: list[np.ndarray], flow: float,
                 ends_on: Optional[int] = None) -> list[int]:
        """Continue stream node ``start`` along ``flats``; returns the new node indices.

        The last new node is a sea stop unless ``ends_on`` (a node index) is
        given, which the path then drains into instead.
        """
        made: list[int] = []
        for j, (flat, p) in enumerate(zip(flats, positions)):
            k = self.base + len(self.nodes)
            last = j == len(flats) - 1
            self.nodes.append(int(flat))
            self.positions.append(np.asarray(p, dtype=np.float32))
            self.receiver.append(-1)
            self.flow.append(float(flow))
            sea = last and ends_on is None
            self.channel.append(not sea)
            self.stop.append(sea)
            if made:
                self.receiver[made[-1] - self.base] = k
            made.append(k)
        if made:
            self.relink[int(start)] = made[0]
            if ends_on is not None:
                self.receiver[made[-1] - self.base] = int(ends_on)
        elif ends_on is not None:
            self.relink[int(start)] = int(ends_on)
        return made

    def apply(self) -> None:
        st = self.streams
        receiver = st.receiver.copy()
        channel = st.channel.copy()
        stop = st.stop.copy()
        for old, new in self.relink.items():
            receiver[old] = new
            channel[old] = True
            stop[old] = False
        nodes, positions, flow = st.nodes, st.positions, st.flow
        if self.nodes:
            nodes = np.concatenate([nodes, np.asarray(self.nodes, dtype=np.int64)])
            positions = np.concatenate([positions, np.asarray(self.positions, dtype=np.float32)])
            receiver = np.concatenate([receiver, np.asarray(self.receiver, dtype=np.int64)])
            flow = np.concatenate([flow, np.asarray(self.flow, dtype=np.float32)])
            channel = np.concatenate([channel, np.asarray(self.channel, dtype=bool)])
            stop = np.concatenate([stop, np.asarray(self.stop, dtype=bool)])
        keep = np.ones(len(nodes), dtype=bool)
        keep[list(self.drop)] = False
        remap = np.full(len(nodes), -1, dtype=np.int64)
        remap[keep] = np.arange(int(keep.sum()))
        receiver = np.where(receiver >= 0, remap[np.maximum(receiver, 0)], -1)
        st.nodes, st.positions, st.flow = nodes[keep], positions[keep], flow[keep]
        st.receiver, st.channel, st.stop = receiver[keep], channel[keep], stop[keep]


def _rejoin(river: River, tail: List[RiverSegment], mouth: Optional[RiverNode], now_plain: np.ndarray,
            idx: TexelIndex) -> None:
    """Re-attach the course ``tail`` a river ran on past its old mouth, from the
    first vertex beyond the plain, with one hidden reach from the new front."""
    if not tail:
        return
    ends = np.stack([s_.end.position for s_ in tail]).astype(np.float64)
    on = np.flatnonzero(now_plain[idx.nearest_inside(ends)])
    first = int(on[-1]) + 1 if on.size else 0
    if first >= len(tail):
        return   # all of it now lies on the plain: the river ends at the front
    nxt = tail[first]
    river.segments.append(RiverSegment(start=river.segments[-1].end, end=nxt.end,
                                       flow_rate=nxt.flow_rate, in_lake=nxt.in_lake))
    river.segments.extend(tail[first + 1:])
    river.mouth = mouth


def _own_stream(streams: StreamNetwork, old_mouth: np.ndarray, now_plain: np.ndarray,
                idx: TexelIndex) -> Optional[int]:
    """The stream node at the delta river's old mouth: among the big nodes now on
    the plain within a texel and a half of it, the nearest."""
    gap = _arc_km(streams.positions.astype(np.float64), old_mouth[None, :])
    near = np.flatnonzero((gap <= 1.5 * idx.spacing_km) & now_plain[streams.nodes])
    if near.size == 0:
        return None
    near = near[streams.flow[near] >= 0.5 * float(streams.flow[near].max())]
    return int(near[np.argmin(gap[near])])


def _course_past(streams: StreamNetwork, own: int, now_plain: np.ndarray) -> tuple[set, Optional[int]]:
    """The nodes downstream of ``own`` on the plain, and the first node past it (if any)."""
    on, k = {own}, int(streams.receiver[own])
    while k >= 0 and k not in on and now_plain[streams.nodes[k]]:
        on.add(k)
        k = int(streams.receiver[k])
    return on, (k if k >= 0 and k not in on else None)


def _carry_across(
    fp: _Footprint,
    idx: TexelIndex,
    on_plain: np.ndarray,
    delta_river: River,
    mouth_index: int,
    rivers: List[River],
    streams: Optional[StreamNetwork],
    ext: Optional[_StreamExtension],
    elev: np.ndarray,
) -> Optional[np.ndarray]:
    """Carry the rivers and streams that now end on the plain out to open water.

    The delta river's mouth is its polyline vertex ``mouth_index`` (its last at
    sea; where it enters a lake it may run on through it).  Returns the delta
    river's new mouth position (None if it could not be routed).
    """
    now_plain = np.zeros(len(elev), dtype=bool)
    now_plain[on_plain] = True
    tree = KDTree(idx.positions[fp.cand])
    front = np.flatnonzero(fp.front)
    near_km = fp.radius_km + 2.0 * idx.spacing_km

    def cand_of(p: np.ndarray) -> int:
        return int(tree.query(np.asarray(p, dtype=np.float64))[1])

    # The delta river, out along the fan's axis.
    old_mouth = np.asarray(delta_river.as_polyline()[mouth_index], dtype=np.float64)
    start = cand_of(old_mouth)
    main = _route(fp, start, _main_target(fp, idx))
    if main is None:
        main = _route(fp, start, front)
    if main is None:
        return None
    main_pts = _extension_points(old_mouth, main, fp, idx)
    final_mouth, tail = delta_river.mouth, delta_river.segments[mouth_index:]
    delta_river.segments = delta_river.segments[:mouth_index]
    _extend_river(delta_river, main_pts, main, fp, idx, elev)
    _rejoin(delta_river, tail, final_mouth, now_plain, idx)
    # The other rivers: to open water, or onto the delta river's new reach.
    joins = main[1:-1]
    for river in rivers:
        if river is delta_river or river.mouth is None or not river.segments:
            continue
        end = np.asarray(river.as_polyline()[-1], dtype=np.float64)
        if float(_arc_km(end, fp.apex)) > near_km:
            continue
        if not now_plain[int(idx.nearest_inside(end[None, :])[0])]:
            continue
        path = _route(fp, cand_of(end), np.concatenate([front, joins]))
        if path is not None:
            _extend_river(river, _extension_points(end, path, fp, idx), path, fp, idx, elev)

    if streams is None or ext is None:
        return main_pts[-1]
    # The delta river's own stream follows its new reach, then drains on into
    # whatever its old course reached past the plain (or stops at the front).
    # Its old course over the plain is dropped.
    own = _own_stream(streams, old_mouth, now_plain, idx)
    old_course: set = set()
    joined: dict = {}
    if own is not None:
        old_course, past = _course_past(streams, own, now_plain)
        old_course.discard(own)
        ext.drop |= old_course
        main_nodes = ext.add_path(own, [int(fp.cand[k]) for k in main[1:]], list(main_pts[1:]),
                                  float(streams.flow[own]), ends_on=past)
        joined = {int(c): j for c, j in zip(main[1:], main_nodes)}
    # The other streams ending on the plain — or on that old course — find
    # their way like the other rivers.
    ends = streams.stop & now_plain[streams.nodes]
    feeds_old = np.isin(streams.receiver, list(old_course)) if old_course else np.zeros(len(streams), dtype=bool)
    stops = [int(k) for k in np.flatnonzero(ends | feeds_old)
             if int(k) not in old_course and int(k) != own]
    for k in stops:
        path = _route(fp, cand_of(streams.positions[k]), np.concatenate([front, joins]))
        if path is None:
            continue
        pts = _extension_points(streams.positions[k], path, fp, idx)
        last = int(path[-1])
        if last in joined:
            ext.add_path(k, [int(fp.cand[j]) for j in path[1:-1]], list(pts[1:-1]),
                         float(streams.flow[k]), ends_on=joined[last])
        else:
            ext.add_path(k, [int(fp.cand[j]) for j in path[1:]], list(pts[1:]), float(streams.flow[k]))
    return main_pts[-1]


# ── entry point ──────────────────────────────────────────────────────────────


def is_delta_river(river: River, ocean: np.ndarray, params: "DeltaParams") -> bool:
    """A river big and long enough, its mouth on an ocean texel (``ocean``: ``(N, H, W)``)."""
    if river.mouth is None or not river.segments:
        return False
    if float(river.segments[-1].flow_rate) < float(params.min_flow):
        return False
    if polyline_length_km(np.stack(river.as_polyline())) < float(params.min_length_km):
        return False
    _, h, w = ocean.shape
    u, v = river.mouth.uv
    row = int(np.clip(round(v * (h - 1)), 0, h - 1))
    col = int(np.clip(round(u * (w - 1)), 0, w - 1))
    return bool(ocean[int(river.mouth.face_id), row, col])


@dataclass
class _Mouth:
    """Where a river meets the water a delta could be built into."""

    river: River
    entry: Optional[RiverSegment]   # a lake mouth's first in-lake segment (None at sea)
    flow: float
    lake: bool
    sediment: float = 1.0    # share of ``flow`` carrying silt here
    load: float = 0.0        # the sediment load arriving (stream-network units)
    node: Optional[int] = None   # its stream node
    body: int = LAND         # the lake it enters

    @property
    def supply(self) -> float:
        """The flow that builds the delta: the water still carrying sediment."""
        return self.flow * self.sediment

    def end(self) -> int:
        """Polyline vertex of the mouth, wherever earlier deltas left it (-1 if
        one built over it): at sea the river's end, at a lake its entry segment's end."""
        if self.entry is None:
            return len(self.river.segments)
        for k, seg in enumerate(self.river.segments):
            if seg is self.entry:
                return k + 1
        return -1


def lake_mouths(rivers: List[River], params: "DeltaParams") -> List[_Mouth]:
    """Every place a river big and long enough runs into a lake (its in-lake
    segments start), however it carries on from there."""
    out: list[_Mouth] = []
    for river in rivers:
        segs = river.segments
        entries = [k for k in range(1, len(segs)) if segs[k].in_lake and not segs[k - 1].in_lake]
        if not entries:
            continue
        pts = np.stack(river.as_polyline()).astype(np.float64)
        steps = np.concatenate([[0.0], np.cumsum(_arc_km(pts[:-1], pts[1:]))])
        for k in entries:
            flow = float(segs[k].flow_rate)
            if flow >= float(params.min_flow) and steps[k + 1] >= float(params.min_length_km):
                out.append(_Mouth(river, segs[k], flow, lake=True))
    return out


def _faces_around(mesh: "PolygonMesh", mouths: List[_Mouth], params: "DeltaParams") -> np.ndarray:
    """Every face a delta at these mouths could reach, with two faces' margin."""
    tree = KDTree(np.asarray(mesh.face_centroids, dtype=np.float64))
    face_km = float(np.sqrt(np.mean(mesh.face_areas))) * EARTH_RADIUS_KM
    out: list[np.ndarray] = []
    for m in mouths:
        size = float(np.sqrt(target_area_km2(m.flow, params)))
        reach = (float(params.apex_distance) + float(params.reach)) * size + 2.0 * face_km
        mouth = np.asarray(m.river.as_polyline()[m.end()], dtype=np.float64)
        out.append(np.asarray(tree.query_ball_point(mouth, chord(reach)), dtype=np.int64))
    return np.unique(np.concatenate(out)) if out else np.zeros(0, dtype=np.int64)


def _water_bodies(idx: TexelIndex, ocean: np.ndarray, lake: np.ndarray) -> np.ndarray:
    """Per flat texel: ``SEA``, ``LAND``, or the lake's number (1, 2, …).

    Lakes are the connected pieces of the in-polygon lake texels on the index;
    an over-hanging lake texel joins the lake nearest it.
    """
    body = np.full(ocean.shape, LAND, dtype=np.int64)
    body[ocean > 0] = SEA
    wet = idx.inside_ids[lake[idx.inside_ids]]
    if wet.size == 0:
        return body
    pos = idx.positions[wet]
    tree = KDTree(pos)
    pairs = tree.query_pairs(_NEIGHBOUR_SPACINGS * idx.spacing, output_type="ndarray")
    graph = csr_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(wet.size, wet.size))
    _, labels = connected_components(graph, directed=False)
    body[wet] = labels + 1
    over = idx.ids[lake[idx.ids] & ~idx.inside[idx.ids]]
    if over.size:
        body[over] = body[wet[tree.query(idx.positions[over], workers=-1)[1]]]
    return body


def _lake_texel(idx: TexelIndex, body: np.ndarray, point: np.ndarray) -> int:
    """The in-polygon lake texel nearest ``point`` within a texel and a half (-1 if none)."""
    near = idx.near(point, 1.5 * idx.spacing_km, inside_only=True)
    near = near[body[near] > SEA]
    if near.size == 0:
        return -1
    return int(near[np.argmin(_arc_km(idx.positions[near], point[None, :]))])


class _Sediment:
    """The stream network's sediment load, re-derived as full lakes spill."""

    def __init__(self, streams: StreamNetwork, node_lake: np.ndarray, trap: float, order: np.ndarray) -> None:
        self.streams, self.node_lake, self.trap, self.order = streams, node_lake, trap, order
        self.spilled: dict[int, float] = {}
        self.load: Optional[np.ndarray] = None

    def lake_rank(self) -> dict[int, int]:
        """Per lake, the drainage-order position of its last node: a lake ranks
        above every lake upstream of it."""
        position = np.empty(len(self.order), dtype=np.int64)
        position[self.order] = np.arange(len(self.order))
        rank: dict[int, int] = {}
        for k in np.flatnonzero(self.node_lake > 0):
            b = int(self.node_lake[k])
            rank[b] = max(rank.get(b, 0), int(position[k]))
        return rank

    def spill(self, lake: int, load: float) -> None:
        if load > 0.0:
            self.spilled[lake] = self.spilled.get(lake, 0.0) + float(load)
            self.load = None

    def at(self, node: int) -> tuple[float, float]:
        """The load arriving at ``node``, and the share of its flow that is."""
        if self.load is None:
            self.load = sediment_load(self.streams, self.node_lake, self.trap, self.spilled, self.order)
        flow = float(self.streams.flow[node])
        load = float(self.load[node])
        return load, (float(np.clip(load / flow, 0.0, 1.0)) if flow > 0 else 1.0)


def build_deltas(
    mesh: "PolygonMesh",
    rivers: List[River],
    elevation: np.ndarray,
    ocean_mask: np.ndarray,
    lake_mask: Optional[np.ndarray],
    filled_height: Optional[np.ndarray],
    streams: Optional[StreamNetwork],
    params: "DeltaParams",
    winds: Optional[np.ndarray] = None,
    seed: int = 0,
) -> DeltaResult:
    """Deposit every delta: into lakes (upstream lakes first), then the sea, largest river first.

    ``winds`` (``(N, K, 3)`` per-season face winds, m/s) set the wave power
    that shapes each delta; without them only swell counts.  ``seed`` makes the
    bird's-foot fingers reproducible.  Edits ``elevation``, ``ocean_mask``,
    ``lake_mask`` and ``filled_height`` (``(N, H, W, 1)``) in place, extends
    ``rivers`` and ``streams`` in place, and returns the fans, the
    ``(N, H, W, 1)`` delta texture and its per-face fractions.
    """
    n_faces, h, w, _ = elevation.shape
    result = DeltaResult(
        delta=np.zeros((n_faces, h, w, 1), dtype=np.float32), plain_fraction=np.zeros(n_faces),
    )
    mouths = [_Mouth(r, None, float(r.segments[-1].flow_rate), lake=False)
              for r in rivers if is_delta_river(r, ocean_mask[..., 0] > 0, params)]
    has_lakes = lake_mask is not None and bool((lake_mask > 0).any())
    if has_lakes:
        mouths += lake_mouths(rivers, params)
    if not mouths:
        return result
    faces = _faces_around(mesh, mouths, params)
    if has_lakes:
        assert lake_mask is not None
        # Whole lakes, so each is labelled (its fetch measured, its silt trapped) in one piece.
        faces = np.union1d(faces, np.flatnonzero((lake_mask[..., 0] > 0).any(axis=(1, 2))))
    idx = TexelIndex(mesh, h, w, faces=faces)
    elev = elevation.reshape(-1)
    ocean = ocean_mask.reshape(-1)
    lake = lake_mask.reshape(-1) if lake_mask is not None else None
    filled = filled_height.reshape(-1) if filled_height is not None else None
    body = _water_bodies(idx, ocean, lake > 0 if lake is not None else np.zeros(elev.shape, dtype=bool))
    # Where each mouth is: its lake, and its stream node for the sediment load.
    for m in mouths:
        point = np.asarray(m.river.as_polyline()[m.end()], dtype=np.float64)
        flat = _lake_texel(idx, body, point) if m.lake else int(idx.nearest_inside(point[None, :])[0])
        if m.lake and flat >= 0:
            m.body = int(body[flat])
        if streams is not None and len(streams) and flat >= 0:
            m.node = mouth_node(streams, flat, point, 1.5 * idx.spacing)
    # Sediment: what the lakes upstream trapped, less what full lakes spilled.
    sediment = None
    if streams is not None and len(streams) and lake is not None:
        node_lake = np.maximum(body[streams.nodes], 0)      # lake labels, before any delta
        order = downstream_order(streams)
        sediment = _Sediment(streams, node_lake, float(params.lake_trap_efficiency), order)
    # Lake deltas first, so a full lake's spill reaches the deltas below it: lakes
    # upstream before lakes downstream (by their last node in drainage order),
    # then the largest river first within a lake, then the sea's, largest first.
    rank = sediment.lake_rank() if sediment is not None else {}
    mouths.sort(key=lambda m: (not m.lake, rank.get(m.body, 0) if m.lake else 0, -m.flow))
    # Each lake's surface (its filled level) and how much of it deltas may take.
    lake_ids = idx.inside_ids[body[idx.inside_ids] > SEA]
    surface = filled if filled is not None else elev
    n_lakes = int(body[lake_ids].max()) if lake_ids.size else 0
    level = np.zeros(n_lakes + 1)
    room = np.full(n_lakes + 1, np.inf)
    if n_lakes:
        order = np.argsort(body[lake_ids], kind="stable")
        groups = np.split(lake_ids[order], np.flatnonzero(np.diff(body[lake_ids][order])) + 1)
        for g in groups:
            b = int(body[g[0]])
            level[b] = float(np.median(surface[g]))
            room[b] = float(params.max_lake_share) * float(idx.texel_area_km2[g].sum())
    delta = np.zeros((elev.size, 1), dtype=np.float32)
    claimed = np.zeros(elev.size, dtype=bool)
    ext = _StreamExtension(streams) if streams is not None else None
    # Open water per face (the sea) or texel (lakes), from before any delta was built.
    waves = WaveField(mesh.face_centroids, ocean_mask[..., 0].mean(axis=(1, 2)) > 0.5, winds,
                      float(np.sqrt(np.mean(mesh.face_areas))))
    lake_waves = None
    if n_lakes:
        ins = idx.inside_ids
        lake_waves = WaveField(idx.positions[ins], np.where(body[ins] > SEA, body[ins], LAND), winds, idx.spacing,
                               point_face=ins // (h * w), step_rad=_LAKE_FETCH_STEP * idx.spacing,
                               start_rad=_LAKE_FETCH_STEP * idx.spacing)

    for i, m in enumerate(mouths):
        river = m.river
        end = m.end()
        if end < 0:
            continue   # its entry lies under a delta built further up the river
        if sediment is not None and m.node is not None:
            m.load, m.sediment = sediment.at(m.node)
        if m.supply < float(params.min_flow):
            continue   # too little silt left
        # The river's last run over land: its apex never lies back in a lake it left.
        wet_before = [k for k in range(end - 1) if river.segments[k].in_lake]
        land_from = wet_before[-1] + 1 if wet_before else 0
        pts = np.stack(river.as_polyline()).astype(np.float64)[land_from: end + 1]
        if len(pts) < 2:
            continue
        if m.lake:
            mouth_flat = _lake_texel(idx, body, pts[-1])
            if mouth_flat < 0:
                continue
        else:
            mouth_flat = int(idx.nearest_inside(pts[-1:])[0])
        target = int(body[mouth_flat])
        if claimed[mouth_flat] or (target == LAND) or (m.lake != (target > SEA)):
            continue  # swallowed by a bigger delta already
        if room[target] <= 0.0:
            if sediment is not None and m.lake:
                sediment.spill(target, m.load)    # the lake is full: its silt passes on
            continue
        seaward = pts[-1] - pts[max(len(pts) - 1 - max(len(pts) // 10, 2), 0)]
        if m.lake:
            assert lake_waves is not None
            power = lake_waves.power(pts[-1], seaward, params, body=target, swell=False)
        else:
            power = waves.power(pts[-1], seaward, params)
        dominance = wave_dominance(power, m.supply, params)
        rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, i])
        fp = _footprint(idx, elev, body, target, float(level[target]), claimed, pts, m.supply, params,
                        dominance, rng, water_budget_km2=float(room[target]))
        if fp is None:
            continue
        room[target] -= fp.water_km2
        if sediment is not None and m.lake and fp.shortfall > 0.0:
            sediment.spill(target, fp.shortfall * m.load)
        on_plain = _apply(idx, fp, elev, ocean, lake, body, target, float(level[target]), filled, delta,
                          claimed, params)
        mouth = _carry_across(fp, idx, on_plain, river, end, rivers, streams, ext, elev)
        result.fans.append(DeltaFan(
            apex=fp.apex.astype(np.float32),
            mouth=(mouth if mouth is not None else pts[-1]).astype(np.float32),
            axis=fp.axis.astype(np.float32), radius_km=fp.radius_km, flow=m.flow, area_km2=fp.area_km2,
            head=np.asarray(fp.shape.head if fp.shape.head is not None else pts[-1], dtype=np.float32),
            wave_dominance=fp.shape.wave_dominance, shape=fp.shape.name,
            fingers=fp.fingers.astype(np.float32), lake=m.lake, sediment=m.sediment,
        ))
    if ext is not None:
        ext.apply()
    result.delta = delta.reshape(n_faces, h, w, 1)
    result.plain_fraction = idx.face_fractions(delta[:, 0], n_faces)
    return result
