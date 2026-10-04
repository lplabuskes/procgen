"""River deltas: the sediment a large river lays down where it meets the sea.

A delta is a low plain at a big river's mouth, built out into the sea from what
the river carries.  Each one here is a *footprint* of global texels around the
river's lower course, grown by a priority flood from that course:

- **Where.**  The fan spreads from an *apex* on the river some way upstream of
  its mouth (the head of the delta, where it starts to branch) and opens
  seaward.  A texel's priority is its distance from the apex, raised the
  further off the river's axis it lies (``lateral_cost``), the deeper the water
  it would fill (``depth_scale_km``), and on land (``land_cost``), so the fan
  hugs the river on land and bulges out over a shallow shelf.  Hills
  (``max_land_elevation_km``) and deep water (``max_depth_km``) are never
  taken, nor lakes, nor another delta's texels.
- **How much.**  The flood stops at an area that grows with the mouth flow,
  ``area_km2 · (flow / 1000) ** area_exponent``.
- **Its shape.**  The balance of river power against the wave power at the
  mouth (:mod:`worldgen.delta.shape`) bends the priority toward long fingers
  in calm water (bird's-foot), a rounded fan, or a blunt cusp with beaches
  running along shore where waves rework the sediment (cuspate).
- **What it becomes.**  Every footprint texel — built land and flattened
  lowland alike — is set to a plain falling from ``apex_height_km`` at the apex
  to ``front_height_km`` at the delta front; the sea in front shoals over
  ``skirt_rings`` rings (the prodelta).  Texels that were sea turn to land.
- **The rivers.**  A river (or saved stream) that ended on the old shore now
  ends inside the plain, so it is carried on across it: the delta's own river
  out to the front along the fan's axis, the others to the nearest open sea or
  onto the delta river's new reach, whichever is nearer.

The flood runs over the texels *inside* their face polygons; a texture's
over-hang copies the texel nearest it, so neighbouring faces agree at seams.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import KDTree

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


@dataclass
class DeltaResult:
    # (N, H, W, 2): plain membership, marsh.
    delta: np.ndarray
    # (N,) share of each face's in-polygon texels on the plain, and mean marsh.
    plain_fraction: np.ndarray
    marsh_fraction: np.ndarray
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


def target_area_km2(flow: float, params: "DeltaParams") -> float:
    return float(params.area_km2) * (max(flow, 0.0) / REFERENCE_FLOW) ** float(params.area_exponent)


def _footprint(
    idx: TexelIndex,
    elev: np.ndarray,
    ocean: np.ndarray,
    lake: np.ndarray,
    claimed: np.ndarray,
    river_pts: np.ndarray,
    flow: float,
    params: "DeltaParams",
    dominance: float = 0.5,
    rng: Optional[np.random.Generator] = None,
) -> Optional[_Footprint]:
    """Grow one delta's footprint from its river's lower course."""
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
    e = elev[cand]
    is_ocean = ocean[cand]
    dist = _arc_km(pos, apex[None, :])
    forward = (pos - apex[None, :]) @ axis * EARTH_RADIUS_KM
    cos_off = np.clip(forward / np.maximum(dist, 1e-9), -1.0, 1.0)

    allowed = (
        ~lake[cand] & ~claimed[cand] & (forward >= -0.5 * idx.spacing_km)
        & np.where(is_ocean, e >= -float(params.max_depth_km), e <= float(params.max_land_elevation_km))
    )
    # The shape metrics start at the shore: the river's last vertex on land.
    on_land = ~ocean[idx.nearest_inside(river_pts)]
    head = _unit(river_pts[int(np.flatnonzero(on_land)[-1])]) if on_land.any() else mouth
    shape = make_shape(dominance, head, axis, flow, size_km, params, rng or np.random.default_rng(0))
    arcuate = dist * (1.0 + float(params.lateral_cost) * (1.0 - cos_off))
    key = shape_metric(pos, apex, axis, arcuate, shape, idx.spacing_km, params) * np.where(
        is_ocean, 1.0 + np.maximum(-e, 0.0) / max(float(params.depth_scale_km), 1e-6), float(params.land_cost),
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
    area = 0.0
    indptr, indices = graph.indptr, graph.indices
    while heap and area < area_target:
        _, k = heapq.heappop(heap)
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
        nbrs &= is_ocean & ~taken & (ring == 0) & ~lake[cand] & ~claimed[cand]
        ring[nbrs] = r
        frontier = nbrs
    front = np.zeros(n, dtype=bool)
    for k in np.flatnonzero(taken):
        front[indices[indptr[k]:indptr[k + 1]]] = True
    front &= is_ocean & ~taken & ~lake[cand]

    return _Footprint(
        apex=apex, axis=axis, ids=cand[taken], skirt_ids=cand[ring > 0], skirt_ring=ring[ring > 0],
        radius_km=float(dist[taken].max()), area_km2=float(weight[taken].sum()),
        cand=cand, graph=graph, in_plain=taken, front=front,
        shape=shape, fingers=finger_tips(pos, taken, shape, idx.spacing_km, params),
    )


def _smoothstep(lo: float, hi: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _apply(
    idx: TexelIndex,
    fp: _Footprint,
    elev: np.ndarray,
    ocean: np.ndarray,
    filled: Optional[np.ndarray],
    delta: np.ndarray,
    claimed: np.ndarray,
    params: "DeltaParams",
) -> np.ndarray:
    """Write a footprint into the flat textures; returns the flat ids now on the plain.

    Every texel near the delta — over-hang included — takes the role of the
    in-polygon texel nearest it, so faces agree across their seams.
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
    surface = toe + (top - toe) * (1.0 - s)
    elev[on_plain] = surface
    ocean[on_plain] = 0
    if filled is not None:
        filled[on_plain] = surface
    delta[on_plain, 0] = 1.0
    # Waves winnow a cuspate delta's front into sandy beach ridges: less marsh.
    marsh_scale = 1.0 - fp.shape.cuspate * (1.0 - float(params.cuspate_marsh))
    delta[on_plain, 1] = marsh_scale * _smoothstep(float(params.marsh_start), 1.0, s)
    claimed[on_plain] = True

    rings = max(int(params.skirt_rings), 1)
    depth = -float(params.skirt_depth_km) * skirt_ring[role[skirt_mask]] / (rings + 1)
    still_sea = ocean[in_skirt] > 0
    elev[in_skirt] = np.where(still_sea, np.maximum(elev[in_skirt], depth), elev[in_skirt])
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
    """New nodes appended to a ``StreamNetwork``, applied in one go at the end."""

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
        if self.nodes:
            st.nodes = np.concatenate([st.nodes, np.asarray(self.nodes, dtype=np.int64)])
            st.positions = np.concatenate([st.positions, np.asarray(self.positions, dtype=np.float32)])
            receiver = np.concatenate([receiver, np.asarray(self.receiver, dtype=np.int64)])
            st.flow = np.concatenate([st.flow, np.asarray(self.flow, dtype=np.float32)])
            channel = np.concatenate([channel, np.asarray(self.channel, dtype=bool)])
            stop = np.concatenate([stop, np.asarray(self.stop, dtype=bool)])
        st.receiver, st.channel, st.stop = receiver, channel, stop


def _carry_across(
    fp: _Footprint,
    idx: TexelIndex,
    on_plain: np.ndarray,
    delta_river: River,
    rivers: List[River],
    streams: Optional[StreamNetwork],
    ext: Optional[_StreamExtension],
    elev: np.ndarray,
) -> Optional[np.ndarray]:
    """Carry the rivers and streams that now end on the plain out to the sea.

    Returns the delta river's new mouth position (None if it could not be routed).
    """
    now_plain = np.zeros(len(elev), dtype=bool)
    now_plain[on_plain] = True
    tree = KDTree(idx.positions[fp.cand])
    front = np.flatnonzero(fp.front)
    near_km = fp.radius_km + 2.0 * idx.spacing_km

    def cand_of(p: np.ndarray) -> int:
        return int(tree.query(np.asarray(p, dtype=np.float64))[1])

    # The delta river, out along the fan's axis.
    old_mouth = np.asarray(delta_river.as_polyline()[-1], dtype=np.float64)
    start = cand_of(old_mouth)
    main = _route(fp, start, _main_target(fp, idx))
    if main is None:
        main = _route(fp, start, front)
    if main is None:
        return None
    main_pts = _extension_points(old_mouth, main, fp, idx)
    _extend_river(delta_river, main_pts, main, fp, idx, elev)
    # The other rivers: to open sea, or onto the delta river's new reach.
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
    # Saved streams ending on the plain: the delta river's own outlet follows its
    # new reach; the rest find their way like the other rivers.
    stops = np.flatnonzero(streams.stop & now_plain[streams.nodes])
    if stops.size == 0:
        return main_pts[-1]
    stop_cand = np.array([cand_of(streams.positions[k]) for k in stops])
    own = int(stops[np.argmax(streams.flow[stops] - 1e6 * (stop_cand != start))])
    main_nodes = ext.add_path(own, [int(fp.cand[k]) for k in main[1:]], list(main_pts[1:]),
                              float(streams.flow[own]))
    joined = {int(c): j for c, j in zip(main[1:], main_nodes)}
    for k, c in zip(stops, stop_cand):
        if int(k) == own:
            continue
        path = _route(fp, int(c), np.concatenate([front, joins]))
        if path is None:
            continue
        pts = _extension_points(streams.positions[k], path, fp, idx)
        last = int(path[-1])
        if last in joined:
            ext.add_path(int(k), [int(fp.cand[j]) for j in path[1:-1]], list(pts[1:-1]),
                         float(streams.flow[k]), ends_on=joined[last])
        else:
            ext.add_path(int(k), [int(fp.cand[j]) for j in path[1:]], list(pts[1:]), float(streams.flow[k]))
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


def _faces_around(mesh: "PolygonMesh", rivers: List[River], params: "DeltaParams") -> np.ndarray:
    """Every face a delta of these rivers could reach, with two faces' margin."""
    tree = KDTree(np.asarray(mesh.face_centroids, dtype=np.float64))
    face_km = float(np.sqrt(np.mean(mesh.face_areas))) * EARTH_RADIUS_KM
    out: list[np.ndarray] = []
    for river in rivers:
        size = float(np.sqrt(target_area_km2(float(river.segments[-1].flow_rate), params)))
        reach = (float(params.apex_distance) + float(params.reach)) * size + 2.0 * face_km
        mouth = np.asarray(river.as_polyline()[-1], dtype=np.float64)
        out.append(np.asarray(tree.query_ball_point(mouth, chord(reach)), dtype=np.int64))
    return np.unique(np.concatenate(out)) if out else np.zeros(0, dtype=np.int64)


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
    """Deposit every delta, largest river first.

    ``winds`` (``(N, K, 3)`` per-season face winds, m/s) set the wave power
    that shapes each delta; without them only swell counts.  ``seed`` makes the
    bird's-foot fingers reproducible.  Edits ``elevation``, ``ocean_mask`` and ``filled_height`` (``(N, H, W, 1)``)
    in place, extends ``rivers`` and ``streams`` in place, and returns the fans,
    the ``(N, H, W, 2)`` delta texture and its per-face fractions.
    """
    n_faces, h, w, _ = elevation.shape
    result = DeltaResult(
        delta=np.zeros((n_faces, h, w, 2), dtype=np.float32),
        plain_fraction=np.zeros(n_faces), marsh_fraction=np.zeros(n_faces),
    )
    candidates = [r for r in rivers if is_delta_river(r, ocean_mask[..., 0] > 0, params)]
    candidates.sort(key=lambda r: -float(r.segments[-1].flow_rate))
    if not candidates:
        return result
    idx = TexelIndex(mesh, h, w, faces=_faces_around(mesh, candidates, params))
    elev = elevation.reshape(-1)
    ocean = ocean_mask.reshape(-1)
    lake = (lake_mask.reshape(-1) > 0) if lake_mask is not None else np.zeros(elev.shape, dtype=bool)
    filled = filled_height.reshape(-1) if filled_height is not None else None
    delta = np.zeros((elev.size, 2), dtype=np.float32)
    claimed = np.zeros(elev.size, dtype=bool)
    ext = _StreamExtension(streams) if streams is not None else None
    # Open sea per face, from the coast before any delta was built.
    waves = WaveField(mesh.face_centroids, ocean_mask[..., 0].mean(axis=(1, 2)) > 0.5, winds,
                      float(np.sqrt(np.mean(mesh.face_areas))))

    for i, river in enumerate(candidates):
        pts = np.stack(river.as_polyline()).astype(np.float64)
        mouth_flat = int(idx.nearest_inside(pts[-1:])[0])
        if claimed[mouth_flat] or not ocean[mouth_flat]:
            continue  # swallowed by a bigger delta already
        flow = float(river.segments[-1].flow_rate)
        seaward = pts[-1] - pts[max(len(pts) - 1 - max(len(pts) // 10, 2), 0)]
        dominance = wave_dominance(waves.power(pts[-1], seaward, params), flow, params)
        rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, i])
        fp = _footprint(idx, elev, ocean > 0, lake, claimed, pts, flow, params, dominance, rng)
        if fp is None:
            continue
        on_plain = _apply(idx, fp, elev, ocean, filled, delta, claimed, params)
        mouth = _carry_across(fp, idx, on_plain, river, rivers, streams, ext, elev)
        result.fans.append(DeltaFan(
            apex=fp.apex.astype(np.float32),
            mouth=(mouth if mouth is not None else pts[-1]).astype(np.float32),
            axis=fp.axis.astype(np.float32), radius_km=fp.radius_km, flow=flow, area_km2=fp.area_km2,
            head=np.asarray(fp.shape.head if fp.shape.head is not None else pts[-1], dtype=np.float32),
            wave_dominance=fp.shape.wave_dominance, shape=fp.shape.name,
            fingers=fp.fingers.astype(np.float32),
        ))
    if ext is not None:
        ext.apply()
    result.delta = delta.reshape(n_faces, h, w, 2)
    result.plain_fraction = idx.face_fractions(delta[:, 0], n_faces)
    result.marsh_fraction = idx.face_fractions(delta[:, 1], n_faces)
    return result
