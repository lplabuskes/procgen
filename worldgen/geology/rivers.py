"""River generation via per-face grid sampling and Priority-Flood routing.

Each face is discretized into a `resolution x resolution` grid of nodes (one
per elevation texture pixel). Intra-face neighbours form a 4-connected grid;
inter-face neighbours are stitched by matching pixels along shared polygon
edges in UV space. Flow is routed using the Priority-Flood algorithm
(O. Planchon & F. Darboux, 2001) which handles depressions implicitly.

Public API:
    RiverGenerator(mesh, elevation_textures, ...).generate() -> list[River]

See `river_generation_spec.md` for design rationale.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from worldgen.mesh.mymesh import PolygonMesh


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class RiverNode:
    position: np.ndarray          # 3D point on globe surface, shape (3,)
    elevation: float
    face_id: int
    uv: Tuple[float, float]       # UV coordinate within the face in [0,1]^2


@dataclass
class RiverSegment:
    start: RiverNode
    end: RiverNode
    flow_rate: float              # accumulated upstream count at segment start
    tributary_count: int = 0


@dataclass
class River:
    segments: List[RiverSegment] = field(default_factory=list)
    source: Optional[RiverNode] = None
    mouth: Optional[RiverNode] = None

    def as_polyline(self) -> List[np.ndarray]:
        if not self.segments:
            return []
        pts = [self.segments[0].start.position]
        pts += [s.end.position for s in self.segments]
        return pts

    def flow_rates(self) -> List[float]:
        """Returns one flow value per segment (parallel to as_polyline edges)."""
        return [s.flow_rate for s in self.segments]


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

def laplacian_smooth_sphere(
    polyline: List[np.ndarray],
    iterations: int = 8,
    alpha: float = 0.5,
) -> List[np.ndarray]:
    """Approximating Laplacian smoothing on the unit sphere.

    Each interior point is pulled toward the midpoint of its two neighbours by
    `alpha`, then re-projected to the sphere. Endpoints (source and mouth) are
    pinned. Unlike an interpolating spline (Catmull-Rom), this displaces the
    control points themselves, so grid-aligned staircases and sharp corners
    are flattened out.
    """
    if len(polyline) < 3:
        return [p.astype(np.float32, copy=True) for p in polyline]

    pts = np.stack(polyline).astype(np.float64)
    for _ in range(iterations):
        mids = 0.5 * (pts[:-2] + pts[2:])
        pts[1:-1] += alpha * (mids - pts[1:-1])
        norms = np.linalg.norm(pts[1:-1], axis=1, keepdims=True)
        np.divide(pts[1:-1], norms, out=pts[1:-1], where=norms > 0)
    return [pts[i].astype(np.float32) for i in range(len(pts))]


# ---------------------------------------------------------------------------
# RiverGenerator
# ---------------------------------------------------------------------------

class RiverGenerator:
    """End-to-end river generation pipeline.

    Internally uses flat node indexing (`face * res^2 + row * res + col`) to
    keep memory and Priority-Flood traversal fast on meshes with thousands
    of faces. Intra-face 4-connectivity is computed on the fly; only the
    sparse inter-face stitching is stored explicitly.
    """

    def __init__(
        self,
        mesh: PolygonMesh,
        elevation_textures: np.ndarray,
        min_flow: float = 50.0,
        sea_level: float = 0.0,
        smoothing_iterations: int = 8,
        smoothing_alpha: float = 0.5,
    ) -> None:
        """
        Args:
            mesh: Polygon mesh on the unit sphere.
            elevation_textures: Per-face elevation as either `(N, H, W)` or
                `(N, H, W, C)` where channel 0 holds the elevation in km.
                H and W must be equal.
            min_flow: Minimum flow accumulation for a node to belong to a river.
            sea_level: Elevation threshold; nodes at or below this drain to the
                ocean (treated as Priority-Flood seeds).
            smoothing_iterations: Laplacian smoothing passes per river polyline.
            smoothing_alpha: Per-iteration pull strength toward neighbour midpoint
                in [0, 1]. Larger values smooth more aggressively per pass.
        """
        if elevation_textures.ndim == 4:
            elev = elevation_textures[:, :, :, 0]
        elif elevation_textures.ndim == 3:
            elev = elevation_textures
        else:
            raise ValueError(
                f"elevation_textures must have 3 or 4 dims, got {elevation_textures.ndim}"
            )

        n_faces = len(mesh.faces)
        if elev.shape[0] != n_faces:
            raise ValueError(
                f"elevation_textures face count {elev.shape[0]} != mesh face count {n_faces}"
            )
        h, w = elev.shape[1], elev.shape[2]
        if h != w:
            raise ValueError(f"non-square texture not supported: {h}x{w}")
        if h < 2:
            raise ValueError(f"texture resolution must be >= 2, got {h}")

        self.mesh = mesh
        self.n_faces = n_faces
        self.resolution = h
        self.n_nodes = n_faces * h * h
        self.min_flow = min_flow
        self.sea_level = sea_level
        self.smoothing_iterations = smoothing_iterations
        self.smoothing_alpha = smoothing_alpha

        # Per-node flat arrays.
        self.elevations: np.ndarray = elev.astype(np.float32).reshape(-1)
        self.positions: np.ndarray = mesh.face_texture_positions(h, h).reshape(-1, 3)

        # Sparse inter-face adjacency: node_idx -> list[node_idx].
        # Intra-face neighbours are derived on the fly in `_neighbors`.
        self._inter_adj: Dict[int, List[int]] = {}
        self._stitch_face_boundaries()

    # ------------------------------------------------------------------
    # Node indexing helpers
    # ------------------------------------------------------------------

    def _node_idx(self, face: int, row: int, col: int) -> int:
        return face * self.resolution * self.resolution + row * self.resolution + col

    def _unpack(self, node: int) -> Tuple[int, int, int]:
        r2 = self.resolution * self.resolution
        face = node // r2
        rem = node % r2
        return face, rem // self.resolution, rem % self.resolution

    def _neighbors(self, node: int) -> List[int]:
        """Intra-face D8 connectivity + any stitched inter-face neighbours.

        Diagonal moves are essential: without them flow is locked to the
        per-face U and V axes which (for ~90% of faces) align with geographic
        east and north, producing visibly cardinal-biased rivers.
        """
        res = self.resolution
        r2 = res * res
        rem = node % r2
        row = rem // res
        col = rem % res
        up = row > 0
        down = row < res - 1
        left = col > 0
        right = col < res - 1
        nbrs: List[int] = []
        if up:
            nbrs.append(node - res)
        if down:
            nbrs.append(node + res)
        if left:
            nbrs.append(node - 1)
        if right:
            nbrs.append(node + 1)
        if up and left:
            nbrs.append(node - res - 1)
        if up and right:
            nbrs.append(node - res + 1)
        if down and left:
            nbrs.append(node + res - 1)
        if down and right:
            nbrs.append(node + res + 1)
        extra = self._inter_adj.get(node)
        if extra:
            nbrs.extend(extra)
        return nbrs

    # ------------------------------------------------------------------
    # Inter-face stitching
    # ------------------------------------------------------------------

    def _stitch_face_boundaries(self) -> None:
        """For each adjacent face pair, find the shared polygon edge in each
        face's UV space and connect grid pixels lying along that edge.

        Pixels are matched by the parameter `t` along the (v0 -> v1) edge,
        since edges in 3D coincide on the great-circle arc between the two
        shared vertices in both faces.
        """
        mesh = self.mesh
        _, _, _, _, _, _, vertex_uvs = mesh.face_tangent_frames()

        for face_a, neighbors in enumerate(mesh.face_adjacency):
            verts_a = mesh.faces[face_a]
            verts_a_set = set(verts_a)
            for face_b in neighbors:
                if face_b <= face_a:
                    continue
                verts_b = mesh.faces[face_b]
                shared = list(verts_a_set & set(verts_b))
                if len(shared) != 2:
                    continue
                v0, v1 = shared
                uv_a0 = np.asarray(vertex_uvs[face_a][verts_a.index(v0)], dtype=np.float64)
                uv_a1 = np.asarray(vertex_uvs[face_a][verts_a.index(v1)], dtype=np.float64)
                uv_b0 = np.asarray(vertex_uvs[face_b][verts_b.index(v0)], dtype=np.float64)
                uv_b1 = np.asarray(vertex_uvs[face_b][verts_b.index(v1)], dtype=np.float64)

                pixels_a = self._pixels_on_edge(uv_a0, uv_a1)
                pixels_b = self._pixels_on_edge(uv_b0, uv_b1)
                if not pixels_a or not pixels_b:
                    continue

                tb = np.array([p[0] for p in pixels_b])
                t_tol = 1.5 / (self.resolution - 1)
                for t_a, r_a, c_a in pixels_a:
                    diffs = np.abs(tb - t_a)
                    k_b = int(np.argmin(diffs))
                    if diffs[k_b] > t_tol:
                        continue
                    _, r_b, c_b = pixels_b[k_b]
                    node_a = self._node_idx(face_a, r_a, c_a)
                    node_b = self._node_idx(face_b, r_b, c_b)
                    self._add_edge(node_a, node_b)

    def _add_edge(self, a: int, b: int) -> None:
        la = self._inter_adj.setdefault(a, [])
        if b not in la:
            la.append(b)
        lb = self._inter_adj.setdefault(b, [])
        if a not in lb:
            lb.append(a)

    def _pixels_on_edge(
        self,
        uv0: np.ndarray,
        uv1: np.ndarray,
    ) -> List[Tuple[float, int, int]]:
        """Find grid pixels lying within ~1 pixel of the UV segment uv0->uv1.

        Returns a list of `(t, row, col)` sorted by parameter `t` along the
        segment. `row` corresponds to the v-axis, `col` to the u-axis
        (matching the convention in `PolygonMesh.face_texture_positions`).
        """
        res = self.resolution
        edge = uv1 - uv0
        edge_len_sq = float(edge @ edge)
        if edge_len_sq < 1e-12:
            return []
        # Half-pixel tolerance keeps interior pixels out while catching the
        # boundary band on each side of the polygon edge.
        tol = 0.75 / (res - 1)
        tol_sq = tol * tol

        coords = np.arange(res) / (res - 1)
        u_grid = np.broadcast_to(coords[np.newaxis, :], (res, res))   # cols
        v_grid = np.broadcast_to(coords[:, np.newaxis], (res, res))   # rows
        du = u_grid - uv0[0]
        dv = v_grid - uv0[1]
        t = (du * edge[0] + dv * edge[1]) / edge_len_sq
        proj_u = uv0[0] + t * edge[0]
        proj_v = uv0[1] + t * edge[1]
        perp_sq = (u_grid - proj_u) ** 2 + (v_grid - proj_v) ** 2

        mask = (t >= -1e-3) & (t <= 1.0 + 1e-3) & (perp_sq < tol_sq)
        rows, cols = np.where(mask)
        if rows.size == 0:
            return []
        ts = t[rows, cols]
        order = np.argsort(ts)
        return [
            (float(ts[k]), int(rows[k]), int(cols[k]))
            for k in order
        ]

    # ------------------------------------------------------------------
    # Priority-Flood + accumulation
    # ------------------------------------------------------------------

    def compute_flow(self) -> Tuple[np.ndarray, np.ndarray]:
        """Route flow and accumulate upstream counts.

        Three stages:
          1. Priority-Flood fills depressions, producing a filled elevation
             field `eff` that has no spurious sinks below sea level.
          2. D8 steepest descent on `eff` (distance-weighted using actual 3D
             positions) picks each node's downstream neighbour. Using gradient
             rather than BFS-pop order removes the directional bias toward
             whichever compass direction was inserted into the heap first.
          3. Flow accumulation in `eff`-descending order ensures every
             upstream node has been summed before its parent reads its value.

        Returns:
            flow_dir: int32 (n_nodes,) — parent node each node drains into,
                -1 for ocean sinks.
            flow_accum: float32 (n_nodes,) — total upstream node count
                (each node contributes 1.0 to itself).
        """
        elev = self.elevations
        n = self.n_nodes
        sea = self.sea_level

        # --- Stage 1: Priority-Flood ---
        eff = elev.astype(np.float32, copy=True)
        pf_parent = np.full(n, -1, dtype=np.int32)
        visited = np.zeros(n, dtype=bool)

        heap: List[Tuple[float, int]] = []
        ocean_idx = np.where(elev <= sea)[0]
        for idx in ocean_idx.tolist():
            heapq.heappush(heap, (float(elev[idx]), idx))
            visited[idx] = True
            eff[idx] = float(elev[idx])

        while heap:
            eff_elev, key = heapq.heappop(heap)
            for nb in self._neighbors(key):
                if visited[nb]:
                    continue
                visited[nb] = True
                nb_elev = float(elev[nb])
                nb_eff = nb_elev if nb_elev > eff_elev else eff_elev
                eff[nb] = nb_eff
                pf_parent[nb] = key
                heapq.heappush(heap, (nb_eff, nb))

        # --- Stage 2: D8 steepest descent on filled DEM ---
        flow_dir = self._steepest_descent(eff, pf_parent)

        # --- Stage 3: Flow accumulation, eff-descending ---
        flow_accum = np.ones(n, dtype=np.float32)
        order = np.argsort(-eff)
        for key in order:
            parent = int(flow_dir[key])
            if parent >= 0:
                flow_accum[parent] += flow_accum[key]
        return flow_dir, flow_accum

    def _steepest_descent(
        self,
        eff: np.ndarray,
        pf_parent: np.ndarray,
    ) -> np.ndarray:
        """For every node pick the neighbour with the steepest descent on
        `eff`, weighted by actual 3D distance so diagonals don't dominate.

        Nodes with no strictly descending neighbour (e.g. on a filled flat)
        fall back to the Priority-Flood parent, guaranteeing every non-ocean
        node has a downstream link.
        """
        n = self.n_nodes
        res = self.resolution
        r2 = res * res
        pos = self.positions

        best_slope = np.full(n, -np.inf, dtype=np.float32)
        best_nb = np.full(n, -1, dtype=np.int32)

        def update(src: np.ndarray, dst: np.ndarray) -> None:
            delta = eff[src] - eff[dst]
            diff = pos[src] - pos[dst]
            d = np.sqrt(np.einsum('ij,ij->i', diff, diff))
            np.maximum(d, 1e-12, out=d)
            slope = delta / d
            better = (slope > best_slope[src]) & (delta > 0)
            idx = src[better]
            best_slope[idx] = slope[better]
            best_nb[idx] = dst[better]

        # Vectorised intra-face neighbours, one direction at a time.
        face_offsets = (np.arange(self.n_faces, dtype=np.int64)[:, None, None] * r2)
        rr, cc = np.meshgrid(np.arange(res), np.arange(res), indexing='ij')
        grid_flat = face_offsets + (rr[None, :, :] * res + cc[None, :, :])

        directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1),
        ]
        for dr, dc in directions:
            r_lo = max(0, -dr); r_hi = res - max(0, dr)
            c_lo = max(0, -dc); c_hi = res - max(0, dc)
            if r_lo >= r_hi or c_lo >= c_hi:
                continue
            src = grid_flat[:, r_lo:r_hi, c_lo:c_hi].reshape(-1)
            dst = grid_flat[:, r_lo + dr:r_hi + dr, c_lo + dc:c_hi + dc].reshape(-1)
            update(src, dst)

        # Inter-face stitched neighbours (sparse).
        if self._inter_adj:
            srcs: List[int] = []
            dsts: List[int] = []
            for s, neighbors in self._inter_adj.items():
                srcs.extend([s] * len(neighbors))
                dsts.extend(neighbors)
            if srcs:
                update(np.asarray(srcs, dtype=np.int64),
                       np.asarray(dsts, dtype=np.int64))

        # Flats: nothing descending, fall back to PF parent.
        flats = (best_nb == -1) & (eff > self.sea_level)
        best_nb[flats] = pf_parent[flats]
        return best_nb

    # ------------------------------------------------------------------
    # River extraction
    # ------------------------------------------------------------------

    def extract_rivers(
        self,
        flow_dir: np.ndarray,
        flow_accum: np.ndarray,
    ) -> List[River]:
        """Trace each headwater downstream until it falls below threshold or
        reaches a sink."""
        threshold = self.min_flow
        is_above = flow_accum >= threshold

        # Build upstream lookup (parent -> children) only for above-threshold nodes.
        upstream: Dict[int, List[int]] = {}
        above_idx = np.where(is_above)[0]
        for node in above_idx.tolist():
            parent = int(flow_dir[node])
            if parent >= 0 and is_above[parent]:
                upstream.setdefault(parent, []).append(node)

        headwaters: List[int] = [
            int(node) for node in above_idx.tolist() if node not in upstream
        ]

        rivers: List[River] = []
        for hw in headwaters:
            river = self._trace_river(hw, flow_dir, flow_accum)
            if river is not None and len(river.segments) > 2:
                rivers.append(river)
        return rivers

    def _trace_river(
        self,
        start: int,
        flow_dir: np.ndarray,
        flow_accum: np.ndarray,
    ) -> Optional[River]:
        path: List[int] = [start]
        current = start
        while True:
            nxt = int(flow_dir[current])
            if nxt < 0:
                break
            path.append(nxt)
            current = nxt
            # Stop once we drop into ocean territory — flow_dir still routes
            # through ocean nodes, but they aren't river-worthy.
            if self.elevations[nxt] <= self.sea_level:
                break
        if len(path) < 2:
            return None

        nodes_obj = [self._make_node(idx) for idx in path]
        river = River(source=nodes_obj[0], mouth=nodes_obj[-1])
        for k in range(1, len(path)):
            river.segments.append(RiverSegment(
                start=nodes_obj[k - 1],
                end=nodes_obj[k],
                flow_rate=float(flow_accum[path[k - 1]]),
            ))
        return river

    def _make_node(self, flat_idx: int) -> RiverNode:
        face, row, col = self._unpack(flat_idx)
        denom = self.resolution - 1
        return RiverNode(
            position=self.positions[flat_idx].copy(),
            elevation=float(self.elevations[flat_idx]),
            face_id=face,
            uv=(col / denom, row / denom),
        )

    # ------------------------------------------------------------------
    # Facade
    # ------------------------------------------------------------------

    def generate(self) -> List[River]:
        flow_dir, flow_accum = self.compute_flow()
        rivers = self.extract_rivers(flow_dir, flow_accum)
        for river in rivers:
            raw = river.as_polyline()
            if len(raw) < 2:
                continue
            smoothed = laplacian_smooth_sphere(
                raw, self.smoothing_iterations, self.smoothing_alpha
            )
            for i, seg in enumerate(river.segments):
                seg.start.position = smoothed[i]
                seg.end.position = smoothed[i + 1]
        return rivers
