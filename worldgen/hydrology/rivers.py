"""River generation via per-face grid sampling and D-infinity routing.

Each face is discretized into a `resolution x resolution` grid of nodes (one
per elevation texture pixel). Intra-face neighbours form a D8 grid;
inter-face neighbours are stitched by matching pixels along shared polygon
edges in UV space. Flow is routed using the Priority-Flood algorithm
(O. Planchon & F. Darboux, 2001) followed by D-infinity flow direction
(D. Tarboton, 1997): each node sends fractional flow to up to two receivers
based on a continuous-angle gradient computed from 8 triangular facets.

River traces are sub-pixel: starting from each headwater, the walk steps
~one grid cell along the continuous flow angle in 3D, snapping back to the
grid only to look up the next angle. This removes the cardinal/diagonal
quantization bias that D8 leaves behind.

Public API:
    RiverGenerator(mesh, elevation_textures, ...).generate() -> list[River]

See `docs/river_generation_spec.md` for design rationale.

Data types, smoothing, and crossing resolution live in sibling modules:
    types.py      — RiverNode, RiverSegment, River, laplacian_smooth_sphere
    crossings.py  — crossing_pairs, resolve_crossings
"""
from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Tuple

import numpy as np

from worldgen.hydrology.crossings import resolve_crossings, resolve_self_crossings
from worldgen.hydrology.types import (
    River,
    RiverNode,
    RiverSegment,
    laplacian_smooth_sphere,
)
from worldgen.mesh.mymesh import PolygonMesh


class RiverGenerator:
    """End-to-end river generation pipeline.

    Internally uses flat node indexing (`face * res^2 + row * res + col`) to
    keep memory and Priority-Flood traversal fast on meshes with thousands
    of faces. Intra-face D8 connectivity is computed on the fly; only the
    sparse inter-face stitching is stored explicitly.
    """

    def __init__(
        self,
        mesh: PolygonMesh,
        elevation_textures: np.ndarray,
        min_flow: float = 50.0,
        sea_level: float = 0.0,
        min_lake_pixels: int = 50,
        smoothing_iterations: int = 8,
        smoothing_alpha: float = 0.5,
        precipitation_textures: np.ndarray | None = None,
        temperature_textures: np.ndarray | None = None,
        pet_baseline_mm_yr: float = 100.0,
        pet_per_degree_mm_yr: float = 60.0,
        pet_max_mm_yr: float = 2500.0,
        ocean_mask_textures: np.ndarray | None = None,
    ) -> None:
        """
        Args:
            mesh: Polygon mesh on the unit sphere.
            elevation_textures: Per-face elevation as either `(N, H, W)` or
                `(N, H, W, C)` where channel 0 holds the elevation in km.
                H and W must be equal.
            min_flow: Minimum flow accumulation for a node to belong to a river.
            sea_level: Elevation threshold used as a fallback ocean mask when
                `ocean_mask_textures` is not provided; nodes at or below this
                drain to the ocean (treated as Priority-Flood seeds).
            ocean_mask_textures: Optional `(N, H, W, 1)` float32 array where
                values > 0.5 mark ocean pixels. When provided, these pixels
                are used as Priority-Flood seeds and river-trace sinks instead
                of the `sea_level` elevation threshold. Necessary when
                sub-sea-level land basins exist that should not drain freely.
            min_lake_pixels: Minimum connected-component size (in pixels) for a
                flooded region to be treated as a lake. Smaller regions are
                ignored — rivers pass through them and they are excluded from the
                lake mask texture.
            smoothing_iterations: Laplacian smoothing passes per river polyline.
            smoothing_alpha: Per-iteration pull strength toward neighbour midpoint
                in [0, 1]. Larger values smooth more aggressively per pass.
            precipitation_textures: Optional `(N, H, W, K)` float32 mm/season.
                When provided, each node's initial flow contribution is scaled by
                its annual precipitation divided by the global mean, so wetter
                pixels source more runoff than drier ones. Required (alongside
                temperature_textures) to enable water-balance lake shrinking.
            temperature_textures: Optional `(N, H, W, K)` float32 °C. The annual
                mean is used to estimate per-pixel potential evapotranspiration
                (PET) for water-balance lake shrinking. When both this and
                precipitation_textures are provided, closed basins where
                inflow < total PET shrink to their equilibrium size; otherwise
                lakes remain at their full Priority-Flood fill level.
            pet_baseline_mm_yr: PET at T = 0 °C, before clamping. Acts as the
                cold-region floor (default 100 mm/yr).
            pet_per_degree_mm_yr: Linear slope of PET vs T for T > 0 °C
                (default 60 mm/yr per °C; PET ≈ 1000 mm/yr at 15 °C).
            pet_max_mm_yr: Upper clamp on PET to keep hot-desert values bounded
                (default 2500 mm/yr).
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
        self._r2 = h * h
        self.min_flow = min_flow
        self.sea_level = sea_level
        self.min_lake_pixels = min_lake_pixels
        self.smoothing_iterations = smoothing_iterations
        self.smoothing_alpha = smoothing_alpha

        # Per-node flat arrays.
        self.elevations: np.ndarray = elev.astype(np.float32).reshape(-1)
        self.positions: np.ndarray = mesh.face_texture_positions(h, h).reshape(-1, 3)

        if precipitation_textures is not None:
            annual = precipitation_textures.sum(axis=-1).astype(np.float32)  # (N, H, W)
            flat = annual.reshape(-1)
            mean = float(flat.mean())
            self._mean_precip: float | None = max(mean, 1e-6)
            self._precip_weights: np.ndarray | None = flat / self._mean_precip
        else:
            self._mean_precip = None
            self._precip_weights = None

        if temperature_textures is not None:
            t_annual = temperature_textures.mean(axis=-1).astype(np.float32)  # (N, H, W)
            self._t_annual_flat: np.ndarray | None = t_annual.reshape(-1)
        else:
            self._t_annual_flat = None
        self._pet_baseline_mm_yr = float(pet_baseline_mm_yr)
        self._pet_per_degree_mm_yr = float(pet_per_degree_mm_yr)
        self._pet_max_mm_yr = float(pet_max_mm_yr)

        if ocean_mask_textures is not None:
            omask = ocean_mask_textures[:, :, :, 0] if ocean_mask_textures.ndim == 4 else ocean_mask_textures
            self._ocean_mask: np.ndarray = omask.reshape(-1).astype(bool)
        else:
            self._ocean_mask = self.elevations <= sea_level

        # Populated by compute_flow / water-balance shrinking.
        self._lake_mask_full: np.ndarray = np.zeros(self.n_nodes, dtype=bool)
        self._endorheic_mask: np.ndarray = np.zeros(self.n_nodes, dtype=bool)

        # Cache per-face tangent frame data for sub-pixel walks.
        frames = mesh.face_tangent_frames()
        self._u_axes = np.asarray(frames[0], dtype=np.float64)
        self._v_axes = np.asarray(frames[1], dtype=np.float64)
        self._u_ranges = np.asarray(frames[4], dtype=np.float64)
        self._vertex_uvs = frames[6]

        # Sparse inter-face adjacency: node_idx -> list[node_idx].
        # Intra-face neighbours are derived on the fly in `_neighbors`.
        self._inter_adj: Dict[int, List[int]] = {}
        self._stitch_face_boundaries()

    # ------------------------------------------------------------------
    # Node indexing helpers
    # ------------------------------------------------------------------

    def _node_idx(self, face: int, row: int, col: int) -> int:
        return face * self._r2 + row * self.resolution + col

    def _unpack(self, node: int) -> Tuple[int, int, int]:
        face = node // self._r2
        rem = node % self._r2
        return face, rem // self.resolution, rem % self.resolution

    def _neighbors(self, node: int) -> List[int]:
        """Intra-face D8 connectivity + any stitched inter-face neighbours.

        Used only by Priority-Flood; D-infinity routing computes its own
        per-direction slices vectorised across all faces.
        """
        res = self.resolution
        rem = node % self._r2
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
        vertex_uvs = self._vertex_uvs

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
    # Priority-Flood + D-inf flow + accumulation
    # ------------------------------------------------------------------

    def compute_flow(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run priority-flood + D-inf flow direction + multi-receiver accumulation.

        Stores the filled DEM as `self._eff` for the trace walk's bilinear
        gradient lookup.

        Returns:
            receiver1, receiver2: int32 (n,) — up to two downstream receivers
                per node, -1 when absent.
            frac1, frac2: float32 (n,) — fractions sent to each receiver
                (sum to 1 where flow exists).
            flow_angle: float32 (n,) — continuous flow direction in the
                source face's local UV plane (radians CCW from +U). NaN where
                no descending direction was found (sinks, true ocean nodes).
                Retained for headwater detection and external inspection;
                the trace walk uses the bilinear elevation gradient instead.
            flow_accum: float32 (n,) — upstream node count (each node
                contributes 1.0 to itself).
        """
        eff, was_raised = self._priority_flood()
        self._eff = eff
        self._was_raised = was_raised
        self._lake_mask = self._build_lake_mask(was_raised)
        # Snapshot the full Priority-Flood lake mask before water-balance shrinking.
        # The full mask is used to gate headwater detection (so dry-bed pixels
        # uncovered by shrinking don't become spurious headwaters); the shrunken
        # mask drives visual rendering and trace termination.
        self._lake_mask_full = self._lake_mask.copy()
        self._endorheic_mask = np.zeros(self.n_nodes, dtype=bool)
        r1, r2, f1, f2, ang = self._compute_dinf_flow(eff)
        self._inter_face_fallback(r1, r2, f1, f2, ang, eff)
        accum = self._accumulate_flow(r1, r2, f1, f2, eff)
        self._shrink_endorheic_lakes(accum)
        return r1, r2, f1, f2, ang, accum

    def _shrink_endorheic_lakes(self, flow_accum: np.ndarray) -> None:
        """Water-balance lake shrinking.

        For each connected component of `self._lake_mask`, compare inflow
        (max `flow_accum` over the component, in normalized-precipitation
        units) against the total potential evaporation at the current
        Priority-Flood fill level.  If inflow ≥ total PET the lake overflows
        and is left untouched (exorheic).  Otherwise the lake is shrunk to
        the equilibrium fill level where partial-coverage evaporation
        balances inflow, and:

        - `self._lake_mask` has the dry-bed pixels cleared (visual / polygon
          footprint becomes the equilibrium lake).
        - `self._endorheic_mask` is set on the remaining lake pixels so the
          trace walk can terminate when entering an endorheic sink.
        - `self._eff` is rewritten over the affected basin: kept pixels are
          pinned to the equilibrium elevation (flat lake surface), dry-bed
          pixels are restored to their natural elevation so external traces
          descend into the new lake instead of routing to the original rim
          spill.

        No-op when precipitation or temperature inputs were not supplied.
        """
        if self._t_annual_flat is None or self._precip_weights is None or self._mean_precip is None:
            return

        components = self._find_connected_components(self._lake_mask, min_size=1)
        if not components:
            return

        pet_mm_yr = np.clip(
            self._pet_baseline_mm_yr + self._pet_per_degree_mm_yr * np.maximum(0.0, self._t_annual_flat),
            0.0,
            self._pet_max_mm_yr,
        ).astype(np.float32)
        pet_norm = pet_mm_yr / self._mean_precip  # normalized to mean-precip units

        for comp in components:
            comp_arr = np.asarray(comp, dtype=np.int64)
            # Inflow proxy: max flow_accum across the basin equals total upstream
            # contribution at the spill pixel (the lake pixel that exits the basin).
            inflow = float(flow_accum[comp_arr].max())
            total_pet_norm = float(pet_norm[comp_arr].sum())
            if inflow >= total_pet_norm:
                continue  # exorheic — lake overflows, keep at full Priority-Flood fill

            elevs = self.elevations[comp_arr]
            order = np.argsort(elevs, kind='stable')
            sorted_pixels = comp_arr[order]
            cum_pet = np.cumsum(pet_norm[sorted_pixels])
            # Find the smallest prefix whose cumulative PET reaches the inflow.
            k = int(np.searchsorted(cum_pet, inflow, side='left')) + 1
            k = max(1, min(k, len(comp_arr)))

            keep = sorted_pixels[:k]
            leave = sorted_pixels[k:]
            equilibrium_elev = float(self.elevations[sorted_pixels[k - 1]])

            self._lake_mask[leave] = False
            self._endorheic_mask[keep] = True
            # Flat lake surface at equilibrium so external traces see the lake as
            # a sink relative to the dry bed.
            self._eff[keep] = equilibrium_elev
            # Restore dry-bed pixels to their natural elevation; the OLD eff (rim+eps)
            # would route walks out the old spill instead of into the new lake.
            self._eff[leave] = self.elevations[leave]

    def _build_lake_mask(self, was_raised: np.ndarray) -> np.ndarray:
        """Return a bool mask containing only flooded land components >= min_lake_pixels."""
        raw = was_raised & ~self._ocean_mask
        mask = np.zeros(self.n_nodes, dtype=bool)
        for comp in self._find_connected_components(raw, min_size=self.min_lake_pixels):
            mask[np.array(comp)] = True
        return mask

    def _priority_flood(self) -> Tuple[np.ndarray, np.ndarray]:
        """Planchon-Darboux depression filling.

        Returns (eff, was_raised) where eff is the filled DEM and
        was_raised[i] is True for every land node whose natural elevation
        is strictly below the fill level when it was first reached — i.e.,
        it sits inside a topographic depression (lake interior).

        Adds a small per-step epsilon so the filled surface inside a
        depression slopes gently toward the spill point. Without it, the
        bilinear gradient on a filled lake is exactly zero and walks coast
        forever on momentum instead of routing to the outlet.
        """
        elev = self.elevations
        n = self.n_nodes
        eff = elev.astype(np.float32, copy=True)
        visited = np.zeros(n, dtype=bool)
        was_raised = np.zeros(n, dtype=bool)
        eps = 1e-6

        heap: List[Tuple[float, int]] = []
        for idx in np.where(self._ocean_mask)[0].tolist():
            heapq.heappush(heap, (float(elev[idx]), idx))
            visited[idx] = True

        while heap:
            eff_elev, key = heapq.heappop(heap)
            for nb in self._neighbors(key):
                if visited[nb]:
                    continue
                visited[nb] = True
                nb_elev = float(elev[nb])
                # Node is inside a depression: its natural elevation is below
                # the fill level being propagated from the spill point.
                if nb_elev < eff_elev:
                    was_raised[nb] = True
                # Imposed-gradient fill: force a tiny downhill from key to nb
                # whenever the real elevation would create a flat or sink.
                nb_eff = max(nb_elev, eff_elev + eps)
                eff[nb] = nb_eff
                heapq.heappush(heap, (nb_eff, nb))
        return eff, was_raised

    def _compute_dinf_flow(
        self,
        eff: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Tarboton (1997) D-infinity flow on each face's regular UV grid.

        For each node, considers the 8 triangular facets formed by adjacent
        cardinal/diagonal neighbour pairs. The steepest descending facet wins;
        flow is split between its cardinal and diagonal vertices proportional
        to the gradient direction angle within the facet.

        Inter-face neighbours are not part of the 8-facet geometry; boundary
        nodes with no descending intra-face facet are handled by
        `_inter_face_fallback`.
        """
        n = self.n_nodes
        res = self.resolution

        receiver1 = np.full(n, -1, dtype=np.int32)
        receiver2 = np.full(n, -1, dtype=np.int32)
        frac1 = np.zeros(n, dtype=np.float32)
        frac2 = np.zeros(n, dtype=np.float32)
        flow_angle = np.full(n, np.nan, dtype=np.float32)
        best_slope = np.full(n, -np.inf, dtype=np.float32)

        d_inv = float(res - 1)                # 1 / (UV grid cell spacing)
        inv_root2 = 1.0 / float(np.sqrt(2.0))
        pi_4 = float(np.pi / 4.0)

        face_offsets = (np.arange(self.n_faces, dtype=np.int64) * self._r2)[:, None, None]

        # Each facet: (cardinal_dr, cardinal_dc, diag_dr, diag_dc, base, sign).
        # +V (row+1) is north, +U (col+1) is east. `base` is the global UV angle
        # of the cardinal direction; `sign` indicates whether the diagonal lies
        # CCW (+1) or CW (-1) from the cardinal in the (U east, V north) frame.
        facets = [
            (0, +1, +1, +1, 0.0,            +1.0),   # E + NE
            (+1, 0, +1, +1, np.pi / 2,      -1.0),   # N + NE
            (+1, 0, +1, -1, np.pi / 2,      +1.0),   # N + NW
            (0, -1, +1, -1, np.pi,          -1.0),   # W + NW
            (0, -1, -1, -1, np.pi,          +1.0),   # W + SW
            (-1, 0, -1, -1, -np.pi / 2,     -1.0),   # S + SW
            (-1, 0, -1, +1, -np.pi / 2,     +1.0),   # S + SE
            (0, +1, -1, +1, 0.0,            -1.0),   # E + SE
        ]

        for c_dr, c_dc, d_dr, d_dc, base, sign in facets:
            r_lo = max(0, -c_dr, -d_dr)
            r_hi = res - max(0, c_dr, d_dr)
            col_lo = max(0, -c_dc, -d_dc)
            col_hi = res - max(0, c_dc, d_dc)
            if r_lo >= r_hi or col_lo >= col_hi:
                continue

            rr = np.arange(r_lo, r_hi)
            cc = np.arange(col_lo, col_hi)
            R, C = np.meshgrid(rr, cc, indexing='ij')
            src_local = (R * res + C)[None, :, :]
            card_local = ((R + c_dr) * res + (C + c_dc))[None, :, :]
            diag_local = ((R + d_dr) * res + (C + d_dc))[None, :, :]

            src_flat = (face_offsets + src_local).reshape(-1)
            card_flat = (face_offsets + card_local).reshape(-1)
            diag_flat = (face_offsets + diag_local).reshape(-1)

            e0 = eff[src_flat]
            e1 = eff[card_flat]
            e2 = eff[diag_flat]

            s1 = (e0 - e1) * d_inv           # cardinal slope
            s2 = (e1 - e2) * d_inv           # orthogonal slope (toward diagonal)
            diag_slope = (e0 - e2) * d_inv * inv_root2

            r = np.arctan2(s2, s1)
            # Outside [0, pi/4]: clamp angle and use the bounding-edge slope.
            slope = np.where(
                r < 0.0,
                s1,
                np.where(r > pi_4, diag_slope, np.hypot(s1, s2)),
            )
            r_clamped = np.clip(r, 0.0, pi_4)

            better = (slope > 0.0) & (slope > best_slope[src_flat])
            if not better.any():
                continue
            idx = src_flat[better]
            sl = slope[better].astype(np.float32)
            rc = r_clamped[better]

            best_slope[idx] = sl
            f2 = (rc / pi_4).astype(np.float32)
            f1 = (1.0 - f2).astype(np.float32)
            receiver1[idx] = card_flat[better]
            receiver2[idx] = diag_flat[better]
            frac1[idx] = f1
            frac2[idx] = f2
            flow_angle[idx] = (base + sign * rc).astype(np.float32)

        return receiver1, receiver2, frac1, frac2, flow_angle

    def _inter_face_fallback(
        self,
        receiver1: np.ndarray,
        receiver2: np.ndarray,
        frac1: np.ndarray,
        frac2: np.ndarray,
        flow_angle: np.ndarray,
        eff: np.ndarray,
    ) -> None:
        """Route boundary nodes with no descending intra-face facet to their
        steepest inter-face neighbour (D8 single receiver). Updates in place.
        """
        if not self._inter_adj:
            return
        for src, neighbors in self._inter_adj.items():
            if receiver1[src] != -1:
                continue
            e0 = float(eff[src])
            if self._ocean_mask[src]:
                continue
            best_slope = 0.0
            best_nb = -1
            for nb in neighbors:
                delta = e0 - float(eff[nb])
                if delta <= 0.0:
                    continue
                diff = self.positions[src] - self.positions[nb]
                d = float(np.sqrt(diff @ diff))
                if d <= 0.0:
                    continue
                slope = delta / d
                if slope > best_slope:
                    best_slope = slope
                    best_nb = nb
            if best_nb < 0:
                continue
            face = src // self._r2
            direction = self.positions[best_nb] - self.positions[src]
            du = float(direction @ self._u_axes[face])
            dv = float(direction @ self._v_axes[face])
            receiver1[src] = best_nb
            receiver2[src] = -1
            frac1[src] = 1.0
            frac2[src] = 0.0
            flow_angle[src] = float(np.arctan2(dv, du))

    def _accumulate_flow(
        self,
        receiver1: np.ndarray,
        receiver2: np.ndarray,
        frac1: np.ndarray,
        frac2: np.ndarray,
        eff: np.ndarray,
    ) -> np.ndarray:
        """Multi-receiver accumulation in `eff`-descending order."""
        n = self.n_nodes
        if self._precip_weights is not None:
            flow_accum = self._precip_weights.copy()
        else:
            flow_accum = np.ones(n, dtype=np.float32)
        order = np.argsort(-eff)
        for key in order.tolist():
            r1 = int(receiver1[key])
            if r1 < 0:
                continue
            contrib = float(flow_accum[key])
            r2 = int(receiver2[key])
            if r2 < 0:
                flow_accum[r1] += contrib
            else:
                flow_accum[r1] += contrib * float(frac1[key])
                flow_accum[r2] += contrib * float(frac2[key])
        return flow_accum

    # ------------------------------------------------------------------
    # River extraction (sub-pixel walk)
    # ------------------------------------------------------------------

    def extract_rivers(
        self,
        receiver1: np.ndarray,
        receiver2: np.ndarray,
        frac1: np.ndarray,
        frac2: np.ndarray,
        flow_accum: np.ndarray,
    ) -> List[River]:
        """Identify headwaters and trace each downstream via sub-pixel walk."""
        threshold = self.min_flow
        # Use the FULL Priority-Flood lake mask (pre-shrinking) for headwater
        # filtering: dry-bed pixels uncovered by water-balance shrinking still
        # have their flow direction pointing at the old rim spill, so without
        # this they'd appear as orphan headwaters in the eps-gradient field.
        lake_nodes = self._lake_mask_full
        is_above = (flow_accum >= threshold) & ~lake_nodes
        # Also exclude pixels immediately downstream of any lake outlet. Without
        # this, the lake's external rim pixel would be flagged as a headwater
        # (since its only upstream sources are lake-interior pixels, which are
        # masked out above), spawning a duplicate post-lake river that overlaps
        # the upstream trace now that traces pass continuously through lakes.
        is_post_lake = np.zeros(self.n_nodes, dtype=bool)
        for p in np.where(lake_nodes)[0].tolist():
            r1i = int(receiver1[p])
            if r1i >= 0 and not lake_nodes[r1i]:
                is_post_lake[r1i] = True
            r2i = int(receiver2[p])
            if r2i >= 0 and not lake_nodes[r2i]:
                is_post_lake[r2i] = True
        is_above &= ~is_post_lake
        above_idx = np.where(is_above)[0]

        # Headwater = above-threshold node receiving no above-threshold inflow.
        has_inflow = np.zeros(self.n_nodes, dtype=bool)
        for node in above_idx.tolist():
            r1 = int(receiver1[node])
            r2 = int(receiver2[node])
            if r1 >= 0 and is_above[r1] and frac1[node] > 0.0:
                has_inflow[r1] = True
            if r2 >= 0 and is_above[r2] and frac2[node] > 0.0:
                has_inflow[r2] = True
        headwaters = above_idx[~has_inflow[above_idx]].tolist()

        rivers: List[River] = []
        for hw in headwaters:
            river = self._trace_river_subpixel(int(hw), flow_accum)
            if river is not None and len(river.segments) > 2:
                rivers.append(river)
        return rivers

    def _trace_river_subpixel(
        self,
        start: int,
        flow_accum: np.ndarray,
    ) -> Optional[River]:
        """Walk sub-pixel along the negative bilinear elevation gradient,
        staying in face-local UV coordinates. The gradient field of the
        filled DEM is curl-free by construction, so the walk follows true
        downhill paths without the D8 grid-snap cardinal-sector bias. We
        only snap to the grid when crossing a face boundary (to switch into
        the neighbour face's UV frame).
        """
        res = self.resolution
        n_minus_1 = res - 1
        uv_step = 1.0 / n_minus_1
        face_init, row_init, col_init = self._unpack(start)

        current_face = face_init
        u = col_init / n_minus_1
        v = row_init / n_minus_1
        current_pos = self.mesh.face_uv_to_3d(current_face, u, v)
        positions: List[np.ndarray] = [current_pos.copy()]
        flow_history: List[float] = [float(flow_accum[start])]
        # Track per-polyline-point lake membership so segments can be tagged
        # for visual hiding without truncating the river data at lake edges.
        in_lake_flags: List[bool] = [bool(self._lake_mask[start])]

        # Momentum carry-over: per-face textures aren't continuous across face
        # boundaries, so the bilinear gradient can flip direction the instant
        # we cross. Blending each step with the previous direction lets the
        # walk push through that boundary band (and across short flats with
        # NaN gradient) instead of stalling out as a dead end.
        momentum = 0.5
        prev_du = 0.0
        prev_dv = 0.0
        has_prev = False

        # Stall detection: if the walk genuinely makes no progress over many
        # steps despite momentum, kill it. Loose enough to let boundary
        # transients resolve, tight enough to avoid runaway loops.
        stall_window = 40
        recent_positions: List[np.ndarray] = [current_pos]
        face_cell_3d = float(self._u_ranges[current_face]) / (res - 1)
        stall_threshold = 0.5 * face_cell_3d

        max_steps = 4 * res * res
        terminated_at_sea = False
        # Track visited grid cells to catch loops that the stall-window check
        # would miss (loops whose circumference exceeds `stall_window` steps).
        # Skip checking the first few steps so normal sub-pixel jitter at the
        # headwater doesn't falsely terminate a valid trace.
        visited_cells: set = set()
        loop_check_start = 4
        for step in range(max_steps):
            ang = self._gradient_angle_at(current_face, u, v)
            if np.isfinite(ang):
                du = float(np.cos(ang))
                dv = float(np.sin(ang))
                if has_prev:
                    du = (1.0 - momentum) * du + momentum * prev_du
                    dv = (1.0 - momentum) * dv + momentum * prev_dv
                    mag = (du * du + dv * dv) ** 0.5
                    if mag < 1e-12:
                        break
                    du /= mag
                    dv /= mag
            elif has_prev:
                # No gradient (flat or boundary glitch) — coast on momentum.
                du = prev_du
                dv = prev_dv
            else:
                break
            prev_du = du
            prev_dv = dv
            has_prev = True

            new_u = u + uv_step * du
            new_v = v + uv_step * dv

            if 0.0 <= new_u <= 1.0 and 0.0 <= new_v <= 1.0:
                u, v = new_u, new_v
                new_pos = self.mesh.face_uv_to_3d(current_face, u, v)
            else:
                # Crossed face boundary — find the neighbour the intended
                # position falls into and switch frames.
                attempted_pos = self.mesh.face_uv_to_3d(current_face, new_u, new_v)
                attempted_pos /= max(float(np.linalg.norm(attempted_pos)), 1e-12)
                # Collect all neighbour candidates sorted by L1 overflow.
                candidates: List[Tuple[float, int, float, float]] = []
                for nb_face in self.mesh.face_adjacency[current_face]:
                    nu, nv = self.mesh.face_3d_to_uv(nb_face, attempted_pos)
                    # L1 distance outside [0,1]²; 0 means the point is inside.
                    out = max(0.0, -nu, nu - 1.0) + max(0.0, -nv, nv - 1.0)
                    candidates.append((out, nb_face, nu, nv))
                candidates.sort()
                best_out = candidates[0][0] if candidates else float('inf')
                # Reject if the best match is implausibly far outside any face
                # (guards against degenerate geometry; 0.5 is well above the
                # ~0.14 max overshoot for a single UV-grid step).
                if best_out > 0.5 or not candidates:
                    break

                # At triple-point vertices two neighbours tie at out≈0. Break
                # the tie with the face whose UV frame the current momentum
                # direction points most "into" (toward centre from corner).
                vertex_tol = 1.5 * uv_step
                tied = [c for c in candidates if c[0] <= best_out + vertex_tol]
                next_face = -1
                next_uv: Optional[Tuple[float, float]] = None
                if has_prev and len(tied) > 1:
                    d_3d = (prev_du * self._u_axes[current_face]
                            + prev_dv * self._v_axes[current_face])
                    best_score = -float('inf')
                    for out, nb_face, nu, nv in tied:
                        mdu = float(d_3d @ self._u_axes[nb_face])
                        mdv = float(d_3d @ self._v_axes[nb_face])
                        # "Into" = momentum component pointing toward face centre.
                        cu = max(0.0, min(1.0, nu))
                        cv = max(0.0, min(1.0, nv))
                        score = mdu * (0.5 - cu) + mdv * (0.5 - cv)
                        if score > best_score:
                            best_score = score
                            next_face = nb_face
                            next_uv = (cu, cv)
                else:
                    _, next_face, nu0, nv0 = candidates[0]
                    next_uv = (max(0.0, min(1.0, nu0)), max(0.0, min(1.0, nv0)))

                if next_face < 0 or next_uv is None:
                    break
                old_face = current_face
                current_face = next_face
                # prev_du/prev_dv carry momentum in the old face's UV frame.
                # Blending them with the new face's gradient angle (a different
                # coordinate system) produces a 3D direction error that can loop
                # the walk back the way it came. Re-express momentum in the new
                # face's UV frame by projecting through 3D.
                if has_prev:
                    prev_du, prev_dv, has_prev = self._reproject_momentum(
                        prev_du, prev_dv, old_face, current_face
                    )
                u, v = next_uv
                new_pos = self.mesh.face_uv_to_3d(current_face, u, v)

            norm = float(np.linalg.norm(new_pos))
            if norm <= 0.0:
                break
            new_pos = new_pos / norm

            col_n = int(round(u * n_minus_1))
            row_n = int(round(v * n_minus_1))
            nearest = self._node_idx(current_face, row_n, col_n)

            positions.append(new_pos)
            in_lake_flags.append(bool(self._lake_mask[nearest]))
            flow_history.append(max(flow_history[-1], float(flow_accum[nearest])))

            # Endorheic basins are real sinks — terminate here so the river
            # doesn't appear to flow out of a closed depression.
            if self._endorheic_mask[nearest]:
                break

            if step >= loop_check_start:
                if nearest in visited_cells:
                    break  # loop detected
                # Seam nodes (face boundaries) can be re-visited briefly due to
                # gradient discontinuities at face edges; only interior nodes
                # count toward loop detection.
                if nearest not in self._inter_adj:
                    visited_cells.add(nearest)
            if self._ocean_mask[nearest]:
                terminated_at_sea = True
                break
            # NOTE: do not threshold on flow_accum mid-trace — with D-inf
            # multi-receiver, accumulation can dip below threshold at branch
            # points yet recover further downstream. Headwater selection
            # already gates river creation.

            # Stall detection: if momentum hasn't been enough to make real
            # progress over `stall_window` steps, give up.
            recent_positions.append(new_pos)
            if len(recent_positions) > stall_window:
                oldest = recent_positions.pop(0)
                if float(np.linalg.norm(new_pos - oldest)) < stall_threshold:
                    break

        if len(positions) < 2:
            return None

        source_node = RiverNode(
            position=positions[0].astype(np.float32),
            elevation=float(self.elevations[start]),
            face_id=face_init,
            uv=(col_init / n_minus_1, row_init / n_minus_1),
        )
        mouth_col = int(round(u * n_minus_1))
        mouth_row = int(round(v * n_minus_1))
        mouth_idx = self._node_idx(current_face, mouth_row, mouth_col)
        mouth_node = RiverNode(
            position=positions[-1].astype(np.float32),
            elevation=float(self.elevations[mouth_idx]),
            face_id=current_face,
            uv=(u, v),
        )
        river = River(source=source_node, mouth=mouth_node)
        for k in range(1, len(positions)):
            seg_start = RiverNode(
                position=positions[k - 1].astype(np.float32),
                elevation=0.0, face_id=current_face, uv=(0.0, 0.0),
            )
            seg_end = RiverNode(
                position=positions[k].astype(np.float32),
                elevation=0.0, face_id=current_face, uv=(0.0, 0.0),
            )
            seg_in_lake = in_lake_flags[k - 1] or in_lake_flags[k]
            river.segments.append(RiverSegment(
                start=seg_start,
                end=seg_end,
                flow_rate=flow_history[k - 1],
                in_lake=seg_in_lake,
            ))
        _ = terminated_at_sea  # for future use; mouth-at-sea info if needed
        return river

    def _reproject_momentum(
        self,
        du: float,
        dv: float,
        old_face: int,
        new_face: int,
    ) -> Tuple[float, float, bool]:
        """Re-express a UV momentum direction from old_face's frame into new_face's.

        Returns (new_du, new_dv, valid). valid=False when the reprojected direction
        has negligible magnitude (degenerate cross-face transition).
        """
        d_3d = du * self._u_axes[old_face] + dv * self._v_axes[old_face]
        new_du = float(d_3d @ self._u_axes[new_face])
        new_dv = float(d_3d @ self._v_axes[new_face])
        pmag = (new_du ** 2 + new_dv ** 2) ** 0.5
        if pmag > 1e-12:
            return new_du / pmag, new_dv / pmag, True
        return 0.0, 0.0, False

    def _gradient_angle_at(self, face: int, u: float, v: float) -> float:
        """Negative bilinear elevation gradient at sub-pixel UV (u, v).

        Returns the flow angle in face-local UV radians (CCW from +U). Walks
        along this field cannot loop — the field is the gradient of a scalar
        (the filled DEM `eff`), and gradient fields are curl-free.
        Returns NaN on perfectly flat cells.
        """
        res = self.resolution
        n_minus_1 = res - 1
        fu = max(0.0, min(float(n_minus_1) - 1e-9, u * n_minus_1))
        fv = max(0.0, min(float(n_minus_1) - 1e-9, v * n_minus_1))
        c0 = int(fu)
        r0 = int(fv)
        c1 = c0 + 1
        r1 = r0 + 1
        tu = fu - c0
        tv = fv - r0
        base = face * self._r2
        e00 = float(self._eff[base + r0 * res + c0])
        e10 = float(self._eff[base + r0 * res + c1])
        e01 = float(self._eff[base + r1 * res + c0])
        e11 = float(self._eff[base + r1 * res + c1])
        # Bilinear: E(tu, tv) = (1-tu)(1-tv) e00 + tu(1-tv) e10 + (1-tu)tv e01 + tu tv e11.
        # Partial derivatives within the cell (scaled by (res-1) to convert
        # to per-UV-unit; constant scale doesn't affect the angle direction).
        dE_du = ((1.0 - tv) * (e10 - e00)) + (tv * (e11 - e01))
        dE_dv = ((1.0 - tu) * (e01 - e00)) + (tu * (e11 - e10))
        # Flow direction = negative gradient.
        flow_du = -dE_du
        flow_dv = -dE_dv
        if abs(flow_du) < 1e-20 and abs(flow_dv) < 1e-20:
            return float('nan')
        return float(np.arctan2(flow_dv, flow_du))

    # ------------------------------------------------------------------
    # Facade
    # ------------------------------------------------------------------

    def generate(self) -> List[River]:
        r1, r2, f1, f2, _ang, accum = self.compute_flow()
        rivers = self.extract_rivers(r1, r2, f1, f2, accum)
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
        rivers = resolve_self_crossings(rivers)
        rivers = resolve_crossings(rivers)
        return rivers

    def _find_connected_components(
        self, mask: np.ndarray, min_size: int = 1
    ) -> List[List[int]]:
        """Return groups of node indices where mask is True, connected via _neighbors."""
        visited = np.zeros(self.n_nodes, dtype=bool)
        components: List[List[int]] = []
        for start in np.where(mask)[0].tolist():
            if visited[start]:
                continue
            comp: List[int] = []
            stack = [start]
            visited[start] = True
            while stack:
                node = stack.pop()
                comp.append(node)
                for nb in self._neighbors(node):
                    if not visited[nb] and mask[nb]:
                        visited[nb] = True
                        stack.append(nb)
            if len(comp) >= min_size:
                components.append(comp)
        return components

    def get_lake_polygons(self) -> List[np.ndarray]:
        """Return per-lake boundary polygons as ordered unit-sphere positions.

        A pixel is a lake if it is above sea level and was raised by
        Priority-Flood. Boundary pixels are traced in pixel-adjacency order:
        starting from any boundary pixel, walk to unvisited boundary neighbors
        until no new neighbor is reachable. This follows the actual shoreline
        for any lake shape without self-intersections. Must be called after
        `generate()`.
        """
        if not self._lake_mask.any():
            return []

        polygons: List[np.ndarray] = []
        for comp in self._find_connected_components(self._lake_mask, min_size=1):
            boundary = [
                node for node in comp
                if any(not self._lake_mask[nb] for nb in self._neighbors(node))
            ]
            if len(boundary) < 3:
                continue

            boundary_set = set(boundary)
            # Adjacency among boundary pixels only.
            adj: Dict[int, List[int]] = {
                node: [nb for nb in self._neighbors(node) if nb in boundary_set]
                for node in boundary
            }
            # Walk the boundary graph in connected order.
            ordered: List[int] = [boundary[0]]
            visited: set = {boundary[0]}
            current = boundary[0]
            while True:
                nxt = next((nb for nb in adj[current] if nb not in visited), None)
                if nxt is None:
                    break
                ordered.append(nxt)
                visited.add(nxt)
                current = nxt

            if len(ordered) < 3:
                continue
            polygons.append(self.positions[np.array(ordered)].astype(np.float32))

        return polygons

    def get_filled_height_texture(self) -> np.ndarray:
        """Return the filled DEM as a (N_faces, H, W, 1) float32 texture.

        Must be called after `generate()`.
        """
        res = self.resolution
        return self._eff.reshape(self.n_faces, res, res, 1).astype(np.float32)

    def get_lake_mask_texture(self) -> np.ndarray:
        """Return the lake mask as a (N_faces, H, W, 1) float32 texture.

        Pixels with value 1.0 are inside a lake (above sea level and filled
        by Priority-Flood). Must be called after `generate()`.
        """
        res = self.resolution
        return self._lake_mask.reshape(self.n_faces, res, res, 1).astype(np.float32)
