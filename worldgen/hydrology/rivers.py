"""River generation via per-face grid sampling and D-infinity routing.

Priority-Flood fills depressions; D-infinity routes flow; headwaters are traced
as sub-pixel bilinear-gradient walks to avoid cardinal-sector bias.

Sibling modules: types.py, crossings.py, stitching.py, routing.py.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from worldgen.hydrology.crossings import resolve_crossings, resolve_self_crossings
from worldgen.hydrology.routing import (
    accumulate_flow,
    compute_dinf_flow,
    inter_face_fallback,
    priority_flood,
)
from worldgen.hydrology.stitching import find_connected_components, stitch_face_boundaries
from worldgen.hydrology.types import (
    River,
    RiverNode,
    RiverSegment,
    laplacian_smooth_sphere,
)
from worldgen.mesh.mymesh import PolygonMesh


class RiverGenerator:
    """End-to-end river generation pipeline.

    Uses flat node indexing (face * res² + row * res + col) across all faces.
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

        self._trace_reasons: Dict[int, str] = {}
        # Sparse inter-face adjacency; intra-face neighbours derived on the fly.
        self._inter_adj: Dict[int, List[int]] = stitch_face_boundaries(
            mesh, self._vertex_uvs, self.resolution
        )

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
        """D8 intra-face connectivity plus stitched inter-face neighbours."""
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
    # Priority-Flood + D-inf flow + accumulation
    # ------------------------------------------------------------------

    def compute_flow(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Priority-Flood → D-inf flow direction → multi-receiver accumulation.

        Stores filled DEM as self._eff. Returns (r1, r2, f1, f2, flow_angle, flow_accum).
        """
        eff, was_raised = priority_flood(self.elevations, self._ocean_mask, self._neighbors)
        self._eff = eff
        self._was_raised = was_raised
        self._lake_mask = self._build_lake_mask(was_raised)
        # Snapshot before water-balance shrinking — gates headwater detection.
        self._lake_mask_full = self._lake_mask.copy()
        self._endorheic_mask = np.zeros(self.n_nodes, dtype=bool)
        r1, r2, f1, f2, ang = compute_dinf_flow(
            eff, self.n_nodes, self.n_faces, self._r2, self.resolution
        )
        inter_face_fallback(
            r1, r2, f1, f2, ang, eff,
            self._inter_adj, self._ocean_mask, self.positions,
            self._u_axes, self._v_axes, self._r2,
        )
        accum = accumulate_flow(r1, r2, f1, f2, eff, self._precip_weights)
        self._shrink_endorheic_lakes(accum)
        return r1, r2, f1, f2, ang, accum

    def _shrink_endorheic_lakes(self, flow_accum: np.ndarray) -> None:
        """Shrink endorheic lakes to equilibrium where inflow equals total PET.

        Clears dry-bed pixels from _lake_mask, sets _endorheic_mask on kept pixels,
        and rewrites _eff so external traces descend into the new lake.
        No-op without climate inputs.
        """
        if self._t_annual_flat is None or self._precip_weights is None or self._mean_precip is None:
            return

        components = find_connected_components(self._lake_mask, self._neighbors, self.n_nodes, min_size=1)
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
            inflow = float(flow_accum[comp_arr].max())  # ≈ upstream contribution at the spill pixel
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
            self._eff[keep] = equilibrium_elev
            self._eff[leave] = self.elevations[leave]  # restore dry bed; old eff would route out the old spill

    def _build_lake_mask(self, was_raised: np.ndarray) -> np.ndarray:
        """Return a bool mask containing only flooded land components >= min_lake_pixels."""
        raw = was_raised & ~self._ocean_mask
        mask = np.zeros(self.n_nodes, dtype=bool)
        for comp in find_connected_components(raw, self._neighbors, self.n_nodes, min_size=self.min_lake_pixels):
            mask[np.array(comp)] = True
        return mask

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
        # Pre-shrinking mask: dry-bed pixels still have old flow directions.
        lake_nodes = self._lake_mask_full
        is_above = (flow_accum >= threshold) & ~lake_nodes
        # Also exclude pixels immediately downstream of a lake outlet to avoid duplicate rivers.
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
        """Walk sub-pixel along the negative bilinear gradient of the filled DEM.

        Momentum blending pushes through face-boundary gradient discontinuities.
        Terminates at ocean, endorheic sink, loop, or stall.
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
        # Per-point lake flag lets the renderer hide in-lake segments without truncating data.
        in_lake_flags: List[bool] = [bool(self._lake_mask[start])]

        # Momentum blending pushes through face-boundary gradient discontinuities and NaN flats.
        momentum = 0.5
        prev_du = 0.0
        prev_dv = 0.0
        has_prev = False

        stall_window = 40
        recent_positions: List[np.ndarray] = [current_pos]
        face_cell_3d = float(self._u_ranges[current_face]) / (res - 1)
        stall_threshold = 0.5 * face_cell_3d

        max_steps = 4 * res * res
        terminated_at_sea = False
        reason = 'max_steps'
        # Loop detection; skip first few steps to avoid false positives at the headwater.
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
                        reason = 'no_direction'
                        break
                    du /= mag
                    dv /= mag
            elif has_prev:
                # No gradient (flat or boundary glitch) — coast on momentum.
                du = prev_du
                dv = prev_dv
            else:
                reason = 'no_gradient'
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
                # 0.5 is well above the ~0.14 max overshoot for one UV-grid step.
                if best_out > 0.5 or not candidates:
                    reason = 'bad_crossing'
                    break

                # At triple-point ties, pick the face momentum points most into.
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
                    reason = 'bad_crossing'
                    break
                old_face = current_face
                current_face = next_face
                # Re-express momentum in new face's UV frame to avoid frame-mismatch loops.
                if has_prev:
                    prev_du, prev_dv, has_prev = self._reproject_momentum(
                        prev_du, prev_dv, old_face, current_face
                    )
                u, v = next_uv
                new_pos = self.mesh.face_uv_to_3d(current_face, u, v)

            norm = float(np.linalg.norm(new_pos))
            if norm <= 0.0:
                reason = 'degenerate'
                break
            new_pos = new_pos / norm

            col_n = int(round(u * n_minus_1))
            row_n = int(round(v * n_minus_1))
            nearest = self._node_idx(current_face, row_n, col_n)

            positions.append(new_pos)
            in_lake_flags.append(bool(self._lake_mask[nearest]))
            flow_history.append(max(flow_history[-1], float(flow_accum[nearest])))

            if self._endorheic_mask[nearest]:  # endorheic sink
                reason = 'endorheic'
                break

            if step >= loop_check_start:
                if nearest in visited_cells:
                    reason = 'loop'
                    break  # loop detected
                if nearest not in self._inter_adj:  # skip seam nodes; gradient discontinuities cause brief re-visits
                    visited_cells.add(nearest)
            if self._ocean_mask[nearest]:
                reason = 'ocean'
                terminated_at_sea = True
                break
            # Don't threshold on flow_accum mid-trace; D-inf accum can dip at branch points.
            recent_positions.append(new_pos)
            if len(recent_positions) > stall_window:
                oldest = recent_positions.pop(0)
                if float(np.linalg.norm(new_pos - oldest)) < stall_threshold:
                    reason = 'stall'
                    break

        self._trace_reasons[start] = reason

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
        """Re-express UV momentum from old_face's frame into new_face's. Returns (du, dv, valid)."""
        d_3d = du * self._u_axes[old_face] + dv * self._v_axes[old_face]
        new_du = float(d_3d @ self._u_axes[new_face])
        new_dv = float(d_3d @ self._v_axes[new_face])
        pmag = (new_du ** 2 + new_dv ** 2) ** 0.5
        if pmag > 1e-12:
            return new_du / pmag, new_dv / pmag, True
        return 0.0, 0.0, False

    def _gradient_angle_at(self, face: int, u: float, v: float) -> float:
        """Flow angle (CCW from +U) from bilinear gradient of filled DEM. NaN on flat cells."""
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
        # Bilinear partial derivatives; constant scale doesn't affect direction.
        dE_du = ((1.0 - tv) * (e10 - e00)) + (tv * (e11 - e01))
        dE_dv = ((1.0 - tu) * (e01 - e00)) + (tu * (e11 - e10))
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

    def get_lake_polygons(self) -> List[np.ndarray]:
        """Lake boundary polygons as ordered unit-sphere positions. Call after generate()."""
        if not self._lake_mask.any():
            return []

        polygons: List[np.ndarray] = []
        for comp in find_connected_components(self._lake_mask, self._neighbors, self.n_nodes, min_size=1):
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
        """(N, H, W, 1) float32 Priority-Flood filled DEM. Call after generate()."""
        res = self.resolution
        return self._eff.reshape(self.n_faces, res, res, 1).astype(np.float32)

    def get_lake_mask_texture(self) -> np.ndarray:
        """(N, H, W, 1) float32 lake mask; 1.0 inside a lake. Call after generate()."""
        res = self.resolution
        return self._lake_mask.reshape(self.n_faces, res, res, 1).astype(np.float32)
