"""River generation via per-face grid sampling and D-infinity routing.

Priority-Flood fills depressions; D-infinity routes flow; rivers follow the
single-receiver accumulation tree, decomposed into main stems and tributaries (Hack ordering).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from worldgen.hydrology.crossings import resolve_self_crossings
from worldgen.hydrology.routing import (
    accumulate_flow,
    compute_dinf_flow,
    inter_face_fallback,
    priority_flood,
)
from worldgen.hydrology.stitching import find_connected_components
from worldgen.hydrology.types import (
    River,
    RiverNode,
    RiverSegment,
    StreamNetwork,
    laplacian_smooth_sphere,
)
from core.mesh import PolygonMesh


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

        # Per-node flat arrays.
        self.elevations: np.ndarray = elev.astype(np.float32).reshape(-1)
        self.positions: np.ndarray = mesh.face_texture_positions(h, h).reshape(-1, 3)

        # Ocean mask first — the precip weighting renormalizes over land.
        if ocean_mask_textures is not None:
            omask = ocean_mask_textures[:, :, :, 0] if ocean_mask_textures.ndim == 4 else ocean_mask_textures
            self._ocean_mask: np.ndarray = omask.reshape(-1).astype(bool)
        else:
            self._ocean_mask = self.elevations <= sea_level

        if precipitation_textures is not None:
            annual = precipitation_textures.sum(axis=-1).astype(np.float32)  # (N, H, W)
            flat = annual.reshape(-1)
            mean = float(flat.mean())
            self._mean_precip: float | None = max(mean, 1e-6)
            # Precip-weighted accumulation: each node contributes precip/mean
            # (mean ≈ 1 over all nodes), so wetter cells carry proportionally
            # more flow.
            self._precip_weights: np.ndarray | None = (flat / self._mean_precip).astype(np.float32)
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

        # Populated by compute_flow / water-balance shrinking.
        self._endorheic_mask: np.ndarray = np.zeros(self.n_nodes, dtype=bool)

        # Per-face tangent frames; the inter-face flow fallback projects D8
        # descent directions into each face's UV frame.
        frames = mesh.face_tangent_frames()
        self._u_axes = np.asarray(frames[0], dtype=np.float64)
        self._v_axes = np.asarray(frames[1], dtype=np.float64)

        # Sparse inter-face adjacency; intra-face neighbours derived on the fly.
        # Derived from the mesh's cached boundary pixel pairs (already built by
        # the elevation/climate seam passes), avoiding a separate stitch pass.
        # The pairing uses a strict t∈[0,1] band vs stitch's ±1e-3 slack, so it
        # differs by ~0.3% of edges — visually negligible, and the cache is free.
        self._inter_adj: Dict[int, List[int]] = self._build_inter_adj_from_pairs(mesh)
        # Static node adjacency as CSR (indptr + flat indices) for the JIT-compiled
        # Priority-Flood. Built once; the grid + stitch topology never changes.
        self._build_neighbor_csr()

    def _build_inter_adj_from_pairs(self, mesh: PolygonMesh) -> Dict[int, List[int]]:
        """Build sparse inter-face node adjacency from cached boundary pixel pairs."""
        fa, fb, ra, ca, rb, cb = mesh.boundary_pixel_pairs(self.resolution)
        r2 = self._r2
        res = self.resolution
        node_a = (fa.astype(np.int64) * r2 + ra * res + ca).tolist()
        node_b = (fb.astype(np.int64) * r2 + rb * res + cb).tolist()
        inter_adj: Dict[int, List[int]] = {}
        for a, b in zip(node_a, node_b):
            la = inter_adj.setdefault(a, [])
            if b not in la:
                la.append(b)
            lb = inter_adj.setdefault(b, [])
            if a not in lb:
                lb.append(a)
        return inter_adj

    def _build_neighbor_csr(self) -> None:
        """Precompute static node adjacency as CSR arrays (indptr, flat).

        Builds the D8 intra-face edges (vectorised across all faces) plus the
        stitched inter-face edges, then stable-sorts them by source node. The
        block concatenation order ([up, down, left, right, UL, UR, DL, DR] then
        inter-face) with a *stable* sort reproduces the exact neighbour ordering
        ``_neighbors`` emits. Consumed by the JIT Priority-Flood in routing.py.
        """
        res = self.resolution
        r2 = self._r2
        n_faces = self.n_faces
        n_nodes = self.n_nodes

        local = np.arange(r2, dtype=np.int32)
        rows = local // res
        cols = local % res
        face_base = (np.arange(n_faces, dtype=np.int32) * r2)[:, None]  # (n_faces, 1)

        directions = (
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1),
        )
        src_blocks: List[np.ndarray] = []
        dst_blocks: List[np.ndarray] = []
        for dr, dc in directions:
            valid = (
                (rows + dr >= 0) & (rows + dr < res)
                & (cols + dc >= 0) & (cols + dc < res)
            )
            ls = local[valid]
            ld = ls + dr * res + dc
            src_blocks.append((face_base + ls[None, :]).reshape(-1))
            dst_blocks.append((face_base + ld[None, :]).reshape(-1))

        if self._inter_adj:
            inter_src: List[int] = []
            inter_dst: List[int] = []
            for node, nbrs in self._inter_adj.items():
                inter_src.extend([node] * len(nbrs))
                inter_dst.extend(nbrs)
            src_blocks.append(np.asarray(inter_src, dtype=np.int32))
            dst_blocks.append(np.asarray(inter_dst, dtype=np.int32))

        src = np.concatenate(src_blocks)
        dst = np.concatenate(dst_blocks)
        order = np.argsort(src, kind="stable")
        self._nbr_flat: np.ndarray = dst[order]
        counts = np.bincount(src, minlength=n_nodes)
        indptr = np.zeros(n_nodes + 1, dtype=np.int64)
        np.cumsum(counts, out=indptr[1:])
        self._nbr_indptr: np.ndarray = indptr

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
        eff, was_raised = priority_flood(
            self.elevations, self._ocean_mask, self._nbr_indptr, self._nbr_flat
        )
        self._eff = eff
        self._lake_mask = self._build_lake_mask(was_raised)
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
    # River extraction (accumulation-graph routing)
    # ------------------------------------------------------------------

    def extract_rivers_graph(
        self,
        receiver1: np.ndarray,
        receiver2: np.ndarray,
        frac1: np.ndarray,
        frac2: np.ndarray,
        flow_accum: np.ndarray,
    ) -> List[River]:
        """Build rivers by routing along the D-infinity accumulation graph.

        Each river follows the primary (max-accumulation) receiver chain from a
        channel headwater down to the sea, an endorheic sink, or a confluence
        with a larger stem. The drawn polyline therefore *is* the accumulation
        channel, so the painted flow grows smoothly and only jumps at real
        confluences — unlike the bilinear-gradient trace, whose path was
        decoupled from the accumulation field (off-channel sampling produced a
        flat-then-leap staircase). Tributaries are separate rivers terminating
        at the confluence node where they meet a higher-order stream, so no two
        rivers share an interior cell (a main-stem / Hack decomposition).
        """
        n = self.n_nodes
        ocean = self._ocean_mask
        endo = self._endorheic_mask
        threshold = self.min_flow

        # Primary single receiver = the valid receiver carrying the larger
        # accumulation (follows the main channel at confluences). -1 at outlets /
        # endorheic sinks. This single-receiver graph is the tree we draw.
        r1 = receiver1.astype(np.int64)
        r2 = receiver2.astype(np.int64)
        valid1 = (r1 >= 0) & (frac1 > 0.0)
        valid2 = (r2 >= 0) & (frac2 > 0.0)
        acc1 = np.full(n, -np.inf)
        acc2 = np.full(n, -np.inf)
        acc1[valid1] = flow_accum[r1[valid1]]
        acc2[valid2] = flow_accum[r2[valid2]]
        prx = np.full(n, -1, dtype=np.int64)
        prx[valid1] = r1[valid1]
        take2 = valid2 & (~valid1 | (acc2 > acc1))
        prx[take2] = r2[take2]

        # Tree accumulation: single-receiver flow along the prx edges we draw.
        # Everything keys off this so the network and its painted flow are one
        # consistent quantity — monotonic down each chain, rising only at *drawn*
        # confluences, with sources born right at the threshold. (D-inf
        # `flow_accum` only chooses the direction; it is a DAG, so using it for
        # magnitude leaps where flow joins via the secondary receiver — a jump
        # with no drawn tributary — and using it for the threshold makes sources
        # appear far above it.) Reuses the JIT accumulator with one receiver.
        tree_accum = accumulate_flow(
            prx.astype(np.int32),
            np.full(n, -1, dtype=np.int32),
            np.ones(n, dtype=np.float32),
            np.zeros(n, dtype=np.float32),
            self._eff,
            self._precip_weights,
        )

        # Kept for stream_network (the same tree, cut at a lower threshold).
        self._tree_receiver = prx
        self._tree_accum = tree_accum

        # Channel nodes: enough tree flow, not open sea. Lake interiors stay in
        # (rivers route through exorheic lakes, flagged in_lake downstream).
        is_channel = (tree_accum >= threshold) & ~ocean
        return [
            self._river_from_path(path, tree_accum)
            for path in main_stem_paths(prx, tree_accum, is_channel, ocean | endo)
        ]

    def _river_from_path(self, path: List[int], flow_accum: np.ndarray) -> River:
        """Assemble a River from an ordered list of node indices (headwater→mouth)."""
        res = self.resolution
        n_minus_1 = res - 1
        nodes = np.asarray(path, dtype=np.int64)
        raw_pos = [self.positions[nd].astype(np.float64) for nd in path]
        smooth = laplacian_smooth_sphere(raw_pos, iterations=8, alpha=0.5)
        # On-channel flow: running max of accumulation keeps it monotonic through
        # the minor dips D-inf splitting can leave between adjacent channel cells.
        running = np.maximum.accumulate(flow_accum[nodes]).astype(np.float64)
        in_lake = self._lake_mask[nodes]

        def _node_rec(idx: int) -> RiverNode:
            nd = int(path[idx])
            face = nd // self._r2
            rem = nd % self._r2
            return RiverNode(
                position=smooth[idx].astype(np.float32),
                elevation=float(self.elevations[nd]),
                face_id=int(face),
                uv=((rem % res) / n_minus_1, (rem // res) / n_minus_1),
            )

        mouth_face = path[-1] // self._r2
        river = River(source=_node_rec(0), mouth=_node_rec(len(path) - 1))
        for k in range(1, len(path)):
            river.segments.append(RiverSegment(
                start=RiverNode(position=smooth[k - 1].astype(np.float32),
                                elevation=0.0, face_id=int(mouth_face), uv=(0.0, 0.0)),
                end=RiverNode(position=smooth[k].astype(np.float32),
                              elevation=0.0, face_id=int(mouth_face), uv=(0.0, 0.0)),
                flow_rate=float(running[k - 1]),
                in_lake=bool(in_lake[k - 1] or in_lake[k]),
            ))
        return river

    # ------------------------------------------------------------------
    # Facade
    # ------------------------------------------------------------------

    def generate(self) -> List[River]:
        r1, r2, f1, f2, _ang, accum = self.compute_flow()
        rivers = self.extract_rivers_graph(r1, r2, f1, f2, accum)
        rivers = resolve_self_crossings(rivers)
        return rivers

    def stream_network(self, threshold: float) -> StreamNetwork:
        """The accumulation tree down to ``threshold`` — the rivers' own tree
        (same receivers, same flow), cut far lower. Call after generate()."""
        ocean = self._ocean_mask
        stop = ocean | self._endorheic_mask
        channel = (self._tree_accum >= threshold) & ~ocean
        ch = np.flatnonzero(channel)
        downstream = self._tree_receiver[ch]
        nodes = np.union1d(ch, downstream[downstream >= 0])
        index = np.full(self.n_nodes, -1, dtype=np.int64)
        index[nodes] = np.arange(len(nodes))
        receiver = np.full(len(nodes), -1, dtype=np.int64)
        is_ch = channel[nodes]
        receiver[is_ch] = index[self._tree_receiver[nodes[is_ch]]]
        # Terminal (non-channel) nodes and stops drain nowhere in the network.
        receiver[stop[nodes] & ~is_ch] = -1
        return StreamNetwork(
            nodes=nodes.astype(np.int64),
            positions=self.positions[nodes].astype(np.float32),
            receiver=receiver,
            flow=self._tree_accum[nodes].astype(np.float32),
            channel=is_ch,
            stop=stop[nodes],
            resolution=self.resolution,
        )

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
        """(N, H, W, 1) uint8 lake mask; 1 inside a lake. Call after generate()."""
        res = self.resolution
        return self._lake_mask.reshape(self.n_faces, res, res, 1).astype(np.uint8)


def main_stem_paths(
    receiver: np.ndarray,
    accum: np.ndarray,
    is_channel: np.ndarray,
    is_stop: np.ndarray,
    min_nodes: int = 4,
) -> List[List[int]]:
    """Main-stem (Hack) decomposition of a single-receiver channel tree.

    ``receiver[i]`` is node ``i``'s downstream node (-1 at an outlet), ``accum``
    the flow along those edges, ``is_channel`` the nodes carrying enough of it to
    be drawn and ``is_stop`` the sinks a river ends on (sea, endorheic lakes, …).
    Returns node paths headwater → mouth: each runs from a channel leaf down to a
    sink or to the confluence where it meets a larger stem (that node included),
    so no two paths share an interior node. Paths shorter than ``min_nodes`` are
    dropped as stubs.
    """
    n = len(receiver)
    # main_child[d] = the channel node draining into d with the largest
    # accumulation — the main-stem continuation upstream of d. Ascending
    # iteration leaves the max as the final writer. A chain stops where its node
    # is not its receiver's main_child; channel leaves (main_child == -1) are the
    # headwaters.
    ch = np.where(is_channel)[0]
    ch_sorted = ch[np.argsort(accum[ch], kind="stable")]
    main_child = np.full(n, -1, dtype=np.int64)
    for node in ch_sorted.tolist():
        d = receiver[node]
        if d >= 0 and is_channel[d]:
            main_child[d] = node

    paths: List[List[int]] = []
    for src in ch[main_child[ch] == -1].tolist():
        path = [src]
        cur = src
        for _ in range(n):
            d = int(receiver[cur])
            if d < 0:
                break
            path.append(d)
            if is_stop[d]:
                break
            if main_child[d] != cur:   # merged into a larger stem here
                break
            cur = d
        if len(path) >= min_nodes:     # drop stubs (matches old >2-segment rule)
            paths.append(path)
    return paths
