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
from worldgen.hydrology.stitching import find_connected_components_csr
from worldgen.hydrology.types import (
    River,
    RiverNode,
    RiverSegment,
    StreamNetwork,
    laplacian_smooth_sphere,
)
from core.mesh import PolygonMesh


def potential_evapotranspiration(
    t_annual_c: np.ndarray, baseline_mm_yr: float, per_degree_mm_yr: float, max_mm_yr: float,
) -> np.ndarray:
    """Annual PET (mm/yr) = ``clamp(baseline + per_degree · max(0, T), 0, max)`` of annual mean °C."""
    return np.clip(baseline_mm_yr + per_degree_mm_yr * np.maximum(0.0, t_annual_c), 0.0, max_mm_yr)


def inter_face_adjacency(mesh: PolygonMesh, resolution: int) -> Dict[int, List[int]]:
    """Sparse inter-face texel adjacency (flat ids) from the mesh's cached boundary pixel pairs."""
    fa, fb, ra, ca, rb, cb = mesh.boundary_pixel_pairs(resolution)
    r2 = resolution * resolution
    node_a = (fa.astype(np.int64) * r2 + ra * resolution + ca).tolist()
    node_b = (fb.astype(np.int64) * r2 + rb * resolution + cb).tolist()
    inter_adj: Dict[int, List[int]] = {}
    for a, b in zip(node_a, node_b):
        la = inter_adj.setdefault(a, [])
        if b not in la:
            la.append(b)
        lb = inter_adj.setdefault(b, [])
        if a not in lb:
            lb.append(a)
    return inter_adj


def texel_neighbor_csr(
    n_faces: int, resolution: int, inter_adj: Dict[int, List[int]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Static texel adjacency over all faces as CSR ``(indptr, flat)``.

    Per-node order — [up, down, left, right, UL, UR, DL, DR] then inter-face —
    must match ``RiverGenerator._neighbors``.
    """
    res = resolution
    r2 = res * res
    n_nodes = n_faces * r2

    local = np.arange(r2, dtype=np.int32)
    rows = local // res
    cols = local % res
    face_base = (np.arange(n_faces, dtype=np.int32) * r2)[:, None]  # (n_faces, 1)

    directions = (
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
    )
    valid = np.stack([
        (rows + dr >= 0) & (rows + dr < res) & (cols + dc >= 0) & (cols + dc < res)
        for dr, dc in directions
    ])                                                       # (8, r2)
    # Slot within a node's row = number of valid directions before it.
    slot = np.cumsum(valid, axis=0) - valid                  # (8, r2)
    n_intra = valid.sum(axis=0)                              # (r2,)

    if inter_adj:
        inter_src = np.fromiter(
            (node for node, nbrs in inter_adj.items() for _ in nbrs), dtype=np.int64)
        inter_dst = np.fromiter(
            (nb for nbrs in inter_adj.values() for nb in nbrs), dtype=np.int32)
    else:
        inter_src = np.zeros(0, dtype=np.int64)
        inter_dst = np.zeros(0, dtype=np.int32)

    intra_counts = np.tile(n_intra, n_faces)
    counts = intra_counts + np.bincount(inter_src, minlength=n_nodes)
    indptr = np.zeros(n_nodes + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    flat = np.empty(int(indptr[-1]), dtype=np.int32)
    for d, (dr, dc) in enumerate(directions):
        ls = local[valid[d]]
        src = (face_base + ls[None, :]).reshape(-1)
        flat[indptr[src] + np.tile(slot[d][ls], n_faces)] = src + (dr * res + dc)
    # Inter-face neighbours follow, in ``inter_adj`` order.
    order = np.argsort(inter_src, kind="stable")
    s_sorted = inter_src[order]
    first = np.searchsorted(s_sorted, s_sorted, side="left")
    rank = np.arange(s_sorted.size) - first
    flat[indptr[s_sorted] + intra_counts[s_sorted] + rank] = inter_dst[order]
    return indptr, flat


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
            # Each node contributes precip/mean (≈ 1 on average) to accumulation.
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

        # For the inter-face flow fallback (D8 descent projected into face UV).
        frames = mesh.face_tangent_frames()
        self._u_axes = np.asarray(frames[0], dtype=np.float64)
        self._v_axes = np.asarray(frames[1], dtype=np.float64)

        # Inter-face adjacency from the mesh's cached boundary pixel pairs (no
        # separate stitch pass; their strict t∈[0,1] band differs from stitch's
        # ±1e-3 slack on ~0.3% of edges — negligible).
        self._inter_adj: Dict[int, List[int]] = inter_face_adjacency(mesh, h)
        self._nbr_indptr, self._nbr_flat = texel_neighbor_csr(n_faces, h, self._inter_adj)

    # ------------------------------------------------------------------
    # Node indexing helpers
    # ------------------------------------------------------------------

    def _node_idx(self, face: int, row: int, col: int) -> int:
        return face * self._r2 + row * self.resolution + col

    def _unpack(self, node: int) -> Tuple[int, int, int]:
        face = node // self._r2
        rem = node % self._r2
        return face, rem // self.resolution, rem % self.resolution

    def _components(self, mask: np.ndarray, min_size: int = 1) -> List[List[int]]:
        """``find_connected_components`` over the node graph, via the neighbour CSR."""
        return find_connected_components_csr(mask, self._nbr_indptr, self._nbr_flat, min_size=min_size)

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
        """Shrink endorheic lakes until total PET equals inflow.

        Updates _lake_mask, _endorheic_mask and _eff (so traces descend into the
        shrunken lake). No-op without climate inputs.
        """
        if self._t_annual_flat is None or self._precip_weights is None or self._mean_precip is None:
            return

        components = self._components(self._lake_mask, min_size=1)
        if not components:
            return

        pet_mm_yr = potential_evapotranspiration(
            self._t_annual_flat, self._pet_baseline_mm_yr, self._pet_per_degree_mm_yr, self._pet_max_mm_yr,
        ).astype(np.float32)
        pet_norm = pet_mm_yr / self._mean_precip  # mean-precip units

        for comp in components:
            comp_arr = np.asarray(comp, dtype=np.int64)
            inflow = float(flow_accum[comp_arr].max())  # ≈ upstream contribution at the spill pixel
            total_pet_norm = float(pet_norm[comp_arr].sum())
            if inflow >= total_pet_norm:
                continue  # exorheic: keep the full fill

            elevs = self.elevations[comp_arr]
            order = np.argsort(elevs, kind='stable')
            sorted_pixels = comp_arr[order]
            cum_pet = np.cumsum(pet_norm[sorted_pixels])
            # Smallest prefix whose cumulative PET reaches the inflow.
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
        """Bool mask of flooded land components with >= min_lake_pixels."""
        raw = was_raised & ~self._ocean_mask
        mask = np.zeros(self.n_nodes, dtype=bool)
        for comp in self._components(raw, min_size=self.min_lake_pixels):
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
        """Build rivers along the primary-receiver tree of the D-infinity graph.

        Each river runs from a headwater to the sea, an endorheic sink, or its
        confluence with a larger stem (Hack decomposition), so the drawn line is
        the accumulation channel and flow jumps only at real confluences.
        """
        n = self.n_nodes
        ocean = self._ocean_mask
        endo = self._endorheic_mask
        threshold = self.min_flow

        # Primary receiver = the valid one with larger accumulation; -1 at outlets / sinks.
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

        # Single-receiver accumulation along the drawn tree: monotonic, rising
        # only at drawn confluences, sources born at the threshold. D-inf
        # `flow_accum` only picks direction — as magnitude it leaps where flow
        # joins via an undrawn secondary receiver.
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

        # Lake interiors stay in: rivers route through exorheic lakes (flagged in_lake).
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
        # Running max smooths minor dips D-inf splitting leaves along the channel.
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
        """The rivers' own accumulation tree, cut at a lower ``threshold``. Call after generate()."""
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
        for comp in self._components(self._lake_mask, min_size=1):
            boundary = [
                node for node in comp
                if any(not self._lake_mask[nb] for nb in self._neighbors(node))
            ]
            if len(boundary) < 3:
                continue

            boundary_set = set(boundary)
            adj: Dict[int, List[int]] = {
                node: [nb for nb in self._neighbors(node) if nb in boundary_set]
                for node in boundary
            }
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

    ``receiver[i]`` is node ``i``'s downstream node (-1 at an outlet);
    ``is_stop`` marks sinks (sea, endorheic lakes). Returns headwater → mouth
    paths, each ending at a sink or at its confluence node with a larger stem,
    so no two share an interior node. Paths under ``min_nodes`` are dropped.
    """
    n = len(receiver)
    # main_child[d] = largest-accumulation channel node draining into d (ascending
    # iteration leaves the max as last writer); leaves (-1) are headwaters.
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
        if len(path) >= min_nodes:
            paths.append(path)
    return paths
