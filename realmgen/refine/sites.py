"""Per-texel site features, shared by city siting and town placement.

**Ownership.** Face textures over-hang their polygons, so a texel counts for
face ``f`` only when ``f`` is its nearest face centroid (the rule
``GlobalSampler`` and the polity label field use). This partitions the sphere,
so a settlement on an owned texel lies inside its face.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.spatial import KDTree


def _neighbor_any(mask: np.ndarray) -> np.ndarray:
    """True where any 4-neighbour (no wrap) is True, over the last two axes of ``(..., H, W)``."""
    out = np.zeros_like(mask)
    out[..., 1:, :] |= mask[..., :-1, :]
    out[..., :-1, :] |= mask[..., 1:, :]
    out[..., :, 1:] |= mask[..., :, :-1]
    out[..., :, :-1] |= mask[..., :, 1:]
    return out


def _box_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted 3×3 box mean over the last two axes of ``(..., H, W)`` (edges use what exists)."""
    num = np.zeros_like(values)
    den = np.zeros_like(values)
    vw = values * weights
    h, w = values.shape[-2:]
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            dst_i = slice(max(0, -di), h - max(0, di))
            dst_j = slice(max(0, -dj), w - max(0, dj))
            src_i = slice(max(0, di), h - max(0, -di))
            src_j = slice(max(0, dj), w - max(0, -dj))
            num[..., dst_i, dst_j] += vw[..., src_i, src_j]
            den[..., dst_i, dst_j] += weights[..., src_i, src_j]
    return np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)


def densify_polylines(polylines, max_step: float) -> Optional[np.ndarray]:
    """All vertices of ``polylines`` plus interpolated points so no gap exceeds
    ``max_step`` (chord). ``None`` when there are no points.

    Per polyline: its vertices, then each split segment's interior points
    ``1/n … (n-1)/n`` in turn.  Vectorised over all polylines at once.
    """
    lines = [p for p in (np.asarray(x, dtype=np.float64) for x in polylines) if p.ndim == 2 and p.shape[0] > 0]
    if not lines:
        return None
    sizes = np.array([len(p) for p in lines], dtype=np.int64)
    pts = np.concatenate(lines, axis=0)
    if max_step <= 0 or len(pts) < 2:
        return pts
    line_of = np.repeat(np.arange(len(lines)), sizes)
    # Segments between consecutive vertices of one polyline.
    seg_ok = line_of[:-1] == line_of[1:]
    a, b = pts[:-1], pts[1:]
    n_sub = np.ceil(np.linalg.norm(b - a, axis=1) / max_step).astype(np.int64)
    count = np.where(seg_ok & (n_sub > 1), n_sub - 1, 0)
    if not count.any():
        return pts
    # Output rows: each polyline's vertices, then its interior points.
    interior = np.bincount(line_of[:-1], weights=count, minlength=len(lines)).astype(np.int64)
    start = np.cumsum(sizes + interior) - (sizes + interior)
    out = np.empty((int((sizes + interior).sum()), 3))
    out[start[line_of] + (np.arange(len(pts)) - np.repeat(np.cumsum(sizes) - sizes, sizes))] = pts
    split = np.flatnonzero(count)
    seg = np.repeat(split, count[split])
    step = np.arange(len(seg)) - np.repeat(np.cumsum(count[split]) - count[split], count[split]) + 1
    t = (step / n_sub[seg])[:, None]
    # Rank of each interior point within its polyline's interior block.
    rank = np.arange(len(seg)) - (np.cumsum(interior) - interior)[line_of[seg]]
    out[start[line_of[seg]] + sizes[line_of[seg]] + rank] = a[seg] + t * (b[seg] - a[seg])
    return out


def _build_confluence_tree(rivers, search_radius: float = 0.01) -> Optional[KDTree]:
    """KD-tree over river confluence points.

    A river mouth is a confluence when it lies within ``search_radius`` (chord)
    of an interior vertex of a different river.
    """
    if not rivers or len(rivers) < 2:
        return None

    mouths: list[tuple[int, np.ndarray]] = []
    interior_chunks: list[np.ndarray] = []
    river_id_list: list[int] = []

    for i, entry in enumerate(rivers):
        pts = np.asarray(entry[0], dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] < 2:
            continue
        mouths.append((i, pts[-1]))
        interior_chunks.append(pts[:-1])
        river_id_list.extend([i] * (len(pts) - 1))

    if not mouths or not interior_chunks:
        return None

    interior_arr = np.concatenate(interior_chunks, axis=0)
    river_ids = np.array(river_id_list, dtype=np.int32)
    interior_tree = KDTree(interior_arr)

    confluence_pts = []
    for river_idx, mouth in mouths:
        neighbors = interior_tree.query_ball_point(mouth, r=search_radius)
        if any(river_ids[nb] != river_idx for nb in neighbors):
            confluence_pts.append(mouth)

    if not confluence_pts:
        return None
    return KDTree(np.stack(confluence_pts))


def _texel_spacing(face_positions: np.ndarray) -> float:
    """Median 3-D distance between horizontally-adjacent texels of one face."""
    grid = np.asarray(face_positions, dtype=np.float64)  # (H, W, 3)
    diffs = np.linalg.norm(grid[:, 1:, :] - grid[:, :-1, :], axis=2)
    if diffs.size == 0:
        return 0.0
    return float(np.median(diffs))


@dataclass
class FaceSites:
    """Site features of one face's ``H × W`` texel grid, flattened to ``(T,)``."""

    face: int
    positions: np.ndarray      # (T, 3) texel centres
    land: np.ndarray           # (T,) not ocean / lake
    owned: np.ndarray          # (T,) nearest face centroid is ``face``
    coast_ocean: np.ndarray    # (T,) land, 4-adjacent to ocean
    coast_lake: np.ndarray     # (T,) land, 4-adjacent to lake
    river: np.ndarray          # (T,) land within the river snap distance of a river
    confluence: np.ndarray     # (T,) land within the river snap distance of a confluence
    stream: np.ndarray         # (T,) land texel on a channel of the stream network
    fertility: np.ndarray      # (T,) 3×3 land-mean fertility in [0, 1]
    spacing: float             # texel spacing (chord)

    @property
    def coast(self) -> np.ndarray:
        return self.coast_ocean | self.coast_lake

    def sites(self) -> np.ndarray:
        """Indices of owned land texels, else any land texel (so a land face always has a site)."""
        own = np.flatnonzero(self.land & self.owned)
        return own if own.size else np.flatnonzero(self.land)


class SiteField:
    """Lazily computed :class:`FaceSites` over a mesh's face textures.

    ``river_threshold_factor`` × texel spacing is the river/confluence snap
    distance. ``positions`` may reuse precomputed ``face_texture_positions``.
    :meth:`prefetch` all needed faces at once; batching is far cheaper.
    """

    def __init__(
        self,
        mesh,
        ocean_tex: np.ndarray,
        lake_tex: Optional[np.ndarray],
        fertility_tex: Optional[np.ndarray],
        rivers,
        *,
        river_threshold_factor: float = 1.5,
        confluence_search_radius: float = 0.01,
        positions: Optional[np.ndarray] = None,
        streams=None,
    ) -> None:
        self.n, self.h, self.w = int(ocean_tex.shape[0]), int(ocean_tex.shape[1]), int(ocean_tex.shape[2])
        self._positions = (
            positions if positions is not None else mesh.face_texture_positions(self.h, self.w)
        )                                                                  # (N, H, W, 3)
        self.centroids = np.asarray(mesh.face_centroids, dtype=np.float64)
        self._centroid_tree = KDTree(self.centroids)
        self._ocean = ocean_tex[..., 0] > 0.5
        self._lake = (lake_tex[..., 0] > 0.5) if lake_tex is not None else np.zeros_like(self._ocean)
        self._fertility = (
            np.clip(np.asarray(fertility_tex[..., 0], dtype=np.float64), 0.0, 1.0)
            if fertility_tex is not None else None
        )
        self._river_factor = float(river_threshold_factor)
        rivers = rivers or []
        spacing = _texel_spacing(self._positions[0]) if self.n else 0.0
        river_pts = densify_polylines([entry[0] for entry in rivers], 0.5 * spacing)
        self._river_tree = KDTree(river_pts) if river_pts is not None else None
        lines = [np.asarray(entry[0], dtype=np.float64) for entry in rivers]
        # (R,) each river's largest flow, ranking the water a city's links can share.
        self.river_flows = np.array([float(np.max(e[1])) if len(e[1]) else 0.0 for e in rivers], dtype=np.float64)
        # Each river on its own, for cities tagged with one (``Settlement.water_links``).
        self.river_proximity = RiverProximity(lambda r: [lines[r]] if 0 <= r < len(lines) else [], 0.5 * spacing)
        self._confluence_tree = _build_confluence_tree(rivers, search_radius=confluence_search_radius)
        # Hydrology ``StreamNetwork`` channel texels; only town siting passes one.
        self._stream = np.zeros(self._ocean.shape, dtype=bool)
        if streams is not None and len(streams):
            self._stream.reshape(-1)[streams.nodes[streams.channel]] = True
        self._cache: dict[int, FaceSites] = {}

    def near_river(self, face: FaceSites, idx: np.ndarray, r: int) -> np.ndarray:
        """``(len(idx),)`` True where ``face``'s texels ``idx`` are within the river snap distance of river ``r``."""
        return self.river_proximity.near(r, face.positions[idx], self._river_factor * face.spacing)

    def owner(self, points: np.ndarray) -> np.ndarray:
        """Nearest-face-centroid owner of each of ``points`` ``(P, 3)``."""
        return np.asarray(self._centroid_tree.query(points, workers=-1)[1], dtype=np.int64)

    def face(self, f: int) -> FaceSites:
        f = int(f)
        if f not in self._cache:
            self.prefetch([f])
        return self._cache[f]

    def prefetch(self, faces) -> None:
        """Compute (and cache) the :class:`FaceSites` of ``faces`` in one batch."""
        faces = np.array(sorted({int(f) for f in faces} - self._cache.keys()), dtype=np.int64)
        if faces.size == 0:
            return
        n_f, t = faces.size, self.h * self.w
        grids = np.asarray(self._positions[faces], dtype=np.float64)       # (F, H, W, 3)
        pos = grids.reshape(n_f, t, 3)
        ocean, lake = self._ocean[faces], self._lake[faces]                 # (F, H, W)
        land2d = ~(ocean | lake)
        land = land2d.reshape(n_f, t)
        owned = self.owner(pos.reshape(-1, 3)).reshape(n_f, t) == faces[:, None]
        spacing = np.median(np.linalg.norm(grids[:, :, 1:] - grids[:, :, :-1], axis=3), axis=(1, 2))

        near = (self._river_factor * spacing)[:, None]                      # (F, 1)
        river = np.zeros_like(land)
        confluence = np.zeros_like(land)
        land_pts = pos[land]
        if land_pts.shape[0]:
            if self._river_tree is not None:
                d = np.full(land.shape, np.inf)
                d[land] = self._river_tree.query(land_pts, workers=-1)[0]
                river = d < near
            if self._confluence_tree is not None:
                d = np.full(land.shape, np.inf)
                d[land] = self._confluence_tree.query(land_pts, workers=-1)[0]
                confluence = d < near

        if self._fertility is not None:
            fert = _box_mean(self._fertility[faces], land2d.astype(np.float64)).reshape(n_f, t)
        else:
            fert = np.zeros(land.shape, dtype=np.float64)
        coast_ocean = (land2d & _neighbor_any(ocean)).reshape(n_f, t)
        coast_lake = (land2d & _neighbor_any(lake)).reshape(n_f, t)
        stream = (land2d & self._stream[faces]).reshape(n_f, t)

        for k, f in enumerate(faces.tolist()):
            self._cache[f] = FaceSites(
                face=f,
                positions=pos[k],
                land=land[k],
                owned=owned[k],
                coast_ocean=coast_ocean[k],
                coast_lake=coast_lake[k],
                river=river[k],
                confluence=confluence[k],
                stream=stream[k],
                fertility=np.where(land[k], fert[k], 0.0),
                spacing=float(spacing[k]),
            )


class RiverProximity:
    """Per world river, a KD-tree of its course, built on demand.

    ``runs(r)`` gives river ``r``'s polylines (a tier's own drawing of it).
    """

    def __init__(self, runs: Callable[[int], list[np.ndarray]], step: float) -> None:
        self._runs = runs
        self._step = float(step)
        self._trees: dict[int, Optional[KDTree]] = {}

    def near(self, r: int, points: np.ndarray, radius: float) -> np.ndarray:
        """``(P,)`` True where ``points`` lie within ``radius`` (chord) of river ``r``."""
        if r not in self._trees:
            pts = densify_polylines(self._runs(r), self._step)
            self._trees[r] = KDTree(pts) if pts is not None else None
        tree = self._trees[r]
        if tree is None or len(points) == 0:
            return np.zeros(len(points), dtype=bool)
        return np.asarray(tree.query(points, distance_upper_bound=radius)[0] < radius)
