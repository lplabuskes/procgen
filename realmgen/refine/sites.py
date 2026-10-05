"""Per-texel site features, shared by city siting and town placement.

**Ownership.** Face textures over-hang their polygons, so a texel counts for
face ``f`` only when ``f`` is its nearest face centroid (the rule
``GlobalSampler`` and the polity label field use). This partitions the sphere,
so a settlement on an owned texel lies inside its face.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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
    ``max_step`` (chord). ``None`` when there are no points."""
    chunks: list[np.ndarray] = []
    for pts in polylines:
        pts = np.asarray(pts, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] == 0:
            continue
        chunks.append(pts)
        if pts.shape[0] < 2 or max_step <= 0:
            continue
        a, b = pts[:-1], pts[1:]
        n_sub = np.ceil(np.linalg.norm(b - a, axis=1) / max_step).astype(np.int64)
        for k in np.flatnonzero(n_sub > 1):
            t = (np.arange(1, n_sub[k]) / n_sub[k])[:, None]
            chunks.append(a[k] + t * (b[k] - a[k]))
    if not chunks:
        return None
    return np.concatenate(chunks, axis=0)


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
        self._confluence_tree = _build_confluence_tree(rivers, search_radius=confluence_search_radius)
        # Hydrology ``StreamNetwork`` channel texels; only town siting passes one.
        self._stream = np.zeros(self._ocean.shape, dtype=bool)
        if streams is not None and len(streams):
            self._stream.reshape(-1)[streams.nodes[streams.channel]] = True
        self._cache: dict[int, FaceSites] = {}

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
