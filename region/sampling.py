"""Sampling the global face model at arbitrary sphere positions.

Regional upsampling needs two things from the globe at every regional texel:

- a **smooth** reading of a global texture (bilinear across the owning face's
  texel grid) — the coarse field the detail is added to;
- the **anchor**: which single global texel that regional texel belongs to.  The
  anchor is the aggregation target — every regional texel sharing an anchor must
  average back to that global texel's value (the conservation invariant in
  ``docs/plans/regional_local_maps_design.md``).

Face ownership is by nearest centroid.  The mesh dual's faces are very nearly the
Voronoi cells of those centroids, so this agrees with the rendered polygons
except within a texel of the seams — and face textures over-hang their polygon,
so a point assigned to a neighbour still reads a valid (seam-smoothed) value.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import KDTree

if TYPE_CHECKING:
    from core.mesh import PolygonMesh


class GlobalSampler:
    """Locate sphere points on the global mesh and read its face textures."""

    def __init__(self, mesh: "PolygonMesh") -> None:
        self.mesh = mesh
        centroids = np.asarray(mesh.face_centroids, dtype=np.float64)
        self._tree = KDTree(centroids / np.linalg.norm(centroids, axis=1, keepdims=True))
        frames = mesh.face_tangent_frames()
        self._u_axes = np.asarray(frames[0], dtype=np.float64)
        self._v_axes = np.asarray(frames[1], dtype=np.float64)
        self._u_offsets = np.asarray(frames[2], dtype=np.float64)
        self._v_offsets = np.asarray(frames[3], dtype=np.float64)
        self._u_ranges = np.asarray(frames[4], dtype=np.float64)
        self._v_ranges = np.asarray(frames[5], dtype=np.float64)
        self._tangent_centroids = np.asarray(mesh._tangent_centroids, dtype=np.float64)

    def locate(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(P, 3)`` unit-sphere points → ``(faces, u, v)``, each ``(P,)``.

        ``u``/``v`` are the owning face's texture coordinates; they leave
        ``[0, 1]`` only for points in that face's texture over-hang.
        """
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        pts = pts / np.maximum(np.linalg.norm(pts, axis=1, keepdims=True), 1e-15)
        _, faces = self._tree.query(pts, k=1, workers=-1)
        faces = np.asarray(faces, dtype=np.int64)
        offset = pts - self._tangent_centroids[faces]
        u_local = np.einsum("ij,ij->i", offset, self._u_axes[faces])
        v_local = np.einsum("ij,ij->i", offset, self._v_axes[faces])
        u = (u_local - self._u_offsets[faces]) / self._u_ranges[faces]
        v = (v_local - self._v_offsets[faces]) / self._v_ranges[faces]
        return faces, u, v

    @staticmethod
    def _texel_coords(u: np.ndarray, v: np.ndarray, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
        """Face UV → continuous texel coordinates on the mesh's endpoint grid."""
        x = np.clip(u, 0.0, 1.0) * (w - 1)
        y = np.clip(v, 0.0, 1.0) * (h - 1)
        return x, y

    def sample(
        self, texture: np.ndarray, faces: np.ndarray, u: np.ndarray, v: np.ndarray
    ) -> np.ndarray:
        """Bilinear read of an ``(N, H, W, C)`` texture → ``(P, C)`` float64."""
        tex = np.asarray(texture)
        _, h, w, c = tex.shape
        x, y = self._texel_coords(u, v, h, w)
        x0 = np.clip(np.floor(x).astype(np.int64), 0, max(w - 2, 0))
        y0 = np.clip(np.floor(y).astype(np.int64), 0, max(h - 2, 0))
        x1 = np.minimum(x0 + 1, w - 1)
        y1 = np.minimum(y0 + 1, h - 1)
        fx = (x - x0)[:, None]
        fy = (y - y0)[:, None]
        t00 = tex[faces, y0, x0].astype(np.float64)
        t01 = tex[faces, y0, x1].astype(np.float64)
        t10 = tex[faces, y1, x0].astype(np.float64)
        t11 = tex[faces, y1, x1].astype(np.float64)
        top = t00 * (1.0 - fx) + t01 * fx
        bot = t10 * (1.0 - fx) + t11 * fx
        return (top * (1.0 - fy) + bot * fy).reshape(-1, c)

    def anchors(
        self, faces: np.ndarray, u: np.ndarray, v: np.ndarray, h: int, w: int
    ) -> np.ndarray:
        """``(P,)`` flat index ``face*H*W + row*W + col`` of the nearest global texel."""
        x, y = self._texel_coords(u, v, h, w)
        col = np.clip(np.rint(x).astype(np.int64), 0, w - 1)
        row = np.clip(np.rint(y).astype(np.int64), 0, h - 1)
        return faces * (h * w) + row * w + col

    @staticmethod
    def anchor_values(texture: np.ndarray, anchors: np.ndarray) -> np.ndarray:
        """``(P, C)`` value of each anchor texel in an ``(N, H, W, C)`` texture."""
        tex = np.asarray(texture)
        return tex.reshape(-1, tex.shape[3])[anchors].astype(np.float64)


def anchor_means(
    values: np.ndarray, anchors: np.ndarray, weights: np.ndarray, n_anchors: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-anchor weighted mean of ``(P, C)`` values → ``((n_anchors, C), counts)``.

    ``weights`` is the ``(P,)`` inclusion weight — 1 for texels that count toward
    the aggregate, 0 for the over-hang duplicates.  Anchors with no weight get a
    mean of 0 and a count of 0; callers must leave those samples uncorrected.
    """
    vals = np.asarray(values, dtype=np.float64)
    if vals.ndim == 1:
        vals = vals[:, None]
    w = np.asarray(weights, dtype=np.float64)
    counts = np.bincount(anchors, weights=w, minlength=n_anchors)
    sums = np.stack(
        [np.bincount(anchors, weights=w * vals[:, k], minlength=n_anchors)
         for k in range(vals.shape[1])],
        axis=1,
    )
    safe = np.maximum(counts, 1e-12)[:, None]
    return sums / safe, counts
