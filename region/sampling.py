"""Sampling the global face model at arbitrary sphere positions.

Provides the smooth (bilinear) read of a global texture and the **anchor**, the
global texel each regional texel must average back to (the conservation
invariant in ``docs/plans/regional_local_maps_design.md``); the tier-agnostic
resampling math is :mod:`utils.resample`.

Face ownership is by nearest centroid, which matches the rendered polygons
except within a texel of the seams; textures over-hang, so reads stay valid there.
"""
from __future__ import annotations

import weakref
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import KDTree

from utils.resample import bilinear_plan, bilinear_read, settle_owners

if TYPE_CHECKING:
    from core.mesh import PolygonMesh


# mesh → {(h, w): owners}, shared by every sampler of a mesh (geometry-only).
_OWNERS: "weakref.WeakKeyDictionary[PolygonMesh, dict[tuple[int, int], np.ndarray]]" = weakref.WeakKeyDictionary()


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
        self._owners = _OWNERS.setdefault(mesh, {})
        # (faces, u, v, h, w) → bilinear_plan, for repeated reads at the same points.
        self._plans: dict[tuple, tuple] = {}

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
        """Bilinear read of an ``(N, H, W, C)`` texture → ``(P, C)`` float64.

        Points just outside the texture square are extrapolated, not clamped,
        so the read keeps its slope instead of flattening into a wedge.
        """
        tex = np.asarray(texture)
        _, h, w, c = tex.shape
        key = (id(faces), id(u), id(v), h, w)
        plan = self._plans.get(key)
        if plan is None:
            x = np.clip(u, -0.5, 1.5) * (w - 1)
            y = np.clip(v, -0.5, 1.5) * (h - 1)
            plan = bilinear_plan(faces, x, y, h, w)
            if len(self._plans) >= 8:
                self._plans.clear()
            self._plans[key] = (faces, u, v, plan)     # the arrays pin their ids
        else:
            plan = plan[3]
        return bilinear_read(tex, plan)

    def anchors(
        self, faces: np.ndarray, u: np.ndarray, v: np.ndarray, h: int, w: int
    ) -> np.ndarray:
        """``(P,)`` flat index ``face*H*W + row*W + col`` of the global texel owning each point.

        The nearest texel, passed through :meth:`owners` so points near a seam
        go to the texel whose centre is on their side.
        """
        return self.owners(h, w)[self._nearest_texels(faces, u, v, h, w)]

    def _nearest_texels(
        self, faces: np.ndarray, u: np.ndarray, v: np.ndarray, h: int, w: int
    ) -> np.ndarray:
        x, y = self._texel_coords(u, v, h, w)
        col = np.clip(np.rint(x).astype(np.int64), 0, w - 1)
        row = np.clip(np.rint(y).astype(np.int64), 0, h - 1)
        return faces * (h * w) + row * w + col

    def owners(self, h: int, w: int) -> np.ndarray:
        """``(N*H*W,)`` the texel owning each global texel's position.

        Its own index when its centre is in its face's territory, else the
        neighbouring face's nearest texel.  Idempotent: an owner owns itself.
        """
        cached = self._owners.get((h, w))
        if cached is not None:
            return cached
        n = len(self._u_axes)
        uu = np.linspace(0.0, 1.0, w)[None, None, :, None]
        vv = np.linspace(0.0, 1.0, h)[None, :, None, None]
        pos = (
            self._tangent_centroids[:, None, None, :]
            + (self._u_offsets[:, None, None, None] + uu * self._u_ranges[:, None, None, None])
            * self._u_axes[:, None, None, :]
            + (self._v_offsets[:, None, None, None] + vv * self._v_ranges[:, None, None, None])
            * self._v_axes[:, None, None, :]
        ).reshape(n * h * w, 3)
        owner = settle_owners(self._nearest_texels(*self.locate(pos), h, w))
        owner.flags.writeable = False
        self._owners[(h, w)] = owner
        return owner
