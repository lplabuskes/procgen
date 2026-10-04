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

import weakref
from typing import TYPE_CHECKING, Optional

import numpy as np
from scipy.spatial import KDTree

if TYPE_CHECKING:
    from core.mesh import PolygonMesh


# mesh → {(h, w): owners}.  Ownership depends only on the geometry, so every
# sampler of a mesh (each region and local map makes its own) shares it.
_OWNERS: "weakref.WeakKeyDictionary[PolygonMesh, dict[tuple[int, int], np.ndarray]]" = weakref.WeakKeyDictionary()


def bilinear_plan(
    items: np.ndarray, x: np.ndarray, y: np.ndarray, h: int, w: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(corners (4, P), fx (P, 1), fy (P, 1))`` for bilinear reads of ``(N, H, W, C)``
    textures at texel coordinates ``(x, y)`` of item ``items``: the flat
    ``(N·H·W)`` indices of the four corners and the fractions between them."""
    x0 = np.clip(np.floor(x).astype(np.int64), 0, max(w - 2, 0))
    y0 = np.clip(np.floor(y).astype(np.int64), 0, max(h - 2, 0))
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    base = np.asarray(items, dtype=np.int64) * (h * w)
    corners = np.stack([base + y0 * w + x0, base + y0 * w + x1, base + y1 * w + x0, base + y1 * w + x1])
    return corners, (x - x0)[:, None], (y - y0)[:, None]


def bilinear_read(texture: np.ndarray, plan: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """``(P, C)`` float64 bilinear read of an ``(N, H, W, C)`` texture along a :func:`bilinear_plan`."""
    corners, fx, fy = plan
    flat = texture.reshape(-1, texture.shape[3])
    t00, t01, t10, t11 = (flat[k].astype(np.float64) for k in corners)
    top = t00 * (1.0 - fx) + t01 * fx
    bot = t10 * (1.0 - fx) + t11 * fx
    return top * (1.0 - fy) + bot * fy


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

        A point just outside its face's texture square (the face's Voronoi cell
        pokes past it near some corners) is extrapolated from the edge texels
        rather than clamped, so the read keeps its slope there instead of
        flattening into a wedge.
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

        The nearest texel of the owning face, passed through :meth:`owners`: a
        point within half a texel of a seam can be nearest a texel whose own
        centre lies across the seam, and then belongs to the neighbour's texel
        there instead, so every anchor's points sit around its own centre.
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

        Its own index when the texel's centre lies in its own face's territory
        (nearest centroid); otherwise the neighbouring face's texel nearest it.
        A seam's duplicated edge texels both point at one of the pair; the
        over-hang past a face's polygon points at the neighbour.  Idempotent:
        an owner owns itself.
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

    @staticmethod
    def anchor_values(texture: np.ndarray, anchors: np.ndarray) -> np.ndarray:
        """``(P, C)`` value of each anchor texel in an ``(N, H, W, C)`` texture."""
        tex = np.asarray(texture)
        return tex.reshape(-1, tex.shape[3])[anchors].astype(np.float64)


def settle_owners(owner: np.ndarray) -> np.ndarray:
    """Make a texel → owning-texel map idempotent (every owner owns itself).

    An owner's own centre can lie just across another seam, so ownership forms
    chains — and, where duplicated seam texels each sit a hair inside the
    other's territory, cycles.  Chains are followed to the cycle they end on
    (a fixed point is a cycle of one), and each cycle goes to its smallest member.
    """
    owner = np.asarray(owner, dtype=np.int64)
    landed = owner.copy()
    for _ in range(16):                       # owner^(2^16): every chain reaches its cycle
        nxt = landed[landed]
        if np.array_equal(nxt, landed):
            return landed
        landed = nxt
    rep, cur = landed.copy(), owner[landed]
    for _ in range(16):                       # min over each (short) cycle
        rep = np.minimum(rep, cur)
        cur = owner[cur]
    return rep


def anchor_means(
    values: np.ndarray, anchors: np.ndarray, weights: np.ndarray, n_anchors: int,
    counts: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-anchor weighted mean of ``(P, C)`` values → ``((n_anchors, C), counts)``.

    ``weights`` is the ``(P,)`` inclusion weight — 1 for texels that count toward
    the aggregate, 0 for the over-hang duplicates.  Anchors with no weight get a
    mean of 0 and a count of 0; callers must leave those samples uncorrected.
    ``counts`` is the per-anchor total weight when the caller already has it.
    """
    vals = np.asarray(values, dtype=np.float64)
    if vals.ndim == 1:
        vals = vals[:, None]
    w = np.asarray(weights, dtype=np.float64)
    if counts is None:
        counts = np.bincount(anchors, weights=w, minlength=n_anchors)
    sums = np.stack(
        [np.bincount(anchors, weights=w * vals[:, k], minlength=n_anchors)
         for k in range(vals.shape[1])],
        axis=1,
    )
    safe = np.maximum(counts, 1e-12)[:, None]
    return sums / safe, counts
