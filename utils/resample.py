"""Resampling textures between tiers: bilinear reads and stencils, and per-anchor means.

An **anchor** is the coarse texel a fine sample belongs to; conservation keeps each
anchor's weighted mean of its fine samples equal to the anchor's own value.
Textures are ``(N, H, W, C)`` tiles; texels are addressed by flat ``n·H·W + row·W + col``.
The numba kernels (:mod:`utils._resample_kernels`) are bit-identical to the numpy paths.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from utils.jit import njit

try:
    from utils import _resample_kernels as _kernels
except ImportError:  # pragma: no cover - numba is an optional accelerator
    _kernels = None  # type: ignore[assignment]


def bilinear_plan(
    items: np.ndarray, x: np.ndarray, y: np.ndarray, h: int, w: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(corners (4, P), fx (P, 1), fy (P, 1))`` for bilinear reads of ``(N, H, W, C)`` textures.

    ``corners`` are flat ``(N·H·W)`` indices at texel coords ``(x, y)`` of ``items``."""
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
    if _kernels is not None:
        if flat.dtype == np.float16:            # numba has no float16: decode its bits
            return _kernels.bilinear_read_f16(flat.view(np.uint16), corners, fx, fy)
        return _kernels.bilinear_read(flat, corners, fx, fy)
    t00, t01, t10, t11 = (flat[k].astype(np.float64) for k in corners)
    top = t00 * (1.0 - fx) + t01 * fx
    bot = t10 * (1.0 - fx) + t11 * fx
    return top * (1.0 - fy) + bot * fy


def bilinear_stencil(
    tiles: np.ndarray, x: np.ndarray, y: np.ndarray, h: int, w: int, owner: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Corners and weights of a bilinear read at texel coords ``(x, y)`` of ``(N, h, w)`` textures.

    Corners are mapped through ``owner`` (flat texel → anchor)."""
    x0 = np.clip(np.floor(x).astype(np.int64), 0, max(w - 2, 0))
    y0 = np.clip(np.floor(y).astype(np.int64), 0, max(h - 2, 0))
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx, fy = x - x0, y - y0
    base = tiles * (h * w)
    idx = np.stack([base + y0 * w + x0, base + y0 * w + x1, base + y1 * w + x0, base + y1 * w + x1], axis=1)
    weights = np.stack([(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy], axis=1)
    return owner[idx], weights


def anchor_values(texture: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """``(P, C)`` value of each anchor texel in an ``(N, H, W, C)`` texture."""
    tex = np.asarray(texture)
    return tex.reshape(-1, tex.shape[3])[anchors].astype(np.float64)


def settle_owners(owner: np.ndarray) -> np.ndarray:
    """Make a texel → owning-texel map idempotent (every owner owns itself).

    Ownership can form chains and (duplicated seam texels) cycles; chains are
    followed to their cycle, and each cycle goes to its smallest member.
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


def anchor_plan(anchors: np.ndarray, n_anchors: int) -> tuple[np.ndarray, np.ndarray]:
    """``(order, indptr)``: each anchor's texels, in index order, for repeated :func:`anchor_means`.

    ``order`` is ``np.argsort(anchors, kind="stable")``, by counting sort.
    """
    anchors = np.asarray(anchors, dtype=np.int64)
    indptr = np.zeros(int(n_anchors) + 1, dtype=np.int64)
    np.cumsum(np.bincount(anchors, minlength=n_anchors), out=indptr[1:])
    if _kernels is None:
        return np.argsort(anchors, kind="stable"), indptr
    return _kernels.counting_order(anchors, indptr), indptr


def anchor_means(
    values: np.ndarray, anchors: np.ndarray, weights: np.ndarray, n_anchors: int,
    counts: Optional[np.ndarray] = None, plan: Optional[tuple[np.ndarray, np.ndarray]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-anchor weighted mean of ``(P, C)`` values → ``((n_anchors, C), counts)``.

    ``weights`` is the ``(P,)`` inclusion weight (0 for over-hang).  Anchors with
    no weight get mean and count 0; callers must leave those uncorrected.
    ``counts`` may be passed if already known; ``plan`` (:func:`anchor_plan` of
    ``anchors``) sums the anchors in parallel, each in the same order.
    """
    vals = np.asarray(values, dtype=np.float64)
    if vals.ndim == 1:
        vals = vals[:, None]
    w = np.asarray(weights, dtype=np.float64)
    if plan is not None and _kernels is not None:
        sums, plan_counts = _kernels.planned_anchor_sums(vals, w, plan[0], plan[1])
        counts = plan_counts if counts is None else counts
        safe = np.maximum(counts, 1e-12)[:, None]
        return sums / safe, counts
    if counts is None:
        counts = np.bincount(anchors, weights=w, minlength=n_anchors)
    sums = _weighted_anchor_sums(vals, np.asarray(anchors, dtype=np.int64), w, int(n_anchors))
    safe = np.maximum(counts, 1e-12)[:, None]
    return sums / safe, counts


@njit(cache=True)
def _weighted_anchor_sums(vals, anchors, w, n_anchors):  # pragma: no cover - compiled
    """``(n_anchors, C)`` per-anchor sums of ``w × vals``; same add order as ``np.bincount``."""
    sums = np.zeros((n_anchors, vals.shape[1]))
    for p in range(vals.shape[0]):
        a = anchors[p]
        for k in range(vals.shape[1]):
            sums[a, k] += w[p] * vals[p, k]
    return sums


def stencil_read(weights: np.ndarray, idx: np.ndarray, values: np.ndarray) -> np.ndarray:
    """``(P, C)`` ``Σ_k weights[p, k] · values[idx[p, k]]``, bit-identical to
    ``np.einsum("pk,pkc->pc", weights, values[idx])`` without the ``(P, K, C)`` gather.

    For ``C > 1`` einsum accumulates ``k`` in order with fused multiply-adds, which
    :func:`utils._resample_kernels.stencil_read` repeats; its one-channel loop is SIMD-laned, so stays on einsum.
    """
    vals = np.asarray(values, dtype=np.float64)
    if _kernels is None or vals.shape[1] == 1:
        return np.einsum("pk,pkc->pc", weights, vals[idx])
    return _kernels.stencil_read(np.ascontiguousarray(weights, dtype=np.float64),
                                 np.ascontiguousarray(idx, dtype=np.int64), np.ascontiguousarray(vals))


def anchor_residual(
    values: np.ndarray, anchors: np.ndarray, weights: np.ndarray, parent: np.ndarray,
) -> dict[str, float]:
    """How far each anchor's weighted mean of ``(P, C)`` ``values`` is from its texel of
    the ``(N, H, W, C)`` ``parent``: ``{"max", "mean", "n_anchors"}`` over anchors with weight."""
    parent_tex = np.asarray(parent, dtype=np.float64)
    n_anchors = parent_tex.shape[0] * parent_tex.shape[1] * parent_tex.shape[2]
    means, counts = anchor_means(values, anchors, weights, n_anchors)
    hit = counts > 0.0
    residual = np.abs(means[hit] - parent_tex.reshape(n_anchors, -1)[hit])
    return {
        "max": float(residual.max()) if residual.size else 0.0,
        "mean": float(residual.mean()) if residual.size else 0.0,
        "n_anchors": int(hit.sum()),
    }
