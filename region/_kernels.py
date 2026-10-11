"""numba kernels for :mod:`region.sampling` (imported only where numba is installed).

Each repeats its numpy counterpart's arithmetic operation for operation, so
results are bit-identical with or without numba.
"""
from __future__ import annotations

import numpy as np
from llvmlite import ir
from numba import njit, prange, types
from numba.core import cgutils
from numba.extending import intrinsic


@njit(cache=True)
def counting_order(anchors, indptr):  # pragma: no cover - compiled
    """Stable argsort of ``anchors`` given each anchor's slot range ``indptr``."""
    slot = indptr[:-1].copy()
    order = np.empty(anchors.shape[0], dtype=np.int64)
    for p in range(anchors.shape[0]):
        order[slot[anchors[p]]] = p
        slot[anchors[p]] += 1
    return order


@njit(cache=True, parallel=True)
def planned_anchor_sums(vals, w, order, indptr):  # pragma: no cover - compiled
    """:func:`region.sampling._weighted_anchor_sums` plus each anchor's weight (as ``np.bincount``),
    anchors in parallel, each summed in index order."""
    n = indptr.shape[0] - 1
    c_n = vals.shape[1]
    sums = np.zeros((n, c_n))
    counts = np.zeros(n)
    for a in prange(n):
        for j in range(indptr[a], indptr[a + 1]):
            p = order[j]
            counts[a] += w[p]
            for k in range(c_n):
                sums[a, k] += w[p] * vals[p, k]
    return sums, counts


@njit(cache=True, inline="always")
def _blend(t00, t01, t10, t11, gx, gy):  # pragma: no cover - compiled
    """:func:`region.sampling.bilinear_read`'s blend, operation for operation."""
    top = t00 * (1.0 - gx) + t01 * gx
    bot = t10 * (1.0 - gx) + t11 * gx
    return top * (1.0 - gy) + bot * gy


@njit(cache=True, parallel=True)
def bilinear_read(flat, corners, fx, fy):  # pragma: no cover - compiled
    """:func:`region.sampling.bilinear_read` fused over its four gathers."""
    p_n, c_n = corners.shape[1], flat.shape[1]
    out = np.empty((p_n, c_n))
    for p in prange(p_n):
        a, b, c, d = corners[0, p], corners[1, p], corners[2, p], corners[3, p]
        for k in range(c_n):
            out[p, k] = _blend(np.float64(flat[a, k]), np.float64(flat[b, k]),
                               np.float64(flat[c, k]), np.float64(flat[d, k]), fx[p, 0], fy[p, 0])
    return out


@njit(cache=True, inline="always")
def _half(h):  # pragma: no cover - compiled
    """The float16 with bits ``h`` (uint16), exactly, as float64."""
    e = (h >> 10) & 0x1F
    m = np.float64(h & 0x3FF)
    if e == 0:
        v = m * 2.0 ** -24                    # subnormal (or zero)
    elif e == 31:
        v = np.inf if m == 0.0 else np.nan
    else:
        v = (1024.0 + m) * 2.0 ** (np.int64(e) - 25)
    return -v if h & 0x8000 else v


@njit(cache=True, parallel=True)
def bilinear_read_f16(bits, corners, fx, fy):  # pragma: no cover - compiled
    """:func:`_bilinear_read` of a float16 texture viewed as uint16."""
    p_n, c_n = corners.shape[1], bits.shape[1]
    out = np.empty((p_n, c_n))
    for p in prange(p_n):
        a, b, c, d = corners[0, p], corners[1, p], corners[2, p], corners[3, p]
        for k in range(c_n):
            out[p, k] = _blend(_half(bits[a, k]), _half(bits[b, k]), _half(bits[c, k]), _half(bits[d, k]),
                               fx[p, 0], fy[p, 0])
    return out


@intrinsic
def _fma(typingctx, a, b, c):  # pragma: no cover - compiled
    """``a * b + c`` rounded once (``llvm.fma``), as numpy's einsum inner loop."""
    def codegen(context, builder, signature, args):
        d = ir.DoubleType()
        fn = cgutils.get_or_insert_function(builder.module, ir.FunctionType(d, [d, d, d]), "llvm.fma.f64")
        return builder.call(fn, args)
    return types.float64(types.float64, types.float64, types.float64), codegen


@njit(cache=True, parallel=True)
def stencil_read(w, idx, vals):  # pragma: no cover - compiled
    """:func:`region.sampling.stencil_read` for ``C > 1``: ``k`` in order, fused multiply-adds."""
    p_n, k_n = idx.shape
    c_n = vals.shape[1]
    out = np.empty((p_n, c_n))
    for p in prange(p_n):
        for c in range(c_n):
            s = 0.0
            for k in range(k_n):
                s = _fma(w[p, k], vals[idx[p, k], c], s)  # pyright: ignore[reportCallIssue]
            out[p, c] = s
    return out
