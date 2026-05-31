"""Vectorized 3-D simplex noise (arbitrary point queries).

Gradient table and skewing/unskewing constants from Gustavson 2012.

`simplex3_vec` is the single-octave vectorized numpy implementation.
`fbm_sum` sums an arbitrary set of octaves and is JIT-compiled with numba when
available (fused scalar kernel, `prange` over points), falling back to a
numpy octave loop otherwise. The two paths are bit-identical.
"""

import numpy as np

try:
    from numba import njit, prange
    _HAVE_NUMBA = True
except ImportError:  # pragma: no cover - numba is an optional accelerator
    _HAVE_NUMBA = False

_GRAD3 = np.array([
    [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
    [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
    [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1],
], dtype=np.float64)
_F3 = 1.0 / 3.0
_G3 = 1.0 / 6.0


def make_perm(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = np.arange(256, dtype=np.int32)
    rng.shuffle(p)
    return np.tile(p, 2)  # shape (512,) — supports index + 0..255 without wrap


def simplex3_vec(xyz: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Single-octave vectorized 3-D simplex noise at arbitrary points.

    Args:
        xyz: (N, 3) float64 coordinates.
        perm: (512,) int32 doubled permutation table from make_perm.

    Returns:
        (N,) float64 noise values in roughly [-1, 1].
    """
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    s = (x + y + z) * _F3
    i = np.floor(x + s).astype(np.int32)
    j = np.floor(y + s).astype(np.int32)
    k = np.floor(z + s).astype(np.int32)

    t = (i + j + k) * _G3
    x0 = x - (i - t)
    y0 = y - (j - t)
    z0 = z - (k - t)

    # Determine which simplex tetrahedron each point is in by comparing
    # offset coordinates. The six orderings of (x0, y0, z0) map to the six
    # possible traversals of the unit simplex.
    x_ge_y = x0 >= y0
    y_ge_z = y0 >= z0
    x_ge_z = x0 >= z0

    i1 = (x_ge_y & x_ge_z).astype(np.int32)        # x is largest
    j1 = (~x_ge_y & y_ge_z).astype(np.int32)       # y is largest
    k1 = (~x_ge_z & ~y_ge_z).astype(np.int32)      # z is largest

    i2 = (x_ge_y | x_ge_z).astype(np.int32)        # x is not smallest
    j2 = (~x_ge_y | y_ge_z).astype(np.int32)       # y is not smallest
    k2 = (~(x_ge_z & y_ge_z)).astype(np.int32)     # z is not smallest

    x1 = x0 - i1 + _G3
    y1 = y0 - j1 + _G3
    z1 = z0 - k1 + _G3
    x2 = x0 - i2 + 2.0 * _G3
    y2 = y0 - j2 + 2.0 * _G3
    z2 = z0 - k2 + 2.0 * _G3
    x3 = x0 - 1.0 + 3.0 * _G3
    y3 = y0 - 1.0 + 3.0 * _G3
    z3 = z0 - 1.0 + 3.0 * _G3

    ii = i & 255
    jj = j & 255
    kk = k & 255

    gi0 = perm[ii + perm[jj + perm[kk]]] % 12
    gi1 = perm[ii + i1 + perm[jj + j1 + perm[kk + k1]]] % 12
    gi2 = perm[ii + i2 + perm[jj + j2 + perm[kk + k2]]] % 12
    gi3 = perm[ii + 1 + perm[jj + 1 + perm[kk + 1]]] % 12

    def _contrib(gi: np.ndarray, dx: np.ndarray, dy: np.ndarray, dz: np.ndarray) -> np.ndarray:
        t_ = 0.6 - dx * dx - dy * dy - dz * dz
        mask = t_ > 0
        t2 = np.where(mask, t_ * t_, 0.0)
        g = _GRAD3[gi]
        dot = g[:, 0] * dx + g[:, 1] * dy + g[:, 2] * dz
        return np.where(mask, t2 * t2 * dot, 0.0)

    return 32.0 * (
        _contrib(gi0, x0, y0, z0)
        + _contrib(gi1, x1, y1, z1)
        + _contrib(gi2, x2, y2, z2)
        + _contrib(gi3, x3, y3, z3)
    )


def _fbm_sum_py(xyz: np.ndarray, perm: np.ndarray, freqs: np.ndarray, amps: np.ndarray) -> np.ndarray:
    """Pure-numpy fallback: sum amps[o] * simplex3_vec(xyz * freqs[o]) over octaves."""
    out = np.zeros(len(xyz), dtype=np.float64)
    for o in range(len(freqs)):
        out += amps[o] * simplex3_vec(xyz * freqs[o], perm)
    return out


if _HAVE_NUMBA:
    @njit(cache=True)
    def _simplex3_point(x, y, z, perm, grad3):  # pragma: no cover - njit
        """Single-octave 3-D simplex noise at one (already-scaled) point.

        Scalar transcription of ``simplex3_vec`` with identical operation order
        so the fused fBm sum is bit-identical to the numpy octave loop.
        """
        s = (x + y + z) * _F3
        i = int(np.floor(x + s))
        j = int(np.floor(y + s))
        k = int(np.floor(z + s))

        t = (i + j + k) * _G3
        x0 = x - (i - t)
        y0 = y - (j - t)
        z0 = z - (k - t)

        x_ge_y = x0 >= y0
        y_ge_z = y0 >= z0
        x_ge_z = x0 >= z0

        i1 = 1 if (x_ge_y and x_ge_z) else 0
        j1 = 1 if ((not x_ge_y) and y_ge_z) else 0
        k1 = 1 if ((not x_ge_z) and (not y_ge_z)) else 0

        i2 = 1 if (x_ge_y or x_ge_z) else 0
        j2 = 1 if ((not x_ge_y) or y_ge_z) else 0
        k2 = 1 if (not (x_ge_z and y_ge_z)) else 0

        x1 = x0 - i1 + _G3
        y1 = y0 - j1 + _G3
        z1 = z0 - k1 + _G3
        x2 = x0 - i2 + 2.0 * _G3
        y2 = y0 - j2 + 2.0 * _G3
        z2 = z0 - k2 + 2.0 * _G3
        x3 = x0 - 1.0 + 3.0 * _G3
        y3 = y0 - 1.0 + 3.0 * _G3
        z3 = z0 - 1.0 + 3.0 * _G3

        ii = i & 255
        jj = j & 255
        kk = k & 255

        gi0 = perm[ii + perm[jj + perm[kk]]] % 12
        gi1 = perm[ii + i1 + perm[jj + j1 + perm[kk + k1]]] % 12
        gi2 = perm[ii + i2 + perm[jj + j2 + perm[kk + k2]]] % 12
        gi3 = perm[ii + 1 + perm[jj + 1 + perm[kk + 1]]] % 12

        n = 0.0
        t0 = 0.6 - x0 * x0 - y0 * y0 - z0 * z0
        if t0 > 0:
            t0_2 = t0 * t0
            dot0 = grad3[gi0, 0] * x0 + grad3[gi0, 1] * y0 + grad3[gi0, 2] * z0
            n += t0_2 * t0_2 * dot0
        t1 = 0.6 - x1 * x1 - y1 * y1 - z1 * z1
        if t1 > 0:
            t1_2 = t1 * t1
            dot1 = grad3[gi1, 0] * x1 + grad3[gi1, 1] * y1 + grad3[gi1, 2] * z1
            n += t1_2 * t1_2 * dot1
        t2 = 0.6 - x2 * x2 - y2 * y2 - z2 * z2
        if t2 > 0:
            t2_2 = t2 * t2
            dot2 = grad3[gi2, 0] * x2 + grad3[gi2, 1] * y2 + grad3[gi2, 2] * z2
            n += t2_2 * t2_2 * dot2
        t3 = 0.6 - x3 * x3 - y3 * y3 - z3 * z3
        if t3 > 0:
            t3_2 = t3 * t3
            dot3 = grad3[gi3, 0] * x3 + grad3[gi3, 1] * y3 + grad3[gi3, 2] * z3
            n += t3_2 * t3_2 * dot3

        return 32.0 * n

    @njit(cache=True, parallel=True)
    def _fbm_sum_njit(xyz, perm, grad3, freqs, amps):  # pragma: no cover - njit
        n_pts = xyz.shape[0]
        n_oct = freqs.shape[0]
        out = np.zeros(n_pts, dtype=np.float64)
        for p in prange(n_pts):
            px = xyz[p, 0]
            py = xyz[p, 1]
            pz = xyz[p, 2]
            acc = 0.0
            for o in range(n_oct):
                f = freqs[o]
                acc += amps[o] * _simplex3_point(px * f, py * f, pz * f, perm, grad3)
            out[p] = acc
        return out


def fbm_sum(xyz: np.ndarray, perm: np.ndarray, freqs: np.ndarray, amps: np.ndarray) -> np.ndarray:
    """Sum ``amps[o] * simplex3_vec(xyz * freqs[o])`` over octaves.

    Returns the un-normalized fBm sum as ``(N,) float64`` — callers handle
    amplitude normalization and reshaping. ``freqs`` and ``amps`` are 1-D
    arrays, one entry per octave (already filtered for any band-limiting).

    Bit-identical between the numba kernel and the numpy fallback.
    """
    freqs = np.ascontiguousarray(freqs, dtype=np.float64)
    amps = np.ascontiguousarray(amps, dtype=np.float64)
    if len(freqs) == 0:
        return np.zeros(len(xyz), dtype=np.float64)
    if _HAVE_NUMBA:
        return _fbm_sum_njit(
            np.ascontiguousarray(xyz, dtype=np.float64),
            np.ascontiguousarray(perm, dtype=np.int64),
            _GRAD3,
            freqs,
            amps,
        )
    return _fbm_sum_py(xyz, perm, freqs, amps)
