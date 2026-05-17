"""Vectorized 3-D simplex noise (arbitrary point queries).

Gradient table and skewing/unskewing constants from Gustavson 2012.
"""

import numpy as np

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

    x1 = x0 - i1 + _G3;  y1 = y0 - j1 + _G3;  z1 = z0 - k1 + _G3
    x2 = x0 - i2 + 2.0 * _G3;  y2 = y0 - j2 + 2.0 * _G3;  z2 = z0 - k2 + 2.0 * _G3
    x3 = x0 - 1.0 + 3.0 * _G3;  y3 = y0 - 1.0 + 3.0 * _G3;  z3 = z0 - 1.0 + 3.0 * _G3

    ii = i & 255;  jj = j & 255;  kk = k & 255

    gi0 = perm[ii      + perm[jj      + perm[kk     ]]] % 12
    gi1 = perm[ii + i1 + perm[jj + j1 + perm[kk + k1]]] % 12
    gi2 = perm[ii + i2 + perm[jj + j2 + perm[kk + k2]]] % 12
    gi3 = perm[ii + 1  + perm[jj + 1  + perm[kk + 1 ]]] % 12

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
