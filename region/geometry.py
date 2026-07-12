"""Equal-area cubed-sphere region geometry.

The sphere is wrapped in a cube; each of the 6 cube faces is a gnomonic patch
carrying gnomonic coordinates ``(u, v) in [-1, 1]**2``.  A separable equal-area
CDF stretch maps gnomonic ``u`` to ``p in [0, 1]`` such that uniform steps in
``p`` carve out equal solid angle (exactly for the 1-D marginal; a few-percent
residual near corners).  Each face is cut into an ``N_SUBDIV x N_SUBDIV`` grid
of cells in ``(pu, pv)`` space, giving ``6 * N_SUBDIV**2`` regions total.

A whole-sphere rotation about Z aligns a cube-face boundary with the viewer's
antimeridian; every public entry point takes that ``antimeridian`` so region
assignment is stable regardless of how the mesh is oriented.

Region ids decompose as ``face * N_SUBDIV**2 + ci * N_SUBDIV + cj`` where
``face in [0, 6)`` and ``ci, cj in [0, N_SUBDIV)``.
"""
from __future__ import annotations

import numpy as np

N_SUBDIV: int = 3
N_REGIONS: int = 6 * N_SUBDIV * N_SUBDIV


def _rotation(antimeridian: float) -> float:
    """Z-rotation (radians) placing a cube-face boundary at ``antimeridian``.

    Cube edges between the 4 equatorial faces sit at ``45 deg + k*90 deg`` in the
    rotated frame, so rotating by ``antimeridian - pi/4`` lands one edge there.
    """
    return antimeridian - np.pi / 4.0


def ea_forward(u: np.ndarray) -> np.ndarray:
    """Equal-area CDF: gnomonic ``u in [-1, 1]`` -> parameter ``p in [0, 1]``."""
    return (3.0 / np.pi) * (np.arcsin(u / np.sqrt(2.0 * (1.0 + u ** 2))) + np.pi / 6.0)


def ea_inverse(p: np.ndarray) -> np.ndarray:
    """Inverse equal-area CDF: parameter ``p in [0, 1]`` -> gnomonic ``u in [-1, 1]``."""
    theta = p * (np.pi / 3.0) - np.pi / 6.0
    s = np.sin(theta)
    c2 = np.maximum(np.cos(2.0 * theta), 1e-10)
    return np.sqrt(2.0) * s / np.sqrt(c2)


def _rotate_into_cube_frame(
    positions: np.ndarray, antimeridian: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotate ECEF ``(N, 3)`` about Z into the cube frame; return ``(xr, yr, zr)``."""
    rot = _rotation(antimeridian)
    cos_r, sin_r = np.cos(rot), np.sin(rot)
    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
    xr = x * cos_r + y * sin_r
    yr = -x * sin_r + y * cos_r
    return xr, yr, z


def _cube_face_gnomonic(
    xr: np.ndarray, yr: np.ndarray, zr: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cube-frame direction -> ``(face, u_g, v_g)`` with gnomonic ``u_g, v_g in [-1, 1]``.

    Face axes:
      0/1 (±X): u = yr/m, v = zr/m     2/3 (±Y): u = xr/m, v = zr/m
      4/5 (±Z): u = xr/m, v = yr/m
    """
    ax, ay, az = np.abs(xr), np.abs(yr), np.abs(zr)
    dom_x = (ax >= ay) & (ax >= az)
    dom_y = (~dom_x) & (ay >= az)

    face = np.empty(xr.shape, dtype=np.int32)
    face[dom_x] = np.where(xr[dom_x] > 0, 0, 1)
    face[dom_y] = np.where(yr[dom_y] > 0, 2, 3)
    dom_z = ~dom_x & ~dom_y
    face[dom_z] = np.where(zr[dom_z] > 0, 4, 5)

    m = np.maximum(ax, np.maximum(ay, az))
    u_g = np.where(dom_x, yr / m, np.where(dom_y, xr / m, xr / m))
    v_g = np.where(dom_x, zr / m, np.where(dom_y, zr / m, yr / m))
    return face, u_g, v_g


def _face_gnomonic_to_ecef(
    face: int, u_g: np.ndarray, v_g: np.ndarray, antimeridian: float
) -> np.ndarray:
    """Gnomonic face coords -> ECEF unit-sphere ``(N, 3)`` in the original frame.

    Inverse of ``_rotate_into_cube_frame`` composed with ``_cube_face_gnomonic``.
    """
    d = np.sqrt(1.0 + u_g ** 2 + v_g ** 2)
    if face == 0:
        xr, yr, zr = 1.0 / d, u_g / d, v_g / d
    elif face == 1:
        xr, yr, zr = -1.0 / d, u_g / d, v_g / d
    elif face == 2:
        xr, yr, zr = u_g / d, 1.0 / d, v_g / d
    elif face == 3:
        xr, yr, zr = u_g / d, -1.0 / d, v_g / d
    elif face == 4:
        xr, yr, zr = u_g / d, v_g / d, 1.0 / d
    else:
        xr, yr, zr = u_g / d, v_g / d, -1.0 / d

    rot = _rotation(antimeridian)
    cos_r, sin_r = np.cos(rot), np.sin(rot)
    px = xr * cos_r - yr * sin_r
    py = xr * sin_r + yr * cos_r
    pz = zr if np.ndim(zr) else np.full_like(px, zr)
    return np.column_stack([px, py, pz])


def ecef_to_cell(positions: np.ndarray, antimeridian: float) -> np.ndarray:
    """Assign each ECEF unit-sphere position ``(N, 3)`` to a region id ``(N,)``."""
    xr, yr, zr = _rotate_into_cube_frame(np.asarray(positions, dtype=np.float64), antimeridian)
    face, u_g, v_g = _cube_face_gnomonic(xr, yr, zr)
    pu = ea_forward(u_g)
    pv = ea_forward(v_g)
    ci = np.clip(pu * N_SUBDIV, 0.0, N_SUBDIV - 1e-9).astype(np.int32)
    cj = np.clip(pv * N_SUBDIV, 0.0, N_SUBDIV - 1e-9).astype(np.int32)
    return face * (N_SUBDIV * N_SUBDIV) + ci * N_SUBDIV + cj


def cell_border_polylines(antimeridian: float, samples: int = 100) -> list[np.ndarray]:
    """Dense ECEF polylines tracing every cell boundary, for overlay rendering.

    Returns one ``(samples, 3)`` array per gridline (constant-``pu`` and
    constant-``pv`` lines on each of the 6 faces).
    """
    p_grid = np.linspace(0.0, 1.0, N_SUBDIV + 1)
    u_grid = ea_inverse(p_grid)                 # gnomonic values at each cell boundary
    param = np.linspace(-1.0, 1.0, samples)     # free coordinate along a gridline

    polys: list[np.ndarray] = []
    for face in range(6):
        for u0 in u_grid:
            polys.append(_face_gnomonic_to_ecef(face, np.full(samples, u0), param, antimeridian))
        for v0 in u_grid:
            polys.append(_face_gnomonic_to_ecef(face, param, np.full(samples, v0), antimeridian))
    return polys


class RegionGeometry:
    """Geometry of a single region cell: maps local ``[0, 1]**2`` <-> the sphere.

    The local frame is equal-area: uniform ``(s, t)`` steps cover equal solid
    angle (up to the CDF residual).  ``s`` runs along the cell's ``pu`` axis,
    ``t`` along ``pv``.  This local frame is what regional upsampling samples in.
    """

    def __init__(self, region_id: int, antimeridian: float) -> None:
        if not 0 <= region_id < N_REGIONS:
            raise ValueError(f"region_id {region_id} out of range [0, {N_REGIONS})")
        self.region_id = int(region_id)
        self.antimeridian = float(antimeridian)
        self.face = self.region_id // (N_SUBDIV * N_SUBDIV)
        rem = self.region_id % (N_SUBDIV * N_SUBDIV)
        self.ci = rem // N_SUBDIV
        self.cj = rem % N_SUBDIV

    def _local_to_p(self, s: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pu = (self.ci + np.asarray(s, dtype=np.float64)) / N_SUBDIV
        pv = (self.cj + np.asarray(t, dtype=np.float64)) / N_SUBDIV
        return pu, pv

    def local_to_ecef(self, s: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Local coords ``(s, t) in [0, 1]`` -> ECEF unit-sphere ``(N, 3)``."""
        pu, pv = self._local_to_p(s, t)
        u_g = ea_inverse(pu)
        v_g = ea_inverse(pv)
        return _face_gnomonic_to_ecef(self.face, u_g, v_g, self.antimeridian)

    def local_to_lonlat(self, s: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Local coords ``(s, t) in [0, 1]`` -> ``(N, 2)`` ``[lon, lat]`` radians."""
        ecef = self.local_to_ecef(s, t)
        lon = np.arctan2(ecef[:, 1], ecef[:, 0])
        lat = np.arcsin(np.clip(ecef[:, 2], -1.0, 1.0))
        return np.column_stack([lon, lat])

    def sample_grid(self, res_s: int, res_t: int) -> np.ndarray:
        """Cell-centred ``(res_t, res_s, 3)`` ECEF grid of sample positions.

        Samples the centres of a ``res_s x res_t`` equal-area lattice over the
        cell — the sampling points a regional field would be evaluated at.
        """
        s = (np.arange(res_s, dtype=np.float64) + 0.5) / res_s
        t = (np.arange(res_t, dtype=np.float64) + 0.5) / res_t
        ss, tt = np.meshgrid(s, t)  # (res_t, res_s)
        ecef = self.local_to_ecef(ss.ravel(), tt.ravel())
        return ecef.reshape(res_t, res_s, 3)

    def center_ecef(self) -> np.ndarray:
        """ECEF position of the cell centre (local ``(0.5, 0.5)``)."""
        return self.local_to_ecef(np.array([0.5]), np.array([0.5]))[0]
