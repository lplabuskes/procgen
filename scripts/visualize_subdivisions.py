#!/usr/bin/env python3
"""
Toy visualization of sphere subdivision schemes for regional maps.

  Cubed sphere  N=2  →  24 regions
  HEALPix      Nside=2  →  48 regions  (cell assignment via nearest pixel center)

Usage:
    python3 scripts/visualize_subdivisions.py
Output:
    scripts/results/sphere_subdivisions.png
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

RES = 0.25  # grid resolution in degrees


# ─────────────────────────────────────────────────────────────────
# Coordinate helpers
# ─────────────────────────────────────────────────────────────────

def lonlat_to_xyz(lon_deg, lat_deg):
    lon = np.radians(lon_deg)
    lat = np.radians(lat_deg)
    return (np.cos(lat) * np.cos(lon),
            np.cos(lat) * np.sin(lon),
            np.sin(lat))


# ─────────────────────────────────────────────────────────────────
# Cubed-sphere  (gnomonic projection, 6·N² cells)
# ─────────────────────────────────────────────────────────────────

def _cs_face_and_uv(x, y, z):
    """Return (face, u, v) with gnomonic u,v ∈ [-1, 1] for each face."""
    ax, ay, az = np.abs(x), np.abs(y), np.abs(z)
    dom_x = (ax >= ay) & (ax >= az)
    dom_y = (~dom_x) & (ay >= az)
    dom_z = ~dom_x & ~dom_y

    face = np.empty_like(x, dtype=int)
    face[dom_x] = np.where(x[dom_x] > 0, 0, 1)
    face[dom_y] = np.where(y[dom_y] > 0, 2, 3)
    face[dom_z] = np.where(z[dom_z] > 0, 4, 5)

    m = np.maximum(ax, np.maximum(ay, az))
    u = np.where(dom_x, y / m, np.where(dom_y, x / m, x / m))
    v = np.where(dom_x, z / m, np.where(dom_y, z / m, y / m))
    return face, u, v


def cubed_sphere_cell(x, y, z, n=2):
    """Gnomonic cubed-sphere cells.  Returns int array [0, 6n²)."""
    face, u, v = _cs_face_and_uv(x, y, z)
    ci = np.clip((u + 1) * n / 2, 0, n - 1e-9).astype(int)
    cj = np.clip((v + 1) * n / 2, 0, n - 1e-9).astype(int)
    return face * n * n + ci * n + cj


def _ea_stretch(t):
    """
    Map gnomonic coordinate t ∈ [-1,1] to equal-area parameter p ∈ [0,1].

    Derived from the CDF of the marginal solid-angle weight on a cube face:
        w(t) ∝ 1 / ((1+t²)·√(2+t²))
    whose antiderivative is arcsin(t / √(2(1+t²))), giving:
        p = (3/π) · (arcsin(t / √(2(1+t²))) + π/6)
    This is exact for the 1-D marginal; the 2-D separable application leaves
    a small residual variation (~few %) near face corners.
    """
    return (3 / np.pi) * (np.arcsin(t / np.sqrt(2 * (1 + t**2))) + np.pi / 6)


def cubed_sphere_cell_ea(x, y, z, n=3):
    """Equal-area cubed-sphere cells (separable CDF stretch).  Returns int array [0, 6n²)."""
    face, u, v = _cs_face_and_uv(x, y, z)
    pu = _ea_stretch(u)   # ∈ [0, 1], uniform area weight along u
    pv = _ea_stretch(v)
    ci = np.clip(pu * n, 0, n - 1e-9).astype(int)
    cj = np.clip(pv * n, 0, n - 1e-9).astype(int)
    return face * n * n + ci * n + cj


# ─────────────────────────────────────────────────────────────────
# HEALPix  (nearest-center Voronoi, ring scheme)
# ─────────────────────────────────────────────────────────────────

def _healpix_centers(nside):
    """
    Return (npix, 3) unit-sphere xyz for all 12·Nside² pixel centers,
    using the HEALPix ring-scheme pix2ang formula.
    """
    npix = 12 * nside * nside
    ncap = 2 * nside * (nside - 1)   # pixels in north polar cap
    p = np.arange(npix)

    theta = np.empty(npix)
    phi = np.empty(npix)

    # North polar cap  (rings 1 … Nside-1, 4·ring pixels each)
    mn = p < ncap
    if mn.any():
        ip = p[mn]
        ir = ((1 + np.sqrt(1 + 2 * ip)) / 2).astype(int)   # 1-indexed ring
        ik = ip - 2 * ir * (ir - 1)                         # pixel within ring
        theta[mn] = np.arccos(1 - ir**2 / (3 * nside**2))
        phi[mn] = (ik + 0.5) * np.pi / (2 * ir)

    # Equatorial belt  (rings Nside … 3·Nside, 4·Nside pixels each)
    me = ~mn & (p < npix - ncap)
    if me.any():
        ip = p[me] - ncap
        ir = (ip // (4 * nside) + nside).astype(int)       # ring number
        ik = ip % (4 * nside)
        # kshift=1 (unshifted) when (ir+nside) is odd, 0 (half-shift) when even
        ks = ((ir + nside) % 2).astype(float)
        theta[me] = np.arccos((2 * nside - ir) * 2 / (3 * nside))
        phi[me] = (ik + 0.5 * (1 - ks)) * np.pi / (2 * nside)

    # South polar cap  (mirror of north)
    ms = p >= npix - ncap
    if ms.any():
        ip = npix - 1 - p[ms]
        ir = ((1 + np.sqrt(1 + 2 * ip)) / 2).astype(int)
        ik = ip - 2 * ir * (ir - 1)
        theta[ms] = np.arccos(-(1 - ir**2 / (3 * nside**2)))
        phi[ms] = (ik + 0.5) * np.pi / (2 * ir)

    cx = np.sin(theta) * np.cos(phi)
    cy = np.sin(theta) * np.sin(phi)
    cz = np.cos(theta)
    return np.stack([cx, cy, cz], axis=1)   # (npix, 3)


def healpix_cell(x, y, z, nside=2):
    """Assign points to the nearest HEALPix pixel center (Voronoi on sphere)."""
    centers = _healpix_centers(nside)                                # (npix, 3)
    pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)   # (N, 3)
    return np.argmax(pts @ centers.T, axis=1).reshape(x.shape)


# ─────────────────────────────────────────────────────────────────
# Rendering
# ─────────────────────────────────────────────────────────────────

def _make_palette(n, seed=42):
    rng = np.random.default_rng(seed)
    hues = np.linspace(0, 1, n, endpoint=False)
    sats = np.where(np.arange(n) % 2 == 0, 0.80, 0.50)
    vals = np.where(np.arange(n) % 2 == 0, 0.88, 0.68)
    palette = hsv_to_rgb(np.stack([hues, sats, vals], axis=1))
    return palette[rng.permutation(n)]


def _edge_mask(cells):
    """True where a pixel borders a cell with a different index."""
    d_up = np.pad(cells[:-1] != cells[1:], ((1, 0), (0, 0)))
    d_down = np.pad(cells[1:] != cells[:-1], ((0, 1), (0, 0)))
    d_left = np.pad(cells[:, :-1] != cells[:, 1:], ((0, 0), (1, 0)))
    d_right = np.pad(cells[:, 1:] != cells[:, :-1], ((0, 0), (0, 1)))
    return d_up | d_down | d_left | d_right


def _area_stats(cells, n_cells, lat_grid):
    """Fraction of sphere area per cell (weighted by cos(lat))."""
    weights = np.cos(np.radians(lat_grid))
    total = weights.sum()
    fracs = np.array([weights[cells == i].sum() / total for i in range(n_cells)])
    ideal = 1.0 / n_cells
    return fracs.min() / ideal, fracs.max() / ideal   # min/max relative to ideal


def render(ax, cells, n_cells, title, lat_grid):
    palette = _make_palette(n_cells)
    rgb = palette[cells].copy()
    rgb[_edge_mask(cells)] = 0.0   # black borders

    ax.imshow(rgb, extent=[-180, 180, -90, 90], origin='lower',
              aspect='auto', interpolation='nearest')
    ax.set_title(title, fontsize=12, pad=8)
    ax.set_xlabel('Longitude °', fontsize=9)
    ax.set_ylabel('Latitude °',  fontsize=9)
    ax.set_xticks(range(-180, 181, 60))
    ax.set_yticks(range(-90,   91, 30))
    ax.tick_params(labelsize=8)
    ax.grid(color='white', linewidth=0.3, alpha=0.35, linestyle=':')

    lo, hi = _area_stats(cells, n_cells, lat_grid)
    ax.text(0.02, 0.04,
            f'area ratio  min={lo:.2f}×  max={hi:.2f}×  (vs ideal)',
            transform=ax.transAxes, fontsize=8, color='white',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.55))


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    lon = np.arange(-179.875, 180, RES)
    lat = np.arange(-89.875,   90, RES)
    lon_g, lat_g = np.meshgrid(lon, lat)
    x_g, y_g, z_g = lonlat_to_xyz(lon_g, lat_g)

    n_cube = 3    # 6·N² = 24 regions
    nside = 2    # 12·Nside² = 48 regions

    print('Computing cubed-sphere cells …')
    cs = cubed_sphere_cell_ea(x_g, y_g, z_g, n=n_cube)

    print('Computing HEALPix cells …')
    hp = healpix_cell(x_g, y_g, z_g, nside=nside)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5))
    fig.suptitle('Sphere Subdivision Schemes — Regional Map Candidates',
                 fontsize=13, y=1.01)

    render(ax1, cs, 6 * n_cube**2,
           f'Cubed Sphere  N={n_cube}  →  {6 * n_cube**2} regions',
           lat_g)
    render(ax2, hp, 12 * nside**2,
           f'HEALPix  Nside={nside}  →  {12 * nside**2} regions  (nearest-center)',
           lat_g)

    plt.tight_layout()
    # out = 'scripts/results/sphere_subdivisions.png'
    # plt.savefig(out, dpi=150, bbox_inches='tight')
    # print(f'Saved → {out}')
    plt.show()


if __name__ == '__main__':
    main()
