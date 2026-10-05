"""Hexagonal scaffolding inside a region cell.

The local ``[0, 1]**2`` frame (:mod:`region.geometry`) is tiled with flat-top
hexagons, each carrying an over-hanging square texture like a mesh face.  Each
hexagon is also the footprint of a local map.

Texture layout (*not* the mesh's endpoint convention):

- Texels are **cell-centred**, so a plain texel mean is an unbiased area mean
  (conservation depends on that).
- Row 0 is the **top** (largest ``t``), for blitting without a flip.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from region.geometry import RegionGeometry

# Hex texture half-width in circumradii (1.0: smallest square containing the hexagon).
_TEXTURE_HALF_WIDTH = 1.0


class RegionHexGrid:
    """Flat-top hexagons tiling one region cell's local ``[0, 1]**2`` frame.

    ``n_cols`` hexagon columns span the local s-axis; rows follow from the aspect ratio.
    """

    def __init__(self, geometry: RegionGeometry, n_cols: int = 14) -> None:
        if n_cols < 1:
            raise ValueError(f"n_cols must be >= 1, got {n_cols}")
        self.geometry = geometry
        self.n_cols = int(n_cols)
        self.radius = 1.0 / (1.5 * self.n_cols + 0.5)
        self.centers = self._layout()
        self._spacing: dict[int, float] = {}

    def __len__(self) -> int:
        return len(self.centers)

    def _layout(self) -> np.ndarray:
        """(M, 2) hexagon centres in local coords, column-major, s then t.

        Hexagons overlapping the cell are kept, so the whole cell is covered;
        neighbouring regions overlap slightly, which is fine as texels anchor globally.
        """
        R = self.radius
        hex_h = math.sqrt(3.0) * R      # full height of a flat-top hex
        dx = 1.5 * R                    # column centre-to-centre spacing
        s_min, s_max = -R, 1.0 + R
        t_min, t_max = -hex_h * 0.5, 1.0 + hex_h * 0.5
        centers: list[tuple[float, float]] = []
        first_col = int(math.floor(s_min / dx))
        last_col = int(math.ceil(s_max / dx))
        for col in range(first_col, last_col + 1):
            s = col * dx
            if not (s_min - 1e-9 <= s <= s_max + 1e-9):
                continue
            t_off = (hex_h * 0.5) if (col % 2) else 0.0
            first_row = int(math.floor((t_min - t_off) / hex_h))
            last_row = int(math.ceil((t_max - t_off) / hex_h))
            for row in range(first_row, last_row + 1):
                t = row * hex_h + t_off
                if t_min - 1e-9 <= t <= t_max + 1e-9:
                    centers.append((s, t))
        if not centers:
            return np.empty((0, 2), dtype=np.float64)
        return np.array(centers, dtype=np.float64)

    def corner_offsets(self) -> np.ndarray:
        """(6, 2) flat-top hexagon corner offsets from a centre, in local units."""
        angles = np.radians(np.arange(6) * 60.0)
        return np.column_stack([self.radius * np.cos(angles), self.radius * np.sin(angles)])

    def centers_ecef(self) -> np.ndarray:
        """(M, 3) unit-sphere position of every hexagon centre."""
        if len(self) == 0:
            return np.empty((0, 3), dtype=np.float64)
        return self.geometry.local_to_ecef(self.centers[:, 0], self.centers[:, 1])

    def texture_local(self, tex_size: int) -> np.ndarray:
        """(M, T, T, 2) cell-centred local ``(s, t)`` coords of every hex texel."""
        if tex_size < 1:
            raise ValueError(f"tex_size must be >= 1, got {tex_size}")
        half = _TEXTURE_HALF_WIDTH * self.radius
        # Cell centres across the texture square; rows run top (+t) to bottom.
        frac = (np.arange(tex_size, dtype=np.float64) + 0.5) / tex_size
        ds = (2.0 * frac - 1.0) * half                    # left → right
        dt = (1.0 - 2.0 * frac) * half                    # top → bottom
        dss, dtt = np.meshgrid(ds, dt)                    # (T, T)
        offsets = np.stack([dss, dtt], axis=-1)           # (T, T, 2)
        return self.centers[:, None, None, :] + offsets[None, :, :, :]

    def texture_positions(self, tex_size: int) -> np.ndarray:
        """(M, T, T, 3) unit-sphere position of every hex texel."""
        local = self.texture_local(tex_size)
        flat = local.reshape(-1, 2)
        if flat.size == 0:
            return np.empty((0, tex_size, tex_size, 3), dtype=np.float64)
        ecef = self.geometry.local_to_ecef(flat[:, 0], flat[:, 1])
        return ecef.reshape(len(self), tex_size, tex_size, 3)

    def texture_in_hexagon(self, tex_size: int) -> np.ndarray:
        """(M, T, T) bool mask: which texels fall inside their own hexagon.

        Only these count when aggregating, so the over-hang never double-counts.
        """
        local = self.texture_local(tex_size)
        d = local - self.centers[:, None, None, :]
        R = self.radius
        x, y = np.abs(d[..., 0]), np.abs(d[..., 1])
        sqrt3 = math.sqrt(3.0)
        # Flat-top hexagon = intersection of three slabs.
        return (
            (y <= sqrt3 * 0.5 * R + 1e-12)
            & (sqrt3 * x + y <= sqrt3 * R + 1e-12)
            & (sqrt3 * x - y <= sqrt3 * R + 1e-12)
        )

    def texel_at(self, s: float, t: float, tex_size: int) -> Optional[tuple[int, int, int]]:
        """``(hex, row, col)`` of the texel drawn at local point ``(s, t)``, or None.

        The owning hexagon is the one with the nearest centre; ``None`` outside the cell.
        """
        hexes, rows, cols, inside = self.texels_at(np.array([s]), np.array([t]), tex_size)
        if not inside[0]:
            return None
        return int(hexes[0]), int(rows[0]), int(cols[0])

    def texels_at(
        self, s: np.ndarray, t: np.ndarray, tex_size: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """:meth:`texel_at` for arrays: ``(hex, row, col, inside)``, zeros outside the cell."""
        s = np.asarray(s, dtype=np.float64).ravel()
        t = np.asarray(t, dtype=np.float64).ravel()
        inside = (s >= 0.0) & (s <= 1.0) & (t >= 0.0) & (t <= 1.0) & (len(self) > 0)
        zeros = np.zeros(s.shape, dtype=np.int64)
        if not inside.any():
            return zeros, zeros.copy(), zeros.copy(), inside
        d2 = (self.centers[None, :, 0] - s[:, None]) ** 2 + (self.centers[None, :, 1] - t[:, None]) ** 2
        hexes = np.where(inside, np.argmin(d2, axis=1), 0)
        half = _TEXTURE_HALF_WIDTH * self.radius
        cs, ct = self.centers[hexes, 0], self.centers[hexes, 1]
        cols = np.clip(((s - (cs - half)) / (2.0 * half) * tex_size).astype(np.int64), 0, tex_size - 1)
        rows = np.clip((((ct + half) - t) / (2.0 * half) * tex_size).astype(np.int64), 0, tex_size - 1)
        return hexes, np.where(inside, rows, 0), np.where(inside, cols, 0), inside

    def texel_spacing_rad(self, tex_size: int) -> float:
        """Mean angular spacing (radians) between neighbouring texels on the sphere (cached)."""
        cached = self._spacing.get(tex_size)
        if cached is not None:
            return cached
        positions = self.texture_positions(tex_size)
        if positions.size == 0 or tex_size < 2:
            return 0.0
        dx = np.linalg.norm(np.diff(positions, axis=2), axis=-1)
        dy = np.linalg.norm(np.diff(positions, axis=1), axis=-1)
        spacing = float(np.mean([dx.mean(), dy.mean()]))
        self._spacing[tex_size] = spacing
        return spacing
