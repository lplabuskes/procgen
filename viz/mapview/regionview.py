from __future__ import annotations

import math
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtWidgets import QWidget

from region.geometry import RegionGeometry


class RegionView(QWidget):
    """Zoomed-in view of a single region.

    The region cell is a unit square in the equal-area ``(s, t)`` local frame
    (see :mod:`region.geometry`).  This view tiles that square with a regular
    hexagonal grid; every hex centre is also mapped to a sphere position via the
    region's :class:`RegionGeometry`, so the grid is anchored to real geography.
    Those sphere positions are what regional upsampling will sample global
    fields at — the geometry mapping this view exists to exercise.
    """

    # Number of hexagon columns spanning the cell's local s-axis.
    _N_COLS: int = 14

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._geometry: Optional[RegionGeometry] = None
        # Hex centres in the cell's local [0, 1]^2 frame and their sphere positions.
        self._hex_local: np.ndarray = np.empty((0, 2), dtype=np.float64)
        self._hex_ecef: np.ndarray = np.empty((0, 3), dtype=np.float64)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def set_region(self, geometry: RegionGeometry) -> None:
        self._geometry = geometry
        self._build_hex_grid()
        self.update()

    @property
    def region_id(self) -> Optional[int]:
        return self._geometry.region_id if self._geometry is not None else None

    @property
    def hex_local(self) -> np.ndarray:
        """(M, 2) hex-centre coordinates in the cell's local [0, 1]^2 frame."""
        return self._hex_local

    @property
    def hex_ecef(self) -> np.ndarray:
        """(M, 3) ECEF unit-sphere position of each hex centre."""
        return self._hex_ecef

    def _hex_radius_local(self) -> float:
        """Flat-top hex circumradius in local units so _N_COLS columns span [0, 1]."""
        return 1.0 / (1.5 * self._N_COLS + 0.5)

    def _build_hex_grid(self) -> None:
        """Lay out flat-top hex centres over the cell's local square and map to sphere."""
        if self._geometry is None:
            self._hex_local = np.empty((0, 2), dtype=np.float64)
            self._hex_ecef = np.empty((0, 3), dtype=np.float64)
            return

        R = self._hex_radius_local()
        hex_h = math.sqrt(3.0) * R          # full height of a flat-top hex
        dx = 1.5 * R                         # column centre-to-centre spacing
        dy = hex_h                           # row centre-to-centre spacing

        centers: list[tuple[float, float]] = []
        n_cols = int(math.ceil(1.0 / dx)) + 1
        n_rows = int(math.ceil(1.0 / dy)) + 1
        for col in range(n_cols):
            s = col * dx
            t_off = (hex_h * 0.5) if (col % 2) else 0.0
            for row in range(n_rows):
                t = row * dy + t_off
                # Keep hexes whose centre falls within the cell square.
                if -1e-9 <= s <= 1.0 + 1e-9 and -1e-9 <= t <= 1.0 + 1e-9:
                    centers.append((s, t))

        self._hex_local = np.array(centers, dtype=np.float64) if centers else np.empty((0, 2))
        if len(self._hex_local):
            self._hex_ecef = self._geometry.local_to_ecef(
                self._hex_local[:, 0], self._hex_local[:, 1]
            )
        else:
            self._hex_ecef = np.empty((0, 3), dtype=np.float64)

    def paintEvent(self, _) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(24, 28, 34))

        W, H = float(self.width()), float(self.height())
        margin = 16.0
        size = min(W, H) - 2.0 * margin
        if size <= 0.0 or self._geometry is None:
            return
        x0 = (W - size) * 0.5
        y0 = (H - size) * 0.5

        self._draw_hex_grid(painter, x0, y0, size)

        painter.setPen(QColor(200, 210, 220))
        font = painter.font()
        font.setPointSizeF(11.0)
        painter.setFont(font)
        rid = self._geometry.region_id
        painter.drawText(12, 22, f"Region {rid}  ({len(self._hex_local)} hexes)")

    def _draw_hex_grid(self, painter: QPainter, x0: float, y0: float, size: float) -> None:
        """Render the local-frame hex centres as a regular grid inside the cell square."""
        R_px = self._hex_radius_local() * size
        # Flat-top hexagon corner offsets (pixels).
        corners = [
            (R_px * math.cos(math.radians(60 * i)), R_px * math.sin(math.radians(60 * i)))
            for i in range(6)
        ]

        painter.save()
        painter.setClipRect(QRectF(x0, y0, size, size))
        pen = QPen(QColor(90, 110, 130))
        pen.setWidthF(1.0)
        painter.setPen(pen)
        painter.setBrush(QColor(34, 40, 48))
        for s, t in self._hex_local:
            cx = x0 + s * size
            cy = y0 + (1.0 - t) * size   # flip so local +t (north) is up on screen
            poly = QPolygonF([QPointF(cx + ox, cy + oy) for ox, oy in corners])
            painter.drawPolygon(poly)
        painter.restore()

        painter.setPen(QPen(QColor(70, 85, 100), 1.0))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(QRectF(x0, y0, size, size))
