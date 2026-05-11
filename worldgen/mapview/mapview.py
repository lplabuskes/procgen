from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QPolygonF
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QVBoxLayout,
    QWidget,
)

from projections import ALL_PROJECTIONS, EquirectangularProjection, Projection

if TYPE_CHECKING:
    from mymesh import PolygonMesh


def _split_at_antimeridian(face_ecef: np.ndarray) -> list[np.ndarray]:
    """
    face_ecef: (N, 3) ECEF unit-sphere vertices of a polygon.
    Returns a list of (M, 2) [lon, lat] (radians) polygons.
    Single-element list if no antimeridian crossing; two elements if crossing.
    """
    n = len(face_ecef)
    lon = np.arctan2(face_ecef[:, 1], face_ecef[:, 0])
    lat = np.arcsin(np.clip(face_ecef[:, 2], -1.0, 1.0))

    # An edge crosses the antimeridian when the direct lon difference exceeds π,
    # meaning the shorter geodesic path goes via ±180°.
    crosses = np.array(
        [abs(lon[i] - lon[(i + 1) % n]) > np.pi for i in range(n)], dtype=bool
    )

    if not crosses.any():
        return [np.column_stack([lon, lat])]

    east: list[list[float]] = []  # lon in [0, π]
    west: list[list[float]] = []  # lon in [-π, 0)

    for i in range(n):
        lo, la = float(lon[i]), float(lat[i])
        (east if lo >= 0 else west).append([lo, la])

        if crosses[i]:
            # Find where the edge crosses y=0 (the antimeridian half-plane x<0, y=0)
            a, b = face_ecef[i], face_ecef[(i + 1) % n]
            t = -a[1] / (b[1] - a[1])
            p = a + t * (b - a)
            p = p / np.linalg.norm(p)
            p_lat = float(np.arcsin(np.clip(p[2], -1.0, 1.0)))

            if lo >= 0:
                east.append([np.pi, p_lat])
                west.append([-np.pi, p_lat])
            else:
                west.append([-np.pi, p_lat])
                east.append([np.pi, p_lat])

    result = []
    if len(east) >= 3:
        result.append(np.array(east, dtype=np.float64))
    if len(west) >= 3:
        result.append(np.array(west, dtype=np.float64))
    return result


class MapWidget(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._mesh: Optional[PolygonMesh] = None
        self._projection: Projection = EquirectangularProjection()
        self._override_colors: Optional[np.ndarray] = None

        # Lon/lat polygons after antimeridian split (coordinate-system-agnostic)
        self._faces_lonlat: list[np.ndarray] = []  # each (N, 2)
        self._face_src: list[int] = []             # original face index per polygon
        self._face_colors: list[np.ndarray] = []   # RGBA color per polygon

        # Projected (u, v) polygons, rebuilt on projection change
        self._proj_faces: list[np.ndarray] = []    # each (N, 2)

    def set_mesh(self, mesh: PolygonMesh) -> None:
        self._mesh = mesh
        self._rebuild_faces()
        self._rebuild_colors()
        self._reproject()
        self.update()

    def set_projection(self, proj: Projection) -> None:
        self._projection = proj
        self._reproject()
        self.update()

    def set_face_colors(self, colors: Optional[np.ndarray]) -> None:
        self._override_colors = colors
        self._rebuild_colors()
        self.update()

    def _rebuild_faces(self) -> None:
        """Convert ECEF faces to lon/lat polygons, splitting at the antimeridian."""
        if self._mesh is None:
            return
        verts = self._mesh.vertices
        self._faces_lonlat = []
        self._face_src = []
        for i, face in enumerate(self._mesh.faces):
            face_ecef = verts[np.array(face, dtype=np.int32)]
            for poly_lonlat in _split_at_antimeridian(face_ecef):
                self._faces_lonlat.append(poly_lonlat)
                self._face_src.append(i)

    def _rebuild_colors(self) -> None:
        """Build per-polygon color list indexed via _face_src."""
        if self._mesh is None or not self._face_src:
            return
        if self._override_colors is not None:
            raw = self._override_colors
        else:
            counts = np.array([len(f) - 2 for f in self._mesh.faces], dtype=np.int32)
            starts = np.concatenate([[0], np.cumsum(counts[:-1])])
            raw = self._mesh.render_colors[starts * 3]
        self._face_colors = [raw[i] for i in self._face_src]

    def _reproject(self) -> None:
        """Project lon/lat polygons to (u, v) map coordinates."""
        self._proj_faces = [
            self._projection.project_lonlat(poly) for poly in self._faces_lonlat
        ]

    def paintEvent(self, _) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = float(self.width()), float(self.height())

        painter.fillRect(self.rect(), QColor(30, 100, 180))

        if not self._proj_faces:
            return

        painter.setPen(Qt.PenStyle.NoPen)
        for poly_uv, color in zip(self._proj_faces, self._face_colors):
            px = (poly_uv[:, 0] + 1.0) * 0.5 * W
            py = (1.0 - poly_uv[:, 1]) * 0.5 * H
            painter.setBrush(
                QBrush(QColor.fromRgbF(
                    float(color[0]), float(color[1]),
                    float(color[2]), float(color[3])
                ))
            )
            painter.drawPolygon(
                QPolygonF([QPointF(float(x), float(y)) for x, y in zip(px, py)])
            )

    def resizeEvent(self, event) -> None:
        self.update()


class MapWindow(QMainWindow):
    def __init__(self, mesh: Optional[PolygonMesh] = None) -> None:
        super().__init__()
        self.setWindowTitle("Map Viewer")
        self.resize(900, 500)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        toolbar = QWidget()
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(0, 0, 0, 0)
        toolbar_layout.addWidget(QLabel("Projection:"))
        self._combo = QComboBox()
        for proj in ALL_PROJECTIONS:
            self._combo.addItem(proj.name)
        self._combo.currentIndexChanged.connect(
            lambda i: self._map.set_projection(ALL_PROJECTIONS[i])
        )
        toolbar_layout.addWidget(self._combo)
        toolbar_layout.addStretch()

        self._map = MapWidget()

        layout.addWidget(toolbar)
        layout.addWidget(self._map, stretch=1)
        self.setCentralWidget(central)

        if mesh is not None:
            self.set_mesh(mesh)

    def set_mesh(self, mesh: PolygonMesh) -> None:
        self._map.set_mesh(mesh)

    def set_face_colors(self, colors: Optional[np.ndarray]) -> None:
        self._map.set_face_colors(colors)
