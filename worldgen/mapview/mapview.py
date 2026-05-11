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

_ANTIMERIDIAN_THRESHOLD = 1.0


class MapWidget(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._mesh: Optional[PolygonMesh] = None
        self._projection: Projection = EquirectangularProjection()
        self._override_colors: Optional[np.ndarray] = None

        self._proj_verts: Optional[np.ndarray] = None
        self._active_colors: Optional[np.ndarray] = None
        self._skip_faces: Optional[np.ndarray] = None
        self._face_vert_indices: Optional[list[np.ndarray]] = None

    def set_mesh(self, mesh: PolygonMesh) -> None:
        self._mesh = mesh
        self._face_vert_indices = [np.array(f, dtype=np.int32) for f in mesh.faces]
        self._recompute()
        self.update()

    def set_projection(self, proj: Projection) -> None:
        self._projection = proj
        self._recompute()
        self.update()

    def set_face_colors(self, colors: Optional[np.ndarray]) -> None:
        self._override_colors = colors
        self._recompute()
        self.update()

    def _recompute(self) -> None:
        if self._mesh is None:
            return

        self._proj_verts = self._projection.project(self._mesh.vertices)

        if self._override_colors is not None:
            self._active_colors = self._override_colors
        else:
            counts = np.array([len(f) - 2 for f in self._mesh.faces], dtype=np.int32)
            starts = np.concatenate([[0], np.cumsum(counts[:-1])])
            self._active_colors = self._mesh.render_colors[starts * 3]

        n = len(self._mesh.faces)
        skip = np.zeros(n, dtype=bool)
        for i, indices in enumerate(self._face_vert_indices):
            u_vals = self._proj_verts[indices, 0]
            if u_vals.max() - u_vals.min() > _ANTIMERIDIAN_THRESHOLD:
                # Face vertices span more than half the map width — almost certainly
                # wrapping across the ±180° antimeridian boundary. Skip rather than
                # draw a huge horizontal band. Proper fix: clip polygon at antimeridian.
                skip[i] = True
        self._skip_faces = skip

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        W, H = float(self.width()), float(self.height())

        painter.fillRect(self.rect(), QColor(30, 100, 180))

        if self._mesh is None or self._proj_verts is None:
            return

        for i, (indices, skip) in enumerate(
            zip(self._face_vert_indices, self._skip_faces)
        ):
            if skip:
                continue
            uvs = self._proj_verts[indices]
            px = (uvs[:, 0] + 1.0) * 0.5 * W
            py = (1.0 - uvs[:, 1]) * 0.5 * H
            color = self._active_colors[i]
            painter.setBrush(
                QBrush(QColor.fromRgbF(float(color[0]), float(color[1]),
                                       float(color[2]), float(color[3])))
            )
            painter.setPen(Qt.NoPen)
            poly = QPolygonF([QPointF(float(x), float(y)) for x, y in zip(px, py)])
            painter.drawPolygon(poly)

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
