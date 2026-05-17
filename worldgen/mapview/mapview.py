from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Sequence

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPolygonF, QTransform
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from mapview.projections import ALL_PROJECTIONS, RobinsonProjection, Projection

if TYPE_CHECKING:
    from mesh.mymesh import PolygonMesh


def _split_at_antimeridian(face_ecef: np.ndarray, face_uv: Optional[np.ndarray] = None) -> list:
    """
    face_ecef: (N, 3) ECEF unit-sphere vertices of a polygon.
    face_uv: optional (N, 2) per-vertex UV coordinates.

    Without face_uv: returns list of (M, 2) [lon, lat] arrays.
    With face_uv: returns list of ((M, 2) lonlat, (M, 2) uv) tuples.
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
        lonlat = np.column_stack([lon, lat])
        return [lonlat] if face_uv is None else [(lonlat, face_uv)]

    east_ll: list[list[float]] = []   # lon in [0, π]
    west_ll: list[list[float]] = []   # lon in [-π, 0)
    east_uv_pts: list[np.ndarray] = []
    west_uv_pts: list[np.ndarray] = []

    for i in range(n):
        lo, la = float(lon[i]), float(lat[i])
        if lo >= 0:
            east_ll.append([lo, la])
            if face_uv is not None:
                east_uv_pts.append(face_uv[i])
        else:
            west_ll.append([lo, la])
            if face_uv is not None:
                west_uv_pts.append(face_uv[i])

        if crosses[i]:
            # Find where the edge crosses y=0 (the antimeridian half-plane x<0, y=0)
            a, b = face_ecef[i], face_ecef[(i + 1) % n]
            t = -a[1] / (b[1] - a[1])
            p = a + t * (b - a)
            p = p / np.linalg.norm(p)
            p_lat = float(np.arcsin(np.clip(p[2], -1.0, 1.0)))

            if lo >= 0:
                east_ll.append([np.pi, p_lat])
                west_ll.append([-np.pi, p_lat])
                if face_uv is not None:
                    uv_cross = face_uv[i] + t * (face_uv[(i + 1) % n] - face_uv[i])
                    east_uv_pts.append(uv_cross)
                    west_uv_pts.append(uv_cross)
            else:
                west_ll.append([-np.pi, p_lat])
                east_ll.append([np.pi, p_lat])
                if face_uv is not None:
                    uv_cross = face_uv[i] + t * (face_uv[(i + 1) % n] - face_uv[i])
                    west_uv_pts.append(uv_cross)
                    east_uv_pts.append(uv_cross)

    result = []
    if len(east_ll) >= 3:
        lonlat = np.array(east_ll, dtype=np.float64)
        result.append(lonlat if face_uv is None else (lonlat, np.array(east_uv_pts, dtype=np.float32)))
    if len(west_ll) >= 3:
        lonlat = np.array(west_ll, dtype=np.float64)
        result.append(lonlat if face_uv is None else (lonlat, np.array(west_uv_pts, dtype=np.float32)))
    return result


def _compute_face_uv(face_ecef: np.ndarray) -> np.ndarray:
    """Tangent-plane UV parameterization. Returns (N, 2) float32 with values in [0, 1]."""
    centroid = np.mean(face_ecef, axis=0)
    n = centroid / np.linalg.norm(centroid)
    ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u_axis = np.cross(ref, n)
    u_axis /= np.linalg.norm(u_axis)
    v_axis = np.cross(n, u_axis)
    offsets = face_ecef - centroid
    us = offsets @ u_axis
    vs = offsets @ v_axis
    u_r = float(us.max() - us.min()) or 1.0
    v_r = float(vs.max() - vs.min()) or 1.0
    return np.stack([(us - us.min()) / u_r, (vs - vs.min()) / v_r], axis=1).astype(np.float32)


class MapWidget(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._mesh: Optional[PolygonMesh] = None
        self._projection: Projection = RobinsonProjection()

        # Lon/lat polygons after antimeridian split (coordinate-system-agnostic)
        self._faces_lonlat: list[np.ndarray] = []  # each (N, 2)
        self._face_src: list[int] = []             # original face index per polygon
        self._face_colors: list[np.ndarray] = []   # RGBA color per polygon
        self._face_uvs: list[np.ndarray] = []      # per-vertex UV (N, 2), populated when textured

        # Projected (u, v) polygons, rebuilt on projection change
        self._proj_faces: list[np.ndarray] = []    # each (N, 2)

        self._texture_qimages: dict[int, QImage] = {}  # face_src → QImage, when textured

        self._color_schemes: dict[str, Callable[[dict], Sequence[float]]] = {}
        self._active_scheme: Optional[str] = None

        self._zoom: float = 1.0
        self._center: np.ndarray = np.zeros(2, dtype=np.float64)
        self._drag_start: Optional[tuple[float, float]] = None
        self._drag_start_center: np.ndarray = np.zeros(2, dtype=np.float64)
        self.setFocusPolicy(Qt.FocusPolicy.WheelFocus)
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    def set_mesh(self, mesh: PolygonMesh) -> None:
        self._mesh = mesh
        self._rebuild_faces()
        self._rebuild_textures()
        self._rebuild_colors()
        self._reproject()
        self.update()

    def set_projection(self, proj: Projection) -> None:
        self._projection = proj
        self._reproject()
        self.update()

    def set_color_schemes(self, schemes: dict[str, Callable[[dict], Sequence[float]]]) -> None:
        self._color_schemes = schemes
        self._active_scheme = next(iter(schemes)) if schemes else None
        self._rebuild_colors()
        self.update()

    def set_color_scheme(self, name: str) -> None:
        self._active_scheme = name
        self._rebuild_colors()
        self.update()

    def _rebuild_faces(self) -> None:
        """Convert ECEF faces to lon/lat polygons, splitting at the antimeridian."""
        if self._mesh is None:
            return
        verts = self._mesh.vertices
        self._faces_lonlat = []
        self._face_src = []
        self._face_uvs = []
        has_textures = self._mesh.face_textures is not None
        for i, face in enumerate(self._mesh.faces):
            face_ecef = verts[np.array(face, dtype=np.int32)]
            if has_textures:
                face_uv = _compute_face_uv(face_ecef)
                for poly_lonlat, poly_uv in _split_at_antimeridian(face_ecef, face_uv):
                    self._faces_lonlat.append(poly_lonlat)
                    self._face_src.append(i)
                    self._face_uvs.append(poly_uv)
            else:
                for poly_lonlat in _split_at_antimeridian(face_ecef):
                    self._faces_lonlat.append(poly_lonlat)
                    self._face_src.append(i)

    def _rebuild_textures(self) -> None:
        """Build cached QImages from mesh face textures, one per face."""
        self._texture_qimages = {}
        if self._mesh is None or self._mesh.face_textures is None:
            return
        textures = self._mesh.face_textures  # (N, H, W, 4) float32
        N, H, W = textures.shape[:3]
        for i in range(N):
            img = np.ascontiguousarray((textures[i] * 255).clip(0, 255).astype(np.uint8))
            self._texture_qimages[i] = QImage(
                img.data, W, H, W * 4, QImage.Format.Format_RGBA8888
            ).copy()

    def _rebuild_colors(self) -> None:
        """Build per-polygon color list indexed via _face_src."""
        if self._mesh is None or not self._face_src:
            return
        if self._active_scheme is not None:
            functor = self._color_schemes[self._active_scheme]
            self._face_colors = [
                np.asarray(functor(self._mesh.face_metadata[i]), dtype=np.float32)
                for i in self._face_src
            ]
        else:
            self._face_colors = [self._mesh.face_colors[i] for i in self._face_src]

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

        cx, cy = self._center[0], self._center[1]
        scale_x = self._zoom * 0.5 * W
        scale_y = self._zoom * 0.5 * H

        painter.setPen(Qt.PenStyle.NoPen)
        for idx, (poly_proj, color) in enumerate(zip(self._proj_faces, self._face_colors)):
            face_src = self._face_src[idx]
            px = (poly_proj[:, 0] - cx) * scale_x + W * 0.5
            py = -(poly_proj[:, 1] - cy) * scale_y + H * 0.5

            if face_src in self._texture_qimages:
                # Per-triangle affine texture mapping: solve image-pixel → screen for each fan triangle.
                qimage = self._texture_qimages[face_src]
                img_w = float(qimage.width())
                img_h = float(qimage.height())
                uvs = self._face_uvs[idx]  # (n_verts, 2)
                n_verts = len(px)
                for k in range(1, n_verts - 1):
                    sx = np.array([px[0], px[k], px[k + 1]])
                    sy = np.array([py[0], py[k], py[k + 1]])
                    tu = uvs[[0, k, k + 1], 0] * img_w
                    tv = uvs[[0, k, k + 1], 1] * img_h
                    A = np.array([[tu[0], tv[0], 1.0],
                                  [tu[1], tv[1], 1.0],
                                  [tu[2], tv[2], 1.0]])
                    if abs(np.linalg.det(A)) < 1e-8:
                        continue
                    X = np.linalg.solve(A, np.array([[sx[0], sy[0]],
                                                     [sx[1], sy[1]],
                                                     [sx[2], sy[2]]]))
                    # QTransform(m11,m12,m21,m22,dx,dy): x'=m11*x+m21*y+dx, y'=m12*x+m22*y+dy
                    brush = QBrush(qimage)
                    brush.setTransform(QTransform(
                        float(X[0, 0]), float(X[0, 1]),
                        float(X[1, 0]), float(X[1, 1]),
                        float(X[2, 0]), float(X[2, 1]),
                    ))
                    painter.setBrush(brush)
                    painter.drawPolygon(QPolygonF([
                        QPointF(sx[0], sy[0]),
                        QPointF(sx[1], sy[1]),
                        QPointF(sx[2], sy[2]),
                    ]))
            else:
                painter.setBrush(
                    QBrush(QColor.fromRgbF(
                        float(color[0]), float(color[1]),
                        float(color[2]), float(color[3])
                    ))
                )
                painter.drawPolygon(
                    QPolygonF([QPointF(float(x), float(y)) for x, y in zip(px, py)])
                )

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = (float(event.position().x()), float(event.position().y()))
            self._drag_start_center = self._center.copy()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event) -> None:
        if self._drag_start is None:
            return
        W, H = float(self.width()), float(self.height())
        dx = float(event.position().x()) - self._drag_start[0]
        dy = float(event.position().y()) - self._drag_start[1]
        scale_x = self._zoom * 0.5 * W
        scale_y = self._zoom * 0.5 * H
        self._center[0] = self._drag_start_center[0] - dx / scale_x
        self._center[1] = self._drag_start_center[1] + dy / scale_y
        self._clamp_center()
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = None
            self.setCursor(Qt.CursorShape.OpenHandCursor)

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        if delta:
            pos = event.position()
            self._zoom_toward(float(pos.x()), float(pos.y()), 1.2 ** (delta / 120.0))
        event.accept()

    def _zoom_toward(self, mx: float, my: float, factor: float) -> None:
        W, H = float(self.width()), float(self.height())
        u = (mx - W * 0.5) / (self._zoom * 0.5 * W) + self._center[0]
        v = -(my - H * 0.5) / (self._zoom * 0.5 * H) + self._center[1]
        new_zoom = max(1.0, min(50.0, self._zoom * factor))
        if new_zoom == 1.0:
            self._zoom = 1.0
            self._center[:] = 0.0
        else:
            self._zoom = new_zoom
            self._center[0] = u - (mx - W * 0.5) / (self._zoom * 0.5 * W)
            self._center[1] = v + (my - H * 0.5) / (self._zoom * 0.5 * H)
        self.update()

    def zoom_in(self) -> None:
        self._zoom_toward(self.width() * 0.5, self.height() * 0.5, 1.2)

    def zoom_out(self) -> None:
        self._zoom_toward(self.width() * 0.5, self.height() * 0.5, 1.0 / 1.2)

    def _clamp_center(self) -> None:
        limit = 1.0 - 1.0 / self._zoom
        self._center[0] = float(np.clip(self._center[0], -limit, limit))
        self._center[1] = float(np.clip(self._center[1], -limit, limit))

    def reset_view(self) -> None:
        self._zoom = 1.0
        self._center[:] = 0.0
        self.update()

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

        self._color_label = QLabel("Coloration:")
        self._color_combo = QComboBox()
        toolbar_layout.addWidget(self._color_label)
        toolbar_layout.addWidget(self._color_combo)
        self._color_label.setVisible(False)
        self._color_combo.setVisible(False)

        toolbar_layout.addStretch()

        zoom_out_btn = QPushButton("−")
        zoom_out_btn.setFixedWidth(30)
        zoom_out_btn.clicked.connect(lambda: self._map.zoom_out())
        toolbar_layout.addWidget(zoom_out_btn)

        zoom_in_btn = QPushButton("+")
        zoom_in_btn.setFixedWidth(30)
        zoom_in_btn.clicked.connect(lambda: self._map.zoom_in())
        toolbar_layout.addWidget(zoom_in_btn)

        reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(lambda: self._map.reset_view())
        toolbar_layout.addWidget(reset_btn)

        bottom_toolbar = QWidget()
        bottom_layout = QHBoxLayout(bottom_toolbar)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.addStretch()
        self._rerun_btn = QPushButton("Re-run")
        self._rerun_btn.setVisible(False)
        self._has_rebuild_callback = False
        bottom_layout.addWidget(self._rerun_btn)

        self._map = MapWidget()
        self._color_combo.currentTextChanged.connect(self._map.set_color_scheme)

        layout.addWidget(toolbar)
        layout.addWidget(self._map, stretch=1)
        layout.addWidget(bottom_toolbar)
        self.setCentralWidget(central)

        if mesh is not None:
            self.set_mesh(mesh)

    def set_mesh(self, mesh: PolygonMesh) -> None:
        self._map.set_mesh(mesh)

    def set_rebuild_callback(self, callback: Callable[[], None]) -> None:
        if self._has_rebuild_callback:
            self._rerun_btn.clicked.disconnect()
        self._rerun_btn.clicked.connect(callback)
        self._has_rebuild_callback = True
        self._rerun_btn.setVisible(True)

    def set_color_schemes(self, schemes: dict[str, Callable[[dict], Sequence[float]]]) -> None:
        self._color_combo.blockSignals(True)
        self._color_combo.clear()
        for name in schemes:
            self._color_combo.addItem(name)
        self._color_combo.blockSignals(False)
        visible = bool(schemes)
        self._color_label.setVisible(visible)
        self._color_combo.setVisible(visible)
        self._map.set_color_schemes(schemes)
