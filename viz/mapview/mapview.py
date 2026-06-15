from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Sequence

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QBrush, QColor, QImage, QLinearGradient, QPainter, QPen, QPolygonF, QTransform
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from viz.mapview.antimeridian import find_ideal_antimeridian
from viz.mapview.projections import ALL_PROJECTIONS, RobinsonProjection, Projection

if TYPE_CHECKING:
    from core.mesh import PolygonMesh
    from core.world import World


def _pip_polygon(pu: float, pv: float, poly: np.ndarray) -> bool:
    """Ray-casting point-in-polygon test. poly is (N, 2) in map coordinates."""
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = float(poly[i, 0]), float(poly[i, 1])
        xj, yj = float(poly[j, 0]), float(poly[j, 1])
        if ((yi > pv) != (yj > pv)) and pu < (xj - xi) * (pv - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def _bary_in_fan(
    pu: float, pv: float, poly_proj: np.ndarray, uvs: np.ndarray
) -> Optional[np.ndarray]:
    """Barycentric UV at (pu, pv) inside a fan-triangulated polygon. Returns None if outside all triangles."""
    n = len(poly_proj)
    ax, ay = float(poly_proj[0, 0]), float(poly_proj[0, 1])
    for k in range(1, n - 1):
        bx, by = float(poly_proj[k, 0]), float(poly_proj[k, 1])
        cx, cy = float(poly_proj[k + 1, 0]), float(poly_proj[k + 1, 1])
        denom = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(denom) < 1e-14:
            continue
        w0 = ((by - cy) * (pu - cx) + (cx - bx) * (pv - cy)) / denom
        w1 = ((cy - ay) * (pu - cx) + (ax - cx) * (pv - cy)) / denom
        w2 = 1.0 - w0 - w1
        if w0 >= -0.02 and w1 >= -0.02 and w2 >= -0.02:
            return w0 * uvs[0] + w1 * uvs[k] + w2 * uvs[k + 1]
    return None


def _split_at_antimeridian(
    face_ecef: np.ndarray,
    face_uv: Optional[np.ndarray] = None,
    cut_lon: float = np.pi,
) -> list:
    """
    face_ecef: (N, 3) ECEF unit-sphere vertices of a polygon.
    face_uv: optional (N, 2) per-vertex UV coordinates.
    cut_lon: longitude (radians) to treat as the antimeridian.  Default π keeps
             the standard ±180° split; any other value rotates the cut.

    Without face_uv: returns list of (M, 2) [lon, lat] arrays in the shifted
    coordinate frame where cut_lon maps to ±π.
    With face_uv: returns list of ((M, 2) lonlat, (M, 2) uv) tuples.
    Single-element list if no crossing; two elements if crossing.
    """
    n = len(face_ecef)
    lon = np.arctan2(face_ecef[:, 1], face_ecef[:, 0])
    lat = np.arcsin(np.clip(face_ecef[:, 2], -1.0, 1.0))

    # Shift so cut_lon maps to ±π.  With cut_lon=π this is the identity on (-π, π].
    lon = ((lon - cut_lon) % (2.0 * np.pi)) - np.pi

    crosses = np.array(
        [abs(lon[i] - lon[(i + 1) % n]) > np.pi for i in range(n)], dtype=bool
    )

    if not crosses.any():
        lonlat = np.column_stack([lon, lat])
        return [lonlat] if face_uv is None else [(lonlat, face_uv)]

    # Precompute the ECEF direction of the cut meridian for crossing interpolation.
    cos_cut = np.cos(cut_lon)
    sin_cut = np.sin(cut_lon)

    east_ll: list[list[float]] = []   # shifted lon in [0, π]
    west_ll: list[list[float]] = []   # shifted lon in [-π, 0)
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
            # Find t where the edge a→b crosses the cut meridian.
            # The cut meridian satisfies: p_y * cos_cut - p_x * sin_cut = 0.
            a, b = face_ecef[i], face_ecef[(i + 1) % n]
            fa = a[1] * cos_cut - a[0] * sin_cut
            fb = b[1] * cos_cut - b[0] * sin_cut
            t = -fa / (fb - fa)
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


class MapWidget(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._mesh: Optional[PolygonMesh] = None
        self._projection: Projection = RobinsonProjection()
        self._antimeridian: float = np.pi  # cut longitude; updated per mesh

        # Lon/lat polygons after antimeridian split (coordinate-system-agnostic)
        self._faces_lonlat: list[np.ndarray] = []  # each (N, 2)
        self._face_src: list[int] = []             # original face index per polygon
        self._face_colors: list[np.ndarray] = []   # RGBA color per polygon
        self._face_uvs: list[np.ndarray] = []      # per-vertex UV (N, 2), populated when textured

        # Projected (u, v) polygons, rebuilt on projection change
        self._proj_faces: list[np.ndarray] = []    # each (N, 2)
        self._face_bbox_min: np.ndarray = np.empty((0, 2), dtype=np.float64)  # (N, 2)
        self._face_bbox_max: np.ndarray = np.empty((0, 2), dtype=np.float64)  # (N, 2)

        self._texture_qimages: dict[int, QImage] = {}  # face_src → QImage, when textured

        self._color_schemes: dict[str, Callable[[dict], Sequence[float]]] = {}
        self._active_scheme: Optional[str] = None
        self._scheme_texture_key: dict[str, str] = {}

        self._globe: Optional["World"] = None
        self._overlay_visible: dict[str, bool] = {}
        self._legends: dict[str, tuple[str, list]] = {}  # scheme → (kind, entries)

        self._hover_callback: Optional[Callable[[Optional[int], Optional[np.ndarray]], None]] = None

        self._zoom: float = 1.0
        self._center: np.ndarray = np.zeros(2, dtype=np.float64)
        self._drag_start: Optional[tuple[float, float]] = None
        self._drag_start_center: np.ndarray = np.zeros(2, dtype=np.float64)
        self.setFocusPolicy(Qt.FocusPolicy.WheelFocus)
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.setMouseTracking(True)

    def _shift_lon(self, lon: np.ndarray) -> np.ndarray:
        """Shift longitudes into the frame where self._antimeridian maps to ±π."""
        return ((lon - self._antimeridian) % (2.0 * np.pi)) - np.pi

    def set_mesh(self, mesh: PolygonMesh) -> None:
        self._mesh = mesh
        self._antimeridian = find_ideal_antimeridian(mesh)
        self._rebuild_faces()
        self._rebuild_textures()
        self._rebuild_colors()
        self._reproject()
        self.update()

    def set_globe(self, globe: "World") -> None:
        self._globe = globe
        self.set_mesh(globe.mesh)

    def set_overlay_visible(self, name: str, visible: bool) -> None:
        self._overlay_visible[name] = visible
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
        self._rebuild_faces()
        self._rebuild_textures()
        self._rebuild_colors()
        self._reproject()
        self.update()

    def associate_scheme_texture(self, scheme_name: str, texture_key: str) -> None:
        """Associate a named face texture with a color scheme for per-pixel rendering."""
        self._scheme_texture_key[scheme_name] = texture_key
        if self._active_scheme == scheme_name:
            self._rebuild_faces()
            self._rebuild_textures()
            self._reproject()
            self.update()

    def _rebuild_faces(self) -> None:
        """Convert ECEF faces to lon/lat polygons, splitting at the antimeridian."""
        if self._mesh is None:
            return
        verts = self._mesh.vertices
        self._faces_lonlat = []
        self._face_src = []
        self._face_uvs = []
        # Always compute UVs when any textures exist — needed for both rendering and hover hit-testing.
        has_textures = bool(self._mesh.face_textures)
        vertex_uvs: Optional[list[np.ndarray]] = None
        if has_textures:
            *_, vertex_uvs = self._mesh.face_tangent_frames()
        cut = self._antimeridian
        for i, face in enumerate(self._mesh.faces):
            face_ecef = verts[np.array(face, dtype=np.int32)]
            if has_textures:
                assert vertex_uvs is not None
                for poly_lonlat, poly_uv in _split_at_antimeridian(face_ecef, vertex_uvs[i], cut):
                    self._faces_lonlat.append(poly_lonlat)
                    self._face_src.append(i)
                    self._face_uvs.append(poly_uv)
            else:
                for poly_lonlat in _split_at_antimeridian(face_ecef, cut_lon=cut):
                    self._faces_lonlat.append(poly_lonlat)
                    self._face_src.append(i)

    def _rebuild_textures(self) -> None:
        """Build cached QImages from mesh face textures, one per face."""
        self._texture_qimages = {}
        if self._mesh is None or not self._mesh.face_textures:
            return
        if self._scheme_texture_key and self._active_scheme is not None:
            tex_key = self._scheme_texture_key.get(self._active_scheme)
            if tex_key is None or tex_key not in self._mesh.face_textures:
                return
            textures = self._mesh.face_textures[tex_key]
        else:
            textures = next(iter(self._mesh.face_textures.values()))  # (N, H, W, 4) float32
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
        if self._proj_faces:
            self._face_bbox_min = np.array([p.min(axis=0) for p in self._proj_faces], dtype=np.float64)
            self._face_bbox_max = np.array([p.max(axis=0) for p in self._proj_faces], dtype=np.float64)
        else:
            self._face_bbox_min = np.empty((0, 2), dtype=np.float64)
            self._face_bbox_max = np.empty((0, 2), dtype=np.float64)

    def paintEvent(self, _) -> None:
        painter = QPainter(self)
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

        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self._overlay_visible.get("rivers") and self._globe is not None:
            self._draw_rivers_overlay(painter, W, H, cx, cy, scale_x, scale_y)
        if self._overlay_visible.get("lakes") and self._globe is not None:
            self._draw_lakes_overlay(painter, W, H, cx, cy, scale_x, scale_y)
        if self._globe is not None and self._globe.has_feature("winds"):
            for season_idx in range(len(self._globe.get_feature("winds") or [])):
                if self._overlay_visible.get(f"winds_season_{season_idx}"):
                    self._draw_winds_overlay(
                        painter, W, H, cx, cy, scale_x, scale_y, season_idx
                    )
        if (self._overlay_visible.get("currents")
                and self._globe is not None
                and self._globe.has_feature("currents")):
            self._draw_currents_overlay(painter, W, H, cx, cy, scale_x, scale_y)
        if (self._overlay_visible.get("deltas")
                and self._globe is not None
                and self._globe.has_feature("deltas")):
            self._draw_deltas_overlay(painter, W, H, cx, cy, scale_x, scale_y)
        if (self._overlay_visible.get("volcanoes")
                and self._globe is not None
                and self._globe.has_feature("volcanoes")):
            self._draw_volcanoes_overlay(painter, W, H, cx, cy, scale_x, scale_y)
        if (self._overlay_visible.get("minerals")
                and self._globe is not None
                and self._globe.has_feature("minerals")):
            self._draw_minerals_overlay(painter, W, H, cx, cy, scale_x, scale_y)
        if (self._overlay_visible.get("rare_bio")
                and self._globe is not None
                and self._globe.has_feature("rare_bio")):
            self._draw_rare_bio_overlay(painter, W, H, cx, cy, scale_x, scale_y)
        if (self._overlay_visible.get("danger_sources")
                and self._globe is not None
                and self._globe.has_feature("danger_sources")):
            self._draw_danger_sources_overlay(painter, W, H, cx, cy, scale_x, scale_y)
        if (self._overlay_visible.get("trade_routes")
                and self._globe is not None
                and self._globe.has_feature("trade_routes")):
            self._draw_trade_routes_overlay(painter, W, H, cx, cy, scale_x, scale_y)
        if (self._overlay_visible.get("settlements")
                and self._globe is not None
                and self._globe.has_feature("settlements")):
            self._draw_settlements_overlay(painter, W, H, cx, cy, scale_x, scale_y)
        if (self._overlay_visible.get("polity_borders")
                and self._globe is not None
                and self._globe.has_feature("polity_borders")):
            self._draw_polity_borders_overlay(painter, W, H, cx, cy, scale_x, scale_y)

        self._draw_legend(painter, W, H)

    def _draw_lakes_overlay(
        self,
        painter: QPainter,
        W: float, H: float,
        cx: float, cy: float,
        scale_x: float, scale_y: float,
    ) -> None:
        polygons = self._globe.get_feature("lakes")  # type: ignore[union-attr]
        if not polygons:
            return
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(80, 140, 220, 255)))
        for verts in polygons:
            for lonlat in _split_at_antimeridian(verts, cut_lon=self._antimeridian):
                proj = self._projection.project_lonlat(lonlat)
                sx = (proj[:, 0] - cx) * scale_x + W * 0.5
                sy = -(proj[:, 1] - cy) * scale_y + H * 0.5
                painter.drawPolygon(
                    QPolygonF([QPointF(float(x), float(y)) for x, y in zip(sx, sy)])
                )

    def _draw_currents_overlay(
        self,
        painter: QPainter,
        W: float, H: float,
        cx: float, cy: float,
        scale_x: float, scale_y: float,
    ) -> None:
        currents_feature = self._globe.get_feature("currents")  # type: ignore[union-attr]
        if not currents_feature:
            return
        entry = currents_feature[0]
        if len(entry) >= 3:
            positions, vectors, temp_delta = entry[0], entry[1], entry[2]
        else:
            positions, vectors = entry[0], entry[1]
            temp_delta = np.zeros(len(positions), dtype=np.float32)
        if len(positions) == 0:
            return

        speeds = np.linalg.norm(vectors, axis=1)
        max_speed = float(speeds.max()) if speeds.size else 0.0
        if max_speed <= 1e-6:
            return

        keep = speeds > max_speed * 0.05
        if not np.any(keep):
            return
        positions = positions[keep]
        vectors = vectors[keep]
        speeds = speeds[keep]
        temp_delta = temp_delta[keep]

        step_max = 0.025 / self._zoom ** 0.5
        step = step_max * (speeds / max_speed)
        unit = vectors / np.maximum(speeds[:, None], 1e-12)
        end_positions = positions + step[:, None] * unit
        end_positions /= np.maximum(np.linalg.norm(end_positions, axis=1, keepdims=True), 1e-12)

        lon0 = self._shift_lon(np.arctan2(positions[:, 1], positions[:, 0]))
        lat0 = np.arcsin(np.clip(positions[:, 2], -1.0, 1.0))
        lon1 = self._shift_lon(np.arctan2(end_positions[:, 1], end_positions[:, 0]))
        lat1 = np.arcsin(np.clip(end_positions[:, 2], -1.0, 1.0))

        proj0 = self._projection.project_lonlat(np.column_stack([lon0, lat0]))
        proj1 = self._projection.project_lonlat(np.column_stack([lon1, lat1]))
        sx0 = (proj0[:, 0] - cx) * scale_x + W * 0.5
        sy0 = -(proj0[:, 1] - cy) * scale_y + H * 0.5
        sx1 = (proj1[:, 0] - cx) * scale_x + W * 0.5
        sy1 = -(proj1[:, 1] - cy) * scale_y + H * 0.5

        crosses = np.abs(lon1 - lon0) > np.pi

        # Diverging blue → neutral → red palette, quantised to keep QPen churn low.
        max_abs = float(np.max(np.abs(temp_delta))) if temp_delta.size else 0.0
        if max_abs > 1e-6:
            norm = np.clip(temp_delta / max_abs, -1.0, 1.0)
        else:
            norm = np.zeros_like(temp_delta)
        bucket_count = 17
        buckets = np.clip(
            np.round((norm + 1.0) * 0.5 * (bucket_count - 1)).astype(np.int32),
            0, bucket_count - 1,
        )
        bucket_pens: dict[int, QPen] = {}

        def _pen_for(bucket: int) -> QPen:
            pen = bucket_pens.get(bucket)
            if pen is not None:
                return pen
            t = (bucket / (bucket_count - 1)) * 2.0 - 1.0  # in [-1, 1]
            # Blue at t=-1, light gray near 0, red at t=+1.
            if t >= 0.0:
                r = int(200 + 30 * t)
                g = int(200 - 130 * t)
                b = int(200 - 150 * t)
            else:
                r = int(200 + 150 * t)
                g = int(200 + 130 * t)
                b = int(200 - 30 * t)
            pen = QPen(QColor(r, g, b, 230))
            pen.setWidthF(1.4)
            bucket_pens[bucket] = pen
            return pen

        active: int = -1
        for i in range(len(positions)):
            if crosses[i]:
                continue
            b = int(buckets[i])
            if b != active:
                painter.setPen(_pen_for(b))
                active = b
            painter.drawLine(
                QPointF(float(sx0[i]), float(sy0[i])),
                QPointF(float(sx1[i]), float(sy1[i])),
            )

    def _draw_winds_overlay(
        self,
        painter: QPainter,
        W: float, H: float,
        cx: float, cy: float,
        scale_x: float, scale_y: float,
        season_idx: int,
    ) -> None:
        winds_feature = self._globe.get_feature("winds")  # type: ignore[union-attr]
        if not winds_feature or season_idx >= len(winds_feature):
            return
        positions, vectors = winds_feature[season_idx]
        if len(positions) == 0:
            return

        speeds = np.linalg.norm(vectors, axis=1)
        max_speed = float(speeds.max()) if speeds.size else 0.0
        if max_speed <= 1e-6:
            return

        # Arc-length per arrow on the unit sphere; tunable via zoom.
        step = 0.025 / self._zoom ** 0.5
        unit = vectors / np.maximum(speeds[:, None], 1e-12)
        end_positions = positions + step * unit
        end_positions /= np.maximum(np.linalg.norm(end_positions, axis=1, keepdims=True), 1e-12)

        lon0 = self._shift_lon(np.arctan2(positions[:, 1], positions[:, 0]))
        lat0 = np.arcsin(np.clip(positions[:, 2], -1.0, 1.0))
        lon1 = self._shift_lon(np.arctan2(end_positions[:, 1], end_positions[:, 0]))
        lat1 = np.arcsin(np.clip(end_positions[:, 2], -1.0, 1.0))

        proj0 = self._projection.project_lonlat(np.column_stack([lon0, lat0]))
        proj1 = self._projection.project_lonlat(np.column_stack([lon1, lat1]))
        sx0 = (proj0[:, 0] - cx) * scale_x + W * 0.5
        sy0 = -(proj0[:, 1] - cy) * scale_y + H * 0.5
        sx1 = (proj1[:, 0] - cx) * scale_x + W * 0.5
        sy1 = -(proj1[:, 1] - cy) * scale_y + H * 0.5

        crosses = np.abs(lon1 - lon0) > np.pi

        pen = QPen(QColor(40, 40, 40, 200))
        pen.setWidthF(1.2)
        painter.setPen(pen)
        for i in range(len(positions)):
            if crosses[i]:
                continue
            painter.drawLine(
                QPointF(float(sx0[i]), float(sy0[i])),
                QPointF(float(sx1[i]), float(sy1[i])),
            )

    def _draw_deltas_overlay(
        self,
        painter: QPainter,
        W: float, H: float,
        cx: float, cy: float,
        scale_x: float, scale_y: float,
    ) -> None:
        positions = self._globe.get_feature("deltas")  # type: ignore[union-attr]
        if positions is None or len(positions) == 0:
            return
        lon = self._shift_lon(np.arctan2(positions[:, 1], positions[:, 0]))
        lat = np.arcsin(np.clip(positions[:, 2], -1.0, 1.0))
        proj = self._projection.project_lonlat(np.column_stack([lon, lat]))
        sx = (proj[:, 0] - cx) * scale_x + W * 0.5
        sy = -(proj[:, 1] - cy) * scale_y + H * 0.5
        r = max(3.0, 5.0 * self._zoom ** 0.3)
        painter.save()
        painter.setPen(QPen(QColor(0, 100, 0, 220), 1.5))
        painter.setBrush(QBrush(QColor(0, 200, 80, 200)))
        for i in range(len(positions)):
            painter.drawEllipse(QPointF(float(sx[i]), float(sy[i])), r, r)
        painter.restore()

    def _draw_volcanoes_overlay(
        self,
        painter: QPainter,
        W: float, H: float,
        cx: float, cy: float,
        scale_x: float, scale_y: float,
    ) -> None:
        positions = self._globe.get_feature("volcanoes")  # type: ignore[union-attr]
        if positions is None or len(positions) == 0:
            return
        lon = self._shift_lon(np.arctan2(positions[:, 1], positions[:, 0]))
        lat = np.arcsin(np.clip(positions[:, 2], -1.0, 1.0))
        proj = self._projection.project_lonlat(np.column_stack([lon, lat]))
        sx = (proj[:, 0] - cx) * scale_x + W * 0.5
        sy = -(proj[:, 1] - cy) * scale_y + H * 0.5
        r = max(2.5, 4.0 * self._zoom ** 0.3)
        painter.save()
        painter.setPen(QPen(QColor(160, 30, 0, 220), 1.5))
        painter.setBrush(QBrush(QColor(255, 80, 0, 200)))
        for i in range(len(positions)):
            painter.drawEllipse(QPointF(float(sx[i]), float(sy[i])), r, r)
        painter.restore()

    def _draw_minerals_overlay(
        self,
        painter: QPainter,
        W: float, H: float,
        cx: float, cy: float,
        scale_x: float, scale_y: float,
    ) -> None:
        from worldgen.resources.minerals import MINERAL_PALETTE, N_MINERALS
        feature = self._globe.get_feature("minerals")  # type: ignore[union-attr]
        if feature is None:
            return
        positions, kinds, abundances = feature
        if len(positions) == 0:
            return
        lon = self._shift_lon(np.arctan2(positions[:, 1], positions[:, 0]))
        lat = np.arcsin(np.clip(positions[:, 2], -1.0, 1.0))
        proj = self._projection.project_lonlat(np.column_stack([lon, lat]))
        sx = (proj[:, 0] - cx) * scale_x + W * 0.5
        sy = -(proj[:, 1] - cy) * scale_y + H * 0.5
        base_r = max(2.0, 3.5 * self._zoom ** 0.3)
        painter.save()
        kinds_i = np.clip(kinds.astype(np.int32), 0, N_MINERALS - 1)
        for k in range(N_MINERALS):
            mask = kinds_i == k
            if not mask.any():
                continue
            rgba = MINERAL_PALETTE[k]
            r_fill = int(rgba[0] * 255)
            g_fill = int(rgba[1] * 255)
            b_fill = int(rgba[2] * 255)
            r_pen = max(0, r_fill - 60)
            g_pen = max(0, g_fill - 60)
            b_pen = max(0, b_fill - 60)
            painter.setPen(QPen(QColor(r_pen, g_pen, b_pen, 220), 1.0))
            painter.setBrush(QBrush(QColor(r_fill, g_fill, b_fill, 210)))
            k_abund = abundances[mask]
            k_sx = sx[mask]
            k_sy = sy[mask]
            for i in range(len(k_sx)):
                r = base_r * (0.7 + 0.6 * float(k_abund[i]))
                painter.drawRect(
                    QRectF(float(k_sx[i]) - r, float(k_sy[i]) - r, 2 * r, 2 * r)
                )
        painter.restore()

    def _draw_rare_bio_overlay(
        self,
        painter: QPainter,
        W: float, H: float,
        cx: float, cy: float,
        scale_x: float, scale_y: float,
    ) -> None:
        from worldgen.resources.rare_bio import N_RARE_BIO, RARE_BIO_PALETTE
        feature = self._globe.get_feature("rare_bio")  # type: ignore[union-attr]
        if feature is None:
            return
        positions, kinds, abundances = feature
        if len(positions) == 0:
            return
        lon = self._shift_lon(np.arctan2(positions[:, 1], positions[:, 0]))
        lat = np.arcsin(np.clip(positions[:, 2], -1.0, 1.0))
        proj = self._projection.project_lonlat(np.column_stack([lon, lat]))
        sx = (proj[:, 0] - cx) * scale_x + W * 0.5
        sy = -(proj[:, 1] - cy) * scale_y + H * 0.5
        base_r = max(2.5, 4.0 * self._zoom ** 0.3)
        painter.save()
        kinds_i = np.clip(kinds.astype(np.int32), 0, N_RARE_BIO - 1)
        for k in range(N_RARE_BIO):
            mask = kinds_i == k
            if not mask.any():
                continue
            rgba = RARE_BIO_PALETTE[k]
            r_fill = int(rgba[0] * 255)
            g_fill = int(rgba[1] * 255)
            b_fill = int(rgba[2] * 255)
            r_pen = max(0, r_fill - 60)
            g_pen = max(0, g_fill - 60)
            b_pen = max(0, b_fill - 60)
            painter.setPen(QPen(QColor(r_pen, g_pen, b_pen, 220), 1.0))
            painter.setBrush(QBrush(QColor(r_fill, g_fill, b_fill, 210)))
            k_abund = abundances[mask]
            k_sx = sx[mask]
            k_sy = sy[mask]
            for i in range(len(k_sx)):
                r = base_r * (0.7 + 0.6 * float(k_abund[i]))
                # Draw as diamond (rotated square) to distinguish from mineral squares.
                cx_i, cy_i = float(k_sx[i]), float(k_sy[i])
                diamond = QPolygonF([
                    QPointF(cx_i, cy_i - r),
                    QPointF(cx_i + r, cy_i),
                    QPointF(cx_i, cy_i + r),
                    QPointF(cx_i - r, cy_i),
                ])
                painter.drawPolygon(diamond)
        painter.restore()

    def _draw_danger_sources_overlay(
        self,
        painter: QPainter,
        W: float, H: float,
        cx: float, cy: float,
        scale_x: float, scale_y: float,
    ) -> None:
        from worldgen.danger.sources import SOURCE_COLORS
        feature = self._globe.get_feature("danger_sources")  # type: ignore[union-attr]
        if feature is None:
            return
        positions, kinds = feature
        if len(positions) == 0:
            return
        lon = self._shift_lon(np.arctan2(positions[:, 1], positions[:, 0]))
        lat = np.arcsin(np.clip(positions[:, 2], -1.0, 1.0))
        proj = self._projection.project_lonlat(np.column_stack([lon, lat]))
        sx = (proj[:, 0] - cx) * scale_x + W * 0.5
        sy = -(proj[:, 1] - cy) * scale_y + H * 0.5
        r = max(3.0, 5.0 * self._zoom ** 0.3)
        painter.save()
        n_colors = len(SOURCE_COLORS)
        for i in range(len(positions)):
            k = int(kinds[i])
            rgba = SOURCE_COLORS[min(k, n_colors - 1)]
            r_fill = int(rgba[0] * 255)
            g_fill = int(rgba[1] * 255)
            b_fill = int(rgba[2] * 255)
            r_pen = max(0, r_fill - 70)
            g_pen = max(0, g_fill - 70)
            b_pen = max(0, b_fill - 70)
            painter.setPen(QPen(QColor(r_pen, g_pen, b_pen, 230), 1.5))
            painter.setBrush(QBrush(QColor(r_fill, g_fill, b_fill, 210)))
            # Draw as triangle (skull-like pointer) pointing downward.
            px, py = float(sx[i]), float(sy[i])
            triangle = QPolygonF([
                QPointF(px, py + r),
                QPointF(px - r * 0.87, py - r * 0.5),
                QPointF(px + r * 0.87, py - r * 0.5),
            ])
            painter.drawPolygon(triangle)
        painter.restore()

    def _draw_trade_routes_overlay(
        self,
        painter: QPainter,
        W: float, H: float,
        cx: float, cy: float,
        scale_x: float, scale_y: float,
    ) -> None:
        routes = self._globe.get_feature("trade_routes")  # type: ignore[union-attr]
        if not routes:
            return

        all_flows = np.array([r[2] for r in routes], dtype=np.float64)
        max_flow = float(all_flows.max()) if all_flows.size > 0 else 0.0
        if max_flow <= 0.0:
            return

        flow_threshold = float(np.median(all_flows))
        min_width, max_width = 0.4, 4.0
        painter.setBrush(Qt.BrushStyle.NoBrush)
        active_pen: QPen | None = None

        for start_pos, end_pos, flow in routes:
            if float(flow) < flow_threshold:
                continue
            lon0 = float(self._shift_lon(np.arctan2(start_pos[1], start_pos[0])))
            lat0 = float(np.arcsin(np.clip(start_pos[2], -1.0, 1.0)))
            lon1 = float(self._shift_lon(np.arctan2(end_pos[1], end_pos[0])))
            lat1 = float(np.arcsin(np.clip(end_pos[2], -1.0, 1.0)))

            delta = lon1 - lon0
            if delta > np.pi:
                # lon0 near -π, lon1 near +π — split at antimeridian
                t = (-np.pi - lon0) / (lon1 - 2 * np.pi - lon0)
                lat_x = lat0 + t * (lat1 - lat0)
                lonlat_pairs = [
                    ((lon0, lat0), (-np.pi, lat_x)),
                    ((np.pi, lat_x), (lon1, lat1)),
                ]
            elif delta < -np.pi:
                # lon0 near +π, lon1 near -π — split at antimeridian
                t = (np.pi - lon0) / (lon1 + 2 * np.pi - lon0)
                lat_x = lat0 + t * (lat1 - lat0)
                lonlat_pairs = [
                    ((lon0, lat0), (np.pi, lat_x)),
                    ((-np.pi, lat_x), (lon1, lat1)),
                ]
            else:
                lonlat_pairs = [((lon0, lat0), (lon1, lat1))]

            w = (
                min_width + (max_width - min_width) * (float(flow) / max_flow) ** 0.5
            ) * self._zoom ** 0.4
            if w < 0.3:
                continue

            if active_pen is None or abs(active_pen.widthF() - w) > 0.08:
                pen = QPen(QColor(220, 180, 60, 180))
                pen.setWidthF(w)
                painter.setPen(pen)
                active_pen = pen

            for (a0, b0), (a1, b1) in lonlat_pairs:
                proj = self._projection.project_lonlat(
                    np.array([[a0, b0], [a1, b1]], dtype=np.float32)
                )
                sx = (proj[:, 0] - cx) * scale_x + W * 0.5
                sy = -(proj[:, 1] - cy) * scale_y + H * 0.5
                painter.drawLine(
                    QPointF(float(sx[0]), float(sy[0])),
                    QPointF(float(sx[1]), float(sy[1])),
                )

    def _draw_settlements_overlay(
        self,
        painter: QPainter,
        W: float, H: float,
        cx: float, cy: float,
        scale_x: float, scale_y: float,
    ) -> None:
        from realmgen.placement.types import TIER_COLORS, TIER_RADII, population_display_tier
        feature = self._globe.get_feature("settlements")  # type: ignore[union-attr]
        if not feature:
            return
        positions = np.array([s.position for s in feature], dtype=np.float64)
        pops = np.array([s.population for s in feature], dtype=np.float64)
        max_pop = float(pops.max()) if pops.size > 0 else 0.0
        tiers = np.array(
            [population_display_tier(float(p), max_pop) for p in pops], dtype=np.int32
        )
        lon = self._shift_lon(np.arctan2(positions[:, 1], positions[:, 0]))
        lat = np.arcsin(np.clip(positions[:, 2], -1.0, 1.0))
        proj = self._projection.project_lonlat(np.column_stack([lon, lat]))
        sx = (proj[:, 0] - cx) * scale_x + W * 0.5
        sy = -(proj[:, 1] - cy) * scale_y + H * 0.5
        zoom_scale = max(1.0, self._zoom ** 0.3)
        # Population-scaled radii: sqrt(pop / max_pop) in [0.6, 1.0] relative
        # to the tier base radius so the shape of the distribution is visible.
        max_pop = float(pops.max()) if pops.size > 0 else 0.0
        if max_pop > 0:
            pop_scale = np.sqrt(np.clip(pops / max_pop, 0.0, 1.0)) * 0.4 + 0.6
        else:
            pop_scale = np.ones(len(feature), dtype=np.float64)
        painter.save()
        n_tiers = len(TIER_COLORS)
        for tier_idx in range(n_tiers):
            mask = tiers == tier_idx
            if not mask.any():
                continue
            rgba = TIER_COLORS[tier_idx]
            r_fill = int(rgba[0] * 255)
            g_fill = int(rgba[1] * 255)
            b_fill = int(rgba[2] * 255)
            r_pen = max(0, r_fill - 70)
            g_pen = max(0, g_fill - 70)
            b_pen = max(0, b_fill - 70)
            pen_width = 1.5 if tier_idx == 0 else 1.0
            painter.setPen(QPen(QColor(r_pen, g_pen, b_pen, 230), pen_width))
            painter.setBrush(QBrush(QColor(r_fill, g_fill, b_fill, 220)))
            base_r = TIER_RADII[tier_idx] * zoom_scale
            t_sx = sx[mask]
            t_sy = sy[mask]
            t_scale = pop_scale[mask]
            for i in range(len(t_sx)):
                r = base_r * float(t_scale[i])
                painter.drawEllipse(QRectF(
                    float(t_sx[i]) - r, float(t_sy[i]) - r, 2 * r, 2 * r,
                ))
        painter.restore()

    def _draw_rivers_overlay(
        self,
        painter: QPainter,
        W: float, H: float,
        cx: float, cy: float,
        scale_x: float, scale_y: float,
    ) -> None:
        river_paths = self._globe.get_feature("rivers")  # type: ignore[union-attr]
        if not river_paths:
            return
        painter.setBrush(Qt.BrushStyle.NoBrush)

        max_flow = max(
            (entry[1][-1] for entry in river_paths if len(entry[1]) > 0),
            default=0.0,
        )
        if max_flow <= 0.0:
            return
        min_width, max_width = 0.5, 4.0

        active_pen: QPen | None = None
        for entry in river_paths:
            positions, flows = entry[0], entry[1]
            in_lake = entry[2] if len(entry) > 2 else None
            if len(positions) < 2:
                continue
            lon = self._shift_lon(np.arctan2(positions[:, 1], positions[:, 0]))
            lat = np.arcsin(np.clip(positions[:, 2], -1.0, 1.0))
            lonlat = np.column_stack([lon, lat])
            proj = self._projection.project_lonlat(lonlat)
            sx = (proj[:, 0] - cx) * scale_x + W * 0.5
            sy = -(proj[:, 1] - cy) * scale_y + H * 0.5
            for j in range(1, len(positions)):
                if in_lake is not None and j - 1 < len(in_lake) and in_lake[j - 1]:
                    continue  # segment lies inside a lake — hide it
                if abs(lon[j] - lon[j - 1]) > np.pi:
                    continue
                w = (min_width + (max_width - min_width) * (flows[j - 1] / max_flow) ** 0.5) * self._zoom ** 0.5
                if w <= 1.0:
                    continue
                if active_pen is None or abs(active_pen.widthF() - w) > 0.05:
                    pen = QPen(QColor(100, 180, 255, 200))
                    pen.setWidthF(w)
                    painter.setPen(pen)
                    active_pen = pen
                painter.drawLine(
                    QPointF(float(sx[j - 1]), float(sy[j - 1])),
                    QPointF(float(sx[j]), float(sy[j])),
                )

    def _draw_polity_borders_overlay(
        self,
        painter: QPainter,
        W: float, H: float,
        cx: float, cy: float,
        scale_x: float, scale_y: float,
    ) -> None:
        borders_by_depth = self._globe.get_feature("polity_borders")  # type: ignore[union-attr]
        if not borders_by_depth:
            return

        painter.setBrush(Qt.BrushStyle.NoBrush)

        # Depth style: (line_width, alpha) — fainter at higher depth.
        def _style(depth: int) -> tuple[float, int]:
            if depth == 0:
                return 2.0, 220
            elif depth == 1:
                return 1.2, 140
            else:
                return 0.7, 75

        # Draw deeper (fainter) borders first so sovereign borders render on top.
        for depth in sorted(borders_by_depth.keys(), reverse=True):
            segs: np.ndarray = borders_by_depth[depth]
            if segs.shape[0] == 0:
                continue
            width, alpha = _style(depth)
            pen = QPen(QColor(0, 0, 0, alpha))
            pen.setWidthF(width * self._zoom ** 0.3)
            painter.setPen(pen)

            pos0 = segs[:, :3]
            pos1 = segs[:, 3:]
            lon0 = self._shift_lon(np.arctan2(pos0[:, 1], pos0[:, 0]))
            lat0 = np.arcsin(np.clip(pos0[:, 2], -1.0, 1.0))
            lon1 = self._shift_lon(np.arctan2(pos1[:, 1], pos1[:, 0]))
            lat1 = np.arcsin(np.clip(pos1[:, 2], -1.0, 1.0))

            # Skip segments that cross the antimeridian.
            valid = np.abs(lon1 - lon0) <= np.pi
            if not np.any(valid):
                continue

            lonlat0 = np.column_stack([lon0[valid], lat0[valid]])
            lonlat1 = np.column_stack([lon1[valid], lat1[valid]])
            proj0 = self._projection.project_lonlat(lonlat0)
            proj1 = self._projection.project_lonlat(lonlat1)
            sx0 = (proj0[:, 0] - cx) * scale_x + W * 0.5
            sy0 = -(proj0[:, 1] - cy) * scale_y + H * 0.5
            sx1 = (proj1[:, 0] - cx) * scale_x + W * 0.5
            sy1 = -(proj1[:, 1] - cy) * scale_y + H * 0.5

            for i in range(len(sx0)):
                painter.drawLine(
                    QPointF(float(sx0[i]), float(sy0[i])),
                    QPointF(float(sx1[i]), float(sy1[i])),
                )

    def register_legend(self, scheme_name: str, kind: str, entries: list) -> None:
        """Register legend data for a color scheme.

        kind: 'gradient' or 'categorical'
        entries: list of (label, rgba) where rgba is (r,g,b,a) floats in [0,1].
        For 'gradient', order from lowest value (bottom of bar) to highest (top).
        """
        self._legends[scheme_name] = (kind, entries)
        self.update()

    def _draw_legend(self, painter: QPainter, W: float, H: float) -> None:
        if self._active_scheme is None:
            return
        legend = self._legends.get(self._active_scheme)
        if legend is None:
            return
        kind, entries = legend
        if len(entries) < 2:
            return
        painter.save()
        if kind == "gradient":
            self._draw_gradient_legend(painter, W, H, entries)
        else:
            self._draw_categorical_legend(painter, W, H, entries)
        painter.restore()

    def _draw_gradient_legend(self, painter: QPainter, W: float, H: float, entries: list) -> None:
        font = painter.font()
        font.setPointSizeF(9.0)
        painter.setFont(font)
        fm = painter.fontMetrics()

        margin = 12
        pad = 6
        bar_w = 18
        bar_h = min(220, max(120, int(H * 0.38)))
        lx_gap = 6

        max_lw = max(fm.horizontalAdvance(lbl) for lbl, _ in entries)

        x0 = margin + pad
        y_bot = int(H) - margin - pad
        y_top = y_bot - bar_h

        bg_w = pad + bar_w + lx_gap + max_lw + pad
        bg_h = pad + bar_h + pad
        painter.fillRect(margin, y_top - pad, bg_w, bg_h, QColor(20, 20, 20, 160))

        gradient = QLinearGradient(x0, y_top, x0, y_bot)
        n = len(entries)
        for i, (_, rgba) in enumerate(entries):
            # entries[0] = lowest = bottom (pos 1.0); entries[-1] = highest = top (pos 0.0)
            pos = 1.0 - i / (n - 1)
            gradient.setColorAt(pos, QColor(int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255), 255))
        painter.fillRect(QRectF(x0, y_top, bar_w, bar_h), QBrush(gradient))
        painter.setPen(QPen(QColor(120, 120, 120, 180), 0.5))
        painter.drawRect(QRectF(x0, y_top, bar_w, bar_h))

        painter.setPen(QColor(240, 240, 240))
        for i, (label, _) in enumerate(entries):
            frac = i / (n - 1)
            cy = y_bot - frac * bar_h
            ty = int(cy + (fm.ascent() - fm.descent()) // 2)
            painter.drawText(int(x0 + bar_w + lx_gap), ty, label)

    def _draw_categorical_legend(self, painter: QPainter, W: float, H: float, entries: list) -> None:
        font = painter.font()
        font.setPointSizeF(9.0)
        painter.setFont(font)
        fm = painter.fontMetrics()

        margin = 12
        pad = 6
        swatch = 13
        swatch_gap = 5
        row_h = max(swatch, fm.height()) + 3

        max_lw = max(fm.horizontalAdvance(lbl) for lbl, _ in entries)
        total_h = pad + len(entries) * row_h + pad
        total_w = pad + swatch + swatch_gap + max_lw + pad

        x0 = margin
        y0 = int(H) - margin - total_h
        painter.fillRect(x0, y0, total_w, total_h, QColor(20, 20, 20, 160))

        for i, (label, rgba) in enumerate(entries):
            row_y = y0 + pad + i * row_h
            sw_x = x0 + pad
            sw_y = row_y + (row_h - swatch) // 2
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255))))
            painter.drawRect(sw_x, sw_y, swatch, swatch)
            painter.setPen(QColor(240, 240, 240))
            cy = row_y + row_h / 2
            ty = int(cy + (fm.ascent() - fm.descent()) // 2)
            painter.drawText(x0 + pad + swatch + swatch_gap, ty, label)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = (float(event.position().x()), float(event.position().y()))
            self._drag_start_center = self._center.copy()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def set_hover_callback(
        self, cb: Optional[Callable[[Optional[int], Optional[np.ndarray]], None]]
    ) -> None:
        self._hover_callback = cb

    def _hit_face(self, mx: float, my: float) -> tuple[Optional[int], Optional[np.ndarray]]:
        """Return (face_src, face_uv) for the face under screen point (mx, my)."""
        if not self._proj_faces or self._face_bbox_min.size == 0:
            return None, None
        W, H = float(self.width()), float(self.height())
        scale_x = self._zoom * 0.5 * W
        scale_y = self._zoom * 0.5 * H
        cx, cy = self._center[0], self._center[1]
        mu = (mx - W * 0.5) / scale_x + cx
        mv = -(my - H * 0.5) / scale_y + cy

        # Bounding-box prefilter (vectorised) before per-polygon PIP test.
        candidates = np.where(
            (self._face_bbox_min[:, 0] <= mu) & (self._face_bbox_max[:, 0] >= mu)
            & (self._face_bbox_min[:, 1] <= mv) & (self._face_bbox_max[:, 1] >= mv)
        )[0]

        hit_src: Optional[int] = None
        hit_uv: Optional[np.ndarray] = None
        for idx in candidates:
            poly = self._proj_faces[idx]
            if not _pip_polygon(mu, mv, poly):
                continue
            hit_src = self._face_src[idx]
            if self._face_uvs:
                hit_uv = _bary_in_fan(mu, mv, poly, self._face_uvs[idx])
        return hit_src, hit_uv

    def mouseMoveEvent(self, event) -> None:
        pos = event.position()
        mx, my = float(pos.x()), float(pos.y())
        if self._drag_start is not None:
            W, H = float(self.width()), float(self.height())
            dx = mx - self._drag_start[0]
            dy = my - self._drag_start[1]
            scale_x = self._zoom * 0.5 * W
            scale_y = self._zoom * 0.5 * H
            self._center[0] = self._drag_start_center[0] - dx / scale_x
            self._center[1] = self._drag_start_center[1] + dy / scale_y
            self._clamp_center()
            self.update()
        if self._hover_callback is not None:
            face_src, uv = self._hit_face(mx, my)
            self._hover_callback(face_src, uv)

    def leaveEvent(self, event) -> None:
        if self._hover_callback is not None:
            self._hover_callback(None, None)

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

        self._save_btn = QPushButton("Save…")
        self._save_btn.setVisible(False)
        self._has_save_callback = False
        toolbar_layout.addWidget(self._save_btn)

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

        self._overlays_btn = QPushButton("Overlays")
        self._overlays_menu = QMenu(self._overlays_btn)
        self._overlays_btn.setMenu(self._overlays_menu)
        self._overlays_btn.setVisible(False)
        self._overlay_actions: dict[str, QAction] = {}
        toolbar_layout.addWidget(self._overlays_btn)

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
        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        bottom_layout.addWidget(self._status_label)
        bottom_layout.addStretch()
        self._params_btn = QPushButton("Parameters…")
        self._params_btn.setVisible(False)
        self._has_params_callback = False
        bottom_layout.addWidget(self._params_btn)
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

    def set_globe(self, globe: "World") -> None:
        self._map.set_globe(globe)

    def set_status(self, text: str) -> None:
        self._status_label.setText(text)

    def set_rebuild_enabled(self, enabled: bool) -> None:
        self._rerun_btn.setEnabled(enabled)
        self._params_btn.setEnabled(enabled)
        self._save_btn.setEnabled(enabled)

    def set_save_callback(self, callback: Callable[[], None]) -> None:
        if self._has_save_callback:
            self._save_btn.clicked.disconnect()
        self._save_btn.clicked.connect(callback)
        self._has_save_callback = True
        self._save_btn.setVisible(True)

    def set_params_callback(self, callback: Callable[[], None]) -> None:
        if self._has_params_callback:
            self._params_btn.clicked.disconnect()
        self._params_btn.clicked.connect(callback)
        self._has_params_callback = True
        self._params_btn.setVisible(True)

    def set_rebuild_callback(self, callback: Callable[[], None]) -> None:
        if self._has_rebuild_callback:
            self._rerun_btn.clicked.disconnect()
        self._rerun_btn.clicked.connect(callback)
        self._has_rebuild_callback = True
        self._rerun_btn.setVisible(True)

    def associate_scheme_texture(self, scheme_name: str, texture_key: str) -> None:
        self._map.associate_scheme_texture(scheme_name, texture_key)

    def register_legend(self, scheme_name: str, kind: str, entries: list) -> None:
        self._map.register_legend(scheme_name, kind, entries)

    def set_hover_callback(
        self, cb: Optional[Callable[[Optional[int], Optional[np.ndarray]], None]]
    ) -> None:
        self._map.set_hover_callback(cb)

    def add_overlay(self, name: str, label: Optional[str] = None, checked: bool = True) -> None:
        """Register a named overlay toggle. Safe to call on every re-run (idempotent)."""
        if name in self._overlay_actions:
            self._map.set_overlay_visible(name, self._overlay_actions[name].isChecked())
            return
        action = QAction(label or name.title(), self._overlays_menu)
        action.setCheckable(True)
        action.setChecked(checked)
        action.toggled.connect(lambda v, n=name: self._map.set_overlay_visible(n, v))
        self._overlays_menu.addAction(action)
        self._overlay_actions[name] = action
        self._overlays_btn.setVisible(True)
        self._map.set_overlay_visible(name, checked)

    @property
    def active_scheme(self) -> "str | None":
        return self._map._active_scheme

    def set_active_scheme(self, name: str) -> None:
        """Switch to a named scheme if it exists in the current scheme list."""
        if name and name in self._map._color_schemes:
            self._color_combo.setCurrentText(name)

    def set_color_schemes(self, schemes: dict[str, Callable[[dict], Sequence[float]]]) -> None:
        prev_scheme = self._color_combo.currentText()
        self._color_combo.blockSignals(True)
        self._color_combo.clear()
        for name in schemes:
            self._color_combo.addItem(name)
        self._color_combo.blockSignals(False)
        visible = bool(schemes)
        self._color_label.setVisible(visible)
        self._color_combo.setVisible(visible)
        self._map.set_color_schemes(schemes)
        if prev_scheme and prev_scheme in schemes:
            self._color_combo.setCurrentText(prev_scheme)
