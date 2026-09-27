from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Sequence

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QBrush, QColor, QImage, QPainter, QPen, QPixmap, QPolygonF, QTransform
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from core.antimeridian import find_ideal_antimeridian
from core.display import LocalHoverFn, SchemeSpec, TextureRecipe
from region.display import describe_texel
from region.geometry import RegionGeometry, cell_border_polylines, ecef_to_cell
from viz.mapview.legend import draw_legend
from viz.mapview.overlays import draw_overlays
from viz.mapview.projections import ALL_PROJECTIONS, RobinsonProjection, Projection
from viz.mapview.regionview import RegionView

# (face, uv, local) -> status fragments; see ``MapWindow.set_hover_parts``.
HoverPartsFn = Callable[[int, Optional[np.ndarray], Optional[LocalHoverFn]], list[str]]

if TYPE_CHECKING:
    from core.mesh import PolygonMesh
    from core.world import World
    from realmgen.builder import RealmParams
    from worldgen.builder import GlobeParams
    from localmap.params import LocalParams
    from region.params import RegionParams


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


class _WorldProjector:
    """Overlay projector for the world map: map projection + antimeridian split."""

    def __init__(self, widget: "MapWidget", W: float, H: float) -> None:
        self._w = widget
        self._W, self._H = W, H
        self._scale_x = widget._zoom * 0.5 * W
        self._scale_y = widget._zoom * 0.5 * H

    @property
    def zoom(self) -> float:
        return self._w._zoom

    @property
    def regional(self) -> bool:
        return False

    def on_land(self, key: str, ecef: np.ndarray) -> np.ndarray:
        return ecef

    def _lonlat(self, ecef: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ecef = np.asarray(ecef).reshape(-1, 3)
        lon = self._w._shift_lon(np.arctan2(ecef[:, 1], ecef[:, 0]))
        lat = np.arcsin(np.clip(ecef[:, 2], -1.0, 1.0))
        return lon, lat

    def _screen(self, lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        uv = self._w._projection.project_lonlat(np.column_stack([lon, lat]))
        cx, cy = self._w._center
        sx = (uv[:, 0] - cx) * self._scale_x + self._W * 0.5
        sy = -(uv[:, 1] - cy) * self._scale_y + self._H * 0.5
        return sx, sy

    def points(self, ecef: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sx, sy = self._screen(*self._lonlat(ecef))
        return sx, sy, np.ones(len(sx), dtype=bool)

    def segments(
        self, p0: np.ndarray, p1: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        lon0, lat0 = self._lonlat(p0)
        lon1, lat1 = self._lonlat(p1)
        sx0, sy0 = self._screen(lon0, lat0)
        sx1, sy1 = self._screen(lon1, lat1)
        return sx0, sy0, sx1, sy1, np.abs(lon1 - lon0) <= np.pi

    def split_segment(self, p0: np.ndarray, p1: np.ndarray) -> list[tuple[float, float, float, float]]:
        (lon0,), (lat0,) = self._lonlat(p0)
        (lon1,), (lat1,) = self._lonlat(p1)
        delta = lon1 - lon0
        if delta > np.pi:
            # lon0 near -π, lon1 near +π — split at the antimeridian.
            t = (-np.pi - lon0) / (lon1 - 2 * np.pi - lon0)
            lat_x = lat0 + t * (lat1 - lat0)
            pairs = [((lon0, lat0), (-np.pi, lat_x)), ((np.pi, lat_x), (lon1, lat1))]
        elif delta < -np.pi:
            t = (np.pi - lon0) / (lon1 + 2 * np.pi - lon0)
            lat_x = lat0 + t * (lat1 - lat0)
            pairs = [((lon0, lat0), (np.pi, lat_x)), ((-np.pi, lat_x), (lon1, lat1))]
        else:
            pairs = [((lon0, lat0), (lon1, lat1))]
        pieces = []
        for (a0, b0), (a1, b1) in pairs:
            sx, sy = self._screen(np.array([a0, a1]), np.array([b0, b1]))
            pieces.append((float(sx[0]), float(sy[0]), float(sx[1]), float(sy[1])))
        return pieces

    def polygons(self, verts: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        return [
            self._screen(lonlat[:, 0], lonlat[:, 1])
            for lonlat in _split_at_antimeridian(verts, cut_lon=self._w._antimeridian)
        ]


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
        self._region_click_callback: Optional[Callable[[int], None]] = None

        self._zoom: float = 1.0
        self._center: np.ndarray = np.zeros(2, dtype=np.float64)
        self._drag_start: Optional[tuple[float, float]] = None
        self._drag_start_center: np.ndarray = np.zeros(2, dtype=np.float64)
        self.setFocusPolicy(Qt.FocusPolicy.WheelFocus)
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.setMouseTracking(True)

        self._region_mode: bool = False
        self._hover_region: Optional[int] = None
        self._region_face_cells: Optional[np.ndarray] = None
        self._region_border_polylines: list[np.ndarray] = []  # each (N, 3) ECEF

        # Cached render of the map body (faces + overlays). Repainting all textured faces is
        # slow, so hover-only changes (region tint) just re-blit this and redraw on top.
        self._base_cache: Optional[QPixmap] = None
        self._base_cache_key: Optional[tuple] = None
        # View-independent face geometry + brushes in projected space; rebuilt lazily.
        self._draw_items: Optional[list[list[tuple[QPolygonF, QBrush]]]] = None

    def _shift_lon(self, lon: np.ndarray) -> np.ndarray:
        """Shift longitudes into the frame where self._antimeridian maps to ±π."""
        return ((lon - self._antimeridian) % (2.0 * np.pi)) - np.pi

    def set_mesh(self, mesh: PolygonMesh, antimeridian: Optional[float] = None) -> None:
        self._mesh = mesh
        self._antimeridian = antimeridian if antimeridian is not None else find_ideal_antimeridian(mesh)
        self._rebuild_faces()
        self._rebuild_textures()
        self._rebuild_colors()
        self._reproject()
        self._region_face_cells = None
        self._region_border_polylines = []
        if self._region_mode:
            self._rebuild_region_cells()
        self._invalidate_base()

    def set_globe(self, globe: "World") -> None:
        self._globe = globe
        self.set_mesh(globe.mesh, antimeridian=globe.antimeridian)

    @property
    def globe(self) -> Optional["World"]:
        return self._globe

    @property
    def antimeridian(self) -> float:
        return self._antimeridian

    def set_overlay_visible(self, name: str, visible: bool) -> None:
        self._overlay_visible[name] = visible
        self._invalidate_base()

    def set_projection(self, proj: Projection) -> None:
        self._projection = proj
        self._reproject()
        self._invalidate_base()

    def set_color_schemes(self, schemes: dict[str, Callable[[dict], Sequence[float]]]) -> None:
        self._color_schemes = schemes
        self._active_scheme = next(iter(schemes)) if schemes else None
        self._rebuild_colors()
        self._invalidate_base()

    def set_color_scheme(self, name: str) -> None:
        self._active_scheme = name
        self._rebuild_faces()
        self._rebuild_textures()
        self._rebuild_colors()
        self._reproject()
        self._invalidate_base()

    def associate_scheme_texture(self, scheme_name: str, texture_key: str) -> None:
        """Associate a named face texture with a color scheme for per-pixel rendering."""
        self._scheme_texture_key[scheme_name] = texture_key
        if self._active_scheme == scheme_name:
            self._rebuild_faces()
            self._rebuild_textures()
            self._reproject()
            self._invalidate_base()

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
        self._draw_items = None
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
        self._draw_items = None
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
        self._draw_items = None
        self._proj_faces = [
            self._projection.project_lonlat(poly) for poly in self._faces_lonlat
        ]
        if self._proj_faces:
            self._face_bbox_min = np.array([p.min(axis=0) for p in self._proj_faces], dtype=np.float64)
            self._face_bbox_max = np.array([p.max(axis=0) for p in self._proj_faces], dtype=np.float64)
        else:
            self._face_bbox_min = np.empty((0, 2), dtype=np.float64)
            self._face_bbox_max = np.empty((0, 2), dtype=np.float64)

    def _build_draw_items(self) -> list[list[tuple[QPolygonF, QBrush]]]:
        """Per-polygon (polygon, brush) pairs in projected (u, v) space, independent of the view.

        Textured faces are split into fan triangles, each with a brush transform mapping
        image pixels → projected coordinates; flat faces get one solid-colour polygon.
        """
        items: list[list[tuple[QPolygonF, QBrush]]] = [[] for _ in self._proj_faces]

        # Gather every textured fan triangle so the affine solves run as one batch.
        tri_poly: list[int] = []
        tri_img: list[np.ndarray] = []   # (3, 2) image-pixel coords
        tri_proj: list[np.ndarray] = []  # (3, 2) projected coords
        for idx, poly_proj in enumerate(self._proj_faces):
            face_src = self._face_src[idx]
            qimage = self._texture_qimages.get(face_src)
            if qimage is None:
                color = self._face_colors[idx]
                brush = QBrush(QColor.fromRgbF(
                    float(color[0]), float(color[1]), float(color[2]), float(color[3])
                ))
                polygon = QPolygonF([QPointF(float(u), float(v)) for u, v in poly_proj])
                items[idx].append((polygon, brush))
                continue
            img = self._face_uvs[idx] * np.array([qimage.width(), qimage.height()], dtype=np.float64)
            for k in range(1, len(poly_proj) - 1):
                tri = [0, k, k + 1]
                tri_poly.append(idx)
                tri_img.append(img[tri])
                tri_proj.append(poly_proj[tri])

        if tri_poly:
            img_pts = np.array(tri_img, dtype=np.float64)                        # (T, 3, 2)
            A = np.concatenate([img_pts, np.ones((len(tri_poly), 3, 1))], axis=2)  # (T, 3, 3)
            B = np.array(tri_proj, dtype=np.float64)                             # (T, 3, 2)
            ok = np.abs(np.linalg.det(A)) >= 1e-8
            X = np.zeros((len(tri_poly), 3, 2))
            X[ok] = np.linalg.solve(A[ok], B[ok])
            for t in np.nonzero(ok)[0]:
                idx = tri_poly[t]
                brush = QBrush(self._texture_qimages[self._face_src[idx]])
                x = X[t]
                brush.setTransform(QTransform(
                    float(x[0, 0]), float(x[0, 1]),
                    float(x[1, 0]), float(x[1, 1]),
                    float(x[2, 0]), float(x[2, 1]),
                ))
                polygon = QPolygonF([QPointF(float(u), float(v)) for u, v in B[t]])
                items[idx].append((polygon, brush))
        return items

    def _invalidate_base(self) -> None:
        """Mark the cached map body stale (data changed) and schedule a repaint."""
        self._base_cache = None
        self.update()

    def paintEvent(self, _) -> None:
        W, H = float(self.width()), float(self.height())
        dpr = self.devicePixelRatioF()
        key = (self.width(), self.height(), dpr, self._zoom, float(self._center[0]), float(self._center[1]))
        if self._base_cache is None or self._base_cache_key != key:
            pixmap = QPixmap(max(1, round(W * dpr)), max(1, round(H * dpr)))
            pixmap.setDevicePixelRatio(dpr)
            base_painter = QPainter(pixmap)
            self._paint_base(base_painter, W, H)
            base_painter.end()
            self._base_cache = pixmap
            self._base_cache_key = key

        painter = QPainter(self)
        painter.drawPixmap(0, 0, self._base_cache)
        if not self._proj_faces:
            return

        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self._region_mode:
            cx, cy = self._center[0], self._center[1]
            self._draw_region_overlay(painter, W, H, cx, cy, self._zoom * 0.5 * W, self._zoom * 0.5 * H)

        self._draw_legend(painter, W, H)

    def _paint_base(self, painter: QPainter, W: float, H: float) -> None:
        """Draw the ocean background, all faces, and world overlays."""
        painter.fillRect(0, 0, int(W) + 1, int(H) + 1, QColor(30, 100, 180))

        if not self._proj_faces:
            return

        cx, cy = self._center[0], self._center[1]
        scale_x = self._zoom * 0.5 * W
        scale_y = self._zoom * 0.5 * H

        if self._draw_items is None:
            self._draw_items = self._build_draw_items()

        # Draw items live in projected (u, v) space; one world transform maps them to the screen.
        # QTransform(m11,m12,m21,m22,dx,dy): x'=m11*x+m21*y+dx, y'=m12*x+m22*y+dy
        painter.save()
        painter.setTransform(QTransform(
            scale_x, 0.0, 0.0, -scale_y, W * 0.5 - cx * scale_x, H * 0.5 + cy * scale_y,
        ))
        painter.setPen(Qt.PenStyle.NoPen)
        half_u, half_v = W * 0.5 / scale_x, H * 0.5 / scale_y
        visible = np.nonzero(
            (self._face_bbox_max[:, 0] >= cx - half_u) & (self._face_bbox_min[:, 0] <= cx + half_u)
            & (self._face_bbox_max[:, 1] >= cy - half_v) & (self._face_bbox_min[:, 1] <= cy + half_v)
        )[0]
        for idx in visible:
            for polygon, brush in self._draw_items[idx]:
                painter.setBrush(brush)
                painter.drawPolygon(polygon)
        painter.restore()

        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self._globe is not None:
            draw_overlays(painter, self._globe, self._overlay_visible, _WorldProjector(self, W, H))

    def register_legend(self, scheme_name: str, kind: str, entries: list) -> None:
        """Register legend data for a color scheme.

        kind: 'gradient' or 'categorical'
        entries: list of (label, rgba) where rgba is (r,g,b,a) floats in [0,1].
        For 'gradient', order from lowest value (bottom of bar) to highest (top).
        """
        self._legends[scheme_name] = (kind, entries)
        self._invalidate_base()

    def _draw_legend(self, painter: QPainter, W: float, H: float) -> None:
        if self._active_scheme is None:
            return
        legend = self._legends.get(self._active_scheme)
        if legend is None:
            return
        kind, entries = legend
        margin = 12
        draw_legend(painter, kind, entries, margin, H - margin, H)

    def legend_for(self, scheme_name: Optional[str]) -> Optional[tuple[str, list]]:
        """The ``(kind, entries)`` legend registered for a scheme, if any."""
        return self._legends.get(scheme_name) if scheme_name is not None else None

    def set_region_mode(self, enabled: bool) -> None:
        self._region_mode = enabled
        if enabled and self._region_face_cells is None and self._mesh is not None:
            self._rebuild_region_cells()
        if not enabled:
            self._hover_region = None
        self.update()

    def _rebuild_region_cells(self) -> None:
        if self._mesh is None:
            return
        centroids = self._mesh.face_centroids.astype(np.float64)  # (N, 3)
        self._region_face_cells = ecef_to_cell(centroids, self._antimeridian)
        self._region_border_polylines = cell_border_polylines(self._antimeridian)

    def _region_at_screen(self, mx: float, my: float) -> Optional[int]:
        """Return the region cell index at screen position, or None if off-map."""
        W, H = float(self.width()), float(self.height())
        scale_x = self._zoom * 0.5 * W
        scale_y = self._zoom * 0.5 * H
        cx, cy = self._center[0], self._center[1]
        mu = (mx - W * 0.5) / scale_x + cx
        mv = -(my - H * 0.5) / scale_y + cy
        lonlat = self._projection.unproject_lonlat(np.array([[mu, mv]], dtype=np.float64))
        if np.any(np.isnan(lonlat)):
            return None
        shifted_lon = float(lonlat[0, 0])
        lat = float(lonlat[0, 1])
        # _shift_lon places the antimeridian at ±π and the map center at lon = ant + π,
        # so the original ECEF longitude is lon = shifted_lon + ant + π.
        lon = shifted_lon + self._antimeridian + np.pi
        cos_lat = np.cos(lat)
        ecef = np.array([[cos_lat * np.cos(lon), cos_lat * np.sin(lon), np.sin(lat)]])
        return int(ecef_to_cell(ecef, self._antimeridian)[0])

    def _draw_region_overlay(
        self,
        painter: QPainter,
        W: float, H: float,
        cx: float, cy: float,
        scale_x: float, scale_y: float,
    ) -> None:
        if self._region_face_cells is None or self._mesh is None:
            return

        # Blue tint over all mesh polygons in the hovered region.
        if self._hover_region is not None:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(80, 140, 255, 80)))
            for idx, poly_proj in enumerate(self._proj_faces):
                if int(self._region_face_cells[self._face_src[idx]]) == self._hover_region:
                    px = (poly_proj[:, 0] - cx) * scale_x + W * 0.5
                    py = -(poly_proj[:, 1] - cy) * scale_y + H * 0.5
                    painter.drawPolygon(
                        QPolygonF([QPointF(float(xx), float(yy)) for xx, yy in zip(px, py)])
                    )

        # Mathematically-defined border polylines (ECEF points, projected each frame).
        if not self._region_border_polylines:
            return
        pen = QPen(QColor(0, 0, 0, 200))
        pen.setWidthF(1.5)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for seg in self._region_border_polylines:
            lon = self._shift_lon(np.arctan2(seg[:, 1], seg[:, 0]))
            lat = np.arcsin(np.clip(seg[:, 2], -1.0, 1.0))
            proj = self._projection.project_lonlat(np.column_stack([lon, lat]))
            sx = (proj[:, 0] - cx) * scale_x + W * 0.5
            sy = -(proj[:, 1] - cy) * scale_y + H * 0.5
            for j in range(len(sx) - 1):
                if abs(lon[j + 1] - lon[j]) > np.pi:
                    continue
                painter.drawLine(
                    QPointF(float(sx[j]), float(sy[j])),
                    QPointF(float(sx[j + 1]), float(sy[j + 1])),
                )

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = (float(event.position().x()), float(event.position().y()))
            self._drag_start_center = self._center.copy()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def set_hover_callback(
        self, cb: Optional[Callable[[Optional[int], Optional[np.ndarray]], None]]
    ) -> None:
        self._hover_callback = cb

    def set_region_click_callback(self, cb: Optional[Callable[[int], None]]) -> None:
        """Callback invoked with a region cell index when one is clicked in region mode."""
        self._region_click_callback = cb

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
        if self._region_mode:
            new_region = self._region_at_screen(mx, my)
            if new_region != self._hover_region:
                self._hover_region = new_region
                self.update()

    def leaveEvent(self, event) -> None:
        if self._hover_callback is not None:
            self._hover_callback(None, None)
        if self._region_mode and self._hover_region is not None:
            self._hover_region = None
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position()
            mx, my = float(pos.x()), float(pos.y())
            # A negligible-movement press-release in region mode is a click, not a drag.
            if (
                self._region_mode
                and self._region_click_callback is not None
                and self._drag_start is not None
                and abs(mx - self._drag_start[0]) < 4.0
                and abs(my - self._drag_start[1]) < 4.0
            ):
                region = self._region_at_screen(mx, my)
                if region is not None:
                    self._region_click_callback(region)
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
    def __init__(self, mesh: Optional[PolygonMesh] = None, debug: bool = False) -> None:
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

        self._proj_label = QLabel("Projection:")
        self._proj_label.setVisible(debug)
        toolbar_layout.addWidget(self._proj_label)
        self._combo = QComboBox()
        for proj in ALL_PROJECTIONS:
            self._combo.addItem(proj.name)
        self._combo.currentIndexChanged.connect(
            lambda i: self._map.set_projection(ALL_PROJECTIONS[i])
        )
        self._combo.setVisible(debug)
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

        self._region_btn = QPushButton("Regions")
        self._region_btn.setCheckable(True)
        self._region_btn.setChecked(False)
        toolbar_layout.addWidget(self._region_btn)

        self._back_btn = QPushButton("← World Map")
        self._back_btn.setVisible(False)
        toolbar_layout.addWidget(self._back_btn)

        # In a region: click a hex to open its local map.
        self._local_btn = QPushButton("Local maps")
        self._local_btn.setCheckable(True)
        self._local_btn.setChecked(False)
        self._local_btn.setVisible(False)
        toolbar_layout.addWidget(self._local_btn)

        toolbar_layout.addStretch()

        zoom_out_btn = QPushButton("−")
        zoom_out_btn.setFixedWidth(30)
        zoom_out_btn.clicked.connect(lambda: self._active_view().zoom_out())
        toolbar_layout.addWidget(zoom_out_btn)

        zoom_in_btn = QPushButton("+")
        zoom_in_btn.setFixedWidth(30)
        zoom_in_btn.clicked.connect(lambda: self._active_view().zoom_in())
        toolbar_layout.addWidget(zoom_in_btn)

        reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(lambda: self._active_view().reset_view())
        toolbar_layout.addWidget(reset_btn)

        bottom_toolbar = QWidget()
        bottom_layout = QHBoxLayout(bottom_toolbar)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        self._prev_status = ""
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
        self._region_view = RegionView()
        self._color_combo.currentTextChanged.connect(self._on_scheme_changed)
        self._hover_parts: Optional[HoverPartsFn] = None
        # Scheme → the recipe its texture was built with; the region view reruns
        # it on refined data.  The world map only needs the finished texture.
        self._scheme_recipes: dict[str, TextureRecipe] = {}
        self._region_view.set_hover_callback(lambda texel: self._zoomed_hover(self._region_view, texel))
        self._region_btn.toggled.connect(self._map.set_region_mode)
        self._map.set_region_click_callback(self._enter_region)
        self._back_btn.clicked.connect(self._go_back)
        # A second view for local maps, so going back shows the region as it was.
        self._local_view = RegionView()
        self._local_view.set_hover_callback(lambda texel: self._zoomed_hover(self._local_view, texel))
        self._local_btn.toggled.connect(self._region_view.set_local_pick)
        self._region_view.set_hex_click_callback(self._enter_local)

        self._stack = QStackedWidget()
        self._stack.addWidget(self._map)          # index 0: world map
        self._stack.addWidget(self._region_view)  # index 1: zoomed region
        self._stack.addWidget(self._local_view)   # index 2: a regional hex's local map

        layout.addWidget(toolbar)
        layout.addWidget(self._stack, stretch=1)
        layout.addWidget(bottom_toolbar)
        self.setCentralWidget(central)

        if mesh is not None:
            self.set_mesh(mesh)

    def set_mesh(self, mesh: PolygonMesh) -> None:
        self._map.set_mesh(mesh)

    def set_globe(self, globe: "World") -> None:
        # A re-run replaces the world under our feet; any region on screen is now
        # stale, so drop back to the world map rather than show old detail.
        if self._stack.currentWidget() is self._local_view:
            self._exit_local()
        if self._stack.currentWidget() is self._region_view:
            self._exit_region()
        self._local_view.clear_local_cache()
        self._map.set_globe(globe)

    def set_realm_params(self, params: "RealmParams") -> None:
        """The realm tuning the world was built with; the region view re-sites towns with it."""
        self._region_view.set_realm_params(params)
        self._local_view.set_realm_params(params)

    def set_world_params(self, params: "GlobeParams") -> None:
        """The physical tuning the world was built with; the region view refines
        resources and re-routes roads with it."""
        self._region_view.set_world_params(params)
        self._local_view.set_world_params(params)

    def set_region_params(self, params: "RegionParams") -> None:
        """Tune the region tier; the region on screen (if any) is refined again,
        and so is a local map on screen (it is built from the region)."""
        local = self._stack.currentWidget() is self._local_view
        showing = local or self._stack.currentWidget() is self._region_view
        if showing:
            self.set_status("Refining region…")
            QApplication.setOverrideCursor(Qt.CursorShape.BusyCursor)
        try:
            self._region_view.set_params(params, self._map.globe)
            # Local maps are built from the region's rivers and roads too, which
            # a region-params change can rebuild without new fields.
            self._local_view.clear_local_cache()
            if local:
                self._reopen_local()
        finally:
            if showing:
                QApplication.restoreOverrideCursor()
                self.set_status((self._local_view if local else self._region_view).status_text())

    def set_local_params(self, params: "LocalParams") -> None:
        """Tune the local tier; the local map on screen (if any) is rebuilt."""
        self._local_view.set_local_params(params)
        if self._stack.currentWidget() is self._local_view:
            self.set_status("Building local map…")
            QApplication.setOverrideCursor(Qt.CursorShape.BusyCursor)
            try:
                self._reopen_local()
            finally:
                QApplication.restoreOverrideCursor()
                self.set_status(self._local_view.status_text())

    def _reopen_local(self) -> None:
        """Rebuild the local map on screen from the region view's current data."""
        local = self._local_view.local_map
        if local is not None:
            self._local_view.set_local(self._region_view, local.hex_index, keep_view=True)

    def _enter_region(self, region_id: int) -> None:
        """Switch from the world map to the zoomed-in view of a clicked region.

        Refinement runs synchronously — it is a few hundred thousand samples, well
        under a second — behind a busy cursor.
        """
        geometry = RegionGeometry(region_id, self._map.antimeridian)
        prev_status = self._status_label.text()
        self.set_status(f"Refining region {region_id}…")
        QApplication.setOverrideCursor(Qt.CursorShape.BusyCursor)
        try:
            self._sync_region_layer()
            self._region_view.set_region(geometry, self._map.globe)
        finally:
            QApplication.restoreOverrideCursor()
        self._prev_status = prev_status
        self.set_status(self._region_view.status_text())
        self._stack.setCurrentWidget(self._region_view)
        self._region_btn.setVisible(False)
        self._back_btn.setText("← World Map")
        self._back_btn.setVisible(True)
        self._local_btn.setVisible(True)

    def _enter_local(self, hex_index: int) -> None:
        """Open the local map of a regional hex clicked in local-maps mode."""
        # Villages only exist on local maps; their toggle appears with the first one.
        self.add_overlay("villages", "Villages")
        self.set_status(f"Building local map of hex {hex_index}…")
        QApplication.setOverrideCursor(Qt.CursorShape.BusyCursor)
        try:
            self._sync_region_layer(self._local_view)
            self._local_view.set_local(self._region_view, hex_index)
        finally:
            QApplication.restoreOverrideCursor()
        self.set_status(self._local_view.status_text())
        self._stack.setCurrentWidget(self._local_view)
        self._back_btn.setText("← Region")
        self._local_btn.setVisible(False)

    def _active_view(self) -> "MapWidget | RegionView":
        """Whichever view is on screen — the zoom buttons act on it."""
        current = self._stack.currentWidget()
        if current is self._region_view or current is self._local_view:
            return current  # type: ignore[return-value]
        return self._map

    def _go_back(self) -> None:
        """One tier up: local map → region → world map."""
        if self._stack.currentWidget() is self._local_view:
            self._exit_local()
        else:
            self._exit_region()

    def _exit_local(self) -> None:
        """Return from a local map to the region it was opened from."""
        self._stack.setCurrentWidget(self._region_view)
        self._local_view.clear_region()
        self._back_btn.setText("← World Map")
        self._local_btn.setVisible(True)
        self.set_status(self._region_view.status_text())

    def _exit_region(self) -> None:
        """Return from the zoomed-in region view to the world map."""
        self._stack.setCurrentWidget(self._map)
        self._back_btn.setVisible(False)
        self._local_btn.setVisible(False)
        self._region_btn.setVisible(True)
        self._region_view.clear_region()
        self.set_status(self._prev_status)

    def _on_scheme_changed(self, name: str) -> None:
        """One Coloration combo drives both views, so the layer survives switching."""
        self._map.set_color_scheme(name)
        current = self._stack.currentWidget()
        if current is self._region_view or current is self._local_view:
            QApplication.setOverrideCursor(Qt.CursorShape.BusyCursor)
            try:
                self._sync_region_layer(current)  # type: ignore[arg-type]
            finally:
                QApplication.restoreOverrideCursor()

    def _sync_region_layer(self, view: Optional[RegionView] = None) -> None:
        """Hand the world map's active scheme to a zoomed view (default: the region's)."""
        view = view if view is not None else self._region_view
        name = self._map._active_scheme
        fn = self._map._color_schemes.get(name) if name is not None else None
        spec = None
        if name is not None and fn is not None:
            spec = SchemeSpec(
                fn, texture=self._map._scheme_texture_key.get(name),
                recipe=self._scheme_recipes.get(name),
            )
        view.set_layer(name, spec)
        view.set_legend(self._map.legend_for(name))

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

    def associate_scheme_recipe(self, scheme_name: str, recipe: TextureRecipe) -> None:
        """How a scheme's texture is made, so the region view can remake it from refined data."""
        self._scheme_recipes[scheme_name] = recipe

    def register_legend(self, scheme_name: str, kind: str, entries: list) -> None:
        self._map.register_legend(scheme_name, kind, entries)

    def set_hover_parts(self, fn: Optional[HoverPartsFn]) -> None:
        """Status-bar fragments for a hovered ``(face, uv)`` — shared by both views.

        ``fn(face, uv, local)``: the region view passes ``local`` to substitute its
        refined values; everything it has not refined comes from the world face.
        """
        self._hover_parts = fn
        self._map.set_hover_callback(self._world_hover if fn is not None else None)

    def _world_hover(self, face: Optional[int], uv: Optional[np.ndarray]) -> None:
        if face is None or self._hover_parts is None:
            self.set_status("")
            return
        self.set_status("  |  ".join(self._hover_parts(face, uv, None)))

    def _zoomed_hover(self, view: RegionView, texel: Optional[tuple[int, int, int]]) -> None:
        """World-map hover for the texel's face, with refined values swapped in
        (and, on a local map, the village under the cursor first)."""
        fields = view.fields
        if texel is None or fields is None:
            self.set_status(view.status_text())
            return
        hex_idx, row, col = texel

        def _local(keys: tuple[str, ...]) -> Optional[list[str]]:
            return describe_texel(fields, hex_idx, row, col, keys)

        if self._hover_parts is None:
            parts = _local(("elevation", "biome", "temperature", "precipitation")) or []
        else:
            uv = np.clip(fields.face_uv[texel], 0.0, 1.0)
            parts = self._hover_parts(int(fields.faces[texel]), uv, _local)
        village = view.village_under_cursor()
        if village is not None:
            parts = [village, *parts]
        self.set_status("  |  ".join(parts))

    def add_overlay(self, name: str, label: Optional[str] = None, checked: bool = True) -> None:
        """Register a named overlay toggle. Safe to call on every re-run (idempotent).

        One toggle drives both views, so an overlay stays on or off across switching.
        """
        if name in self._overlay_actions:
            self._set_overlay_visible(name, self._overlay_actions[name].isChecked())
            return
        action = QAction(label or name.title(), self._overlays_menu)
        action.setCheckable(True)
        action.setChecked(checked)
        action.toggled.connect(lambda v, n=name: self._set_overlay_visible(n, v))
        self._overlays_menu.addAction(action)
        self._overlay_actions[name] = action
        self._overlays_btn.setVisible(True)
        self._set_overlay_visible(name, checked)

    def _set_overlay_visible(self, name: str, visible: bool) -> None:
        self._map.set_overlay_visible(name, visible)
        self._region_view.set_overlay_visible(name, visible)
        self._local_view.set_overlay_visible(name, visible)

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
