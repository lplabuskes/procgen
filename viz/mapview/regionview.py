from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable, Optional, Sequence

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPainterPath, QPen, QPolygonF, QTransform
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtWidgets import QWidget

from region.display import build_layer_texture
from viz.mapview.legend import draw_legend, legend_size
from viz.mapview.overlays import draw_overlays
from region.geometry import RegionGeometry
from region.grid import RegionHexGrid
from region.upsample import RegionFields, RegionUpsampleParams, build_region_fields

if TYPE_CHECKING:
    from core.world import World


class RegionView(QWidget):
    """Zoomed-in view of a single region.

    The region cell is a unit square in the equal-area ``(s, t)`` local frame
    (see :mod:`region.geometry`), tiled with flat-top hexagons by
    :class:`~region.grid.RegionHexGrid`.  Each hexagon carries its own texture
    holding the refined fields from :mod:`region.upsample` — the global map
    upsampled with slope-proportional detail, with every global texel's average
    preserved.  Those textures are drawn directly: local ``(s, t)`` maps to
    screen by a single flip-and-scale, so a hex texture is an axis-aligned image
    clipped to its hexagon.

    Without a globe (or before a region is picked) the view falls back to drawing
    the bare hex scaffolding.

    Scroll to zoom (1×–``MAX_ZOOM``, anchored to the cursor) and drag to pan, as
    on the world map.  The zoom is a screen-space magnification of the laid-out
    cell, so it composes with the north-up rotation and the legend stays put.
    """

    MAX_ZOOM = 20.0

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._geometry: Optional[RegionGeometry] = None
        self._grid: Optional[RegionHexGrid] = None
        self._fields: Optional[RegionFields] = None
        self._params = RegionUpsampleParams()
        self._globe: Optional["World"] = None
        # The world map's active scheme: name, color function, per-face texture key.
        self._layer: Optional[str] = None
        self._scheme_fn: Optional[Callable[[dict], Sequence[float]]] = None
        self._texture_key: Optional[str] = None
        self._legend: Optional[tuple[str, list]] = None
        self._overlay_visible: dict[str, bool] = {}
        self._hex_images: list[QImage] = []
        # (x0, y0, size, rotation) of the cell as last painted — hover hit-tests against it.
        self._cell_layout: Optional[tuple[float, float, float, QTransform]] = None
        self._hover_callback: Optional[Callable[[Optional[tuple[int, int, int]]], None]] = None
        # Magnification about the cell's footprint centre, and the pan offset in
        # units of the footprint width (so it survives a resize).
        self._zoom: float = 1.0
        self._pan = np.zeros(2, dtype=np.float64)
        # (centre x, centre y, footprint px) of the last paint — zoom/pan anchor.
        self._view_frame: Optional[tuple[float, float, float]] = None
        self._drag_start: Optional[tuple[float, float]] = None
        self._drag_start_pan = np.zeros(2, dtype=np.float64)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    # ── data ─────────────────────────────────────────────────────────────────

    def set_region(self, geometry: RegionGeometry, globe: Optional["World"] = None) -> None:
        """Show a region, refining the global fields inside it when a globe is given."""
        self._geometry = geometry
        self._globe = globe
        self._grid = RegionHexGrid(geometry, int(self._params.n_cols))
        self._fields = None
        self._hex_images = []
        self._zoom = 1.0
        self._pan[:] = 0.0
        if globe is not None and "elevation" in globe.mesh.face_textures:
            self._fields = build_region_fields(globe, geometry.region_id, self._params)
            self._grid = self._fields.grid
            self._rebuild_images()
        self.update()

    def clear_region(self) -> None:
        """Drop the current region (and its refined fields) until the next ``set_region``."""
        self._geometry = None
        self._globe = None
        self._grid = None
        self._fields = None
        self._hex_images = []
        self._cell_layout = None
        self._view_frame = None
        self._drag_start = None

    def set_layer(
        self,
        layer: Optional[str],
        scheme_fn: Optional[Callable[[dict], Sequence[float]]] = None,
        texture_key: Optional[str] = None,
    ) -> None:
        """Show one of the world map's layers (its scheme name, color fn, texture key)."""
        if (layer, scheme_fn, texture_key) == (self._layer, self._scheme_fn, self._texture_key):
            return
        self._layer = layer
        self._scheme_fn = scheme_fn
        self._texture_key = texture_key
        self._rebuild_images()
        self.update()

    def set_overlay_visible(self, name: str, visible: bool) -> None:
        """Toggle one of the world map's feature overlays (same names, same styling)."""
        self._overlay_visible[name] = visible
        self.update()

    def set_legend(self, legend: Optional[tuple[str, list]]) -> None:
        """The active layer's ``(kind, entries)`` legend, drawn beside the cell."""
        self._legend = legend
        self.update()

    def set_hover_callback(
        self, cb: Optional[Callable[[Optional[tuple[int, int, int]]], None]]
    ) -> None:
        """Called with the ``(hex, row, col)`` texel under the cursor, or None off the map."""
        self._hover_callback = cb

    @property
    def layer(self) -> Optional[str]:
        return self._layer

    @property
    def region_id(self) -> Optional[int]:
        return self._geometry.region_id if self._geometry is not None else None

    @property
    def fields(self) -> Optional[RegionFields]:
        return self._fields

    @property
    def hex_local(self) -> np.ndarray:
        """(M, 2) hex-centre coordinates in the cell's local [0, 1]^2 frame."""
        if self._grid is None:
            return np.empty((0, 2), dtype=np.float64)
        return self._grid.centers

    @property
    def hex_ecef(self) -> np.ndarray:
        """(M, 3) ECEF unit-sphere position of each hex centre."""
        if self._grid is None:
            return np.empty((0, 3), dtype=np.float64)
        return self._grid.centers_ecef()

    def status_text(self) -> str:
        """One-line summary of the refinement, for the window's status bar."""
        if self._fields is None or self._grid is None:
            return f"Region {self.region_id}"
        f = self._fields
        return (
            f"Region {f.region_id} · {f.n_hexes} hexes × {f.tex_size}² texels "
            f"(~{f.texel_area_km2:.0f} km²/texel) · {f.detail_octaves} detail octaves"
        )

    def _rebuild_images(self) -> None:
        """Convert each hexagon's refined texture to a QImage for painting."""
        self._hex_images = []
        if self._fields is None or self._layer is None:
            return
        rgba = build_layer_texture(                                 # (M, T, T, 4)
            self._fields, self._layer,
            self._globe.mesh if self._globe is not None else None,
            scheme_fn=self._scheme_fn, texture_key=self._texture_key,
        )
        data = np.ascontiguousarray((rgba * 255.0).clip(0, 255).astype(np.uint8))
        m, h, w, _ = data.shape
        for i in range(m):
            plane = np.ascontiguousarray(data[i])
            self._hex_images.append(
                QImage(plane.data, w, h, w * 4, QImage.Format.Format_RGBA8888).copy()
            )

    # ── painting ─────────────────────────────────────────────────────────────

    def paintEvent(self, _) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(24, 28, 34))
        self._cell_layout = None

        if self._grid is None:
            return
        W, H = float(self.width()), float(self.height())
        margin = 16.0
        legend_gap = 12.0
        legend = self._legend if self._legend is not None and len(self._legend[1]) >= 2 else None
        legend_w, legend_h = (
            legend_size(painter, legend[0], legend[1], H - 2.0 * margin) if legend else (0, 0)
        )
        reserved = legend_w + legend_gap if legend else 0.0

        # Size the cell so its *rotated* footprint fits beside the legend; the
        # rotation about the centre swings the corners out by |cos| + |sin|.
        angle = self._up_angle()
        spread = abs(math.cos(angle)) + abs(math.sin(angle))
        footprint = min(H - 2.0 * margin, W - 2.0 * margin - reserved)
        size = footprint / spread
        if size <= 0.0:
            return
        # Centred when there is room, otherwise pushed right to clear the legend.
        fx0 = max((W - footprint) * 0.5, margin + reserved)
        fy0 = (H - footprint) * 0.5
        x0 = fx0 + 0.5 * (footprint - size)
        y0 = fy0 + 0.5 * (footprint - size)
        cx, cy = fx0 + 0.5 * footprint, fy0 + 0.5 * footprint
        self._view_frame = (cx, cy, footprint)
        transform = self._cell_transform(angle, x0, y0, size) * self._view_transform()
        self._cell_layout = (x0, y0, size, transform)

        painter.save()
        # Zoomed in, the map fills everything right of the legend but never covers it.
        if legend:
            left = fx0 - 0.5 * legend_gap
            painter.setClipRect(QRectF(left, 0.0, W - left, H))
        painter.setTransform(transform, True)
        if self._hex_images:
            self._draw_textured_hexes(painter, x0, y0, size)
        else:
            self._draw_hex_grid(painter, x0, y0, size)

        # The cell outline; the run's summary goes to the window's status bar
        # (``status_text``) rather than over the map.
        painter.setPen(_cosmetic_pen(QColor(70, 85, 100)))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(QRectF(x0, y0, size, size))
        painter.restore()

        # Overlays are drawn in screen space (so markers stay upright and sized
        # in pixels), clipped to the rotated cell.
        if self._globe is not None and self._geometry is not None:
            painter.save()
            if legend:
                left = fx0 - 0.5 * legend_gap
                painter.setClipRect(QRectF(left, 0.0, W - left, H))
            cell = QPainterPath()
            cell.addPolygon(transform.map(QPolygonF(QRectF(x0, y0, size, size))))
            painter.setClipPath(cell, Qt.ClipOperation.IntersectClip)
            draw_overlays(
                painter, self._globe, self._overlay_visible,
                _RegionProjector(self._geometry, x0, y0, size, transform, self._equivalent_zoom(size, W)),
            )
            painter.restore()

        # Outside the map, bottom-aligned with the cell's footprint to its left.
        if legend:
            draw_legend(
                painter, legend[0], legend[1],
                fx0 - legend_gap - legend_w, fy0 + footprint, H - 2.0 * margin,
            )

    def _up_angle(self) -> float:
        """Screen rotation (radians) that makes "up" read correctly for this cell.

        ``RegionGeometry.up_direction()`` gives the local ``(ds, dt)`` that
        should render as up — true north ordinarily, or the antimeridian
        tangent on the two regions centred on a pole, where north has no
        direction. The ``(s, screen-x) / (t, screen-y flipped)`` mapping already
        treats +t as up with no rotation, so this is 0 for every cell where that
        already points at the target.
        """
        if self._geometry is None:
            return 0.0
        ds, dt = self._geometry.up_direction()
        angle = math.atan2(-ds, dt)
        return 0.0 if abs(angle) < 1e-9 else angle

    def _equivalent_zoom(self, size: float, W: float) -> float:
        """The world-map zoom showing this cell's ground at its on-screen scale.

        Overlays size their markers and lines by zoom; handing them this makes
        a region look like the world map zoomed in to the same place.  The
        world map at zoom 1 spans 2π of longitude across the view's width.
        """
        assert self._geometry is not None
        ends = self._geometry.local_to_ecef(np.array([0.0, 1.0]), np.array([0.5, 0.5]))
        side_rad = float(np.arccos(np.clip(np.dot(ends[0], ends[1]), -1.0, 1.0)))
        px_per_rad = size * self._zoom / max(side_rad, 1e-9)
        return px_per_rad / (W / (2.0 * np.pi))

    def _view_transform(self) -> QTransform:
        """Zoom about the footprint centre, then pan — applied after the rotation."""
        t = QTransform()
        if self._view_frame is None:
            return t
        cx, cy, footprint = self._view_frame
        t.translate(cx + self._pan[0] * footprint, cy + self._pan[1] * footprint)
        t.scale(self._zoom, self._zoom)
        t.translate(-cx, -cy)
        return t

    # ── zoom / pan ───────────────────────────────────────────────────────────

    @property
    def zoom(self) -> float:
        return self._zoom

    def zoom_toward(self, mx: float, my: float, factor: float) -> None:
        """Scale by ``factor`` keeping the screen point ``(mx, my)`` fixed."""
        if self._view_frame is None:
            return
        cx, cy, footprint = self._view_frame
        new_zoom = max(1.0, min(self.MAX_ZOOM, self._zoom * factor))
        if new_zoom == 1.0:
            self._pan[:] = 0.0
        else:
            # The unzoomed point under the cursor must land under it again.
            px = cx + self._pan[0] * footprint
            py = cy + self._pan[1] * footprint
            qx = (mx - px) / self._zoom
            qy = (my - py) / self._zoom
            self._pan[0] = (mx - cx - new_zoom * qx) / footprint
            self._pan[1] = (my - cy - new_zoom * qy) / footprint
        self._zoom = new_zoom
        self._clamp_pan()
        self.update()

    def zoom_in(self) -> None:
        self.zoom_toward(self.width() * 0.5, self.height() * 0.5, 1.2)

    def zoom_out(self) -> None:
        self.zoom_toward(self.width() * 0.5, self.height() * 0.5, 1.0 / 1.2)

    def reset_view(self) -> None:
        self._zoom = 1.0
        self._pan[:] = 0.0
        self.update()

    def _clamp_pan(self) -> None:
        """Keep the footprint centre's view covered — no panning off into the void."""
        limit = 0.5 * (self._zoom - 1.0)
        self._pan[:] = np.clip(self._pan, -limit, limit)

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        if delta:
            pos = event.position()
            self.zoom_toward(float(pos.x()), float(pos.y()), 1.2 ** (delta / 120.0))
        event.accept()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position()
            self._drag_start = (float(pos.x()), float(pos.y()))
            self._drag_start_pan = self._pan.copy()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = None
            self.setCursor(Qt.CursorShape.OpenHandCursor)

    @staticmethod
    def _cell_transform(angle: float, x0: float, y0: float, size: float) -> QTransform:
        """Rotation of the whole cell about its centre by ``angle``."""
        cx, cy = x0 + 0.5 * size, y0 + 0.5 * size
        t = QTransform()
        t.translate(cx, cy)
        t.rotate(math.degrees(angle))
        t.translate(-cx, -cy)
        return t

    # ── hover ────────────────────────────────────────────────────────────────

    def texel_at_screen(self, mx: float, my: float) -> Optional[tuple[int, int, int]]:
        """``(hex, row, col)`` of the refined texel under a screen point, or None."""
        if self._cell_layout is None or self._fields is None:
            return None
        x0, y0, size, transform = self._cell_layout
        inverse, ok = transform.inverted()
        if not ok:
            return None
        p = inverse.map(QPointF(mx, my))
        s = (p.x() - x0) / size
        t = 1.0 - (p.y() - y0) / size
        return self._fields.grid.texel_at(s, t, self._fields.tex_size)

    def mouseMoveEvent(self, event) -> None:
        pos = event.position()
        mx, my = float(pos.x()), float(pos.y())
        if self._drag_start is not None and self._view_frame is not None:
            footprint = self._view_frame[2]
            self._pan[0] = self._drag_start_pan[0] + (mx - self._drag_start[0]) / footprint
            self._pan[1] = self._drag_start_pan[1] + (my - self._drag_start[1]) / footprint
            self._clamp_pan()
            self.update()
        if self._hover_callback is not None:
            self._hover_callback(self.texel_at_screen(mx, my))

    def leaveEvent(self, event) -> None:
        if self._hover_callback is not None:
            self._hover_callback(None)

    def _hex_polygon(self, cx: float, cy: float, r_px: float) -> QPolygonF:
        return QPolygonF([
            QPointF(cx + r_px * math.cos(math.radians(60 * i)),
                    cy + r_px * math.sin(math.radians(60 * i)))
            for i in range(6)
        ])

    def _draw_textured_hexes(
        self, painter: QPainter, x0: float, y0: float, size: float
    ) -> None:
        """Blit each hexagon's texture, clipped to the hexagon itself.

        The texture square spans ``2R`` in local units centred on the hex, with
        row 0 at the top — so the image-to-screen map is a plain scale plus
        translate, and the over-hanging corners are simply clipped away.
        """
        assert self._grid is not None and self._fields is not None
        r_px = self._grid.radius * size
        tex_size = self._fields.tex_size
        scale = (2.0 * r_px) / tex_size
        # Draw a hair past the hexagon so antialiased neighbours overlap instead
        # of leaving a seam; the extra pixel comes from the texture's over-hang,
        # which is real data for the adjacent hexagon anyway.  Divided by the
        # zoom so the overlap stays sub-pixel on screen.
        r_draw = r_px + 0.75 / self._zoom

        painter.save()
        painter.setClipRect(QRectF(x0, y0, size, size), Qt.ClipOperation.IntersectClip)
        painter.setPen(Qt.PenStyle.NoPen)
        for i, (s, t) in enumerate(self._grid.centers):
            cx = x0 + s * size
            cy = y0 + (1.0 - t) * size       # local +t is up; _up_angle makes that true north
            brush = QBrush(self._hex_images[i])
            brush.setTransform(
                QTransform(scale, 0.0, 0.0, scale, cx - r_px, cy - r_px)
            )
            painter.setBrush(brush)
            painter.drawPolygon(self._hex_polygon(cx, cy, r_draw))

        # Faint outlines: the hexagons are the scaffolding for local maps, so
        # they stay legible without competing with the terrain.
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(_cosmetic_pen(QColor(20, 25, 30, 60)))
        for s, t in self._grid.centers:
            painter.drawPolygon(
                self._hex_polygon(x0 + s * size, y0 + (1.0 - t) * size, r_px)
            )
        painter.restore()

    def _draw_hex_grid(self, painter: QPainter, x0: float, y0: float, size: float) -> None:
        """Fallback: bare scaffolding when no globe data is available."""
        assert self._grid is not None
        r_px = self._grid.radius * size
        painter.save()
        painter.setClipRect(QRectF(x0, y0, size, size), Qt.ClipOperation.IntersectClip)
        painter.setPen(_cosmetic_pen(QColor(90, 110, 130)))
        painter.setBrush(QColor(34, 40, 48))
        for s, t in self._grid.centers:
            cx = x0 + s * size
            cy = y0 + (1.0 - t) * size
            painter.drawPolygon(self._hex_polygon(cx, cy, r_px))
        painter.restore()


def _cosmetic_pen(color: QColor) -> QPen:
    """A 1-px line that stays 1 px however far the view is zoomed."""
    pen = QPen(color, 1.0)
    pen.setCosmetic(True)
    return pen


class _RegionProjector:
    """Overlay projector for a region: the cell's local frame, then its screen transform.

    Features beyond ``_MARGIN`` of the cell (in local units) are culled rather
    than drawn under the clip, and the far side of the globe has no image.
    """

    _MARGIN = 0.15

    def __init__(
        self, geometry: RegionGeometry, x0: float, y0: float, size: float,
        transform: QTransform, zoom: float,
    ) -> None:
        self._geometry = geometry
        self._x0, self._y0, self._size = x0, y0, size
        self._t = transform
        self._zoom = zoom

    @property
    def zoom(self) -> float:
        return self._zoom

    def _project(self, ecef: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """``(sx, sy, valid, near)``: screen position, has-an-image, near-the-cell."""
        s, t, valid = self._geometry.ecef_to_local(np.asarray(ecef).reshape(-1, 3))
        m = self._MARGIN
        near = valid & (s > -m) & (s < 1.0 + m) & (t > -m) & (t < 1.0 + m)
        x = self._x0 + s * self._size
        y = self._y0 + (1.0 - t) * self._size
        # QTransform maps x' = m11 x + m21 y + dx, y' = m12 x + m22 y + dy.
        tr = self._t
        sx = tr.m11() * x + tr.m21() * y + tr.dx()
        sy = tr.m12() * x + tr.m22() * y + tr.dy()
        return sx, sy, valid, near

    def points(self, ecef: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sx, sy, _, near = self._project(ecef)
        return sx, sy, near

    def segments(
        self, p0: np.ndarray, p1: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        sx0, sy0, valid0, near0 = self._project(p0)
        sx1, sy1, valid1, near1 = self._project(p1)
        return sx0, sy0, sx1, sy1, valid0 & valid1 & (near0 | near1)

    def split_segment(self, p0: np.ndarray, p1: np.ndarray) -> list[tuple[float, float, float, float]]:
        sx0, sy0, sx1, sy1, keep = self.segments(p0, p1)
        if not keep[0]:
            return []
        return [(float(sx0[0]), float(sy0[0]), float(sx1[0]), float(sy1[0]))]

    def polygons(self, verts: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        sx, sy, valid, near = self._project(verts)
        if not valid.all() or not near.any():
            return []
        return [(sx, sy)]
