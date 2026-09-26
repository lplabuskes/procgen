from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable, Optional, Sequence

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPolygonF, QTransform
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtWidgets import QWidget

from region.display import build_layer_texture
from viz.mapview.legend import draw_legend, legend_size
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
    """

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
        self._hex_images: list[QImage] = []
        # (x0, y0, size, rotation) of the cell as last painted — hover hit-tests against it.
        self._cell_layout: Optional[tuple[float, float, float, QTransform]] = None
        self._hover_callback: Optional[Callable[[Optional[tuple[int, int, int]]], None]] = None
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)

    # ── data ─────────────────────────────────────────────────────────────────

    def set_region(self, geometry: RegionGeometry, globe: Optional["World"] = None) -> None:
        """Show a region, refining the global fields inside it when a globe is given."""
        self._geometry = geometry
        self._globe = globe
        self._grid = RegionHexGrid(geometry, int(self._params.n_cols))
        self._fields = None
        self._hex_images = []
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
        transform = self._cell_transform(angle, x0, y0, size)
        self._cell_layout = (x0, y0, size, transform)

        painter.save()
        painter.setTransform(transform, True)
        if self._hex_images:
            self._draw_textured_hexes(painter, x0, y0, size)
        else:
            self._draw_hex_grid(painter, x0, y0, size)

        # The cell outline; the run's summary goes to the window's status bar
        # (``status_text``) rather than over the map.
        painter.setPen(QPen(QColor(70, 85, 100), 1.0))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(QRectF(x0, y0, size, size))
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
        if self._hover_callback is not None:
            pos = event.position()
            self._hover_callback(self.texel_at_screen(float(pos.x()), float(pos.y())))

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
        # which is real data for the adjacent hexagon anyway.
        r_draw = r_px + 0.75

        painter.save()
        painter.setClipRect(QRectF(x0, y0, size, size))
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
        painter.setPen(QPen(QColor(20, 25, 30, 60), 1.0))
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
        painter.setClipRect(QRectF(x0, y0, size, size))
        pen = QPen(QColor(90, 110, 130))
        pen.setWidthF(1.0)
        painter.setPen(pen)
        painter.setBrush(QColor(34, 40, 48))
        for s, t in self._grid.centers:
            cx = x0 + s * size
            cy = y0 + (1.0 - t) * size
            painter.drawPolygon(self._hex_polygon(cx, cy, r_px))
        painter.restore()
