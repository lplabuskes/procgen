"""Measuring tape: two points clicked on any map and the ground distance between them.

Ends are ECEF unit vectors, so one tape works on the world map, a region, a local
map and a settlement map alike; the distance is the great-circle one on the
Earth-sized globe and the line is drawn along the great circle.
"""
from __future__ import annotations

import math
from typing import Optional, Protocol

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFontMetricsF, QPainter, QPen, QPolygonF

from worldgen.hydrology.discharge import EARTH_RADIUS_KM

# Arc samples per radian of ground (~one per 0.25°), so the line bends with the projection.
_SAMPLES_PER_RAD = 230.0
_MAX_SAMPLES = 512

_LINE_COLOR = QColor(255, 220, 60)
_CASING_COLOR = QColor(20, 20, 20, 200)
_LABEL_BG = QColor(20, 20, 20, 210)
_LABEL_FG = QColor(255, 240, 180)


class _Projector(Protocol):
    def points(self, ecef: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...

    def segments(
        self, p0: np.ndarray, p1: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...


class MeasureTape:
    """One view's tape: the first click sets the start, the second the end, a third starts over.

    Between the two clicks the line follows ``cursor`` (the ground under the mouse).
    """

    def __init__(self) -> None:
        self.start: Optional[np.ndarray] = None
        self.end: Optional[np.ndarray] = None
        self.cursor: Optional[np.ndarray] = None

    def click(self, point: np.ndarray) -> None:
        point = np.asarray(point, dtype=np.float64).reshape(3)
        if self.start is None or self.end is not None:
            self.start, self.end = point, None
        else:
            self.end = point
        self.cursor = None

    def clear(self) -> None:
        self.start = self.end = self.cursor = None

    @property
    def waiting(self) -> bool:
        """The start is set and the line follows the cursor to the end."""
        return self.start is not None and self.end is None

    def ends(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """The ``(start, end)`` to draw: the fixed end, else the cursor; None before both exist."""
        b = self.end if self.end is not None else self.cursor
        return None if self.start is None or b is None else (self.start, b)


def distance_km(a: np.ndarray, b: np.ndarray) -> float:
    """Great-circle ground distance (km) between two unit vectors."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    angle = math.atan2(float(np.linalg.norm(np.cross(a, b))), float(np.dot(a, b)))
    return angle * EARTH_RADIUS_KM


def arc_points(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``(N, 3)`` points along the great circle from ``a`` to ``b`` (both included)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    angle = distance_km(a, b) / EARTH_RADIUS_KM
    n = int(min(_MAX_SAMPLES, max(2, math.ceil(angle * _SAMPLES_PER_RAD) + 1)))
    t = np.linspace(0.0, 1.0, n)[:, None]
    if angle < 1e-9:
        return np.repeat(a[None, :], n, axis=0)
    # Slerp; (near-)antipodal ends have no unique great circle — any one will do.
    s = math.sin(angle)
    if s < 1e-9:
        return np.vstack([a, b])
    pts = (np.sin((1.0 - t) * angle) * a + np.sin(t * angle) * b) / s
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)


def format_distance(km: float) -> str:
    """Metres under a kilometre, else kilometres with precision falling as the distance grows."""
    if km < 1.0:
        return f"{km * 1000.0:.0f} m"
    if km < 10.0:
        return f"{km:.2f} km"
    if km < 100.0:
        return f"{km:.1f} km"
    return f"{km:,.0f} km"


def tape_polylines(projector: _Projector, a: np.ndarray, b: np.ndarray) -> list[QPolygonF]:
    """Screen polylines of the great circle ``a`` → ``b``, broken where the projector
    drops a segment (the world map's antimeridian, a region's far side)."""
    arc = arc_points(a, b)
    sx0, sy0, sx1, sy1, keep = projector.segments(arc[:-1], arc[1:])
    lines: list[QPolygonF] = []
    current: list[QPointF] = []
    for i in range(len(keep)):
        if not keep[i]:
            if len(current) >= 2:
                lines.append(QPolygonF(current))
            current = []
            continue
        if not current:
            current.append(QPointF(float(sx0[i]), float(sy0[i])))
        current.append(QPointF(float(sx1[i]), float(sy1[i])))
    if len(current) >= 2:
        lines.append(QPolygonF(current))
    return lines


def draw_tape(painter: QPainter, tape: MeasureTape, projector: _Projector) -> None:
    """The tape's line (cased, along the great circle), its ends and the distance label."""
    ends = tape.ends()
    if ends is None:
        if tape.start is not None:
            _draw_ends(painter, projector, tape.start[None, :])
        return
    a, b = ends
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    lines = tape_polylines(projector, a, b)
    for color, width in ((_CASING_COLOR, 4.0), (_LINE_COLOR, 2.0)):
        pen = QPen(color, width)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        if color is _LINE_COLOR and tape.end is None:
            pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        for line in lines:
            painter.drawPolyline(line)
    painter.restore()
    _draw_ends(painter, projector, np.vstack([a, b]) if tape.end is not None else a[None, :])

    # The label sits beside the moving (or second) end.
    sx, sy, keep = projector.points(b[None, :])
    if keep[0]:
        _draw_label(painter, format_distance(distance_km(a, b)), float(sx[0]), float(sy[0]))


def _draw_ends(painter: QPainter, projector: _Projector, pts: np.ndarray) -> None:
    sx, sy, keep = projector.points(pts)
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(QPen(_CASING_COLOR, 1.5))
    painter.setBrush(_LINE_COLOR)
    for x, y, k in zip(sx, sy, keep):
        if k:
            painter.drawEllipse(QPointF(float(x), float(y)), 4.0, 4.0)
    painter.restore()


def _draw_label(painter: QPainter, text: str, x: float, y: float) -> None:
    painter.save()
    metrics = QFontMetricsF(painter.font())
    w, h = metrics.horizontalAdvance(text) + 10.0, metrics.height() + 4.0
    # Up-right of the point, flipped inside the viewport near its edges.
    view = painter.viewport()
    left = x + 10.0 if x + 10.0 + w <= view.width() else x - 10.0 - w
    top = y - 10.0 - h if y - 10.0 - h >= 0.0 else y + 10.0
    rect = QRectF(left, top, w, h)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(_LABEL_BG)
    painter.drawRoundedRect(rect, 4.0, 4.0)
    painter.setPen(_LABEL_FG)
    painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)
    painter.restore()
