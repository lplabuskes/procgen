"""Feature overlays, drawn the same way on the world map and the region view.

Every overlay feature is stored on the sphere (ECEF unit vectors), so the only
thing a view has to supply is how sphere points land on its screen — an
:class:`OverlayProjector`.  The world map projects through its map projection
(splitting at the antimeridian); the region view goes through the cell's local
frame and its north-up rotation.  Styling, filtering and size-vs-zoom rules live
here once, so the two views cannot drift apart.

Marker sizes, line widths and wind/current arrow lengths all scale with
``projector.zoom``: the world map's magnification, or, for a region, the world
map zoom that would show the same ground at the same pixel size.  Rivers are
the exception: each map level sizes them on its own scale (``view_zoom``), and a
local map draws them at their physical width (``px_per_km``).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional, Protocol, Sequence
import weakref

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QPolygonF

if TYPE_CHECKING:
    from core.world import World

# Screen-space line pieces: (x0, y0, x1, y1).
Piece = tuple[float, float, float, float]


class OverlayProjector(Protocol):
    """How a view maps sphere points to its screen, for overlay drawing."""

    @property
    def zoom(self) -> float:
        """World-map-equivalent magnification, for sizing markers and lines."""
        ...

    @property
    def view_zoom(self) -> float:
        """Magnification relative to this view's own unzoomed extent (the world
        map's zoom on the world map, the region view's own zoom in a region)."""
        ...

    @property
    def px_per_km(self) -> Optional[float]:
        """Screen pixels per km of ground when this view draws rivers at their
        physical width (a local map); None to size them cartographically."""
        ...

    @property
    def regional(self) -> bool:
        """True in a region view — region-scale features (towns) draw only there."""
        ...

    def on_land(self, key: str, ecef: np.ndarray) -> np.ndarray:
        """Settlement positions ``(N, 3)`` as this view should place them: moved
        off any water the view draws finer than the globe (``key`` names the
        feature, so a view can cache the result). Identity on the world map."""
        ...

    def points(self, ecef: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(N, 3)`` → ``(sx, sy, keep)``; ``keep`` drops points with no image."""
        ...

    def segments(
        self, p0: np.ndarray, p1: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Two ``(N, 3)`` endpoint arrays → ``(sx0, sy0, sx1, sy1, keep)``.

        ``keep`` drops segments the view cannot draw as one straight line
        (e.g. crossing the world map's antimeridian).
        """
        ...

    def split_segment(self, p0: np.ndarray, p1: np.ndarray) -> list[Piece]:
        """One segment as the screen pieces to draw — split rather than dropped."""
        ...

    def polygons(self, verts: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """``(N, 3)`` ring → the ``(sx, sy)`` screen rings that draw it."""
        ...


def draw_overlays(
    painter: QPainter, globe: "World", visible: Mapping[str, bool], proj: OverlayProjector,
    refined: Mapping[str, Any] | None = None,
) -> None:
    """Draw every visible overlay the globe has data for, in a fixed stacking order.

    ``refined`` holds a view's own finer version of a feature, in the world
    feature's form, drawn in its place (the region view's rivers, roads and
    deltas; a local map's also its towns at local positions). Widths and
    thresholds still come from the world feature, so a refined feature is drawn
    exactly as heavily as the world's.  ``refined["villages"]`` (a local map's
    :class:`localmap.villages.Villages`) is drawn under the towns.
    """
    refined = refined or {}

    def on(name: str) -> bool:
        return bool(visible.get(name)) and globe.has_feature(name)

    if on("rivers"):
        world_rivers = globe.get_feature("rivers") or []
        paths = refined.get("rivers", world_rivers)
        _draw_rivers(painter, paths, _max_river_flow(world_rivers), proj, _discharge_scale(globe))
    if on("lakes"):
        _draw_lakes(painter, globe, proj)
    if globe.has_feature("winds"):
        for season_idx in range(len(globe.get_feature("winds") or [])):
            if visible.get(f"winds_season_{season_idx}"):
                _draw_winds(painter, globe, proj, season_idx)
    if on("currents"):
        _draw_currents(painter, globe, proj)
    if on("deltas"):
        _draw_points(
            painter, globe, proj, "deltas", QColor(0, 100, 0, 220), QColor(0, 200, 80, 200),
            max(3.0, 5.0 * proj.zoom ** 0.3), positions=refined.get("deltas"),
        )
    if on("volcanoes"):
        _draw_points(
            painter, globe, proj, "volcanoes", QColor(160, 30, 0, 220), QColor(255, 80, 0, 200),
            max(2.5, 4.0 * proj.zoom ** 0.3),
        )
    if on("minerals"):
        _draw_minerals(painter, globe, proj)
    if on("rare_bio"):
        _draw_rare_bio(painter, globe, proj)
    if on("danger_sources"):
        _draw_danger_sources(painter, globe, proj)
    if on("trade_routes"):
        _draw_trade_routes(painter, globe, proj, refined.get("trade_routes"))
    # Villages exist only on a local map, never as a world feature.
    if visible.get("villages", True) and refined.get("villages") is not None:
        _draw_villages(painter, refined["villages"], proj)
    if on("towns"):
        _draw_towns(painter, refined.get("towns", globe.get_feature("towns")), proj)
    if on("settlements"):
        _draw_settlements(painter, globe, proj)
    if on("polity_borders"):
        _draw_polity_borders(painter, globe, proj)


def _line(painter: QPainter, x0: float, y0: float, x1: float, y1: float) -> None:
    painter.drawLine(QPointF(x0, y0), QPointF(x1, y1))


# ── points ───────────────────────────────────────────────────────────────────

def _draw_points(
    painter: QPainter, globe: "World", proj: OverlayProjector, feature_name: str,
    pen_color: QColor, brush_color: QColor, base_r: float, positions: Optional[np.ndarray] = None,
) -> None:
    """A uniformly-styled circle at each position of a point feature (or at
    ``positions``, a view's refined version of it)."""
    if positions is None:
        positions = globe.get_feature(feature_name)
    if positions is None or len(positions) == 0:
        return
    sx, sy, keep = proj.points(np.asarray(positions))
    painter.save()
    painter.setPen(QPen(pen_color, 1.5))
    painter.setBrush(QBrush(brush_color))
    for i in np.flatnonzero(keep):
        painter.drawEllipse(QPointF(float(sx[i]), float(sy[i])), base_r, base_r)
    painter.restore()


def _draw_resource(
    painter: QPainter, globe: "World", proj: OverlayProjector, feature_name: str,
    palette: np.ndarray, n_kinds: int, base_r: float,
    draw_shape: Callable[[QPainter, float, float, float], None],
) -> None:
    """A palette-per-kind resource overlay, sized by abundance."""
    feature = globe.get_feature(feature_name)
    if feature is None:
        return
    positions, kinds, abundances = feature
    if len(positions) == 0:
        return
    sx, sy, keep = proj.points(np.asarray(positions))
    painter.save()
    kinds_i = np.clip(kinds.astype(np.int32), 0, n_kinds - 1)
    for k in range(n_kinds):
        mask = (kinds_i == k) & keep
        if not mask.any():
            continue
        rgba = palette[k]
        r_fill, g_fill, b_fill = int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
        painter.setPen(QPen(QColor(max(0, r_fill - 60), max(0, g_fill - 60), max(0, b_fill - 60), 220), 1.0))
        painter.setBrush(QBrush(QColor(r_fill, g_fill, b_fill, 210)))
        for x, y, a in zip(sx[mask], sy[mask], abundances[mask]):
            draw_shape(painter, float(x), float(y), base_r * (0.7 + 0.6 * float(a)))
    painter.restore()


def _draw_minerals(painter: QPainter, globe: "World", proj: OverlayProjector) -> None:
    from worldgen.resources.minerals import MINERAL_PALETTE, N_MINERALS

    def _square(p: QPainter, x: float, y: float, r: float) -> None:
        p.drawRect(QRectF(x - r, y - r, 2 * r, 2 * r))

    _draw_resource(
        painter, globe, proj, "minerals", MINERAL_PALETTE, N_MINERALS,
        max(2.0, 3.5 * proj.zoom ** 0.3), _square,
    )


def _draw_rare_bio(painter: QPainter, globe: "World", proj: OverlayProjector) -> None:
    from worldgen.resources.rare_bio import N_RARE_BIO, RARE_BIO_PALETTE

    def _diamond(p: QPainter, x: float, y: float, r: float) -> None:
        p.drawPolygon(QPolygonF([
            QPointF(x, y - r), QPointF(x + r, y), QPointF(x, y + r), QPointF(x - r, y),
        ]))

    _draw_resource(
        painter, globe, proj, "rare_bio", RARE_BIO_PALETTE, N_RARE_BIO,
        max(2.5, 4.0 * proj.zoom ** 0.3), _diamond,
    )


def _draw_danger_sources(painter: QPainter, globe: "World", proj: OverlayProjector) -> None:
    from worldgen.danger.sources import SOURCE_COLORS
    feature = globe.get_feature("danger_sources")
    if feature is None:
        return
    positions, kinds = feature
    if len(positions) == 0:
        return
    sx, sy, keep = proj.points(np.asarray(positions))
    r = max(3.0, 5.0 * proj.zoom ** 0.3)
    painter.save()
    n_colors = len(SOURCE_COLORS)
    for i in np.flatnonzero(keep):
        rgba = SOURCE_COLORS[min(int(kinds[i]), n_colors - 1)]
        r_fill, g_fill, b_fill = int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
        painter.setPen(QPen(QColor(max(0, r_fill - 70), max(0, g_fill - 70), max(0, b_fill - 70), 230), 1.5))
        painter.setBrush(QBrush(QColor(r_fill, g_fill, b_fill, 210)))
        # Downward-pointing triangle.
        px, py = float(sx[i]), float(sy[i])
        painter.drawPolygon(QPolygonF([
            QPointF(px, py + r),
            QPointF(px - r * 0.87, py - r * 0.5),
            QPointF(px + r * 0.87, py - r * 0.5),
        ]))
    painter.restore()


# City markers in a region view: size multiplier over the world-map marker.
_REGION_CITY_SCALE = 1.5
# Largest town dot radius at zoom 1 (``_draw_towns``: 0.9 + 0.7 at the top size).
_TOWN_MAX_RADIUS = 1.6


def _draw_settlements(painter: QPainter, globe: "World", proj: OverlayProjector) -> None:
    from realmgen.placement.types import TIER_COLORS, TIER_RADII, population_display_tier
    feature = globe.get_feature("settlements")
    if not feature:
        return
    positions = np.array([s.position for s in feature], dtype=np.float64)
    pops = np.array([s.population for s in feature], dtype=np.float64)
    max_pop = float(pops.max()) if pops.size > 0 else 0.0
    tiers = np.array([population_display_tier(float(p), max_pop) for p in pops], dtype=np.int32)
    sx, sy, keep = proj.points(proj.on_land("settlements", positions))
    zoom_scale = max(1.0, proj.zoom ** 0.3)
    # Population-scaled radii: sqrt(pop / max_pop) in [0.6, 1.0] relative to the
    # tier base radius so the shape of the distribution is visible.
    if max_pop > 0:
        pop_scale = np.sqrt(np.clip(pops / max_pop, 0.0, 1.0)) * 0.4 + 0.6
    else:
        pop_scale = np.ones(len(feature), dtype=np.float64)
    # In a region, cities share the map with the towns: draw them bigger (never
    # smaller than twice the largest town), opaque, and ringed dark-then-white so
    # they read as a different class of place rather than big towns.
    regional = proj.regional
    size_scale = _REGION_CITY_SCALE if regional else 1.0
    min_r = 2.0 * _TOWN_MAX_RADIUS * proj.zoom ** 0.3 if regional else 0.0
    painter.save()
    for tier_idx in range(len(TIER_COLORS)):
        mask = (tiers == tier_idx) & keep
        if not mask.any():
            continue
        rgba = TIER_COLORS[tier_idx]
        r_fill, g_fill, b_fill = int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
        base_r = TIER_RADII[tier_idx] * zoom_scale * size_scale
        radii = [max(base_r * float(sc), min_r) for sc in pop_scale[mask]]
        centres = list(zip(sx[mask], sy[mask]))
        if regional:
            painter.setPen(QPen(QColor(255, 255, 255, 235), 2.5))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            for (x, y), r in zip(centres, radii):
                h = r + 1.6
                painter.drawEllipse(QRectF(float(x) - h, float(y) - h, 2 * h, 2 * h))
            painter.setPen(QPen(QColor(25, 22, 18, 245), 1.6))
            painter.setBrush(QBrush(QColor(r_fill, g_fill, b_fill, 255)))
        else:
            pen_width = 1.5 if tier_idx == 0 else 1.0
            painter.setPen(QPen(
                QColor(max(0, r_fill - 70), max(0, g_fill - 70), max(0, b_fill - 70), 230), pen_width))
            painter.setBrush(QBrush(QColor(r_fill, g_fill, b_fill, 220)))
        for (x, y), r in zip(centres, radii):
            painter.drawEllipse(QRectF(float(x) - r, float(y) - r, 2 * r, 2 * r))
    painter.restore()


def _draw_towns(painter: QPainter, towns, proj: OverlayProjector) -> None:
    """Small kind-coloured dots, sized by population; only in the region view."""
    from realmgen.towns.types import TOWN_KIND_COLORS
    if towns is None or len(towns) == 0 or not proj.regional:
        return
    sx, sy, keep = proj.points(proj.on_land("towns", towns.positions))
    pop = towns.population
    lo, hi = float(pop.min()), float(pop.max())
    frac = np.sqrt((pop - lo) / (hi - lo)) if hi > lo else np.zeros_like(pop)
    radii = (_TOWN_MAX_RADIUS - 0.7 + 0.7 * frac) * proj.zoom ** 0.3
    painter.save()
    painter.setPen(QPen(QColor(40, 36, 30, 200), 0.8))
    for kind, rgba in enumerate(TOWN_KIND_COLORS):
        mask = keep & (towns.kind == kind)
        if not mask.any():
            continue
        painter.setBrush(QBrush(QColor.fromRgbF(*rgba)))
        for x, y, r in zip(sx[mask], sy[mask], radii[mask]):
            painter.drawEllipse(QRectF(float(x) - r, float(y) - r, 2 * r, 2 * r))
    painter.restore()


# Farming, river, fishing, roadside (localmap.villages.VILLAGE_* order).
_VILLAGE_COLORS = (
    QColor(214, 190, 120), QColor(110, 170, 235), QColor(90, 205, 190), QColor(200, 150, 110),
)
# Dot radius (px) for the smallest and largest village: under the towns'
# ``_TOWN_MAX_RADIUS`` scale, so a village never reads as a town.
_VILLAGE_RADIUS = (1.3, 2.4)


def _draw_villages(painter: QPainter, villages, proj: OverlayProjector) -> None:
    """Small kind-coloured dots, sized by population; only on a local map."""
    if villages is None or len(villages) == 0:
        return
    sx, sy, keep = proj.points(villages.positions)
    pop = np.asarray(villages.population, dtype=np.float64)
    lo, hi = float(pop.min()), float(pop.max())
    frac = np.sqrt((pop - lo) / (hi - lo)) if hi > lo else np.zeros_like(pop)
    radii = _VILLAGE_RADIUS[0] + (_VILLAGE_RADIUS[1] - _VILLAGE_RADIUS[0]) * frac
    painter.save()
    painter.setPen(QPen(QColor(35, 30, 25, 210), 0.7))
    for kind, color in enumerate(_VILLAGE_COLORS):
        mask = keep & (villages.kind == kind)
        if not mask.any():
            continue
        painter.setBrush(QBrush(color))
        for x, y, r in zip(sx[mask], sy[mask], radii[mask]):
            painter.drawEllipse(QRectF(float(x) - r, float(y) - r, 2 * r, 2 * r))
    painter.restore()


# ── lines ────────────────────────────────────────────────────────────────────

def _arrow_ends(positions: np.ndarray, unit: np.ndarray, step: np.ndarray | float) -> np.ndarray:
    """Step ``positions`` along the tangent ``unit`` by ``step`` radians, back onto the sphere."""
    ends = positions + np.reshape(step, (-1, 1)) * unit
    return ends / np.maximum(np.linalg.norm(ends, axis=1, keepdims=True), 1e-12)


def _draw_winds(painter: QPainter, globe: "World", proj: OverlayProjector, season_idx: int) -> None:
    winds_feature = globe.get_feature("winds")
    if not winds_feature or season_idx >= len(winds_feature):
        return
    positions, vectors = winds_feature[season_idx]
    if len(positions) == 0:
        return
    speeds = np.linalg.norm(vectors, axis=1)
    if float(speeds.max()) <= 1e-6:
        return
    # Arc length per arrow on the unit sphere, shrinking as the view zooms in.
    step = 0.025 / proj.zoom ** 0.5
    unit = vectors / np.maximum(speeds[:, None], 1e-12)
    sx0, sy0, sx1, sy1, keep = proj.segments(positions, _arrow_ends(positions, unit, step))
    pen = QPen(QColor(40, 40, 40, 200))
    pen.setWidthF(1.2)
    painter.setPen(pen)
    for i in np.flatnonzero(keep):
        _line(painter, float(sx0[i]), float(sy0[i]), float(sx1[i]), float(sy1[i]))


def _draw_currents(painter: QPainter, globe: "World", proj: OverlayProjector) -> None:
    currents_feature = globe.get_feature("currents")
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
    keep_fast = speeds > max_speed * 0.05
    if not np.any(keep_fast):
        return
    positions, vectors = positions[keep_fast], vectors[keep_fast]
    speeds, temp_delta = speeds[keep_fast], temp_delta[keep_fast]

    step = (0.025 / proj.zoom ** 0.5) * (speeds / max_speed)
    unit = vectors / np.maximum(speeds[:, None], 1e-12)
    sx0, sy0, sx1, sy1, keep = proj.segments(positions, _arrow_ends(positions, unit, step))

    # Diverging blue → neutral → red palette, quantised to keep QPen churn low.
    max_abs = float(np.max(np.abs(temp_delta))) if temp_delta.size else 0.0
    norm = np.clip(temp_delta / max_abs, -1.0, 1.0) if max_abs > 1e-6 else np.zeros_like(temp_delta)
    bucket_count = 17
    buckets = np.clip(
        np.round((norm + 1.0) * 0.5 * (bucket_count - 1)).astype(np.int32), 0, bucket_count - 1,
    )
    bucket_pens: dict[int, QPen] = {}

    def _pen_for(bucket: int) -> QPen:
        pen = bucket_pens.get(bucket)
        if pen is not None:
            return pen
        t = (bucket / (bucket_count - 1)) * 2.0 - 1.0  # in [-1, 1]
        # Blue at t=-1, light gray near 0, red at t=+1.
        if t >= 0.0:
            r, g, b = int(200 + 30 * t), int(200 - 130 * t), int(200 - 150 * t)
        else:
            r, g, b = int(200 + 150 * t), int(200 + 130 * t), int(200 - 30 * t)
        pen = QPen(QColor(r, g, b, 230))
        pen.setWidthF(1.4)
        bucket_pens[bucket] = pen
        return pen

    active = -1
    for i in np.flatnonzero(keep):
        b = int(buckets[i])
        if b != active:
            painter.setPen(_pen_for(b))
            active = b
        _line(painter, float(sx0[i]), float(sy0[i]), float(sx1[i]), float(sy1[i]))


def _max_river_flow(river_paths: Sequence) -> float:
    return float(max((entry[1][-1] for entry in river_paths if len(entry[1]) > 0), default=0.0))


# World → m³/s per unit of river flow (a mean over the precipitation textures).
_DISCHARGE_SCALES: "weakref.WeakKeyDictionary[World, float]" = weakref.WeakKeyDictionary()


def _discharge_scale(globe: "World") -> float:
    scale = _DISCHARGE_SCALES.get(globe)
    if scale is None:
        from worldgen.hydrology.discharge import discharge_per_flow_unit
        scale = _DISCHARGE_SCALES[globe] = discharge_per_flow_unit(globe.mesh)
    return scale


# The narrowest line a local map draws a river with, so a river narrower than a
# pixel still shows as a hairline.
_MIN_PHYSICAL_RIVER_PX = 1.0
# Region-view rivers are drawn this much heavier than the world map's.
_REGION_RIVER_SCALE = 1.5


def _river_widths(flows: np.ndarray, max_flow: float, proj: OverlayProjector, discharge_scale: float) -> np.ndarray:
    """Screen width (px) of each river segment carrying ``flows``; ≤ 1 px is not drawn.

    On a local map (``proj.px_per_km`` set) a river is as wide as its channel:
    discharge from the flow, width from the hydraulic geometry, at least a
    hairline.  Elsewhere the width is cartographic — by flow relative to the
    world's largest river, on this map level's own zoom — so the world map and a
    region at the same zoom draw the same line weights.
    """
    flows = np.asarray(flows, dtype=np.float64)
    px_per_km = proj.px_per_km
    if px_per_km is not None and discharge_scale > 0.0:
        from worldgen.hydrology.discharge import channel_width_km
        width = channel_width_km(flows * discharge_scale) * px_per_km
        return np.maximum(width, _MIN_PHYSICAL_RIVER_PX + 1e-3)
    min_width, max_width = 0.5, 4.0
    scale = _REGION_RIVER_SCALE if proj.regional else 1.0
    return (min_width + (max_width - min_width) * (flows / max_flow) ** 0.5) * proj.view_zoom ** 0.5 * scale


def _draw_rivers(
    painter: QPainter, river_paths: Sequence, max_flow: float, proj: OverlayProjector, discharge_scale: float = 0.0,
) -> None:
    """River polylines, width by flow (see :func:`_river_widths`). ``max_flow``
    is the world's largest, so a refined river keeps its world river's weight."""
    if not river_paths or max_flow <= 0.0:
        return
    painter.setBrush(Qt.BrushStyle.NoBrush)

    active_pen: QPen | None = None
    for entry in river_paths:
        positions, flows = entry[0], entry[1]
        in_lake = entry[2] if len(entry) > 2 else None
        if len(positions) < 2:
            continue
        sx0, sy0, sx1, sy1, keep = proj.segments(positions[:-1], positions[1:])
        widths = _river_widths(flows, max_flow, proj, discharge_scale)
        for j in np.flatnonzero(keep):
            if in_lake is not None and j < len(in_lake) and in_lake[j]:
                continue  # segment lies inside a lake — hide it
            w = float(widths[j])
            if w <= 1.0:
                continue
            if active_pen is None or abs(active_pen.widthF() - w) > 0.05:
                active_pen = QPen(QColor(100, 180, 255, 200))
                active_pen.setWidthF(w)
                active_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                painter.setPen(active_pen)
            _line(painter, float(sx0[j]), float(sy0[j]), float(sx1[j]), float(sy1[j]))


def _draw_lakes(painter: QPainter, globe: "World", proj: OverlayProjector) -> None:
    polygons = globe.get_feature("lakes")
    if not polygons:
        return
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QBrush(QColor(80, 140, 220, 255)))
    for verts in polygons:
        for sx, sy in proj.polygons(np.asarray(verts)):
            painter.drawPolygon(QPolygonF([QPointF(float(x), float(y)) for x, y in zip(sx, sy)]))


def _position_key(pos) -> tuple[float, float, float]:
    from realmgen.refine.settlements import position_key
    return position_key(np.asarray(pos))


def _moved_settlements(globe: "World", proj: OverlayProjector) -> dict:
    """``{position_key: drawn position}`` for settlements the view moved onto
    land (``proj.on_land``), so trade routes still end at their markers."""
    feature = globe.get_feature("settlements")
    if not feature:
        return {}
    positions = np.array([s.position for s in feature], dtype=np.float64)
    drawn = proj.on_land("settlements", positions)
    return {
        _position_key(p): q for p, q in zip(positions, drawn) if not np.array_equal(p, q)
    }


def _draw_trade_routes(
    painter: QPainter, globe: "World", proj: OverlayProjector, refined: Optional[list] = None,
) -> None:
    """Route hops, width by flow. ``refined`` (a view's re-routed roads, already
    ending on the settlements' drawn markers) replaces the world's hops; the
    width scale and the drawing threshold are the world's either way."""
    world_routes = globe.get_feature("trade_routes")
    if not world_routes:
        return
    all_flows = np.array([r[2] for r in world_routes], dtype=np.float64)
    max_flow = float(all_flows.max()) if all_flows.size > 0 else 0.0
    if max_flow <= 0.0:
        return

    flow_threshold = float(np.median(all_flows))
    min_width, max_width = 0.4, 4.0
    zoom_scale = proj.zoom ** 0.4
    routes = world_routes if refined is None else refined
    moved = _moved_settlements(globe, proj) if refined is None else {}
    painter.setBrush(Qt.BrushStyle.NoBrush)
    active_pen: QPen | None = None
    for start_pos, end_pos, flow in routes:
        if float(flow) < flow_threshold:
            continue
        w = (min_width + (max_width - min_width) * (float(flow) / max_flow) ** 0.5) * zoom_scale
        if w < 0.3:
            continue
        if moved:
            start_pos = moved.get(_position_key(start_pos), start_pos)
            end_pos = moved.get(_position_key(end_pos), end_pos)
        pieces = proj.split_segment(np.asarray(start_pos), np.asarray(end_pos))
        if not pieces:
            continue
        if active_pen is None or abs(active_pen.widthF() - w) > 0.08:
            active_pen = QPen(QColor(220, 180, 60, 180))
            active_pen.setWidthF(w)
            painter.setPen(active_pen)
        for x0, y0, x1, y1 in pieces:
            _line(painter, x0, y0, x1, y1)


def _draw_polity_borders(painter: QPainter, globe: "World", proj: OverlayProjector) -> None:
    borders_by_depth = globe.get_feature("polity_borders")
    if not borders_by_depth:
        return
    painter.setBrush(Qt.BrushStyle.NoBrush)

    # Depth style: (line_width, alpha) — fainter at higher depth.
    def _style(depth: int) -> tuple[float, int]:
        if depth == 0:
            return 2.0, 220
        if depth == 1:
            return 1.2, 140
        return 0.7, 75

    # Deeper (fainter) borders first so sovereign borders render on top.
    for depth in sorted(borders_by_depth.keys(), reverse=True):
        segs: np.ndarray = borders_by_depth[depth]
        if segs.shape[0] == 0:
            continue
        sx0, sy0, sx1, sy1, keep = proj.segments(segs[:, :3], segs[:, 3:])
        if not np.any(keep):
            continue
        width, alpha = _style(depth)
        pen = QPen(QColor(0, 0, 0, alpha))
        pen.setWidthF(width * proj.zoom ** 0.3)
        painter.setPen(pen)
        for i in np.flatnonzero(keep):
            _line(painter, float(sx0[i]), float(sy0[i]), float(sx1[i]), float(sy1[i]))
