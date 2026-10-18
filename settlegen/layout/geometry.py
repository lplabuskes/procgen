"""Polygon helpers for the layout: multipolygons, blobs and discs, the old core, smoothing, a polygon's axes."""
from __future__ import annotations

import math

import numpy as np
import shapely
from shapely.geometry import LineString, MultiPolygon, Point, Polygon


def _multi(geom) -> MultiPolygon:  # type: ignore[no-untyped-def]
    """Polygons of ``geom`` (dropping lines and points) as one MultiPolygon."""
    if geom is None or geom.is_empty:
        return MultiPolygon()
    polys = [g for g in shapely.get_parts(geom) if isinstance(g, Polygon) and not g.is_empty]
    return MultiPolygon(polys)


def _area_disc(xy: np.ndarray, area: float, sides: int, rng: np.random.Generator, wobble: float) -> Polygon:
    """A wobbly ``sides``-gon of about ``area`` m² at ``xy``."""
    if wobble > 0:
        theta = np.sort(rng.uniform(0.0, 2.0 * np.pi, sides))
    else:
        theta = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    r = math.sqrt(area / (0.5 * sides * math.sin(2.0 * math.pi / sides)))
    radii = r * (1.0 + wobble * rng.uniform(-1.0, 1.0, sides))
    pts = np.column_stack([xy[0] + radii * np.cos(theta), xy[1] + radii * np.sin(theta)])
    hull = shapely.make_valid(Polygon(pts)).convex_hull
    return hull if isinstance(hull, Polygon) else Polygon()


def _blob(centre: np.ndarray, rng: np.random.Generator, wobble: float, n: int = 96) -> np.ndarray:
    """``(n,)`` radius multipliers of a smooth wobbly loop (a few low harmonics)."""
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    out = np.ones(n)
    for k in (2, 3, 4, 5):
        out += wobble / k * rng.standard_normal() * np.cos(k * theta + rng.uniform(0.0, 2.0 * np.pi))
    return np.clip(out, 0.4, None)


def core_polygon(footprint: MultiPolygon, centre: np.ndarray, share: float, rng: np.random.Generator,
                 wobble: float) -> MultiPolygon:
    """The footprint inside a smooth blob about ``centre`` holding ``share`` of its area
    (pieces under 5% of the largest dropped)."""
    if footprint.is_empty or share >= 0.999:
        return footprint
    target = share * footprint.area
    mult = _blob(centre, rng, wobble)
    theta = np.linspace(0.0, 2.0 * np.pi, len(mult), endpoint=False)

    def cut(r: float):  # type: ignore[no-untyped-def]
        loop = Polygon(np.column_stack([centre[0] + r * mult * np.cos(theta), centre[1] + r * mult * np.sin(theta)]))
        return footprint.intersection(shapely.make_valid(loop))

    lo, hi = 0.0, 2.0 * math.sqrt(footprint.area / math.pi) + float(np.max(np.abs(
        np.asarray(footprint.bounds).reshape(2, 2) - centre)))
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if cut(mid).area < target:
            lo = mid
        else:
            hi = mid
    core = _multi(cut(hi))
    if core.is_empty:
        return core
    biggest = max(g.area for g in core.geoms)
    return MultiPolygon([g for g in core.geoms if g.area >= 0.05 * biggest])


def chaikin(xy: np.ndarray, iterations: int = 3) -> np.ndarray:
    """Corner-cut ``xy`` (ends kept)."""
    for _ in range(iterations):
        if len(xy) < 3:
            return xy
        a, b = xy[:-1], xy[1:]
        q, r = 0.75 * a + 0.25 * b, 0.25 * a + 0.75 * b
        xy = np.concatenate([xy[:1], np.stack([q, r], axis=1).reshape(-1, 2)[1:-1], xy[-1:]])
    return xy


def _axes(poly: Polygon) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """``(centre, long unit, short unit, length, depth)`` of ``poly``'s minimum rotated rectangle."""
    with np.errstate(invalid="ignore", divide="ignore"):
        rect = np.asarray(shapely.get_coordinates(shapely.minimum_rotated_rectangle(poly)))[:4]
    e0, e1 = rect[1] - rect[0], rect[2] - rect[1]
    l0, l1 = float(np.linalg.norm(e0)), float(np.linalg.norm(e1))
    centre = rect.mean(axis=0)
    if l0 >= l1:
        u, length, depth = e0 / max(l0, 1e-9), l0, l1
    else:
        u, length, depth = e1 / max(l1, 1e-9), l1, l0
    return centre, u, np.array([-u[1], u[0]]), length, depth


def part_containing(geom: MultiPolygon, xy: np.ndarray) -> Polygon:
    """The piece of ``geom`` holding (else nearest) ``xy``."""
    pt = Point(float(xy[0]), float(xy[1]))
    return min(geom.geoms, key=lambda g: g.distance(pt))


def ray_exit(part: Polygon, origin: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Where a ray from ``origin`` along ``direction`` first leaves ``part`` (``origin`` if it never enters)."""
    x0, y0, x1, y1 = part.bounds
    reach = 2.0 * math.hypot(x1 - x0, y1 - y0) + 1.0
    o = Point(float(origin[0]), float(origin[1]))
    hit = LineString([origin, origin + reach * direction]).intersection(part)
    pieces = [g for g in shapely.get_parts(hit) if isinstance(g, LineString) and not g.is_empty]
    if not pieces:
        return np.asarray(origin, dtype=np.float64)
    first = min(pieces, key=lambda g: g.distance(o))
    ends = np.asarray(first.coords)[[0, -1]]
    return ends[int(np.argmax(np.linalg.norm(ends - origin, axis=1)))]
