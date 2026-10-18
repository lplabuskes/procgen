"""Where the site's roads meet things: crossings of river and stream channels (bridges and fords),
and gates where they leave the footprint."""
from __future__ import annotations

import numpy as np
import shapely
from shapely.geometry import LineString, MultiPolygon, Point

from settlegen.site.types import Crossing, Gate, RoadLine, SiteWater


# Points closer than this (m) along one road are one gate / crossing.
_SAME_POINT_M = 2.0


def _nearest_edge(xy: np.ndarray, p: np.ndarray) -> int:
    """Index of the edge of polyline ``xy`` nearest point ``p``."""
    a, b = xy[:-1], xy[1:]
    ab = b - a
    t = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-18), 0.0, 1.0)
    return int(np.argmin(np.linalg.norm(a + t[:, None] * ab - p, axis=1)))


def _points(geom) -> list[np.ndarray]:  # type: ignore[no-untyped-def]
    """The points of an intersection (a touching stretch counts by its ends)."""
    if geom.is_empty:
        return []
    if isinstance(geom, Point):
        return [np.array([geom.x, geom.y])]
    if isinstance(geom, LineString):
        c = np.asarray(geom.coords)
        return [c[0], c[-1]]
    return [q for g in getattr(geom, "geoms", []) for q in _points(g)]


def _dedupe(points: list[np.ndarray]) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for p in points:
        if all(np.linalg.norm(p - q) >= _SAME_POINT_M for q in out):
            out.append(p)
    return out


def crossings(
    lines: list[RoadLine], water: SiteWater, bridge_width_m: float, always_bridge: bool, merge_m: float,
) -> list[Crossing]:
    """Every point a road crosses a dry river or stream centreline; roads that
    share a way (two routes re-traced onto one path) cross once, flows summed."""
    out: list[Crossing] = []
    for j, river in enumerate(water.lines):
        dry = ~river.in_water
        if not dry.any():
            continue
        segs = shapely.linestrings(np.stack([river.xy[:-1][dry], river.xy[1:][dry]], axis=1))
        rline = shapely.union_all(segs)
        for i, road in enumerate(lines):
            for p in _dedupe(_points(LineString(road.xy).intersection(rline))):
                flow = float(road.flows[_nearest_edge(road.xy, p[None])])
                same = [c for c in out if c.river == j and np.linalg.norm(c.xy - p) < merge_m]
                if same:
                    same[0].flow += flow
                    continue
                width = float(river.width_m[_nearest_edge(river.xy, p[None])])
                out.append(Crossing(
                    xy=p, road=i, river=j, width_m=width, flow=flow,
                    bridge=bool(always_bridge or width >= bridge_width_m),
                ))
    return out


def gates(lines: list[RoadLine], footprint: MultiPolygon, merge_m: float) -> list[Gate]:
    """Where each road leaves ``footprint`` for good, facing out along the road: the
    inner end of each stretch outside it that runs on to the road's end (a stretch
    between two parts of the footprint — a bridge, a bay — is no gate).  Roads that
    share a way leave by one gate, flows summed."""
    if footprint.is_empty:
        return []
    edge = footprint.boundary
    out: list[Gate] = []
    for i, road in enumerate(lines):
        ends = (road.xy[0], road.xy[-1])
        outside = LineString(road.xy).difference(footprint)
        for part in getattr(outside, "geoms", [outside]):
            if part.is_empty or not isinstance(part, LineString):
                continue
            c = np.asarray(part.coords)
            first_on, last_on = (edge.distance(Point(c[k])) < 1e-6 for k in (0, -1))
            leads_out = [any(np.linalg.norm(c[k] - e) < 1e-6 for e in ends) for k in (0, -1)]
            if first_on and leads_out[1] and not last_on:
                p, ahead = c[0], c[1]
            elif last_on and leads_out[0] and not first_on:
                p, ahead = c[-1], c[-2]
            else:
                continue
            k = _nearest_edge(road.xy, p[None])
            same = [g for g in out if np.linalg.norm(g.xy - p) < merge_m]
            if same:
                same[0].flow += float(road.flows[k])
                continue
            along = (ahead - p) / max(float(np.linalg.norm(ahead - p)), 1e-12)
            out.append(Gate(xy=p, direction=along, flow=float(road.flows[k]), road=i))
    return out
