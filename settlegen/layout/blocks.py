"""Blocks and lots: the footprint cut into blocks by minor streets, lots along their sides,
burgage plots along a town's main street, and closes and alleys opening up deep blocks."""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import shapely
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, Polygon
from shapely.ops import substring

from settlegen.layout.geometry import _axes, _multi
from settlegen.layout.params import SettlementLayoutParams


# A block this many lot depths deep has yards in its middle (``strip_lots``'s rows stop short).
_DEEP_BLOCK = 2.6


def split_blocks(
    ward: Polygon, target_m2, rng: np.random.Generator, params: SettlementLayoutParams,  # type: ignore[no-untyped-def]
    planned: bool, grid_angle: float,
) -> tuple[list[Polygon], list[LineString]]:
    """``ward`` cut by streets until every piece is about ``target_m2`` (m², or a function
    of the piece): ``(blocks, street centre lines)``."""
    blocks: list[Polygon] = []
    streets: list[LineString] = []
    half_street = 0.5 * float(params.street_m)
    goal_of = target_m2 if callable(target_m2) else (lambda _p: float(target_m2))
    jitter = float(params.block_jitter)
    stack = [(ward, float(np.exp(jitter * rng.standard_normal())))]
    while stack:
        poly, jit = stack.pop()
        goal = float(goal_of(poly)) * jit  # type: ignore[operator]
        if poly.area <= goal * 1.4:
            blocks.append(poly)
            continue
        centre, u, v, length, depth = _axes(poly)
        if planned:
            # The grid direction nearest the long axis.
            ga = np.array([math.cos(grid_angle), math.sin(grid_angle)])
            gb = np.array([-ga[1], ga[0]])
            u = ga if abs(float(u @ ga)) >= abs(float(u @ gb)) else gb
            offset = 0.0
        else:
            a = math.radians(float(params.organic_angle_deg)) * rng.uniform(-1.0, 1.0)
            u = np.array([u[0] * math.cos(a) - u[1] * math.sin(a), u[0] * math.sin(a) + u[1] * math.cos(a)])
            offset = float(params.organic_offset) * rng.uniform(-1.0, 1.0) * 0.5 * length
        cut_dir = np.array([-u[1], u[0]])
        at = centre + offset * u
        reach = length + depth
        cut = LineString([at - reach * cut_dir, at + reach * cut_dir])
        pieces = _multi(poly.difference(cut.buffer(half_street, cap_style="flat")))
        if len(pieces.geoms) < 2:
            blocks.append(poly)
            continue
        street = cut.intersection(poly)
        for s in getattr(street, "geoms", [street]):
            if isinstance(s, LineString) and s.length > 0:
                streets.append(s)
        for piece in pieces.geoms:
            if piece.area >= 0.12 * goal:
                stack.append((piece, jit))
    return blocks, streets


def strip_lots(
    block: Polygon, frontage: float, depth: float, rng: np.random.Generator, planned: bool,
) -> tuple[np.ndarray, MultiLineString]:
    """Strips along ``block``'s long sides: ``frontage`` wide (jittered unless ``planned``),
    ``depth`` deep (one row through a thin block, two back to back in a middling one; a
    deep one's middle left as yards).  ``(lots, the lines between them and along the rows'
    backs, inside the block)``."""
    centre, u, v, length, d = _axes(block)
    if length <= 0 or d <= 0:
        return np.zeros(0, dtype=object), MultiLineString()
    if d < 1.3 * depth:
        rows = [(-0.5 * d, 0.5 * d)]
        backs: list[float] = []
    elif d < 2.6 * depth:
        rows = [(-0.5 * d, 0.0), (0.0, 0.5 * d)]
        backs = [0.0]
    else:
        rows = [(-0.5 * d, -0.5 * d + depth), (0.5 * d - depth, 0.5 * d)]
        backs = [-0.5 * d + depth, 0.5 * d - depth]
    boxes = []
    cuts = []
    j = 0.05 if planned else 0.3
    for b0, b1 in rows:
        a = -0.5 * length
        while a < 0.5 * length - 0.3 * frontage:
            w = frontage * rng.uniform(1.0 - j, 1.0 + j)
            a1 = min(a + w, 0.5 * length)
            if 0.5 * length - a1 < 0.5 * frontage:
                a1 = 0.5 * length
            corners = [(a, b0), (a1, b0), (a1, b1), (a, b1), (a, b0)]
            boxes.append([centre + p * u + q * v for p, q in corners])
            if a1 < 0.5 * length:
                cuts.append([centre + a1 * u + b0 * v, centre + a1 * u + b1 * v])
            a = a1
    if not boxes:
        return np.zeros(0, dtype=object), MultiLineString()
    rects = shapely.polygons(np.asarray(boxes))
    lots = np.asarray(shapely.intersection(rects, block), dtype=object)
    area = shapely.area(lots)
    keep = area >= 0.3 * frontage * min(depth, d)
    kept = np.asarray([g if isinstance(g, Polygon) else max(shapely.get_parts(g), key=lambda p: p.area)
                       for g in lots[keep] if not g.is_empty], dtype=object)
    reach = 0.5 * length + 1.0
    cuts += [[centre - reach * u + q * v, centre + reach * u + q * v] for q in backs]
    if not cuts:
        return kept, MultiLineString()
    segs = np.asarray(shapely.linestrings(np.asarray(cuts)), dtype=object)
    lines = shapely.intersection(shapely.multilinestrings(segs), block)
    parts = shapely.get_parts(lines)
    keep_line = (shapely.get_type_id(parts) == 1) & (shapely.length(parts) > 1e-6)
    return kept, MultiLineString(list(parts[keep_line]))


def _run_on(line: LineString, by: float, start: bool = True, end: bool = True) -> LineString:
    """``line`` carried straight on ``by`` m past its ``start`` and/or ``end``: a lane cut
    through to the street even where it meets the block's edge off square."""
    xy = np.asarray(line.coords, dtype=np.float64)
    if start:
        t = xy[0] - xy[1]
        xy = np.vstack([xy[0] + by * t / max(float(np.linalg.norm(t)), 1e-9), xy])
    if end:
        t = xy[-1] - xy[-2]
        xy = np.vstack([xy, xy[-1] + by * t / max(float(np.linalg.norm(t)), 1e-9)])
    return LineString(xy)


def lanes_of(
    block: Polygon, depth: float, rng: np.random.Generator, params: SettlementLayoutParams, planned: bool,
) -> tuple[list[Polygon], list[LineString], list[LineString]]:
    """Lanes into ``block`` if it is too deep for ``depth``-deep lots round its edge:
    ``(pieces, closes, alleys)``.  Organic places cut lanes in across the block every
    ``lane_spacing_block_m`` from a random long side — closes stopping a lot short of the far
    side, or (``alley_share``) alleys through; a ``planned`` one runs a back alley down the
    middle, between the back-to-back rows.  A block too shallow keeps itself, lane-less."""
    centre, u, v, length, d = _axes(block)
    if d < _DEEP_BLOCK * depth or length < 2.0 * depth:
        return [block], [], []
    half = 0.5 * float(params.lane_m)
    reach = length + d
    run = 2.0 * float(params.lane_m) + 2.0          # past the block's edge at an open end
    if planned:
        cut = LineString([centre - reach * u, centre + reach * u]).intersection(block)
        alleys = [g for g in shapely.get_parts(cut) if isinstance(g, LineString) and g.length > depth]
        if not alleys:
            return [block], [], []
        cuts = [_run_on(g, run) for g in alleys]
        pieces = _multi(block.difference(shapely.union_all(cuts).buffer(half, cap_style="flat")))
        return list(pieces.geoms), [], alleys
    n = max(1, int(round(length / float(params.lane_spacing_block_m))))
    span = length - 1.2 * depth                         # keep a lot's depth clear of each end
    if span <= 0:
        return [block], [], []
    closes: list[LineString] = []
    alleys = []
    exterior = block.exterior
    for i in range(n):
        a = -0.5 * span + (i + 0.5 + 0.3 * rng.uniform(-1.0, 1.0)) * span / n
        side = 1.0 if rng.random() < 0.5 else -1.0
        tilt = math.radians(8.0) * rng.uniform(-1.0, 1.0)
        w = -side * (v * math.cos(tilt) + u * math.sin(tilt))      # inward, from the ``side`` edge
        start = centre + a * u + side * (0.5 * d + 10.0) * v
        line = LineString([start, start + (d + 20.0) * w]).intersection(block)
        parts = [g for g in shapely.get_parts(line) if isinstance(g, LineString) and g.length > 1.0]
        if not parts:
            continue
        first = min(parts, key=lambda g: g.distance(Point(start)))
        xy = np.asarray(first.coords)
        if np.linalg.norm(xy[0] - start) > np.linalg.norm(xy[-1] - start):
            xy = xy[::-1]
        lane = LineString(xy)
        if exterior.distance(Point(xy[0])) > 1.0 or lane.length < 1.5 * depth:
            continue
        through = rng.random() < float(params.alley_share) and exterior.distance(Point(xy[-1])) < 1.0
        if through:
            alleys.append(lane)
        else:
            close = substring(lane, 0.0, lane.length - depth)
            if isinstance(close, LineString):
                closes.append(close)
    if not closes and not alleys:
        return [block], [], []
    cuts = [_run_on(g, run, end=False) for g in closes] + [_run_on(g, run) for g in alleys]
    cut = shapely.union_all(cuts).buffer(half, cap_style="flat")
    pieces = [g for g in _multi(block.difference(cut)).geoms if g.area > 0.5 * depth * depth]
    return pieces, closes, alleys


def yards_of(block: Polygon, depth: float):  # type: ignore[no-untyped-def]
    """What :func:`strip_lots`'s ``depth``-deep rows leave in the middle of ``block`` (None
    when they fill it)."""
    centre, u, v, length, d = _axes(block)
    if d < _DEEP_BLOCK * depth:
        return None
    a, b = 0.5 * length + 1.0, 0.5 * d - depth
    rect = Polygon([centre + p * u + q * v for p, q in ((-a, -b), (a, -b), (a, b), (-a, b))])
    yards = block.intersection(rect)
    return None if yards.is_empty else yards


def lane_lots(
    yards, lanes: Sequence[LineString], frontage: float, depth: float, half_lane: float,  # type: ignore[no-untyped-def]
    rng: np.random.Generator, planned: bool,
) -> np.ndarray:
    """Lots fronting ``lanes`` in a block's ``yards`` (see :func:`yards_of`): ``frontage``
    along the lane, up to ``depth`` back from it either side, and one facing down from a
    close's head."""
    free = yards
    j = 0.05 if planned else 0.3
    boxes = []
    for lane in lanes:                              # straight, from the street inward
        xy = np.asarray(lane.coords)
        length = float(np.linalg.norm(xy[-1] - xy[0]))
        t = (xy[-1] - xy[0]) / max(length, 1e-9)
        nrm = np.array([-t[1], t[0]])
        a = 0.5 * depth                             # the street's rows take what they hold
        while a < length - 0.3 * frontage:
            a1 = min(a + frontage * rng.uniform(1.0 - j, 1.0 + j), length)
            p0, p1 = xy[0] + a * t, xy[0] + a1 * t
            for s in (1.0, -1.0):
                q0, q1 = p0 + s * half_lane * nrm, p1 + s * half_lane * nrm
                boxes.append([q0, q1, q1 + s * depth * nrm, q0 + s * depth * nrm, q0])
            a = a1
        if not free.contains(Point(xy[-1] + 2.0 * half_lane * t)):
            continue                                # an alley: out onto the street
        head = xy[-1] + half_lane * t               # a close's head: a lot facing down it
        boxes.append([head - frontage * nrm, head + frontage * nrm, head + frontage * nrm + depth * t,
                      head - frontage * nrm + depth * t, head - frontage * nrm])
    if not boxes:
        return np.zeros(0, dtype=object)
    cut = np.asarray(shapely.intersection(shapely.polygons(np.asarray(boxes)), free), dtype=object)
    out: list[Polygon] = []
    for g in cut:                                   # first come first served where two lanes' lots meet
        if g.is_empty or g.area < 0.4 * frontage * depth:
            continue
        hits = [o for o in out if o.intersects(g)]
        if hits:
            g = g.difference(shapely.union_all(hits))
        parts = [q for q in shapely.get_parts(g) if isinstance(q, Polygon) and not q.is_empty]
        if parts:
            q = max(parts, key=lambda q: q.area)
            if q.area >= 0.4 * frontage * depth:
                out.append(q)
    return np.asarray(out, dtype=object)


def burgage_band(
    main: Sequence[LineString], land: MultiPolygon, rng: np.random.Generator, params: SettlementLayoutParams,
) -> tuple[list[Polygon], MultiPolygon, list[LineString]]:
    """Plots along the main street: a band ``burgage_depth_m`` deep each side, cut by cross
    lanes every ``lane_spacing_m`` (running on past it) and closed by a back lane.
    ``(band blocks, the land beyond, the lanes)``."""
    main_geom = shapely.union_all(list(main))
    half_art, depth, hs = 0.5 * float(params.arterial_m), float(params.burgage_depth_m), 0.5 * float(params.street_m)
    lanes: list[LineString] = []
    reach = half_art + depth + 120.0
    for line in main:
        s = float(rng.uniform(0.3, 0.7)) * float(params.lane_spacing_m)
        while s < line.length - 20.0:
            a = np.asarray(line.interpolate(max(s - 5.0, 0.0)).coords[0])
            b = np.asarray(line.interpolate(min(s + 5.0, line.length)).coords[0])
            t = (b - a) / max(float(np.linalg.norm(b - a)), 1e-9)
            n = np.array([-t[1], t[0]])
            p = np.asarray(line.interpolate(s).coords[0])
            lanes.append(LineString([p - reach * n, p + reach * n]))
            s += float(params.lane_spacing_m) * float(rng.uniform(0.8, 1.2))
    back = main_geom.buffer(half_art + depth + hs).boundary
    cutters = shapely.union_all([*lanes, back]).buffer(hs, cap_style="flat")
    land2 = land.difference(cutters)
    band = _multi(land2.intersection(main_geom.buffer(half_art + depth + 0.01)))
    rest = _multi(land2.difference(main_geom.buffer(half_art + depth + 2.0 * hs)))
    min_m2 = 2.0 * float(params.burgage_frontage_m) * depth
    blocks = [g for g in band.geoms if g.area >= min_m2]
    streets = [g for g in shapely.get_parts(shapely.union_all([*lanes, back]).intersection(land))
               if isinstance(g, LineString) and g.length > 10.0]
    return blocks, rest, streets
