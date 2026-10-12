"""A village's layout: farmsteads (tofts) along its streets, not a little city.

A village of ``P`` people has ``P / household_people`` households, each a toft —
house, yard and garden — fronting a street, sized so the tofts together cover
about ``toft_share`` of the site's footprint (between 300 and 3,000 m², 2.5 times
as deep as wide).  No street arm runs more than ``_ARM_CAP_M`` from the middle:
more households, more streets.  Forms:

- **harbour** (a fishing village within ``_SHORE_REACH_M`` of the sea or a lake):
  it gathers about its jetty — rows of streets parallel to the shore (offsets
  of the water's edge, so they follow it), as many as keep it about as wide as
  deep, a lane up from the jetty through them and lanes joining their ends;
  rows lengthen and multiply where the land runs out (:func:`harbour_rows`);
- otherwise, seeded: **street** (45%) — one street along the main axis (the
  footprint's long axis, or the way to a gate near it); **green** (30%) — an
  oval green with the street round it and on along the axis both ways;
  **cluster** (25%) — three to six lanes out of the middle, frontages uneven.
  A street or green village short of room adds back lanes a plot-pair behind
  its arms (:func:`back_lanes`).

Tofts go nearest the middle (the jetty) first, on dry, buildable ground as far
as the streets run, clear of each other and of every street; ~12% of slots are
skipped (gaps).  The parish church and churchyard take the slot nearest the
middle; a village of ``_MANOR_PEOPLE`` or more may have a manor beside it.
Streets end a little past the last toft; lanes run to the road gates and the
jetty (cut out of the plots they cross).  A well on the green, else mid-street.
Streets off the shore are routed on the slope like a town's.  Qt-free.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np
import shapely
import shapely.ops
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry

from settlegen.docks import Docks, build_docks
from settlegen.layout import (
    Block, SettlementLayout, SettlementLayoutParams, _dock_spec, _multi, choose_anchors, main_axis, part_containing,
    route_arterials,
)
from settlegen.profile import SettlementProfile

if TYPE_CHECKING:
    from settlegen.site import SettlementSite

FORMS = ("street", "green", "cluster")
_FORM_WEIGHTS = (0.45, 0.30, 0.25)
_MANOR_PEOPLE = 400.0
_MANOR_CHANCE = 0.5
_GAP_CHANCE = 0.12
_SHORE_REACH_M = 300.0
# A street arm runs at most this far from the middle; more people, more streets.
_ARM_CAP_M = 300.0
_MAX_ROWS = 6


def _unit(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-12)


def _plot(line: LineString, s: float, side: float, width: float, depth: float, offset: float) -> Polygon:
    """A ``width`` × ``depth`` rectangle fronting ``line`` at ``s``, ``offset`` back from its centre line."""
    a = np.asarray(line.interpolate(max(s - 4.0, 0.0)).coords[0])
    b = np.asarray(line.interpolate(min(s + 4.0, line.length)).coords[0])
    t = _unit(b - a)
    n = side * np.array([-t[1], t[0]])
    p = np.asarray(line.interpolate(s).coords[0])
    w = 0.5 * width * t
    near, far = p + offset * n, p + (offset + depth) * n
    return Polygon([near - w, near + w, far + w, far - w])


def tofts_along(
    lines: Sequence[LineString], n: int, frontage: float, depth: float, half_street: float, allowed,
    centre: np.ndarray, rng: np.random.Generator, jitter: float, taken: Sequence[Polygon] = (),
) -> list[Polygon]:
    """Up to ``n`` plots fronting ``lines``, nearest ``centre`` first, inside ``allowed``,
    clear of each other and of ``taken``."""
    slots = []
    for li, line in enumerate(lines):
        k = int(line.length // frontage)
        for j in range(k):
            s = (j + 0.5) * line.length / max(k, 1)
            p = np.asarray(line.interpolate(s).coords[0])
            for side in (1.0, -1.0):
                slots.append((float(np.linalg.norm(p - centre)) + float(rng.uniform(0.0, 0.5 * frontage)), li, s, side))
    slots.sort(key=lambda x: x[0])
    placed: list[Polygon] = []
    blocked = shapely.union_all(list(taken)) if taken else Polygon()
    for _, li, s, side in slots:
        if len(placed) >= n:
            break
        if rng.random() < _GAP_CHANCE:
            continue
        w = frontage * float(rng.uniform(1.0 - jitter, 1.0 + jitter))
        d = depth * float(rng.uniform(1.0 - 0.5 * jitter, 1.0 + 0.5 * jitter))
        poly = _plot(lines[li], s, side, w, d, half_street + 0.5).intersection(allowed)
        if not blocked.is_empty:
            poly = poly.difference(blocked)
        parts = [g for g in shapely.get_parts(poly) if isinstance(g, Polygon)]
        if not parts:
            continue
        best = max(parts, key=lambda g: g.area)
        if best.area < 0.55 * w * d:
            continue
        placed.append(best)
        blocked = blocked.union(best.buffer(0.5))
    return placed


def _trimmed(plots: Sequence[Polygon], cut, keep_all: bool = False) -> list:  # type: ignore[no-untyped-def]
    """``plots`` less ``cut``: each its largest remaining part, dropped once under half its
    size — unless ``keep_all`` (the church, the manor), which keeps any part (None if none)."""
    out: list = []
    for p in plots:
        parts = [g for g in shapely.get_parts(p.difference(cut)) if isinstance(g, Polygon)]
        best = max(parts, key=lambda g: g.area) if parts else None
        if best is not None and (keep_all or best.area >= 0.5 * p.area):
            out.append(best)
        elif keep_all:
            out.append(None)
    return out


def _shore_point(site: "SettlementSite", centre: np.ndarray) -> Optional[np.ndarray]:
    """The nearest point of the sea or a lake, if it is within reach."""
    still = shapely.union_all([site.water.sea, site.water.lakes])
    pt = Point(float(centre[0]), float(centre[1]))
    if still.is_empty or still.distance(pt) > _SHORE_REACH_M:
        return None
    return np.asarray(shapely.ops.nearest_points(still.boundary, pt)[0].coords[0])


def harbour_rows(
    still, shore: np.ndarray, households: int, frontage: float, depth: float, street_m: float,
    max_rows: int = _MAX_ROWS, allowed=None,  # type: ignore[no-untyped-def]
) -> list[LineString]:
    """A fishing village's streets: rows parallel to the shore (offsets of the water's edge,
    so they follow it), as many as keep the village about as wide as it is deep about
    its jetty at ``shore``, then a lane from the shore up through every row and lanes
    joining the rows' ends.  Where the land runs out (a headland, a narrow shore) the rows
    lengthen and multiply until their frontage on ``allowed`` would hold every household."""
    spacing = 2.0 * depth + street_m + 1.0
    setback = 0.5 * street_m + 10.0
    needed = households * frontage / 0.85                 # street frontage the tofts need
    rows_n = 2
    while rows_n < max_rows and needed / (2 * rows_n - 1) > 1.5 * rows_n * spacing:
        rows_n += 1
    length = float(np.clip(needed / (2 * rows_n - 1), 4.0 * frontage, 2.0 * _ARM_CAP_M))
    pt = Point(float(shore[0]), float(shore[1]))

    def build(rows_n: int, length: float) -> list[LineString]:
        rows: list[LineString] = []
        for k in range(rows_n):
            off = setback + k * spacing
            ring = still.buffer(off, quad_segs=16).boundary
            piece = ring.intersection(pt.buffer(math.hypot(0.5 * length, off)))
            parts = [g for g in shapely.get_parts(shapely.line_merge(piece) if not piece.is_empty else piece)
                     if isinstance(g, LineString) and g.length > 2.0 * frontage]
            if not parts:
                break
            rows.append(min(parts, key=lambda g: g.distance(pt)))
        return rows

    def frontage_of(rows: list[LineString]) -> float:
        """Street frontage with room behind it (both sides but the front row's seaward one)."""
        if allowed is None:
            return sum(r.length * (1 if k == 0 else 2) for k, r in enumerate(rows))
        total = 0.0
        for k, r in enumerate(rows):
            room = allowed.intersection(r.buffer(0.5 * street_m + depth, cap_style="flat")).area
            total += min(room / depth, r.length * (1 if k == 0 else 2))
        return total

    rows = build(rows_n, length)
    for _ in range(8):
        if not rows or frontage_of(rows) * 0.8 >= needed * 0.85:
            break
        if rows_n < max_rows:
            rows_n += 1
        length *= 1.3
        rows = build(rows_n, length)
    if not rows:
        return []
    lanes = []
    spine = [shore] + [np.asarray(shapely.ops.nearest_points(r, pt)[0].coords[0]) for r in rows]
    lanes.append(LineString(spine))
    for a, b in zip(rows[:-1], rows[1:]):
        for end in (np.asarray(a.coords[0]), np.asarray(a.coords[-1])):
            q = shapely.ops.nearest_points(b, Point(float(end[0]), float(end[1])))[0]
            if q.distance(Point(float(end[0]), float(end[1]))) < 2.0 * spacing:
                lanes.append(LineString([end, (q.x, q.y)]))
    return rows + [g for g in lanes if g.length > 1.0]


def back_lanes(
    streets: Sequence[LineString], depth: float, street_m: float, others: Sequence[LineString] = (),
    keep_out: Optional[BaseGeometry] = None,
) -> list[LineString]:
    """Lanes parallel to ``streets`` one plot-pair back on each side, and short links to them.

    Arms meeting end to end are offset as one line (no lanes crossing at the bend).
    A lane stops short of every other street (``others``, the other arms and the lanes
    already made) and of ``keep_out`` (a green); a link that would cross one is dropped."""
    gap = 2.0 * depth + street_m + 1.0
    # Stubs (the router's short spurs at the middle) would stop the arms merging there.
    merged = shapely.line_merge(shapely.union_all([g for g in streets if g.length > 4.0 * depth]))
    arms = [g for g in shapely.get_parts(merged) if isinstance(g, LineString)]
    out: list[LineString] = []
    for st in arms:
        near: list[BaseGeometry] = [g for g in [*arms, *others] if g is not st]
        if keep_out is not None and not keep_out.is_empty:
            near.append(keep_out.boundary)
        barrier = shapely.union_all(near) if near else LineString()
        cut = barrier.buffer(0.9 * gap) if not barrier.is_empty else None
        for side in (gap, -gap):
            off = st.offset_curve(side, quad_segs=8, join_style="round")
            if cut is not None:
                off = off.difference(cut)
            for g in shapely.get_parts(shapely.line_merge(off)):
                if not isinstance(g, LineString) or g.length <= 4.0 * depth:
                    continue
                out.append(g)
                for end in (np.asarray(g.coords[0]), np.asarray(g.coords[-1])):
                    q = shapely.ops.nearest_points(st, Point(float(end[0]), float(end[1])))[0]
                    link = LineString([end, (q.x, q.y)])
                    if link.length > 1.5 * gap or (not barrier.is_empty and link.crosses(barrier)) or \
                            (keep_out is not None and link.crosses(keep_out)):
                        continue
                    out.append(link)
    return out


def village_layout(
    site: "SettlementSite", profile: SettlementProfile, slope: np.ndarray, rng: np.random.Generator,
    params: SettlementLayoutParams,
) -> SettlementLayout:
    """See the module docstring."""
    footprint = site.footprint
    anchors = choose_anchors(site, profile, params)
    centre = anchors["market"]
    anchors = {"market": centre}
    pop = float(profile.population)
    households = max(int(round(pop / float(params.household_people))), 3)
    toft_m2 = float(np.clip(float(params.toft_share) * footprint.area / households, 300.0, 3000.0))
    frontage = math.sqrt(toft_m2 / 2.5)
    depth = 2.5 * frontage
    street_m = float(params.village_street_m)
    hs = 0.5 * street_m
    water = site.water.bodies
    part = part_containing(footprint, centre)
    gates = [np.asarray(g.xy, dtype=np.float64) for g in site.roads.gates]
    form = FORMS[int(rng.choice(len(FORMS), p=_FORM_WEIGHTS))]
    shore = _shore_point(site, centre) if profile.kind == "fishing" else None
    if shore is not None:
        form = "harbour"
        centre = shore                     # the village gathers about its jetty
        anchors["market"] = centre
    u = main_axis(part, centre, gates, float(params.main_street_gate_deg))
    half = site.frame.half
    roads = [ln.xy for ln in site.roads.lines]
    route = SettlementLayoutParams(**{**params.__dict__, "road_follow": 1.0})
    jitter = 0.4 if form == "cluster" else 0.2
    reach = min(households * frontage / 4.0 * 1.25 + 40.0, _ARM_CAP_M)     # both sides of one street

    def run(ends: list[np.ndarray], start: np.ndarray = centre) -> list[LineString]:
        return route_arterials(half, slope, water, footprint, roads, start, ends, route)

    green = MultiPolygon()
    arms: list[LineString] = []            # the streets back lanes may run behind
    if form == "harbour":
        still = shapely.union_all([site.water.sea, site.water.lakes])
        ground = site.buildable.difference(still.buffer(5.0)) if not still.is_empty else site.buildable
        streets = harbour_rows(still, centre, households, frontage, depth, street_m, allowed=ground)
        reach = max([float(np.max(np.linalg.norm(np.asarray(g.coords) - centre, axis=1))) for g in streets]
                    or [reach])
    elif form == "street":
        streets = run([centre + reach * u, centre - reach * u])
        arms = list(streets)
    elif form == "green":
        area = float(np.clip(8.0 * pop, 2000.0, 15000.0))
        a = math.sqrt(area * 1.8 / math.pi)
        b = area / (math.pi * a)
        theta = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
        v = np.array([-u[1], u[0]])
        ring = centre + np.outer(a * np.cos(theta), u) + np.outer(b * np.sin(theta), v)
        oval = Polygon(ring).intersection(footprint.buffer(20.0)).difference(water)
        green = _multi(oval)
        loop = [LineString(g.exterior.coords) for g in _multi(green.buffer(hs)).geoms]
        out = min(households * frontage / 0.85 / 4.0 + 20.0, _ARM_CAP_M - a)
        arms = run([centre + (a + out) * u], centre + a * u) + run([centre - (a + out) * u], centre - a * u)
        streets = loop + arms
    else:
        # Enough lanes, each at most an arm long, for every household (both sides of each).
        need = int(math.ceil(households * frontage / 0.85 / (2.0 * _ARM_CAP_M)))
        k = int(np.clip(max(int(rng.integers(3, 5)), need), 3, 6))
        base = math.atan2(u[1], u[0])
        ends = []
        for j in range(k):
            ang = base + 2.0 * math.pi * j / k + math.radians(float(rng.uniform(-25.0, 25.0)))
            lane = min(households * frontage / 0.85 / (2.0 * k) + 20.0, _ARM_CAP_M)
            ends.append(centre + lane * np.array([math.cos(ang), math.sin(ang)]))
        streets = run(ends)
    streets = [g for g in streets if g.length > 1.0]
    street_geom = shapely.union_all(streets) if streets else LineString()

    # Where tofts may go: dry, buildable ground as far out as the streets run.
    spread = max(1.6 * math.sqrt(households * toft_m2 / math.pi), reach) + depth
    allowed = site.buildable.intersection(Point(float(centre[0]), float(centre[1])).buffer(spread))
    if not water.is_empty:
        allowed = allowed.difference(water.buffer(5.0))
    allowed = allowed.difference(street_geom.buffer(hs)).difference(green)
    # The church first, nearest the middle; a manor beside it.
    special: list[tuple[str, Polygon]] = []
    church = tofts_along(streets, 1, 1.8 * frontage, 0.9 * depth, hs, allowed, centre, rng, 0.0)
    if church:
        special.append(("church", church[0]))
    if pop >= _MANOR_PEOPLE and rng.random() < _MANOR_CHANCE:
        manor = tofts_along(streets, 1, 2.5 * frontage, 1.2 * depth, hs, allowed, centre, rng, 0.0,
                            [p for _, p in special])
        if manor:
            special.append(("manor", manor[0]))
    tofts = tofts_along(streets, households, frontage, depth, hs, allowed, centre, rng, jitter,
                        [p for _, p in special])
    # Short of room on capped streets (a street or green village): back lanes a plot-pair behind, up to twice.
    for _ in range(2):
        if not arms or len(tofts) >= 0.9 * households:
            break
        others = [g for g in streets if not any(g is a for a in arms)]
        extra = [g for g in back_lanes(arms, depth, street_m, others, green)
                 if allowed.intersection(g.buffer(depth)).area > 2.0 * frontage * depth]
        if not extra:
            break
        streets = streets + extra
        arms = [g for g in extra if g.length > 4.0 * depth]
        lanes_geom = shapely.union_all(extra).buffer(hs)
        allowed = allowed.difference(lanes_geom)
        tofts = _trimmed(tofts, lanes_geom)
        tofts += tofts_along(extra, households - len(tofts), frontage, depth, hs, allowed, centre, rng, jitter,
                             [*tofts, *(p for _, p in special)])
    # The streets end a little past the last toft; lanes run on to the gates the roads come in by.
    built = shapely.union_all([*tofts, *(p for _, p in special)])
    if not built.is_empty:
        keep = built.convex_hull.buffer(frontage)
        green_lines = [g for g in streets if not green.is_empty and g.distance(green) < street_m]
        streets = green_lines + [g for st in streets if st not in green_lines
                                 for g in shapely.get_parts(st.intersection(keep))
                                 if isinstance(g, LineString) and g.length > 5.0]
    n_kept = len(streets)
    net = shapely.union_all(streets) if streets else Point(float(centre[0]), float(centre[1]))
    for g in gates:
        gp = Point(float(g[0]), float(g[1]))
        d = net.distance(gp)
        if 1.0 < d < 300.0:
            q = shapely.ops.nearest_points(net, gp)[0]
            streets.append(LineString([(q.x, q.y), (float(g[0]), float(g[1]))]))
    street_geom = shapely.union_all(streets) if streets else LineString()
    docks = Docks()
    if profile.kind == "fishing" and not water.is_empty:
        weight = max([i.weight for i in profile.industries if i.name == "fishing"] or [0.5])
        docks = build_docks(footprint, site.water, centre, [], pop, weight, 5, rng, _dock_spec(params))
        for pier in docks.piers:                  # a lane down to it
            if not street_geom.is_empty and street_geom.distance(pier) > hs:
                a, b = shapely.ops.nearest_points(street_geom, pier)
                streets.append(LineString([(a.x, a.y), (b.x, b.y)]))
        street_geom = shapely.union_all(streets) if streets else LineString()
    # The lanes added after the plots (to gates and jetties) are cut out of them.
    if len(streets) > n_kept:
        lanes = shapely.union_all(streets[n_kept:]).buffer(hs + 0.5)
        tofts = _trimmed(tofts, lanes)
        trimmed = _trimmed([p for _, p in special], lanes, keep_all=True)
        special = [(k, p) for (k, _), p in zip(special, trimmed) if p is not None]
    blocks = [Block(polygon=t, district="residential", core=True, lots=np.asarray([t], dtype=object)) for t in tofts]
    precinct = MultiPolygon()
    for kind, poly in special:
        if kind == "church":
            precinct = MultiPolygon([poly])
            anchors["sacred"] = np.asarray(poly.centroid.coords[0])
        else:
            blocks.append(Block(polygon=poly, district="elite", core=True, lots=np.asarray([poly], dtype=object)))
    well = np.asarray(green.centroid.coords[0]) if not green.is_empty else \
        np.asarray(shapely.ops.nearest_points(street_geom, Point(float(centre[0]), float(centre[1])))[0].coords[0]) \
        if not street_geom.is_empty else centre
    bridges = np.zeros((0, 2))
    if streets and not site.water.channels.is_empty:
        hit = _multi(street_geom.buffer(0.5).intersection(site.water.channels))
        bridges = np.asarray([(g.centroid.x, g.centroid.y) for g in hit.geoms]).reshape(-1, 2)
    return SettlementLayout(
        anchors=anchors, core=footprint, walls=None, citadel=MultiPolygon(), plaza=MultiPolygon(),
        precinct=precinct, arterials=streets, bridges=bridges, streets=[], blocks=blocks,
        points={"well": np.asarray(well, dtype=np.float64).reshape(1, 2), "shrine": np.zeros((0, 2))},
        arterial_m=street_m, quays=docks.quays, piers=docks.piers, green=green, fill_footprint=False,
    )
