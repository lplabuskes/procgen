"""A settlement's layout: walls, streets, districts and lots (plan phase 6).

:func:`build_layout` reads only the :class:`SettlementSite`, the
:class:`SettlementProfile` and the ground's slope, so it runs on hand-built sites.
Block subdivision (user choice, 2026-10-11), planned by tier — a city, a town and
a village are different places, not one plan at three sizes.  Villages:
:mod:`settlegen.village_layout`.  Towns and cities, in order:

1. **Anchors** from the site's candidates by profile: always a market; a
   citadel for a capital, a march (``march`` ≥ ``citadel_march``) or a strongly
   walled town or city; a waterfront where it ships or fishes; a sacred site
   apart from the others.
2. **Main streets**, least-cost on a ``route_step_m`` grid (slope, water, off
   the footprint; along the site's roads cheaply — they hold the river
   crossings — but not led by them).  A **town**'s main street runs through its
   market along the footprint's long axis (or to a gate near that line); a
   **city**'s run out of its market to every gate, and to the footprint's edge
   in the widest gaps between them until at least ``radial_streets`` leave it.
   Both reach the waterfront, citadel and sacred site.  Bridges where they
   cross a channel.
3. **Market**: a town's is its main street widened (``market_street_m``) about
   the market; a city's a plaza.  The citadel and the sacred precinct are open
   ground sized by population.
4. **Old core**: a smooth blob about the market (and citadel) holding
   ``core_share`` of the footprint (more for strong walls, young places and
   towns).  Only a walled place shows it: walls (``palisade`` or ``stone``,
   towers on stone) run round it, open along the water, with suburbs outside.
   Unwalled, blocks and lots just grow coarser toward the edge.
5. **Docks** (:mod:`settlegen.docks`) where there is a waterfront: quays along
   the shore, piers into the water.
6. **Blocks**: a town's main street is lined with burgage plots
   (``burgage_depth_m`` deep, cut by cross lanes every ``lane_spacing_m``,
   closed by a back lane); the rest of the footprint — less streets, walls,
   quays and open ground — is cut recursively across its long axis by minor
   streets into blocks (``block_ha``; a town's × ``town_block_mult``).  Organic
   places jitter the cuts; planned ones (``planned`` stub) cut on a grid.
7. **Districts** per block: market, quay, elite (near the citadel, upwind),
   craft quarters from ``industries`` (noxious trades downwind, downstream,
   outside the walls; water trades on the water), graveyards, squares; the rest
   houses (suburb outside walls, gardens on sparse ground).  A town keeps to
   its market, quay, two leading trades, its churchyard and — only by a castle
   — an elite quarter.
8. **Lots**: strips along each block's long sides; wells in the core, shrines
   along the streets.

:func:`describe_at` names what is at a point (for hover).  No names
(Decision 9).  Seeded per (world, settlement).  Qt-free.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, Polygon

from core.params import param_field as _f, paramclass
from settlegen.docks import Docks, DockSpec, build_docks
from settlegen.profile import SettlementProfile

if TYPE_CHECKING:
    from settlegen.site import SettlementSite

_LAYOUT_STREAM = 61
# (min, max) m² of the open grounds.
_PLAZA_M2, _CITADEL_M2, _PRECINCT_M2 = (300.0, 15000.0), (1500.0, 60000.0), (300.0, 12000.0)

# Craft quarters: industry → (district, noxious, by the water).
CRAFTS: dict[str, tuple[str, bool, bool]] = {
    "tanning": ("tanners", True, True),
    "dyeing": ("dyers", True, True),
    "smithing": ("smiths", True, False),
    "butchery": ("butchers", True, True),
    "brewing": ("brewers", False, True),
    "milling": ("millers", False, True),
    "sugar": ("sugar mills", False, True),
    "weaving": ("weavers", False, False),
    "timber": ("carpenters", False, False),
    "shipping": ("shipwrights", False, True),
    "salt works": ("salters", False, True),
    "fishing": ("fishers", False, True),
    "carving": ("artisans", False, False),
    "pearling": ("artisans", False, True),
}
# Districts with no lots.
OPEN_DISTRICTS = ("graveyard", "square")
DISTRICTS = ("market", "quay", "elite", "residential", "suburb", "gardens", "graveyard", "square",
             *sorted({v[0] for v in CRAFTS.values()}))


@paramclass
class SettlementLayoutParams:
    """Tuning for the layout."""

    citadel_march: float = _f(0.3, label="March that builds a citadel", min=0.0, max=1.0, step=0.05)
    citadel_wall_strength: float = _f(0.6, label="Wall strength that builds a citadel", min=0.0, max=2.0, step=0.05)
    core_share: float = _f(0.45, label="Old core share of the footprint", min=0.05, max=1.0, step=0.05)
    core_wall_gain: float = _f(0.25, label="Core share per wall strength", min=0.0, max=1.0, step=0.05)
    core_young_gain: float = _f(0.2, label="Core share for a young place", min=0.0, max=1.0, step=0.05)
    young_years: float = _f(300.0, label="Young below (years)", min=10.0, max=3000.0, step=10.0)
    core_wobble: float = _f(0.12, label="Core outline wobble", min=0.0, max=0.5, step=0.01)
    stone_wall_strength: float = _f(0.4, label="Stone walls from strength", min=0.0, max=2.0, step=0.05)
    wall_m: float = _f(5.0, label="Wall width (m)", min=1.0, max=30.0, step=0.5)
    wall_water_m: float = _f(15.0, label="No wall within (m) of water", min=0.0, max=200.0, step=1.0)
    tower_spacing_m: float = _f(70.0, label="Tower spacing at strength 1 (m)", min=10.0, max=500.0, step=5.0)
    # Open grounds scale with people, within _PLAZA_M2 / _CITADEL_M2 / _PRECINCT_M2.
    plaza_m2_per_person: float = _f(0.08, label="Market plaza (m² per person)", min=0.0, max=2.0, step=0.01)
    citadel_m2_per_person: float = _f(0.15, label="Citadel (m² per person)", min=0.0, max=5.0, step=0.01)
    precinct_m2_per_person: float = _f(0.05, label="Sacred precinct (m² per person)", min=0.0, max=2.0, step=0.01)
    route_step_m: float = _f(20.0, label="Arterial routing grid (m)", min=5.0, max=100.0, step=1.0)
    slope_weight: float = _f(10.0, label="Arterial slope cost", min=0.0, max=100.0, step=1.0)
    water_cost: float = _f(40.0, label="Arterial water cost (×)", min=1.0, max=500.0, step=1.0)
    road_follow: float = _f(0.6, label="Arterial cost along roads (×)", min=0.1, max=1.0, step=0.05)
    outside_cost: float = _f(1.5, label="Arterial cost off the footprint (×)", min=1.0, max=10.0, step=0.1)
    arterial_m: float = _f(10.0, label="Arterial width (m)", min=2.0, max=40.0, step=0.5)
    street_m: float = _f(5.0, label="Street width (m)", min=1.0, max=20.0, step=0.5)
    block_ha: float = _f(0.35, label="Core block (ha)", min=0.02, max=5.0, step=0.01)
    suburb_block_mult: float = _f(2.0, label="Suburb block (×)", min=1.0, max=10.0, step=0.1)
    sparse_block_mult: float = _f(4.0, label="Sparse-ground block (×)", min=1.0, max=20.0, step=0.1)
    block_jitter: float = _f(0.35, label="Block size jitter (lognormal σ)", min=0.0, max=1.0, step=0.05)
    organic_offset: float = _f(0.25, label="Organic cut offset (share)", min=0.0, max=0.5, step=0.01)
    organic_angle_deg: float = _f(14.0, label="Organic cut angle jitter (°)", min=0.0, max=45.0, step=1.0)
    lot_frontage_m: float = _f(8.0, label="Lot frontage (m)", min=2.0, max=50.0, step=0.5)
    lot_depth_m: float = _f(22.0, label="Lot depth (m)", min=5.0, max=100.0, step=1.0)
    suburb_lot_mult: float = _f(1.5, label="Suburb lot (×)", min=1.0, max=5.0, step=0.1)
    garden_lot_mult: float = _f(3.0, label="Garden lot (×)", min=1.0, max=10.0, step=0.1)
    elite_share: float = _f(0.12, label="Elite share of core blocks", min=0.0, max=0.5, step=0.01)
    craft_share: float = _f(0.06, label="Craft quarter share of blocks (per weight)", min=0.0, max=0.5, step=0.01)
    quay_m: float = _f(40.0, label="Quay reach from the water (m)", min=5.0, max=200.0, step=5.0)
    graveyard_people: float = _f(20000.0, label="People per graveyard", min=500.0, max=100000.0, step=500.0)
    square_people: float = _f(12000.0, label="People per small square", min=500.0, max=100000.0, step=500.0)
    shrine_people: float = _f(1500.0, label="People per shrine", min=100.0, max=50000.0, step=100.0)
    well_ha: float = _f(1.5, label="Core hectares per well", min=0.1, max=20.0, step=0.1)
    radial_streets: int = _f(3, label="Main streets out of a city's market (at least)", min=0, max=12)
    # Towns: a main street through the market, burgage plots along it.
    town_block_mult: float = _f(1.5, label="Town block (×)", min=0.5, max=10.0, step=0.1)
    main_street_gate_deg: float = _f(35.0, label="Main street takes a gate within (°)", min=0.0, max=90.0, step=1.0)
    burgage_depth_m: float = _f(45.0, label="Burgage plot depth (m)", min=10.0, max=150.0, step=1.0)
    burgage_frontage_m: float = _f(7.0, label="Burgage plot frontage (m)", min=2.0, max=30.0, step=0.5)
    lane_spacing_m: float = _f(140.0, label="Cross-lane spacing (m)", min=40.0, max=500.0, step=5.0)
    market_street_m: float = _f(24.0, label="Town market street width (m)", min=5.0, max=80.0, step=1.0)
    # Villages: tofts along a street, round a green or along a few lanes.
    household_people: float = _f(5.0, label="People per household", min=1.0, max=20.0, step=0.5)
    toft_share: float = _f(0.85, label="Tofts' share of a village footprint", min=0.1, max=2.0, step=0.05)
    village_street_m: float = _f(6.0, label="Village street width (m)", min=2.0, max=20.0, step=0.5)
    # Docks.
    quay_width_m: float = _f(12.0, label="Quay width (m)", min=2.0, max=60.0, step=1.0)
    pier_spacing_m: float = _f(60.0, label="Pier spacing (m)", min=10.0, max=500.0, step=5.0)
    pier_length_m: float = _f(60.0, label="Pier length on still water (m)", min=5.0, max=300.0, step=5.0)
    pier_width_m: float = _f(10.0, label="Pier width (m)", min=2.0, max=40.0, step=1.0)
    pier_min_river_m: float = _f(40.0, label="No piers on rivers narrower than (m)", min=0.0, max=500.0, step=5.0)


@dataclass
class Walls:
    """The core's walls."""

    kind: str                    # "palisade" / "stone"
    line: MultiLineString        # open along the water
    towers: np.ndarray           # (n, 2)
    gates: np.ndarray            # (n, 2): where arterials pass
    width_m: float = 5.0


@dataclass
class Block:
    """One block between streets."""

    polygon: Polygon
    district: str
    core: bool                   # inside the old core
    lots: np.ndarray             # (k,) shapely polygons


@dataclass
class SettlementLayout:
    """See the module docstring.  All in frame metres."""

    anchors: dict[str, np.ndarray]           # kind → (2,)
    core: MultiPolygon
    walls: Optional[Walls]
    citadel: MultiPolygon
    plaza: MultiPolygon
    precinct: MultiPolygon
    arterials: list[LineString]
    bridges: np.ndarray                      # (n, 2)
    streets: list[LineString]                # minor streets' centre lines
    blocks: list[Block]
    points: dict[str, np.ndarray] = field(default_factory=dict)   # wells, shrines: (n, 2)
    arterial_m: float = 10.0
    quays: MultiPolygon = field(default_factory=MultiPolygon)
    piers: list[Polygon] = field(default_factory=list)
    green: MultiPolygon = field(default_factory=MultiPolygon)        # a village green
    fill_footprint: bool = True              # the footprint is streets between the blocks (not a village)

    @staticmethod
    def empty() -> "SettlementLayout":
        e = MultiPolygon()
        return SettlementLayout({}, e, None, e, e, e, [], np.zeros((0, 2)), [], [])

    def district_areas_ha(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for b in self.blocks:
            out[b.district] = out.get(b.district, 0.0) + b.polygon.area / 1e4
        return out

    @property
    def n_lots(self) -> int:
        return int(sum(len(b.lots) for b in self.blocks))


# ── geometry helpers ─────────────────────────────────────────────────────────

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


# ── arterials ────────────────────────────────────────────────────────────────

def _grid_graph(cost: np.ndarray, step_m: float):  # type: ignore[no-untyped-def]
    n_r, n_c = cost.shape
    idx = np.arange(n_r * n_c).reshape(n_r, n_c)
    src, dst, w = [], [], []
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        r0, r1 = 0, n_r - dr
        c0, c1 = max(0, -dc), n_c - max(0, dc)
        a = idx[r0:r1, c0:c1]
        b = idx[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
        e = 0.5 * (cost[r0:r1, c0:c1] + cost[r0 + dr:r1 + dr, c0 + dc:c1 + dc]) * math.hypot(dr, dc) * step_m
        src.append(a.reshape(-1))
        dst.append(b.reshape(-1))
        w.append(e.reshape(-1))
    n = n_r * n_c
    return coo_matrix((np.concatenate(w), (np.concatenate(src), np.concatenate(dst))), shape=(n, n)).tocsr()


def route_arterials(
    half: float, slope: np.ndarray, wet: MultiPolygon, footprint: MultiPolygon, roads: Sequence[np.ndarray],
    source: np.ndarray, targets: Sequence[np.ndarray], params: SettlementLayoutParams,
) -> list[LineString]:
    """Least-cost paths from ``source`` to each target over a ``route_step_m`` grid of the
    ``2·half`` frame (``slope``: the frame's (T, T) slope), merged into one smoothed network."""
    if not targets:
        return []
    step = float(params.route_step_m)
    n = max(int(math.ceil(2.0 * half / step)), 2)
    step = 2.0 * half / n
    c = -half + (np.arange(n) + 0.5) * step
    xx, yy = np.meshgrid(c, c[::-1])
    t = slope.shape[0]
    rows = np.clip(((half - yy) / (2.0 * half) * t).astype(int), 0, t - 1)
    cols = np.clip(((xx + half) / (2.0 * half) * t).astype(int), 0, t - 1)
    cost = 1.0 + float(params.slope_weight) * slope[rows, cols]
    on_road = np.zeros((n, n), dtype=bool)
    for xy in roads:
        xy = np.asarray(xy, dtype=np.float64)
        if len(xy) < 2:
            continue
        line = LineString(xy)
        pts = np.asarray([(p.x, p.y) for p in shapely.line_interpolate_point(
            line, np.append(np.arange(0.0, line.length, 0.4 * step), line.length))]).reshape(-1, 2)
        r = ((half - pts[:, 1]) / step).astype(int)
        q = ((pts[:, 0] + half) / step).astype(int)
        ok = (r >= 0) & (r < n) & (q >= 0) & (q < n)
        on_road[r[ok], q[ok]] = True
    if not wet.is_empty:
        is_wet = shapely.contains_xy(wet, xx, yy)
        cost = np.where(is_wet & ~on_road, cost * float(params.water_cost), cost)
    if not footprint.is_empty:
        cost = np.where(shapely.contains_xy(footprint, xx, yy), cost, cost * float(params.outside_cost))
    cost = np.where(on_road, cost * float(params.road_follow), cost)
    graph = _grid_graph(cost, step)

    def cell(p: np.ndarray) -> int:
        r = int(np.clip((half - p[1]) / step, 0, n - 1))
        q = int(np.clip((p[0] + half) / step, 0, n - 1))
        return r * n + q

    s = cell(source)
    _, pred = dijkstra(graph, directed=False, indices=s, return_predecessors=True)
    centres = np.column_stack([xx.reshape(-1), yy.reshape(-1)])
    segs: list[LineString] = []
    for target in targets:
        k = cell(target)
        path = [k]
        while path[-1] != s and pred[path[-1]] >= 0:
            path.append(int(pred[path[-1]]))
        if path[-1] != s:
            continue
        xy = centres[path]
        segs.append(LineString([target, xy[0]]))
        segs.extend(LineString([xy[i], xy[i + 1]]) for i in range(len(xy) - 1))
    segs.append(LineString([centres[s], source]))
    segs = [g for g in segs if g.length > 1e-6]
    if not segs:
        return []
    merged = shapely.line_merge(shapely.union_all(segs))
    # The source and targets stay ends (a merge can run a line through one, and smoothing would cut it).
    pins = np.asarray([source, *targets], dtype=np.float64).reshape(-1, 2)
    out = []
    for line in getattr(merged, "geoms", [merged]):
        xy = np.asarray(line.coords)
        hit = np.flatnonzero(np.min(np.linalg.norm(xy[1:-1, None, :] - pins[None], axis=2), axis=1) < 1e-6) + 1 \
            if len(xy) > 2 else np.zeros(0, dtype=np.int64)
        for a, b in zip([0, *hit], [*hit, len(xy) - 1]):
            piece = LineString(xy[a:b + 1])
            if piece.length > 1e-6:
                out.append(LineString(chaikin(np.asarray(piece.simplify(0.6 * step).coords), 3)))
    return out


# ── blocks and lots ──────────────────────────────────────────────────────────

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


def lots_of(block: Polygon, frontage: float, depth: float, rng: np.random.Generator, planned: bool) -> np.ndarray:
    """Strips along ``block``'s long sides: ``frontage`` wide (jittered unless ``planned``),
    ``depth`` deep (one row through a thin block, two back to back in a middling one; a
    deep one's middle left as yards)."""
    centre, u, v, length, d = _axes(block)
    if length <= 0 or d <= 0:
        return np.zeros(0, dtype=object)
    if d < 1.3 * depth:
        rows = [(-0.5 * d, 0.5 * d)]
    elif d < 2.6 * depth:
        rows = [(-0.5 * d, 0.0), (0.0, 0.5 * d)]
    else:
        rows = [(-0.5 * d, -0.5 * d + depth), (0.5 * d - depth, 0.5 * d)]
    boxes = []
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
            a = a1
    if not boxes:
        return np.zeros(0, dtype=object)
    rects = shapely.polygons(np.asarray(boxes))
    lots = np.asarray(shapely.intersection(rects, block), dtype=object)
    area = shapely.area(lots)
    keep = area >= 0.3 * frontage * min(depth, d)
    return np.asarray([g if isinstance(g, Polygon) else max(shapely.get_parts(g), key=lambda p: p.area)
                       for g in lots[keep] if not g.is_empty], dtype=object)


# ── districts ────────────────────────────────────────────────────────────────

def _nearest_cluster(score: np.ndarray, centroids: np.ndarray, free: np.ndarray, n: int) -> np.ndarray:
    """Up to ``n`` free blocks: the best-scoring one and those nearest it."""
    idx = np.flatnonzero(free & np.isfinite(score))
    if idx.size == 0 or n <= 0:
        return np.zeros(0, dtype=np.int64)
    seed = idx[int(np.argmax(score[idx]))]
    d = np.linalg.norm(centroids[idx] - centroids[seed], axis=1)
    return idx[np.argsort(d, kind="stable")[:n]]


def assign_districts(
    site: "SettlementSite", profile: SettlementProfile, blocks: list[Polygon], in_core: np.ndarray,
    anchors: dict[str, np.ndarray], plaza: MultiPolygon, precinct: MultiPolygon, rng: np.random.Generator,
    params: SettlementLayoutParams, walled: bool = True,
) -> list[str]:
    """Each block's district.  See the module docstring: a town keeps to a market,
    a quay, its two leading trades, an elite quarter only by a castle and its
    churchyard; suburbs lie only outside walls."""
    n = len(blocks)
    district = np.array(["" for _ in range(n)], dtype=object)
    if n == 0:
        return []
    geoms = np.asarray(blocks, dtype=object)
    centroids = np.asarray([(p.x, p.y) for p in shapely.centroid(geoms)])
    centre = np.asarray(anchors["market"] if "market" in anchors else centroids.mean(axis=0), dtype=np.float64)
    radius = max(math.sqrt(max(site.footprint.area, 1.0) / math.pi), 1.0)
    wind = np.asarray(profile.wind_xy, dtype=np.float64)
    speed = float(np.linalg.norm(wind))
    downwind = (centroids - centre) @ (wind / speed) / radius if speed > 0.5 else np.zeros(n)
    water = site.water.bodies
    d_water = shapely.distance(geoms, water) if not water.is_empty else np.full(n, np.inf)
    downstream = np.zeros(n)
    rivers = [ln for ln in site.water.lines if len(ln.xy) >= 2 and not ln.minor]
    if rivers:
        main = max(rivers, key=lambda ln: float(np.mean(ln.width_m)))
        line = LineString(main.xy)
        s = shapely.line_locate_point(line, shapely.points(centroids))
        downstream = (s - line.project(Point(float(centre[0]), float(centre[1])))) / radius
        downstream = np.where(shapely.distance(line, shapely.points(centroids)) < radius, downstream, 0.0)
    sparse = site.sparse
    in_sparse = shapely.contains_xy(sparse, centroids[:, 0], centroids[:, 1]) if not sparse.is_empty \
        else np.zeros(n, dtype=bool)
    free = np.ones(n, dtype=bool)
    village = profile.tier == 5
    town = profile.tier == 4

    def take(idx: np.ndarray, name: str) -> None:
        idx = np.asarray(idx, dtype=np.int64)
        idx = idx[free[idx]]
        district[idx] = name
        free[idx] = False

    # Market: round the plaza.
    if not plaza.is_empty:
        d = shapely.distance(geoms, plaza)
        near = np.flatnonzero(d <= 1.5 * float(params.street_m) + 1.0)
        take(near[np.argsort(d[near], kind="stable")][:8 if town else 12], "market")
    # Quay: along the water by the waterfront.
    if "waterfront" in anchors and not village:
        reach = float(np.clip(200.0 * math.sqrt(profile.population / 5000.0), 150.0, 1200.0))
        near = np.linalg.norm(centroids - anchors["waterfront"], axis=1) <= reach
        take(np.flatnonzero(near & (d_water <= float(params.quay_m))), "quay")
    # Elite: core, near the citadel (else the precinct, the market), upwind.
    if not village and (not town or "citadel" in anchors):
        focus = anchors.get("citadel", anchors.get("sacred", centre))
        score = -np.linalg.norm(centroids - focus, axis=1) / radius - 0.5 * downwind
        score = np.where(in_core & free, score, -np.inf)
        k = int(round(float(params.elite_share) * int(in_core.sum())))
        take(_nearest_cluster(score, centroids, free & in_core, min(k, 4) if town else k), "elite")
    # Craft quarters.
    crafts = [(i.name, i.weight) for i in profile.industries if i.name in CRAFTS]
    if town:
        crafts = [c for c in sorted(crafts, key=lambda c: -c[1]) if c[1] >= 0.4][:2]
    elif not village:
        crafts.append(("butchery", 0.5))
    for name, weight in crafts:
        quarter, noxious, wet = CRAFTS[name]
        if quarter == "shipwrights" and "waterfront" not in anchors:
            continue
        k = int(round(weight * float(params.craft_share) * n))
        if k == 0 and weight >= 0.3:
            k = 1
        score = rng.uniform(0.0, 0.1, n) - np.linalg.norm(centroids - centre, axis=1) / (3.0 * radius)
        if noxious:
            score += downwind + 0.5 * downstream + 0.5 * ~in_core
        if wet:
            score += np.where(d_water <= float(params.quay_m), 1.0, -1.0)
        take(_nearest_cluster(score, centroids, free, k), quarter)
    # Graveyards: the churchyard by the precinct, the rest outside on the edge.
    n_graves = 1 if town or village else max(1, int(round(profile.population / float(params.graveyard_people))))
    if not precinct.is_empty:
        d = np.where(free, shapely.distance(geoms, precinct), np.inf)
        if np.isfinite(d).any():
            take(np.array([int(np.argmin(d))]), "graveyard")
            n_graves -= 1
    if n_graves > 0:
        edge = np.linalg.norm(centroids - centre, axis=1) / radius
        areas = shapely.area(geoms)
        cand = np.flatnonzero(free & ~in_core & (edge >= np.quantile(edge, 0.6)) & (areas <= np.median(areas)))
        if cand.size:
            take(rng.choice(cand, size=min(n_graves, cand.size), replace=False), "graveyard")
    # Small squares in the core.
    n_squares = 0 if town or village else int(round(profile.population / float(params.square_people)))
    areas = shapely.area(geoms)
    cand = np.flatnonzero(free & in_core & (areas <= np.quantile(areas, 0.3)))
    if n_squares and cand.size:
        take(rng.choice(cand, size=min(n_squares, cand.size), replace=False), "square")
    rest = np.flatnonzero(free)
    outside = ~in_core[rest] & walled
    district[rest] = np.where(in_sparse[rest], "gardens", np.where(outside, "suburb", "residential"))
    return [str(d) for d in district]


# ── the layout ───────────────────────────────────────────────────────────────

def _first(cands, avoid: Sequence[np.ndarray], apart_m: float):  # type: ignore[no-untyped-def]
    """The best candidate at least ``apart_m`` from every point in ``avoid`` (else the best)."""
    for a in cands:
        if all(np.linalg.norm(a.xy - p) >= apart_m for p in avoid):
            return a
    return cands[0] if cands else None


def choose_anchors(site: "SettlementSite", profile: SettlementProfile,
                   params: SettlementLayoutParams) -> dict[str, np.ndarray]:
    """Kind → position.  See the module docstring."""
    out: dict[str, np.ndarray] = {}
    cand = site.anchors
    if cand["market"]:
        out["market"] = np.asarray(cand["market"][0].xy, dtype=np.float64)
    elif not site.footprint.is_empty:
        c = site.footprint.representative_point()
        out["market"] = np.array([c.x, c.y])
    else:
        return out
    pol = profile.polity
    wants_citadel = profile.tier <= 4 and (
        pol.capital or pol.march >= float(params.citadel_march)
        or (profile.walls and profile.wall_strength >= float(params.citadel_wall_strength)))
    apart = math.sqrt(max(_PLAZA_M2[1], 1.0))
    if wants_citadel and cand["citadel"]:
        a = _first(cand["citadel"], [out["market"]], apart)
        assert a is not None
        out["citadel"] = np.asarray(a.xy, dtype=np.float64)
    ships = any(i.name in ("shipping", "fishing") and i.weight >= 0.5 for i in profile.industries)
    if ships and cand["waterfront"]:
        out["waterfront"] = np.asarray(cand["waterfront"][0].xy, dtype=np.float64)
    if cand["sacred"]:
        a = _first(cand["sacred"], [p for p in out.values()], 0.5 * apart)
        assert a is not None
        out["sacred"] = np.asarray(a.xy, dtype=np.float64)
    return out


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


def main_axis(part: Polygon, centre: np.ndarray, gates: Sequence[np.ndarray], max_deg: float) -> np.ndarray:
    """The footprint's long axis — or the way to a gate within ``max_deg`` of it."""
    _, u, _, _, _ = _axes(part)
    best: Optional[tuple[float, np.ndarray]] = None
    for g in gates:
        d = np.asarray(g, dtype=np.float64) - centre
        n = float(np.linalg.norm(d))
        if n < 1.0:
            continue
        c = abs(float(d @ u)) / n
        if c >= math.cos(math.radians(max_deg)) and (best is None or c > best[0]):
            best = (c, d / n)
    return best[1] if best is not None else u


def radial_targets(part: Polygon, centre: np.ndarray, taken: Sequence[np.ndarray], n: int) -> list[np.ndarray]:
    """Points on ``part``'s edge in the widest gaps between the ways already out of
    ``centre`` (``taken``), until ``n`` ways leave it."""
    angles = [math.atan2(float(p[1] - centre[1]), float(p[0] - centre[0])) for p in taken
              if np.linalg.norm(np.asarray(p) - centre) > 1.0]
    radius = math.sqrt(part.area / math.pi)
    out: list[np.ndarray] = []
    if not angles:
        _, u, _, _, _ = _axes(part)
        angles_new = [math.atan2(u[1], u[0])]
    else:
        angles_new = []
    for _ in range(2 * n):
        if len(angles) + len(angles_new) >= n:
            break
        if angles_new:
            a = angles_new.pop()
        else:
            ring = np.sort(np.mod(angles, 2.0 * np.pi))
            gaps = np.diff(np.append(ring, ring[0] + 2.0 * np.pi))
            k = int(np.argmax(gaps))
            if gaps[k] < 1.2 * 2.0 * np.pi / n:
                break
            a = float(ring[k] + 0.5 * gaps[k])
        angles.append(a)
        end = ray_exit(part, centre, np.array([math.cos(a), math.sin(a)]))
        if np.linalg.norm(end - centre) > 0.3 * radius:
            out.append(end)
    return out


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


def _smoothstep(x: np.ndarray | float) -> np.ndarray | float:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def build_layout(
    site: "SettlementSite", profile: SettlementProfile, slope: np.ndarray, seed: Sequence[int],
    params: Optional[SettlementLayoutParams] = None,
) -> SettlementLayout:
    """The layout of ``site``.  ``slope``: (T, T) km/km over the frame; ``seed``: the
    settlement's stream key.  See the module docstring."""
    params = params or SettlementLayoutParams()
    footprint = site.footprint
    if footprint.is_empty:
        return SettlementLayout.empty()
    rng = np.random.default_rng([*seed, _LAYOUT_STREAM])
    if profile.tier == 5:
        from settlegen.village_layout import village_layout
        return village_layout(site, profile, slope, rng, params)
    anchors = choose_anchors(site, profile, params)
    pop = float(profile.population)
    town = profile.tier == 4
    market = anchors["market"]
    water = site.water.bodies
    roads = [ln.xy for ln in site.roads.lines]
    half = site.frame.half
    radius = math.sqrt(footprint.area / math.pi)

    def sized(per_person: float, bounds: tuple) -> float:
        return float(np.clip(per_person * pop, bounds[0], bounds[1]))

    # ── citadel and precinct ──
    citadel = MultiPolygon()
    if "citadel" in anchors:
        citadel = _multi(_area_disc(anchors["citadel"], sized(params.citadel_m2_per_person, _CITADEL_M2),
                                    7, rng, 0.25).intersection(footprint))
    precinct = MultiPolygon()
    if "sacred" in anchors:
        precinct = _multi(_area_disc(anchors["sacred"], sized(params.precinct_m2_per_person, _PRECINCT_M2),
                                     5, rng, 0.2).intersection(footprint).difference(citadel))

    # ── main streets: a town's through its market; a city's out of its market every way ──
    part = part_containing(footprint, market)
    gates = [np.asarray(g.xy, dtype=np.float64) for g in site.roads.gates]
    main: list[LineString] = []
    if town:
        u = main_axis(part, market, gates, float(params.main_street_gate_deg))
        ends = [e for e in (ray_exit(part, market, u), ray_exit(part, market, -u))
                if np.linalg.norm(e - market) > 30.0]
        main = route_arterials(half, slope, water, footprint, roads, market, ends, params)
        others = [g for g in gates if all(np.linalg.norm(g - e) > 60.0 for e in ends)]
    else:
        n_radial = int(np.clip(max(int(params.radial_streets), round(2.0 + math.log2(max(pop, 1.0) / 5000.0))), 0, 8))
        others = gates + radial_targets(part, market, gates, n_radial)
    others += [anchors[k] for k in ("waterfront", "citadel", "sacred") if k in anchors]
    side = route_arterials(half, slope, water, footprint, roads, market, others, params)
    arterials = main + side
    art_geom = shapely.union_all(arterials) if arterials else LineString()
    bridges = np.zeros((0, 2))
    if arterials and not site.water.channels.is_empty:
        hit = _multi(art_geom.buffer(0.5).intersection(site.water.channels))
        bridges = np.asarray([(g.centroid.x, g.centroid.y) for g in hit.geoms]).reshape(-1, 2)

    # ── the market: a town's widened main street, a city's plaza ──
    if town and main:
        area = float(np.clip(0.5 * pop, 800.0, 5000.0))
        width = float(params.market_street_m)
        stretch = shapely.union_all(main).intersection(Point(float(market[0]), float(market[1])).buffer(
            0.5 * area / width))
        plaza = _multi(stretch.buffer(0.5 * width, cap_style="flat"))
    else:
        plaza = _multi(_area_disc(market, sized(params.plaza_m2_per_person, _PLAZA_M2), 6, rng, 0.2))
    plaza = _multi(plaza.intersection(footprint).difference(citadel))
    precinct = _multi(precinct.difference(plaza))

    # ── the old core, walled only if the place has walls ──
    young = 1.0 - min(float(profile.founding_age) / float(params.young_years), 1.0)
    share = float(np.clip(
        float(params.core_share) + float(params.core_wall_gain) * float(profile.wall_strength)
        + float(params.core_young_gain) * young + (0.25 if town else 0.0), 0.05, 1.0))
    centre = market if "citadel" not in anchors else 0.6 * market + 0.4 * anchors["citadel"]
    core = core_polygon(_multi(footprint.difference(site.sparse)) if not site.sparse.is_empty else footprint,
                        centre, share, rng, float(params.core_wobble))
    if not citadel.is_empty:
        core = _multi(shapely.union_all([core, citadel]))
    walls: Optional[Walls] = None
    if profile.walls and not core.is_empty:
        kind = "stone" if profile.wall_strength >= float(params.stone_wall_strength) else "palisade"
        # Round each piece's outside only (not round the holes in it).
        line = MultiLineString([LineString(g.exterior.coords) for g in core.geoms])
        if not water.is_empty:
            line = line.difference(water.buffer(float(params.wall_water_m)))
        parts = [g for g in shapely.get_parts(shapely.line_merge(line)) if isinstance(g, LineString) and g.length > 5]
        wall_line = MultiLineString(parts)
        towers = np.zeros((0, 2))
        if kind == "stone" and parts:
            spacing = max(float(params.tower_spacing_m) / max(float(profile.wall_strength), 0.25), 20.0)
            towers = np.asarray([(p.x, p.y) for g in parts for p in shapely.line_interpolate_point(
                g, np.arange(0.0, g.length, spacing))]).reshape(-1, 2)
        gate_pts = wall_line.intersection(art_geom) if arterials and parts else Point()
        wall_gates = np.asarray([(p.x, p.y) for p in shapely.get_parts(gate_pts)
                                 if isinstance(p, Point)]).reshape(-1, 2)
        walls = Walls(kind=kind, line=wall_line, towers=towers, gates=wall_gates, width_m=float(params.wall_m))

    # ── docks ──
    docks = Docks()
    if "waterfront" in anchors:
        weight = max([i.weight for i in profile.industries if i.name in ("shipping", "fishing")] or [0.0])
        docks = build_docks(footprint, site.water, anchors["waterfront"], [la.xy for la in site.roads.landings],
                            pop, weight, profile.tier, rng, _dock_spec(params))

    # ── wards → blocks ──
    cut: list = [plaza, citadel, precinct, docks.quays]
    if arterials:
        cut.append(art_geom.buffer(0.5 * float(params.arterial_m)))
    if walls is not None and not walls.line.is_empty:
        cut.append(walls.line.buffer(0.5 * float(params.wall_m)))
    land = _multi(footprint.difference(shapely.union_all(cut)))
    blocks: list[Polygon] = []
    streets: list[LineString] = []
    burgage: list[bool] = []
    if town and main:
        band, land, lanes = burgage_band(main, land, rng, params)
        blocks += band
        burgage += [True] * len(band)
        streets += lanes
    grid_angle = 0.0
    if arterials:
        longest = max(arterials, key=lambda g: g.length)
        xy = np.asarray(longest.coords)
        d = xy[-1] - xy[0]
        grid_angle = math.atan2(d[1], d[0])
    walled = walls is not None

    def outward(poly: Polygon) -> float:
        """0 in the dense middle, 1 in the outskirts: by the walls if walled, else by distance."""
        c = poly.centroid
        if not site.sparse.is_empty and site.sparse.contains(c):
            return -1.0
        if walled:
            return 0.0 if core.contains(c) else 1.0
        d = math.hypot(c.x - float(market[0]), c.y - float(market[1])) / max(radius, 1.0)
        return float(_smoothstep((d - 0.4) / 0.6))

    base = float(params.block_ha) * 1e4 * (float(params.town_block_mult) if town else 1.0)

    def goal(poly: Polygon) -> float:
        g = outward(poly)
        if g < 0:
            return base * float(params.sparse_block_mult)
        return base * (1.0 + (float(params.suburb_block_mult) - 1.0) * g)

    wards = [(g, True) for g in _multi(land.intersection(core)).geoms] + \
        [(g, False) for g in _multi(land.difference(core)).geoms] if walled else [(g, False) for g in land.geoms]
    for ward, _ in wards:
        b, st = split_blocks(ward, goal, rng, params, bool(profile.planned), grid_angle)
        blocks += b
        streets += st
        burgage += [False] * len(b)
    cents = shapely.centroid(np.asarray(blocks, dtype=object)) if blocks else np.zeros(0, dtype=object)
    in_core = np.asarray([core.contains(c) for c in cents], dtype=bool)
    districts = assign_districts(site, profile, blocks, in_core, anchors, plaza, precinct, rng, params, walled)

    # ── lots ──
    out_blocks: list[Block] = []
    for poly, dist, is_core, is_burgage in zip(blocks, districts, in_core, burgage):
        if dist in OPEN_DISTRICTS:
            lots = np.zeros(0, dtype=object)
        elif is_burgage:
            lots = lots_of(poly, float(params.burgage_frontage_m), 1.4 * float(params.burgage_depth_m), rng,
                           bool(profile.planned))
        else:
            g = outward(poly)
            mult = float(params.garden_lot_mult) if dist == "gardens" or g < 0 else \
                1.0 if dist in ("market", "quay") else 1.0 + (float(params.suburb_lot_mult) - 1.0) * g
            lots = lots_of(poly, float(params.lot_frontage_m) * mult, float(params.lot_depth_m) * math.sqrt(mult),
                           rng, bool(profile.planned))
        out_blocks.append(Block(polygon=poly, district=dist, core=bool(is_core), lots=lots))

    # ── wells and shrines on the streets ──
    lanes_all = streets or [g for g in arterials]
    core_streets = [st for st in lanes_all if core.contains(st.centroid)] or lanes_all
    n_wells = max(1, int(round(core.area / 1e4 / float(params.well_ha))))
    points = {
        "well": on_streets(core_streets, n_wells, rng),
        "shrine": on_streets(lanes_all, int(round(pop / float(params.shrine_people))), rng),
    }
    return SettlementLayout(
        anchors=anchors, core=core, walls=walls, citadel=citadel, plaza=plaza, precinct=precinct,
        arterials=arterials, bridges=bridges, streets=streets, blocks=out_blocks, points=points,
        arterial_m=float(params.arterial_m), quays=docks.quays, piers=docks.piers,
    )


def on_streets(lines: Sequence[LineString], k: int, rng: np.random.Generator) -> np.ndarray:
    """``(k, 2)`` points along ``lines``, by length."""
    if k <= 0 or not lines:
        return np.zeros((0, 2))
    lengths = np.array([g.length for g in lines])
    if lengths.sum() <= 0:
        return np.zeros((0, 2))
    pick = rng.choice(len(lines), size=k, p=lengths / lengths.sum())
    return np.asarray([(p.x, p.y) for p in (lines[i].interpolate(rng.uniform(0.0, lines[i].length))
                                            for i in pick)]).reshape(-1, 2)


def _dock_spec(params: SettlementLayoutParams) -> DockSpec:
    return DockSpec(quay_width_m=float(params.quay_width_m), pier_spacing_m=float(params.pier_spacing_m),
                    pier_length_m=float(params.pier_length_m), pier_width_m=float(params.pier_width_m),
                    pier_min_river_m=float(params.pier_min_river_m))


# ── what is where (for hover) ────────────────────────────────────────────────

_DISTRICT_LABELS = {
    "residential": "Houses", "suburb": "Suburb", "gardens": "Gardens", "elite": "Elite quarter",
    "market": "Market", "quay": "Quayside", "graveyard": "Graveyard", "square": "Square",
    "sugar mills": "Sugar mills",
}


def district_label(district: str, tier: int) -> str:
    """How a district reads (a village's houses are farmsteads, its elite plot the manor)."""
    if tier == 5:
        return {"residential": "Farmstead", "elite": "Manor"}.get(district, district.capitalize())
    if tier == 4 and district == "graveyard":
        return "Churchyard"
    if district in _DISTRICT_LABELS:
        return _DISTRICT_LABELS[district]
    return f"{district.capitalize()}' quarter"


def block_at(layout: SettlementLayout, xy: np.ndarray) -> Optional[Block]:
    """The block holding ``xy`` (an STR tree, built once per layout)."""
    tree = getattr(layout, "_block_tree", None)
    if tree is None:
        tree = shapely.STRtree([b.polygon for b in layout.blocks])
        setattr(layout, "_block_tree", tree)
    hits = tree.query(Point(float(xy[0]), float(xy[1])), predicate="within")
    return layout.blocks[int(hits[0])] if len(hits) else None


def describe_at(layout: SettlementLayout, site: "SettlementSite", xy: np.ndarray, tier: int) -> Optional[str]:
    """What of the settlement is at ``xy`` (frame metres), or None."""
    pt = Point(float(xy[0]), float(xy[1]))
    village = tier == 5
    if any(p.contains(pt) for p in layout.piers):
        return "Jetty" if village else "Pier"
    walls = layout.walls
    if walls is not None and not walls.line.is_empty and walls.line.distance(pt) <= 0.5 * walls.width_m + 2.0:
        near_tower = len(walls.towers) and float(np.min(np.linalg.norm(walls.towers - xy, axis=1))) <= walls.width_m
        return "Tower" if near_tower else ("Stone wall" if walls.kind == "stone" else "Palisade")
    for geom, label in ((layout.citadel, "Citadel"), (layout.plaza, "Market street" if tier == 4 else "Market place"),
                        (layout.precinct, "Church and churchyard" if tier >= 4 else "Temple precinct"),
                        (layout.green, "Village green"), (layout.quays, "Quay")):
        if not geom.is_empty and geom.contains(pt):
            return label
    if layout.arterials and shapely.union_all(layout.arterials).distance(pt) <= 0.5 * layout.arterial_m:
        return "Village street" if village else "Main street"
    block = block_at(layout, xy)
    if block is not None:
        label = district_label(block.district, tier)
        for lot in block.lots:
            if lot.contains(pt):
                _, _, _, length, depth = _axes(lot)
                return f"{label} · plot ~{min(length, depth):.0f} × {max(length, depth):.0f} m"
        return label if block.district in OPEN_DISTRICTS or not len(block.lots) else f"{label} · yard"
    if not site.water.channels.is_empty and site.water.channels.contains(pt):
        return "River"                   # the sea and lakes the ground's own readout names
    if layout.fill_footprint and site.footprint.contains(pt):
        return "Street"
    return None


def summary(layout: SettlementLayout, profile: SettlementProfile) -> str:
    """One line on the settlement and its layout."""
    pol = profile.polity
    who = f"{profile.tier_name} ({profile.kind}) · {profile.population:,.0f} people"
    rule = "" if pol.control == "none" else f" · {pol.control}" + (" capital" if pol.capital else "")
    if pol.march >= 0.3:
        rule += " march"
    walls = f"{layout.walls.kind} walls" if layout.walls is not None else "unwalled"
    if profile.tier == 5:
        plan = f"{len(layout.blocks)} farmsteads"
    else:
        plan = f"{len(layout.blocks):,} blocks, {layout.n_lots:,} plots" + (" (planned)" if profile.planned else "")
    n = len(layout.piers)
    docks = f" · {n} pier{'s' if n != 1 else ''}" if n else ""
    return f"{who}{rule} · {walls} · {plan}{docks}"
