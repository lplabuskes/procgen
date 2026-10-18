"""A settlement's layout: walls, streets, districts and lots (plan phase 6).

:func:`build_layout` reads only the :class:`SettlementSite`, the
:class:`SettlementProfile` and the ground's slope, so it runs on hand-built sites.
Block subdivision (user choice, 2026-10-11), planned by tier — a city, a town and
a village are different places, not one plan at three sizes.  Villages:
:mod:`settlegen.layout.village`.  Towns and cities, in order:

1. **Anchors** from the site's candidates by profile: always a market; a
   citadel for a capital, a march (``march`` ≥ ``citadel_march``) or a strongly
   walled town or city; a waterfront where it ships or fishes; a sacred site
   apart from the others.
2. **Main streets**, least-cost on a ``route_step_m`` grid (slope, water, off
   the footprint; along the site's roads cheaply — they hold the river
   crossings — but not led by them).  A **town**'s main street runs through its
   market along the footprint's long axis (or to a gate near that line); a
   **city**'s run out of its market to every gate, and to the footprint's landward
   edge (not the shore), each as far along it from the others as can be, until
   at least ``radial_streets`` leave it (no nearer than ``radial_spacing_m``).
   Both reach the waterfront, citadel and sacred site.  Bridges where they
   cross a channel.  After the walls, **secondary arterials**
   (:func:`infill_arterials`) fill the gaps until the dense ground lies within
   half of ``arterial_spacing_m`` (towns: ``town_arterial_spacing_m``) of an
   arterial, through the walls only at their gates.
3. **Market**: a town's is its main street widened (``market_street_m``) about
   the market; a city's a plaza.  The citadel and the sacred precinct are open
   ground sized by population.
4. **Old core**: a smooth blob about the market (and citadel) holding
   ``core_share`` of the footprint (more for strong walls, young places and
   towns).  Only a walled place shows it: walls (``palisade`` or ``stone``,
   towers on stone) run round it, open along the water, with suburbs outside,
   and a wall street (:func:`wall_street`) just inside them.
   Unwalled, blocks and lots just grow coarser toward the edge.
5. **Docks** (:mod:`settlegen.layout.docks`) where there is a waterfront: quays along
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
   along the streets.  Blocks too deep for their lots (yards in the middle) are
   opened up by closes and alleys (:func:`lanes_of`) with lots fronting them
   (:func:`lane_lots`).

:func:`describe_at` names what is at a point (for hover).  No names
(Decision 9).  Seeded per (world, settlement).  Qt-free.
"""
from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np
import shapely
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, Polygon

from settlegen.layout.blocks import burgage_band, lane_lots, lanes_of, lots_of, split_blocks, yards_of
from settlegen.layout.districts import LANE_DISTRICTS, OPEN_DISTRICTS, assign_districts
from settlegen.layout.docks import Docks, DockSpec, build_docks
from settlegen.layout.geometry import _area_disc, _multi, core_polygon, part_containing, ray_exit
from settlegen.layout.params import SettlementLayoutParams
from settlegen.layout.streets import (
    infill_arterials, main_axis, radial_targets, route_arterials, route_grid, wall_street,
)
from settlegen.layout.types import Block, SettlementLayout, Walls
from settlegen.profile.builder import SettlementProfile
from utils.numeric import smoothstep

if TYPE_CHECKING:
    from settlegen.site.builder import SettlementSite


_LAYOUT_STREAM = 61


# (min, max) m² of the open grounds.
_PLAZA_M2, _CITADEL_M2, _PRECINCT_M2 = (300.0, 15000.0), (1500.0, 60000.0), (300.0, 12000.0)


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
        from settlegen.layout.village import village_layout
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
        others = gates + radial_targets(part, market, gates, n_radial, water, float(params.radial_spacing_m))
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
    # Sparse ground (gardens, orchards) inside the line is walled in with the rest, not walled round.
    core = core_polygon(footprint, centre, share, rng, float(params.core_wobble))
    if not citadel.is_empty:
        core = _multi(shapely.union_all([core, citadel]))
    grown: Optional[MultiPolygon] = None
    if profile.walls and not core.is_empty and float(params.wall_closing) > 0.0:
        # The wall takes the short way across a notch in the core's edge (and round the
        # citadel); dry ground it takes in beyond the footprint is walled gardens.
        r = float(params.wall_closing) * math.sqrt(core.area / math.pi)
        dry = site.frame.box().difference(site.water.bodies)
        closed = _multi(core.buffer(r, quad_segs=16).buffer(-r, quad_segs=16).intersection(dry))
        biggest = max([g.area for g in closed.geoms] or [0.0])
        core = _multi(shapely.union_all([core, *(g for g in closed.geoms if g.area >= 0.05 * biggest)]))
        added = _multi(core.difference(footprint))
        if added.area > 1.0:
            footprint = grown = _multi(footprint.union(added))
            site = dataclasses.replace(site, footprint=footprint, sparse=_multi(site.sparse.union(added)))
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

    # ── the wall street and secondary arterials ──
    n_main = len(arterials)
    kinds = ["main"] * n_main
    secondary_m = float(params.secondary_m)
    if walls is not None:
        ring = wall_street(core, walls.line, 0.5 * float(params.wall_m) + 0.5 * secondary_m, citadel)
        arterials += ring
        kinds += ["wall"] * len(ring)
    if arterials:
        spacing = float(params.town_arterial_spacing_m if town else params.arterial_spacing_m)
        serve = _multi(footprint.difference(shapely.union_all([site.sparse, plaza, citadel, precinct, docks.quays])))
        grid = route_grid(half, slope, water, footprint, roads, params)
        extra = infill_arterials(grid, arterials, serve, footprint, spacing,
                                 walls=walls.line if walls is not None else None,
                                 gates=walls.gates if walls is not None else None,
                                 avoid=_multi(shapely.union_all([citadel, precinct])))
        arterials += extra
        kinds += ["secondary"] * len(extra)

    # ── wards → blocks ──
    cut: list = [plaza, citadel, precinct, docks.quays]
    for which, width in ((("main",), float(params.arterial_m)), (("wall", "secondary"), secondary_m)):
        lines = [g for g, k in zip(arterials, kinds) if k in which]
        if lines:
            cut.append(shapely.union_all(lines).buffer(0.5 * width))
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
    if n_main:
        longest = max(arterials[:n_main], key=lambda g: g.length)
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
        return float(smoothstep(0.0, 1.0, (d - 0.4) / 0.6))

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
    closes: list[LineString] = []
    alleys: list[LineString] = []
    planned = bool(profile.planned)
    for poly, dist, is_core, is_burgage in zip(blocks, districts, in_core, burgage):
        if dist in OPEN_DISTRICTS:
            out_blocks.append(Block(polygon=poly, district=dist, core=bool(is_core), lots=np.zeros(0, dtype=object)))
            continue
        if is_burgage:
            lots = lots_of(poly, float(params.burgage_frontage_m), 1.4 * float(params.burgage_depth_m), rng, planned)
            out_blocks.append(Block(polygon=poly, district=dist, core=bool(is_core), lots=lots))
            continue
        g = outward(poly)
        mult = float(params.garden_lot_mult) if dist == "gardens" or g < 0 else \
            1.0 if dist in ("market", "quay") else 1.0 + (float(params.suburb_lot_mult) - 1.0) * g
        frontage, depth = float(params.lot_frontage_m) * mult, float(params.lot_depth_m) * math.sqrt(mult)
        pieces, c, a = lanes_of(poly, depth, rng, params, planned) if dist in LANE_DISTRICTS and g >= 0 \
            else ([poly], [], [])
        closes += c
        alleys += a
        for piece in pieces:
            lots = lots_of(piece, frontage, depth, rng, planned)
            near = [ln for ln in (*c, *a) if ln.distance(piece) < float(params.lane_m)]
            yards = yards_of(piece, depth) if near else None
            if yards is not None:
                extra = lane_lots(yards, near, frontage, depth, 0.5 * float(params.lane_m), rng, planned)
                lots = np.concatenate([lots, extra]) if len(extra) else lots
            out_blocks.append(Block(polygon=piece, district=dist, core=bool(is_core), lots=lots))

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
        arterial_m=float(params.arterial_m), quays=docks.quays, piers=docks.piers, footprint=grown,
        arterial_kinds=kinds, secondary_m=secondary_m, closes=closes, alleys=alleys, lane_m=float(params.lane_m),
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
