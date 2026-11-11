"""Lots as boundaries: derived from a block's ground and dividers, packed along village streets."""
import math

import numpy as np
import pytest
import shapely
from shapely.geometry import LineString, Point, box

from settlegen.layout.blocks import block_sides, front_trees, perimeter_lots, street_fronts, strip_lots
from settlegen.layout.frontage import PlotSpec, cut_blocks, pack_frontage, ranked, street_grounds
from settlegen.layout.lots import merge_slivers
from settlegen.layout.types import Block


def _tile(lots) -> None:  # type: ignore[no-untyped-def]
    """``lots`` don't overlap."""
    assert shapely.union_all(list(lots)).area == pytest.approx(sum(g.area for g in lots), rel=1e-6)


def test_a_block_round_trips_through_its_dividers():
    block = shapely.Polygon([(0, 0), (130, 8), (124, 70), (-6, 60)])
    lots, _ = strip_lots(block, 8.0, 22.0, np.random.default_rng(0), planned=False)
    blk = Block.of_lots(block, "residential", True, lots)
    assert len(blk.lots) == len(lots)
    for a, b in zip(lots, blk.lots):
        assert b.symmetric_difference(a).area < 1e-6 * a.area
    _tile(blk.lots)
    # The dividers lie inside the ground, off its edge.
    assert blk.dividers.difference(block.boundary.buffer(1e-6)).length > 0
    assert blk.dividers.difference(block.buffer(1e-6)).length < 1e-6


def _ringed_block():  # type: ignore[no-untyped-def]
    """A 100 × 60 m block (lots 22 m deep: yards in the middle) with a main street along its
    south side, minor streets along the others."""
    block = box(0, 0, 100, 60)
    main = [LineString([(-20, -5.0), (120, -5.0)])]                       # 10 m wide
    minor = [LineString([(-2.5, -20), (-2.5, 80)]), LineString([(102.5, -20), (102.5, 80)]),
             LineString([(-20, 62.5), (120, 62.5)])]                     # 5 m wide
    fronts = front_trees(street_fronts(main, ["main"], 10.0, 7.0, minor, 5.0, [], 3.0, []))
    return block, block_sides(block, fronts)


def test_block_sides_are_its_street_sides_ranked():
    block, sides = _ringed_block()
    assert sorted(r for _, r in sides) == [1, 1, 1, 3]
    main = next(g for g, r in sides if r == 3)
    assert main.length == pytest.approx(100.0) and max(abs(y) for _, y in main.coords) < 1e-9
    # Counter-clockwise: the block on each side's left.
    for g, _ in sides:
        (x0, y0), (x1, y1) = g.coords[0], g.coords[-1]
        assert block.contains(Point(0.5 * (x0 + x1) - 1e-3 * (y1 - y0), 0.5 * (y0 + y1) + 1e-3 * (x1 - x0)))


def test_perimeter_lots_line_every_street_side():
    block, sides = _ringed_block()
    lots, _ = perimeter_lots(block, sides, 8.0, 22.0, 0.3, np.random.default_rng(0), planned=True)
    _tile(lots)
    assert all(block.buffer(1e-6).contains(g) for g in lots)
    # Every side is built along: a lot within 1 m of each street side's middle.
    for g, _ in sides:
        assert any(lot.distance(g.interpolate(0.5, normalized=True)) < 1e-6 for lot in lots)
    # The middle is yards.
    assert not any(lot.intersects(box(40, 25, 60, 35)) for lot in lots)
    # The corner lots front the main street: they run the full 22 m up from it, the west and
    # east rows starting behind them.
    for x in (0.5, 99.5):
        corner = next(lot for lot in lots if lot.contains(Point(x, 0.5)))
        assert corner.bounds[3] == pytest.approx(22.0)
    # Frontage is dear: narrower on the main street (×0.7) than on the minor ones.
    south = [lot for lot in lots if lot.bounds[1] < 1e-6 and lot.bounds[3] <= 22.0 + 1e-6]
    north = [lot for lot in lots if lot.bounds[3] > 60.0 - 1e-6 and lot.bounds[1] >= 38.0 - 1e-6]
    width = [float(np.mean([g.bounds[2] - g.bounds[0] for g in row])) for row in (south, north)]
    assert width[0] == pytest.approx(0.7 * width[1], rel=0.15)


def test_no_dividers_one_lot_is_the_ground():
    block = box(0, 0, 20, 10)
    blk = Block(polygon=block, district="residential", core=True, seeds=np.array([[5.0, 5.0]]))
    assert blk.lots[0].equals(block)


def _arms():  # type: ignore[no-untyped-def]
    """Two village arms out of the middle at 60° and the ground round them."""
    arms = [LineString([(0, 0), (300, 0)]), LineString([(0, 0), (150, 260)])]
    allowed = box(-400, -400, 400, 400).difference(shapely.union_all(arms).buffer(3.0))
    return arms, allowed


def test_the_longer_street_keeps_the_corner():
    arms, allowed = _arms()
    arms = [arms[0], LineString([(0, 0), (100, 173)])]           # the 60° arm, shorter
    grounds = street_grounds(arms, allowed, 40.0)
    long_ = shapely.union_all([g for line, _, g in grounds if line.length > 250.0])
    short = shapely.union_all([g for line, _, g in grounds if line.length < 250.0])
    assert long_.intersection(short).area < 1e-6
    # In the corner between them the longer keeps its whole band (no slant across it)...
    p = Point(60.0, 35.0)                         # 35 m off the long arm, ~13 m off the short
    assert long_.contains(p) and not short.contains(p)
    # ... and the shorter's ground starts square to it, past the long arm's band.
    assert short.intersection(box(-400, -400, 400, 43.0)).area < 1e-6


def test_arms_meeting_end_to_end_are_one_line():
    arms = [LineString([(0, 0), (100, 0)]), LineString([(0, 0), (-80, 30)]), LineString([(0, 0), (0, 50)])]
    lines = ranked(arms[:2])
    assert len(lines) == 1 and lines[0].length == pytest.approx(100.0 + math.hypot(80, 30))
    assert len(ranked(arms)) == 3                 # three meeting: no single line through


def test_plots_where_arms_meet_tile_their_ground():
    arms, allowed = _arms()
    rng = np.random.default_rng(1)
    centre = np.zeros(2)
    plots = [PlotSpec("farm", float(rng.uniform(12, 18)), 35.0, centre) for _ in range(60)]
    blocks, placed = pack_frontage(arms, allowed, plots, centre, 3.0, rng, gap_chance=0.1)
    lots = [lot for b in blocks for lot in b.lots]
    assert len(lots) == sum(placed) >= 40
    _tile(lots)
    streets = shapely.union_all(arms).buffer(3.0)
    assert all(lot.intersection(streets).area < 1e-3 for lot in lots)
    # Grown out from the middle: the plots nearest it are taken, both arms, both sides.
    near = [lot for lot in lots if lot.distance(Point(0, 0)) < 40.0]
    assert len(near) >= 4
    # Each plot fronts its street: within a few metres of it.
    assert all(lot.distance(shapely.union_all(arms)) < 4.0 for lot in lots)


def test_a_free_plot_goes_by_its_target():
    arms, allowed = _arms()
    rng = np.random.default_rng(2)
    mill = np.array([280.0, -30.0])
    plots = [PlotSpec("farm", 15.0, 35.0, np.zeros(2)) for _ in range(5)] + \
        [PlotSpec("miller", 12.0, 30.0, mill, free=True)]
    blocks, placed = pack_frontage(arms, allowed, plots, np.zeros(2), 3.0, rng, gap_chance=0.0)
    assert all(placed)
    miller = next(lot for b in blocks for i, lot in enumerate(b.lots) if b.use(i) == "miller")
    assert miller.distance(Point(*mill)) < 40.0


def test_a_lane_cut_through_plots_trims_or_drops_them():
    arms, allowed = _arms()
    rng = np.random.default_rng(3)
    plots = [PlotSpec("farm", 15.0, 35.0, np.zeros(2)) for _ in range(20)]
    blocks, _ = pack_frontage(arms[:1], allowed, plots, np.zeros(2), 3.0, rng, gap_chance=0.0)
    before = sum(len(b.lots) for b in blocks)
    lane = LineString([(100.0, -100.0), (100.0, 100.0)]).buffer(2.0)
    after = cut_blocks(blocks, lane)
    lots = [lot for b in after for lot in b.lots]
    assert before - 4 <= len(lots) <= before
    assert all(lot.intersection(lane).area < 1e-3 for lot in lots)
    _tile(lots)


def test_both_sides_of_a_street_get_plots_even_where_its_ground_is_one_piece():
    # The ground not cut by the street (one piece round it): each side is still its own run.
    street = LineString([(0, 0), (200, 0)])
    rng = np.random.default_rng(4)
    plots = [PlotSpec("farm", 15.0, 30.0, np.zeros(2)) for _ in range(20)]
    blocks, placed = pack_frontage([street], box(-300, -300, 300, 300), plots, np.zeros(2), 3.0, rng, 0.0)
    ys = [lot.centroid.y for b in blocks for lot in b.lots]
    assert all(placed)
    assert sum(y > 0 for y in ys) >= 8 and sum(y < 0 for y in ys) >= 8


def test_free_street_ends_run_on_as_far_as_the_ground_goes():
    from settlegen.layout.village import _run_on_free_ends
    a = LineString([(0, 0), (100, 0)])
    b = LineString([(0, 0), (0, 100)])
    ground = box(-300, -300, 300, 120)
    a2, b2 = _run_on_free_ends([a, b], 30.0, ground)
    assert a2.coords[0] == (0, 0) and a2.coords[-1] == pytest.approx((130.0, 0.0))     # shared start stays
    assert b2.coords[-1] == pytest.approx((0.0, 120.0))                               # stops at the ground's edge


def test_no_plot_spans_a_sharp_bend():
    street = LineString([(0, 0), (100, 0), (100, 100)])           # a right angle
    rng = np.random.default_rng(5)
    plots = [PlotSpec("farm", 15.0, 30.0, np.zeros(2)) for _ in range(20)]
    blocks, _ = pack_frontage([street], box(-300, -300, 300, 300), plots, np.zeros(2), 3.0, rng, 0.0)
    # Every plot is near-rectangular: its area close to its minimum rotated rectangle's.
    for lot in (lot for b in blocks for lot in b.lots):
        assert lot.area > 0.8 * lot.minimum_rotated_rectangle.area


def test_a_sliver_joins_the_neighbour_it_shares_most_with():
    # Three strips off a street and a wedge left along the block's slanted top edge.
    strips = [box(x, 0, x + 8, 20) for x in (0, 8, 16)]
    wedge = shapely.Polygon([(0, 20), (24, 20), (24, 23), (0, 21)])
    merged = merge_slivers(np.asarray([*strips, wedge], dtype=object), 3.5)
    assert len(merged) == 3
    _tile(merged)
    assert sum(g.area for g in merged) == pytest.approx(sum(g.area for g in (*strips, wedge)))
    assert sorted(g.area for g in merged)[-1] == pytest.approx(160.0 + wedge.area)    # into one strip
    # Nothing thin: the lots come back as they were.
    assert all(a.equals(b) for a, b in zip(merge_slivers(np.asarray(strips, dtype=object), 3.5), strips))
