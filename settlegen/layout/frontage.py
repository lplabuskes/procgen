"""Plots fronting streets, packed as boundaries: a village's tofts.

Streets meeting end to end are one line (:func:`ranked`); then, longest first,
each street claims the ground within ``reach`` of it that no street before it has
(:func:`street_grounds`) — where two meet, the corner plots front the longer and
the shorter's plots start behind them, cut square to the longer.  Each side of a
street's ground is a **run**.  A plot is its run's ground between two
**dividers** — lines square off the street at stations along it — out to the
plot's depth, never across a bend sharper than ``_MAX_BEND_COS``.  Plots grow
outward from each run's point nearest the middle, both ways (two cursors a run):
each plot in turn takes the cursor whose next slot lies nearest its target, so the
first placed sit in the middle; a ``free`` plot (the miller, by the water) takes
the slot nearest its target anywhere.  Neighbours share their divider, so plots
tile their run and nothing overlaps; skipped slots (gaps), the ground behind
shallow plots and what no plot reaches are open ground.  Each run becomes a
:class:`Block` (its ground, the dividers and backs, a seed per plot).  Qt-free.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import shapely
from shapely.geometry import LineString, MultiLineString, Point, Polygon
from shapely.geometry.base import BaseGeometry

from settlegen.layout.lots import as_lines
from settlegen.layout.types import Block
from utils.sphere import unit

# Street tangents are taken over ± this (m): dividers square off a smoothed street.
_TANGENT_M = 6.0
# No plot spans a bend sharper than this (its dividers would fan): the corner stays open.
_MAX_BEND_COS = float(np.cos(np.radians(25.0)))


@dataclass
class PlotSpec:
    """One plot to place."""

    use: str
    width: float                     # frontage (m)
    depth: float                     # back from the street's edge (m)
    target: np.ndarray               # (2,) placed as near this as it can be
    free: bool = False               # anywhere along the streets (else grown out from the middle)
    gaps: bool = True                # a slot may be skipped before it


def ranked(streets: Sequence[LineString]) -> list[LineString]:
    """``streets`` with those meeting end to end merged into one line, longest first: the
    order they claim their ground in."""
    merged = shapely.line_merge(MultiLineString([g for g in streets if g.length > 1.0]))
    lines = [g for g in shapely.get_parts(merged) if isinstance(g, LineString) and g.length > 1.0]
    return sorted(lines, key=lambda g: -g.length)


def street_grounds(
    streets: Sequence[LineString], allowed: BaseGeometry, reach: float,
) -> list[tuple[LineString, float, BaseGeometry]]:
    """``(street, side, ground)`` per side of each of the :func:`ranked` streets: the ground
    of ``allowed`` within ``reach`` of it on that side (+1 left of its way, −1 right) that no
    street before it has claimed."""
    out = []
    claimed: BaseGeometry = Polygon()
    for line in ranked(streets):
        for side in (1.0, -1.0):
            band = line.buffer(side * reach, single_sided=True)
            out.append((line, side, allowed.intersection(band).difference(claimed)))
        claimed = claimed.union(line.buffer(reach, cap_style="flat"))
    return out


class _Run:
    """One side of one street within one piece of its ground."""

    def __init__(self, line: LineString, side: float, ground: Polygon, half_street: float, reach: float):
        self.line, self.side, self.ground = line, side, ground
        self.hs, self.reach = half_street, reach
        # Dividers start just over the street's centre line, so cut to the ground they always
        # meet its edge (the street cut out of it or not).
        self.front = -half_street - 0.5
        s = shapely.line_locate_point(line, shapely.points(shapely.get_coordinates(ground.exterior)))
        self.s_lo, self.s_hi = float(np.min(s)), float(np.max(s))
        self.taken: list[tuple[float, float]] = []
        self.plots: list[tuple[float, float, float, str, Polygon]] = []   # (a, b, depth, use, polygon)

    def point_normal(self, s: float) -> tuple[np.ndarray, np.ndarray]:
        L = self.line.length
        a = np.asarray(self.line.interpolate(max(s - _TANGENT_M, 0.0)).coords[0])
        b = np.asarray(self.line.interpolate(min(s + _TANGENT_M, L)).coords[0])
        t = unit(b - a)
        return np.asarray(self.line.interpolate(s).coords[0]), self.side * np.array([-t[1], t[0]])

    def at(self, s: float, out: float) -> np.ndarray:
        """``out`` m off the street's edge, square to it at station ``s``."""
        p, n = self.point_normal(s)
        return p + (self.hs + out) * n

    def within(self, a: float, b: float) -> bool:
        """Stations ``a``–``b`` lie along the run's ground."""
        return self.s_lo - 1e-6 <= a and b <= self.s_hi + 1e-6

    def blocker(self, a: float, b: float) -> Optional[tuple[float, float]]:
        """The stretch of a plot already here overlapping stations ``a``–``b``, if any."""
        return next(((x0, x1) for x0, x1 in self.taken if not (b <= x0 + 1e-6 or a >= x1 - 1e-6)), None)

    def try_place(self, a: float, b: float, depth: float) -> Optional[Polygon]:
        """The plot between stations ``a`` and ``b``, ``depth`` deep, if it fits: on the run's
        ground, at least 55% of its size, not across a sharp bend, its dividers not crossing,
        clear of the plots here."""
        if not self.within(a, b) or self.blocker(a, b) is not None:
            return None
        if float(self.point_normal(a)[1] @ self.point_normal(b)[1]) < _MAX_BEND_COS:
            return None
        a0, b0 = self.at(a, self.front), self.at(b, self.front)
        if LineString([a0, self.at(a, self.reach)]).intersects(LineString([b0, self.at(b, self.reach)])):
            return None
        quad = Polygon([a0, b0, self.at(b, depth), self.at(a, depth)])
        if not quad.is_valid:
            return None
        parts = [g for g in shapely.get_parts(quad.intersection(self.ground)) if isinstance(g, Polygon)]
        if not parts:
            return None
        best = max(parts, key=lambda g: g.area)
        if best.area < 0.55 * (b - a) * depth:
            return None
        if any(best.intersection(p).area > 0.5 for *_, p in self.plots):
            return None
        return best

    def place(self, a: float, b: float, depth: float, use: str, poly: Polygon) -> None:
        self.taken.append((a, b))
        self.plots.append((a, b, depth, use, poly))

    def block(self, district: str) -> Optional[Block]:
        """The run as a block: its ground, the dividers between its plots (each as deep as
        the deeper plot it bounds) and their backs, a seed per plot."""
        if not self.plots:
            return None
        # Each station's divider, through the back corner of each plot it bounds (neighbours share
        # their station exactly, so the corners are the backs' own ends).
        depths: dict[float, set[float]] = {}
        for a, b, d, *_ in self.plots:
            for s in (a, b):
                depths.setdefault(s, set()).add(d)
        lines = [LineString([self.at(s, self.front), *(self.at(s, d) for d in sorted(ds))])
                 for s, ds in depths.items()]
        lines += [LineString([self.at(a, d), self.at(b, d)]) for a, b, d, *_ in self.plots]
        dividers = as_lines(shapely.union_all(lines).intersection(self.ground))
        seeds = shapely.get_coordinates(shapely.point_on_surface(
            np.asarray([p for *_, p in self.plots], dtype=object))).reshape(-1, 2)
        uses = np.asarray([u for *_, u, _ in self.plots], dtype=object)
        return Block(polygon=self.ground, district=district, core=True, dividers=dividers, seeds=seeds, uses=uses)


def _runs(streets: Sequence[LineString], allowed: BaseGeometry, half_street: float, reach: float) -> list[_Run]:
    """A run per piece of each street's ground on each side (see :func:`street_grounds`)."""
    return [_Run(line, side, piece, half_street, reach)
            for line, side, ground in street_grounds(streets, allowed, half_street + reach)
            for piece in shapely.get_parts(ground) if isinstance(piece, Polygon) and piece.area >= 1.0]


class _Cursor:
    """Where a run's plots have grown to, one way along it."""

    def __init__(self, run: _Run, s: float, sign: float):
        self.run, self.s, self.sign = run, s, sign
        self._memo: tuple = (None, 0.0)

    def slot(self, w: float) -> tuple[float, float]:
        """The stations of the next ``w`` m wide slot."""
        return (self.s, self.s + w) if self.sign > 0 else (self.s - w, self.s)

    def distance(self, spec: PlotSpec) -> float:
        """From the middle of the next slot for ``spec`` to its target (kept while unmoved)."""
        key = (self.s, spec.width, spec.depth, id(spec))
        if self._memo[0] != key:
            r = self.run
            mid = r.at(float(np.clip(self.s + 0.5 * self.sign * spec.width, r.s_lo, r.s_hi)), 0.5 * spec.depth)
            self._memo = (key, float(np.hypot(*(mid - spec.target))))
        return self._memo[1]


def pack_frontage(
    streets: Sequence[LineString], allowed: BaseGeometry, plots: Sequence[PlotSpec], centre: np.ndarray,
    half_street: float, rng: np.random.Generator, gap_chance: float, district: str = "residential",
) -> tuple[list[Block], list[bool]]:
    """``plots`` fronting ``streets`` on ``allowed`` ground, in order: ``(a block per run that
    holds any, placed per plot)``.  See the module docstring."""
    streets = [g for g in streets if g.length > 1.0]
    if not streets or not plots:
        return [], [False] * len(plots)
    reach = 1.05 * max(p.depth for p in plots)
    runs = _runs(streets, allowed, half_street, reach)
    middle = Point(float(centre[0]), float(centre[1]))
    cursors = []
    for r in runs:
        s0 = float(np.clip(r.line.project(middle), r.s_lo, r.s_hi))
        cursors += [_Cursor(r, s0, 1.0), _Cursor(r, s0, -1.0)]
    placed = [False] * len(plots)
    for k, spec in enumerate(plots):
        placed[k] = _place_free(runs, spec) if spec.free else _grow(cursors, spec, rng, gap_chance)
    blocks = [blk for blk in (r.block(district) for r in runs) if blk is not None]
    return blocks, placed


def _grow(cursors: list[_Cursor], spec: PlotSpec, rng: np.random.Generator, gap_chance: float) -> bool:
    """``spec`` on the next slot of the cursor nearest its target that takes it; each cursor
    tried steps on past what stops it (a plot there, a skipped gap, a slot that won't fit)."""
    w, d = spec.width, spec.depth
    live = [c for c in cursors if c.run.within(*c.slot(w))]
    while live:
        c = min(live, key=lambda c: c.distance(spec))
        a, b = c.slot(w)
        if not c.run.within(a, b):
            live.remove(c)
            continue
        blocked = c.run.blocker(a, b)
        if blocked is not None:                    # on past a plot already there
            c.s = blocked[1] if c.sign > 0 else blocked[0]
            continue
        if spec.gaps and rng.random() < gap_chance:
            c.s += c.sign * w
            continue
        poly = c.run.try_place(a, b, d)
        if poly is None:
            c.s += c.sign * 0.25 * w
            continue
        c.run.place(a, b, d, spec.use, poly)
        c.s = b if c.sign > 0 else a
        return True
    return False


def _place_free(runs: list[_Run], spec: PlotSpec) -> bool:
    """``spec`` on the slot nearest its target on any run."""
    w, d = spec.width, spec.depth
    pt = Point(float(spec.target[0]), float(spec.target[1]))
    order = sorted(runs, key=lambda r: r.ground.distance(pt))
    for r in order[:6]:
        s = float(r.line.project(pt))
        for shift in (0.0, 0.5, -0.5, 1.0, -1.0):
            a = float(np.clip(s - 0.5 * w + shift * w, r.s_lo, max(r.s_hi - w, r.s_lo)))
            poly = r.try_place(a, a + w, d)
            if poly is not None:
                r.place(a, a + w, d, spec.use, poly)
                return True
    return False


def cut_blocks(blocks: Sequence[Block], cut: BaseGeometry, keep: Sequence[str] = ()) -> list[Block]:
    """``blocks`` with ``cut`` (lanes) taken out of their ground: a lot losing more than
    half of itself goes (unless its use is in ``keep``), the rest keep what remains; a
    block split in two becomes two."""
    out: list[Block] = []
    for blk in blocks:
        if not blk.polygon.intersects(cut):
            out.append(blk)
            continue
        kept: list[tuple[Polygon, str]] = []
        for i, lot in enumerate(blk.lots):
            parts = [g for g in shapely.get_parts(lot.difference(cut)) if isinstance(g, Polygon)]
            best = max(parts, key=lambda g: g.area) if parts else None
            if best is not None and (best.area >= 0.5 * lot.area or blk.use(i) in keep):
                kept.append((best, blk.use(i)))
        ground = blk.polygon.difference(cut)
        lines = blk.dividers.difference(cut)
        for piece in shapely.get_parts(ground):
            if not isinstance(piece, Polygon) or piece.is_empty:
                continue
            mine = [(g, u) for g, u in kept if piece.contains(g.representative_point())]
            if not mine:
                continue
            seeds = shapely.get_coordinates(shapely.point_on_surface(
                np.asarray([g for g, _ in mine], dtype=object))).reshape(-1, 2)
            out.append(Block(polygon=piece, district=blk.district, core=blk.core,
                             dividers=as_lines(lines.intersection(piece)) if not lines.is_empty
                             else MultiLineString(), seeds=seeds,
                             uses=np.asarray([u for _, u in mine], dtype=object)))
    return out
