"""Buildings on the lots: houses, barns, workshops, halls and the church (plan phase 3).

Each lot is read in its own frame (:func:`lot_frames`): its **front** — of the lot's
edges on a street (an arterial, minor street, close or alley, the market, a quay, the
green), the one most nearly along its narrow side (a strip's street end, not a corner
lot's long flank), the grander street between two such ends (a burgage plot faces the
market street, not its back lane); with none on a street, the edge most nearly along
its narrow side —
from which ``u`` runs along the street and ``n`` into the lot, with the front's
``width`` and the lot's ``depth`` back from it.

A **typology** maps a lot's type (:func:`lot_type`: its use, else its district and the
tier) to a placer (:data:`TYPOLOGY`) that lays rectangles in the frames of every lot of
that type at once; each is clipped to its lot and kept while most of it is left.  A lot
skewed to its street (its own axes — its minimum rectangle's — turned more than
``_SKEW_DEG`` from the frame's) is built twice with the same draws, square to the street
and square to itself (:func:`square_frames`), and keeps the second when its buildings'
floor, each counted by how nearly it is a rectangle, is ``lot_align_margin`` more: no
wedges where a skewed lot's sides cut a street-square house.  New
types (L-shaped farmhouses, courtyard houses, halls) are new entries.  First cut:

- **townhouse** (houses, market, quay): terraced — the whole frontage, at the street —
  on a narrow lot (≤ ``terrace_frontage_m``) in the core, the market or the quay; else
  detached, a gap either side, set back a little.  A dense lot is built back: the house
  up to half the lot deep (``backland_depth_m`` at most) and, on ``backland_share`` of
  them, a rear range down one side;
- **workshop** (a trade's lot, a craft quarter): a townhouse and a workshop at the back;
- **elite**: a house the whole frontage and a wing down one side behind it;
- **suburb**: detached;
- **garden**: a cottage (``garden_cottage_chance``) and now and then (``shed_chance``) a
  shed at the back — sparse ground holds people too;
- a village's **farm**: a house along the street and a barn behind; **cottage**
  (fishers, miners): a small house; **craft**: a house and a workshop behind;
  **manor**: a hall set back and a wing.

The church stands on the sacred precinct, its nave east–west and a chancel at the
east end (a tower at the west in a place of ``_TOWER_PEOPLE``), sized by people and
no longer than the precinct and shrunk (or turned along it) to fit.  Each other piece of
the precinct (a temple block, :func:`settlegen.layout.districts.assign_districts`) holds
a temple (:func:`temple`): a hall behind a portico or a rotunda, sized by its share of
the people.  Qt-free.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
import shapely
from shapely.affinity import rotate, translate
from shapely.geometry import Point, Polygon, box
from shapely.geometry.base import BaseGeometry

from settlegen.layout.blocks import street_fronts
from settlegen.layout.districts import CRAFTS, OPEN_DISTRICTS
from settlegen.layout.households import TRADES
from settlegen.layout.params import SettlementLayoutParams
from settlegen.layout.types import Block, Buildings, SettlementLayout

# Between two ends on streets, each step of a street's rank (0 lanes … 3 the market and
# main streets) counts this much — under the narrow-side preference (1 at most).
_RANK_WEIGHT = 0.3
# An edge fronts its street if at least this share of the lot's narrow side (not a tapered end).
_FRONT_SHARE = 0.6
# A building keeps at least this share of itself, clipped to its lot.
_MIN_KEPT = 0.6
# A lot whose own axes turn more than this (°) from its street's is tried square to itself too.
_SKEW_DEG = 3.0
# A back range down one side of a lot is at least this wide (m); a house this wide between its gaps.
_RANGE_M = 3.5
# A lot this much longer than wide has a long axis its frame keeps to (see :func:`square_frames`).
_ELONGATED = 1.3
_TOWER_PEOPLE = 800.0
_CRAFT_QUARTERS = frozenset(v[0] for v in CRAFTS.values())

BUILDING_LABELS = {
    "house": "House", "row": "Town house", "workshop": "Workshop", "barn": "Barn", "shed": "Shed",
    "farmhouse": "Farmhouse", "cottage": "Cottage", "hall": "Hall", "wing": "Wing", "church": "Church",
    "chancel": "Chancel", "tower": "Tower", "temple": "Temple", "portico": "Temple portico", "rotunda": "Temple",
}


@dataclass
class LotFrames:
    """Each lot's frame: see the module docstring.  All ``(k, …)``."""

    origin: np.ndarray       # (k, 2) the front edge's start
    u: np.ndarray            # (k, 2) along the front
    n: np.ndarray            # (k, 2) into the lot
    width: np.ndarray        # (k,) the front's length
    depth: np.ndarray        # (k,) how far the lot runs back from it

    def __len__(self) -> int:
        return len(self.width)

    def take(self, sel: np.ndarray) -> "LotFrames":
        return LotFrames(self.origin[sel], self.u[sel], self.n[sel], self.width[sel], self.depth[sel])

    def rects(self, x0: np.ndarray, x1: np.ndarray, y0: np.ndarray, y1: np.ndarray) -> np.ndarray:
        """``(k,)`` rectangles, ``x`` along the front and ``y`` back from it, in each frame."""
        corners = np.stack([np.stack([x0, y0], 1), np.stack([x1, y0], 1), np.stack([x1, y1], 1),
                            np.stack([x0, y1], 1), np.stack([x0, y0], 1)], 1)                   # (k, 5, 2)
        xy = self.origin[:, None, :] + corners[..., :1] * self.u[:, None, :] + corners[..., 1:] * self.n[:, None, :]
        return np.asarray(shapely.polygons(xy), dtype=object)


def _segments(lines: Sequence[BaseGeometry]) -> np.ndarray:
    """Every straight piece of ``lines`` (or polygon outlines) as its own two-point line."""
    geoms = np.asarray([g for g in lines if not g.is_empty], dtype=object)
    if not len(geoms):
        return np.zeros(0, dtype=object)
    xy, idx = shapely.get_coordinates(geoms, return_index=True)
    same = idx[:-1] == idx[1:]
    return np.asarray(shapely.linestrings(np.stack([xy[:-1][same], xy[1:][same]], 1)), dtype=object)


def lot_frames(lots: np.ndarray, fronts: Sequence[tuple[Sequence[BaseGeometry], float, int]]) -> LotFrames:
    """The frame of each of ``lots`` (see the module docstring).  ``fronts``: ``(lines or
    polygons, reach, rank)`` — a lot edge whose middle lies within ``reach`` of them is on a
    street of that rank."""
    k = len(lots)
    xy, idx = shapely.get_coordinates(shapely.get_exterior_ring(lots), return_index=True)
    same = idx[:-1] == idx[1:]
    a, b, owner = xy[:-1][same], xy[1:][same], idx[:-1][same]
    d = b - a
    length = np.hypot(d[:, 0], d[:, 1])
    dirs = d / np.maximum(length, 1e-9)[:, None]
    mids = shapely.points(0.5 * (a + b))
    rank = np.full(len(a), -1.0)                                         # -1: on no street
    for geoms, reach, r in fronts:
        segs = _segments(geoms)
        if len(segs):
            hit = shapely.STRtree(segs).query(mids, predicate="dwithin", distance=reach)[0]
            rank[hit] = np.maximum(rank[hit], float(r))
    # The narrow side of each lot's oriented envelope (none for a sliver: no edge preferred).
    with np.errstate(divide="ignore", invalid="ignore"):
        env = shapely.oriented_envelope(lots)
    narrow = np.zeros((k, 2))
    narrow_m = np.zeros(k)
    ok = shapely.get_type_id(env) == 3                                   # a polygon, not a collapsed line
    if ok.any():
        env_xy, env_idx = shapely.get_coordinates(shapely.get_exterior_ring(env[ok]), return_index=True)
        first = np.searchsorted(env_idx, np.arange(int(ok.sum())))
        s1 = env_xy[first + 1] - env_xy[first]
        s2 = env_xy[first + 2] - env_xy[first + 1]
        l1, l2 = np.hypot(*s1.T), np.hypot(*s2.T)
        narrow_m[ok] = np.minimum(l1, l2)
        narrow[ok] = np.where((l1 <= l2)[:, None], s1, s2) / np.maximum(narrow_m[ok], 1e-9)[:, None]
    narrow = np.nan_to_num(narrow)
    along = np.abs(np.sum(dirs * narrow[owner], axis=1))
    fronting = (rank >= 0) & (length >= _FRONT_SHARE * narrow_m[owner])
    score = along + np.where(fronting, 2.0 + _RANK_WEIGHT * rank, 0.0)
    order = np.lexsort((score, owner))
    best = order[np.r_[np.flatnonzero(np.diff(owner[order])), len(order) - 1]]    # each lot's top edge
    origin, u, width = a[best], dirs[best], length[best]
    n = np.stack([-u[:, 1], u[:, 0]], 1)
    inward = np.sum((shapely.get_coordinates(shapely.centroid(lots)) - origin) * n, axis=1) >= 0
    n = np.where(inward[:, None], n, -n)
    depth = np.zeros(k)
    np.maximum.at(depth, idx, np.sum((xy - origin[idx]) * n[idx], axis=1))
    return LotFrames(origin, u, n, width, depth)


def square_frames(lots: np.ndarray, street: LotFrames) -> tuple[LotFrames, np.ndarray]:
    """Each lot's frame square to its own sides: the axes of its minimum rectangle, ``n``
    (still into the lot from the street) the long one where the street front is the narrow
    end, the short where it is the long side — else the one nearer ``street.n`` — ``u`` across it;
    ``x`` over the lot's whole extent across, ``y = 0`` level with the middle of the street
    front (so the front corners a skew leaves stick out and are clipped about equally).
    ``(frames, skewed)``: ``skewed`` where the axes turn more than ``_SKEW_DEG`` from the
    street's (elsewhere the frame is the street's)."""
    k = len(lots)
    out = street.take(np.arange(k))
    with np.errstate(divide="ignore", invalid="ignore"):
        env = shapely.oriented_envelope(lots)
    ok = np.flatnonzero(shapely.get_type_id(env) == 3)
    skewed = np.zeros(k, dtype=bool)
    if not len(ok):
        return out, skewed
    env_xy, env_idx = shapely.get_coordinates(shapely.get_exterior_ring(env[ok]), return_index=True)
    first = np.searchsorted(env_idx, np.arange(len(ok)))
    s1 = env_xy[first + 1] - env_xy[first]
    l1, l2 = np.hypot(*s1.T), np.hypot(*(env_xy[first + 2] - env_xy[first + 1]).T)
    s1 = s1 / np.maximum(l1, 1e-9)[:, None]
    s2 = np.stack([-s1[:, 1], s1[:, 0]], 1)
    sn = street.n[ok]
    c1, c2 = np.sum(s1 * sn, axis=1), np.sum(s2 * sn, axis=1)
    # ``n`` down the lot's long axis if the street front is its narrow end, across it if
    # its long side; a squarish lot's (or squarish frame's), the axis nearer the street's.
    on_1 = np.abs(c1) >= np.abs(c2)
    long_1 = l1 >= l2
    W, D = street.width[ok], street.depth[ok]
    clear = (np.maximum(l1, l2) >= _ELONGATED * np.minimum(l1, l2)) & \
        (np.maximum(W, D) >= _ELONGATED * np.minimum(W, D))
    on_1 = np.where(clear, long_1 == (W <= D), on_1)
    n = np.where(on_1[:, None], s1, s2)
    mid = street.origin[ok] + 0.5 * street.width[ok, None] * street.u[ok]
    inward = np.sum((shapely.get_coordinates(shapely.centroid(lots[ok])) - mid) * n, axis=1) >= 0
    n = np.where(inward[:, None], n, -n)
    u = np.stack([-n[:, 1], n[:, 0]], 1)
    u = np.where((np.sum(u * street.u[ok], axis=1) >= 0)[:, None], u, -u)
    tilt = np.sum(n * sn, axis=1) < math.cos(math.radians(_SKEW_DEG))
    xy, idx = shapely.get_coordinates(shapely.get_exterior_ring(lots[ok]), return_index=True)
    pu, pn = np.sum(xy * u[idx], axis=1), np.sum(xy * n[idx], axis=1)
    lo_u, hi_u, hi_n = np.full(len(ok), np.inf), np.full(len(ok), -np.inf), np.full(len(ok), -np.inf)
    np.minimum.at(lo_u, idx, pu)
    np.maximum.at(hi_u, idx, pu)
    np.maximum.at(hi_n, idx, pn)
    y0 = np.sum(mid * n, axis=1)
    sel = ok[tilt]
    skewed[sel] = True
    out.origin[sel] = (lo_u[:, None] * u + y0[:, None] * n)[tilt]
    out.u[sel], out.n[sel] = u[tilt], n[tilt]
    out.width[sel] = (hi_u - lo_u)[tilt]
    out.depth[sel] = (hi_n - y0)[tilt]
    return out, skewed


# ── the typology ─────────────────────────────────────────────────────────────

# (kind, rectangles, which of the frames they stand in)
Placed = list[tuple[str, np.ndarray, np.ndarray]]
Placer = Callable[[LotFrames, np.ndarray, np.random.Generator, SettlementLayoutParams], Placed]


def _put(f: LotFrames, kind: str, which: np.ndarray, x0, x1, y0, y1,  # type: ignore[no-untyped-def]
         ) -> tuple[str, np.ndarray, np.ndarray]:
    """``kind`` on the frames ``which`` (bool or index), where it has room (over a metre each way)."""
    which = np.flatnonzero(which) if np.asarray(which).dtype == bool else np.asarray(which)
    x0, x1, y0, y1 = (np.broadcast_to(np.asarray(v, dtype=np.float64), (len(f),))[which] for v in (x0, x1, y0, y1))
    room = (x1 - x0 > 1.0) & (y1 - y0 > 1.0)
    which = which[room]
    return kind, f.take(which).rects(x0[room], x1[room], y0[room], y1[room]), which


def _houses(f: LotFrames, dense: np.ndarray, rng: np.random.Generator, p: SettlementLayoutParams,
            ) -> tuple[Placed, np.ndarray]:
    """Town houses: terraced on narrow dense lots, else detached.  ``(placed, where each ends back)``."""
    k = len(f)
    W, D = f.width, f.depth
    # Terraced on narrow dense lots, and on any too narrow for a house between gaps.
    row = (dense | (W < _RANGE_M + 2.0 * float(p.house_gap_m))) & (W <= float(p.terrace_frontage_m))
    g = np.where(row, 0.0, np.minimum(float(p.house_gap_m), 0.15 * W))
    w = np.where(row, W, np.minimum(W - 2.0 * g, float(p.max_house_width_m)))
    x0 = g + rng.uniform(0.0, 1.0, k) * np.maximum(W - 2.0 * g - w, 0.0)
    y0 = np.where(row, 0.0, rng.uniform(0.0, 2.0, k))
    deep = np.where(dense, np.clip(0.5 * D, float(p.house_depth_m), float(p.backland_depth_m)), float(p.house_depth_m))
    y1 = y0 + np.minimum(deep * rng.uniform(0.85, 1.15, k), 0.7 * D)
    return [_put(f, "row", row, x0, x0 + w, y0, y1), _put(f, "house", ~row, x0, x0 + w, y0, y1)], y1


def _townhouse(f: LotFrames, dense: np.ndarray, rng: np.random.Generator, p: SettlementLayoutParams) -> Placed:
    """Town houses; a dense lot's back built up by a range down one side or, on a lot too
    narrow for one beside a yard, a back house across it behind a yard."""
    placed, ends = _houses(f, dense, rng, p)
    W, D = f.width, f.depth
    k = len(f)
    back = dense & (rng.random(k) < float(p.backland_share))
    left = rng.random(k) < 0.5
    side = 0.5 * W >= _RANGE_M
    w = np.minimum(5.0, 0.5 * W)
    x0 = np.where(left, 0.0, W - w)
    y0 = ends + 3.0
    y1 = np.minimum(y0 + float(p.house_depth_m), D - 1.5)
    return placed + [_put(f, "wing", back & side & (D - 1.5 - ends >= 4.0), x0, x0 + w, ends, D - 1.5),
                     _put(f, "wing", back & ~side & (y1 - y0 >= 4.0), 0.0, W, y0, y1)]


def _suburb(f: LotFrames, dense: np.ndarray, rng: np.random.Generator, p: SettlementLayoutParams) -> Placed:
    return _houses(f, np.zeros(len(f), dtype=bool), rng, p)[0]


def _workshop(f: LotFrames, dense: np.ndarray, rng: np.random.Generator, p: SettlementLayoutParams) -> Placed:
    """A town house and a workshop across the back of the lot."""
    placed, ends = _houses(f, dense, rng, p)
    W, D = f.width, f.depth
    wd = float(p.workshop_depth_m)
    w = np.minimum(0.8 * W, 10.0)
    x0 = 0.5 * (W - w)
    back = D >= ends + 2.0 + wd + 1.0
    return placed + [_put(f, "workshop", back, x0, x0 + w, D - 1.0 - wd, D - 1.0)]


def _elite(f: LotFrames, dense: np.ndarray, rng: np.random.Generator, p: SettlementLayoutParams) -> Placed:
    """A house the whole frontage, a wing down one side behind it (round a yard)."""
    W, D = f.width, f.depth
    y1 = np.minimum(1.3 * float(p.house_depth_m), 0.6 * D)
    left = rng.random(len(f)) < 0.5
    wx0 = np.where(left, 0.5, W - 6.5)
    wing_y1 = np.minimum(D - 2.0, y1 + 0.5 * D)
    return [_put(f, "house", np.ones(len(f), dtype=bool), 0.5, W - 0.5, 0.0, y1),
            _put(f, "wing", W >= 14.0, wx0, wx0 + 6.0, y1, wing_y1)]


def _garden(f: LotFrames, dense: np.ndarray, rng: np.random.Generator, p: SettlementLayoutParams) -> Placed:
    """A cottage at the front, now and then; a shed at the back, now and then."""
    W, D = f.width, f.depth
    k = len(f)
    cottage = rng.random(k) < float(p.garden_cottage_chance)
    w = np.minimum(W - 2.0, 8.0)
    x0 = 1.0 + rng.uniform(0.0, 1.0, k) * np.maximum(W - 2.0 - w, 0.0)
    y0 = rng.uniform(1.0, 4.0, k)
    y1 = y0 + 0.75 * float(p.house_depth_m)
    shed = (rng.random(k) < float(p.shed_chance)) & (~cottage | (D - 5.0 >= y1 + 1.0))
    return [_put(f, "cottage", cottage, x0, x0 + w, y0, y1),
            _put(f, "shed", shed, W - 5.0, W - 1.0, D - 5.0, D - 1.0)]


def _village_house(f: LotFrames, rng: np.random.Generator, p: SettlementLayoutParams, widest: float,
                   depth_mult: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(x0, x1, y0, y1)`` of a village house along the street, set back a little."""
    k = len(f)
    W = f.width
    w = np.minimum(W - 2.0, widest) * rng.uniform(0.85, 1.0, k)
    x0 = 1.0 + rng.uniform(0.0, 1.0, k) * np.maximum(W - 2.0 - w, 0.0)
    y0 = rng.uniform(1.0, 4.0, k)
    y1 = y0 + float(p.village_house_depth_m) * depth_mult * rng.uniform(0.9, 1.1, k)
    return x0, x0 + w, y0, np.minimum(y1, 0.6 * f.depth)


def _farm(f: LotFrames, dense: np.ndarray, rng: np.random.Generator, p: SettlementLayoutParams) -> Placed:
    """A house along the street (a longhouse, up to 14 m) and a barn behind."""
    x0, x1, y0, y1 = _village_house(f, rng, p, 14.0)
    W, D = f.width, f.depth
    wb = np.minimum(W - 2.0, 12.0) * rng.uniform(0.8, 1.0, len(f))
    bx0 = 1.0 + rng.uniform(0.0, 1.0, len(f)) * np.maximum(W - 2.0 - wb, 0.0)
    by0 = y1 + rng.uniform(3.0, 6.0, len(f))
    by1 = by0 + float(p.barn_depth_m)
    return [_put(f, "farmhouse", np.ones(len(f), dtype=bool), x0, x1, y0, y1),
            _put(f, "barn", by1 <= D - 1.0, bx0, bx0 + wb, by0, by1)]


def _cottage(f: LotFrames, dense: np.ndarray, rng: np.random.Generator, p: SettlementLayoutParams) -> Placed:
    x0, x1, y0, y1 = _village_house(f, rng, p, 8.0, 0.75)
    return [_put(f, "cottage", np.ones(len(f), dtype=bool), x0, x1, y0, y1)]


def _craft(f: LotFrames, dense: np.ndarray, rng: np.random.Generator, p: SettlementLayoutParams) -> Placed:
    """A house and a workshop behind it."""
    x0, x1, y0, y1 = _village_house(f, rng, p, 9.0)
    W, D = f.width, f.depth
    ww = np.minimum(W - 2.0, 7.0)
    wy0 = y1 + 2.0
    wy1 = wy0 + float(p.workshop_depth_m)
    return [_put(f, "house", np.ones(len(f), dtype=bool), x0, x1, y0, y1),
            _put(f, "workshop", wy1 <= D - 0.5, 1.0, 1.0 + ww, wy0, wy1)]


def _manor(f: LotFrames, dense: np.ndarray, rng: np.random.Generator, p: SettlementLayoutParams) -> Placed:
    """A hall set back from the street and a wing back from one end of it."""
    W, D = f.width, f.depth
    y0 = np.full(len(f), 4.0)
    y1 = y0 + 1.4 * float(p.village_house_depth_m)
    left = rng.random(len(f)) < 0.5
    wx0 = np.where(left, 0.15 * W, 0.85 * W - 6.0)
    return [_put(f, "hall", np.ones(len(f), dtype=bool), 0.15 * W, 0.85 * W, y0, y1),
            _put(f, "wing", np.ones(len(f), dtype=bool), wx0, wx0 + 6.0, y1, np.minimum(y1 + 15.0, y1 + 0.4 * D))]


TYPOLOGY: dict[str, Placer] = {
    "townhouse": _townhouse, "suburb": _suburb, "workshop": _workshop, "elite": _elite, "garden": _garden,
    "farm": _farm, "cottage": _cottage, "craft": _craft, "manor": _manor,
}


def lot_type(block: Block, i: int, tier: int) -> Optional[str]:
    """The typology entry for lot ``i`` of ``block`` (None: nothing is built on it)."""
    use, district = block.use(i), block.lot_district(i)
    if tier == 5:
        if use in ("farm", "manor"):
            return use
        return "craft" if use in TRADES else "cottage"
    if district in OPEN_DISTRICTS:
        return None
    if district in ("gardens", "elite", "suburb"):
        return {"gardens": "garden"}.get(district, district)
    if use in TRADES or district in _CRAFT_QUARTERS:
        return "workshop"
    return "townhouse"


# ── the church ───────────────────────────────────────────────────────────────

def church(precinct: BaseGeometry, people: float) -> list[tuple[str, Polygon]]:
    """The church on the largest piece of ``precinct``: ``(kind, polygon)`` for the nave,
    the chancel and (``_TOWER_PEOPLE`` and up) the tower; none if it won't fit."""
    pieces = [g for g in shapely.get_parts(precinct) if isinstance(g, Polygon)]
    if not pieces:
        return []
    ground = max(pieces, key=lambda g: g.area)
    room = ground.buffer(-1.5)
    if room.is_empty:
        return []
    c = ground.centroid if ground.contains(ground.centroid) else ground.representative_point()
    env = shapely.get_coordinates(ground.minimum_rotated_rectangle)
    sides = [env[1] - env[0], env[2] - env[1]]
    long_side = max(sides, key=lambda v: float(np.hypot(*v)))
    # By people, but no longer than the precinct runs (tower and chancel included: 1.3 × the nave's span).
    length = min(float(np.clip(5.0 + max(people, 0.0) ** 0.45, 18.0, 90.0)), 0.8 * float(np.hypot(*long_side)) / 1.3)
    for angle in (0.0, math.degrees(math.atan2(long_side[1], long_side[0]))):   # east–west, else along the ground
        for scale in (1.0, 0.85, 0.7, 0.55, 0.4):
            L, w = length * scale, 0.36 * length * scale
            parts = [("church", box(-0.5 * L, -0.5 * w, 0.25 * L, 0.5 * w)),
                     ("chancel", box(0.25 * L, -0.35 * w, 0.5 * L, 0.35 * w))]
            if people >= _TOWER_PEOPLE:
                t = 0.8 * w
                parts.append(("tower", box(-0.5 * L - t, -0.5 * t, -0.5 * L, 0.5 * t)))
            placed = [(k, translate(rotate(g, angle, origin=(0, 0)), c.x, c.y)) for k, g in parts]
            if all(room.contains(g) for _, g in placed):
                return placed
    return []


def temple(ground: Polygon, people: float, rng: np.random.Generator) -> list[tuple[str, Polygon]]:
    """A temple on ``ground`` for ``people``: a hall behind a portico or (one in three) a
    rotunda, along the ground's long axis (else east–west), shrunk to fit; none if it won't."""
    room = ground.buffer(-1.5)
    if room.is_empty:
        return []
    c = ground.centroid if ground.contains(ground.centroid) else ground.representative_point()
    env = shapely.get_coordinates(ground.minimum_rotated_rectangle)
    long_side = max([env[1] - env[0], env[2] - env[1]], key=lambda v: float(np.hypot(*v)))
    length = min(float(np.clip(6.0 + max(people, 0.0) ** 0.4, 12.0, 50.0)), 0.8 * float(np.hypot(*long_side)))
    round_plan = rng.uniform() < 1.0 / 3.0
    facing = 180.0 * float(rng.integers(2))        # the portico at one end or the other
    for angle in (math.degrees(math.atan2(long_side[1], long_side[0])) + facing, facing):
        for scale in (1.0, 0.85, 0.7, 0.55, 0.4):
            L = length * scale
            if round_plan:
                r = 0.35 * L
                drum = Point(0.0, 0.0).buffer(r, quad_segs=8)
                # The portico butts onto the drum (an overlap would be cut out by the odd-even fill).
                parts = [("rotunda", drum),
                         ("portico", box(r - 0.1 * L, -0.2 * L, r + 0.2 * L, 0.2 * L).difference(drum))]
            else:
                w = 0.5 * L
                parts = [("temple", box(-0.5 * L, -0.5 * w, 0.3 * L, 0.5 * w)),
                         ("portico", box(0.3 * L, -0.5 * w, 0.5 * L, 0.5 * w))]
            placed = [(k, translate(rotate(g, angle, origin=(0, 0)), c.x, c.y)) for k, g in parts]
            if all(room.contains(g) for _, g in placed):
                return placed
    return []


def sacred_buildings(precinct: BaseGeometry, sacred: Optional[np.ndarray], people: float,
                     rng: np.random.Generator) -> list[tuple[str, Polygon]]:
    """The church on the main piece of ``precinct`` (the one at ``sacred``, else the largest) and
    a temple on each other piece, the people shared out among them all."""
    pieces = [g for g in shapely.get_parts(precinct) if isinstance(g, Polygon) and not g.is_empty]
    if not pieces:
        return []
    if sacred is not None:
        at = Point(float(sacred[0]), float(sacred[1]))
        main = int(np.argmin([g.distance(at) for g in pieces]))
    else:
        main = int(np.argmax([g.area for g in pieces]))
    out = church(pieces[main], people)
    share = people / len(pieces)
    for i, g in enumerate(pieces):
        if i != main:
            out += temple(g, share, rng)
    return out


# ── all of them ──────────────────────────────────────────────────────────────

def _fronts(layout: SettlementLayout, params: SettlementLayoutParams) -> list[tuple[list, float, int]]:
    """``(lines or open grounds, reach, rank)`` a lot's front can lie on (see :func:`lot_frames`
    and :func:`settlegen.layout.blocks.street_fronts`)."""
    return street_fronts(layout.arterials, layout.arterial_kinds, layout.arterial_m, layout.secondary_m,
                         layout.streets, float(params.street_m), [*layout.closes, *layout.alleys], layout.lane_m,
                         [(layout.plaza, 3), (layout.green, 2), (layout.quays, 2)])


def _largest(g: BaseGeometry) -> Optional[Polygon]:
    parts = [q for q in shapely.get_parts(g) if isinstance(q, Polygon) and not q.is_empty]
    return max(parts, key=lambda q: q.area) if parts else None


def _floor(buildings: np.ndarray) -> np.ndarray:
    """Each building's floor, counted by how nearly it is a rectangle: its area × its share of
    its minimum rectangle (a lot's sides cutting a corner off cost it twice)."""
    area = shapely.area(buildings)
    with np.errstate(divide="ignore", invalid="ignore"):
        rect = shapely.area(shapely.oriented_envelope(buildings))
    return area * area / np.maximum(rect, 1e-9)


def _clip(rects: np.ndarray, lot_i: np.ndarray, lots: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``rects`` (on ``lots[lot_i]``) clipped to their lots, those keeping ``_MIN_KEPT`` of
    themselves: ``(polygons, their lot indices)``.  Mostly single polygons; a lot's odd edge
    can leave pieces (the largest is kept)."""
    if not len(rects):
        return np.zeros(0, dtype=object), np.zeros(0, dtype=np.int64)
    # On a fixed grid: a rectangle sharing its lot's front edge can otherwise come back unclipped.
    clipped = shapely.intersection(rects, lots[lot_i], grid_size=1e-9)
    keep = shapely.area(clipped) >= _MIN_KEPT * shapely.area(rects)
    single = keep & (shapely.get_type_id(clipped) == 3)
    split = np.flatnonzero(keep & ~single)
    pieces = [_largest(g) for g in clipped[split]]
    got = np.concatenate([np.flatnonzero(single), split[[q is not None for q in pieces]]]).astype(np.int64)
    rest = np.asarray([q for q in pieces if q is not None], dtype=object)
    return np.concatenate([clipped[single], rest]), lot_i[got]


def place_buildings(layout: SettlementLayout, people: float, tier: int, rng: np.random.Generator,
                    params: SettlementLayoutParams) -> Buildings:
    """Every building of ``layout``.  See the module docstring."""
    refs: list[tuple[int, int]] = []
    lots: list[Polygon] = []
    types: list[str] = []
    dense: list[bool] = []
    for bi, b in enumerate(layout.blocks):
        for i, lot in enumerate(b.lots):
            t = lot_type(b, i, tier)
            if t is not None:
                refs.append((bi, i))
                lots.append(lot)
                types.append(t)
                dense.append(b.core or b.lot_district(i) in ("market", "quay"))
    polys: list[np.ndarray] = []                   # arrays of buildings, their kinds and (block, lot)s
    kinds: list[np.ndarray] = []
    where: list[np.ndarray] = []
    if lots:
        lots_a = np.asarray(lots, dtype=object)
        refs_a = np.asarray(refs, dtype=np.int64)
        types_a = np.asarray(types)
        dense_a = np.asarray(dense, dtype=bool)
        frames = lot_frames(lots_a, _fronts(layout, params))
        square, skewed = square_frames(lots_a, frames)
        margin = 1.0 + float(params.lot_align_margin)
        for t in sorted(set(types)):
            sel = np.flatnonzero(types_a == t)
            # Square to the street everywhere and, with the same draws, to the lot on skewed lots.
            state = rng.bit_generator.state
            placed = TYPOLOGY[t](frames.take(sel), dense_a[sel], rng, params)
            on_street = [_clip(rects, sel[which], lots_a) for _, rects, which in placed]
            on_lot: list[tuple[np.ndarray, np.ndarray]] = []
            if skewed[sel].any():
                after = rng.bit_generator.state
                rng.bit_generator.state = state
                for _, rects, which in TYPOLOGY[t](square.take(sel), dense_a[sel], rng, params):
                    tilted = skewed[sel[which]]
                    on_lot.append(_clip(rects[tilted], sel[which[tilted]], lots_a))
                rng.bit_generator.state = after
            # A skewed lot builds square to itself when that keeps ``margin`` × the floor (all its
            # buildings'), each counted by how nearly it is a rectangle (see :func:`_floor`).
            use_lot = np.zeros(len(lots_a), dtype=bool)
            if on_lot:
                floor_street, floor_lot = np.zeros(len(lots_a)), np.zeros(len(lots_a))
                for (built, lot_i), floor in [*((b, floor_street) for b in on_street),
                                              *((b, floor_lot) for b in on_lot)]:
                    tilted = skewed[lot_i]
                    np.add.at(floor, lot_i[tilted], _floor(built[tilted]))
                use_lot = skewed & (floor_lot > margin * floor_street)
            for j, (kind, _, _) in enumerate(placed):
                for (built, lot_i), take in [(on_street[j], ~use_lot), *([(on_lot[j], use_lot)] if on_lot else [])]:
                    m = take[lot_i]
                    polys.append(built[m])
                    kinds.append(np.full(int(m.sum()), kind, dtype=object))
                    where.append(refs_a[lot_i[m]])
    for kind, poly in sacred_buildings(layout.precinct, layout.anchors.get("sacred"), people, rng):
        polys.append(np.asarray([poly], dtype=object))
        kinds.append(np.asarray([kind], dtype=object))
        where.append(np.full((1, 2), -1, dtype=np.int64))
    if not polys:
        return Buildings.empty()
    return Buildings(np.concatenate(polys), np.concatenate(kinds), np.concatenate(where).reshape(-1, 2))
