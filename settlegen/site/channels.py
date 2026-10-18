"""River channels as polygons: each line widened to its bankfull width, wandering, with islands (eyots)
in wide rivers (:func:`natural_channels`), or plainly (:func:`channel_polygons`)."""
from __future__ import annotations

import zlib
from typing import Optional, Sequence

import numpy as np
import shapely
from scipy.ndimage import gaussian_filter1d
from shapely.geometry import LineString, MultiPolygon, Point, Polygon

from settlegen.site.params import SettlementRiverParams
from settlegen.site.types import RiverLine, _as_multipolygon


# Narrowest channel polygon (m): streams are drawn, however small.
_MIN_CHANNEL_M = 1.0


def channel_polygons(lines: Sequence[RiverLine], clip: Optional[Polygon] = None) -> MultiPolygon:
    """Each dry segment widened to its bankfull width (clipped to ``clip``, if given):
    the channels' plain outline, without :func:`natural_channels`' wander and islands."""
    a, b, half = [], [], []
    for line in lines:
        dry = ~line.in_water
        a.append(line.xy[:-1][dry])
        b.append(line.xy[1:][dry])
        half.append(np.maximum(0.5 * line.width_m[dry], 0.5 * _MIN_CHANNEL_M))
    if not a or not sum(len(x) for x in a):
        return MultiPolygon()
    a_xy, b_xy, h = np.concatenate(a), np.concatenate(b), np.concatenate(half)
    # Flat ends, so a channel stops square at a shore or the frame instead of in a disc
    # its full width; each joint between two segments rounded at the narrower one's width.
    parts = list(shapely.buffer(shapely.linestrings(np.stack([a_xy, b_xy], axis=1)), h, cap_style="flat"))
    joint = np.linalg.norm(a_xy[1:] - b_xy[:-1], axis=1) < 1e-6
    if joint.any():
        k = np.flatnonzero(joint)
        parts.extend(shapely.buffer(shapely.points(a_xy[k + 1]), np.minimum(h[k], h[k + 1]), quad_segs=4))
    merged = shapely.union_all(parts)
    return _as_multipolygon(merged.intersection(clip) if clip is not None else merged)


_DENSIFY_M = 10.0


# How far (× its width) a channel runs on past an end that meets water, the frame's edge or another channel.
_END_RUN_WIDTHS = 2.0


def _line_rng(line: RiverLine) -> np.random.Generator:
    """A stream of its own per line (its ends, to the metre), so a channel is the same however often it is built."""
    key = np.round(np.concatenate([line.xy[0], line.xy[-1]])).astype(np.int64)
    return np.random.default_rng([zlib.crc32(key.tobytes()), len(line.xy)])


def _smooth_noise(n: int, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """``n`` samples of unit-variance noise correlated over ``sigma`` samples."""
    sigma = max(float(sigma), 0.5)
    pad = int(np.ceil(3.0 * sigma))
    raw = gaussian_filter1d(rng.standard_normal(n + 2 * pad), sigma, mode="wrap")[pad:pad + n]
    # Filtered white noise has std 1 / sqrt(2 sqrt(pi) sigma): scaled by theory, not by a short sample.
    return raw * np.sqrt(2.0 * np.sqrt(np.pi) * sigma)


def _lens(u: np.ndarray, power: float) -> np.ndarray:
    """1 at ``u`` = 0.5, falling to 0 at 0 and 1 (and outside)."""
    return np.clip(1.0 - (2.0 * u - 1.0) ** 2, 0.0, 1.0) ** power


def _end_heading(xy: np.ndarray, reach: float) -> np.ndarray:
    """Unit direction out of ``xy``'s last vertex: the chord from ``reach`` back along it
    (a chord, not the last segment, so a kink at the very end doesn't steer it)."""
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy[::-1], axis=0), axis=1))]
    back = xy[::-1][int(min(np.searchsorted(s, reach), len(xy) - 1))]
    d = xy[-1] - back
    if float(np.linalg.norm(d)) < 1e-9:
        d = xy[-1] - xy[-2]
    return d / max(float(np.linalg.norm(d)), 1e-12)


def _line_channel(
    line: RiverLine, params: SettlementRiverParams,
    extend: tuple[Optional[np.ndarray], Optional[np.ndarray]] = (None, None),
    avoid: Sequence[tuple[LineString, float]] = (),
) -> tuple[list[Polygon], list[Polygon]]:
    """``(bank quads, islands)`` of one line's channel.  ``avoid``: the other lines and
    their widths, which islands keep clear of (a confluence partway along this line).
    See :func:`natural_channels`."""
    xy = np.asarray(line.xy, dtype=np.float64)
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    keep = np.r_[True, seg > 1e-9]
    xy = xy[keep]
    width = np.r_[line.width_m, line.width_m[-1:]][keep]
    wet = np.r_[line.in_water, line.in_water[-1:]][keep]
    if len(xy) < 2:
        return [], []
    # Ends that run on (into water, past the frame, into another channel) carried
    # straight on, so the band overruns what cuts it rather than stopping square short.
    length0 = float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))
    pre = post = 0.0

    def heading(given: np.ndarray, out_of: np.ndarray, w: float) -> np.ndarray:
        own = _end_heading(out_of, min(w, length0 / 3.0))
        if float(np.linalg.norm(given)) < 0.5:
            return own
        # Into still water: halfway between the line's own heading and the water's middle.
        mid = own + given
        norm = float(np.linalg.norm(mid))
        return mid / norm if norm > 1e-6 else given

    if extend[0] is not None:
        pre = _END_RUN_WIDTHS * max(float(width[0]), _MIN_CHANNEL_M)
        xy = np.vstack([xy[0] + heading(extend[0], xy[::-1], float(width[0])) * pre, xy])
        width, wet = np.r_[width[:1], width], np.r_[wet[:1], wet]
    if extend[1] is not None:
        post = _END_RUN_WIDTHS * max(float(width[-1]), _MIN_CHANNEL_M)
        xy = np.vstack([xy, xy[-1] + heading(extend[1], xy, float(width[-1])) * post])
        width, wet = np.r_[width, width[-1:]], np.r_[wet, wet[-1:]]
    # Densified, so the wander and the islands have vertices to bend.
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    n = np.maximum(np.ceil(seg / _DENSIFY_M).astype(int), 1)
    k = np.repeat(np.arange(len(seg)), n)
    u = np.concatenate([np.arange(m) / m for m in n])
    p = np.concatenate([xy[k] + u[:, None] * (xy[k + 1] - xy[k]), xy[-1:]])
    w = np.maximum(np.r_[width[k], width[-2:-1]], _MIN_CHANNEL_M)
    dry = ~np.r_[wet[k], wet[-2:-1]]
    # The course smoothed over about a width: a channel's bends are wide for its width
    # (offset from a tight bend, a wide river's inner bank balloons into a lobe).
    sigma_c = float(params.course_smoothing) * float(np.median(w)) / _DENSIFY_M
    if sigma_c > 0.5 and len(p) > 3:
        p = np.column_stack([gaussian_filter1d(p[:, j], sigma_c, mode="nearest") for j in range(2)])
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    # The line's own stretch (islands keep to it), less the runs on.
    s_lo = float(s[-1]) * pre / max(pre + length0 + post, 1e-9)
    s_hi = float(s[-1]) * (pre + length0) / max(pre + length0 + post, 1e-9)
    tangent = np.gradient(p, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])

    rng = _line_rng(line)
    sigma = float(params.wander_widths) * float(np.median(w)) / _DENSIFY_M
    half = 0.5 * w * np.exp(float(params.width_wander) * _smooth_noise(len(p), sigma, rng))
    shift = float(params.bank_wander) * w * _smooth_noise(len(p), sigma, rng)
    left = np.maximum(half + shift, 0.25 * w)
    right = np.maximum(half - shift, 0.25 * w)

    islands: list[Polygon] = []
    total = s_hi - s_lo
    count = rng.poisson(float(params.islands_per_km) * total / 1000.0) if total > 0 else 0
    taken: list[tuple[float, float]] = []
    for at in np.sort(rng.uniform(s_lo, s_hi, count)):
        i = int(np.searchsorted(s, at))
        i = min(i, len(p) - 1)
        if w[i] < float(params.island_min_width_m):
            continue
        breadth = float(params.island_width) * w[i] * rng.uniform(0.7, 1.3)
        length = float(params.island_length) * breadth * rng.uniform(0.7, 1.3)
        a, b = at - 0.5 * length, at + 0.5 * length
        # Clear of the line's ends (a mouth, a confluence, the frame) and of other islands, by half a width.
        lo, hi = a - 0.5 * w[i], b + 0.5 * w[i]
        if lo < s_lo or hi > s_hi or any(lo < t1 and hi > t0 for t0, t1 in taken):
            continue
        span = (s >= lo) & (s <= hi)
        if not dry[span].all() or (w[span] < float(params.island_min_width_m)).any():
            continue
        ends = np.flatnonzero(span)[[0, -1]]
        if float(np.dot(tangent[ends[0]], tangent[ends[1]])) < np.cos(np.radians(60.0)):
            continue              # not on a sharp bend
        # Clear of a channel joining (or leaving) partway along: half a width past both banks.
        stretch = LineString(p[span]) if span.sum() >= 2 else Point(p[i])
        if any(other.distance(stretch) < w[i] + 0.5 * ow for other, ow in avoid):
            continue
        taken.append((lo, hi))
        # The channel widens by a share of the island's breadth, tapered over twice its
        # length, so the banks swell gently rather than bulge.
        ta, tb = a - 0.5 * length, b + 0.5 * length
        widen = 0.5 * float(params.island_widening) * breadth * _lens((s - ta) / (tb - ta), 1.0)
        left, right = left + widen, right + widen
        on = (s >= a) & (s <= b)
        if on.sum() < 3:
            continue
        # Blunt at its head, tapering to a long tail downstream (lines run downstream),
        # on one line along the channel's middle there.
        u = (s[on] - a) / (b - a)
        shape = u ** 0.5 * (1.0 - u) ** 1.5
        thick = 0.5 * breadth * shape / max(float(shape.max()), 1e-12)
        centre = float(np.mean(0.5 * (left[on] - right[on])))
        mid = p[on] + normal[on] * centre
        ring = np.concatenate([mid + normal[on] * thick[:, None], (mid - normal[on] * thick[:, None])[::-1]])
        isle = shapely.make_valid(Polygon(ring))
        parts = [g for g in getattr(isle, "geoms", [isle]) if isinstance(g, Polygon) and not g.is_empty]
        if parts:
            islands.append(max(parts, key=lambda g: g.area))

    lb, rb = p + normal * left[:, None], p - normal * right[:, None]
    quads = shapely.make_valid(shapely.polygons(np.stack([lb[:-1], lb[1:], rb[1:], rb[:-1], lb[:-1]], axis=1)))
    return list(np.asarray(quads)), islands


_RunOn = Optional[np.ndarray]


def _runs_on(line: RiverLine, k: int, lines: Sequence[RiverLine], others: list[LineString],
             still: MultiPolygon, half: Optional[float]) -> tuple[_RunOn, _RunOn]:
    """How each end of ``line`` runs on: ``None`` (a source: it doesn't), a unit heading
    (into the still water within about a width: towards its middle there, which
    :func:`_line_channel` blends with the line's own heading), or a zero vector (on along the line: past the
    frame's edge, ``half``, or into another line's channel at a confluence)."""
    out: list[_RunOn] = []
    for end, w in ((line.xy[0], float(line.width_m[0])), (line.xy[-1], float(line.width_m[-1]))):
        reach = 1.5 * max(w, _MIN_CHANNEL_M) + 20.0
        pt = Point(end)
        if not still.is_empty and still.distance(pt) < reach:
            water = still.intersection(pt.buffer(2.0 * reach))
            to = np.array([water.centroid.x, water.centroid.y]) - end if not water.is_empty else np.zeros(2)
            norm = float(np.linalg.norm(to))
            out.append(to / norm if norm > 1e-6 else np.zeros(2))
        elif (half is not None and half - float(np.max(np.abs(end))) < reach) \
                or any(j != k and others[j].distance(pt) < reach + 0.5 * float(np.max(lines[j].width_m))
                       for j in range(len(lines))):
            out.append(np.zeros(2))
        else:
            out.append(None)
    return out[0], out[1]


def natural_channels(
    lines: Sequence[RiverLine], still: MultiPolygon, params: Optional[SettlementRiverParams] = None,
    half: Optional[float] = None,
) -> MultiPolygon:
    """The ground the channels cover, less the ``still`` water (sea and lakes) they run
    into and their islands.  Each line's channel is its bankfull width, wandering
    (``width_wander``, lognormal), its centre off the line (``bank_wander`` × width),
    both correlated over ``wander_widths`` widths; rivers at least
    ``island_min_width_m`` wide get ``islands_per_km`` islands (Poisson): long and
    narrow, ``island_width`` × the width broad and ``island_length`` × that long
    (±30%), blunt upstream and tapering downstream, along the channel's middle,
    clear of the line's ends, sharp bends and other lines (a width plus half theirs); the channel widens round each by
    ``island_widening`` × its breadth, tapered over twice its length.  Unclipped: as far as the lines reach.
    In-water stretches are widened too, so a channel meets its sea or lake whole, and
    land between the two narrower than ``min_land_m`` goes under the channel.  The
    course is the line smoothed over ``course_smoothing`` widths, and an end that
    meets water, another line or the frame's edge (``half``, the frame's half-side)
    runs on straight two widths (into still water, turned towards its middle), so the band
    is cut there, not stopped square."""
    params = params or SettlementRiverParams()
    quads, islands = [], []
    others = [LineString(line.xy) for line in lines]
    for k, line in enumerate(lines):
        avoid = [(others[j], float(np.max(lines[j].width_m))) for j in range(len(lines)) if j != k]
        q, i = _line_channel(line, params, _runs_on(line, k, lines, others, still, half), avoid)
        quads.extend(q)
        islands.extend(i)
    if not quads:
        return MultiPolygon()
    merged = shapely.union_all(quads)
    d = 0.5 * float(params.min_land_m)
    if d > 0 and not still.is_empty:
        # Closed against the still water, near the channels only.
        near = merged.buffer(4.0 * d)
        closed = shapely.union_all([merged, still.intersection(near)]).buffer(d).buffer(-d)
        merged = shapely.union_all([merged, closed.intersection(merged.buffer(2.0 * d))])
    # The quads' union can leave slivers unfilled on tight bends: only islands are holes.
    merged = _as_multipolygon(merged)
    merged = MultiPolygon([Polygon(g.exterior) for g in merged.geoms])
    cut = shapely.union_all([*islands, still]) if islands or not still.is_empty else None
    if cut is not None and not cut.is_empty:
        merged = merged.difference(cut)
    return _as_multipolygon(merged)
