"""River valleys carved into a refined map once its rivers are routed.

A re-routed river seeks the valleys of its corridor (:mod:`region.rivers`) but
nothing makes it run downhill: over the relief spectrum's bumps it climbs.  After
routing, each tier carves its drawn rivers (and streams) into its elevation:

- **Bed.**  Along each drawn line, the ground (nearest texel, every
  ``step_texels``) is fitted with a never-rising profile: isotonic regression at
  ``quantile`` (0 cuts every rise down to the lowest ground upstream; 0.5 is the
  median fit, half cut and half filled).  Each dry stretch ends no lower than
  the water it runs into (a lake's surface, or the bed of the stem it joins, so
  stems go first; never below sea level), unless it runs over lower ground
  (then it climbs in rather than ride a causeway), and starts no higher than the
  lake it leaves; dry hollows under sea level are fitted as at it.  Where floors
  overlap (at a junction), the stem's bed holds.
- **Valley.**  Texels within ``floor_texels`` of a bed take its height; over
  ``wall_texels`` beyond, they blend back to their own (smoothstep).  Water
  texels are never edited, and no dry texel is cut below the water beside it
  or below sea level.
- **Conservation.**  Each parent texel's mean is restored by raising (or
  lowering) its ground outside the valley, weighted by how far out of it each
  texel lies, read smoothly across parent texels and closed by an exact
  per-parent-texel step, as :func:`region.upsample.conserve_smooth` does.  A
  parent texel whose valleys would need its ground moved by more than
  ``absorb_relief`` × its own relief range on average is cut less.

The tier below conserves against the carved means, so it inherits the valleys.
Re-carving starts from the uncarved elevation (:attr:`RegionFields.uncarved`).
"""
from __future__ import annotations

from bisect import insort
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np
from scipy.spatial import KDTree

from core.params import param_field as _f, paramclass
from region.sampling import anchor_means, stencil_read
from region.source import Located
from region.upsample import EARTH_RADIUS_KM, SEA_LEVEL_KM, _neighbour_min, _smooth_setup

if TYPE_CHECKING:
    from region.rivers import RiverPath
    from region.source import ParentSource
    from region.upsample import RegionFields

# Dry ground is kept at least this far above the water beside it (km).
_ABOVE_WATER_KM = 0.001
# A line's end within this many texels of an earlier line's bed joins it (a junction is on
# its stem's line or, for a region's streams, the raster node beside it).
_JOIN_TEXELS = 1.0
# A line starting within this many texels of an earlier one starts on it (a distributary).
_START_TEXELS = 0.5
# Bed points looked at per texel (for the earliest line's floor it lies on).
_NEAREST = 8
# How many texels a water surface spreads onto the shore as a floor for the cut.
_SHORE_TEXELS = 2
# A parent texel with less than this share of its weight outside the valleys isn't carved
# (what's left of its residual goes on all its ground).
_MIN_COVER = 0.1
# Smooth rounds easing the cut off where a parent texel can't absorb it.
_LIMIT_ROUNDS = 4
# Smooth absorption passes, stopping once the worst residual is _TOLERANCE of its start.
_MAX_PASSES = 24
_TOLERANCE = 0.01
# Over-relaxation per smooth pass (each alone takes back only part of a residual).
_RELAXATION = 1.6
# Rounds of holding ground the absorption pushed under water at its floor.
_MAX_HOLDS = 4
# An anchor mean further off than this (km) after the holds is restored on all its ground.
_EXACT_KM = 1e-6


@paramclass
class RiverCarveParams:
    """Tuning for the valleys carved along a tier's routed rivers (:mod:`region.carve`)."""

    carve: bool = _f(True, label="Carve river valleys", min=0, max=1)
    # The flat valley floor's half-width, in this tier's texels: wide enough that the
    # texel nearest any point of the drawn (smoothed) line is on it.
    floor_texels: float = _f(0.75, label="Valley floor half-width (texels)", min=0.0, max=6.0, step=0.25)
    # Width of the walls blending the floor back into the ground, in texels.  Wider
    # valleys cut more than a parent texel can absorb beside them on rugged ground.
    wall_texels: float = _f(1.0, label="Valley wall width (texels)", min=0.5, max=12.0, step=0.25)
    # 0: every rise is cut to the lowest ground upstream; 0.5: the median fit, cutting
    # rises and filling hollows alike, which leaves the least for the ground beside to absorb.
    quantile: float = _f(0.5, label="Bed fit quantile (cut ↔ fill)", min=0.0, max=0.5, step=0.05)
    step_texels: float = _f(0.5, label="Bed sample spacing (texels)", min=0.1, max=2.0, step=0.1)
    # The most a parent texel's ground outside the valleys is raised (or lowered) on average,
    # as a share of that texel's own relief range; a valley needing more is cut less there.
    absorb_relief: float = _f(1.0, label="Max absorption (× parent texel relief)", min=0.0, max=5.0, step=0.1)


# ── the bed ──────────────────────────────────────────────────────────────────


def _quantile_sorted(values: list[float], q: float) -> float:
    """``np.quantile`` (linear) of an already sorted list."""
    pos = q * (len(values) - 1)
    i = int(pos)
    return values[i] if i + 1 >= len(values) else values[i] + (pos - i) * (values[i + 1] - values[i])


def decreasing_fit(z: np.ndarray, quantile: float) -> np.ndarray:
    """Never-rising fit to ``z``: isotonic regression at ``quantile`` (pool-adjacent-violators).

    Each pooled block takes its members' ``quantile``: 0 is the running minimum.
    """
    blocks: list[list[float]] = []      # each block's members, sorted
    levels: list[float] = []
    for x in np.asarray(z, dtype=np.float64).tolist():
        block, level = [x], x
        while levels and levels[-1] < level:
            prev = blocks.pop()
            levels.pop()
            if len(prev) < len(block):
                prev, block = block, prev
            for v in block:
                insort(prev, v)
            block, level = prev, _quantile_sorted(prev, quantile)
        blocks.append(block)
        levels.append(level)
    return np.repeat(np.asarray(levels), [len(b) for b in blocks])


def _densify(positions: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray]:
    """Points every ``step`` (chord) along a polyline, and the segment each lies on."""
    pts = np.asarray(positions, dtype=np.float64)
    seg_len = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    k = np.maximum(np.ceil(seg_len / max(step, 1e-12)).astype(np.int64), 1)
    seg = np.repeat(np.arange(len(k)), k)
    frac = np.arange(len(seg)) - np.repeat(np.cumsum(k) - k, k)
    out = pts[seg] + (frac / np.repeat(k, k))[:, None] * (pts[seg + 1] - pts[seg])
    out = np.vstack([out, pts[-1:]])
    seg = np.append(seg, len(k) - 1)
    return out / np.linalg.norm(out, axis=1, keepdims=True), seg


def line_beds(
    fields: "RegionFields", elevation: np.ndarray, surface: np.ndarray, water: np.ndarray,
    paths: Sequence["RiverPath"], params: RiverCarveParams,
    half_widths: Optional[Sequence[np.ndarray]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(points (K, 3), beds (K,), floors (K,), lines (K,))`` along the dry stretches of
    ``paths`` (in order: stems first); ``floors`` is each point's floor half-width (chord),
    ``lines`` its line's order.

    ``elevation``, ``surface`` (water surface, inf on land) and ``water`` are ``(M, T, T)``;
    ``half_widths`` per path, per segment (km), widens a floor past ``floor_texels``.
    """
    spacing = fields.grid.texel_spacing_rad(fields.tex_size)
    step = float(params.step_texels) * spacing
    keep_line = [len(p[0]) >= 2 for p in paths]
    lines = [p for p, k in zip(paths, keep_line) if k]
    halves = [h for h, k in zip(half_widths, keep_line) if k] if half_widths is not None else None
    if not lines:
        return np.zeros((0, 3)), np.zeros(0), np.zeros(0), np.zeros(0, dtype=np.int64)
    # Every line's points at once: one texel lookup, one tree for the joins.
    parts = [_densify(p[0], step) for p in lines]
    floors = np.full(sum(len(q) for q, _ in parts), float(params.floor_texels) * spacing)
    if halves is not None:
        half = [np.asarray(h, dtype=np.float64).reshape(-1)[seg] for h, (_, seg) in zip(halves, parts)]
        floors = np.maximum(floors, np.concatenate(half) / EARTH_RADIUS_KM)
    pts = np.concatenate([q for q, _ in parts])
    line_of = np.repeat(np.arange(len(parts)), [len(q) for q, _ in parts])
    first = np.concatenate([[0], np.cumsum([len(q) for q, _ in parts])])
    s, t, valid = fields.geometry.ecef_to_local(pts)
    hexes, rows, cols, covered = fields.grid.nearest_texels(s, t, fields.tex_size)
    at = (hexes, rows, cols)
    ok_all = valid & covered
    z_all, wet_all, surf_all = elevation[at], water[at], surface[at]
    bed = np.full(len(pts), np.nan)
    tree = KDTree(pts)

    def earlier_bed(k: int, p: np.ndarray, reach: float) -> Optional[float]:
        near = np.asarray(tree.query_ball_point(p, reach), dtype=np.int64)
        near = near[(line_of[near] < k) & np.isfinite(bed[near])]
        if near.size == 0:
            return None
        return float(bed[near[np.argmin(np.linalg.norm(pts[near] - p[None, :], axis=1))]])

    for k, ((_, seg), (_, _, in_water)) in enumerate(zip(parts, lines)):
        a, b = first[k], first[k + 1]
        n = b - a
        ok, wet_texel, z, surf = ok_all[a:b], wet_all[a:b], z_all[a:b], surf_all[a:b]
        hidden = np.asarray(in_water, dtype=bool).reshape(-1)
        wet = (hidden[seg] if hidden.size == len(lines[k][0]) - 1 else np.zeros(n, dtype=bool)) | wet_texel
        idx = np.flatnonzero(ok & ~wet)
        for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1) if idx.size else []:
            i, j = int(run[0]), int(run[-1])
            # Land is never cut below sea level (carve_rivers), so the bed isn't fitted below it either:
            # a dry hollow under sea level along the way counts as at it.
            fit = decreasing_fit(np.maximum(z[i:j + 1], SEA_LEVEL_KM + _ABOVE_WATER_KM), float(params.quantile))
            floor, ceiling = SEA_LEVEL_KM + _ABOVE_WATER_KM, np.inf
            if j + 1 < n and ok[j + 1] and wet_texel[j + 1]:
                floor = float(surf[j + 1]) + _ABOVE_WATER_KM
            elif j == n - 1:
                joined = earlier_bed(k, pts[a + j], _JOIN_TEXELS * spacing)
                if joined is not None:
                    floor = max(floor, joined)
            if i > 0 and ok[i - 1] and wet_texel[i - 1]:
                ceiling = float(surf[i - 1]) + _ABOVE_WATER_KM
            elif i == 0:
                parent = earlier_bed(k, pts[a], _START_TEXELS * spacing)
                if parent is not None:
                    ceiling = parent
            # Never above the stretch's own lowest ground: a lake (or stem) the routing put
            # higher than ground the stretch runs over doesn't lift it all (it climbs in).
            floor = min(floor, float(z[i:j + 1].min()))
            fit = np.maximum(fit, floor)
            if np.isfinite(ceiling):
                fit = np.minimum(fit, max(ceiling, floor))
            bed[a + i:a + j + 1] = fit
    keep = np.isfinite(bed)
    return pts[keep], bed[keep], floors[keep], line_of[keep]


# ── conservation ─────────────────────────────────────────────────────────────


def restore_means(
    values: np.ndarray, mask: np.ndarray, ground: np.ndarray, source: "ParentSource", loc: Located,
    weights: np.ndarray,
) -> np.ndarray:
    """``(P,)`` ``values`` with each anchor's weighted mean back at the parent's, changed only
    in proportion to ``mask`` (``(P,)`` in [0, 1]).

    Smooth passes (the per-anchor residual over its masked share, read bilinearly)
    then an exact per-anchor step.  An anchor with under ``_MIN_COVER`` of its
    weight masked absorbs on all its ``ground`` (``(P,)`` bool) instead (one with
    no ground was never edited).
    """
    anchors, idx, w, n, plan = _smooth_setup(source, loc)
    tgt = np.zeros(n)
    tgt[anchors] = source.anchor_values("elevation", loc)[:, 0]
    weights = np.asarray(weights, dtype=np.float64)
    counts = np.bincount(anchors, weights=weights, minlength=n)
    live = counts > 0.0
    safe = np.where(live, counts, 1.0)
    m = np.asarray(mask, dtype=np.float64)
    cover = np.bincount(anchors, weights=weights * m, minlength=n) / safe
    thin = live & (cover < _MIN_COVER)
    # The smooth step is read(residual) / read(cover): smooth, and no blow-up beside a thin anchor,
    # whose residual waits for the exact step.
    # (An anchor with no weight here reads as covered, so beside one nothing is amplified.)
    read_cover = np.maximum(stencil_read(w, idx, np.where(live, cover, 1.0)[:, None])[:, 0], _MIN_COVER)
    m_end, cover_end = m, cover
    if thin.any():
        m_end = np.where(thin[anchors], np.maximum(m, ground.astype(np.float64)), m)
        cover_end = np.bincount(anchors, weights=weights * m_end, minlength=n) / safe
    inv = np.where(live & (cover_end > 0.0), 1.0 / np.where(cover_end > 0.0, cover_end, 1.0), 0.0)

    def residual(v: np.ndarray) -> np.ndarray:
        means, _ = anchor_means(v[:, None], anchors, weights, n, counts=counts, plan=plan)
        return np.where(live, tgt - means[:, 0], 0.0)

    v = np.asarray(values, dtype=np.float64).copy()
    kept, first, best = v, None, np.inf
    for _ in range(_MAX_PASSES):
        r = residual(v)
        worst = float(np.abs(r[~thin]).max(initial=0.0))      # thin anchors wait for the exact step
        if worst >= best:
            v = kept
            break
        kept, best = v, worst
        first = worst if first is None else first
        if worst <= _TOLERANCE * first:
            break
        v = v + _RELAXATION * m * stencil_read(w, idx, np.where(thin, 0.0, r)[:, None])[:, 0] / read_cover
    return v + m_end * (residual(v) * inv)[anchors]


def limit_to_capacity(
    before: np.ndarray, after: np.ndarray, mask: np.ndarray, ground: np.ndarray, source: "ParentSource",
    loc: Located, weights: np.ndarray, factor: float,
) -> np.ndarray:
    """``after`` with each anchor's edit scaled back so its masked ground (``mask``-weighted
    share ``cover``) would move by at most ``factor`` × the anchor's relief range on average.

    The per-anchor scale is read smoothly, so the cut eases off rather than steps.
    """
    anchors, idx, w, n, _ = _smooth_setup(source, loc)
    weights = np.asarray(weights, dtype=np.float64)
    counts = np.bincount(anchors, weights=weights, minlength=n)
    live = counts > 0.0
    safe = np.where(live, counts, 1.0)
    cover = np.bincount(anchors, weights=weights * mask, minlength=n) / safe
    hi, lo = np.full(n, -np.inf), np.full(n, np.inf)
    np.maximum.at(hi, anchors[ground], before[ground])
    np.minimum.at(lo, anchors[ground], before[ground])
    span = np.where(hi > lo, hi - lo, 0.0)
    # A parent texel the valleys all but fill (cut by a map's edge, say) can't take any.
    cap = np.where(cover >= _MIN_COVER, float(factor) * cover * span, 0.0)
    delta = after - before
    eased = np.ones(len(after))
    for _ in range(_LIMIT_ROUNDS):
        need = np.abs(np.bincount(anchors, weights=weights * delta * eased, minlength=n)) / safe
        scale = np.where(live & (need > cap), cap / np.maximum(need, 1e-12), 1.0)
        if np.all(scale >= 1.0):
            break
        eased *= np.clip(stencil_read(w, idx, scale[:, None])[:, 0], 0.0, 1.0)
    return before + delta * eased


def anchor_residual(values: np.ndarray, source: "ParentSource", loc: Located, weights: np.ndarray) -> np.ndarray:
    """``(n,)`` parent value minus weighted mean, per compact anchor (0 where unweighted)."""
    anchors, _, _, n, plan = _smooth_setup(source, loc)
    weights = np.asarray(weights, dtype=np.float64)
    tgt = np.zeros(n)
    tgt[anchors] = source.anchor_values("elevation", loc)[:, 0]
    means, counts = anchor_means(np.asarray(values, dtype=np.float64)[:, None], anchors, weights, n, plan=plan)
    return np.where(counts > 0.0, tgt - means[:, 0], 0.0)


# ── entry point ──────────────────────────────────────────────────────────────


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def carve_rivers(
    fields: "RegionFields", source: "ParentSource", paths: Sequence["RiverPath"],
    params: Optional[RiverCarveParams] = None, weights: Optional[np.ndarray] = None,
    half_widths: Optional[Sequence[np.ndarray]] = None,
) -> None:
    """Carve ``paths`` (stems before the lines that join them) into ``fields``' elevation, in place.

    ``source`` is the tier ``fields`` conserves against and ``weights`` its
    ``(M, T, T)`` conservation weights (default the domain).  ``half_widths``
    (per path, per segment, km; e.g. half the channel's width) widens the floor
    where it is wider than ``floor_texels``.  Updates ``filled_height`` off the
    lakes to match.  See the module docstring.
    """
    params = params or RiverCarveParams()
    base = fields.uncarved if fields.uncarved is not None else fields.textures["elevation"]
    fields.uncarved = base
    fields.textures["elevation"] = base.copy()
    if not params.carve or not paths:
        _sync_filled(fields)
        return
    e = base[..., 0].astype(np.float64)
    shape = e.shape
    ocean = fields.textures["ocean_mask"][..., 0] > 0.5 if "ocean_mask" in fields.textures else np.zeros(shape, bool)
    lake = fields.textures["lake_mask"][..., 0] > 0.5 if "lake_mask" in fields.textures else np.zeros(shape, bool)
    water = ocean | lake
    surface = np.where(ocean, SEA_LEVEL_KM, np.inf)
    if lake.any() and "filled_height" in fields.textures:
        surface = np.where(lake, fields.textures["filled_height"][..., 0].astype(np.float64), surface)

    points, beds, floors, line = line_beds(fields, e, surface, water, paths, params, half_widths)
    if not len(points):
        _sync_filled(fields)
        return
    spacing = fields.grid.texel_spacing_rad(fields.tex_size)
    wall_r = max(float(params.wall_texels), 1e-6) * spacing
    flat_pos = np.asarray(fields.positions, dtype=np.float64).reshape(-1, 3)
    found = KDTree(points).query(flat_pos, k=_NEAREST, distance_upper_bound=float(floors.max()) + wall_r,
                                 workers=-1)
    dk, kk = np.asarray(found[0], dtype=np.float64), np.asarray(found[1], dtype=np.int64)
    d, k = dk[:, 0].copy(), kk[:, 0].copy()
    # On several floors (near a junction), the stem's: a tributary's bed neither lifts nor sinks it.
    safe_k = np.minimum(kk, len(beds) - 1)
    on_floor = np.isfinite(dk) & (dk <= floors[safe_k])
    if on_floor[:, 1:].any():
        stem = np.argmin(np.where(on_floor, line[safe_k], np.iinfo(np.int64).max), axis=1)
        rows = np.flatnonzero(on_floor.any(axis=1))
        d[rows], k[rows] = dk[rows, stem[rows]], kk[rows, stem[rows]]
    near = np.isfinite(d)
    blend = np.ones(len(flat_pos))                     # 0 on a floor, 1 untouched
    blend[near] = _smoothstep((d[near] - floors[k[near]]) / wall_r)
    ground = ~water.reshape(-1)
    edit = near & ground
    flat_e = e.reshape(-1)
    carved = flat_e.copy()
    bed = beds[np.minimum(k, len(beds) - 1)]
    carved[edit] = bed[edit] + (flat_e[edit] - bed[edit]) * blend[edit]

    # Never below the water beside it (surfaces spread a little onto the shore), nor below sea level.
    shore = np.where(water, surface, SEA_LEVEL_KM)
    for _ in range(_SHORE_TEXELS):
        shore = np.maximum(shore, -_neighbour_min(-shore))
    shore = (shore + _ABOVE_WATER_KM).reshape(-1)
    carved = np.where(ground, np.maximum(carved, np.minimum(flat_e, shore)), flat_e)

    w = (fields.domain if weights is None else weights).reshape(-1)
    if fields.located is None:
        # The build's own anchors: re-locating the float32 positions moves a few edge texels.
        stored = fields.parent_anchors if fields.parent_anchors is not None else fields.anchors
        fields.located = Located(np.asarray(stored, dtype=np.int64).reshape(-1), source.locate(flat_pos).data)
    loc = fields.located
    mask = np.where(ground, blend, 0.0)
    carved = limit_to_capacity(flat_e, carved, mask, ground, source, loc, w, float(params.absorb_relief))
    out = restore_means(carved, mask, ground, source, loc, w)
    # The absorption can lower ground too (where beds were mostly filled): keep it off the water.
    held = np.zeros(len(out), dtype=bool)
    for _ in range(_MAX_HOLDS):
        low = ground & ~held & (out < shore) & (out < carved)
        if not low.any():
            break
        held |= low
        out = restore_means(np.where(held, np.maximum(out, np.minimum(carved, shore)), out),
                            np.where(ground & ~held, blend, 0.0), ground & ~held, source, loc, w)
    # An anchor whose ground was all held takes the rest on it anyway: the means stay exact.
    off = np.abs(anchor_residual(out, source, loc, w)) > _EXACT_KM
    if off.any():
        anchors = _smooth_setup(source, loc)[0]
        out = restore_means(out, np.where(off[anchors] & ground, 1.0, 0.0), ground, source, loc, w)
    fields.textures["elevation"] = out.reshape(shape)[..., None].astype(np.float32)
    _sync_filled(fields)


def _sync_filled(fields: "RegionFields") -> None:
    """``filled_height`` is the ground off the lakes."""
    filled = fields.textures.get("filled_height")
    if filled is None:
        return
    lake = fields.textures["lake_mask"][..., 0] > 0.5 if "lake_mask" in fields.textures else None
    elev = fields.textures["elevation"]
    fields.textures["filled_height"] = (elev if lake is None else np.where(lake[..., None], filled, elev)).astype(
        np.float32)
