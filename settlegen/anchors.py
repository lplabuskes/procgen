"""A settlement's anchor candidates: where its citadel, market, waterfront and sacred
site could go, scored, not chosen (the layout picks, plan phase 6).

All inside (or, for the waterfront, on the edge of) the footprint:

- **citadel**: high ground — local maxima of the relief (smoothed over
  ``smoothing_m``) scored by height above their surroundings (the relief less
  its ``surround_m`` blur, against the site's highest) and the steepness around
  them (a defensible knoll); on ground with too few knolls, its highest ground;
- **market**: along the roads, scored by the trade flowing past (every road
  within ``market_reach_m``, summed), nearness to the centre and flat ground;
  without roads, the flat ground nearest the centre;
- **waterfront**: points along the footprint's edge within ``waterfront_reach_m``
  of water, scored by the water (sea and lakes over rivers, wide rivers over
  narrow), flat ground and the landings near them (a landing is itself one);
- **sacred**: central high ground — the citadel's measure of height and
  nearness to the centre, half each.

Each kind keeps its ``per_kind`` best, at least ``separation_m`` apart, scores
in [0, 1].  Qt-free.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import shapely
from scipy.ndimage import gaussian_filter, maximum_filter
from shapely.geometry import LineString, MultiPolygon

from core.params import param_field as _f, paramclass
from region.upsample import RegionFields
from settlegen.footprint import site_slope
from settlegen.frame import Frame
from settlegen.roads import SiteRoads
from settlegen.water import SiteWater, channel_polygons

AnchorKind = Literal["citadel", "market", "waterfront", "sacred"]
ANCHOR_KINDS: tuple[AnchorKind, ...] = ("citadel", "market", "waterfront", "sacred")


@paramclass
class SettlementAnchorParams:
    """Tuning for the anchor candidates."""

    per_kind: int = _f(6, label="Candidates per kind", min=1, max=30)
    separation_m: float = _f(150.0, label="Candidate separation (m)", min=10.0, max=1000.0, step=10.0)
    smoothing_m: float = _f(30.0, label="Relief smoothing (m)", min=0.0, max=200.0, step=5.0)
    surround_m: float = _f(300.0, label="High-ground surround (m)", min=50.0, max=2000.0, step=25.0)
    sample_m: float = _f(30.0, label="Road / shore sample spacing (m)", min=5.0, max=200.0, step=5.0)
    market_reach_m: float = _f(75.0, label="Market road reach (m)", min=10.0, max=500.0, step=5.0)
    waterfront_reach_m: float = _f(25.0, label="Waterfront reach (m)", min=1.0, max=200.0, step=1.0)
    # Slope (km/km) at which ground stops counting as flat.
    flat_slope: float = _f(0.1, label="Flat-ground slope (km/km)", min=0.01, max=1.0, step=0.01)


@dataclass
class Anchor:
    """One candidate."""

    kind: AnchorKind
    xy: np.ndarray            # (2,) frame metres
    score: float              # [0, 1], within its kind
    # What it scored on, by name (height_m, flow, water, …), for the layout and for checks.
    detail: dict = field(default_factory=dict)


@dataclass
class AnchorCandidates:
    """The candidates of each kind, best first."""

    by_kind: dict[str, list[Anchor]]

    def __getitem__(self, kind: str) -> list[Anchor]:
        return self.by_kind.get(kind, [])


def _pick(kind: AnchorKind, xy: np.ndarray, score: np.ndarray, details: list[dict],
          params: SettlementAnchorParams) -> list[Anchor]:
    """The best ``per_kind``, at least ``separation_m`` apart, scores scaled to [0, 1]."""
    if len(xy) == 0:
        return []
    top = float(np.max(score))
    norm = score / top if top > 0 else np.ones_like(score)
    out: list[Anchor] = []
    for k in np.argsort(-score, kind="stable"):
        if len(out) >= int(params.per_kind):
            break
        if all(np.linalg.norm(a.xy - xy[k]) >= float(params.separation_m) for a in out):
            out.append(Anchor(kind=kind, xy=np.asarray(xy[k], dtype=np.float64), score=float(norm[k]),
                              detail=details[k]))
    return out


def _along(lines, spacing: float) -> np.ndarray:  # type: ignore[no-untyped-def]
    """Points every ``spacing`` along (multi)line geometry."""
    pts = []
    for line in getattr(lines, "geoms", [lines]):
        if line.is_empty or line.length <= 0:
            continue
        n = max(int(line.length // spacing), 1)
        pts.extend((p.x, p.y) for p in shapely.line_interpolate_point(line, np.linspace(0.0, line.length, n + 1)))
    return np.asarray(pts, dtype=np.float64).reshape(-1, 2)


def build_anchor_candidates(
    fields: RegionFields, frame: Frame, water: SiteWater, roads: SiteRoads, footprint: MultiPolygon,
    params: Optional[SettlementAnchorParams] = None,
) -> AnchorCandidates:
    """The candidates.  See the module docstring."""
    params = params or SettlementAnchorParams()
    by_kind: dict[str, list[Anchor]] = {k: [] for k in ANCHOR_KINDS}
    if footprint.is_empty:
        return AnchorCandidates(by_kind)
    t = fields.tex_size
    texel_m = frame.side_m / t
    xy = frame.texel_xy(t)
    inside = shapely.contains_xy(footprint, xy[..., 0], xy[..., 1])
    if not inside.any():
        return AnchorCandidates(by_kind)
    elev = fields.textures["elevation"][0, ..., 0].astype(np.float64) * 1000.0
    relief = gaussian_filter(elev, float(params.smoothing_m) / texel_m) if params.smoothing_m > 0 else elev
    height = relief - gaussian_filter(elev, float(params.surround_m) / texel_m)      # above its surroundings (m)
    slope = site_slope(fields, frame)
    flat = 1.0 - np.clip(slope / float(params.flat_slope), 0.0, 1.0)
    steep_round = gaussian_filter(slope, 2.0 * float(params.smoothing_m) / texel_m)
    centre = footprint.centroid
    reach = max(float(np.sqrt(footprint.area / np.pi)), texel_m)
    near_centre = np.clip(1.0 - np.hypot(xy[..., 0] - centre.x, xy[..., 1] - centre.y) / (2.0 * reach), 0.0, 1.0)

    def at(points: np.ndarray, grid: np.ndarray) -> np.ndarray:
        cols = np.clip(((points[:, 0] + frame.half) / texel_m).astype(int), 0, t - 1)
        rows = np.clip(((frame.half - points[:, 1]) / texel_m).astype(int), 0, t - 1)
        return grid[rows, cols]

    # ── high ground: the citadel's and the sacred site's ──
    window = max(int(round(float(params.separation_m) / texel_m)), 3)
    peaks = inside & (relief >= maximum_filter(relief, size=window)) & (height > 0.0)
    if int(peaks.sum()) < int(params.per_kind):
        # Few knolls (a plain, a slope): the highest ground it has stands in.
        flat_idx = np.flatnonzero(inside.reshape(-1))
        best = flat_idx[np.argsort(-height.reshape(-1)[flat_idx])[:max(int(params.per_kind) * window ** 2, 1)]]
        peaks.reshape(-1)[best] = True
    rows, cols = np.nonzero(peaks)
    hp = xy[rows, cols]
    h = height[rows, cols]
    h_rel = np.clip(h, 0.0, None) / max(float(h.max()), 1e-9) if h.size else h
    steep = np.clip(steep_round[rows, cols] / float(params.flat_slope), 0.0, 1.0)
    details = [{"height_m": float(a), "slope": float(b)} for a, b in zip(h, slope[rows, cols])]
    by_kind["citadel"] = _pick("citadel", hp, 0.7 * h_rel + 0.3 * steep, details, params)
    sacred_xy = np.concatenate([hp, [[centre.x, centre.y]]]) if inside.any() else hp
    sacred_h = np.concatenate([h_rel, [0.0]])
    sacred_details = details + [{"height_m": 0.0, "slope": float(at(np.array([[centre.x, centre.y]]), slope)[0])}]
    by_kind["sacred"] = _pick("sacred", sacred_xy, 0.5 * sacred_h + 0.5 * at(sacred_xy, near_centre),
                              sacred_details, params)

    # ── market: where the trade passes, near the middle, on the flat ──
    spacing = float(params.sample_m)
    road_lines = [LineString(r.xy) for r in roads.lines if len(r.xy) >= 2]
    pts = _along(shapely.union_all(road_lines).intersection(footprint), spacing) if road_lines else np.zeros((0, 2))
    if len(pts):
        flow = np.zeros(len(pts))
        for road in roads.lines:
            seg = np.asarray(shapely.linestrings(np.stack([road.xy[:-1], road.xy[1:]], axis=1)))
            d = shapely.distance(np.asarray(shapely.points(pts))[:, None], seg[None, :])     # (P, E)
            k = np.argmin(d, axis=1)
            near = d[np.arange(len(pts)), k] <= float(params.market_reach_m)
            flow += np.where(near, road.flows[k], 0.0)
        conv = flow / max(float(flow.max()), 1e-9)
        details = [{"flow": float(f)} for f in flow]
        score = 0.5 * conv + 0.3 * at(pts, near_centre) + 0.2 * at(pts, flat)
    else:
        rows, cols = np.nonzero(inside)
        pts = xy[rows, cols]
        details = [{"flow": 0.0} for _ in range(len(pts))]
        score = 0.6 * near_centre[rows, cols] + 0.4 * flat[rows, cols]
    by_kind["market"] = _pick("market", pts, score, details, params)

    # ── waterfront: the footprint's edge along the water ──
    channels = channel_polygons(water.lines, frame.box())
    wet = shapely.union_all([water.sea, water.lakes, channels])
    if not wet.is_empty:
        edge = footprint.boundary.intersection(wet.buffer(float(params.waterfront_reach_m)))
        pts = _along(edge, spacing)
        if len(roads.landings):
            pts = np.concatenate([pts, np.array([la.xy for la in roads.landings])])
        if len(pts):
            p = shapely.points(pts)
            reach = float(params.waterfront_reach_m) + 1.0
            on_sea = shapely.distance(water.sea, p) <= reach if not water.sea.is_empty else np.zeros(len(pts), bool)
            on_lake = shapely.distance(water.lakes, p) <= reach if not water.lakes.is_empty \
                else np.zeros(len(pts), bool)
            width = np.zeros(len(pts))
            for line in water.lines:
                seg = np.asarray(shapely.linestrings(np.stack([line.xy[:-1], line.xy[1:]], axis=1)))
                d = shapely.distance(np.asarray(p)[:, None], seg[None, :])
                k = np.argmin(d, axis=1)
                close = d[np.arange(len(pts)), k] <= 0.5 * line.width_m[k] + reach
                width = np.where(close, np.maximum(width, line.width_m[k]), width)
            quality = np.where(on_sea, 1.0, np.where(on_lake, 0.8, np.clip(width / 100.0, 0.1, 0.7)))
            landed = np.zeros(len(pts))
            total = sum(la.flow for la in roads.landings) or 1.0
            for la in roads.landings:
                landed += la.flow / total * (np.linalg.norm(pts - la.xy, axis=1) <= float(params.separation_m))
            kinds = np.where(on_sea, "sea", np.where(on_lake, "lake", np.where(width > 0, "river", "other")))
            details = [{"water": str(k), "width_m": float(w), "landing": float(lv)}
                       for k, w, lv in zip(kinds, width, landed)]
            score = 0.4 * quality + 0.2 * at(pts, flat) + 0.4 * landed
            keep = kinds != "other"
            by_kind["waterfront"] = _pick("waterfront", pts[keep], score[keep],
                                          [d for d, k in zip(details, keep) if k], params)
    return AnchorCandidates(by_kind)
