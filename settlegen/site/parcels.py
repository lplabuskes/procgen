"""The hinterland's fields and gardens cut into parcels, shaped and coloured by the crop.

Each farmer's fields (:class:`settlegen.site.hinterland.Hinterland`: the settlement's
own and each neighbour's) are split into *furlongs* — Voronoi cells of a jittered
grid ``furlong_m`` apart — and each furlong into parallel strips ``strip_m`` wide
(jittered), cut across every ``length_m`` where the crop has short parcels.  A
furlong's strips run:

- downslope on open fields (drainage: the ridge-and-furrow way), where there is a slope;
- along the contour for rice paddies and coffee terraces;
- on the flat, along a grain that drifts across the frame (over ``_GRAIN_M``),
  with some furlongs (``_ACROSS``) turned square to it.

The crop is the frame's ``dominant_crop`` (:mod:`worldgen.resources.crops`); a
city's own ring is market gardens whatever it grows.  Each parcel takes one of
the crop's *states* (sown, ripening, fallow, …): on open fields by which of its
farmer's three great fields it lies in (three-field rotation: sectors round the
farmer), on the rest by furlong — save a ``mix`` share in a state of their own;
then a brightness jitter, quantised
to ``_SHADES`` levels so the viewer fills a handful of paths.

Only ``arable_share`` of a farmer's fields is cropland (``ha_per_farmer`` counts
meadow and pasture too): its outer and steeper furlongs, with some noise, are
meadow and pasture instead, merged into unstriped commons.  A city's market
gardens hold their own pasture.

Display-only, Qt-free, deterministic.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import shapely
from scipy.ndimage import gaussian_filter
from shapely.geometry import MultiPolygon, Polygon

from core.params import param_field as _f, paramclass
from region.upsample import RegionFields
from settlegen.frame import Frame
from settlegen.site.hinterland import Hinterland
from worldgen.resources.crops import CROP_NAMES

# Brightness levels each state's colour is quantised to.
_SHADES = 5
# Slope (km/km) below which a furlong's strips run any which way.
_FLAT = 0.01
# How far the grain on the flat drifts over (m), and the share of furlongs turned square to it.
_GRAIN_M = 800.0
_GRAIN_TEXELS = 64
_ACROSS = 0.35
# Which furlongs go to pasture: the farthest from the farmer (0–1 over its fields), plus
# this much for steep ground (saturating at _STEEP km/km), plus noise smooth over _PASTURE_NOISE_M.
_PASTURE_SLOPE = 0.5
_STEEP = 0.15
_PASTURE_NOISE = 0.25
_PASTURE_NOISE_M = 500.0
# Parcel vertices snap to this grid (m).
_GRID_M = 0.01
# Parcels smaller than this share of a full strip cell are slivers.
_MIN_SHARE = 0.05


@dataclass(frozen=True)
class CropStyle:
    """How one crop's fields are cut and coloured."""

    furlong_m: float
    strip_m: float
    length_m: Optional[float]                   # parcel length along the strip (None: the furlong's)
    orient: Literal["downslope", "contour", "any"]
    states: tuple[tuple[float, float, float], ...]
    weights: tuple[float, ...]
    state_by: Literal["sector", "furlong"]
    mix: float                                  # share of parcels in a state of their own
    jitter: float                               # ± brightness per parcel
    edge: tuple[float, float, float, float]     # RGBA of the balks, bunds, hedges between parcels


# Three-field rotation: winter corn, spring corn, fallow.
_OPEN_FIELD = dict(furlong_m=220.0, strip_m=18.0, length_m=None, orient="downslope",
                   weights=(1.0, 1.0, 1.0), state_by="sector", mix=0.05, jitter=0.07, edge=(0.45, 0.40, 0.25, 0.35))
CROP_STYLES: dict[str, CropStyle] = {
    "Wheat": CropStyle(states=((0.88, 0.75, 0.36), (0.74, 0.76, 0.42), (0.60, 0.64, 0.42)),
                       **_OPEN_FIELD),  # type: ignore[arg-type]
    "Barley": CropStyle(states=((0.86, 0.80, 0.52), (0.70, 0.73, 0.44), (0.60, 0.62, 0.44)),
                        **_OPEN_FIELD),  # type: ignore[arg-type]
    # No crop does well: rough grazing between a few small strips.
    "None": CropStyle(furlong_m=200.0, strip_m=40.0, length_m=None, orient="any",
                      states=((0.66, 0.70, 0.46), (0.60, 0.64, 0.42), (0.78, 0.72, 0.48)),
                      weights=(2.0, 1.0, 1.0), state_by="furlong", mix=0.15, jitter=0.06,
                      edge=(0.40, 0.42, 0.28, 0.35)),
    # Paddies on the contour, between bunds: flooded, transplanted, ripening.
    "Rice": CropStyle(furlong_m=160.0, strip_m=28.0, length_m=55.0, orient="contour",
                      states=((0.56, 0.68, 0.62), (0.52, 0.70, 0.42), (0.72, 0.74, 0.44)),
                      weights=(1.0, 1.5, 1.0), state_by="furlong", mix=0.1, jitter=0.05, edge=(0.84, 0.80, 0.66, 0.65)),
    "Maize": CropStyle(furlong_m=180.0, strip_m=45.0, length_m=None, orient="downslope",
                       states=((0.50, 0.64, 0.30), (0.68, 0.67, 0.36), (0.68, 0.58, 0.40)),
                       weights=(2.0, 1.0, 0.6), state_by="furlong", mix=0.25, jitter=0.06,
                       edge=(0.38, 0.36, 0.22, 0.35)),
    "Millet": CropStyle(furlong_m=160.0, strip_m=40.0, length_m=None, orient="any",
                        states=((0.80, 0.70, 0.46), (0.64, 0.66, 0.38), (0.74, 0.62, 0.46)),
                        weights=(1.5, 1.0, 1.0), state_by="furlong", mix=0.25, jitter=0.06,
                        edge=(0.45, 0.38, 0.26, 0.35)),
    "Cotton": CropStyle(furlong_m=180.0, strip_m=50.0, length_m=None, orient="downslope",
                        states=((0.50, 0.62, 0.38), (0.80, 0.80, 0.68), (0.68, 0.60, 0.46)),
                        weights=(1.5, 0.7, 0.6), state_by="furlong", mix=0.25, jitter=0.05,
                        edge=(0.40, 0.38, 0.28, 0.35)),
    # Big blocks of cane, some cut.
    "Sugarcane": CropStyle(furlong_m=300.0, strip_m=100.0, length_m=None, orient="any",
                           states=((0.34, 0.54, 0.26), (0.44, 0.62, 0.30), (0.70, 0.64, 0.46)),
                           weights=(2.0, 1.5, 0.6), state_by="furlong", mix=0.25, jitter=0.05,
                           edge=(0.26, 0.34, 0.18, 0.40)),
    # Terraces on the contour, under shade trees.
    "Coffee": CropStyle(furlong_m=120.0, strip_m=25.0, length_m=80.0, orient="contour",
                        states=((0.28, 0.42, 0.24), (0.34, 0.48, 0.28), (0.24, 0.38, 0.22)),
                        weights=(2.0, 1.0, 1.0), state_by="furlong", mix=0.3, jitter=0.06,
                        edge=(0.55, 0.48, 0.32, 0.45)),
}
# A city's market gardens: orchards, vegetables, dairy pasture, dug earth.
GARDEN_STYLE = CropStyle(furlong_m=90.0, strip_m=25.0, length_m=40.0, orient="any",
                         states=((0.40, 0.54, 0.28), (0.52, 0.64, 0.32), (0.64, 0.72, 0.44), (0.62, 0.52, 0.38)),
                         weights=(1.0, 1.5, 1.0, 0.6), state_by="furlong", mix=0.6, jitter=0.06,
                         edge=(0.38, 0.30, 0.20, 0.45))
# The fields' meadow and pasture: unstriped commons in two tones (hay meadow, rough grazing), hedged.
PASTURE_STYLE = CropStyle(furlong_m=0.0, strip_m=0.0, length_m=None, orient="any",
                          states=((0.60, 0.69, 0.42), (0.67, 0.71, 0.47)), weights=(1.0, 1.0),
                          state_by="furlong", mix=0.0, jitter=0.05, edge=(0.30, 0.40, 0.22, 0.55))


@paramclass
class SettlementParcelParams:
    enabled: bool = _f(True, label="Field parcels", min=0, max=1)
    size_scale: float = _f(1.0, label="Parcel size (×)", min=0.25, max=4.0, step=0.05)
    alpha: float = _f(0.75, label="Parcel opacity", min=0.0, max=1.0, step=0.05)
    # Of ha_per_farmer's 1.5 ha, ~1 is cropland; the rest meadow and pasture (not a city's gardens).
    arable_share: float = _f(0.67, label="Arable share of fields (rest meadow, pasture)", min=0.0, max=1.0,
                             step=0.05)


@dataclass
class FieldParcels:
    """The hinterland's parcels.  ``owner``: 0 the settlement, ``1 + k`` its neighbour ``k``;
    ``shade`` indexes ``palette`` and ``edge`` (state × brightness), ``(P,)`` each."""

    style: str                 # crop name, or "Gardens" for the settlement's own when a city
    polygons: list[Polygon]
    owner: np.ndarray          # (P,) int32
    shade: np.ndarray          # (P,) int32
    palette: np.ndarray        # (C, 4) float32 RGBA fill
    edge: np.ndarray           # (C, 4) float32 RGBA of the balks, bunds, hedges round a parcel
    strip_m: float             # narrowest parcel width (m)
    pasture: np.ndarray        # (P,) bool: meadow or pasture, not cropland


def _crop_name(fields: RegionFields) -> str:
    tex = fields.textures.get("dominant_crop")
    if tex is None:
        return "None"
    idx = int(tex.reshape(-1)[0])
    name = CROP_NAMES[idx] if 0 <= idx < len(CROP_NAMES) else "None"
    return name if name in CROP_STYLES else "None"


def _gradient(fields: RegionFields, frame: Frame, sigma_m: float) -> tuple[np.ndarray, float]:
    """``(T, T, 2)`` elevation gradient (km/km, frame x then y) smoothed over ``sigma_m``; and the texel size (m)."""
    t = fields.tex_size
    texel_m = frame.side_m / t
    elev = gaussian_filter(fields.textures["elevation"][0, ..., 0].astype(np.float64) * 1000.0,
                           max(sigma_m / texel_m, 0.5), mode="nearest")
    d_row, d_col = np.gradient(elev, texel_m)
    # Row 0 is the top (largest y).
    return np.stack([d_col, -d_row], axis=-1), texel_m


def _furlongs(geom: MultiPolygon, spacing: float, rng: np.random.Generator, frame: Frame) -> list[Polygon]:
    """``geom`` cut into Voronoi cells of a jittered grid ``spacing`` apart."""
    x0, y0, x1, y1 = geom.bounds
    xs = np.arange(x0 - 0.5 * spacing, x1 + spacing, spacing)
    ys = np.arange(y0 - 0.5 * spacing, y1 + spacing, spacing)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()]) + rng.uniform(-0.4, 0.4, (gx.size, 2)) * spacing
    if len(pts) < 2:
        return [p for p in geom.geoms]
    cells = shapely.get_parts(shapely.voronoi_polygons(
        shapely.multipoints(pts), extend_to=frame.box().buffer(spacing)))
    cells = cells[shapely.intersects(cells, geom)]
    out: list[Polygon] = []
    for piece in shapely.intersection(cells, geom):
        out.extend(g for g in shapely.get_parts(piece) if isinstance(g, Polygon) and not g.is_empty)
    return out


def _strips(furlongs: list[Polygon], angles: np.ndarray, style: CropStyle, scale: float,
            rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Each furlong cut into strips along its angle (radians from frame x), and across every
    ``length_m``: ``(parcels, furlong index)``, ``(P,)`` each.  All furlongs in one overlay."""
    if not furlongs:
        return np.empty(0, dtype=object), np.empty(0, dtype=np.int64)
    geoms = np.array(furlongs, dtype=object)
    centre = shapely.get_coordinates(shapely.centroid(geoms))
    c, s = np.cos(angles), np.sin(angles)
    # Into each strip frame: u along the strips, v across.
    xy, at = shapely.get_coordinates(shapely.get_exterior_ring(geoms), return_index=True)
    d = xy - centre[at]
    u = d[:, 0] * c[at] + d[:, 1] * s[at]
    v = -d[:, 0] * s[at] + d[:, 1] * c[at]
    n = len(geoms)
    lo_u, hi_u, lo_v, hi_v = (np.full(n, np.inf), np.full(n, -np.inf), np.full(n, np.inf), np.full(n, -np.inf))
    np.minimum.at(lo_u, at, u)
    np.maximum.at(hi_u, at, u)
    np.minimum.at(lo_v, at, v)
    np.maximum.at(hi_v, at, v)
    lo_u -= 1.0
    hi_u += 1.0
    width = style.strip_m * scale
    rows: list[np.ndarray] = []         # (k, 5): furlong, u0, v0, u1, v1
    for i in range(n):
        # Strip edges from lo_v until one passes hi_v.
        k = int(np.ceil((hi_v[i] - lo_v[i]) / (0.7 * width))) + 1
        ev = lo_v[i] + np.concatenate([[0.0], np.cumsum(width * rng.uniform(0.7, 1.3, k))])
        ev = ev[:int(np.searchsorted(ev, hi_v[i])) + 1]
        if style.length_m is None:
            rows.append(np.column_stack([np.full(len(ev) - 1, i), np.full(len(ev) - 1, lo_u[i]), ev[:-1],
                                         np.full(len(ev) - 1, hi_u[i]), ev[1:]]))
            continue
        length = style.length_m * scale
        for v0, v1 in zip(ev[:-1], ev[1:]):
            # Staggered from strip to strip, as bunds and terraces are.
            step = length * rng.uniform(0.85, 1.15)
            cuts = np.arange(lo_u[i] - rng.uniform(0.0, length), hi_u[i] + step, step)
            m = len(cuts) - 1
            rows.append(np.column_stack([np.full(m, i), cuts[:-1], np.full(m, v0), cuts[1:], np.full(m, v1)]))
    box = np.concatenate(rows)
    owner = box[:, 0].astype(np.int64)
    corners = np.stack([box[:, [1, 2]], box[:, [3, 2]], box[:, [3, 4]], box[:, [1, 4]], box[:, [1, 2]]], axis=1)
    cu, cv = corners[..., 0], corners[..., 1]
    oc, os_ = c[owner][:, None], s[owner][:, None]
    world = np.stack([cu * oc - cv * os_ + centre[owner, 0][:, None], cu * os_ + cv * oc + centre[owner, 1][:, None]],
                     axis=-1)
    cells = np.asarray(shapely.polygons(world), dtype=object)
    # On a centimetre grid: neighbours' shared corners are then exactly shared (robust overlays downstream).
    shapely.prepare(geoms)
    inside = shapely.contains_properly(geoms[owner], cells)
    cut = np.empty(len(cells), dtype=object)
    cut[inside] = shapely.set_precision(cells[inside], _GRID_M)
    cut[~inside] = shapely.set_precision(shapely.intersection(cells[~inside], geoms[owner[~inside]]), _GRID_M)
    parts, idx = shapely.get_parts(cut, return_index=True)
    full = width * (style.length_m * scale if style.length_m is not None else width)
    keep = (shapely.get_type_id(parts) == 3) & (shapely.area(parts) >= _MIN_SHARE * full)
    return parts[keep], owner[idx[keep]]


def _palette(style: CropStyle, alpha: float) -> np.ndarray:
    """``(states × _SHADES, 4)`` RGBA: each state at ``_SHADES`` brightnesses over ``± jitter``."""
    levels = 1.0 + style.jitter * np.linspace(-1.0, 1.0, _SHADES)
    rgb = np.clip(np.asarray(style.states)[:, None, :] * levels[None, :, None], 0.0, 1.0).reshape(-1, 3)
    return np.column_stack([rgb, np.full(len(rgb), alpha)]).astype(np.float32)


def build_parcels(
    fields: RegionFields, frame: Frame, hinterland: Hinterland, seed: list[int],
    params: Optional[SettlementParcelParams] = None,
) -> Optional[FieldParcels]:
    """The hinterland's parcels (the settlement at the frame's centre).  See the module
    docstring.  ``None`` when switched off."""
    p = params or SettlementParcelParams()
    if not p.enabled:
        return None
    scale = float(p.size_scale)
    crop = _crop_name(fields)
    crop_style = CROP_STYLES[crop]
    own_style = GARDEN_STYLE if hinterland.own_kind == "gardens" else crop_style
    styles = [crop_style] if own_style is crop_style else [crop_style, own_style]
    styles.append(PASTURE_STYLE)
    # One palette per style, end to end.
    offsets: dict[int, int] = {}
    blocks, edges = [], []
    for s in styles:
        offsets[id(s)] = sum(len(b) for b in blocks)
        blocks.append(_palette(s, float(p.alpha)))
        edges.append(np.tile(np.asarray(s.edge, dtype=np.float32), (len(blocks[-1]), 1)))

    rng = np.random.default_rng(seed)
    grad, texel_m = _gradient(fields, frame, 0.5 * crop_style.furlong_m * scale)
    t = fields.tex_size

    # The grain: a smooth angle field (halved, so a strip's two ends agree), on a coarse grid.
    sigma = _GRAIN_M / (frame.side_m / _GRAIN_TEXELS)
    noise = rng.standard_normal((2, _GRAIN_TEXELS, _GRAIN_TEXELS))
    grain = np.angle(gaussian_filter(noise[0], sigma, mode="wrap") + 1j * gaussian_filter(noise[1], sigma, mode="wrap"))
    grain = grain / 2.0
    # Where commons lie beyond the outer fields: unit-variance noise, smooth over _PASTURE_NOISE_M.
    commons = gaussian_filter(rng.standard_normal((_GRAIN_TEXELS, _GRAIN_TEXELS)),
                              _PASTURE_NOISE_M / (frame.side_m / _GRAIN_TEXELS), mode="wrap")
    commons /= max(float(commons.std()), 1e-12)

    def texel(x: float, y: float, n: int = t) -> tuple[int, int]:
        return (int(np.clip((frame.half - y) / frame.side_m * n, 0, n - 1)),
                int(np.clip((x + frame.half) / frame.side_m * n, 0, n - 1)))

    farmers = [(hinterland.own, np.zeros(2), own_style)]
    farmers += [(n.fields, np.asarray(n.xy, dtype=np.float64), crop_style) for n in hinterland.neighbours]
    polygons: list[Polygon] = []
    owners: list[int] = []
    shades: list[int] = []
    pastures: list[bool] = []
    for k, (geom, home, style) in enumerate(farmers):
        if geom.is_empty:
            continue
        n_states = len(style.states)
        weights = np.asarray(style.weights) / float(np.sum(style.weights))
        # Three great fields (on open fields): sectors round the farmer, at a random turn.
        turn = rng.uniform(0.0, 2.0 * math.pi)
        furlongs = _furlongs(geom, style.furlong_m * scale, rng, frame)
        if not furlongs:
            continue
        centres = shapely.get_coordinates(shapely.centroid(np.array(furlongs, dtype=object)))
        angles = np.empty(len(furlongs))
        main = np.empty(len(furlongs), dtype=np.int64)
        slope = np.empty(len(furlongs))
        lumps = np.empty(len(furlongs))
        for i, (cx, cy) in enumerate(centres):
            at = texel(cx, cy)
            g = grad[at] / 1000.0   # m/m
            slope[i] = float(np.hypot(*g))
            lumps[i] = commons[texel(cx, cy, _GRAIN_TEXELS)]
            if style.orient == "any" or float(np.hypot(*g)) < _FLAT:
                angle = float(grain[texel(cx, cy, _GRAIN_TEXELS)]) + (0.5 * math.pi if rng.random() < _ACROSS else 0.0)
            else:
                angle = math.atan2(g[1], g[0]) + (0.5 * math.pi if style.orient == "contour" else 0.0)
            angles[i] = angle + rng.normal(0.0, 0.1)
            if style.state_by == "sector":
                bearing = (math.atan2(cy - home[1], cx - home[0]) - turn) % (2.0 * math.pi)
                main[i] = int(bearing / (2.0 * math.pi) * n_states) % n_states
            else:
                main[i] = int(rng.choice(n_states, p=weights))
        # Meadow and pasture: the outer and steeper furlongs, (1 − arable_share) of the area.
        geoms = np.array(furlongs, dtype=object)
        pasture = np.zeros(len(furlongs), dtype=bool)
        if style is not GARDEN_STYLE and p.arable_share < 1.0:
            area = shapely.area(geoms)
            dist = np.hypot(centres[:, 0] - home[0], centres[:, 1] - home[1])
            score = (dist / max(float(dist.max()), 1e-9) + _PASTURE_SLOPE * np.clip(slope / _STEEP, 0.0, 1.0)
                     + _PASTURE_NOISE * lumps)
            order = np.argsort(-score)
            n_take = int(np.searchsorted(np.cumsum(area[order]), (1.0 - float(p.arable_share)) * area.sum()))
            pasture[order[:n_take]] = True
        arable = np.flatnonzero(~pasture)
        strips, of = _strips([furlongs[i] for i in arable], angles[arable], style, scale, rng)
        own = rng.random(len(strips)) < style.mix
        state = np.where(own, rng.choice(n_states, size=len(strips), p=weights), main[arable][of])
        level = rng.integers(0, _SHADES, size=len(strips))
        polygons.extend(strips.tolist())
        owners.extend([k] * len(strips))
        shades.extend((offsets[id(style)] + state * _SHADES + level).tolist())
        pastures.extend([False] * len(strips))
        if pasture.any():
            # Neighbouring pasture furlongs are one common.
            parts = shapely.get_parts(shapely.set_precision(shapely.union_all(geoms[pasture]), _GRID_M))
            parts = [g for g in parts if isinstance(g, Polygon) and not g.is_empty]
            tone = rng.integers(0, len(PASTURE_STYLE.states), size=len(parts))
            level = rng.integers(0, _SHADES, size=len(parts))
            polygons.extend(parts)
            owners.extend([k] * len(parts))
            shades.extend((offsets[id(PASTURE_STYLE)] + tone * _SHADES + level).tolist())
            pastures.extend([True] * len(parts))
    return FieldParcels(
        style="Gardens" if own_style is GARDEN_STYLE else crop, polygons=polygons,
        owner=np.asarray(owners, dtype=np.int32), shade=np.asarray(shades, dtype=np.int32),
        palette=np.concatenate(blocks), edge=np.concatenate(edges),
        strip_m=min(s.strip_m for s in styles if s is not PASTURE_STYLE) * scale,
        pasture=np.asarray(pastures, dtype=bool),
    )
