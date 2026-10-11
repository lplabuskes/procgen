"""A settlement's footprint and buildable ground, as polygons.

The settlement tier is the authority for the footprint: it claims every
settlement's built-up ground fresh from its own terrain, water and roads, and
owes nothing to the local map's coarse footprint (which stays a preview there).

**Density** (people per hectare of built-up ground, :func:`base_density_per_ha`)
is derived, with a random spread:

- size: ``density_per_ha`` at ``density_reference_population`` people, scaled
  by ``(population / reference) ** density_size_exponent`` — bigger places are
  denser;
- walls pack people in (``walled_density_mult``);
- a lognormal jitter of ``density_jitter`` per settlement;
- the site, once its terrain is known (:func:`build_footprint`): ground hemmed in
  by water or steep slopes packs denser, by ``(buildable share) ** -constraint_exponent``
  of the ground around the unconstrained footprint, at most ``constraint_max_mult``.

The frame is sized before the terrain exists (:mod:`settlegen.chain`), from
everything but the site; the site only raises density, so the footprint fits.

**The claim**: ``population / density`` of dry ground, by priority — nearness to
the centre, a pull towards the roads (ribbon growth along them,
``road_pull`` within ``road_pull_m``) and the water (a river or harbour
town grows to its waterfront, ``water_pull`` within ``water_pull_m``), spatially correlated noise along the
edge for a ragged outline (``edge_roughness``), less a penalty on ground steeper
than ``slope_free`` (the slope smoothed over ``slope_smoothing_m``).  Never water
or a river channel: a city spans its rivers.  Detached pieces under
``min_piece_fraction`` of it go back to the main body, grown at its edge by priority.
The claimed texels are traced (marching squares on the lightly smoothed claim,
the level matched to the claimed area), simplified, and the water cut out exactly.

The **buildable** ground is the frame less the water, the channels and ground
steeper than ``buildable_max_slope``.

Seeded per (world, settlement kind, index) (Decision 3).  Qt-free.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np
import shapely
from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter, label
from shapely.geometry import MultiPolygon

from core.params import param_field as _f, paramclass
from region.resources import refined_slope
from region.upsample import RegionFields
from settlegen.contour import regions_below
from settlegen.frame import Frame
from settlegen.water import SiteWater, _as_multipolygon, channel_polygons

if TYPE_CHECKING:
    from core.world import World
    from settlegen.chain import SettlementRef

_M2_PER_HA = 10_000.0
# Smallest piece of buildable ground kept, in fine texels.
_MIN_PIECE_TEXELS = 4.0
# RNG stream tags under the settlement's seed.
_DENSITY_STREAM, _CLAIM_STREAM = 1, 2


@paramclass
class SettlementFootprintParams:
    """Tuning for the density, footprint and buildable ground."""

    density_per_ha: float = _f(70.0, label="Built-up density at the reference size (people/ha)",
                               min=5.0, max=500.0, step=5.0)
    density_reference_population: float = _f(1000.0, label="Density reference population",
                                             min=100.0, max=100_000.0, step=100.0)
    density_size_exponent: float = _f(0.2, label="Density size exponent", min=0.0, max=1.0, step=0.05)
    walled_density_mult: float = _f(1.3, label="Walled density (×)", min=1.0, max=3.0, step=0.05)
    density_jitter: float = _f(0.3, label="Density jitter (lognormal σ)", min=0.0, max=1.0, step=0.05)
    constraint_exponent: float = _f(0.5, label="Hemmed-in density exponent", min=0.0, max=2.0, step=0.05)
    constraint_max_mult: float = _f(2.0, label="Hemmed-in density max (×)", min=1.0, max=5.0, step=0.1)
    min_density_per_ha: float = _f(10.0, label="Min density (people/ha)", min=1.0, max=100.0, step=1.0)
    max_density_per_ha: float = _f(450.0, label="Max density (people/ha)", min=50.0, max=2000.0, step=10.0)
    road_pull: float = _f(0.3, label="Road pull", min=0.0, max=2.0, step=0.05)
    road_pull_m: float = _f(120.0, label="Road pull reach (m)", min=10.0, max=1000.0, step=10.0)
    water_pull: float = _f(0.3, label="Waterfront pull", min=0.0, max=2.0, step=0.05)
    water_pull_m: float = _f(150.0, label="Waterfront pull reach (m)", min=10.0, max=1000.0, step=10.0)
    edge_roughness: float = _f(1.0, label="Footprint edge roughness", min=0.0, max=3.0, step=0.05)
    # Correlation length of the edge noise.
    roughness_m: float = _f(100.0, label="Edge roughness scale (m)", min=5.0, max=500.0, step=5.0)
    slope_free: float = _f(0.05, label="Footprint slope free (km/km)", min=0.0, max=1.0, step=0.01)
    # The slope the penalty reads, smoothed: hillsides, not the street-scale bumps.
    slope_smoothing_m: float = _f(40.0, label="Footprint slope smoothing (m)", min=0.0, max=300.0, step=5.0)
    # Detached pieces smaller than this share of the footprint go back to its main body.
    min_piece_fraction: float = _f(0.03, label="Smallest footprint piece (share)", min=0.0, max=0.5, step=0.01)
    slope_weight: float = _f(4.0, label="Footprint slope penalty", min=0.0, max=50.0, step=0.5)
    buildable_max_slope: float = _f(0.3, label="Buildable max slope (km/km)", min=0.01, max=2.0, step=0.01)


@dataclass
class Footprint:
    """The claim on the raster (for checks), its polygon and the density it was claimed at."""

    polygon: MultiPolygon
    claimed: np.ndarray       # (T, T) bool
    target: int               # fine texels the claim was due
    density_per_ha: float     # with the site's squeeze
    constraint: float         # the site's squeeze (×, ≥ 1)


def _seed(globe_seed: int, ref: "SettlementRef", stream: int) -> list[int]:
    kind = zlib.crc32(ref.kind.encode())
    where = list(ref.map) if ref.map is not None else []
    return [int(globe_seed) & 0xFFFFFFFF, kind, int(ref.index), *where, stream]


def base_density_per_ha(
    globe: "World", ref: "SettlementRef", population: float, params: Optional[SettlementFootprintParams] = None,
) -> float:
    """The settlement's density before its site is known: size, walls and jitter."""
    params = params or SettlementFootprintParams()
    size = max(float(population), 1.0) / max(float(params.density_reference_population), 1.0)
    density = float(params.density_per_ha) * size ** float(params.density_size_exponent)
    walled = False
    if ref.kind == "city":
        walled = bool(globe.get_feature("settlements")[ref.index].has_walls)
    elif ref.kind == "town":
        walled = bool(globe.get_feature("towns").has_walls[ref.index])
    if walled:
        density *= float(params.walled_density_mult)
    rng = np.random.default_rng(_seed(int(globe.seed or 0), ref, _DENSITY_STREAM))
    density *= float(np.exp(float(params.density_jitter) * rng.standard_normal()))
    return float(np.clip(density, params.min_density_per_ha, params.max_density_per_ha))


def wet_texels(fields: RegionFields, frame: Frame, water: SiteWater) -> np.ndarray:
    """``(T, T)`` texels under the sea, a lake or a river channel."""
    shape = (fields.tex_size, fields.tex_size)
    wet = np.zeros(shape, dtype=bool)
    for key in ("ocean_mask", "lake_mask"):
        if key in fields.textures:
            wet |= fields.textures[key][0, ..., 0] > 0.5
    if not water.channels.is_empty:
        xy = frame.texel_xy(fields.tex_size)
        wet |= shapely.contains_xy(water.channels, xy[..., 0], xy[..., 1])
    return wet


def site_slope(fields: RegionFields, frame: Frame) -> np.ndarray:
    """``(T, T)`` ground slope (km/km)."""
    texel_km = frame.side_m / fields.tex_size / 1000.0
    return refined_slope(fields.textures["elevation"][..., 0], texel_km)[0]


def _noise(shape: tuple[int, int], sigma_texels: float, rng: np.random.Generator) -> np.ndarray:
    """Unit-variance noise correlated over ``sigma_texels``."""
    n = gaussian_filter(rng.standard_normal(shape), max(float(sigma_texels), 0.5), mode="wrap")
    return n / max(float(n.std()), 1e-12)


def _road_distance_m(roads: Sequence[np.ndarray], frame: Frame, tex_size: int) -> Optional[np.ndarray]:
    """``(T, T)`` distance (m) from each texel to the nearest road; None without roads."""
    texel_m = frame.side_m / tex_size
    on_road = np.zeros((tex_size, tex_size), dtype=bool)
    for xy in roads:
        xy = np.asarray(xy, dtype=np.float64)
        if len(xy) < 2:
            continue
        seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        n = np.maximum(np.ceil(seg / (0.5 * texel_m)).astype(int), 1)
        pts = np.concatenate([xy[k] + np.linspace(0.0, 1.0, n[k], endpoint=False)[:, None] * (xy[k + 1] - xy[k])
                              for k in range(len(seg))] + [xy[-1:]])
        cols = np.floor((pts[:, 0] + frame.half) / texel_m).astype(int)
        rows = np.floor((frame.half - pts[:, 1]) / texel_m).astype(int)
        ok = (rows >= 0) & (rows < tex_size) & (cols >= 0) & (cols < tex_size)
        on_road[rows[ok], cols[ok]] = True
    if not on_road.any():
        return None
    return np.asarray(distance_transform_edt(~on_road)) * texel_m


def _claim(priority: np.ndarray, allowed: np.ndarray, n: int) -> np.ndarray:
    """The ``n`` allowed texels of highest priority."""
    flat = np.where(allowed.reshape(-1), priority.reshape(-1), -np.inf)
    n = int(min(max(n, 0), int(allowed.sum())))
    claimed = np.zeros(flat.size, dtype=bool)
    if n:
        claimed[np.argpartition(-flat, n - 1)[:n]] = True
    return claimed.reshape(priority.shape)


def _consolidate(claimed: np.ndarray, priority: np.ndarray, allowed: np.ndarray, min_texels: float) -> np.ndarray:
    """``claimed`` less its detached pieces under ``min_texels``, their texels re-claimed
    on the edge of what is left, by priority (the largest piece always stays)."""
    labelled = label(claimed, structure=np.ones((3, 3), dtype=bool))
    labels, n = np.asarray(labelled[0], dtype=np.int64), int(labelled[1])  # type: ignore[index]
    if n <= 1:
        return claimed
    sizes = np.bincount(labels.reshape(-1), minlength=n + 1)
    small = sizes < min_texels
    small[0] = False
    small[int(np.argmax(sizes[1:])) + 1] = False
    if not small.any():
        return claimed
    out = claimed & ~small[labels]
    owed = int(claimed.sum() - out.sum())
    for _ in range(10_000):
        if owed <= 0:
            break
        frontier = binary_dilation(out, structure=np.ones((3, 3), dtype=bool)) & allowed & ~out
        idx = np.flatnonzero(frontier.reshape(-1))
        if idx.size == 0:
            break
        # A ring at a time, so the body grows evenly by priority.
        take = idx[np.argsort(-priority.reshape(-1)[idx], kind="stable")[:min(owed, max(idx.size // 4, 1))]]
        out.reshape(-1)[take] = True
        owed -= take.size
    return out


def _trace(claimed: np.ndarray, frame: Frame, wet_poly: MultiPolygon) -> MultiPolygon:
    """The claimed texels as polygons: contoured on the claim lightly smoothed, water cut out."""
    if not claimed.any():
        return MultiPolygon()
    soft = gaussian_filter(claimed.astype(np.float64), 0.7, mode="constant")
    # The level whose contour holds as many texels as were claimed (smoothing rounds corners off).
    level = float(np.sort(soft.reshape(-1))[::-1][int(claimed.sum()) - 1])
    level = min(max(level, 0.3), 0.7)
    poly = regions_below(level - soft, frame.texel_xy(claimed.shape[0]), frame.box())
    texel_m = frame.side_m / claimed.shape[0]
    # Simplified before the water is cut out, so the shore stays exact.
    poly = shapely.make_valid(poly.simplify(0.25 * texel_m))
    if not wet_poly.is_empty:
        poly = poly.difference(wet_poly)
    # Slivers the water's cut leaves.
    return MultiPolygon([g for g in _as_multipolygon(poly).geoms if g.area >= _MIN_PIECE_TEXELS * texel_m ** 2])


def build_footprint(
    fields: RegionFields, frame: Frame, water: SiteWater, ref: "SettlementRef", population: float,
    density_per_ha: float, globe_seed: int, roads: Sequence[np.ndarray] = (),
    params: Optional[SettlementFootprintParams] = None,
) -> Footprint:
    """The settlement's footprint.  See the module docstring.  ``density_per_ha`` is
    :func:`base_density_per_ha`; ``roads`` are the site's road polylines in frame metres."""
    params = params or SettlementFootprintParams()
    t = fields.tex_size
    texel_m = frame.side_m / t
    wet = wet_texels(fields, frame, water)
    slope = site_slope(fields, frame)
    xy = frame.texel_xy(t)
    dist = np.hypot(xy[..., 0], xy[..., 1])

    # The site's squeeze: the buildable share of the ground the unconstrained footprint would reach.
    def radius_m(density: float) -> float:
        return max(np.sqrt(float(population) / max(density, 1e-9) * _M2_PER_HA / np.pi), texel_m)

    near = dist <= 1.5 * radius_m(density_per_ha)
    ok = ~wet & (slope <= float(params.buildable_max_slope))
    share = float(ok[near].mean()) if near.any() else 1.0
    constraint = float(np.clip(max(share, 1e-3) ** -float(params.constraint_exponent),
                               1.0, float(params.constraint_max_mult)))
    density = float(np.clip(density_per_ha * constraint, params.min_density_per_ha, params.max_density_per_ha))
    r = radius_m(density)
    target = int(round(float(population) / density * _M2_PER_HA / texel_m ** 2))

    # Nearness, 0.5 at the radius; the roads pull growth along them out to twice that.
    base = np.clip(1.0 - dist / (2.0 * r), 0.0, 1.0)
    reach = base > 0
    road_d = _road_distance_m(roads, frame, t)
    if road_d is not None and params.road_pull > 0:
        base = base + float(params.road_pull) * np.exp(-road_d / float(params.road_pull_m)) * reach
    if wet.any() and params.water_pull > 0:
        water_d = np.asarray(distance_transform_edt(~wet)) * texel_m
        base = base + float(params.water_pull) * np.exp(-water_d / float(params.water_pull_m)) * reach
    rng = np.random.default_rng(_seed(globe_seed, ref, _CLAIM_STREAM))
    noise = _noise(base.shape, float(params.roughness_m) / texel_m, rng)
    # Only along the edge: none deep inside or far out.
    edge = 4.0 * np.clip(base, 0.0, 1.0) * (1.0 - np.clip(base, 0.0, 1.0))
    hills = gaussian_filter(slope, float(params.slope_smoothing_m) / texel_m) if params.slope_smoothing_m > 0 \
        else slope
    steep = np.maximum(hills - float(params.slope_free), 0.0)
    priority = base + 0.25 * float(params.edge_roughness) * edge * noise - float(params.slope_weight) * steep
    claimed = _consolidate(_claim(priority, ~wet, target), priority, ~wet,
                           max(float(params.min_piece_fraction) * target, 1.0))
    wet_poly = _as_multipolygon(shapely.union_all([water.sea, water.lakes, water.channels]))
    return Footprint(polygon=_trace(claimed, frame, wet_poly), claimed=claimed, target=target,
                     density_per_ha=density, constraint=constraint)


def buildable_ground(
    fields: RegionFields, frame: Frame, water: SiteWater, params: Optional[SettlementFootprintParams] = None,
) -> MultiPolygon:
    """The frame less its water, channels and ground steeper than ``buildable_max_slope``."""
    params = params or SettlementFootprintParams()
    slope = site_slope(fields, frame)
    texel_m = frame.side_m / fields.tex_size
    steep = regions_below(float(params.buildable_max_slope) - slope, frame.texel_xy(fields.tex_size), frame.box())
    # The slope contour simplified, the water cut out exactly.
    blocked = shapely.union_all([water.sea, water.lakes, channel_polygons(water.lines, frame.box()),
                                 shapely.make_valid(steep.simplify(0.25 * texel_m))])
    ground = frame.box().difference(blocked)
    pieces = [g for g in getattr(ground, "geoms", [ground])
              if g.geom_type == "Polygon" and g.area >= _MIN_PIECE_TEXELS * texel_m ** 2]
    return MultiPolygon(pieces)
