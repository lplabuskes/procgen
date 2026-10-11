"""A settlement's footprint and buildable ground, as polygons.

The **footprint** is the built-up ground, claimed on the terrain raster and
traced to polygons (plan Decision 4):

- **A city** has a local footprint (:mod:`localmap.cities`): each local texel's
  built-up share, 1 or 0, from the owning local map (:func:`city_shares`).  The
  share is blurred over ``edge_smoothing_local_texels`` and the local
  footprint's area is claimed by priority over the whole frame.  So the total
  matches the local area to a fine texel, and each local texel's share holds
  approximately, not exactly (seed 4: off 2–4% on average, 8% where the fine
  water differs most; :meth:`Footprint.share_error`): exact quotas per local
  texel forced claims into every edge texel whatever its priority, leaving
  specks and the local texels' squares in the outline.
- **A town or village** has none: it claims ``people / density`` fresh, by
  the same priority, nearness to its centre standing in for the share.

Priority: the blurred share, plus spatially correlated noise along the edge
for a ragged outline (``edge_roughness``), less a penalty on ground
steeper than ``slope_free``.  Never water or a river channel: a city spans its
rivers.  The claimed texels are traced (marching squares on the lightly
smoothed claim) and the water cut out exactly.

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
from scipy.ndimage import gaussian_filter
from shapely.geometry import MultiPolygon

from core.params import param_field as _f, paramclass
from localmap.builder import LocalMap
from region.resources import refined_slope
from region.upsample import RegionFields
from settlegen.contour import regions_below
from settlegen.frame import Frame
from settlegen.water import SiteWater, _as_multipolygon, channel_polygons

if TYPE_CHECKING:
    from settlegen.chain import SettlementRef

_KM2_PER_HA = 0.01
# Smallest piece of buildable ground kept, in fine texels.
_MIN_PIECE_TEXELS = 4.0


@paramclass
class SettlementFootprintParams:
    """Tuning for the footprint and buildable ground."""

    # σ of the blur that turns the local footprint's 0/1 shares into a smooth edge.
    edge_smoothing_local_texels: float = _f(
        0.5, label="Footprint edge smoothing (local texels)", min=0.0, max=2.0, step=0.05)
    edge_roughness: float = _f(1.0, label="Footprint edge roughness", min=0.0, max=3.0, step=0.05)
    # Correlation length of the edge noise.
    roughness_m: float = _f(100.0, label="Edge roughness scale (m)", min=5.0, max=500.0, step=5.0)
    slope_free: float = _f(0.05, label="Footprint slope free (km/km)", min=0.0, max=1.0, step=0.01)
    slope_weight: float = _f(4.0, label="Footprint slope penalty", min=0.0, max=50.0, step=0.5)
    buildable_max_slope: float = _f(0.3, label="Buildable max slope (km/km)", min=0.01, max=2.0, step=0.01)


@dataclass
class Footprint:
    """The claim on the raster (for checks) and its polygon."""

    polygon: MultiPolygon
    claimed: np.ndarray       # (T, T) bool
    target: int               # fine texels the claim was due
    # Cities: the local share under each fine texel and that local texel's flat index (-1 off the map).
    share: Optional[np.ndarray] = None
    texel: Optional[np.ndarray] = None

    def share_error(self, full: float = 0.8) -> tuple[float, float]:
        """``(mean, max)`` over the local texels in the frame (at least ``full`` of
        the fullest one's fine texels) of ``|claimed share − local share|``; zeros for a town."""
        if self.share is None or self.texel is None:
            return 0.0, 0.0
        _, label = np.unique(self.texel.reshape(-1), return_inverse=True)
        n = np.bincount(label)
        got = np.bincount(label, self.claimed.reshape(-1)) / n
        want = np.bincount(label, self.share.reshape(-1)) / n
        err = np.abs(got - want)[n >= full * n.max()]
        return (float(err.mean()), float(err.max())) if err.size else (0.0, 0.0)


def _seed(globe_seed: int, ref: "SettlementRef") -> list[int]:
    kind = zlib.crc32(ref.kind.encode())
    where = list(ref.map) if ref.map is not None else []
    return [int(globe_seed) & 0xFFFFFFFF, kind, int(ref.index), *where]


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


def _priority(base: np.ndarray, slope: np.ndarray, frame: Frame, params: SettlementFootprintParams,
              rng: np.random.Generator) -> np.ndarray:
    texel_m = frame.side_m / base.shape[0]
    noise = _noise(base.shape, float(params.roughness_m) / texel_m, rng)
    # Only along the edge: none deep inside (share 1) or far out (share 0).
    edge = 4.0 * np.clip(base, 0.0, 1.0) * (1.0 - np.clip(base, 0.0, 1.0))
    steep = np.maximum(slope - float(params.slope_free), 0.0)
    return base + 0.25 * float(params.edge_roughness) * edge * noise - float(params.slope_weight) * steep


def _claim(priority: np.ndarray, allowed: np.ndarray, n: int) -> np.ndarray:
    """The ``n`` allowed texels of highest priority."""
    flat = np.where(allowed.reshape(-1), priority.reshape(-1), -np.inf)
    n = int(min(max(n, 0), int(allowed.sum())))
    claimed = np.zeros(flat.size, dtype=bool)
    if n:
        claimed[np.argpartition(-flat, n - 1)[:n]] = True
    return claimed.reshape(priority.shape)


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
    return _as_multipolygon(poly)


def city_shares(fields: RegionFields, owner: LocalMap, city: int) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """``((T, T) share, (T, T) local texel)``: the 0/1 built-up share for world city
    ``city`` of the owning local map's texel under each fine texel, and that
    texel's flat index (``-1`` off the map); None if the map has no footprint for it.

    The owning map's alone: it claims the whole city, over-hang past its parent
    hexagon included, where each neighbour claims it again around its own drawn
    position (their footprints are jittered per map), so mixing maps would count
    the city twice across a seam.
    """
    if owner.city_footprints is None:
        return None
    hit = np.flatnonzero(np.asarray(owner.city_index) == city)
    if hit.size == 0:
        return None
    local = owner.fields
    t = fields.tex_size
    s, u, valid = local.geometry.ecef_to_local(fields.positions.reshape(-1, 3).astype(np.float64))
    hexes, rows, cols, inside = local.grid.texels_at(s, u, local.tex_size)
    inside &= valid
    texel = np.where(inside, np.ravel_multi_index((hexes, rows, cols), local.in_hex.shape), -1)
    owned = owner.city_footprints.owner.reshape(-1) == int(hit[0])
    share = np.where(inside, owned[np.maximum(texel, 0)], False).astype(np.float64)
    return share.reshape(t, t), texel.reshape(t, t)


def build_footprint(
    fields: RegionFields, frame: Frame, maps: Sequence[LocalMap], water: SiteWater, ref: "SettlementRef",
    area_km2: float, globe_seed: int, params: Optional[SettlementFootprintParams] = None,
) -> Footprint:
    """The settlement's footprint.  See the module docstring.  ``area_km2`` is its
    built-up area (a city's: its local footprint's)."""
    params = params or SettlementFootprintParams()
    t = fields.tex_size
    rng = np.random.default_rng(_seed(globe_seed, ref))
    wet = wet_texels(fields, frame, water)
    slope = site_slope(fields, frame)
    texel_m = frame.side_m / t
    wet_poly = _as_multipolygon(shapely.union_all([water.sea, water.lakes, water.channels]))

    target = int(round(float(area_km2) * 1e6 / texel_m ** 2))
    shares = city_shares(fields, maps[0], ref.index) if ref.kind == "city" else None
    if shares is not None:
        share, texel = shares
        local_m = np.sqrt(maps[0].fields.texel_area_km2) * 1000.0
        sigma = float(params.edge_smoothing_local_texels) * local_m / texel_m
        base = gaussian_filter(share, sigma, mode="constant") if sigma > 0 else share
    else:
        share = texel = None
        # Nearness to the centre on the same scale as a city's blurred share (0.5 at the radius).
        xy = frame.texel_xy(t)
        radius_m = max(np.sqrt(max(float(area_km2), 0.0) / np.pi) * 1000.0, texel_m)
        base = np.clip(1.0 - np.hypot(xy[..., 0], xy[..., 1]) / (2.0 * radius_m), 0.0, 1.0)
    claimed = _claim(_priority(base, slope, frame, params, rng), ~wet, target)
    return Footprint(polygon=_trace(claimed, frame, wet_poly), claimed=claimed, target=target,
                     share=share, texel=texel)


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
