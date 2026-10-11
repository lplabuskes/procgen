"""How a settlement map's terrain is coloured: the Elevation layer at street scale.

The world's elevation colours span kilometres; a settlement frame's land spans
metres to tens of metres, so it would be one flat colour.  :func:`relief_rgba`
stretches a muted hypsometric ramp over the frame's own land (its 2nd–98th
percentile, at least ``min_span_m``) and hillshades it with a vertical
exaggeration that makes its typical slope visible (at most
``_MAX_EXAGGERATION``: more lifts the local maps' hex seams into ridges).  The
water is the site's vector water's to draw (:data:`WATER_RGB`): the raster
colours its texels by their bed, like the land, so where the two disagree by a
texel the shore follows the vector.  Qt-free.
"""
from __future__ import annotations

import numpy as np

from region.display import hillshade
from region.upsample import RegionFields

# The site's water: sea, lakes and river channels alike.
WATER_RGB = (0.35, 0.60, 0.85)

# Low → high, by share of the frame's land range.
_STOPS = [
    (0.00, (0.47, 0.60, 0.38)),
    (0.30, (0.62, 0.68, 0.44)),
    (0.55, (0.76, 0.73, 0.52)),
    (0.80, (0.72, 0.62, 0.47)),
    (1.00, (0.80, 0.76, 0.70)),
]
# Hillshade: the exaggeration that brings the land's 90th-percentile slope to this (km/km), within [1, MAX].
_SHADE_SLOPE = 0.15
_MAX_EXAGGERATION = 8.0


def _land(fields: RegionFields) -> np.ndarray:
    wet = np.zeros(fields.textures["elevation"].shape[:3], dtype=bool)
    for key in ("ocean_mask", "lake_mask"):
        if key in fields.textures:
            wet |= fields.textures[key][..., 0] > 0.5
    return ~wet


def relief_range_m(fields: RegionFields, min_span_m: float = 5.0) -> tuple[float, float]:
    """``(low, high)`` metres the ramp spans: the land's 2nd–98th percentile, widened about
    its middle to ``min_span_m``."""
    elev = fields.textures["elevation"][..., 0] * 1000.0
    land = _land(fields)
    values = elev[land] if land.any() else elev.reshape(-1)
    lo, hi = (float(v) for v in np.percentile(values, [2.0, 98.0]))
    if hi - lo < min_span_m:
        mid = 0.5 * (lo + hi)
        lo, hi = mid - 0.5 * min_span_m, mid + 0.5 * min_span_m
    return lo, hi


def ramp(share: np.ndarray) -> np.ndarray:
    """``(..., 3)`` colours of the ramp at ``share`` in [0, 1]."""
    x = np.clip(np.asarray(share, dtype=np.float64), 0.0, 1.0)
    at = np.array([a for a, _ in _STOPS])
    rgb = np.array([c for _, c in _STOPS])
    return np.stack([np.interp(x, at, rgb[:, k]) for k in range(3)], axis=-1)


def relief_rgba(fields: RegionFields) -> np.ndarray:
    """``(M, T, T, 4)`` float32 colours of the frame's terrain.  See the module docstring."""
    elev = fields.textures["elevation"][..., 0].astype(np.float64) * 1000.0
    lo, hi = relief_range_m(fields)
    land = _land(fields)
    rgba = np.ones((*elev.shape, 4), dtype=np.float32)
    rgba[..., :3] = ramp((elev - lo) / (hi - lo))
    texel_km = float(np.sqrt(max(fields.texel_area_km2, 1e-12)))
    gy, gx = np.gradient(elev / 1000.0, texel_km, axis=(1, 2))
    slope = np.hypot(gx, gy)[land]
    p90 = float(np.percentile(slope, 90.0)) if slope.size else 0.0
    exaggeration = float(np.clip(_SHADE_SLOPE / max(p90, 1e-9), 1.0, _MAX_EXAGGERATION))
    return hillshade(rgba, fields.textures["elevation"], texel_km / exaggeration, water=~land)


def relief_legend(fields: RegionFields, n: int = 5) -> list[tuple[str, tuple]]:
    """``(label, rgba)`` entries over the ramp, low → high, in metres."""
    lo, hi = relief_range_m(fields)
    out = []
    for share in np.linspace(0.0, 1.0, n):
        r, g, b = (float(c) for c in ramp(np.array(share)))
        out.append((f"{lo + share * (hi - lo):+.0f} m", (r, g, b, 1.0)))
    return out


# The layout's districts (settlegen.layout.DISTRICTS), muted so the lots read over them.
DISTRICT_RGB: dict[str, tuple[float, float, float]] = {
    "residential": (0.80, 0.66, 0.52),
    "suburb": (0.82, 0.74, 0.62),
    "gardens": (0.68, 0.74, 0.52),
    "elite": (0.86, 0.58, 0.50),
    "market": (0.93, 0.78, 0.45),
    "quay": (0.62, 0.72, 0.80),
    "graveyard": (0.55, 0.65, 0.50),
    "square": (0.90, 0.87, 0.80),
    "tanners": (0.62, 0.50, 0.40),
    "dyers": (0.66, 0.50, 0.66),
    "smiths": (0.55, 0.52, 0.52),
    "butchers": (0.60, 0.30, 0.36),
    "brewers": (0.80, 0.68, 0.40),
    "millers": (0.86, 0.80, 0.60),
    "sugar mills": (0.88, 0.82, 0.70),
    "weavers": (0.74, 0.70, 0.82),
    "carpenters": (0.72, 0.60, 0.42),
    "shipwrights": (0.56, 0.64, 0.72),
    "salters": (0.85, 0.85, 0.88),
    "fishers": (0.58, 0.70, 0.74),
    "artisans": (0.78, 0.66, 0.74),
}
STREET_RGB = (0.93, 0.90, 0.84)        # the footprint between blocks
ARTERIAL_RGB = (0.97, 0.95, 0.90)
OPEN_RGB = {"plaza": (0.94, 0.90, 0.80), "precinct": (0.92, 0.92, 0.86), "citadel": (0.60, 0.48, 0.42),
            "green": (0.56, 0.70, 0.42), "quays": (0.80, 0.78, 0.74), "pier": (0.62, 0.48, 0.34)}
