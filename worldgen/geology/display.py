from __future__ import annotations

import colorsys

import numpy as np


OCEAN_COLOR = (0.25, 0.55, 0.85, 1.0)
LAND_COLOR = (0.72, 0.52, 0.25, 1.0)
BOUNDARY_COLORS = {
    "interior":   (0.70, 0.70, 0.70, 1.0),
    "convergent": (0.85, 0.30, 0.10, 1.0),
    "divergent":  (0.95, 0.90, 0.30, 1.0),
    "transform":  (0.60, 0.20, 0.80, 1.0),
}

# Separate elevation color scales for ocean and land pixels.
# Ocean pixels are colored by depth; land pixels by elevation (including
# below-sea-level depressions that are not ocean-connected).
_OCEAN_ELEVATION_STOPS = [
    (-8.0, (0.02, 0.06, 0.35, 1.0)),  # abyssal plain
    (-3.5, (0.06, 0.22, 0.62, 1.0)),  # deep ocean
    (-0.5, (0.16, 0.48, 0.82, 1.0)),  # continental shelf
    (0.0,  (0.20, 0.52, 0.88, 1.0)),  # sea level
]

_LAND_ELEVATION_STOPS = [
    (-0.5, (0.62, 0.52, 0.28, 1.0)),  # below-sea-level land (arid depression)
    (0.0,  (0.78, 0.70, 0.42, 1.0)),  # beach / sand
    (0.8,  (0.33, 0.60, 0.20, 1.0)),  # lowland / plains
    (2.5,  (0.50, 0.38, 0.18, 1.0)),  # highland
    (5.5,  (0.92, 0.92, 0.92, 1.0)),  # snow-capped peaks
]


def _plate_palette(n: int) -> np.ndarray:
    """Return (n, 4) float32 RGBA colors evenly spaced around the HSV hue wheel."""
    hues = np.linspace(0, 1, n, endpoint=False)
    rgba = np.ones((n, 4), dtype=np.float32)
    for i, h in enumerate(hues):
        r, g, b = colorsys.hsv_to_rgb(h, 0.65, 0.85)
        rgba[i, :3] = r, g, b
    return rgba


def _interp_stops_scalar(e: float, stops: list) -> tuple:
    """Interpolate a scalar elevation through a stops list."""
    if e <= stops[0][0]:
        return stops[0][1]
    if e >= stops[-1][0]:
        return stops[-1][1]
    for i in range(len(stops) - 1):
        e0, c0 = stops[i]
        e1, c1 = stops[i + 1]
        if e0 == e1:
            continue
        if e0 <= e <= e1:
            t = (e - e0) / (e1 - e0)
            return tuple(c0[j] + t * (c1[j] - c0[j]) for j in range(4))
    return stops[-1][1]


def _elevation_color(meta: dict) -> tuple:
    e = meta["elevation"]
    is_ocean = bool(meta.get("is_ocean", e < 0.0))
    stops = _OCEAN_ELEVATION_STOPS if is_ocean else _LAND_ELEVATION_STOPS
    return _interp_stops_scalar(e, stops)


def elevation_legend_entries(n: int = 5) -> list[tuple[str, tuple]]:
    """Ordered (label, rgba) entries for a land-elevation gradient legend, low→high.

    Entries are evenly spaced in elevation across the full stop range, with
    colors interpolated from _LAND_ELEVATION_STOPS.
    """
    e_min = _LAND_ELEVATION_STOPS[0][0]
    e_max = _LAND_ELEVATION_STOPS[-1][0]
    elevations = np.linspace(e_min, e_max, n)
    entries = []
    for e in elevations:
        color = _interp_stops_scalar(e, _LAND_ELEVATION_STOPS)
        label = f"{e:+.1f} km" if e != 0.0 else "0 km"
        entries.append((label, color))
    return entries


def _apply_stops_vec(elevs: np.ndarray, stops: list) -> np.ndarray:
    """Linear interpolation through stops for an arbitrary-shape elevation array.

    Returns (..., 4) float32.
    """
    shape = elevs.shape
    result = np.full((*shape, 4), stops[-1][1], dtype=np.float32)
    for i in range(len(stops) - 2, -1, -1):
        e0, c0 = stops[i]
        e1, c1 = stops[i + 1]
        if e0 == e1:
            continue
        mask = (elevs >= e0) & (elevs <= e1)
        t = np.where(mask, (elevs - e0) / (e1 - e0), 0.0)
        for ch in range(4):
            result[..., ch] = np.where(mask, c0[ch] + t * (c1[ch] - c0[ch]), result[..., ch])
    result[elevs <= stops[0][0]] = np.array(stops[0][1], dtype=np.float32)
    result[elevs >= stops[-1][0]] = np.array(stops[-1][1], dtype=np.float32)
    return result


def _elevation_color_vec(elevs: np.ndarray, ocean_mask: np.ndarray | None = None) -> np.ndarray:
    """Vectorized elevation coloring with separate ocean and land scales.

    Args:
        elevs: (...) float array of elevations in km.
        ocean_mask: optional bool array, same shape as elevs. True = ocean pixel.
            Falls back to elevs < 0 when not provided (backward-compatible).

    Returns:
        (..., 4) float32 RGBA array.
    """
    if ocean_mask is None:
        ocean_mask = elevs < 0.0
    is_ocean = ocean_mask.astype(bool)
    result = np.empty((*elevs.shape, 4), dtype=np.float32)
    result[is_ocean] = _apply_stops_vec(elevs[is_ocean], _OCEAN_ELEVATION_STOPS)
    result[~is_ocean] = _apply_stops_vec(elevs[~is_ocean], _LAND_ELEVATION_STOPS)
    return result


def build_elevation_display_texture(mesh) -> np.ndarray:
    """RGBA (N, H, W, 4) elevation texture from the single-channel ``"elevation"``
    data layer, painting ocean pixels by depth and lake pixels in lake blue."""
    from worldgen.hydrology.builder import LAKE_COLOR

    elev_tex = mesh.face_textures["elevation"]          # (N, H, W, 1)
    ocean_mask = None
    if "ocean_mask" in mesh.face_textures:
        ocean_mask = mesh.face_textures["ocean_mask"][:, :, :, 0] > 0  # (N, H, W) bool
    rgba = _elevation_color_vec(elev_tex[:, :, :, 0], ocean_mask=ocean_mask)  # (N, H, W, 4)
    if "lake_mask" in mesh.face_textures:
        lake_tex = mesh.face_textures["lake_mask"]       # (N, H, W, 1)
        rgba[lake_tex[:, :, :, 0] > 0] = LAKE_COLOR
    return rgba
