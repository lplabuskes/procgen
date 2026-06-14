from __future__ import annotations

import colorsys

import numpy as np

from viz.color_stops import interp_stops, interp_stops_scalar, gradient_scheme, categorical_scheme


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


_OCEAN_RGBA = np.array(OCEAN_COLOR, dtype=np.float32)


def is_ocean_face(meta: dict) -> bool:
    if "is_ocean" in meta:
        return bool(meta["is_ocean"])
    return float(meta.get("elevation", 0.0)) < 0.0


def apply_water_masks(
    rgba: np.ndarray,
    *,
    elevation_tex: np.ndarray | None = None,
    ocean_mask_tex: np.ndarray | None = None,
    lake_mask_tex: np.ndarray | None = None,
) -> np.ndarray:
    """Overwrite ocean and lake pixels in an (N, H, W, 4) RGBA array in-place."""
    if ocean_mask_tex is not None:
        rgba[ocean_mask_tex[..., 0] > 0] = _OCEAN_RGBA
    elif elevation_tex is not None:
        rgba[elevation_tex[..., 0] < 0.0] = _OCEAN_RGBA
    if lake_mask_tex is not None:
        from worldgen.hydrology.builder import LAKE_COLOR
        rgba[lake_mask_tex[..., 0] > 0] = LAKE_COLOR
    return rgba


def ocean_gradient_scheme(meta_key: str, stops: list, default: float = 0.0):
    """Return a gradient scheme function that paints ocean faces with OCEAN_COLOR."""
    base = gradient_scheme(meta_key, stops, default)

    def fn(meta: dict) -> tuple:
        if is_ocean_face(meta):
            return OCEAN_COLOR
        return base(meta)
    return fn


def ocean_categorical_scheme(meta_key: str, palette: np.ndarray, n: int, default: int = 0):
    """Return a categorical scheme function that paints ocean faces with OCEAN_COLOR."""
    base = categorical_scheme(meta_key, palette, n, default)

    def fn(meta: dict) -> tuple:
        if is_ocean_face(meta):
            return OCEAN_COLOR
        return base(meta)
    return fn


def _plate_palette(n: int) -> np.ndarray:
    """Return (n, 4) float32 RGBA colors evenly spaced around the HSV hue wheel."""
    hues = np.linspace(0, 1, n, endpoint=False)
    rgba = np.ones((n, 4), dtype=np.float32)
    for i, h in enumerate(hues):
        r, g, b = colorsys.hsv_to_rgb(h, 0.65, 0.85)
        rgba[i, :3] = r, g, b
    return rgba


def _elevation_color(meta: dict) -> tuple:
    e = meta["elevation"]
    stops = _OCEAN_ELEVATION_STOPS if is_ocean_face(meta) else _LAND_ELEVATION_STOPS
    return interp_stops_scalar(e, stops)


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
        color = interp_stops_scalar(e, _LAND_ELEVATION_STOPS)
        label = f"{e:+.1f} km" if e != 0.0 else "0 km"
        entries.append((label, color))
    return entries


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
    result[is_ocean] = interp_stops(elevs[is_ocean], _OCEAN_ELEVATION_STOPS)
    result[~is_ocean] = interp_stops(elevs[~is_ocean], _LAND_ELEVATION_STOPS)
    return result


def build_elevation_display_texture(mesh) -> np.ndarray:
    """RGBA (N, H, W, 4) elevation texture from the single-channel ``"elevation"``
    data layer, painting ocean pixels by depth and lake pixels in lake blue."""
    elev_tex = mesh.face_textures["elevation"]          # (N, H, W, 1)
    ocean_mask = None
    if "ocean_mask" in mesh.face_textures:
        ocean_mask = mesh.face_textures["ocean_mask"][:, :, :, 0] > 0  # (N, H, W) bool
    rgba = _elevation_color_vec(elev_tex[:, :, :, 0], ocean_mask=ocean_mask)  # (N, H, W, 4)
    return apply_water_masks(rgba, lake_mask_tex=mesh.face_textures.get("lake_mask"))
