from __future__ import annotations

import colorsys
from typing import Callable

import numpy as np

from worldgen.geology.elevation_builder import ElevationBuilder
from worldgen.mapview.mapview import MapWindow
from worldgen.mesh.mymesh import PolygonMesh


OCEAN_COLOR = (0.25, 0.55, 0.85, 1.0)
LAND_COLOR = (0.72, 0.52, 0.25, 1.0)
BOUNDARY_COLORS = {
    "interior":   (0.70, 0.70, 0.70, 1.0),
    "convergent": (0.85, 0.30, 0.10, 1.0),
    "divergent":  (0.95, 0.90, 0.30, 1.0),
    "transform":  (0.60, 0.20, 0.80, 1.0),
}

# Elevation color stops: (elevation_km, RGBA)
# Two stops at 0.0 create a hard break between ocean (blue) and land (sand/green).
_ELEVATION_STOPS = [
    (-8.0, (0.02, 0.06, 0.35, 1.0)),  # abyssal plain
    (-3.5, (0.06, 0.22, 0.62, 1.0)),  # deep ocean
    (-0.5, (0.16, 0.48, 0.82, 1.0)),  # continental shelf
    (0.0, (0.20, 0.52, 0.88, 1.0)),  # sea level (ocean side)
    (0.0, (0.78, 0.70, 0.42, 1.0)),  # sea level (land side — beach/sand)
    (0.8, (0.33, 0.60, 0.20, 1.0)),  # lowland / plains
    (2.5, (0.50, 0.38, 0.18, 1.0)),  # highland
    (5.5, (0.92, 0.92, 0.92, 1.0)),  # snow-capped peaks
]


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
    stops = _ELEVATION_STOPS
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


def _elevation_color_vec(elevs: np.ndarray) -> np.ndarray:
    """Vectorized elevation color scheme. Returns (..., 4) float32."""
    shape = elevs.shape
    result = np.full((*shape, 4), _ELEVATION_STOPS[-1][1], dtype=np.float32)
    for i in range(len(_ELEVATION_STOPS) - 2, -1, -1):
        e0, c0 = _ELEVATION_STOPS[i]
        e1, c1 = _ELEVATION_STOPS[i + 1]
        if e0 == e1:
            continue
        mask = (elevs >= e0) & (elevs <= e1)
        t = np.where(mask, (elevs - e0) / (e1 - e0), 0.0)
        for ch in range(4):
            result[..., ch] = np.where(mask, c0[ch] + t * (c1[ch] - c0[ch]), result[..., ch])
    result[elevs <= _ELEVATION_STOPS[0][0]] = np.array(_ELEVATION_STOPS[0][1], dtype=np.float32)
    result[elevs >= _ELEVATION_STOPS[-1][0]] = np.array(_ELEVATION_STOPS[-1][1], dtype=np.float32)
    return result


def build_elevation_display_texture(mesh: PolygonMesh) -> np.ndarray:
    """Convert elevation_delta face textures to smooth RGBA elevation colors."""
    delta_tex = mesh.face_textures["elevation_delta"]          # (N, H, W, 4)
    face_elevs = np.array([m["elevation"] for m in mesh.face_metadata], dtype=np.float32)
    elev_map = face_elevs[:, np.newaxis, np.newaxis] + delta_tex[:, :, :, 0]  # (N, H, W)
    return _elevation_color_vec(elev_map)                      # (N, H, W, 4)


def apply_geology_display(window: MapWindow, builder: ElevationBuilder) -> None:
    """Wire elevation-textured display and all geology color schemes into the window."""
    mesh = builder.mesh
    palette = _plate_palette(builder.params.n_plates)
    max_speed = builder.max_speed

    mesh.set_face_colors(palette[np.array(builder.plate_ids, dtype=np.int32)])

    elevation_rgba = build_elevation_display_texture(mesh)
    mesh.set_face_textures("elevation_rgba", elevation_rgba)

    def _speed_color(meta: dict) -> tuple:
        t = meta["surface_speed"] / max_speed
        return (t, 0.2 + 0.3 * (1.0 - t), 1.0 - t, 1.0)

    def _volcanic_color(meta: dict) -> tuple:
        t = meta["volcanic_activity"]
        if t <= 0.0:
            return (0.35, 0.35, 0.35, 1.0)
        elif t < 0.5:
            s = t / 0.5
            return (0.35 + 0.60 * s, 0.35 - 0.10 * s, 0.35 - 0.35 * s, 1.0)
        else:
            s = (t - 0.5) / 0.5
            return (0.95, 0.25 - 0.20 * s, 0.0, 1.0)

    schemes: dict[str, Callable[[dict], tuple]] = {
        "Elevation": _elevation_color,
        "Plates": lambda meta: tuple(palette[meta["plate_id"]]),
        "Plate Type": lambda meta: OCEAN_COLOR if meta["plate_type"] == "oceanic" else LAND_COLOR,
        "Speed": _speed_color,
        "Boundary": lambda meta: BOUNDARY_COLORS[meta["boundary_type"]],
        "Volcanic Activity": _volcanic_color,
    }

    window.set_color_schemes(schemes)
    window.associate_scheme_texture("Elevation", "elevation_rgba")
