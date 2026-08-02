"""RGBA display textures for the region view.

Deliberately thin: every layer routes through the same palettes the world map
uses, so a region reads as a zoomed-in continuation of the globe rather than a
differently-coloured map.
"""
from __future__ import annotations

import math

import numpy as np

from region.upsample import RegionFields

# Layer keys in the order the region view offers them.
REGION_LAYERS: list[str] = ["Terrain", "Elevation", "Temperature", "Precipitation"]


def build_layer_texture(fields: RegionFields, layer: str) -> np.ndarray:
    """(M, T, T, 4) float32 RGBA for one named layer of a refined region."""
    from viz.color_stops import categorical_texture
    from worldgen.climate.display import (
        BIOME_PALETTE,
        build_precipitation_display_texture,
        build_temperature_display_texture,
    )
    from worldgen.geology.display import apply_water_masks, elevation_rgba

    tex = fields.textures
    elevation = tex.get("elevation")
    ocean_mask = tex.get("ocean_mask")
    lake_mask = tex.get("lake_mask")

    if layer == "Elevation" or "biome" not in tex:
        if elevation is None:
            raise KeyError("region has no elevation texture")
        mask = ocean_mask[..., 0] > 0 if ocean_mask is not None else None
        rgba = elevation_rgba(elevation[..., 0], ocean_mask=mask)
    elif layer == "Temperature":
        rgba = build_temperature_display_texture(
            tex["temperature"], elevation_tex=elevation, ocean_mask_tex=ocean_mask
        )
    elif layer == "Precipitation":
        rgba = build_precipitation_display_texture(
            tex["precipitation"], elevation_tex=elevation, ocean_mask_tex=ocean_mask
        )
    elif layer == "Terrain":
        rgba = categorical_texture(tex["biome"], BIOME_PALETTE, len(BIOME_PALETTE))
    else:
        raise ValueError(f"unknown region layer {layer!r}")

    rgba = np.ascontiguousarray(rgba, dtype=np.float32)
    if layer in ("Terrain", "Elevation") and elevation is not None:
        rgba = hillshade(rgba, elevation, math.sqrt(max(fields.texel_area_km2, 1e-6)))
    return apply_water_masks(
        rgba,
        elevation_tex=elevation if ocean_mask is None else None,
        ocean_mask_tex=ocean_mask,
        lake_mask_tex=lake_mask,
    )


def hillshade(
    rgba: np.ndarray,
    elevation: np.ndarray,
    texel_km: float,
    *,
    strength: float = 0.55,
) -> np.ndarray:
    """Relight ``(M, T, T, 4)`` colours by the slope of the refined elevation.

    Without this the added relief is invisible — a biome or elevation ramp at
    ~16 km²/texel simply cannot show a 100 m bump.  Gradients are taken inside
    each hex texture (which over-hangs its hexagon, so the visible area is lit
    from real neighbours) with the light from the north-west, the cartographic
    convention.
    """
    elev = np.asarray(elevation, dtype=np.float32)[..., 0]
    dz_ds = np.gradient(elev, axis=2) / texel_km     # east-west, columns
    dz_dt = -np.gradient(elev, axis=1) / texel_km    # north-south (row 0 is north)
    normal = np.stack([-dz_ds, -dz_dt, np.ones_like(elev)], axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    light = np.array([-0.5, 0.5, 0.707], dtype=np.float32)
    light /= np.linalg.norm(light)
    intensity = np.clip(normal @ light, 0.0, 1.0)
    shade = (1.0 - strength) + strength * (intensity / 0.707)
    out = rgba.copy()
    out[..., :3] = np.clip(out[..., :3] * np.clip(shade, 0.4, 1.6)[..., None], 0.0, 1.0)
    return out
