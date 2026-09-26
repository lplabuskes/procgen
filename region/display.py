"""RGBA display textures for the region view.

Deliberately thin: the region view offers exactly the world map's layers, under
the same names.  Layers the upsampler refines route through the same palettes
the world map uses; every other layer is read straight off the globe, so a
region reads as a zoomed-in continuation of it rather than a different map.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable, Sequence

import numpy as np

from region.sampling import GlobalSampler
from region.upsample import RegionFields

if TYPE_CHECKING:
    from core.mesh import PolygonMesh

# World-map layers the region view renders from its own refined fields — the
# names are the world map's scheme names, so one combo drives both views.  Every
# other scheme is sampled from the global map (``sample_global_layer``).
REFINED_LAYERS: list[str] = [
    "Elevation",
    "Biome",
    "Annual Temperature",
    "Summer Temperature",
    "Winter Temperature",
    "Annual Precipitation",
]

# Season channel per temperature layer (None = annual mean), matching the world
# map's climate display.
_TEMPERATURE_SEASON: dict[str, int | None] = {
    "Annual Temperature": None,
    "Summer Temperature": 1,
    "Winter Temperature": 3,
}

# The texture each refined layer needs; without it the layer is sampled instead.
_REFINED_SOURCE: dict[str, str] = {
    "Elevation": "elevation",
    "Biome": "biome",
    "Annual Temperature": "temperature",
    "Summer Temperature": "temperature",
    "Winter Temperature": "temperature",
    "Annual Precipitation": "precipitation",
}


def has_refined_layer(fields: RegionFields, layer: str) -> bool:
    """True when ``layer`` can be drawn from the region's own refined fields."""
    key = _REFINED_SOURCE.get(layer)
    return key is not None and key in fields.textures


def build_layer_texture(
    fields: RegionFields,
    layer: str,
    mesh: "PolygonMesh | None" = None,
    *,
    scheme_fn: "Callable[[dict], Sequence[float]] | None" = None,
    texture_key: str | None = None,
) -> np.ndarray:
    """(M, T, T, 4) float32 RGBA for one world-map layer over a refined region.

    Layers in ``REFINED_LAYERS`` are drawn from the refined fields; any other
    layer is sampled from the global map (``mesh`` plus the scheme's
    ``texture_key`` / ``scheme_fn``), so a region offers every world-map layer.
    """
    if has_refined_layer(fields, layer):
        return _refined_layer_texture(fields, layer)
    if mesh is None:
        raise ValueError(f"layer {layer!r} is not refined and no global mesh was given")
    return sample_global_layer(fields, mesh, scheme_fn=scheme_fn, texture_key=texture_key)


def sample_global_layer(
    fields: RegionFields,
    mesh: "PolygonMesh",
    *,
    scheme_fn: "Callable[[dict], Sequence[float]] | None" = None,
    texture_key: str | None = None,
) -> np.ndarray:
    """(M, T, T, 4) RGBA reading a world-map layer exactly as the globe draws it.

    A per-face RGBA texture is read at each texel's anchor (nearest global
    texel); otherwise the scheme colours each texel's owning face flat — the
    same two rendering paths the world map uses.  No detail is invented.
    """
    shape = fields.faces.shape
    if texture_key is not None and texture_key in mesh.face_textures:
        rgba = GlobalSampler.anchor_values(mesh.face_textures[texture_key], fields.anchors.reshape(-1))
        return np.ascontiguousarray(rgba.reshape(*shape, -1), dtype=np.float32)
    if scheme_fn is None:
        raise ValueError("layer has neither a face texture nor a color scheme")
    faces = fields.faces.reshape(-1)
    unique, inverse = np.unique(faces, return_inverse=True)
    colors = np.array(
        [np.asarray(scheme_fn(mesh.face_metadata[int(f)]), dtype=np.float32) for f in unique],
        dtype=np.float32,
    ).reshape(len(unique), -1)
    if colors.shape[1] == 3:
        colors = np.concatenate([colors, np.ones((len(unique), 1), dtype=np.float32)], axis=1)
    return np.ascontiguousarray(colors[inverse].reshape(*shape, 4))


def _refined_layer_texture(fields: RegionFields, layer: str) -> np.ndarray:
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

    if layer == "Elevation":
        assert elevation is not None
        mask = ocean_mask[..., 0] > 0 if ocean_mask is not None else None
        rgba = elevation_rgba(elevation[..., 0], ocean_mask=mask)
    elif layer in _TEMPERATURE_SEASON:
        rgba = build_temperature_display_texture(
            tex["temperature"], season=_TEMPERATURE_SEASON[layer],
            elevation_tex=elevation, ocean_mask_tex=ocean_mask,
        )
    elif layer == "Annual Precipitation":
        rgba = build_precipitation_display_texture(
            tex["precipitation"], elevation_tex=elevation, ocean_mask_tex=ocean_mask
        )
    elif layer == "Biome":
        rgba = categorical_texture(tex["biome"], BIOME_PALETTE, len(BIOME_PALETTE))
    else:
        raise ValueError(f"unknown refined layer {layer!r}")

    rgba = np.ascontiguousarray(rgba, dtype=np.float32)
    if layer in ("Biome", "Elevation") and elevation is not None:
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
