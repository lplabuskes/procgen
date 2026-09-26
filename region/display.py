"""RGBA display textures for the region view.

Deliberately thin: the region view offers exactly the world map's layers, under
the same names, coloured by the same code.  A scheme whose texture has a
:class:`~core.display.TextureRecipe` is rebuilt from the region's refined
fields by running that recipe on them — the very function its stage ran on the
globe's face textures.  Every other layer is read straight off the globe, so a
region reads as a zoomed-in continuation of it rather than a different map.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable, Sequence

import numpy as np

from core.display import POSITIONS_KEY, SchemeSpec
from region.sampling import GlobalSampler
from region.upsample import RegionFields

if TYPE_CHECKING:
    from core.mesh import PolygonMesh


def recipe_inputs(fields: RegionFields) -> dict[str, np.ndarray]:
    """What a recipe may read here: the refined textures plus the texel positions.

    Positions let a layer that is a function of place on the sphere (the polity
    fill) be evaluated at the region's own resolution.
    """
    return {**fields.textures, POSITIONS_KEY: fields.positions}


def has_refined_layer(fields: RegionFields, spec: SchemeSpec) -> bool:
    """True when the layer can be drawn from the region's own refined fields."""
    return spec.recipe is not None and spec.recipe.available(recipe_inputs(fields))


def build_layer_texture(
    fields: RegionFields, spec: SchemeSpec, mesh: "PolygonMesh | None" = None
) -> np.ndarray:
    """(M, T, T, 4) float32 RGBA for one world-map layer over a refined region.

    A layer with a recipe whose sources the region has refined is rebuilt from
    them (hillshaded by the refined relief when the recipe asks for it); any
    other layer is sampled from the global map (``mesh`` plus the scheme's
    texture or colour function), so a region offers every world-map layer.
    """
    if has_refined_layer(fields, spec):
        assert spec.recipe is not None
        rgba = np.ascontiguousarray(spec.recipe(recipe_inputs(fields)), dtype=np.float32)
        if spec.recipe.relief and "elevation" in fields.textures:
            rgba = hillshade(
                rgba, fields.textures["elevation"], math.sqrt(max(fields.texel_area_km2, 1e-6)),
                water=_water_mask(fields),
            )
        return rgba
    if mesh is None:
        raise ValueError("layer is not refined and no global mesh was given")
    return sample_global_layer(fields, mesh, scheme_fn=spec.fn, texture_key=spec.texture)


def _water_mask(fields: RegionFields) -> np.ndarray | None:
    """(M, T, T) True on refined ocean or lake texels, which hillshading leaves flat."""
    masks = [fields.textures[k][..., 0] > 0 for k in ("ocean_mask", "lake_mask") if k in fields.textures]
    if not masks:
        return None
    return np.logical_or.reduce(masks)


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


def describe_texel(
    fields: RegionFields, hex_idx: int, row: int, col: int, keys: Sequence[str]
) -> list[str] | None:
    """Status-bar fragments for the refined ``keys`` at one texel.

    Mirrors the world map's hover contributors for those textures (see
    ``DisplayRegistry.add_hover(refined=...)``); ``None`` when any key is not
    refined here, so the caller falls back to the global value.
    """
    tex = fields.textures
    if not keys or any(k not in tex for k in keys):
        return None
    at = (hex_idx, row, col)
    parts: list[str] = []
    if "elevation" in keys:
        parts.append(f"[px: {float(tex['elevation'][at][0]):.2f} km]")
    if "biome" in keys:
        if "ocean_mask" in tex and tex["ocean_mask"][at][0] > 0:
            parts.append("Ocean")
        elif "lake_mask" in tex and tex["lake_mask"][at][0] > 0:
            parts.append("Lake")
        else:
            from worldgen.climate.classification import BIOME_NAMES
            parts.append(BIOME_NAMES[int(tex["biome"][at][0])])
    if "temperature" in keys:
        parts.append(f"T: {float(tex['temperature'][at].mean()):.1f}°C")
    if "precipitation" in keys:
        parts.append(f"P: {float(tex['precipitation'][at].sum()):.0f} mm/yr")
    return parts


def hillshade(
    rgba: np.ndarray,
    elevation: np.ndarray,
    texel_km: float,
    *,
    water: np.ndarray | None = None,
    strength: float = 0.55,
) -> np.ndarray:
    """Relight ``(M, T, T, 4)`` colours by the slope of the refined elevation.

    Without this the added relief is invisible — a biome or elevation ramp at
    ~16 km²/texel simply cannot show a 100 m bump.  Gradients are taken inside
    each hex texture (which over-hangs its hexagon, so the visible area is lit
    from real neighbours) with the light from the north-west, the cartographic
    convention.  Pixels under ``water`` (``(M, T, T)`` bool) are left as they are.
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
    if water is not None:
        shade = np.where(water, 1.0, shade)
    out = rgba.copy()
    out[..., :3] = np.clip(out[..., :3] * np.clip(shade, 0.4, 1.6)[..., None], 0.0, 1.0)
    return out
