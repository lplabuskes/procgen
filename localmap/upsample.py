"""Local refinement: a regional hex's fields refined onto its local map.

:mod:`region.upsample`'s recipe one tier down (:func:`region.upsample.refine_fields`
with the region as parent via :class:`region.source.RegionSource`); downsampling
recovers the region's texels to float noise (:func:`local_conservation_report`).

Defaults (``n_cols=16``, ``tex_size=64``): local hex ~23 km, texel ~365 m vs the
region's ~4.5 km — a 12× refinement, ~3 octaves of detail.

The noise stream is shared by every local map of a world, so neighbouring local
maps of one region agree at their seam; regions, seeded per region, do not.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

from core.params import param_field as _f, paramclass
from localmap.geometry import LocalGeometry
from localmap.grid import LocalHexGrid
from region.rivers import RiverPath
from region.sampling import GlobalSampler
from region.source import RegionSource
from region.upsample import RegionFields, RegionUpsampleParams, anchor_residuals, refine_fields, worked_mine_points

if TYPE_CHECKING:
    from core.world import World
    from worldgen.resources.builder import ResourcesParams

_SEED_SALT_LOCAL = 0x10CA


@paramclass
class LocalUpsampleParams(RegionUpsampleParams):
    """:class:`RegionUpsampleParams` with the local tier's defaults (relief not yet tuned)."""

    n_cols: int = _f(16, label="Hex columns per local map", min=4, max=64)
    tex_size: int = _f(64, label="Hex texture size", min=4, max=128)
    # Regional sub-grid slopes run ~0.003–0.035 km/km: saturating at 0.03 gives
    # mountains a few hundred metres of local relief; the floor keeps plains to ~15 m.
    slope_reference: float = _f(
        0.03, label="Slope saturation (km/km)", min=0.0005, max=0.1, step=0.0005
    )
    slope_floor: float = _f(0.03, label="Flat-ground detail floor", min=0.0, max=1.0)
    max_relief_km: float = _f(0.6, label="Max detail relief (km)", min=0.0, max=5.0)
    # About one regional texel (~4.5 km / ~365 m).
    water_spread_texels: int = _f(12, label="Shoreline spread (texels)", min=0, max=32)
    # ~One regional texel, so biome edges follow the region rather than local speckle.
    biome_smoothing: int = _f(13, label="Biome climate averaging (texels)", min=0, max=31)
    # The region's refined slopes are steeper and local relief pits deeper; at the
    # region's 6 local maps hold ~5× Earth's share of 0.1–40 km² lakes.
    lake_breach_parent_texels: float = _f(
        10.0, label="Small lake breach length (parent texels)", min=0.0, max=30.0, step=0.5
    )
    # Local slopes over ~365 m texels are far steeper than the region's.
    slope_saturation: float = _f(
        0.1, label="Slope saturation, resources (km/km)", min=0.002, max=0.5, step=0.002
    )


def build_local_fields(
    globe: "World",
    region: RegionFields,
    hex_index: int,
    params: Optional[LocalUpsampleParams] = None,
    *,
    resources: Optional["ResourcesParams"] = None,
    fertility_exponent: float = 2.0,
    drains: Optional[np.ndarray] = None,
) -> RegionFields:
    """Refine regional hex ``hex_index`` of ``region`` onto its local map.

    ``drains`` (``(K, 2, 3)`` unit-vector segments; :func:`path_segments` of the
    region's rivers) drain any basin they cross, so no small lake forms on them.

    Returns a :class:`RegionFields` so region consumers read it unchanged:
    ``anchors`` / ``faces`` / ``face_uv`` stay global, ``parent_anchors`` holds
    the regional texel each local texel conserves, ``in_map`` marks the parent
    hexagon, and ``region_id`` is the parent region's.
    """
    params = params or LocalUpsampleParams()
    tex_size = int(params.tex_size)
    geometry = LocalGeometry(region.geometry, region.grid, hex_index)
    grid = LocalHexGrid(geometry, int(params.n_cols))
    positions = grid.texture_positions(tex_size)                 # (M, T, T, 3)
    in_hex = grid.texture_in_hexagon(tex_size)
    in_map = grid.texture_in_map(tex_size)
    domain = in_hex & in_map
    m = len(grid)

    parent_area_km2 = region.texel_area_km2 * float(region.in_hex[hex_index].sum())
    texel_area_km2 = parent_area_km2 / max(int(domain.sum()), 1)

    textures, loc, octaves = refine_fields(
        RegionSource(region), positions, domain, grid.texel_spacing_rad(tex_size), texel_area_km2,
        params, int(globe.seed or 0) ^ _SEED_SALT_LOCAL,
        resources=resources, fertility_exponent=fertility_exponent,
        mine_points=worked_mine_points(globe), grid=grid, drains=drains,
    )
    gh, gw = globe.mesh.face_textures["elevation"].shape[1:3]
    sampler = GlobalSampler(globe.mesh)
    faces, u, v = sampler.locate(positions.reshape(-1, 3))
    shape = (m, tex_size, tex_size)
    return RegionFields(
        region_id=region.region_id,
        geometry=geometry,
        grid=grid,
        tex_size=tex_size,
        positions=positions,
        in_hex=in_hex,
        anchors=sampler.anchors(faces, u, v, gh, gw).reshape(shape),
        faces=faces.reshape(shape),
        face_uv=np.stack([u, v], axis=-1).reshape(*shape, 2),
        textures=textures,
        detail_octaves=octaves,
        texel_area_km2=texel_area_km2,
        parent_anchors=loc.anchors.reshape(shape),
        in_map=in_map,
    )


def path_segments(paths: Sequence[RiverPath]) -> np.ndarray:
    """``(K, 2, 3)`` consecutive-vertex segments of ``(positions, flows, in_water)`` polylines."""
    segs = []
    for positions, _, _ in paths:
        p = np.asarray(positions, dtype=np.float64)
        if len(p) >= 2:
            segs.append(np.stack([p[:-1], p[1:]], axis=1))
    return np.concatenate(segs) if segs else np.zeros((0, 2, 3))


def local_conservation_report(local: RegionFields, region: RegionFields) -> dict[str, dict[str, float]]:
    """:func:`region.upsample.conservation_report` one tier down: per regional
    texel in the parent hexagon, the local mean against the regional value."""
    if local.parent_anchors is None:
        raise ValueError("not a local map: no parent_anchors")
    return anchor_residuals(local.textures, local.parent_anchors, local.domain, region.textures)
