"""Local refinement: a regional hex's fields refined onto its local map.

:mod:`region.upsample`'s recipe one tier down (:func:`region.upsample.refine_fields`
with the region as parent via :class:`region.source.RegionSource`); downsampling
recovers the region's texels to float noise (:func:`local_conservation_report`).

Defaults (``n_cols=16``, ``tex_size=64``): local hex ~23 km, texel ~365 m vs the
region's ~4.5 km — a 12× refinement, ~3 octaves of detail.

The noise stream is shared by every local map of a world (as the region tier's
is), and each map conserves over a halo past its parent hexagon
(``conservation_halo_parent_texels``), so neighbouring local maps agree at their
seam: each conserves the regional texels its edge cuts over their whole area,
not just its own part of them, and the smooth correction is the same on both
sides.  (Over its own part alone, such a texel is no longer exact.)
"""
from __future__ import annotations

import dataclasses
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
    """:class:`RegionUpsampleParams` with the local tier's defaults (the relief spectrum is the region's).

    Only the fields redeclared here are the local tier's own; the rest are the
    region's (:data:`SHARED_WITH_REGION`, copied down by :func:`with_region_detail`).
    """

    n_cols: int = _f(16, label="Hex columns per local map", min=4, max=64)
    tex_size: int = _f(64, label="Hex texture size", min=4, max=128)
    # About one regional texel (~4.5 km / ~365 m).
    water_spread_texels: int = _f(12, label="Shoreline spread (texels)", min=0, max=32)
    # ~One regional texel, so biome edges follow the region rather than local speckle.
    biome_smoothing: int = _f(13, label="Biome climate averaging (texels)", min=0, max=31)
    # Lower again than the region's (0.4): fens in hollows, marsh strips along streams.
    wetland_threshold: float = _f(0.3, label="Wetland threshold", min=0.0, max=1.0, step=0.05)
    wetland_relief_gain: float = _f(0.08, label="Wetland relief swing", min=0.0, max=0.5, step=0.01)
    wetland_relief_scale_m: float = _f(3.0, label="Wetland relief scale (m)", min=0.5, max=200.0, step=0.5)
    wetland_riparian_width_km: float = _f(1.5, label="Riparian width at ref flow (km)", min=0.0, max=50.0, step=0.1)
    wetland_riparian_slope_m_per_km: float = _f(
        8.0, label="Riparian flatness slope scale (m/km)", min=0.1, max=50.0, step=0.1)
    # ~7 km, near the region's ~15 km.  Uncapped this floods whole districts (L=7
    # was needed); capped, it only sets how many sub-texel lakes (seed 4: 0.27% of land).
    lake_breach_parent_texels: float = _f(
        1.5, label="Small lake breach length (parent texels)", min=0.0, max=30.0, step=0.5
    )
    # Only lakes below the region's resolution (~20 km²): bigger basins the region left dry stay dry.
    lake_max_parent_texels: float = _f(
        1.0, label="Small lake max size (parent texels, 0 = none)", min=0.0, max=50.0, step=0.5
    )
    # ~3 local texels (~1 km): rounds the conservation staircase's parent-texel pits off.
    lake_smoothing_parent_texels: float = _f(
        0.25, label="Small lake smoothing (parent texels)", min=0.0, max=3.0, step=0.05
    )
    # Local slopes over ~365 m texels are far steeper than the region's.
    slope_saturation: float = _f(
        0.1, label="Slope saturation, resources (km/km)", min=0.002, max=0.5, step=0.002
    )
    # Conserving only over the parent hexagon, neighbouring maps' corrections fade out
    # differently at their seam (seed 4: 5.5 m mean / 24 m max disagreement); over ~2
    # regional texels past it, they agree to ~0.5 m.  0: conserve over the hexagon only.
    conservation_halo_parent_texels: float = _f(
        2.0, label="Conservation halo past the hexagon (regional texels)", min=0.0, max=4.0, step=0.5
    )


# The detail fields the local tier takes from the region: those it doesn't redeclare.
SHARED_WITH_REGION: frozenset[str] = frozenset(
    f.name for f in dataclasses.fields(RegionUpsampleParams)
) - frozenset(vars(LocalUpsampleParams).get("__annotations__", {}))
LocalUpsampleParams.DERIVED = RegionUpsampleParams.DERIVED | SHARED_WITH_REGION


def with_region_detail(local: LocalUpsampleParams, region: RegionUpsampleParams) -> LocalUpsampleParams:
    """``local`` with the region's values for the shared detail fields."""
    return dataclasses.replace(local, **{name: getattr(region, name) for name in SHARED_WITH_REGION})


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
    # The halo in local units: a regional texel is 1 / T_region of the local map's width.
    halo = float(params.conservation_halo_parent_texels) / float(region.tex_size)
    grid = LocalHexGrid(geometry, int(params.n_cols), halo=halo)
    positions = grid.texture_positions(tex_size)                 # (M, T, T, 3)
    in_hex = grid.texture_in_hexagon(tex_size)
    in_map = grid.texture_in_map(tex_size)
    domain = in_hex & in_map
    conserved = in_hex & grid.texture_in_halo(tex_size)
    m = len(grid)

    parent_area_km2 = region.texel_area_km2 * float(region.in_hex[hex_index].sum())
    texel_area_km2 = parent_area_km2 / max(int(domain.sum()), 1)

    textures, loc, octaves = refine_fields(
        RegionSource(region), positions, conserved, grid.texel_spacing_rad(tex_size), texel_area_km2,
        params, int(globe.seed or 0) ^ _SEED_SALT_LOCAL,
        resources=resources, fertility_exponent=fertility_exponent,
        mine_points=worked_mine_points(globe), neighbours=grid.texel_neighbours(tex_size, in_map),
        grid=grid, drains=drains,
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
        positions=positions.astype(np.float32),
        in_hex=in_hex,
        anchors=sampler.anchors(faces, u, v, gh, gw).reshape(shape).astype(np.int32),
        faces=faces.reshape(shape).astype(np.int32),
        face_uv=np.stack([u, v], axis=-1).reshape(*shape, 2).astype(np.float32),
        textures=textures,
        detail_octaves=octaves,
        texel_area_km2=texel_area_km2,
        parent_anchors=loc.anchors.reshape(shape).astype(np.int32),
        in_map=in_map,
        conserved=conserved,
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
    texel the map conserves, the local mean over the texels it conserves it with
    (the parent hexagon and its halo) against the regional value."""
    if local.parent_anchors is None:
        raise ValueError("not a local map: no parent_anchors")
    weights = local.conserved if local.conserved is not None else local.domain
    return anchor_residuals(local.textures, local.parent_anchors, weights, region.textures)
