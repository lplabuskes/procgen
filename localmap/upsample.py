"""Local refinement: a regional hex's fields refined onto its local map.

The same recipe as :mod:`region.upsample` one tier down — bilinear read of the
parent, detail above the parent grid's Nyquist, per-parent-texel conservation —
run by :func:`region.upsample.refine_fields` with the region as the parent
(:class:`region.source.RegionSource`).  Downsampling a local map by averaging
recovers the region's texels to float noise (:func:`local_conservation_report`).

At the defaults (``n_cols=16``, ``tex_size=64``) a local hex is ~23 km across
and a texel ~365 m, against the region's ~4.5 km: a 12× refinement, which buys
the detail band ~3 octaves.

The noise stream is shared by every local map of a world, so two neighbouring
local maps of one region agree where they meet (up to their parents'
per-texel corrections); regions, seeded per region, do not.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from core.params import param_field as _f, paramclass
from localmap.geometry import LocalGeometry
from localmap.grid import LocalHexGrid
from region.sampling import GlobalSampler
from region.source import RegionSource
from region.upsample import RegionFields, RegionUpsampleParams, anchor_residuals, refine_fields

if TYPE_CHECKING:
    from core.world import World
    from worldgen.resources.builder import ResourcesParams

_SEED_SALT_LOCAL = 0x10CA


@paramclass
class LocalUpsampleParams(RegionUpsampleParams):
    """:class:`RegionUpsampleParams` with the local tier's defaults.

    The relief numbers are a first cut, not yet tuned against real terrain.
    """

    n_cols: int = _f(16, label="Hex columns per local map", min=4, max=64)
    tex_size: int = _f(64, label="Hex texture size", min=4, max=128)
    # The parent is the region, whose sub-grid slopes run ~0.003–0.035 km/km
    # over a ~9 km detail wavelength: saturating at 0.03 gives mountains a few
    # hundred metres of local relief, and the lower floor keeps plains to ~15 m.
    slope_reference: float = _f(
        0.03, label="Slope saturation (km/km)", min=0.0005, max=0.1, step=0.0005
    )
    slope_floor: float = _f(0.03, label="Flat-ground detail floor", min=0.0, max=1.0)
    max_relief_km: float = _f(0.6, label="Max detail relief (km)", min=0.0, max=5.0)
    # About one regional texel (~4.5 km / ~365 m).
    water_spread_texels: int = _f(12, label="Shoreline spread (texels)", min=0, max=32)
    # The per-regional-texel conservation step leaves ~4.5 km blocks in the
    # climate; the classifiers average over about one of them.
    biome_smoothing: int = _f(13, label="Biome climate averaging (texels)", min=0, max=31)
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
) -> RegionFields:
    """Refine regional hex ``hex_index`` of ``region`` onto its local map.

    Returns a :class:`RegionFields` so every consumer of a region (display,
    hover, corridors) reads a local map the same way: ``anchors`` / ``faces`` /
    ``face_uv`` stay *global*, ``parent_anchors`` holds the regional texel each
    local texel conserves, and ``in_map`` marks the parent hexagon.
    ``region_id`` and ``geometry.region_id`` are the parent region's.
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


def local_conservation_report(local: RegionFields, region: RegionFields) -> dict[str, dict[str, float]]:
    """Does downsampling the local map recover its region's texels?

    :func:`region.upsample.conservation_report` one tier down: per regional
    texel inside the parent hexagon, the local mean against the regional value.
    """
    if local.parent_anchors is None:
        raise ValueError("not a local map: no parent_anchors")
    return anchor_residuals(local.textures, local.parent_anchors, local.domain, region.textures)
