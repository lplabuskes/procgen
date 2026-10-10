"""Settlement refinement: local maps' fields refined onto one settlement's frame.

:mod:`localmap.upsample`'s recipe one tier down
(:func:`region.upsample.refine_fields`, with the local maps the frame touches as
parent via :class:`settlegen.source.LocalMapsSource`); downsampling recovers
every local texel to float noise over its in-frame part
(:func:`settlement_conservation_report`).

Defaults: ``texel_m = 15`` against the local map's ~365 m — a ~24× refinement,
~4–5 octaves of detail.  Small lakes are off (the frame has no hex grid; the
local tier already draws lakes down to ~10 km²); cut-off sea is reconnected.

The noise stream is shared by every settlement map of a world, so overlapping
frames share their relief (up to each one's conservation of the texels its
edge cuts).
"""
from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

from core.params import param_field as _f, paramclass
from localmap.upsample import LocalUpsampleParams
from region.sampling import GlobalSampler
from region.upsample import EARTH_RADIUS_KM, RegionFields, anchor_residuals, refine_fields, worked_mine_points
from settlegen.geometry import SettlementGeometry
from settlegen.grid import SettlementGrid
from settlegen.source import LocalMapsSource

if TYPE_CHECKING:
    from core.world import World
    from worldgen.resources.builder import ResourcesParams

_SEED_SALT_SETTLEMENT = 0x5E77


@paramclass
class SettlementUpsampleParams(LocalUpsampleParams):
    """:class:`LocalUpsampleParams` with the settlement tier's defaults (relief not yet tuned).

    Only the fields redeclared here are the settlement tier's own; the rest are
    the local tier's (:data:`SHARED_WITH_LOCAL`, copied down by :func:`with_local_detail`).
    """

    texel_m: float = _f(15.0, label="Texel size (m)", min=2.0, max=100.0, step=1.0)
    # Ground kept around the footprint and garden ring.
    margin_km: float = _f(0.5, label="Frame margin (km)", min=0.0, max=5.0, step=0.1)
    # Local slopes over ~365 m texels run to ~0.1–0.3 km/km in hills; the floor
    # keeps plains to a couple of metres.
    slope_reference: float = _f(
        0.1, label="Slope saturation (km/km)", min=0.001, max=1.0, step=0.001
    )
    slope_floor: float = _f(0.03, label="Flat-ground detail floor", min=0.0, max=1.0)
    max_relief_km: float = _f(0.03, label="Max detail relief (km)", min=0.0, max=1.0, step=0.005)
    # About one local texel (~365 m / 15 m).
    water_spread_texels: int = _f(24, label="Shoreline spread (texels)", min=0, max=128)
    biome_smoothing: int = _f(25, label="Biome climate averaging (texels)", min=0, max=63)
    # Slopes over 15 m texels are steeper again.
    slope_saturation: float = _f(
        0.3, label="Slope saturation, resources (km/km)", min=0.01, max=2.0, step=0.01
    )


# The detail fields the settlement tier takes from the local tier: those it doesn't redeclare.
SHARED_WITH_LOCAL: frozenset[str] = frozenset(
    f.name for f in dataclasses.fields(LocalUpsampleParams)
) - frozenset(vars(SettlementUpsampleParams).get("__annotations__", {}))
# Hidden in the dialog: set once on the local tier (its grid shape, n_cols / tex_size, unused here).
SettlementUpsampleParams.DERIVED = LocalUpsampleParams.DERIVED | SHARED_WITH_LOCAL


def with_local_detail(settlement: SettlementUpsampleParams, local: LocalUpsampleParams) -> SettlementUpsampleParams:
    """``settlement`` with the local tier's values for the shared detail fields."""
    return dataclasses.replace(settlement, **{name: getattr(local, name) for name in SHARED_WITH_LOCAL})


def km_per_local_unit(local: RegionFields) -> float:
    """Ground km per unit of a local map's ``(s, t)`` (the map is one unit across)."""
    texel_units = 2.0 * local.grid.radius / local.tex_size
    return local.grid.texel_spacing_rad(local.tex_size) * EARTH_RADIUS_KM / texel_units


def build_settlement_fields(
    globe: "World",
    maps: Sequence[RegionFields],
    center: np.ndarray,
    side_km: float,
    params: Optional[SettlementUpsampleParams] = None,
    *,
    resources: Optional["ResourcesParams"] = None,
    fertility_exponent: float = 2.0,
) -> RegionFields:
    """Refine the local maps ``maps`` onto a ``side_km`` square centred on ``center`` (unit vector).

    ``maps[0]`` is the owning local map: the frame is laid out in its ``(s, t)``.
    ``maps`` must cover the frame (every point inside some map's parent
    hexagon), or this raises.

    Returns a :class:`RegionFields` on a one-cell :class:`SettlementGrid`, so
    local consumers read it unchanged: ``anchors`` / ``faces`` / ``face_uv`` stay
    global, ``parent_anchors`` holds the local texel each texel conserves (in
    :class:`LocalMapsSource`'s numbering over ``maps``), ``in_map`` is all True.
    """
    params = params or SettlementUpsampleParams()
    owner = maps[0]
    km_per_unit = km_per_local_unit(owner)
    s, t, valid = owner.geometry.ecef_to_local(np.asarray(center, dtype=np.float64).reshape(1, 3))
    if not valid[0]:
        raise ValueError("centre is on the far side of the owning local map's region")
    side = float(side_km) / km_per_unit
    tex_size = max(2, int(math.ceil(float(side_km) * 1000.0 / float(params.texel_m))))
    geometry = SettlementGeometry(owner.geometry, np.array([s[0], t[0]]), side)
    grid = SettlementGrid(geometry)
    positions = grid.texture_positions(tex_size)                 # (1, T, T, 3)
    weights = np.ones((1, tex_size, tex_size), dtype=bool)
    # The parent's texel area scaled by the texels' width ratio, as the local map is to its region.
    local_texel_units = 2.0 * owner.grid.radius / owner.tex_size
    texel_area_km2 = float(owner.texel_area_km2) * (side / tex_size / local_texel_units) ** 2

    source = LocalMapsSource(maps, globe.antimeridian)
    textures, loc, octaves = refine_fields(
        source, positions, weights, grid.texel_spacing_rad(tex_size), texel_area_km2,
        params, int(globe.seed or 0) ^ _SEED_SALT_SETTLEMENT,
        resources=resources, fertility_exponent=fertility_exponent,
        mine_points=worked_mine_points(globe), neighbours=grid.texel_neighbours(tex_size),
    )
    gh, gw = globe.mesh.face_textures["elevation"].shape[1:3]
    sampler = GlobalSampler(globe.mesh)
    faces, u, v = sampler.locate(positions.reshape(-1, 3))
    shape = (1, tex_size, tex_size)
    return RegionFields(
        region_id=owner.region_id,
        geometry=geometry,
        grid=grid,
        tex_size=tex_size,
        positions=positions.astype(np.float32),
        in_hex=weights,
        anchors=sampler.anchors(faces, u, v, gh, gw).reshape(shape).astype(np.int32),
        faces=faces.reshape(shape).astype(np.int32),
        face_uv=np.stack([u, v], axis=-1).reshape(*shape, 2).astype(np.float32),
        textures=textures,
        detail_octaves=octaves,
        texel_area_km2=texel_area_km2,
        parent_anchors=loc.anchors.reshape(shape).astype(np.int32),
        in_map=np.ones(shape, dtype=bool),
    )


def settlement_conservation_report(
    fields: RegionFields, maps: Sequence[RegionFields], antimeridian: float,
) -> dict[str, dict[str, float]]:
    """:func:`localmap.upsample.local_conservation_report` one tier down: per
    local texel (of every map in ``maps``, as passed to
    :func:`build_settlement_fields`) the settlement mean against the local value."""
    if fields.parent_anchors is None:
        raise ValueError("not a settlement map: no parent_anchors")
    parent = LocalMapsSource(maps, antimeridian).parent_textures()
    return anchor_residuals(fields.textures, fields.parent_anchors, fields.domain, parent)
