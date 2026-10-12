"""Settlement refinement: local maps' terrain refined onto one settlement's frame.

Only the relief is refined — :mod:`localmap.upsample`'s recipe one tier down,
stopped after the terrain (:func:`region.upsample.refine_terrain`, with the local
maps the frame touches as parent via :class:`settlegen.source.LocalMapsSource`):
elevation, the water masks and shoreline, coast style and delta.  Downsampling
recovers every local texel's elevation to float noise over its in-frame part
(:func:`settlement_conservation_report`).

Everything else (climate, biome, wetland, resources, rural density, danger) is
**one value for the frame**: the local map's at the settlement, broadcast
(read-only, no copy) so local consumers read the textures unchanged.  A town or
city spans a few km; its climate and soils don't vary on that scale in any way
the map uses.  The small-lake flag is the exception: it follows the refined lakes.

Defaults: ``texel_m = 15`` against the local map's ~365 m — a ~24× refinement,
~4–5 octaves of detail.  No new small lakes (the frame has no hex grid; the
local tier already draws lakes down to ~10 km²); cut-off sea is reconnected.
Where two of a local map's sea texels touch only at a corner, across two land
texels (a strait one texel wide running on diagonally), the sea wins: the frame
cuts a ``saddle_channel_m`` channel between them (:func:`join_ocean_saddles`),
so the strait stays open and the land beyond stays an island, where smooth
interpolation would close it.
Land narrower than ``min_land_m`` between waters (a spit, a sliver of an islet:
on a flat shore the refined relief sits within centimetres of the water and
pokes through in one-texel ridges) goes to the water nearest it
(:func:`clear_thin_land`).  The masks aren't conserved, only the elevation.

The noise stream is shared by every settlement map of a world, so overlapping
frames share their relief (up to each one's conservation of the texels its
edge cuts).
"""
from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np
from scipy.ndimage import binary_opening, distance_transform_edt

from core.params import param_field as _f, paramclass
from localmap.upsample import LocalUpsampleParams
from region.sampling import GlobalSampler
from region.upsample import EARTH_RADIUS_KM, SMALL_LAKE_KEY, RegionFields, anchor_residuals, refine_terrain
from settlegen.frame import Frame
from settlegen.geometry import SettlementGeometry
from settlegen.grid import SettlementGrid
from settlegen.source import LocalMapsSource

if TYPE_CHECKING:
    from core.world import World

_SEED_SALT_SETTLEMENT = 0x5E77


@paramclass
class SettlementUpsampleParams(LocalUpsampleParams):
    """:class:`LocalUpsampleParams` with the settlement tier's defaults (the relief spectrum is the region's).

    Only the fields redeclared here are the settlement tier's own; the rest are
    the local tier's (:data:`SHARED_WITH_LOCAL`, copied down by :func:`with_local_detail`).
    """

    texel_m: float = _f(15.0, label="Texel size (m)", min=2.0, max=100.0, step=1.0)
    # Ground kept around the footprint and garden ring.
    margin_km: float = _f(0.5, label="Frame margin (km)", min=0.0, max=5.0, step=0.1)
    # About one local texel (~365 m / 15 m).
    water_spread_texels: int = _f(24, label="Shoreline spread (texels)", min=0, max=128)
    min_land_m: float = _f(75.0, label="Narrowest land between waters (m)", min=0.0, max=500.0, step=5.0)
    saddle_channel_m: float = _f(100.0, label="Channel across a diagonal sea saddle (m)", min=0.0, max=500.0,
                                 step=5.0)


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


def ocean_saddles(maps: Sequence[RegionFields]) -> np.ndarray:
    """``(K, 2, 3)`` pairs of sea texel centres (unit vectors) that touch only at a
    corner across two dry texels, within one hexagon's texture of any of ``maps``."""
    out = []
    for f in maps:
        if "ocean_mask" not in f.textures:
            continue
        sea = f.textures["ocean_mask"][..., 0] > 0.5
        lake = f.textures["lake_mask"][..., 0] > 0.5 if "lake_mask" in f.textures else np.zeros_like(sea)
        dry = ~sea & ~lake
        drawn = f.in_hex
        pos = f.positions
        a, b = (slice(None), slice(None, -1), slice(None, -1)), (slice(None), slice(1, None), slice(1, None))
        c, d = (slice(None), slice(1, None), slice(None, -1)), (slice(None), slice(None, -1), slice(1, None))
        for (p, q), (u, v) in (((a, b), (c, d)), ((c, d), (a, b))):
            hit = sea[p] & sea[q] & dry[u] & dry[v] & drawn[p] & drawn[q]
            if hit.any():
                out.append(np.stack([pos[p][hit], pos[q][hit]], axis=1))
    return np.concatenate(out).astype(np.float64) if out else np.zeros((0, 2, 3))


def join_ocean_saddles(textures: dict, frame: Frame, maps: Sequence[RegionFields], width_m: float) -> None:
    """In place: sea along a ``width_m`` channel between each pair of :func:`ocean_saddles`
    in the frame (masks only, as :func:`clear_thin_land`)."""
    ocean = textures.get("ocean_mask")
    if ocean is None or width_m <= 0.0:
        return
    pairs = ocean_saddles(maps)
    if not len(pairs):
        return
    ends = frame.to_xy(pairs.reshape(-1, 3)).reshape(-1, 2, 2)
    tex = ocean.shape[1]
    texel = frame.side_m / tex
    keep = (np.abs(ends.mean(axis=1)) < frame.half + width_m).all(axis=1)
    sea = ocean[0, ..., 0] > 0.5
    lake = textures.get("lake_mask")
    wet_lake = lake[0, ..., 0] > 0.5 if lake is not None else np.zeros_like(sea)
    half_w = 0.5 * float(width_m)
    for p0, p1 in ends[keep]:
        lo, hi = np.minimum(p0, p1) - half_w, np.maximum(p0, p1) + half_w
        # Texel columns run with x, rows down from the top (largest y).
        c0, c1 = (np.clip(np.floor((np.r_[lo[0], hi[0]] + frame.half) / texel), 0, tex - 1)).astype(int)
        r0, r1 = (np.clip(np.floor((frame.half - np.r_[hi[1], lo[1]]) / texel), 0, tex - 1)).astype(int)
        cols = (np.arange(c0, c1 + 1) + 0.5) * texel - frame.half
        rows = frame.half - (np.arange(r0, r1 + 1) + 0.5) * texel
        xy = np.stack(np.meshgrid(cols, rows), axis=-1)
        seg = p1 - p0
        t = np.clip(((xy - p0) @ seg) / max(float(seg @ seg), 1e-12), 0.0, 1.0)
        near = np.linalg.norm(xy - (p0 + t[..., None] * seg), axis=-1) <= half_w
        sea[r0:r1 + 1, c0:c1 + 1] |= near & ~wet_lake[r0:r1 + 1, c0:c1 + 1]
    textures["ocean_mask"] = sea[None, ..., None].astype(ocean.dtype)


def clear_thin_land(textures: dict, texel_m: float, min_land_m: float) -> None:
    """In place: land (neither sea nor lake) the opening by a disc ``min_land_m``
    across removes goes to its nearest water's mask.  The frame's edge doesn't
    erode it (the land is extended past it)."""
    ocean, lake = textures.get("ocean_mask"), textures.get("lake_mask")
    if ocean is None:
        return
    r = int(np.ceil((float(min_land_m) / float(texel_m) - 1.0) / 2.0))
    if r < 1:
        return
    sea = ocean[0, ..., 0] > 0.5
    wet_lake = lake[0, ..., 0] > 0.5 if lake is not None else np.zeros_like(sea)
    wet = sea | wet_lake
    if not wet.any() or wet.all():
        return
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    disc = xx ** 2 + yy ** 2 <= r * r + r
    land = np.pad(~wet, r, mode="edge")
    opened = np.asarray(binary_opening(land, disc), dtype=bool)
    thin = (land & ~opened)[r:-r, r:-r]
    if not thin.any():
        return
    near = np.asarray(distance_transform_edt(~wet, return_distances=False, return_indices=True))
    to_sea = sea[near[0], near[1]]
    new_sea = sea | (thin & to_sea)
    textures["ocean_mask"] = new_sea[None, ..., None].astype(ocean.dtype)
    if lake is not None:
        textures["lake_mask"] = (wet_lake | (thin & ~to_sea))[None, ..., None].astype(lake.dtype)
        filled = textures.get("filled_height")
        if filled is not None:
            # The lake's surface carried onto the land it takes.
            f = filled[0, ..., 0]
            textures["filled_height"] = np.where(thin & ~to_sea, f[near[0], near[1]], f)[None, ..., None].astype(
                filled.dtype)


def build_settlement_fields(
    globe: "World",
    maps: Sequence[RegionFields],
    center: np.ndarray,
    side_km: float,
    params: Optional[SettlementUpsampleParams] = None,
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
    terrain = refine_terrain(
        source, positions, weights, grid.texel_spacing_rad(tex_size), params,
        int(globe.seed or 0) ^ _SEED_SALT_SETTLEMENT, neighbours=grid.texel_neighbours(tex_size),
    )
    textures, loc = terrain.textures, terrain.located
    shape = (1, tex_size, tex_size)
    join_ocean_saddles(textures, Frame(geometry, float(side_km) * 1000.0), maps, float(params.saddle_channel_m))
    clear_thin_land(textures, float(side_km) * 1000.0 / tex_size, float(params.min_land_m))
    if source.has(SMALL_LAKE_KEY) and "lake_mask" in textures:
        near_small = source.coarse(SMALL_LAKE_KEY, loc)[:, 0].reshape(shape) > 0.0
        textures[SMALL_LAKE_KEY] = ((textures["lake_mask"][..., 0] > 0.5) & near_small)[..., None].astype(np.uint8)
    # Every other field: the owning map's value at the settlement, for the whole frame.
    at_center = source.locate(np.asarray(center, dtype=np.float64).reshape(1, 3))
    for key, tex in owner.textures.items():
        if key in textures or not source.has(key):
            continue
        value = source.anchor_values(key, at_center)[0].astype(tex.dtype)
        textures[key] = np.broadcast_to(value, shape + value.shape)
    gh, gw = globe.mesh.face_textures["elevation"].shape[1:3]
    sampler = GlobalSampler(globe.mesh)
    faces, u, v = sampler.locate(positions.reshape(-1, 3))
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
        detail_octaves=terrain.detail_octaves,
        texel_area_km2=texel_area_km2,
        parent_anchors=loc.anchors.reshape(shape).astype(np.int32),
        in_map=np.ones(shape, dtype=bool),
    )


def settlement_conservation_report(
    fields: RegionFields, maps: Sequence[RegionFields], antimeridian: float,
) -> dict[str, dict[str, float]]:
    """:func:`localmap.upsample.local_conservation_report` one tier down, for the
    one refined field: per local texel (of every map in ``maps``, as passed to
    :func:`build_settlement_fields`) the settlement's mean elevation against the
    local value.  (Nothing else is conserved: the rest is constant over the frame.)"""
    if fields.parent_anchors is None:
        raise ValueError("not a settlement map: no parent_anchors")
    parent = LocalMapsSource(maps, antimeridian).parent_textures()
    keys = ("elevation", "ocean_mask", "lake_mask")
    return anchor_residuals(
        {k: fields.textures[k] for k in keys if k in fields.textures}, fields.parent_anchors, fields.domain,
        {k: parent[k] for k in keys if k in parent},
    )
