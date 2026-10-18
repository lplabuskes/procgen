"""A settlement site's water, as vectors: the sea, lakes, shorelines, rivers and their channels.

- **Sea and lakes** are traced from the refined terrain (:mod:`settlegen.site.contour`):
  the zero crossing of height above the water surface — sea level, or the nearest
  lake's surface — with its sign forced to the refined masks, so the polygons
  are the masks' water with smooth shores rather than a texel staircase.  Where
  the shore is within ``_FLAT_SHORE_M`` of the water, the heights can't place it
  (a forced sign alone traces the staircase): the mask, smoothed, does.
- **Rivers and streams** are the local maps' (:mod:`settlegen.site.river_chain`: each clipped to the ground its map
  owns, and chained across a seam between maps) re-traced over the frame's
  relief within about one local texel (:func:`localmap.rivers.build_local_rivers`
  one tier down), then widened into channels (:func:`settlegen.site.channels.natural_channels`): their
  bankfull width, wandering by ``width_wander`` and each bank by
  ``bank_wander`` over a few widths, with islands (eyots) in rivers at least
  ``island_min_width_m`` wide, the channel widened around each so both arms
  keep the river's width.  A channel runs on into the sea or lake it meets, so
  the two meet without a shore line across the mouth.  A line's end that hooks
  back in at the frame's edge (the router, pinned at its raster's edge) is cut
  where it leaves, and an end left just inside is carried straight out, so
  channels reach the frame's edge whole (:func:`settlegen.site.river_chain._tidy_ends`).

Qt-free.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np
import shapely
from scipy.ndimage import distance_transform_edt, gaussian_filter
from shapely.geometry import LineString, MultiLineString, MultiPolygon

from localmap.builder import LocalMap
from localmap.rivers import build_local_rivers
from region.rivers import RegionRivers
from region.upsample import SEA_LEVEL_KM, RegionFields
from settlegen.frame import Frame
from settlegen.site.channels import natural_channels
from settlegen.site.contour import regions_below
from settlegen.site.params import SettlementRiverParams
from settlegen.site.river_chain import parent_rivers, river_lines
from settlegen.site.types import SiteWater, _as_multipolygon
from worldgen.hydrology.discharge import discharge_per_flow_unit

if TYPE_CHECKING:
    from core.world import World


# Water / dry decided by the masks, the shore placed by the heights: never exactly on the level.
_EPS_M = 0.01


# Below this height above (or depth under) the water, the smoothed mask places the shore (σ in texels).
_FLAT_SHORE_M = 1.0


_FLAT_SHORE_SIGMA = 1.5


def water_regions(fields: RegionFields, frame: Frame) -> tuple[MultiPolygon, MultiPolygon]:
    """``(sea, lakes)`` polygons of the refined masks, shores at the water surface."""
    t = fields.tex_size
    xy = frame.texel_xy(t)
    elev_m = fields.textures["elevation"][0, ..., 0].astype(np.float64) * 1000.0
    out = []
    for key, level in (("ocean_mask", None), ("lake_mask", "lake")):
        mask = fields.textures.get(key)
        if mask is None or not (mask[0, ..., 0] > 0.5).any():
            out.append(MultiPolygon())
            continue
        wet = mask[0, ..., 0] > 0.5
        if level is None:
            surface = np.full_like(elev_m, SEA_LEVEL_KM * 1000.0)
        else:
            # Each texel reads the surface of its nearest lake texel.
            lake_surface = fields.textures["filled_height"][0, ..., 0].astype(np.float64) * 1000.0
            nearest = np.asarray(distance_transform_edt(~wet, return_distances=False, return_indices=True))
            surface = lake_surface[nearest[0], nearest[1]]
        height = elev_m - surface
        smooth = np.abs(gaussian_filter(np.where(wet, -1.0, 1.0), _FLAT_SHORE_SIGMA, mode="nearest"))
        least = np.maximum(_FLAT_SHORE_M * smooth, _EPS_M)
        signed = np.where(wet, np.minimum(height, -least), np.maximum(height, least))
        out.append(regions_below(signed, xy, frame.box()))
    return out[0], out[1]


def shorelines(water: MultiPolygon, frame: Frame, channels: Optional[MultiPolygon] = None) -> MultiLineString:
    """The water's edge inside the frame (its boundary, less the frame's own edge and
    the mouths of ``channels``)."""
    if water.is_empty:
        return MultiLineString()
    edge = frame.box().exterior.buffer(1e-3)
    if channels is not None and not channels.is_empty:
        edge = edge.union(channels.buffer(0.5))
    lines = shapely.line_merge(shapely.difference(water.boundary, edge))
    if lines.is_empty:
        return MultiLineString()
    parts = [lines] if isinstance(lines, LineString) else list(getattr(lines, "geoms", []))
    return MultiLineString([p for p in parts if isinstance(p, LineString) and not p.is_empty])


def build_site_water(
    globe: "World", fields: RegionFields, frame: Frame, maps: Sequence[LocalMap],
    params: Optional[SettlementRiverParams] = None,
) -> SiteWater:
    """The site's water.  See the module docstring."""
    params = params or SettlementRiverParams()
    sea, lakes = water_regions(fields, frame)
    parent, streams = parent_rivers(maps, globe.antimeridian)
    rivers = (build_local_rivers(fields, maps[0].fields, parent, globe, params, streams)
              if parent.paths or streams else RegionRivers(region_id=fields.region_id, paths=[]))
    lines = river_lines(rivers, frame, discharge_per_flow_unit(globe.mesh))
    still = _as_multipolygon(shapely.union_all([sea, lakes]))
    channels = natural_channels(lines, still, params, frame.half)
    return SiteWater(
        sea=sea, lakes=lakes, shorelines=shorelines(still, frame, channels),
        lines=lines, channels=channels, rivers=rivers,
    )
