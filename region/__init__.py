"""Regional maps: equal-area cubed-sphere subdivision and per-region refinement.

The globe is divided into ``6 · N_SUBDIV**2`` equal-area regions via a gnomonic
cubed-sphere parameterization with a separable equal-area CDF stretch
(:mod:`region.geometry`).  Each region's local ``[0, 1]**2`` frame is tiled with
hexagons carrying their own textures (:mod:`region.grid`), and
:mod:`region.upsample` fills those textures by refining the global fields:
bilinear upsample + slope-proportional detail, with every global texel's average
preserved exactly.  All of it is Qt-free; the viewer sits on top.
"""
from region.geometry import (
    N_SUBDIV,
    N_REGIONS,
    RegionGeometry,
    cell_border_polylines,
    ea_forward,
    ea_inverse,
    ecef_to_cell,
)
from region.grid import RegionHexGrid
from region.sampling import GlobalSampler
from region.upsample import (
    RegionFields,
    RegionUpsampleParams,
    build_region_fields,
    conservation_report,
)

__all__ = [
    "N_SUBDIV",
    "N_REGIONS",
    "GlobalSampler",
    "RegionFields",
    "RegionGeometry",
    "RegionHexGrid",
    "RegionUpsampleParams",
    "build_region_fields",
    "cell_border_polylines",
    "conservation_report",
    "ea_forward",
    "ea_inverse",
    "ecef_to_cell",
]
