"""Regional maps: equal-area cubed-sphere subdivision and per-region refinement.

``6 · N_SUBDIV**2`` equal-area cubed-sphere regions (:mod:`region.geometry`),
each tiled with textured hexagons (:mod:`region.grid`) that
:mod:`region.upsample` fills by refining the global fields while preserving every
global texel's average.  Qt-free.
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
from region.params import RegionParams
from region.rivers import RegionRiverParams, RegionRivers, build_region_rivers
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
    "RegionParams",
    "RegionRiverParams",
    "RegionRivers",
    "RegionUpsampleParams",
    "build_region_fields",
    "build_region_rivers",
    "cell_border_polylines",
    "conservation_report",
    "ea_forward",
    "ea_inverse",
    "ecef_to_cell",
]
