"""Local maps: each regional hexagon refined onto its own finer hex grid.

The third tier, under :mod:`region`: a local map covers one regional hex
(:mod:`localmap.geometry`), tiles it with smaller hexagons
(:mod:`localmap.grid`) and fills their textures by refining the region's
exactly the way a region refines the globe (:mod:`localmap.upsample`).
See ``docs/plans/local_maps.md``.  Qt-free.
"""
from localmap.geometry import LocalGeometry
from localmap.grid import LocalHexGrid
from localmap.upsample import LocalUpsampleParams, build_local_fields, local_conservation_report

__all__ = [
    "LocalGeometry",
    "LocalHexGrid",
    "LocalUpsampleParams",
    "build_local_fields",
    "local_conservation_report",
]
