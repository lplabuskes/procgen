"""Local maps: each regional hexagon refined onto its own finer hex grid.

The third tier, under :mod:`region`: a local map covers one regional hex
(:mod:`localmap.geometry`), tiles it with smaller hexagons
(:mod:`localmap.grid`) and fills their textures by refining the region's
exactly the way a region refines the globe (:mod:`localmap.upsample`).  It
re-routes the region's rivers and roads (:mod:`localmap.rivers`,
:mod:`localmap.roads`) and sites villages of 200–1000 people
(:mod:`localmap.villages`); :mod:`localmap.builder` runs it all.
See ``docs/local.md``.  Qt-free.
"""
from localmap.builder import LocalMap, build_local_map
from localmap.geometry import LocalGeometry
from localmap.grid import LocalHexGrid
from localmap.params import LocalParams
from localmap.rivers import LocalRiverParams, build_local_rivers
from localmap.roads import LocalRoadParams, build_local_roads
from localmap.upsample import LocalUpsampleParams, build_local_fields, local_conservation_report
from localmap.villages import VillageParams, Villages, place_villages, village_budget_report

__all__ = [
    "LocalGeometry",
    "LocalHexGrid",
    "LocalMap",
    "LocalParams",
    "LocalRiverParams",
    "LocalRoadParams",
    "LocalUpsampleParams",
    "VillageParams",
    "Villages",
    "build_local_fields",
    "build_local_map",
    "build_local_rivers",
    "build_local_roads",
    "local_conservation_report",
    "place_villages",
    "village_budget_report",
]
