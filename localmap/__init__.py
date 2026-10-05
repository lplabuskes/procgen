"""Local maps: each regional hexagon refined onto its own finer hex grid.

The third tier under :mod:`region`, refining the region the way a region refines
the globe; :mod:`localmap.builder` runs it.  See ``docs/local.md``.  Qt-free.
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
