"""Settlement maps: one town or city refined from its local map(s) onto a street-scale frame.

The fourth tier under :mod:`localmap`, refining the local maps a settlement's
frame touches the way a local map refines its region.  See
``docs/plans/settlement_maps.md``.  Qt-free.
"""
from settlegen.builder import SettlementMap, build_settlement_map
from settlegen.chain import SettlementMaps, SettlementRef, ensure_local_maps
from settlegen.geometry import SettlementGeometry
from settlegen.grid import SettlementGrid
from settlegen.params import SettlementMapParams
from settlegen.source import LocalMapsSource
from settlegen.upsample import (
    SettlementUpsampleParams, build_settlement_fields, settlement_conservation_report, with_local_detail,
)

__all__ = [
    "LocalMapsSource",
    "SettlementMap",
    "SettlementMapParams",
    "SettlementMaps",
    "SettlementRef",
    "SettlementGeometry",
    "SettlementGrid",
    "SettlementUpsampleParams",
    "build_settlement_fields",
    "build_settlement_map",
    "ensure_local_maps",
    "settlement_conservation_report",
    "with_local_detail",
]
