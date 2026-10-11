"""The chain from a city, town or village to the local maps its settlement map refines.

:func:`ensure_local_maps` takes a settlement to every local map its frame needs,
asking a :class:`localmap.store.MapStore` for each — so a region or local map is
built only if the store doesn't already hold it (the viewer and the chain share
one store):

1. the settlement's region (it refines the settlement there, if not yet);
2. its owning local map — the regional hex nearest its region position
   (:func:`localmap.settlements.owning_hex`) — which refines it locally: its
   position is final from here on.  A village has no world record: it is one of
   a local map's villages, named by that map (``SettlementRef.map``) and its
   index there, valid while that map's params don't change;
3. the frame: a square centred where that local map draws the settlement, of side
   ``2 · (footprint radius + garden ring + margin)``, the radius the settlement
   tier's own (its people at :func:`settlegen.footprint.base_density_per_ha`;
   the local map's footprint is only a preview there);
4. every other local map whose ground the frame reaches (each regional hexagon,
   within its own region's cell, the frame overlaps), with their regions.

The chain never resets a settlement; the store's params do (see its docstring).
Qt-free.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Optional

import numpy as np
from shapely.geometry import Polygon, box

from localmap.builder import LocalMap
from localmap.settlements import owning_hex
from region.geometry import RegionGeometry, ecef_to_cell
from region.grid import RegionHexGrid
from settlegen.footprint import base_density_per_ha
from settlegen.upsample import km_per_local_unit

if TYPE_CHECKING:
    from core.world import World
    from localmap.store import MapStore
    from settlegen.footprint import SettlementFootprintParams

_KM2_PER_HA = 0.01
# Points per frame edge when mapping the frame into another region's frame (≤10 km: ~150 m apart).
_EDGE_SAMPLES = 64


@dataclass(frozen=True)
class SettlementRef:
    """A city (index into the ``settlements`` feature), a town (index into ``towns``)
    or a village (index into ``LocalMap.villages`` of local map ``map``)."""

    kind: Literal["city", "town", "village"]
    index: int
    map: Optional[tuple[int, int]] = None      # (region_id, hex_index): villages only

    def __post_init__(self) -> None:
        if (self.kind == "village") != (self.map is not None):
            raise ValueError("a village, and only a village, names its local map")


def site_position(globe: "World", ref: SettlementRef) -> np.ndarray:
    """The settlement's stage site — which region refines it."""
    if ref.kind == "city":
        return np.asarray(globe.get_feature("settlements")[ref.index].site_position, dtype=np.float64)
    return np.asarray(globe.get_feature("towns").site_positions[ref.index], dtype=np.float64)


def region_position(globe: "World", ref: SettlementRef) -> np.ndarray:
    """Where its region put it — which local map refines it."""
    if ref.kind == "city":
        return np.asarray(globe.get_feature("settlements")[ref.index].region_position, dtype=np.float64)
    return np.asarray(globe.get_feature("towns").region_positions[ref.index], dtype=np.float64)


def position(globe: "World", ref: SettlementRef) -> np.ndarray:
    """Its current position (the finest tier that has refined it)."""
    if ref.kind == "city":
        return np.asarray(globe.get_feature("settlements")[ref.index].position, dtype=np.float64)
    return np.asarray(globe.get_feature("towns").positions[ref.index], dtype=np.float64)


def population(globe: "World", ref: SettlementRef) -> float:
    if ref.kind == "city":
        return float(globe.get_feature("settlements")[ref.index].population)
    return float(globe.get_feature("towns").population[ref.index])


def is_local_refined(globe: "World", ref: SettlementRef) -> bool:
    if ref.kind == "village":
        return True                       # born on its local map
    if ref.kind == "city":
        return bool(globe.get_feature("settlements")[ref.index].local_refined)
    return bool(globe.get_feature("towns").local_refined[ref.index])


def region_of(globe: "World", ref: SettlementRef) -> int:
    """The region that refines the settlement: the cell of its stage site (a village: its map's region)."""
    if ref.map is not None:
        return int(ref.map[0])
    return int(ecef_to_cell(site_position(globe, ref)[None], globe.antimeridian)[0])


@dataclass
class SettlementMaps:
    """The local maps one settlement's frame needs, and the frame."""

    ref: SettlementRef
    keys: list[tuple[int, int]]          # (region_id, hex_index), owning map first
    maps: list[LocalMap]                 # in ``keys`` order
    center: np.ndarray                   # (3,) where the owning local map draws the settlement
    radius_km: float                     # footprint radius at ``density_per_ha``
    population: float
    density_per_ha: float                # before the site's squeeze (settlegen.footprint)
    side_km: float                       # frame side
    built: list[tuple[str, object]] = field(default_factory=list)   # what this call had the store build

    @property
    def owner(self) -> LocalMap:
        return self.maps[0]


def drawn_position(local: LocalMap, ref: SettlementRef) -> np.ndarray:
    """Where ``local`` draws the settlement (its position, moved just off a river channel)."""
    if ref.kind == "village":
        return np.asarray(local.villages.positions[ref.index], dtype=np.float64)
    index, positions = (local.city_index, local.city_positions) if ref.kind == "city" else \
        (local.town_index, local.town_positions)
    hit = np.flatnonzero(np.asarray(index) == ref.index)
    if hit.size == 0:
        raise ValueError(f"{ref.kind} {ref.index} is not on local map {local.hex_index}")
    return np.asarray(positions[hit[0]], dtype=np.float64)


def settlement_population(globe: "World", local: LocalMap, ref: SettlementRef) -> float:
    """Its people (a village's: its local map's)."""
    if ref.kind == "village":
        return float(local.villages.population[ref.index])
    return population(globe, ref)


def footprint_radius_km(population: float, density_per_ha: float) -> float:
    """The radius of a disc of ``population`` people at ``density_per_ha``."""
    area = max(float(population), 0.0) / (max(float(density_per_ha), 1e-9) / _KM2_PER_HA)
    return math.sqrt(area / math.pi)


def frame_side_km(radius_km: float, garden_ring_km: float, margin_km: float) -> float:
    return 2.0 * (float(radius_km) + float(garden_ring_km) + float(margin_km))


def frame_keys(
    globe: "World", owner: LocalMap, center: np.ndarray, side_km: float, n_cols: int,
) -> list[tuple[int, int]]:
    """``(region_id, hex_index)`` of every local map whose ground the frame overlaps,
    the owning map first.  Exact on the regional hex grids (no region is built):
    the frame square, in each touched region's ``(s, t)``, against that region's
    hexagons clipped to its cell ``[0, 1]²`` — the ground its local maps own."""
    geo = owner.fields.geometry
    s, t, _ = geo.ecef_to_local(np.asarray(center, dtype=np.float64).reshape(1, 3))
    half = 0.5 * float(side_km) / km_per_local_unit(owner.fields)
    edge = np.linspace(-half, half, _EDGE_SAMPLES, endpoint=False)
    ring = np.concatenate([
        np.column_stack([edge, np.full_like(edge, -half)]), np.column_stack([np.full_like(edge, half), edge]),
        np.column_stack([-edge, np.full_like(edge, half)]), np.column_stack([np.full_like(edge, -half), -edge]),
    ]) + np.array([s[0], t[0]])
    outline = geo.local_to_ecef(ring[:, 0], ring[:, 1])                  # (R, 3) frame boundary
    owner_key = (owner.fields.region_id, owner.hex_index)
    keys = [owner_key]
    # A ≤10 km frame can't hold a whole region cell, so its outline meets every cell it touches.
    for region_id in np.unique(ecef_to_cell(outline, globe.antimeridian)):
        region = RegionGeometry(int(region_id), globe.antimeridian)
        a, b, valid = region.ecef_to_local(outline)
        if not valid.all():
            continue
        frame = Polygon(np.column_stack([a, b])).intersection(box(0.0, 0.0, 1.0, 1.0))
        if frame.is_empty or frame.area <= 0.0:
            continue
        grid = RegionHexGrid(region, n_cols)
        corners = grid.corner_offsets()
        for h in range(len(grid)):
            key = (int(region_id), h)
            if key in keys:
                continue
            if Polygon(grid.centers[h] + corners).intersection(frame).area > 0.0:
                keys.append(key)
    return keys


def ensure_local_maps(
    store: "MapStore", ref: SettlementRef, margin_km: float = 0.5,
    footprint: Optional["SettlementFootprintParams"] = None,
) -> SettlementMaps:
    """Every local map ``ref``'s frame needs, from ``store``.  See the module docstring."""
    globe = store.globe
    if globe is None:
        raise ValueError("the store has no globe")
    builds_before = len(store.builds)
    if ref.map is not None:
        owner = store.local(*ref.map)
        if not 0 <= ref.index < len(owner.villages):
            raise ValueError(f"local map {ref.map} has no village {ref.index}")
    else:
        region_id = region_of(globe, ref)
        region = store.region(region_id)
        hexes, valid = owning_hex(region.fields, region_position(globe, ref)[None])
        if not valid[0]:
            raise ValueError(f"{ref.kind} {ref.index} lies on the far side of region {region_id}")
        owner = store.local(region_id, int(hexes[0]))
        if not is_local_refined(globe, ref):
            raise RuntimeError(f"{ref.kind} {ref.index} was not refined by its local map ({region_id}, {hexes[0]})")

    local_params = store.local_params
    center = drawn_position(owner, ref)
    people = settlement_population(globe, owner, ref)
    density = base_density_per_ha(globe, ref, people, footprint)
    radius = footprint_radius_km(people, density)
    side = frame_side_km(radius, local_params.farmland.city_garden_ring_km, margin_km)
    keys = frame_keys(globe, owner, center, side, int(store.region_params.upsample.n_cols))
    maps = [owner] + [store.local(*key) for key in keys[1:]]
    return SettlementMaps(
        ref=ref, keys=keys, maps=maps, center=center, radius_km=radius, population=people,
        density_per_ha=density, side_km=side,
        built=list(store.builds[builds_before:]),
    )
