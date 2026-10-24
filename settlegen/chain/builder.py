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
3. the frame: a square round where that local map draws the settlement, of side
   ``2 · (footprint radius + garden ring + margin)`` (a city's garden ring by
   :func:`localmap.farmland.garden_ring_km`; a town's or village's 1.5 km of
   fields), the radius the settlement
   tier's own (its people at :func:`settlegen.site.footprint.base_density_per_ha`;
   the local map's footprint is only a preview there).  Squared to north, and
   slid toward land by up to ``land_shift`` of the ring (:func:`land_shifted_center`),
   so a coastal settlement's frame isn't half sea; the footprint and margin stay inside;
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
from localmap.farmland import garden_ring_km
from localmap.settlements import owning_hex
from region.geometry import RegionGeometry, ecef_to_cell
from region.grid import RegionHexGrid
from region.source import RegionSource
from settlegen.geometry import SettlementGeometry
from settlegen.site.footprint import base_density_per_ha
from settlegen.terrain.builder import km_per_local_unit

if TYPE_CHECKING:
    from core.world import World
    from localmap.store import MapStore
    from settlegen.site.params import SettlementFootprintParams

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
    frame_center: np.ndarray             # (3,) the frame's centre: ``center`` slid toward land
    radius_km: float                     # footprint radius at ``density_per_ha``
    population: float
    density_per_ha: float                # before the site's squeeze (settlegen.site.footprint)
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


# The ground kept round a town's or village's footprint for its fields.
_FIELDS_KM = 1.5


def footprint_radius_km(population: float, density_per_ha: float) -> float:
    """The radius of a disc of ``population`` people at ``density_per_ha``."""
    area = max(float(population), 0.0) / (max(float(density_per_ha), 1e-9) / _KM2_PER_HA)
    return math.sqrt(area / math.pi)


def frame_side_km(radius_km: float, garden_ring_km: float, margin_km: float) -> float:
    return 2.0 * (float(radius_km) + float(garden_ring_km) + float(margin_km))


# Candidate shifts per axis, land samples across the frame, and the land share
# a full shift must gain (less is spent on smaller shifts in proportion).
_SHIFT_STEPS = 9
_SHIFT_SAMPLES = 48
_SHIFT_PENALTY = 0.05


def land_shifted_center(owner: LocalMap, center: np.ndarray, side_km: float, max_shift_km: float) -> np.ndarray:
    """The frame centre (unit vector): ``center`` moved east / north by at most
    ``max_shift_km`` each way to the side ``side_km`` square holding the most land,
    as the owning local map's water (bilinear, its over-hang past the hexagon
    included) has it.  ``center`` itself unless the land gained pays for the move."""
    center = np.asarray(center, dtype=np.float64).reshape(3)
    if not max_shift_km > 0.0:
        return center
    f = owner.fields
    s, t, _ = f.geometry.ecef_to_local(center[None])
    ext = float(side_km) + 2.0 * float(max_shift_km)
    reach = SettlementGeometry(f.geometry, np.array([s[0], t[0]]), ext / km_per_local_unit(f))
    n = int(math.ceil(_SHIFT_SAMPLES * ext / float(side_km)))
    c = (np.arange(n) + 0.5) / n
    aa, bb = np.meshgrid(c, c)                                            # rows along b (north)
    source = RegionSource(f)
    loc = source.locate(reach.local_to_ecef(aa.ravel(), bb.ravel()))
    wet = np.zeros(n * n)
    for key in ("ocean_mask", "lake_mask"):
        if source.has(key):
            wet += source.coarse(key, loc)[:, 0]
    wet = np.clip(wet, 0.0, 1.0).reshape(n, n)
    xs = (c - 0.5) * ext                                                  # km from the settlement
    best, shift = -np.inf, (0.0, 0.0)
    steps = np.linspace(-max_shift_km, max_shift_km, _SHIFT_STEPS)
    for dx in steps:
        cols = np.abs(xs - dx) <= 0.5 * side_km
        for dy in steps:
            rows = np.abs(xs - dy) <= 0.5 * side_km
            land = 1.0 - float(wet[np.ix_(rows, cols)].mean())
            score = land - _SHIFT_PENALTY * math.hypot(dx, dy) / max_shift_km
            if score > best + 1e-12:
                best, shift = score, (dx, dy)
    if shift == (0.0, 0.0):
        return center
    return reach.local_to_ecef(np.array([0.5 + shift[0] / ext]), np.array([0.5 + shift[1] / ext]))[0]


def frame_keys(
    globe: "World", owner: LocalMap, center: np.ndarray, side_km: float, n_cols: int,
) -> list[tuple[int, int]]:
    """``(region_id, hex_index)`` of every local map whose ground the frame overlaps,
    the owning map first.  Exact on the regional hex grids (no region is built):
    the north-squared frame, in each touched region's ``(s, t)``, against that region's
    hexagons clipped to its cell ``[0, 1]²`` — the ground its local maps own."""
    geo = owner.fields.geometry
    s, t, _ = geo.ecef_to_local(np.asarray(center, dtype=np.float64).reshape(1, 3))
    frame_geo = SettlementGeometry(geo, np.array([s[0], t[0]]), float(side_km) / km_per_local_unit(owner.fields))
    edge = np.linspace(0.0, 1.0, _EDGE_SAMPLES, endpoint=False)
    lo, hi = np.zeros_like(edge), np.ones_like(edge)
    ring = np.concatenate([
        np.column_stack([edge, lo]), np.column_stack([hi, edge]),
        np.column_stack([1.0 - edge, hi]), np.column_stack([lo, 1.0 - edge]),
    ])
    outline = frame_geo.local_to_ecef(ring[:, 0], ring[:, 1])            # (R, 3) frame boundary
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
    footprint: Optional["SettlementFootprintParams"] = None, land_shift: float = 1.0,
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
    ring = garden_ring_km(people, math.pi * radius ** 2, local_params.farmland) if ref.kind == "city" \
        else _FIELDS_KM
    side = frame_side_km(radius, ring, margin_km)
    frame_center = land_shifted_center(owner, center, side, max(float(land_shift), 0.0) * ring)
    keys = frame_keys(globe, owner, frame_center, side, int(store.region_params.upsample.n_cols))
    maps = [owner] + [store.local(*key) for key in keys[1:]]
    return SettlementMaps(
        ref=ref, keys=keys, maps=maps, center=center, frame_center=frame_center, radius_km=radius, population=people,
        density_per_ha=density, side_km=side,
        built=list(store.builds[builds_before:]),
    )
