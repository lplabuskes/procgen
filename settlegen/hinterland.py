"""A settlement's hinterland, as vectors: its neighbours, their fields, its own fields or gardens.

Claimed at the settlement tier, owing nothing to the local maps' coarse farmland
(the settlement map owns its footprint, plan Decision 4, so the local garden
ring would not fit it).  Only the neighbours themselves come from the local
maps: every village, town and city inside the frame, each from the map that
owns its ground.

The farmland rules are the local tier's (:mod:`localmap.farmland`,
``LocalParams.farmland``):

- the settlement itself farms from its footprint's edge: a city a ring of market
  gardens ``city_garden_ring_km`` wide (wider over good ground), a town
  ``town_farming_share`` of its people (a mining town ``mining_farming_share``),
  a village its farmers (``Villages.farming_share``: fishers and miners need no
  fields), at ``ha_per_farmer`` each;
- each neighbouring village or town from its position, by the same rules
  (neighbouring cities claim nothing: their gardens lie past the frame);
- gardens first, then the rest cheapest (settlement, texel) pair first, cost =
  distance × the local stretch (poor soil, slope, forest) × jitter, so
  neighbours share ground;
- never water, a channel or the footprint; a village farms only its own bank of
  a river at least ``river_barrier_width_m`` wide.

On a raster of at most ``_MAX_TEXELS`` across (fields are hectares; the terrain's
15 m is more than they need), traced to polygons.  Qt-free, deterministic.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional, Sequence

import numpy as np
import shapely
from scipy.ndimage import distance_transform_edt, label, maximum_filter
from scipy.spatial import KDTree
from shapely.geometry import MultiPolygon

from localmap.builder import LocalMap
from localmap.farmland import FarmlandParams, _claim
from localmap.villages import VILLAGE_MINING
from realmgen.towns.types import TOWN_MINING
from region.resources import refined_slope
from region.upsample import RegionFields
from settlegen.footprint import _trace
from settlegen.frame import Frame
from settlegen.source import LocalMapsSource
from settlegen.water import SiteWater, _as_multipolygon, channel_polygons

if TYPE_CHECKING:
    from core.world import World
    from settlegen.chain import SettlementRef

# Widest hinterland raster, in texels.
_MAX_TEXELS = 320
_M2_PER_HA = 10_000.0
_HINTERLAND_STREAM = 3


@dataclass
class Neighbour:
    """Another settlement inside the frame."""

    xy: np.ndarray                         # (2,) frame metres
    kind: Literal["city", "town", "village"]
    index: int                             # world city / town index, or village index on ``map``
    map: tuple[int, int]                   # (region_id, hex_index) of the local map it came from
    population: float
    mining: bool
    fields: MultiPolygon                   # what it farms inside the frame
    farming_share: float = 1.0             # a village's people who farm (the rest fish or mine)


@dataclass
class Hinterland:
    """The settlement's own fields (a city's: market gardens) and its neighbours."""

    own: MultiPolygon
    own_kind: Literal["gardens", "fields"]
    own_target_ha: float                   # what it wanted (a city: its whole ring)
    neighbours: list[Neighbour]


def neighbours_in_frame(
    globe: "World", frame: Frame, maps: Sequence[LocalMap], keys: Sequence[tuple[int, int]], ref: "SettlementRef",
) -> list[Neighbour]:
    """Every village, town and city in the frame but ``ref``, each from the map owning its ground."""
    source = LocalMapsSource([m.fields for m in maps], globe.antimeridian)
    box = frame.box().bounds
    towns = globe.get_feature("towns")
    cities = globe.get_feature("settlements") or []
    out: list[Neighbour] = []

    def add(i: int, positions: np.ndarray, kind, index, people, mining,  # type: ignore[no-untyped-def]
            farming=None) -> None:
        pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        if len(pos) == 0:
            return
        xy = frame.to_xy(pos)
        inside = (xy[:, 0] >= box[0]) & (xy[:, 0] <= box[2]) & (xy[:, 1] >= box[1]) & (xy[:, 1] <= box[3])
        inside &= source.map_of(pos) == i
        for k in np.flatnonzero(inside):
            if kind == ref.kind and int(index[k]) == ref.index and (kind != "village" or tuple(keys[i]) == ref.map):
                continue
            out.append(Neighbour(xy=xy[k], kind=kind, index=int(index[k]), map=tuple(keys[i]),  # type: ignore
                                 population=float(people[k]), mining=bool(mining[k]), fields=MultiPolygon(),
                                 farming_share=float(farming[k]) if farming is not None else 1.0))

    for i, m in enumerate(maps):
        v = m.villages
        add(i, v.positions, "village", np.arange(len(v)), v.population, np.asarray(v.kind) == VILLAGE_MINING,
            village_farming(v))
        t_idx = np.asarray(m.town_index, dtype=np.int64)
        if towns is not None and len(t_idx):
            add(i, m.town_positions, "town", t_idx, np.asarray(towns.population)[t_idx],
                np.asarray(towns.kind)[t_idx] == TOWN_MINING)
        c_idx = np.asarray(m.city_index, dtype=np.int64)
        if len(c_idx):
            add(i, m.city_positions, "city", c_idx, np.array([cities[c].population for c in c_idx]),
                np.zeros(len(c_idx), dtype=bool))
    return out


def village_farming(villages) -> np.ndarray:  # type: ignore[no-untyped-def]
    """Each village's share of people who farm (all of them, on maps built before it was kept)."""
    share = np.asarray(villages.farming_share, dtype=np.float64)
    return share if len(share) == len(villages) else np.ones(len(villages))


def _farmers_ha(kind: str, people: float, mining: bool, p: FarmlandParams, farming_share: float = 1.0) -> float:
    """Hectares farmed: a village's farmers (``farming_share``: fishers and miners
    need none), a town's ``town_farming_share`` (a mining town's ``mining_farming_share``)."""
    if kind == "village":
        share = float(np.clip(farming_share, 0.0, 1.0))
    else:
        share = float(p.town_farming_share) if kind == "town" else 0.0
        if mining:
            share = min(share, float(p.mining_farming_share))
    return max(float(people), 0.0) * share * float(p.ha_per_farmer)


def build_hinterland(
    globe: "World", fields: RegionFields, frame: Frame, water: SiteWater, footprint: MultiPolygon,
    maps: Sequence[LocalMap], keys: Sequence[tuple[int, int]], ref: "SettlementRef", population: float,
    mining: bool = False, params: Optional[FarmlandParams] = None, neighbours: Optional[list[Neighbour]] = None,
    farming_share: float = 1.0,
) -> Hinterland:
    """The hinterland.  See the module docstring.  ``neighbours``: given, rather than found in ``maps``;
    ``farming_share``: a village's people who farm."""
    p = params or FarmlandParams()
    if neighbours is None:
        neighbours = neighbours_in_frame(globe, frame, maps, keys, ref)
    t_fine = fields.tex_size
    step = max(1, math.ceil(t_fine / _MAX_TEXELS))
    t = max(2, t_fine // step)
    texel_m = frame.side_m / t
    texel_ha = texel_m ** 2 / _M2_PER_HA
    xy = frame.texel_xy(t)
    # Coarse texel ← the fine texels it covers: any water makes it wet, the mean slope its slope.
    fine_texel = frame.side_m / t_fine
    idx = np.clip(((np.arange(t) + 0.5) * texel_m / fine_texel).astype(int), 0, t_fine - 1)
    wet_fine = np.zeros((t_fine, t_fine), dtype=bool)
    for key in ("ocean_mask", "lake_mask"):
        if key in fields.textures:
            wet_fine |= fields.textures[key][0, ..., 0] > 0.5
    wet = np.asarray(maximum_filter(wet_fine, size=step), dtype=bool)[np.ix_(idx, idx)]
    if not water.channels.is_empty:
        wet |= shapely.contains_xy(water.channels, xy[..., 0], xy[..., 1])
    built = shapely.contains_xy(footprint, xy[..., 0], xy[..., 1]) if not footprint.is_empty else \
        np.zeros((t, t), dtype=bool)
    free = ~wet & ~built

    # The local stretch: poor soil, slope (relative to its saturation), forest; jittered.
    tex = fields.textures
    slope = refined_slope(tex["elevation"][..., 0], fine_texel / 1000.0)[0]
    slope = slope[np.ix_(idx, idx)]
    stretch = 1.0 + float(p.slope_weight) * np.clip(slope / max(float(p.slope_saturation), 1e-9), 0.0, 1.0)
    if "fertility" in tex:
        stretch += float(p.fertility_weight) * (1.0 - float(np.clip(tex["fertility"].reshape(-1)[0], 0.0, 1.0)))
    if "forest_density" in tex:
        stretch += float(p.forest_weight) * float(np.clip(tex["forest_density"].reshape(-1)[0], 0.0, 1.0))
    from settlegen.footprint import _seed
    rng = np.random.default_rng(_seed(int(globe.seed or 0), ref, _HINTERLAND_STREAM))
    stretch = stretch * (1.0 + float(p.edge_roughness) * rng.random((t, t)))

    taken = np.full(t * t, -1, dtype=np.int64)       # claimant per texel: 0 = the settlement, 1 + k = neighbour k
    from_footprint = (np.asarray(distance_transform_edt(~built)) * texel_m) if built.any() else \
        np.hypot(xy[..., 0], xy[..., 1])
    own_kind: Literal["gardens", "fields"] = "gardens" if ref.kind == "city" else "fields"

    # ── a city's market gardens: the ring round its footprint, wider over good ground ──
    if ref.kind == "city":
        ring_m = float(p.city_garden_ring_km) * 1000.0
        rel = stretch / float(np.median(stretch[free])) if free.any() else stretch
        garden = free & (from_footprint * rel <= ring_m)
        taken[garden.reshape(-1)] = 0
        own_target_ha = float(footprint.buffer(ring_m).difference(footprint).area) / _M2_PER_HA
    else:
        own_target_ha = _farmers_ha(ref.kind, population, mining, p, farming_share)

    # ── fields: cheapest (settlement, texel) pair first ──
    farming = [k for k, n in enumerate(neighbours) if n.kind != "city"]
    quotas = ([0 if ref.kind == "city" else own_target_ha / texel_ha]
              + [_farmers_ha(neighbours[k].kind, neighbours[k].population, neighbours[k].mining, p,
                             neighbours[k].farming_share) / texel_ha for k in farming])
    cells = np.flatnonzero((free.reshape(-1)) & (taken < 0))
    if cells.size and any(q > 0 for q in quotas):
        # Villages keep to their own bank of a river too wide to ford (and of the water).
        barrier = np.array(wet, dtype=bool)
        wide = [line for line in water.lines if (line.width_m >= float(p.river_barrier_width_m)).any()]
        if wide:
            deep = channel_polygons(wide)
            barrier |= shapely.contains_xy(deep, xy[..., 0], xy[..., 1])
        banks = np.asarray(label(~barrier)[0]).reshape(-1)  # type: ignore[index]
        tree = KDTree(xy.reshape(-1, 2)[cells])
        who_l, texel_l, cost_l = [], [], []
        flat_stretch = stretch.reshape(-1)
        for c, q in enumerate(quotas):
            if q <= 0:
                continue
            if c == 0:
                d = from_footprint.reshape(-1)[cells]
                keep = np.flatnonzero(np.isfinite(d))
                who_l.append(np.zeros(keep.size, dtype=np.int64))
                texel_l.append(keep)
                cost_l.append(d[keep] * flat_stretch[cells[keep]])
                continue
            n = neighbours[farming[c - 1]]
            reach = 3.0 * math.sqrt(q * texel_ha * _M2_PER_HA / math.pi) * (1.0 + float(p.edge_roughness)) \
                + 2.0 * texel_m
            near = np.asarray(tree.query_ball_point(n.xy, reach), dtype=np.int64)
            if n.kind == "village" and near.size:
                col = int(np.clip((n.xy[0] + frame.half) / texel_m, 0, t - 1))
                row = int(np.clip((frame.half - n.xy[1]) / texel_m, 0, t - 1))
                home = banks[row * t + col]
                if home > 0:
                    near = near[banks[cells[near]] == home]
            who_l.append(np.full(near.size, c, dtype=np.int64))
            texel_l.append(near)
            cost_l.append(np.linalg.norm(xy.reshape(-1, 2)[cells[near]] - n.xy, axis=1)
                          * flat_stretch[cells[near]])
        who = np.concatenate(who_l)
        texel = np.concatenate(texel_l)
        order = np.argsort(np.concatenate(cost_l), kind="stable").astype(np.int64)
        owner = np.full(cells.size, -1, dtype=np.int64)
        _claim(order, who, texel, np.round(np.asarray(quotas)).astype(np.int64), owner)
        got = owner >= 0
        taken[cells[got]] = np.where(owner[got] == 0, 0, np.asarray(farming + [0])[np.maximum(owner[got] - 1, 0)] + 1)

    wet_poly = _as_multipolygon(shapely.union_all([water.sea, water.lakes, water.channels, footprint]))
    grid = taken.reshape(t, t)
    own = _trace(grid == 0, frame, wet_poly)
    for k, n in enumerate(neighbours):
        mask = grid == k + 1
        if mask.any():
            n.fields = _trace(mask, frame, wet_poly)
    return Hinterland(own=own, own_kind=own_kind, own_target_ha=own_target_ha, neighbours=neighbours)
