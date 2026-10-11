"""Which water a face can actually reach, and the water links cities are tagged with.

The trade graph is per face, but a face is ~220 km across: touching a water face,
or being crossed by a navigable river, does not put water within reach of a city
standing in it.  :class:`WaterAccess` reads the texels:

- a **port** (land-only face → water-only neighbour) needs the face to own coast
  of that water's kind (an owned land texel 4-adjacent to ocean, or to a lake);
- a **river face's water edge** needs a river shared with the neighbour (it flows
  on into it), or — onto a water-only neighbour — coast of that water's kind.

Each city is then tagged with its **water links**: per land↔water link its trade
uses, the tags of the water that serves it — a world river's index into
``"rivers"``, :data:`TAG_OCEAN` or :data:`TAG_LAKE` (by kind; any globe-lake
coast serves).  Every siting tier keeps the city on that water
(:func:`water_link_mask`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

from realmgen.refine.sites import SiteField, densify_polylines

TAG_OCEAN = -1
TAG_LAKE = -2


@dataclass
class WaterAccess:
    """Per face, the water its own texels reach."""

    rivers: list[frozenset]   # per face: world rivers (``"rivers"`` indices) running through it
    coast_ocean: np.ndarray   # (N,) bool — owns a land texel 4-adjacent to ocean
    coast_lake: np.ndarray    # (N,) bool — owns a land texel 4-adjacent to a lake
    ocean: np.ndarray         # (N,) bool — an ocean face (a water-only face that isn't is a lake)
    land_passable: np.ndarray  # (N,) bool
    sea_passable: np.ndarray   # (N,) bool

    def link_tags(self, f: int, g: int) -> tuple[int, ...]:
        """Tags of the water by which land face ``f`` reaches its sea-passable neighbour ``g``.

        Empty when its texels reach none (no such link).  A river face reaches ``g``
        by a river they share; any land face reaches a water-only ``g`` by coast of
        its kind.
        """
        tags: list[int] = []
        if self.sea_passable[f]:
            tags += sorted(self.rivers[f] & self.rivers[g])
        if not self.land_passable[g]:
            if self.ocean[g] and self.coast_ocean[f]:
                tags.append(TAG_OCEAN)
            elif not self.ocean[g] and self.coast_lake[f]:
                tags.append(TAG_LAKE)
        return tuple(tags)

    def sea_edge_ok(self, f: int, g: int) -> bool:
        """Whether sea-layer edge ``f``–``g`` (both sea-passable) is real water."""
        if not self.land_passable[f] and not self.land_passable[g]:
            return True
        if not self.land_passable[f]:
            f, g = g, f
        return bool(self.link_tags(f, g))


def build_water_access(
    mesh,
    ocean_tex: np.ndarray,
    lake_tex: Optional[np.ndarray],
    rivers: Optional[Sequence],
    is_ocean: np.ndarray,
    land_passable: np.ndarray,
    sea_passable: np.ndarray,
    sites: Optional[SiteField] = None,
) -> WaterAccess:
    """:class:`WaterAccess` from the world's water masks and ``"rivers"``.

    Face ownership is :class:`SiteField`'s (nearest centroid), so a link found
    here is one the city's siting can meet.
    """
    n = len(mesh.faces)
    land_p = np.asarray(land_passable, dtype=bool)
    sea_p = np.asarray(sea_passable, dtype=bool)
    sites = sites or SiteField(mesh, ocean_tex, lake_tex, None, None)

    # Rivers by the faces owning their (densified) course.
    face_rivers: list[set] = [set() for _ in range(n)]
    lines = [np.asarray(r[0], dtype=np.float64) for r in (rivers or [])]
    step = 0.5 * float(sites.face(0).spacing) if n else 0.0
    for r, line in enumerate(lines):
        pts = densify_polylines([line], step)
        if pts is None:
            continue
        for f in np.unique(sites.owner(pts)).tolist():
            face_rivers[f].add(r)

    # Coast only matters on land faces beside a water-only face.
    water_only = sea_p & ~land_p
    beside = np.array([land_p[f] and any(water_only[g] for g in adj)
                       for f, adj in enumerate(mesh.face_adjacency)], dtype=bool)
    coast_ocean = np.zeros(n, dtype=bool)
    coast_lake = np.zeros(n, dtype=bool)
    faces = np.flatnonzero(beside)
    sites.prefetch(faces.tolist())
    for f in faces.tolist():
        fs = sites.face(f)
        idx = fs.sites()
        coast_ocean[f] = bool(fs.coast_ocean[idx].any())
        coast_lake[f] = bool(fs.coast_lake[idx].any())

    return WaterAccess(
        rivers=[frozenset(s) for s in face_rivers],
        coast_ocean=coast_ocean, coast_lake=coast_lake,
        ocean=np.asarray(is_ocean, dtype=bool), land_passable=land_p, sea_passable=sea_p,
    )


def port_links(mesh, access: WaterAccess) -> np.ndarray:
    """``(E,)`` bool per directed adjacency edge (``face_adjacency`` flattened in
    order): a land face's link onto a water face its texels reach."""
    return np.fromiter(
        (bool(access.link_tags(f, g)) if access.land_passable[f] and access.sea_passable[g] else False
         for f, adj in enumerate(mesh.face_adjacency) for g in adj),
        dtype=bool,
    )


def mask_sea_edges(mesh, sea_edge_costs: list[np.ndarray], access: WaterAccess) -> list[np.ndarray]:
    """``sea_edge_costs`` with edges between faces whose texels share no water set to ``inf``."""
    out = []
    land_p = access.land_passable
    for f, (adj, costs) in enumerate(zip(mesh.face_adjacency, sea_edge_costs)):
        c = np.array(costs, dtype=np.float64, copy=True)
        for k, g in enumerate(adj[:c.size]):
            # A finite sea edge joins two sea-passable faces; only one with a river face can be false.
            if np.isfinite(c[k]) and (land_p[f] or land_p[g]) and not access.sea_edge_ok(f, int(g)):
                c[k] = np.inf
        out.append(c)
    return out


def settlement_water_links(
    access: WaterAccess, faces: Sequence[int], links: Sequence[Sequence[tuple[int, float]]],
) -> list[list[tuple[int, ...]]]:
    """Per settlement, the tags of each of its ``links`` (``(water face, flow)``), strongest first.

    Links to the same water merge (their flows add); links whose water the face's
    texels don't reach are dropped.
    """
    out = []
    for f, own in zip(faces, links):
        flow: dict[tuple[int, ...], float] = {}
        for g, fl in own:
            tags = access.link_tags(int(f), int(g))
            if tags:
                flow[tags] = flow.get(tags, 0.0) + float(fl)
        out.append([t for t, _ in sorted(flow.items(), key=lambda kv: (-kv[1], kv[0]))])
    return out


def river_flows(rivers: Optional[Sequence]) -> np.ndarray:
    """``(R,)`` each world river's largest flow (at its mouth)."""
    return np.array([float(np.max(r[1])) if len(r[1]) else 0.0 for r in (rivers or [])], dtype=np.float64)


def water_link_mask(
    links: Sequence[Sequence[int]],
    coast_ocean: np.ndarray,
    coast_lake: np.ndarray,
    near_river: Callable[[int], np.ndarray],
    river_flows: np.ndarray,
) -> Optional[np.ndarray]:
    """``(P,)`` the candidates on the water of a city's ``links``.

    A link is met by any of its tags: on river ``t`` (``near_river(t)``), or on
    ocean / lake coast.  Candidates meeting every link when some do.  Otherwise
    the best place the links' waters meet, by class: a confluence of tagged
    rivers > a tagged river's mouth on tagged coast > tagged coast > one tagged
    river.  Within the class, the highest flow of the tagged rivers met
    (``river_flows`` per world river), then the links strongest first.  None when
    no candidate is on any of it.
    """
    if not links:
        return None
    n = len(coast_ocean)
    near = {r: near_river(r) for r in sorted({int(t) for tags in links for t in tags if t >= 0})}

    def met(tags: Sequence[int]) -> np.ndarray:
        ok = np.zeros(n, dtype=bool)
        for t in tags:
            ok |= coast_ocean if t == TAG_OCEAN else coast_lake if t == TAG_LAKE else near[int(t)]
        return ok

    every = np.logical_and.reduce([met(tags) for tags in links])
    if every.any():
        return every
    tagged = {int(t) for tags in links for t in tags}
    coast = np.zeros(n, dtype=bool)
    if TAG_OCEAN in tagged:
        coast |= coast_ocean
    if TAG_LAKE in tagged:
        coast |= coast_lake
    count = np.zeros(n, dtype=np.int64)
    flow = np.zeros(n, dtype=np.float64)
    for r, ok in near.items():
        count += ok
        flow += np.where(ok, river_flows[r] if 0 <= r < len(river_flows) else 0.0, 0.0)
    for cls in (count >= 2, (count > 0) & coast, coast, count > 0):
        if cls.any():
            mask = cls & (flow == flow[cls].max())
            for tags in links:
                both = mask & met(tags)
                if both.any():
                    mask = both
            return mask
    return None
