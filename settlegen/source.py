"""Several local maps as the parent of one settlement map.

A settlement's frame can cross its local map's edge (the parent regional
hexagon) into a neighbouring local map, maybe in another region.  Each point
belongs to the local map whose regional hexagon holds it — the region by
:func:`region.geometry.ecef_to_cell`, the hexagon by the local map's own domain
(``in_map``), as :func:`localmap.settlements.local_owned` assigns settlements —
and is read from that map's :class:`region.source.RegionSource`.

Anchors are numbered in one space: map ``i``'s texels follow the maps before it
(:attr:`LocalMapsSource.offsets`), so conservation runs per local texel of
every map unchanged.

Across a seam a map's over-hang is not the neighbour's ground: it was never
conserved, so it is off by the neighbour's correction (seed 4: ~3 m of
elevation step against ~0.2 m between texels).  So with several maps every
smooth read goes through :meth:`stencil`, whose corners are re-owned across
maps (:meth:`owners`) as :class:`RegionSource` re-owns them across hexes.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

from localmap.geometry import PARENT_CENTER, PARENT_RADIUS
from localmap.grid import in_flat_hexagon
from region.geometry import ecef_to_cell
from region.resources import refined_slope
from region.sampling import settle_owners
from region.source import Located, ParentSource, RegionSource, _bilinear_stencil

if TYPE_CHECKING:
    from region.upsample import RegionFields


class LocalMapsSource(ParentSource):
    """Local maps' fields as one parent tier.  See the module docstring."""

    def __init__(self, maps: Sequence["RegionFields"], antimeridian: float) -> None:
        if not maps:
            raise ValueError("no local maps")
        self.maps = list(maps)
        self.antimeridian = float(antimeridian)
        self.sources = [RegionSource(f) for f in self.maps]
        sizes = np.array([s.n_anchors for s in self.sources], dtype=np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]])
        self.n_anchors = int(sizes.sum())
        self.spacing_rad = float(np.mean([s.spacing_rad for s in self.sources]))
        self._owner: Optional[np.ndarray] = None
        # key → (n_anchors, C) float64, for the cross-map smooth reads.
        self._per_anchor: dict[str, np.ndarray] = {}
        self._stencils: dict[int, tuple[Located, tuple[np.ndarray, np.ndarray]]] = {}

    def map_of(self, points: np.ndarray) -> np.ndarray:
        """``(P,)`` index of the map owning each point, ``-1`` where none does."""
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        which = np.full(len(points), -1, dtype=np.int64)
        cell = ecef_to_cell(points, self.antimeridian)
        for i, f in enumerate(self.maps):
            idx = np.flatnonzero((which < 0) & (cell == f.region_id))
            if idx.size == 0:
                continue
            a, b, valid = f.geometry.ecef_to_local(points[idx])
            inside = valid & in_flat_hexagon(np.column_stack([a, b]) - PARENT_CENTER, PARENT_RADIUS)
            which[idx[inside]] = i
        return which

    def has(self, key: str) -> bool:
        return all(s.has(key) for s in self.sources)

    def locate(self, points: np.ndarray) -> Located:
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        which = self.map_of(points)
        if (which < 0).any():
            raise ValueError(f"{int((which < 0).sum())} points lie outside every given local map")
        anchors = np.empty(len(points), dtype=np.int64)
        groups = []
        for i, source in enumerate(self.sources):
            idx = np.flatnonzero(which == i)
            if idx.size == 0:
                continue
            loc = source.locate(points[idx])
            # Re-owned across maps too: a texel straddling the parent hexagon's
            # edge (centre past it) is the neighbour's, never this map's over-hang.
            anchors[idx] = self.owners()[loc.anchors + self.offsets[i]]
            groups.append((i, idx, loc))
        return Located(anchors, (len(points), groups))

    def _anchor_field(self, key: str) -> np.ndarray:
        cached = self._per_anchor.get(key)
        if cached is None:
            if key == "slope":
                parts = [refined_slope(f.textures["elevation"][..., 0], math.sqrt(max(f.texel_area_km2, 1e-9)))
                         .reshape(-1, 1) for f in self.maps]
            else:
                parts = [np.asarray(f.textures[key], dtype=np.float64).reshape(s.n_anchors, -1)
                         for f, s in zip(self.maps, self.sources)]
            cached = self._per_anchor[key] = np.concatenate(parts, axis=0)
        return cached

    def coarse(self, key: str, loc: Located) -> np.ndarray:
        if len(self.maps) == 1:
            return self.sources[0].coarse(key, loc.data[1][0][2])
        return self.interpolate(self._anchor_field(key), loc)

    def anchor_values(self, key: str, loc: Located) -> np.ndarray:
        # At the re-owned anchor, which may be another map's texel.  Read at ``loc.anchors``
        # (as the multi-map case must), not the sub-locations': a caller may pin them
        # (region.carve keeps the build's anchors), and in a build they're the same.
        return self._anchor_field(key)[loc.anchors]

    def owners(self) -> np.ndarray:
        """Each map's own owners, except over-hang past its parent hexagon that
        another map owns: that map's texel there."""
        if self._owner is None:
            owner = np.concatenate([s.owners() + off for s, off in zip(self.sources, self.offsets)])
            for i, f in enumerate(self.maps if len(self.maps) > 1 else []):
                if f.in_map is None:
                    continue
                outside = np.flatnonzero(~f.in_map.reshape(-1))
                points = f.positions.reshape(-1, 3)[outside].astype(np.float64)
                which = self.map_of(points)
                for j in range(len(self.maps)):
                    if j == i:
                        continue
                    sel = which == j
                    if sel.any():
                        owner[self.offsets[i] + outside[sel]] = self._domain_texel(j, points[sel]) + self.offsets[j]
            self._owner = settle_owners(owner)
        return self._owner

    def _domain_texel(self, j: int, points: np.ndarray) -> np.ndarray:
        """Map ``j``'s texel nearest each point among those in its domain, around the one under it.

        The texel under a point by the seam may have its centre past it, in the
        other map; owning it would chain back there (a cycle of two over-hang
        texels, neither conserved)."""
        f = self.maps[j]
        t = f.tex_size
        domain = f.domain.reshape(-1)
        positions = f.positions.reshape(-1, 3)
        under = self.sources[j]._locate_nearest(points).anchors
        hexes, rows, cols = under // (t * t), (under // t) % t, under % t
        best = under.copy()
        best_d2 = np.full(len(points), np.inf)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                r2, c2 = np.clip(rows + dr, 0, t - 1), np.clip(cols + dc, 0, t - 1)
                cand = hexes * (t * t) + r2 * t + c2
                d2 = ((positions[cand].astype(np.float64) - points) ** 2).sum(axis=1)
                better = domain[cand] & (d2 < best_d2)
                best = np.where(better, cand, best)
                best_d2 = np.where(better, d2, best_d2)
        return best

    def stencil(self, loc: Located) -> tuple[np.ndarray, np.ndarray]:
        n, groups = loc.data
        cached = self._stencils.get(id(loc))
        if cached is not None and cached[0] is loc:
            return cached[1]
        idx_out = np.zeros((n, 4), dtype=np.int64)
        w_out = np.zeros((n, 4), dtype=np.float64)
        for i, idx, sub in groups:
            hexes, x, y = sub.data
            t = self.maps[i].tex_size
            # The map's bilinear corners as flat texels of this space, re-owned across maps.
            corners, weights = _bilinear_stencil(hexes, x, y, t, t, np.arange(self.sources[i].n_anchors))
            idx_out[idx] = self.owners()[corners + self.offsets[i]]
            w_out[idx] = weights
        if len(self._stencils) >= 4:
            self._stencils.clear()
        self._stencils[id(loc)] = (loc, (idx_out, w_out))    # loc pins its id
        return idx_out, w_out

    def slope(self, loc: Located) -> np.ndarray:
        if len(self.maps) == 1:
            return self.sources[0].slope(loc.data[1][0][2])
        return self.interpolate(self._anchor_field("slope"), loc)[:, 0]

    def danger(self, loc: Located, positions: np.ndarray) -> Optional[tuple[np.ndarray, Optional[np.ndarray]]]:
        if not self.has("danger"):
            return None
        base = self.coarse("danger", loc)[:, 0]
        return base, self.anchor_values("danger", loc)[:, 0]

    def parent_textures(self) -> dict[str, np.ndarray]:
        """Every texture all maps share, concatenated in anchor order (for the conservation report)."""
        keys = set(self.maps[0].textures)
        for f in self.maps[1:]:
            keys &= set(f.textures)
        out = {}
        for key in sorted(keys):
            parts = [np.asarray(f.textures[key]) for f in self.maps]
            if all(p.shape[1:] == parts[0].shape[1:] for p in parts):
                out[key] = np.concatenate(parts, axis=0)
        return out
