"""The coarse tier a refinement reads from: the globe, or a region.

At each child texel, a refinement (:func:`region.upsample.refine_fields`) needs:

- a smooth (bilinear) read of a parent texture, which detail is added to;
- the **anchor**: the parent texel it belongs to, whose value is the conservation target;
- the parent's slope, texel spacing, water surfaces and danger.

:class:`GlobalSource` serves the region tier from face textures,
:class:`RegionSource` the local tier from a region's hex textures.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
from scipy.spatial import KDTree

from region.resources import _danger_field, refined_slope
from region.sampling import GlobalSampler, bilinear_plan, bilinear_read, settle_owners
from worldgen.resources.terrain import global_slope

if TYPE_CHECKING:
    from core.world import World
    from region.upsample import RegionFields

EARTH_RADIUS_KM = 6371.0


def _bilinear_stencil(
    tiles: np.ndarray, x: np.ndarray, y: np.ndarray, h: int, w: int, owner: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Corners and weights of a bilinear read at texel coords ``(x, y)`` of ``(N, h, w)`` textures.

    Corners are mapped through ``owner`` (flat texel → anchor)."""
    x0 = np.clip(np.floor(x).astype(np.int64), 0, max(w - 2, 0))
    y0 = np.clip(np.floor(y).astype(np.int64), 0, max(h - 2, 0))
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx, fy = x - x0, y - y0
    base = tiles * (h * w)
    idx = np.stack([base + y0 * w + x0, base + y0 * w + x1, base + y1 * w + x0, base + y1 * w + x1], axis=1)
    weights = np.stack([(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy], axis=1)
    return owner[idx], weights


@dataclass
class Located:
    """Child texels located on the parent: anchors plus ``data`` (opaque, for the source's reads)."""

    anchors: np.ndarray          # (P,) flat parent texel index
    data: tuple


class ParentSource:
    """What a refinement reads from the tier above.  See the module docstring."""

    #: Number of parent texels.
    n_anchors: int
    #: Mean angular spacing (radians) of the parent's texels.
    spacing_rad: float

    def has(self, key: str) -> bool:
        raise NotImplementedError

    def locate(self, points: np.ndarray) -> Located:
        raise NotImplementedError

    def coarse(self, key: str, loc: Located) -> np.ndarray:
        """``(P, C)`` float64 bilinear read of parent texture ``key``."""
        raise NotImplementedError

    def anchor_values(self, key: str, loc: Located) -> np.ndarray:
        """``(P, C)`` float64 value of each anchor texel in texture ``key``."""
        raise NotImplementedError

    def owners(self) -> np.ndarray:
        """``(n_anchors,)`` the anchor owning each parent texel's *position*.

        A texel's own index unless its position is over-hang in a neighbour's
        territory; then the neighbour's texel there.
        """
        raise NotImplementedError

    def stencil(self, loc: Located) -> tuple[np.ndarray, np.ndarray]:
        """``((P, 4) anchors, (P, 4) weights)``: the bilinear read of a per-anchor field.

        Each corner is replaced by the anchor owning its position, so the read
        stays continuous across hex / face seams.
        """
        raise NotImplementedError

    def interpolate(
        self, per_anchor: np.ndarray, loc: Located,
        stencil: Optional[tuple[np.ndarray, np.ndarray]] = None,
    ) -> np.ndarray:
        """``(P, C)`` smooth read of an ``(n_anchors, C)`` per-anchor field (see :meth:`stencil`)."""
        idx, w = stencil if stencil is not None else self.stencil(loc)
        values = np.asarray(per_anchor, dtype=np.float64).reshape(self.n_anchors, -1)
        return np.einsum("pk,pkc->pc", w, values[idx])

    def slope(self, loc: Located) -> np.ndarray:
        """``(P,)`` surface slope (km/km) of the parent's elevation."""
        raise NotImplementedError

    def danger(self, loc: Located, positions: np.ndarray) -> Optional[tuple[np.ndarray, Optional[np.ndarray]]]:
        """``(base, targets)`` for the danger refinement, or None without danger.

        ``base`` is a smooth ``(P,)`` surface; ``targets`` each anchor group's
        mean, or None to keep ``base``'s own.
        """
        raise NotImplementedError


class GlobalSource(ParentSource):
    """The globe's face textures as the parent of a region."""

    def __init__(self, globe: "World") -> None:
        mesh = globe.mesh
        if "elevation" not in mesh.face_textures:
            raise ValueError("global mesh has no 'elevation' texture — run the elevation stage first")
        self.mesh = mesh
        self.sampler = GlobalSampler(mesh)
        self._danger = globe.get_feature("danger")
        n_faces, gh, gw, _ = mesh.face_textures["elevation"].shape
        self._hw = (gh, gw)
        self.n_anchors = n_faces * gh * gw
        self.spacing_rad = float(np.sqrt(np.mean(mesh.face_areas))) / max(gw - 1, 1)

    def has(self, key: str) -> bool:
        return key in self.mesh.face_textures

    def owners(self) -> np.ndarray:
        return self.sampler.owners(*self._hw)

    def stencil(self, loc: Located) -> tuple[np.ndarray, np.ndarray]:
        faces, u, v = loc.data
        gh, gw = self._hw
        x, y = GlobalSampler._texel_coords(u, v, gh, gw)
        return _bilinear_stencil(faces, x, y, gh, gw, self.owners())

    def locate(self, points: np.ndarray) -> Located:
        faces, u, v = self.sampler.locate(points)
        return Located(self.sampler.anchors(faces, u, v, *self._hw), (faces, u, v))

    def coarse(self, key: str, loc: Located) -> np.ndarray:
        return self.sampler.sample(self.mesh.face_textures[key], *loc.data)

    def anchor_values(self, key: str, loc: Located) -> np.ndarray:
        return GlobalSampler.anchor_values(self.mesh.face_textures[key], loc.anchors)

    def slope(self, loc: Located) -> np.ndarray:
        elev = np.asarray(self.mesh.face_textures["elevation"], dtype=np.float64)
        return self.sampler.sample(global_slope(self.mesh, elev), *loc.data)[:, 0]

    def danger(self, loc: Located, positions: np.ndarray) -> Optional[tuple[np.ndarray, Optional[np.ndarray]]]:
        if self._danger is None:
            return None
        # Per-face values made continuous with exact face means.
        return _danger_field(self.mesh, np.asarray(self._danger, dtype=np.float64))(positions), None


class RegionSource(ParentSource):
    """A region's refined hex textures as the parent of a local map.

    Points are located in the region's ``(s, t)`` frame on the hexagon with the
    nearest centre and read from its (over-hanging) texture.
    """

    def __init__(self, fields: "RegionFields") -> None:
        if "elevation" not in fields.textures:
            raise ValueError("region has no 'elevation' texture")
        self.fields = fields
        grid, t = fields.grid, fields.tex_size
        self._tree = KDTree(grid.centers)
        self.n_anchors = len(grid) * t * t
        self.spacing_rad = grid.texel_spacing_rad(t)
        self._slope: Optional[np.ndarray] = None
        self._owner: Optional[np.ndarray] = None
        # (loc, h, w) → bilinear_plan, for repeated reads at the same points.
        self._plans: dict[tuple, tuple] = {}

    def has(self, key: str) -> bool:
        return key in self.fields.textures

    def owners(self) -> np.ndarray:
        if self._owner is None:
            self._owner = settle_owners(self._locate_nearest(self.fields.positions.reshape(-1, 3)).anchors)
        return self._owner

    def stencil(self, loc: Located) -> tuple[np.ndarray, np.ndarray]:
        hexes, x, y = loc.data
        t = self.fields.tex_size
        return _bilinear_stencil(hexes, x, y, t, t, self.owners())

    def locate(self, points: np.ndarray) -> Located:
        # Anchor on the texel owning the nearest one, as GlobalSampler.anchors does.
        loc = self._locate_nearest(points)
        return Located(self.owners()[loc.anchors], loc.data)

    def _locate_nearest(self, points: np.ndarray) -> Located:
        f = self.fields
        grid, t = f.grid, f.tex_size
        s, tt, _ = f.geometry.ecef_to_local(points)
        _, hexes = self._tree.query(np.column_stack([s, tt]), k=1, workers=-1)
        hexes = np.asarray(hexes, dtype=np.int64)
        half = grid.radius                    # texture half-width (grid._TEXTURE_HALF_WIDTH == 1)
        # Continuous texel coordinates: texel i spans [i, i + 1), centre at i + 0.5.
        fx = (s - (grid.centers[hexes, 0] - half)) / (2.0 * half) * t
        fy = ((grid.centers[hexes, 1] + half) - tt) / (2.0 * half) * t
        col = np.clip(np.floor(fx).astype(np.int64), 0, t - 1)
        row = np.clip(np.floor(fy).astype(np.int64), 0, t - 1)
        anchors = hexes * (t * t) + row * t + col
        return Located(anchors, (hexes, np.clip(fx - 0.5, 0.0, t - 1.0), np.clip(fy - 0.5, 0.0, t - 1.0)))

    def _bilinear(self, texture: np.ndarray, loc: Located) -> np.ndarray:
        tex = np.asarray(texture)
        _, h, w, _ = tex.shape
        key = (id(loc), h, w)
        cached = self._plans.get(key)
        if cached is None:
            hexes, x, y = loc.data
            cached = (loc, bilinear_plan(hexes, x, y, h, w))    # loc pins its id
            if len(self._plans) >= 8:
                self._plans.clear()
            self._plans[key] = cached
        return bilinear_read(tex, cached[1])

    def coarse(self, key: str, loc: Located) -> np.ndarray:
        return self._bilinear(self.fields.textures[key], loc)

    def anchor_values(self, key: str, loc: Located) -> np.ndarray:
        return GlobalSampler.anchor_values(self.fields.textures[key], loc.anchors)

    def slope(self, loc: Located) -> np.ndarray:
        if self._slope is None:
            f = self.fields
            elev = f.textures["elevation"][..., 0]
            self._slope = refined_slope(elev, math.sqrt(max(f.texel_area_km2, 1e-9)))[..., None]
        return self._bilinear(self._slope, loc)[:, 0]

    def danger(self, loc: Located, positions: np.ndarray) -> Optional[tuple[np.ndarray, Optional[np.ndarray]]]:
        if "danger" not in self.fields.textures:
            return None
        # The region's danger is already continuous; each regional texel keeps its value.
        return self.coarse("danger", loc)[:, 0], self.anchor_values("danger", loc)[:, 0]
