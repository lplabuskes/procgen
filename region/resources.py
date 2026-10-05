"""Regional refinement of the resource, population and danger fields.

Runs inside :func:`region.upsample.build_region_fields` after the physical fields,
with the same conservation rule.  No noise of its own: all sub-grid variation
comes from the refined relief (slope, anomaly, treeline, shoreline), so the layers
line up with the terrain.

- **Forest density** — the treeline on the refined elevation.  Conserved over land.
- **Fertility** — richer valley floors, poorer slopes (``fertility_valley_gain``,
  ``fertility_slope_penalty``), forest penalty from the refined forest.
  Conserved over land.
- **Dominant crop** — reclassified, with climate area-averaged as for ``biome``.
- **Rural density** — farmers over refined land by ``fertility ** γ``, fishers
  (``fishing_density``) within ``fishing_shore_km`` of water, miners
  (``mining_density``) around their deposit (``mining_camp_km``).  Totals
  conserved exactly; none left on refined water unless a texel has no land.
- **Danger** — made continuous with :class:`region.smooth.FaceFanField`, then
  nudged up on rugged/forested ground (``danger_terrain_gain``) and
  renormalised per global texel.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
from scipy.spatial import KDTree

from region.sampling import anchor_means
from region.smooth import FaceFanField

if TYPE_CHECKING:
    from core.mesh import PolygonMesh
    from region.source import Located, ParentSource
    from region.upsample import RegionUpsampleParams
    from worldgen.resources.builder import ResourcesParams


_EARTH_RADIUS_KM = 6371.0


@dataclass
class RefineContext:
    """Shared inputs for the field refinements.

    Per-texel arrays are flat ``(P,)`` over the child's ``shape`` = ``(M, T, T)``
    textures; ``source`` is the parent tier.
    """

    source: "ParentSource"
    located: "Located"
    anchors: np.ndarray
    weights: np.ndarray          # in-hexagon inclusion weight
    n_anchors: int
    shape: tuple[int, int, int]
    positions: np.ndarray        # (P, 3)
    elevation: np.ndarray        # refined, km
    elev_coarse: np.ndarray      # bilinear global, km
    slope: np.ndarray            # refined, km/km
    land: np.ndarray             # refined dry ground
    anchor_land: np.ndarray      # the anchor texel is dry on the globe
    texel_km: float = 1.0        # child texel spacing
    mine_points: Optional[np.ndarray] = None   # (K, 3) worked deposits
    # (temperature, precipitation) as averaged for the biome; None to average here.
    climate_avg: Optional[tuple[np.ndarray, np.ndarray]] = None

    def has(self, key: str) -> bool:
        return self.source.has(key)

    def coarse(self, key: str) -> np.ndarray:
        return self.source.coarse(key, self.located)[:, 0]

    def anchor(self, key: str) -> np.ndarray:
        return self.source.anchor_values(key, self.located)[:, 0]

    def conserve_unit(self, values: np.ndarray, targets: np.ndarray, mask: np.ndarray, rounds: int = 8) -> np.ndarray:
        """:meth:`conserve_mean` for a ``[0, 1]`` field: scale and clip alternate, then settle exactly."""
        out = np.clip(values, 0.0, 1.0)
        for _ in range(rounds):
            out = np.clip(self.conserve_mean(out, targets, mask), 0.0, 1.0)
        return self._settle_unit(out, targets, mask)

    def _settle_unit(self, values: np.ndarray, targets: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Close what the alternation left open, exactly and inside ``[0, 1]``.

        Short groups move toward 1 in proportion to headroom; over ones scale
        down.  Needed for targets near 1, which the rounds only approach geometrically.
        """
        w = self.weights * mask
        means, counts = anchor_means(values, self.anchors, w, self.n_anchors)
        m = means[self.anchors, 0]
        t = np.clip(targets, 0.0, 1.0)
        ok = counts[self.anchors] > 0.0
        up = ok & (m < t)
        down = ok & (m > t) & (m > 1e-12)
        out = values.copy()
        out[up] = 1.0 - (1.0 - values[up]) * (1.0 - t[up]) / (1.0 - m[up])
        out[down] = values[down] * t[down] / m[down]
        return out

    def conserve_mean(self, values: np.ndarray, targets: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Scale each anchor group so its mean over ``mask`` texels is ``targets``.

        Groups with no ``mask`` texel, or a ~0 mean, are left alone.
        """
        w = self.weights * mask
        means, counts = anchor_means(values, self.anchors, w, self.n_anchors)
        m = means[self.anchors, 0]
        ok = (counts[self.anchors] > 0.0) & (np.abs(m) > 1e-12)
        return values * np.where(ok, targets / np.where(ok, m, 1.0), 1.0)


# Last danger field built, keyed by mesh and danger array (the solve is global).
_DANGER_CACHE: list = []


def _danger_field(mesh: "PolygonMesh", danger: np.ndarray) -> FaceFanField:
    if _DANGER_CACHE and _DANGER_CACHE[0] is mesh and np.array_equal(_DANGER_CACHE[1], danger):
        return _DANGER_CACHE[2]
    field = FaceFanField(mesh, danger)
    _DANGER_CACHE[:] = [mesh, np.array(danger, copy=True), field]
    return field


def _spread_into(values: np.ndarray, where: np.ndarray, shape: tuple[int, int, int], passes: int = 12) -> np.ndarray:
    """Fill NaN ``values`` inside ``where`` from known 4-neighbours, a ring per pass; flat ``(P,)``."""
    v = values.reshape(shape).copy()
    inside = where.reshape(shape)
    for _ in range(passes):
        todo = np.isnan(v) & inside
        if not todo.any():
            break
        pad = np.pad(v, ((0, 0), (1, 1), (1, 1)), constant_values=np.nan)
        nbrs = np.stack([pad[:, :-2, 1:-1], pad[:, 2:, 1:-1], pad[:, 1:-1, :-2], pad[:, 1:-1, 2:]])
        known = np.isfinite(nbrs)
        count = known.sum(axis=0)
        mean = np.where(known, nbrs, 0.0).sum(axis=0) / np.maximum(count, 1)
        v = np.where(todo & (count > 0), mean, v)
    return v.reshape(-1)


def _or_fallback(ctx: RefineContext, w: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """``w``, with every anchor group that has no weight in it replaced by ``fallback``."""
    mass, _ = anchor_means(w, ctx.anchors, ctx.weights, ctx.n_anchors)
    return np.where(mass[ctx.anchors, 0] > 0.0, w, fallback)


def _box_mean(values: np.ndarray, r: int) -> np.ndarray:
    """``(2r+1)``² mean over each ``(M, T, T)`` hex texture, edges replicated."""
    out = np.asarray(values, dtype=np.float64)
    for axis in (1, 2):
        pad = [(0, 0)] * 3
        pad[axis] = (r + 1, r)
        c = np.cumsum(np.pad(out, pad, mode="edge"), axis=axis)
        n = out.shape[axis]
        hi = np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
        lo = np.take(c, np.arange(n), axis=axis)
        out = (hi - lo) / (2 * r + 1)
    return out


def refined_slope(elevation: np.ndarray, texel_km: float) -> np.ndarray:
    """``(M, T, T)`` gradient magnitude (km/km) of refined elevation, per hex texture."""
    dz_r, dz_c = np.gradient(np.asarray(elevation, dtype=np.float64), axis=(1, 2))
    return np.hypot(dz_r, dz_c) / max(texel_km, 1e-9)


def refine_resources(
    ctx: RefineContext,
    textures: dict[str, np.ndarray],
    params: "RegionUpsampleParams",
    world: Optional["ResourcesParams"],
    fertility_exponent: float,
    danger: Optional[tuple[np.ndarray, Optional[np.ndarray]]],
) -> dict[str, np.ndarray]:
    """The refined resource, rural-density and danger textures present on the parent.

    ``danger`` is :meth:`ParentSource.danger`'s ``(base, targets)``, or None.
    """
    from region.upsample import _box_filter
    from worldgen.resources.builder import ResourcesParams
    world = world or ResourcesParams()
    shape = ctx.shape
    out: dict[str, np.ndarray] = {}

    def texture(values: np.ndarray, dtype=np.float32) -> np.ndarray:
        return values.reshape(*shape, 1).astype(dtype)

    slope_n = np.clip(ctx.slope / max(float(params.slope_saturation), 1e-9), 0.0, 1.0)
    dry = ctx.land & ctx.anchor_land

    # ── forest density: the treeline on the refined relief ───────────────────
    forest = None
    if ctx.has("forest_density"):
        gp = world.forest_density
        falloff = max(float(gp.treeline_falloff_km), 1e-3)

        def treeline(e: np.ndarray) -> np.ndarray:
            return np.clip((float(gp.treeline_elevation_km) - e) / falloff, 0.0, 1.0)

        coarse = ctx.coarse("forest_density")
        forest = coarse * (treeline(ctx.elevation) + 0.05) / (treeline(ctx.elev_coarse) + 0.05)
        forest = np.where(ctx.land, ctx.conserve_unit(forest, ctx.anchor("forest_density"), dry), 0.0)
        out["forest_density"] = texture(forest)

    # ── fertility: valley floors richer, steep slopes poorer ─────────────────
    fertility = None
    if ctx.has("fertility"):
        coarse = ctx.coarse("fertility")
        valley = np.clip(1.0 - float(params.fertility_valley_gain) * (ctx.elevation - ctx.elev_coarse), 0.5, 1.5)
        mod = valley * (1.0 - float(params.fertility_slope_penalty) * slope_n)
        s = float(world.fertility.forest_penalty_strength)
        if forest is not None and s > 0.0:
            forest_coarse = np.clip(ctx.coarse("forest_density"), 0.0, 1.0)
            mod = mod * (1.0 - s * forest) / np.maximum(1.0 - s * forest_coarse, 1e-3)
        target = ctx.anchor("fertility")
        fertility = ctx.conserve_unit(coarse * mod, target, dry)
        # Water keeps the parent's aquatic fertility; newly flooded texels take their neighbours'.
        aquatic = _spread_into(np.where(~ctx.land & ~ctx.anchor_land, target, np.nan), ~ctx.land, shape)
        fertility = np.clip(np.where(ctx.land, fertility, np.where(np.isfinite(aquatic), aquatic, coarse)), 0.0, 1.0)
        out["fertility"] = texture(fertility)

    # ── dominant crop: reclassified from the refined fields ──────────────────
    if fertility is not None and ctx.has("dominant_crop") and "temperature" in textures and "precipitation" in textures:
        from worldgen.resources.crops import classify_crops
        k = int(params.biome_smoothing)
        temperature, precipitation = ctx.climate_avg or (
            _box_filter(textures["temperature"], k), _box_filter(textures["precipitation"], k))
        out["dominant_crop"] = classify_crops(
            temperature,
            precipitation,
            ctx.elevation.reshape(*shape, 1),
            fertility.reshape(*shape, 1),
            (~ctx.land).reshape(*shape, 1).astype(np.float32),
        )

    # ── rural density: each global texel's people over its refined land ──────
    if ctx.has("rural_density"):
        everywhere = np.ones(len(ctx.land), dtype=bool)
        target = ctx.anchor("rural_density")
        fish_target = (
            np.clip(ctx.anchor("fishing_density"), 0.0, target) if ctx.has("fishing_density")
            else np.zeros(len(target))
        )
        fert = fertility if fertility is not None else np.ones(len(target))
        w = np.where(ctx.land, np.clip(fert, 0.0, 1.0) ** float(fertility_exponent), 0.0)
        # All-barren land: spread evenly over it; no land left: keep them in place.
        w = _or_fallback(ctx, w, ctx.land.astype(np.float64))
        w = _or_fallback(ctx, w, np.ones(len(w)))
        mine_target = (
            np.clip(ctx.anchor("mining_density"), 0.0, target - fish_target) if ctx.has("mining_density")
            else np.zeros(len(target))
        )
        rho = ctx.conserve_mean(w, target - fish_target - mine_target, everywhere)
        if ctx.has("mining_density"):
            # Land weighted by a Gaussian about the nearest worked deposit; with
            # no such land, miners stay with the farmers.
            if ctx.mine_points is not None and len(ctx.mine_points):
                d, _ = KDTree(ctx.mine_points).query(ctx.positions, workers=-1)
                camp_rad = float(params.mining_camp_km) / _EARTH_RADIUS_KM
                q = np.where(ctx.land & (ctx.weights > 0), (np.asarray(d) / camp_rad) ** 2, np.inf)
                # Relative to each anchor's closest land, so distant deposits don't underflow.
                q_min = np.full(ctx.n_anchors, np.inf)
                np.minimum.at(q_min, ctx.anchors, q)
                finite = np.isfinite(q)
                rel = np.subtract(q, q_min[ctx.anchors], out=np.full_like(q, np.inf), where=finite)
                near_mine = np.exp(-rel)
            else:
                near_mine = np.zeros(len(w))
            wm = _or_fallback(ctx, near_mine, w)
            mine = ctx.conserve_mean(wm, mine_target, everywhere)
            out["mining_density"] = texture(mine)
            rho = rho + mine
        if ctx.has("fishing_density"):
            # Land weighted by nearby water fraction; with no shore, fishers join the farmers.
            r = int(round(float(params.fishing_shore_km) / max(ctx.texel_km, 1e-9)))
            near = _box_mean((~ctx.land).astype(np.float64).reshape(shape), max(r, 1)).reshape(-1)
            ws = _or_fallback(ctx, np.where(ctx.land, near, 0.0), w)
            fish = ctx.conserve_mean(ws, fish_target, everywhere)
            out["fishing_density"] = texture(fish)
            rho = rho + fish
        out["rural_density"] = texture(rho)

    # ── danger: continuous across faces, nudged by terrain ───────────────────
    if danger is not None:
        base, targets = danger
        # Area-averaged (texel slope is too noisy); water scores as flat ground so shores aren't seams.
        score = np.where(ctx.land, slope_n + (forest if forest is not None else 0.0), 0.0)
        score = _box_filter(score.reshape(*shape, 1), int(params.biome_smoothing)).reshape(-1)
        nudged = base * np.exp(float(params.danger_terrain_gain) * score)
        if targets is None:
            means, _ = anchor_means(base, ctx.anchors, ctx.weights, ctx.n_anchors)
            targets = means[ctx.anchors, 0]
        nudged = ctx.conserve_mean(nudged, targets, np.ones(len(base), dtype=bool))
        out["danger"] = texture(nudged)

    return out
