"""Small lakes — basins in a tier's refined relief, too small for its parent.

The globe keeps lakes of ``min_lake_pixels`` global texels and up; a region (and
a local map, one tier down) finds its own in its refined relief, on a raster at
its texel spacing over the map plus a margin:

- **Basins.** Priority-flood (:mod:`region.drainage`) from the outlets: water
  already there (sea, the parent's lakes), uncovered nodes and the raster edge,
  and the parent's channels (``drains``) widened by
  ``drain_clearance_parent_texels`` — a river that big would have cut through
  any sill on its course, and the tier re-routes it anywhere within about a
  parent texel of it, so a lake any closer would sit tangent to it or be
  clipped by it.
- **Water balance.** A basin's inflow is its catchment's runoff (Fu's Budyko
  curve from precipitation and PET, the globe's PET formula); it fills to its
  spill level when that covers evaporation off the full lake, else to the level
  where it does, as the globe's endorheic lakes.  None where it's arid
  (precipitation / PET under ``lake_min_aridity_index``: playas, not lakes) or
  on ice cap.
- **Breaching.** Erosion the relief doesn't model cuts sills fastest on steep
  ground, so a lake must be deeper than the parent's surface drops over
  ``lake_breach_parent_texels`` (parent slope × length) — lakes gather on flats
  and valley floors — and than ``lake_min_depth_km``, and at least
  ``lake_min_texels`` big.  In the cold the breach length shrinks by
  ``lake_glacial_factor``: glacially scoured ground is lake country.
- **Size cap.** Optionally no bigger than ``lake_max_parent_texels`` parent
  texels: a basin the parent could have drawn a lake in, but didn't, stays dry
  (the local tier only adds lakes below the region's resolution).

All of it on the relief smoothed over ``smoothing_parent_texels`` (a Gaussian,
water left out): the raw texels carry texel-scale detail and, a tier down, the
conservation staircase (metre steps at parent-texel edges), whose pits and sills
would give lacy shores, speckle islands and lakes filling whole parent texels.
The lake surface is then spread a node onto the shore and every texel whose
smoothed height is below it (and not already water) becomes lake; the caller
edits the raw relief to agree (:func:`region.upsample.reshape_to_target`).
Deterministic; no noise of its own.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, cast

import numpy as np
from scipy import ndimage

from region.drainage import accumulate, priority_flood
from region.grid import RegionHexGrid
from worldgen.hydrology.builder import HydrologyParams

if TYPE_CHECKING:
    from region.source import ParentSource

EARTH_RADIUS_KM = 6371.0

# Fu's Budyko shape parameter (the global mean fit).
_BUDYKO_W = 2.6

# Annual means (°C) between which glacial lake country fades out: roughly the
# ice-age ice sheets' reach (now ~0 °C in central Canada, ~10 °C at their rim).
_GLACIAL_COLD_C = 0.0
_GLACIAL_WARM_C = 10.0


def budyko_runoff_ratio(pet: np.ndarray, precipitation: np.ndarray) -> np.ndarray:
    """Share of precipitation that runs off (Fu's curve), from PET and precipitation (same units)."""
    phi = np.asarray(pet, dtype=np.float64) / np.maximum(np.asarray(precipitation, dtype=np.float64), 1e-9)
    phi = np.minimum(phi, 1e6)
    return np.clip((1.0 + phi ** _BUDYKO_W) ** (1.0 / _BUDYKO_W) - phi, 0.0, 1.0)


def potential_evaporation(temperature: np.ndarray, hydrology: Optional[HydrologyParams] = None) -> np.ndarray:
    """PET (mm/yr) from annual mean temperature (°C), as the globe's lake shrinking."""
    hp = hydrology or HydrologyParams()
    return np.clip(
        float(hp.pet_baseline_mm_yr) + float(hp.pet_per_degree_mm_yr) * np.maximum(temperature, 0.0),
        0.0, float(hp.pet_max_mm_yr),
    )


def glacial_breach_scale(temperature: np.ndarray, factor: float) -> np.ndarray:
    """Breach-length multiplier: ``factor`` at an annual mean ≤ ``_GLACIAL_COLD_C``, 1 from ``_GLACIAL_WARM_C``.

    Ice-age glaciers scoured the cold lands into basins and left hummocky
    moraine (the lake districts of Canada and Finland), so lakes there survive
    shallower.
    """
    ramp = np.clip((np.asarray(temperature, dtype=np.float64) - _GLACIAL_COLD_C)
                   / (_GLACIAL_WARM_C - _GLACIAL_COLD_C), 0.0, 1.0)
    return float(factor) + (1.0 - float(factor)) * ramp


def drain_nodes(
    grid: RegionHexGrid, segments: np.ndarray, lo: float, step: float, n: int,
) -> np.ndarray:
    """``(n²,)`` mask of the raster nodes the ``(K, 2, 3)`` unit-vector segments cross."""
    hit = np.zeros(n * n, dtype=bool)
    segments = np.asarray(segments, dtype=np.float64).reshape(-1, 2, 3)
    if len(segments) == 0:
        return hit
    s, t, valid = grid.geometry.ecef_to_local(segments.reshape(-1, 3))
    x = ((s - lo) / step).reshape(-1, 2)
    y = ((t - lo) / step).reshape(-1, 2)
    hi = n - 1
    keep = (valid.reshape(-1, 2).all(axis=1)
            & (x.max(axis=1) >= 0) & (x.min(axis=1) <= hi) & (y.max(axis=1) >= 0) & (y.min(axis=1) <= hi))
    x, y = x[keep], y[keep]
    if len(x) == 0:
        return hit
    # Two samples per node step along each segment.
    count = np.ceil(2.0 * np.hypot(x[:, 1] - x[:, 0], y[:, 1] - y[:, 0])).astype(np.int64) + 1
    seg = np.repeat(np.arange(len(x)), count)
    frac = (np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count)) / np.repeat(
        np.maximum(count - 1, 1), count)
    xs = np.rint(x[seg, 0] + frac * (x[seg, 1] - x[seg, 0])).astype(np.int64)
    ys = np.rint(y[seg, 0] + frac * (y[seg, 1] - y[seg, 0])).astype(np.int64)
    inside = (xs >= 0) & (xs < n) & (ys >= 0) & (ys < n)
    hit[ys[inside] * n + xs[inside]] = True
    return hit


def _neighbour_min_2d(values: np.ndarray) -> np.ndarray:
    """Per-node minimum over the 8 neighbours of an ``(n, n)`` array (edges: +inf)."""
    padded = np.pad(values, 1, mode="constant", constant_values=np.inf)
    n = values.shape[0]
    out = np.full_like(values, np.inf)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr or dc:
                out = np.minimum(out, padded[1 + dr:1 + dr + n, 1 + dc:1 + dc + n])
    return out


def smooth_land(elevation: np.ndarray, water: np.ndarray, sigma_nodes: float) -> np.ndarray:
    """``(n, n)`` ``elevation`` Gaussian-smoothed over ``sigma_nodes``, averaging land (``~water``) only.

    Water nodes keep their height, so neither a lake bed nor the sea floor drags its shore down.
    """
    e = np.asarray(elevation, dtype=np.float64)
    if sigma_nodes <= 0.0:
        return e.copy()
    land = ~np.asarray(water, dtype=bool)
    w = ndimage.gaussian_filter(land.astype(np.float64), sigma_nodes, mode="nearest")
    v = ndimage.gaussian_filter(np.where(land, e, 0.0), sigma_nodes, mode="nearest")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(w > 1e-6, v / w, e)
    return np.where(land, out, e)


def basin_lake_levels(
    elevation: np.ndarray, outlet: np.ndarray, n: int, *,
    runoff: Optional[np.ndarray] = None, evaporation: Optional[np.ndarray] = None,
    breach: Optional[np.ndarray] = None, allowed: Optional[np.ndarray] = None,
    min_depth: float = 0.0, min_nodes: int = 1, max_nodes: int = 0,
) -> np.ndarray:
    """``(n²,)`` lake surface on every lake node of an ``n × n`` raster, ``+inf`` elsewhere.

    Basins are the priority-flood's raised nodes; ``runoff`` (per node) and
    ``evaporation`` (per node, same units) shrink a basin whose inflow can't
    cover evaporation off its full lake (none given → every basin is full).
    A lake is kept when at least ``min_nodes`` big (and, ``max_nodes`` > 0, at
    most that), and deeper than both
    ``min_depth`` and its nodes' mean ``breach`` (per node, elevation units),
    and (given ``allowed``, per node) mostly over allowed nodes.
    """
    elevation = np.asarray(elevation, dtype=np.float64)
    parent, order, filled = priority_flood(elevation, np.asarray(outlet, dtype=bool), n)
    out = np.full(n * n, np.inf)
    basin = (filled > elevation) & ~outlet
    labels, k = cast(tuple[np.ndarray, int],
                     ndimage.label(basin.reshape(n, n), structure=np.ones((3, 3), dtype=bool)))
    if k == 0:
        return out
    labels = labels.ravel()
    ids = np.arange(1, k + 1)
    spill = np.asarray(ndimage.maximum(filled, labels, ids), dtype=np.float64)
    level = spill.copy()
    if runoff is not None and evaporation is not None:
        inflow = np.asarray(ndimage.maximum(accumulate(parent, order, np.where(outlet, 0.0, runoff)), labels, ids))
        loss = np.asarray(ndimage.sum(evaporation, labels, ids))
        nodes_of = ndimage.value_indices(labels.reshape(n, n), ignore_value=0) if np.any(inflow < loss) else {}
        for i in np.flatnonzero(inflow < loss):
            rows, cols = nodes_of[i + 1]
            nodes = rows * n + cols
            nodes = nodes[np.argsort(elevation[nodes], kind="stable")]
            # The smallest set of lowest nodes whose evaporation reaches the inflow.
            j = int(np.searchsorted(np.cumsum(evaporation[nodes]), inflow[i], side="left"))
            level[i] = elevation[nodes[min(j, len(nodes) - 1)]] if j > 0 else -np.inf
    node_level = np.where(labels > 0, level[np.maximum(labels, 1) - 1], -np.inf)
    wet = node_level > elevation
    # Breached: too shallow or too small.
    depth = np.asarray(ndimage.maximum(np.where(wet, node_level - elevation, 0.0), labels, ids))
    size = np.bincount(labels[wet], minlength=k + 1)[1:]
    keep = (depth >= float(min_depth)) & (size >= int(min_nodes)) & (depth > 0.0)
    if max_nodes > 0:
        keep &= size <= int(max_nodes)
    if breach is not None:
        keep &= depth >= np.asarray(ndimage.mean(breach, labels, ids))
    if allowed is not None:
        keep &= np.asarray(ndimage.mean(np.asarray(allowed, dtype=np.float64), labels, ids)) >= 0.5
    lake = wet & keep[np.maximum(labels, 1) - 1] & (labels > 0)
    out[lake] = node_level[lake]
    return out


def find_small_lakes(
    source: "ParentSource",
    grid: RegionHexGrid,
    elevation: np.ndarray,
    water: np.ndarray,
    *,
    temperature: Optional[np.ndarray] = None,
    precipitation: Optional[np.ndarray] = None,
    frozen: Optional[np.ndarray] = None,
    drains: Optional[np.ndarray] = None,
    min_depth_km: float = 0.002,
    breach_parent_texels: float = 6.0,
    min_aridity_index: float = 0.2,
    glacial_factor: float = 0.3,
    min_texels: int = 2,
    max_parent_texels: float = 0.0,
    smoothing_parent_texels: float = 0.5,
    drain_clearance_parent_texels: float = 0.0,
    inherited: Optional[np.ndarray] = None,
    hydrology: Optional[HydrologyParams] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The tier's own lakes over its ``(M, T, T)`` refined ``elevation``: ``(lake, surface, smoothed)``.

    ``water`` is the water already there (an outlet; never made lake),
    ``temperature`` / ``precipitation`` the annual mean (°C) / total (mm) per
    texel (both or neither; neither → every basin fills; a basin mostly where
    precipitation / PET < ``min_aridity_index`` holds no lake), ``frozen`` ground
    that never thaws (a basin mostly on ice cap holds no lake); the breach
    length shrinks by ``glacial_factor`` in the cold (:func:`glacial_breach_scale`).  ``drains`` the
    parent's channels as ``(K, 2, 3)`` unit-vector segments, with no lake within
    ``drain_clearance_parent_texels`` of them.  A lake must be
    deeper than ``min_depth_km`` and than the parent's surface drops over
    ``breach_parent_texels`` of its texels (parent slope × that length).
    ``inherited`` (``(M, T, T)`` bool, part of ``water``) is the parent's small
    lakes: smoothed over as land (the caller re-shores them on ``smoothed``),
    and no ``drains`` through them (the parent's channel runs under them).
    Basins and shores are of the relief smoothed over ``smoothing_parent_texels``
    (:func:`smooth_land`); ``smoothed`` is that relief at each texel.
    ``surface`` is the lake surface (km) on lake texels and about a parent
    texel onto their shores, ``+inf`` elsewhere: the band where the raw relief
    is edited to agree, wide enough to keep each parent texel's mean.
    """
    m, tex = elevation.shape[0], elevation.shape[1]
    step = 2.0 * grid.radius / tex
    lo = -grid.radius               # the textures reach ~2 radii past the hexes
    n = int(np.ceil((1.0 - 2.0 * lo) / step)) + 1
    cols, rows = np.meshgrid(np.arange(n), np.arange(n))
    s, t = lo + step * cols.ravel(), lo + step * rows.ravel()
    hexes, hr, hc, covered = grid.nearest_texels(s, t, tex)

    e = elevation[hexes, hr, hc].astype(np.float64)
    loc = source.locate(grid.geometry.local_to_ecef(s, t))
    # Uncovered nodes are outlets at the parent's (smooth) height.
    e[~covered] = source.coarse("elevation", loc)[~covered, 0]
    breach = source.slope(loc) * float(breach_parent_texels) * source.spacing_rad * EARTH_RADIUS_KM
    edge = np.zeros((n, n), dtype=bool)
    edge[[0, -1], :] = edge[:, [0, -1]] = True
    wet = covered & water[hexes, hr, hc]
    outlet = ~covered | wet | edge.ravel()
    seed = np.zeros(n * n, dtype=bool)
    if inherited is not None:
        seed = covered & inherited[hexes, hr, hc]
        wet &= ~seed
    n_t = source.spacing_rad / max(grid.texel_spacing_rad(tex), 1e-12)
    if drains is not None:
        on = drain_nodes(grid, drains, lo, step, n)
        if drain_clearance_parent_texels > 0.0 and on.any():
            pair = grid.geometry.local_to_ecef(np.array([0.5, 0.5 + step]), np.array([0.5, 0.5]))
            reach = float(drain_clearance_parent_texels) * source.spacing_rad / float(np.linalg.norm(pair[1] - pair[0]))
            on = (np.asarray(ndimage.distance_transform_edt(~on.reshape(n, n))) <= reach).ravel()
        outlet |= on & ~seed
    sigma = float(smoothing_parent_texels) * n_t
    e = smooth_land(e.reshape(n, n), wet.reshape(n, n), sigma).ravel()

    runoff = evaporation = None
    allowed = np.ones(n * n, dtype=bool)
    if temperature is not None and precipitation is not None:
        pet = potential_evaporation(temperature[hexes, hr, hc].astype(np.float64), hydrology)
        rain = np.maximum(precipitation[hexes, hr, hc].astype(np.float64), 0.0)
        runoff, evaporation = rain * budyko_runoff_ratio(pet, rain), pet
        allowed &= rain >= float(min_aridity_index) * pet
        breach = breach * glacial_breach_scale(temperature[hexes, hr, hc], glacial_factor)
    if frozen is not None:
        allowed &= ~frozen[hexes, hr, hc]

    levels = basin_lake_levels(
        e, outlet, n, runoff=runoff, evaporation=evaporation,
        breach=breach, allowed=allowed, min_depth=float(min_depth_km), min_nodes=int(min_texels),
        max_nodes=int(np.ceil(float(max_parent_texels) * n_t * n_t)) if max_parent_texels > 0.0 else 0,
    ).reshape(n, n)
    # Carry each surface a node onto its shore so texels between nodes see it
    # (not onto outlets: nothing stands on a drain).
    levels = np.where(np.isfinite(levels) | outlet.reshape(n, n), levels, _neighbour_min_2d(levels))
    band = levels
    for _ in range(max(int(np.ceil(n_t)), 1)):
        band = np.where(np.isfinite(band) | outlet.reshape(n, n), band, _neighbour_min_2d(band))

    local = grid.texture_local(tex).reshape(-1, 2)
    x, y = (local[:, 0] - lo) / step, (local[:, 1] - lo) / step
    c, r = np.rint(x).astype(np.int64), np.rint(y).astype(np.int64)
    inside = (c >= 0) & (c < n) & (r >= 0) & (r < n)
    level = np.full(m * tex * tex, np.inf)
    level[inside] = levels[r[inside], c[inside]]
    level = level.reshape(m, tex, tex)
    surface = np.full(m * tex * tex, np.inf)
    surface[inside] = band[r[inside], c[inside]]
    surface = np.where(water, np.inf, surface.reshape(m, tex, tex))
    smoothed = ndimage.map_coordinates(e.reshape(n, n), [y, x], order=1, mode="nearest").reshape(m, tex, tex)
    lake = np.isfinite(level) & ~water & (smoothed < level)
    return lake, np.where(lake, level, surface), smoothed
