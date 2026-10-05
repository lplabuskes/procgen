"""Regional refinement of the global fields — upsampling with conservation.

See ``docs/plans/regional_local_maps_design.md``.  For every field:

1. read the global texture bilinearly at each regional texel;
2. add detail above the global grid's Nyquist limit;
3. force each global texel's average back to its global value
   (:func:`conserve_smooth`, a smooth correction, so no steps at parent texel edges).

Downsampling a region therefore recovers the global map (:func:`conservation_report`).

Elevation is the only field with its own noise stream.  ``biome`` thresholds
temperature and precipitation, so independent noise in either would dissolve
biome boundaries into speckle; deriving both from the relief makes them follow
ridges and valleys instead.

- **Elevation** — slope-proportional fBm with a floor, faded under shallow water;
  near water the coast smoothness restyles it (:func:`coast_style`).  Additive.
- **Temperature** — lapse-rate correction on the refined relief.  Additive.
- **Precipitation** — orographic response to the relief.  Multiplicative.
- **Biome** — reclassified from the refined climate with the global classifier.
- **Ocean / lake masks** — water surfaces are taken from the globe (sea level, a
  lake's Priority-Flood level); only the shoreline moves, by a level-constrained
  flood seeded from the global mask.

Octaves run from the global grid's Nyquist to the regional grid's, so a region no
finer than the globe gets no detail (``RegionFields.detail_octaves == 0``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

from core.params import param_field as _f, paramclass
from region.geometry import N_REGIONS, RegionGeometry
from region.grid import RegionHexGrid
from region.resources import RefineContext, refine_resources, refined_slope
from region.sampling import GlobalSampler, anchor_means
from region.source import GlobalSource, Located, ParentSource
from utils.simplex import fbm_sum, make_perm
from worldgen.resources.minerals import mineral_workings, site_workers

if TYPE_CHECKING:
    from core.world import World
    from worldgen.resources.builder import ResourcesParams

EARTH_RADIUS_KM = 6371.0

# Hard-coded as the shore by worldgen.geology.elevation and worldgen.hydrology.builder.
SEA_LEVEL_KM = 0.0

_SEED_SALT_ELEVATION = 0x9E37
_SEED_SALT_COAST = 0xC0A5


@paramclass
class RegionUpsampleParams:
    """Tuning for regional detail generation."""

    n_cols: int = _f(14, label="Hex columns per region", min=4, max=64)
    # 64 gives ~4 km texels vs the globe's ~15 km: two detail octaves.
    tex_size: int = _f(64, label="Hex texture size", min=4, max=128)
    # Amplitude = relief_factor × detail wavelength × local slope.  Global slopes
    # at ~15 km only reach ~0.007 km/km, hence the small saturation point.
    relief_factor: float = _f(1.0, label="Detail relief factor", min=0.0, max=8.0)
    slope_reference: float = _f(
        0.005, label="Slope saturation (km/km)", min=0.0005, max=0.05, step=0.0005
    )
    slope_floor: float = _f(0.10, label="Flat-ground detail floor", min=0.0, max=1.0)
    max_relief_km: float = _f(1.0, label="Max detail relief (km)", min=0.0, max=5.0)
    # 0.5/km: a 400 m sub-grid ridge catches ~20% more rain, a hollow ~20% less.
    orographic_gain: float = _f(
        0.5, label="Orographic precip gain (/km)", min=0.0, max=3.0, step=0.05
    )
    lapse_rate_c_per_km: float = _f(6.5, label="Lapse rate (°C/km)", min=0.0, max=15.0)
    # Detail fades over this depth so shallow lake beds stay submerged (median
    # lake depth ~40 m); 0 disables the fade.
    water_relief_depth_km: float = _f(
        0.1, label="Underwater detail fade (km)", min=0.0, max=2.0, step=0.05
    )
    # Off: masks are copied from the anchor texel and the coast stays blocky.
    refine_water_masks: bool = _f(True, label="Refine shoreline", min=0, max=1)
    # How far water may advance into global land texels, in regional texels;
    # about one global texel keeps the shore inside the coast's own texels.
    water_spread_texels: int = _f(4, label="Shoreline spread (texels)", min=0, max=16)
    # The biome classifier reads box-averaged climate: ~1 °C texel-to-texel swings
    # would otherwise dither every boundary.  The climate layers stay sharp.
    # 0 or 1 disables it.
    biome_smoothing: int = _f(5, label="Biome climate averaging (texels)", min=0, max=15)
    octaves: int = _f(4, label="Detail octaves (max)", min=0, max=8)
    lacunarity: float = _f(2.0, label="Detail lacunarity", min=1.1, max=4.0)
    persistence: float = _f(0.5, label="Detail persistence", min=0.1, max=0.95)
    nyquist_safety: float = _f(0.5, label="Nyquist safety factor", min=0.1, max=1.0)
    # ── coast style (from ``coast_smoothness``).  Coasts are nearly flat at the
    # parent's scale, so near water the amplitude is ``wander × detail wavelength ×
    # parent slope`` and the shore moves ~``wander`` wavelengths whatever the
    # gradient.  Smooth (1): little wander, low persistence; rugged (0): far
    # wander, high persistence, and skerries (no underwater fade over the sea).
    # Fades out where the parent water fraction is within ``coast_band`` of 0 or 1.
    coast_band: float = _f(0.25, label="Coast style band (water fraction)", min=0.01, max=0.5, step=0.01)
    coast_persistence_smooth: float = _f(0.25, label="Smooth-coast persistence", min=0.05, max=0.95, step=0.05)
    coast_persistence_rugged: float = _f(0.8, label="Rugged-coast persistence", min=0.05, max=0.95, step=0.05)
    coast_wander_smooth: float = _f(0.03, label="Smooth-coast wander (wavelengths)", min=0.0, max=2.0, step=0.01)
    coast_wander_rugged: float = _f(0.5, label="Rugged-coast wander (wavelengths)", min=0.0, max=2.0, step=0.01)
    # ── shore reshaping (sea and lakes).  Conservation pins a coast to the
    # parent's texel staircase, so a target coastline is drawn instead: the
    # elevation box-blurred over ``coast_blur_texels`` parent texels, fine detail
    # kept by ruggedness, plus bay-scale noise one octave below the detail band on
    # rugged coasts.  Wrong-side texels are set just across the surface and the
    # rest of their parent texel absorbs the difference by scaling (parent means
    # exact).  Parent texels needing more than ``coast_max_scale`` are left alone.
    reshape_coast: bool = _f(True, label="Reshape shores", min=0, max=1)
    coast_blur_texels: float = _f(1.5, label="Coast blur (parent texels)", min=0.0, max=4.0, step=0.25)
    coast_target_wander_smooth: float = _f(
        0.0, label="Smooth-coast bay size (wavelengths)", min=0.0, max=2.0, step=0.05
    )
    coast_target_wander_rugged: float = _f(
        0.5, label="Rugged-coast bay size (wavelengths)", min=0.0, max=2.0, step=0.05
    )
    coast_max_scale: float = _f(1.0, label="Coast edit max scale", min=0.0, max=5.0, step=0.1)
    # Bounds on how far past the surface an edited texel is set (km).
    coast_edit_min_km: float = _f(0.001, label="Coast edit min depth/height (km)", min=0.0, max=0.05, step=0.001)
    coast_edit_max_km: float = _f(0.02, label="Coast edit max depth/height (km)", min=0.0, max=0.2, step=0.005)
    # ── resources, population and danger (region/resources.py) ──
    # Refined slope (km/km) at which slope-driven terms saturate.
    slope_saturation: float = _f(
        0.02, label="Slope saturation, resources (km/km)", min=0.002, max=0.2, step=0.002
    )
    # Fertility gained per km below the coarse surface (valley floors), lost above.
    fertility_valley_gain: float = _f(
        2.0, label="Fertility valley gain (/km)", min=0.0, max=10.0, step=0.25
    )
    # Fraction of fertility lost on saturated slopes.
    fertility_slope_penalty: float = _f(
        0.5, label="Fertility slope penalty", min=0.0, max=1.0, step=0.05
    )
    # Danger × exp(gain · (slope + forest)), renormalised per global texel.
    danger_terrain_gain: float = _f(
        1.0, label="Danger terrain gain", min=0.0, max=4.0, step=0.1
    )
    # Inland band of the refined shore holding the fishers' rural density.
    fishing_shore_km: float = _f(
        2.0, label="Fishing shore band (km)", min=0.1, max=20.0, step=0.1
    )
    # Gaussian scale of the miners' rural density around their deposit.
    mining_camp_km: float = _f(
        3.0, label="Mining camp radius (km)", min=0.1, max=20.0, step=0.1
    )


@dataclass
class RegionFields:
    """Refined per-hex textures for one region (or local map), mirroring ``mesh.face_textures``.

    ``textures`` uses the globe's keys as ``(M, T, T, C)`` arrays over the map's
    ``M`` hexagons, plus ``"danger"`` (per face on the globe, a texture here).
    ``filled_height`` is the lake surface on lake texels and ground elsewhere;
    the tier below reads its lake levels there.
    """

    region_id: int
    geometry: RegionGeometry
    grid: RegionHexGrid
    tex_size: int
    positions: np.ndarray                    # (M, T, T, 3) unit sphere
    in_hex: np.ndarray                       # (M, T, T) bool — aggregation weight
    anchors: np.ndarray                      # (M, T, T) global texel index
    faces: np.ndarray                        # (M, T, T) owning global face
    face_uv: np.ndarray                      # (M, T, T, 2) texture coords on that face
    textures: dict[str, np.ndarray]
    detail_octaves: int
    texel_area_km2: float
    # Local maps only: each texel's anchor on the region (``hex*T*T + row*T + col``),
    # the conservation target one tier up (``anchors``/``faces`` stay global), and
    # which texels lie inside the parent hexagon.
    parent_anchors: Optional[np.ndarray] = None   # (M, T, T)
    in_map: Optional[np.ndarray] = None           # (M, T, T) bool

    @property
    def n_hexes(self) -> int:
        return len(self.grid)

    @property
    def domain(self) -> np.ndarray:
        """(M, T, T) bool: texels inside their hexagon (and, on a local map, the parent hexagon)."""
        return self.in_hex if self.in_map is None else self.in_hex & self.in_map


# ── detail machinery ─────────────────────────────────────────────────────────


def _octave_bands(
    base_frequency: float, max_frequency: float, params: RegionUpsampleParams
) -> tuple[np.ndarray, np.ndarray]:
    """Frequencies/amplitudes for the band between the global and regional grids."""
    freqs: list[float] = []
    amps: list[float] = []
    freq, amp = float(base_frequency), 1.0
    for _ in range(max(int(params.octaves), 0)):
        if freq > max_frequency:
            break
        freqs.append(freq)
        amps.append(amp)
        amp *= params.persistence
        freq *= params.lacunarity
    return np.asarray(freqs, dtype=np.float64), np.asarray(amps, dtype=np.float64)


def _normalized_fbm(
    points: np.ndarray, seed: int, freqs: np.ndarray, amps: np.ndarray
) -> np.ndarray:
    """fBm over ``(P, 3)`` points normalized to roughly ``[-1, 1]``; zeros if no octaves."""
    if freqs.size == 0:
        return np.zeros(len(points), dtype=np.float64)
    total = float(amps.sum()) or 1.0
    return fbm_sum(points, make_perm(seed), freqs, amps) / total


def _varying_persistence_fbm(
    points: np.ndarray, seed: int, freqs: np.ndarray, persistence: np.ndarray
) -> np.ndarray:
    """:func:`_normalized_fbm` with per-point persistence (octave ``k`` weighs ``persistence ** k``)."""
    if freqs.size == 0:
        return np.zeros(len(points), dtype=np.float64)
    perm = make_perm(seed)
    total = np.zeros(len(points), dtype=np.float64)
    weight = np.ones(len(points), dtype=np.float64)
    norm = np.zeros(len(points), dtype=np.float64)
    for freq in freqs:
        total += weight * fbm_sum(points, perm, np.array([freq]), np.array([1.0]))
        norm += weight
        weight = weight * persistence
    return total / norm


@dataclass
class CoastStyle:
    """Per-texel detail style near water, from the parent's coast smoothness."""

    near: np.ndarray         # (P,) blend weight: 1 at the shore, 0 away from it
    smooth: np.ndarray       # (P,) the parent's coast smoothness
    persistence: np.ndarray  # (P,) fBm persistence
    wander: np.ndarray       # (P,) shoreline wander, in detail wavelengths


def coast_style(
    source: ParentSource, loc: Located, params: RegionUpsampleParams
) -> Optional[CoastStyle]:
    """The near-shore detail style, or None without ``coast_smoothness`` (plain fBm).

    ``near`` is a tent on the parent's bilinear water fraction, zero within
    ``coast_band`` of all-land or all-water.
    """
    if not source.has("coast_smoothness"):
        return None
    water = np.zeros(len(loc.anchors), dtype=np.float64)
    for key in ("ocean_mask", "lake_mask"):
        if source.has(key):
            water += source.coarse(key, loc)[:, 0]
    water = np.clip(water, 0.0, 1.0)
    near = np.clip(np.minimum(water, 1.0 - water) / max(float(params.coast_band), 1e-6), 0.0, 1.0)
    smooth = np.clip(source.coarse("coast_smoothness", loc)[:, 0], 0.0, 1.0)
    p_rugged, p_smooth = float(params.coast_persistence_rugged), float(params.coast_persistence_smooth)
    w_rugged, w_smooth = float(params.coast_wander_rugged), float(params.coast_wander_smooth)
    coast_persistence = p_rugged + (p_smooth - p_rugged) * smooth
    base = float(params.persistence)
    return CoastStyle(
        near=near,
        smooth=smooth,
        persistence=base + (coast_persistence - base) * near,
        wander=w_rugged + (w_smooth - w_rugged) * smooth,
    )


def _box_filter(texture: np.ndarray, size: int) -> np.ndarray:
    """``size``×``size`` mean over each hex texture of an ``(M, T, T, C)`` array.

    Edges replicate; only the invisible over-hang corners see that.
    """
    if size <= 1:
        return texture
    r = size // 2
    padded = np.pad(np.asarray(texture, dtype=np.float64), ((0, 0), (r, r), (r, r), (0, 0)), mode="edge")
    _, h, w, _ = texture.shape
    cols = np.zeros((padded.shape[0], h + 2 * r, w, padded.shape[3]), dtype=np.float64)
    for j in range(size):
        cols += padded[:, :, j:j + w, :]
    total = np.zeros(texture.shape, dtype=np.float64)
    for i in range(size):
        total += cols[:, i:i + h, :, :]
    return (total / float(size * size)).astype(texture.dtype)


def _effective_slope(slope: np.ndarray, params: RegionUpsampleParams) -> np.ndarray:
    """Map raw slope (km/km) to the amplitude multiplier, floored and saturated."""
    ref = max(float(params.slope_reference), 1e-9)
    norm = np.clip(slope / ref, 0.0, 1.0)
    floor = float(params.slope_floor)
    return ref * (floor + (1.0 - floor) * norm)


def _shallow_water_damping(depth: np.ndarray, params: RegionUpsampleParams) -> np.ndarray:
    """Fade the detail out as the water above the ground shallows (1 = full detail).

    Lake beds sit ~40 m below their surface against ~15–30 m of sub-grid relief,
    so unscaled noise would turn shallow lakes into confetti.  Scaling by depth
    keeps bumps below the surface so only the shoreline moves.
    """
    scale = float(params.water_relief_depth_km)
    if scale <= 0.0:
        return np.ones_like(depth)
    return np.clip(depth / scale, 0.0, 1.0)


# ── water masks ──────────────────────────────────────────────────────────────


def _neighbour_min(values: np.ndarray) -> np.ndarray:
    """Per-texel minimum over the 4 neighbours of an ``(M, T, T)`` array (edge replicates)."""
    padded = np.pad(values, ((0, 0), (1, 1), (1, 1)), mode="edge")
    return np.minimum(
        np.minimum(padded[:, :-2, 1:-1], padded[:, 2:, 1:-1]),
        np.minimum(padded[:, 1:-1, :-2], padded[:, 1:-1, 2:]),
    )


def flood_to_level(
    elevation: np.ndarray,
    seed: np.ndarray,
    level: np.ndarray,
    iterations: int,
    blocked: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Grow water from ``seed`` texels into neighbours that lie below its surface.

    All arrays are ``(M, T, T)``.  ``seed`` is the parent's water mask, ``level``
    its surface (km); a seed texel stays water only below its surface, so sub-grid
    rises become islands.  A level-constrained flood (as
    ``compute_ocean_mask_texture``): stops at ridges above the surface.
    ``iterations`` caps the travel; ``blocked`` texels belong to another water body.
    Each hex texture floods on its own, which is fine while ``iterations`` stays
    inside the over-hang.
    """
    return _flood(elevation, seed, level, iterations, blocked)[0]


def _flood(
    elevation: np.ndarray,
    seed: np.ndarray,
    level: np.ndarray,
    iterations: int,
    blocked: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """:func:`flood_to_level`, also returning each texel's surface level (``+inf`` when dry)."""
    elev = np.asarray(elevation, dtype=np.float64)
    water = np.asarray(seed, dtype=bool) & (elev <= level)
    water_level = np.where(seed, level, np.inf)
    forbidden = np.zeros_like(water) if blocked is None else np.asarray(blocked, dtype=bool)
    for _ in range(max(int(iterations), 0)):
        nbr = _neighbour_min(np.where(water, water_level, np.inf))
        grow = ~water & ~forbidden & np.isfinite(nbr) & (elev <= nbr)
        if not grow.any():
            break
        water_level = np.where(grow, np.minimum(water_level, nbr), water_level)
        water |= grow
    return water, np.where(water, water_level, np.inf)


# ── conservation ─────────────────────────────────────────────────────────────


def reshape_to_target(
    elevation: np.ndarray, target: np.ndarray, anchors: np.ndarray, weights: np.ndarray,
    n_anchors: int, *, level: float | np.ndarray = 0.0, active: Optional[np.ndarray] = None,
    max_scale: float = 1.0, min_offset: float = 0.001, max_offset: float = 0.02,
) -> np.ndarray:
    """Edit ``elevation`` so its side of ``level`` matches ``target``'s, keeping every anchor's mean.

    ``level`` is the water surface (scalar or per texel); only ``active`` texels
    with a finite level take part.  Wrong-side texels are set to ``target``
    clipped to ``[min_offset, max_offset]`` past the level; the anchor's mean
    change is absorbed by scaling its texels on the side that can take it without
    crossing (dry raised or wet deepened, proportional to height/depth).  Anchors
    needing more than ``max_scale`` ×, or with nothing to absorb, are unchanged.
    """
    e0 = np.asarray(elevation, dtype=np.float64)
    lvl = np.broadcast_to(np.asarray(level, dtype=np.float64), e0.shape)
    act = np.isfinite(lvl) if active is None else (np.asarray(active, dtype=bool) & np.isfinite(lvl))
    lvl = np.where(act, lvl, 0.0)
    e = e0 - lvl
    t = np.asarray(target, dtype=np.float64) - lvl
    wet_t = t <= 0.0
    flip = act & (wet_t != (e <= 0.0))
    mag = np.clip(np.abs(t), min_offset, max(max_offset, min_offset))
    new = np.where(flip, np.where(wet_t, -mag, mag), e)

    delta = np.bincount(anchors, weights=weights * (new - e), minlength=n_anchors)
    dry = new > 0.0
    # delta > 0: the anchor rose, so its wet texels deepen; delta < 0: its dry ones rise.
    absorb = act & np.where(delta[anchors] > 0.0, ~dry, dry)
    capacity = np.bincount(anchors, weights=np.where(absorb, weights * np.abs(new), 0.0), minlength=n_anchors)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(capacity > 0.0, np.abs(delta) / capacity, np.inf)
    ok = (delta == 0.0) | (k <= max_scale)
    k = np.where(ok, np.where(np.isfinite(k), k, 0.0), 0.0)
    out = np.where(absorb, new * (1.0 + k[anchors]), new)
    return np.where(ok[anchors], out + lvl, e0)


def conserve_additive(
    values: np.ndarray, anchors: np.ndarray, targets: np.ndarray,
    weights: np.ndarray, n_anchors: int,
) -> np.ndarray:
    """Shift each anchor group so its weighted mean equals the global value."""
    means, counts = anchor_means(values, anchors, weights, n_anchors)
    correction = targets - means[anchors]
    correction[counts[anchors] <= 0.0] = 0.0
    return values + correction


def conserve_multiplicative(
    values: np.ndarray, anchors: np.ndarray, targets: np.ndarray,
    weights: np.ndarray, n_anchors: int,
) -> np.ndarray:
    """Scale each anchor group so its weighted mean equals the global value.

    For strictly-positive fields.  Groups with mean ~0 or no in-hex sample are left alone.
    """
    means, counts = anchor_means(values, anchors, weights, n_anchors)
    sample_means = means[anchors]
    scale = np.where(
        np.abs(sample_means) > 1e-12,
        targets / np.where(sample_means == 0.0, 1.0, sample_means),
        1.0,
    )
    scale[counts[anchors] <= 0.0] = 1.0
    return values * scale


# Over-relaxation per smoothing pass: a bilinear read returns ~1/4 (checkerboard)
# to all (smooth) of a residual, so 1.4 converges ~1.5x faster without overshoot.
_RELAXATION = 1.4


def conserve_smooth(
    values: np.ndarray, source: ParentSource, loc: Located, targets: np.ndarray,
    weights: np.ndarray, *, multiplicative: bool = False,
    tolerance: float = 0.01, max_iterations: int = 16,
) -> np.ndarray:
    """:func:`conserve_additive` / :func:`conserve_multiplicative` with a smooth correction.

    A per-anchor constant leaves steps at parent-texel edges.  Instead residuals
    are read back smoothly (:meth:`ParentSource.interpolate`) and added, each pass
    halving them, until the well-covered anchors' worst residual is ``tolerance``
    of its start; an exact per-anchor step then fixes the crumbs (and rim
    slivers).  Multiplicative mode works in log space.

    Convergence relies on every anchor owning its own position
    (:meth:`ParentSource.owners`); anchors no read touches can make it diverge.
    """
    # Renumber compactly to the anchors this map touches (the parent has millions).
    idx, w = source.stencil(loc)
    renumber = np.zeros(source.n_anchors, dtype=np.int64)
    renumber[loc.anchors] = 1
    renumber[idx.reshape(-1)] = 1
    used = np.flatnonzero(renumber)
    renumber[used] = np.arange(len(used))
    anchors, idx, n = renumber[loc.anchors], renumber[idx], len(used)
    tgt = np.zeros((n, targets.shape[1]))
    tgt[anchors] = targets
    v = np.asarray(values, dtype=np.float64).copy()
    counts = np.bincount(anchors, weights=np.asarray(weights, dtype=np.float64), minlength=n)
    live = counts > 0.0
    # Clamped reads pile samples onto texture corners, so judge coverage by the median.
    well_covered = counts >= 0.5 * np.median(counts[live]) if live.any() else live
    kept, first, best = v, None, np.inf
    for _ in range(max_iterations):
        means, _ = anchor_means(v, anchors, weights, n, counts=counts)
        if multiplicative:
            ok = live[:, None] & (means > 1e-12) & (tgt > 1e-12)
            resid = np.where(ok, np.log(np.clip(tgt, 1e-12, None) / np.clip(means, 1e-12, None)), 0.0)
            resid = np.clip(resid, -3.0, 3.0)
        else:
            resid = np.where(live[:, None], tgt - means, 0.0)
        worst = float(np.abs(resid[well_covered]).max(initial=0.0))
        if worst >= best:            # a pass stopped helping: keep the best so far
            v = kept
            break
        kept, best = v, worst
        first = worst if first is None else first
        if worst <= tolerance * first:
            break
        step = _RELAXATION * np.einsum("pk,pkc->pc", w, resid[idx])   # ParentSource.interpolate
        v = v * np.exp(step) if multiplicative else v + step
    finish = conserve_multiplicative if multiplicative else conserve_additive
    return finish(v, anchors, targets, weights, n)


# ── the pipeline ─────────────────────────────────────────────────────────────


@dataclass
class _AnchorWater:
    """The parent's water at each texel's anchor; flat ``(P,)`` arrays.

    ``surface`` is the water surface (km): sea level, a lake's fill height, or
    ``+inf`` for no water or an unknown lake level (no hydrology).
    """

    ocean: np.ndarray
    lake: np.ndarray
    surface: np.ndarray
    # Which parent textures exist: a child must not invent a mask.
    has_ocean: bool
    has_lake: bool
    has_lake_levels: bool

    def depth(self, elevation: np.ndarray) -> np.ndarray:
        """Water depth over ``elevation`` (km): ``+inf`` on dry ground, negative where ground pokes out."""
        return self.surface - elevation


def _anchor_water(source: ParentSource, loc: Located) -> _AnchorWater:
    """Read the parent's water masks and their surface elevations at every anchor."""
    n = len(loc.anchors)
    zeros = np.zeros(n, dtype=bool)

    def _mask(key: str) -> np.ndarray:
        if not source.has(key):
            return zeros
        return source.anchor_values(key, loc)[:, 0] > 0.5

    ocean, lake = _mask("ocean_mask"), _mask("lake_mask")
    surface = np.full(n, np.inf)
    surface[ocean] = SEA_LEVEL_KM
    has_levels = source.has("filled_height")
    if has_levels:
        levels = source.anchor_values("filled_height", loc)[:, 0]
        surface[lake] = levels[lake]
    return _AnchorWater(
        ocean=ocean,
        lake=lake,
        surface=surface,
        has_ocean=source.has("ocean_mask"),
        has_lake=source.has("lake_mask"),
        has_lake_levels=has_levels,
    )


def _refine_water_masks(
    water: _AnchorWater,
    elevation: np.ndarray,
    params: RegionUpsampleParams,
) -> dict[str, np.ndarray]:
    """Re-flood ``ocean_mask``/``lake_mask`` against the ``(M, T, T)`` refined relief.

    Water surfaces are the parent's; only the shore shape changes.  Ocean and
    lake flood separately and may not take each other's texels, so a
    below-sea-level coastal lake isn't swallowed by the sea.  Without
    ``filled_height`` (no hydrology) lakes copy the anchor mask; with it,
    ``filled_height`` is written back for the tier below.
    """
    shape = elevation.shape
    masks: dict[str, np.ndarray] = {}
    refine = bool(params.refine_water_masks)
    spread = int(params.water_spread_texels)
    ocean_seed = water.ocean.reshape(shape)
    lake_seed = water.lake.reshape(shape)

    ocean = ocean_seed
    if refine and water.has_ocean:
        ocean = flood_to_level(
            elevation, ocean_seed, np.full(shape, SEA_LEVEL_KM), spread, blocked=lake_seed
        )
    if water.has_ocean:
        masks["ocean_mask"] = ocean[..., None].astype(np.float32)

    if water.has_lake:
        if refine and water.has_lake_levels:
            lake, level = _flood(
                elevation, lake_seed, water.surface.reshape(shape), spread, blocked=ocean_seed
            )
            lake &= ~ocean
        else:
            lake, level = lake_seed, water.surface.reshape(shape)
        masks["lake_mask"] = lake[..., None].astype(np.float32)
        if water.has_lake_levels:
            masks["filled_height"] = np.where(lake, level, elevation)[..., None].astype(np.float32)

    return masks


def _shore_band(source: ParentSource, loc: Located, key: str, params: RegionUpsampleParams) -> np.ndarray:
    """The tent on the parent's bilinear ``key`` fraction (see :func:`coast_style`)."""
    frac = np.clip(source.coarse(key, loc)[:, 0], 0.0, 1.0)
    return np.clip(np.minimum(frac, 1.0 - frac) / max(float(params.coast_band), 1e-6), 0.0, 1.0)


def _reshape_coasts(
    elevation: np.ndarray, water: _AnchorWater, style: CoastStyle, source: ParentSource,
    loc: Located, flat_pos: np.ndarray, seed: int, weights: np.ndarray, slope: np.ndarray,
    shape: tuple[int, int], base_frequency: float, feature_length_km: float, spacing_rad: float,
    params: RegionUpsampleParams,
) -> np.ndarray:
    """Draw the target shores and edit the conserved ``(P, 1)`` elevation to them.

    See ``RegionUpsampleParams.reshape_coast``.  Two passes, each in its shore
    band: the sea against sea level, then lakes against their surfaces, which
    are spread ~the blur's reach onto the shore (lowest neighbouring surface).
    """
    m, tex_size = shape
    n_t = float(source.spacing_rad / max(spacing_rad, 1e-12))
    sea_band = _shore_band(source, loc, "ocean_mask", params) if water.has_ocean else None
    lake_band = (
        _shore_band(source, loc, "lake_mask", params)
        if water.has_lake and water.has_lake_levels else None
    )
    if not any(b is not None and b.any() for b in (sea_band, lake_band)):
        return elevation

    # Blur the staircase away, then add back fine detail in proportion to ruggedness.
    e = elevation[:, 0]
    blur = int(round(float(params.coast_blur_texels) * n_t))
    wide = e
    if blur > 1:
        size = blur | 1
        wide = _box_filter(_box_filter(e.reshape(m, tex_size, tex_size, 1), size), size).reshape(-1)
    goal = wide + (e - wide) * (1.0 - style.smooth)
    w_rugged, w_smooth = float(params.coast_target_wander_rugged), float(params.coast_target_wander_smooth)
    wander = w_rugged + (w_smooth - w_rugged) * style.smooth
    if np.any(wander > 0.0):
        # The scales conservation removes from the elevation itself.
        freqs = np.array([0.5 * base_frequency, base_frequency])
        bays = _varying_persistence_fbm(flat_pos, seed ^ _SEED_SALT_COAST, freqs, style.persistence)
        goal = goal + bays * np.clip(wander * 2.0 * feature_length_km * slope, 0.0, float(params.max_relief_km))

    edit = dict(
        max_scale=float(params.coast_max_scale),
        min_offset=float(params.coast_edit_min_km), max_offset=float(params.coast_edit_max_km),
    )
    out = e
    if sea_band is not None and sea_band.any():
        out = reshape_to_target(
            out, out + sea_band * (goal - out), loc.anchors, weights, source.n_anchors,
            level=SEA_LEVEL_KM, active=~water.lake, **edit,
        )
    if lake_band is not None and lake_band.any():
        surface = np.where(water.lake, water.surface, np.inf).reshape(m, tex_size, tex_size)
        for _ in range(max(int(round(2.0 * n_t)), 1)):
            surface = np.where(np.isfinite(surface), surface, _neighbour_min(surface))
        out = reshape_to_target(
            out, out + lake_band * (goal - out), loc.anchors, weights, source.n_anchors,
            level=surface.reshape(-1), active=~water.ocean, **edit,
        )
    # Float crumbs only: the edit is mean-preserving by construction.
    return conserve_additive(out[:, None], loc.anchors, source.anchor_values("elevation", loc), weights,
                             source.n_anchors)


def refine_fields(
    source: ParentSource,
    positions: np.ndarray,
    weights: np.ndarray,
    spacing_rad: float,
    texel_area_km2: float,
    params: RegionUpsampleParams,
    seed: int,
    *,
    resources: Optional["ResourcesParams"] = None,
    fertility_exponent: float = 2.0,
    mine_points: Optional[np.ndarray] = None,
) -> tuple[dict[str, np.ndarray], Located, int]:
    """Refine ``source``'s fields onto a child tier's ``(M, T, T, 3)`` texels.

    Tier-agnostic: a region refines the globe (:class:`GlobalSource`), a local
    map a region (:class:`region.source.RegionSource`).  ``weights`` is the
    ``(M, T, T)`` aggregation mask, ``spacing_rad`` the child's texel spacing.
    ``mine_points`` are worked deposits (unit vectors) for the miners' density.
    Returns ``(textures, located, detail_octaves)``.
    """
    m, tex_size = positions.shape[0], positions.shape[1]
    flat_pos = positions.reshape(-1, 3)
    weights = weights.reshape(-1).astype(np.float64)
    loc = source.locate(flat_pos)
    anchors, n_anchors = loc.anchors, source.n_anchors

    base_frequency = 0.5 / max(source.spacing_rad, 1e-12)
    max_frequency = (
        float(params.nyquist_safety) / spacing_rad if spacing_rad > 0 else 0.0
    )
    freqs, amps = _octave_bands(base_frequency, max_frequency, params)
    feature_length_km = EARTH_RADIUS_KM / base_frequency

    slope = source.slope(loc)
    eff_slope = _effective_slope(slope, params)
    water = _anchor_water(source, loc)

    textures: dict[str, np.ndarray] = {}

    # ── elevation ────────────────────────────────────────────────────────────
    elev_coarse = source.coarse("elevation", loc)                            # (P, 1)
    style = coast_style(source, loc, params)
    damping = _shallow_water_damping(water.depth(elev_coarse[:, 0]), params)
    amplitude = np.clip(
        float(params.relief_factor) * feature_length_km * eff_slope, 0.0, float(params.max_relief_km),
    ) * damping
    # Near water: coast-style persistence and wander amplitude; the underwater
    # fade is lifted over the sea (islands) but kept for lakes.
    if style is None:
        noise = _normalized_fbm(flat_pos, seed ^ _SEED_SALT_ELEVATION, freqs, amps)
    else:
        noise = _varying_persistence_fbm(flat_pos, seed ^ _SEED_SALT_ELEVATION, freqs, style.persistence)
        coast_amplitude = np.clip(
            style.wander * feature_length_km * slope, 0.0, float(params.max_relief_km),
        ) * np.where(water.lake, damping, 1.0)
        amplitude = amplitude + (coast_amplitude - amplitude) * style.near
    elevation = elev_coarse + (noise * amplitude)[:, None]
    elevation = conserve_smooth(elevation, source, loc, source.anchor_values("elevation", loc), weights)
    if style is not None and params.reshape_coast:
        elevation = _reshape_coasts(
            elevation, water, style, source, loc, flat_pos, seed, weights, slope,
            (m, tex_size), base_frequency, feature_length_km, spacing_rad, params,
        )
    textures["elevation"] = elevation.reshape(m, tex_size, tex_size, 1).astype(np.float32)

    # Sub-grid relief (km), ~zero-mean per anchor: drives every derived field.
    relief_anomaly = elevation[:, 0] - elev_coarse[:, 0]

    # ── water masks: the water surface is the parent's, the shoreline is refined
    textures.update(
        _refine_water_masks(
            water, elevation[:, 0].reshape(m, tex_size, tex_size), params
        )
    )

    # Copied (bilinear, not conserved) so the tier below styles its shores alike.
    if style is not None:
        textures["coast_smoothness"] = source.coarse("coast_smoothness", loc).reshape(
            m, tex_size, tex_size, 1
        ).astype(np.float32)

    # Delta plain and marsh, copied likewise (for region.distributaries).
    if source.has("delta"):
        textures["delta"] = np.clip(source.coarse("delta", loc), 0.0, 1.0).reshape(
            m, tex_size, tex_size, -1
        ).astype(np.float32)

    # ── temperature: lapse rate on the refined relief ────────────────────────
    if source.has("temperature"):
        t_coarse = source.coarse("temperature", loc)                        # (P, K)
        # Sub-sea-level ground counts as sea level, as in the global stage.
        delta_elev = np.maximum(elevation[:, 0], 0.0) - np.maximum(elev_coarse[:, 0], 0.0)
        temperature = t_coarse - float(params.lapse_rate_c_per_km) * delta_elev[:, None]
        temperature = conserve_smooth(
            temperature, source, loc, source.anchor_values("temperature", loc), weights
        )
        textures["temperature"] = temperature.reshape(
            m, tex_size, tex_size, t_coarse.shape[1]
        ).astype(np.float32)

    # ── precipitation: orographic response to the refined relief ─────────────
    if source.has("precipitation"):
        p_coarse = source.coarse("precipitation", loc)                      # (P, K)
        wobble = np.clip(float(params.orographic_gain) * relief_anomaly, -0.9, 0.9)
        precipitation = p_coarse * (1.0 + wobble)[:, None]
        precipitation = conserve_smooth(
            precipitation, source, loc, source.anchor_values("precipitation", loc),
            weights, multiplicative=True,
        )
        textures["precipitation"] = np.maximum(precipitation, 0.0).reshape(
            m, tex_size, tex_size, p_coarse.shape[1]
        ).astype(np.float32)

    # ── biome: reclassify from the area-averaged refined climate ─────────────
    climate_avg = None
    if "temperature" in textures and "precipitation" in textures:
        from worldgen.climate.classification import classify_textures
        k = int(params.biome_smoothing)
        climate_avg = (_box_filter(textures["temperature"], k), _box_filter(textures["precipitation"], k))
        textures["biome"] = classify_textures(*climate_avg, np.zeros(m))

    # ── resources, rural density and danger (region/resources.py) ────────────
    shape = (m, tex_size, tex_size)
    water_now = np.zeros(len(flat_pos), dtype=bool)
    for key in ("ocean_mask", "lake_mask"):
        if key in textures:
            water_now |= textures[key].reshape(-1) > 0.5
    textures.update(refine_resources(
        RefineContext(
            source=source, located=loc, anchors=anchors,
            weights=weights, n_anchors=n_anchors, shape=shape, positions=flat_pos,
            elevation=elevation[:, 0], elev_coarse=elev_coarse[:, 0],
            slope=refined_slope(elevation[:, 0].reshape(shape), np.sqrt(texel_area_km2)).reshape(-1),
            land=~water_now, anchor_land=~(water.ocean | water.lake),
            texel_km=float(np.sqrt(texel_area_km2)),
            mine_points=mine_points, climate_avg=climate_avg,
        ),
        textures, params, resources, fertility_exponent,
        source.danger(loc, flat_pos),
    ))
    return textures, loc, int(freqs.size)


def worked_mine_points(globe: "World") -> Optional[np.ndarray]:
    """``(K, 3)`` unit vectors of the deposits' workings with miners (None without any)."""
    workings = mineral_workings(globe)
    workers = globe.get_feature("mine_workers")
    if workings is None or workers is None:
        return None
    pos, deposit, share = workings
    pts = pos[site_workers(deposit, share, workers) > 0]
    return pts if len(pts) else None


def build_region_fields(
    globe: "World",
    region_id: int,
    params: Optional[RegionUpsampleParams] = None,
    *,
    resources: Optional["ResourcesParams"] = None,
    fertility_exponent: float = 2.0,
) -> RegionFields:
    """Refine the global fields inside one region onto its hex-texture grid.

    Seeded with ``world.seed XOR region_id``.  ``resources`` and
    ``fertility_exponent`` (the population model's γ) are the world's tuning.
    """
    params = params or RegionUpsampleParams()
    source = GlobalSource(globe)

    tex_size = int(params.tex_size)
    geometry = RegionGeometry(region_id, globe.antimeridian)
    grid = RegionHexGrid(geometry, int(params.n_cols))
    positions = grid.texture_positions(tex_size)                 # (M, T, T, 3)
    in_hex = grid.texture_in_hexagon(tex_size)                   # (M, T, T)
    m = len(grid)

    # Equal-area frame: a texel's area is its share of the unit-square cell (the
    # in-hex count would undercount, as hexes spill past the cell edge).
    region_area_km2 = 4.0 * np.pi * EARTH_RADIUS_KM ** 2 / N_REGIONS
    texel_area_km2 = float(region_area_km2 * (2.0 * grid.radius / tex_size) ** 2)

    textures, loc, octaves = refine_fields(
        source, positions, in_hex, grid.texel_spacing_rad(tex_size), texel_area_km2,
        params, int(globe.seed or 0) ^ int(region_id),
        resources=resources, fertility_exponent=fertility_exponent,
        mine_points=worked_mine_points(globe),
    )
    faces, u, v = loc.data
    return RegionFields(
        region_id=int(region_id),
        geometry=geometry,
        grid=grid,
        tex_size=tex_size,
        positions=positions,
        in_hex=in_hex,
        anchors=loc.anchors.reshape(m, tex_size, tex_size),
        faces=faces.reshape(m, tex_size, tex_size),
        face_uv=np.stack([u, v], axis=-1).reshape(m, tex_size, tex_size, 2),
        textures=textures,
        detail_octaves=octaves,
        texel_area_km2=texel_area_km2,
    )


def conservation_report(fields: RegionFields, globe: "World") -> dict[str, dict[str, float]]:
    """Per-global-texel residual of the regional mean vs the global value.

    Returns ``{field: {"max": …, "mean": …, "n_anchors": …}}`` in field units.
    Fertility and forest density are compared over texels dry on both maps.
    """
    return anchor_residuals(fields.textures, fields.anchors, fields.domain, globe.mesh.face_textures)


def anchor_residuals(
    textures: dict[str, np.ndarray],
    anchors: np.ndarray,
    domain: np.ndarray,
    parent_textures: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    """:func:`conservation_report` against any parent tier's ``(N, H, W, C)`` textures."""
    anchors = np.asarray(anchors).reshape(-1)
    in_hex = np.asarray(domain).reshape(-1).astype(np.float64)
    water = np.zeros(len(anchors), dtype=bool)
    anchor_water = np.zeros(len(anchors), dtype=bool)
    for key in ("ocean_mask", "lake_mask"):
        if key in textures:
            water |= textures[key].reshape(-1) > 0.5
        if key in parent_textures:
            anchor_water |= GlobalSampler.anchor_values(parent_textures[key], anchors)[:, 0] > 0.5
    dry = in_hex * (~water & ~anchor_water)
    report: dict[str, dict[str, float]] = {}
    for key, weights in (
        ("elevation", in_hex), ("temperature", in_hex), ("precipitation", in_hex), ("rural_density", in_hex),
        ("fishing_density", in_hex), ("mining_density", in_hex), ("fertility", dry), ("forest_density", dry),
    ):
        if key not in textures or key not in parent_textures:
            continue
        parent_tex = np.asarray(parent_textures[key], dtype=np.float64)
        n_anchors = parent_tex.shape[0] * parent_tex.shape[1] * parent_tex.shape[2]
        values = textures[key].reshape(-1, textures[key].shape[3]).astype(np.float64)
        means, counts = anchor_means(values, anchors, weights, n_anchors)
        hit = counts > 0.0
        residual = np.abs(means[hit] - parent_tex.reshape(n_anchors, -1)[hit])
        report[key] = {
            "max": float(residual.max()) if residual.size else 0.0,
            "mean": float(residual.mean()) if residual.size else 0.0,
            "n_anchors": int(hit.sum()),
        }
    return report
