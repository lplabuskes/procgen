"""Regional refinement of the global fields — upsampling with conservation.

Implements the mechanism in ``docs/plans/regional_local_maps_design.md``: a
region is a *refinement* of the global map, never a replacement.  For every
field the recipe is

1. read the global texture smoothly (bilinear) at each regional texel;
2. add fine detail that exists only above the global grid's Nyquist limit;
3. force each global texel's average back to its exact global value.

Step 3 is what makes the result legal: downsampling a region by averaging
recovers the global map bit-for-bit (see :func:`conservation_report`).  It is a
per-anchor constant nudge, so it costs the detail nothing visually.

**Elevation is the only field with a noise stream of its own.**  Every other
field is a physical consequence of the refined relief, or is copied verbatim.
That is deliberate: ``biome`` is a threshold classifier over temperature and
precipitation, so an *independent* noise stream in either input makes neighbouring
texels land on opposite sides of a cut at random — the boundary dissolves into
speckle.  Driving both from the same relief instead makes biome boundaries follow
ridges and valleys, which is both prettier and closer to the truth.

Field-by-field:

- **Elevation** — slope-proportional fBm.  Amplitude follows the local gradient
  of the coarse field (flat plains stay flat, mountains get rugged), with a floor
  so nowhere is glassy, and fades out under shallow water so a lake bed cannot
  poke back through its own surface.  Conserved additively.
- **Temperature** — no noise: the refined elevation drives a lapse-rate
  correction, which is exactly the physical signal a finer grid should reveal.
  Conserved additively.
- **Precipitation** — no noise: an orographic response to the refined relief
  (high ground catches more rain), applied multiplicatively and conserved
  multiplicatively so values stay positive.
- **Biome** — reclassified per regional texel from the refined climate, using the
  same classifier the global stage uses.
- **Ocean / lake masks** — re-flooded against the refined relief.  A water body's
  *surface* is a global fact (the ocean sits at sea level, a lake at the fill
  level Priority-Flood gave it), so the level is taken verbatim from the globe and
  only the shoreline moves: a regional texel is water when the refined elevation
  lies below the surface of a water body it is connected to.  This is the same
  level-constrained flood fill ``compute_ocean_mask_texture`` runs on the global
  grid, seeded from the global mask instead of from deep-ocean faces.

The detail band is honest about resolution: octaves start at the global grid's
Nyquist and stop at the regional grid's, so a region no finer than the globe
gets no detail at all (``RegionFields.detail_octaves == 0``) rather than fake
structure.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

from core.params import param_field as _f, paramclass
from region.geometry import N_REGIONS, RegionGeometry
from region.grid import RegionHexGrid
from region.sampling import GlobalSampler, anchor_means
from utils.simplex import fbm_sum, make_perm

if TYPE_CHECKING:
    from core.mesh import PolygonMesh
    from core.world import World

EARTH_RADIUS_KM = 6371.0

# The elevation the global stages call the shore (worldgen.geology.elevation and
# worldgen.hydrology.builder both hard-code it).
SEA_LEVEL_KM = 0.0

# Elevation is the only field with a noise stream; see the module docstring.
_SEED_SALT_ELEVATION = 0x9E37


@paramclass
class RegionUpsampleParams:
    """Tuning for regional detail generation."""

    n_cols: int = _f(14, label="Hex columns per region", min=4, max=64)
    # 64 puts regional texels ~4 km across against the globe's ~15 km, which is
    # what buys the detail band its two octaves; see build_region_fields.
    tex_size: int = _f(64, label="Hex texture size", min=4, max=128)
    # Detail amplitude is relief_factor × (detail wavelength) × (local slope), so
    # the factor reads as "sub-grid relief per unit of grid-scale slope".  The
    # global field is smooth at ~15 km, where even mountain slopes only reach
    # ~0.007 km/km — hence the small saturation point and the factor above 1.
    relief_factor: float = _f(2.0, label="Detail relief factor", min=0.0, max=8.0)
    slope_reference: float = _f(
        0.005, label="Slope saturation (km/km)", min=0.0005, max=0.05, step=0.0005
    )
    slope_floor: float = _f(0.10, label="Flat-ground detail floor", min=0.0, max=1.0)
    max_relief_km: float = _f(1.0, label="Max detail relief (km)", min=0.0, max=5.0)
    # Precipitation has no noise of its own — it responds to the refined relief.
    # 0.5/km means a 400 m sub-grid ridge catches ~20% more rain than the global
    # texel it sits in, and the matching hollow ~20% less.  0 disables it.
    orographic_gain: float = _f(
        0.5, label="Orographic precip gain (/km)", min=0.0, max=3.0, step=0.05
    )
    lapse_rate_c_per_km: float = _f(6.5, label="Lapse rate (°C/km)", min=0.0, max=15.0)
    # Sub-grid relief fades out under water as the depth runs out, so a shallow
    # lake bed cannot poke back through its own surface.  0.1 km clears the real
    # lake-depth distribution (median ~40 m); 0 disables the fade.
    water_relief_depth_km: float = _f(
        0.1, label="Underwater detail fade (km)", min=0.0, max=2.0, step=0.05
    )
    # Water surfaces are global facts; only the shoreline is refined.  Off, the
    # masks are copied from the anchor texel and the coast stays blocky.
    refine_water_masks: bool = _f(True, label="Refine shoreline", min=0, max=1)
    # How far water may advance into a texel the globe called land, in regional
    # texels.  The refined shoreline should stay inside the global texels the
    # coast already runs through, so this wants to be about one global texel —
    # ~4 regional texels at the default resolutions.
    water_spread_texels: int = _f(4, label="Shoreline spread (texels)", min=0, max=16)
    # Biome is a hard threshold classifier, and the refined temperature swings
    # ~1 °C between neighbouring texels — enough to fray every boundary into a
    # dithered band.  A biome is a property of a neighbourhood rather than of one
    # texel, so the classifier reads an area average; the temperature and
    # precipitation layers themselves stay sharp.  0 or 1 disables it.
    biome_smoothing: int = _f(5, label="Biome climate averaging (texels)", min=0, max=15)
    octaves: int = _f(4, label="Detail octaves (max)", min=0, max=8)
    lacunarity: float = _f(2.0, label="Detail lacunarity", min=1.1, max=4.0)
    persistence: float = _f(0.5, label="Detail persistence", min=0.1, max=0.95)
    nyquist_safety: float = _f(0.5, label="Nyquist safety factor", min=0.1, max=1.0)


@dataclass
class RegionFields:
    """Refined per-hex textures for one region, mirroring ``mesh.face_textures``.

    ``textures`` maps the same keys the globe uses (``"elevation"``,
    ``"temperature"``, ``"precipitation"``, ``"biome"``, ``"ocean_mask"``,
    ``"lake_mask"``) to ``(M, T, T, C)`` arrays over the region's ``M`` hexagons.
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

    @property
    def n_hexes(self) -> int:
        return len(self.grid)


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


def global_slope(mesh: "PolygonMesh", elevation_tex: np.ndarray) -> np.ndarray:
    """(N, H, W, 1) gradient magnitude (km/km) of the global elevation texture.

    Differences are taken across each face's own texel grid and scaled by that
    face's tangent-frame extent, so the result is a true surface slope rather
    than a per-index difference.
    """
    elev = np.asarray(elevation_tex, dtype=np.float64)[..., 0]        # (N, H, W)
    _, h, w = elev.shape
    frames = mesh.face_tangent_frames()
    u_ranges = np.asarray(frames[4], dtype=np.float64) * EARTH_RADIUS_KM
    v_ranges = np.asarray(frames[5], dtype=np.float64) * EARTH_RADIUS_KM
    d_v, d_u = np.gradient(elev, axis=(1, 2))
    du_km = np.maximum(u_ranges / max(w - 1, 1), 1e-9)[:, None, None]
    dv_km = np.maximum(v_ranges / max(h - 1, 1), 1e-9)[:, None, None]
    slope = np.hypot(d_u / du_km, d_v / dv_km)
    return slope[..., None].astype(np.float64)


def _box_filter(texture: np.ndarray, size: int) -> np.ndarray:
    """``size``×``size`` mean over each hex texture of an ``(M, T, T, C)`` array.

    Edges replicate.  That is fine even at the hexagon's rim: the texture
    over-hangs its hexagon, so the visible border averages over real neighbouring
    data, and only the invisible corners see the replicated edge.
    """
    if size <= 1:
        return texture
    r = size // 2
    padded = np.pad(texture, ((0, 0), (r, r), (r, r), (0, 0)), mode="edge")
    _, h, w, _ = texture.shape
    total = np.zeros(texture.shape, dtype=np.float64)
    for i in range(size):
        for j in range(size):
            total += padded[:, i:i + h, j:j + w, :]
    return (total / float(size * size)).astype(texture.dtype)


def _effective_slope(slope: np.ndarray, params: RegionUpsampleParams) -> np.ndarray:
    """Map raw slope (km/km) to the amplitude multiplier, floored and saturated."""
    ref = max(float(params.slope_reference), 1e-9)
    norm = np.clip(slope / ref, 0.0, 1.0)
    floor = float(params.slope_floor)
    return ref * (floor + (1.0 - floor) * norm)


def _shallow_water_damping(depth: np.ndarray, params: RegionUpsampleParams) -> np.ndarray:
    """Fade the detail out as the water above the ground shallows (1 = full detail).

    Without this, refining the masks turns every shallow lake into confetti.  A
    lake bed sits within a few tens of metres of its own surface (median depth on
    a real world is ~40 m, against ~15–30 m of sub-grid relief), so zero-mean
    noise pokes half of it back into the air and the mask, being a threshold on
    the refined elevation, dithers across the whole water body rather than only
    along its shore.

    Scaling the amplitude by the depth keeps the bump strictly smaller than the
    water it sits under, so shallow water stays water and only the *edge* — where
    the depth genuinely runs out — moves.  It costs nothing visible: submerged
    relief is painted over by the water anyway.  The deep ocean (median 3.4 km) is
    far past the fade and keeps its full detail.
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

    All arrays are ``(M, T, T)`` over the region's hex textures.  ``seed`` marks
    the texels the *global* mask calls water and ``level`` carries that water
    body's surface elevation (km); a seed texel is water only if the refined
    ``elevation`` is still below its own surface, which is what lets a sub-grid
    rise inside a global water texel emerge as an island.

    Growth is a level-constrained flood — the same rule
    ``worldgen.geology.elevation.compute_ocean_mask_texture`` uses one level up —
    so water spreads through low ground but stops at any ridge above the surface,
    rather than teleporting over it the way a plain dilation would.  ``iterations``
    caps how far it can travel; ``blocked`` texels are never taken (they belong to
    another water body).

    Each hex texture is flooded on its own.  The textures over-hang their
    hexagons, so the visible area is grown from real neighbouring data as long as
    ``iterations`` stays well inside the over-hang.
    """
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
    return water


# ── conservation ─────────────────────────────────────────────────────────────


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

    Used for strictly-positive fields, where an additive shift could go negative.
    Groups whose mean is ~0 (or which received no in-hex sample) are left alone.
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


# ── the pipeline ─────────────────────────────────────────────────────────────


@dataclass
class _AnchorWater:
    """What the globe says about the water at each regional texel's anchor.

    ``surface`` is the water-surface elevation (km) — sea level under the ocean,
    the Priority-Flood fill height under a lake — and ``+inf`` where there is no
    water, or where a lake's level is unknown because the world has no hydrology.
    All arrays are flat ``(P,)`` over the region's texels.
    """

    ocean: np.ndarray
    lake: np.ndarray
    surface: np.ndarray
    # Which of the globe's textures actually existed — a region must not invent a
    # mask the world map does not have.
    has_ocean: bool
    has_lake: bool
    has_lake_levels: bool

    def depth(self, elevation: np.ndarray) -> np.ndarray:
        """How deep the water stands over ``elevation`` (km).

        ``+inf`` on dry ground (no surface above it) and negative where ground
        already pokes out of a water texel, so callers can treat "deep" and "not
        underwater at all" the same way.
        """
        return self.surface - elevation


def _anchor_water(
    mesh: "PolygonMesh", sampler: GlobalSampler, anchors: np.ndarray
) -> _AnchorWater:
    """Read the global water masks and their surface elevations at every anchor."""
    n = len(anchors)
    zeros = np.zeros(n, dtype=bool)

    def _mask(key: str) -> np.ndarray:
        if key not in mesh.face_textures:
            return zeros
        return sampler.anchor_values(mesh.face_textures[key], anchors)[:, 0] > 0.5

    ocean, lake = _mask("ocean_mask"), _mask("lake_mask")
    surface = np.full(n, np.inf)
    surface[ocean] = SEA_LEVEL_KM
    filled = mesh.face_textures.get("filled_height")
    if filled is not None:
        levels = sampler.anchor_values(filled, anchors)[:, 0]
        surface[lake] = levels[lake]
    return _AnchorWater(
        ocean=ocean,
        lake=lake,
        surface=surface,
        has_ocean="ocean_mask" in mesh.face_textures,
        has_lake="lake_mask" in mesh.face_textures,
        has_lake_levels=filled is not None,
    )


def _refine_water_masks(
    water: _AnchorWater,
    elevation: np.ndarray,
    params: RegionUpsampleParams,
) -> dict[str, np.ndarray]:
    """Re-flood ``ocean_mask``/``lake_mask`` against the ``(M, T, T)`` refined relief.

    Each water body keeps the surface elevation the globe gave it — sea level for
    the ocean, the Priority-Flood fill height for a lake — so the two tiers agree
    on *where the water is*, and disagree only on the sub-texel shape of the
    shore.  Ocean and lake are flooded separately and neither may take the
    other's texels, which keeps a coastal lake sitting below sea level from being
    swallowed by the sea it is walled off from on the world map.

    A lake's surface lives in the ``filled_height`` texture the hydrology stage
    writes; without it (a world built with elevation only) lakes fall back to the
    anchor mask verbatim.
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
            lake = flood_to_level(
                elevation, lake_seed, water.surface.reshape(shape), spread, blocked=ocean_seed
            )
            lake &= ~ocean
        else:
            lake = lake_seed
        masks["lake_mask"] = lake[..., None].astype(np.float32)

    return masks


def build_region_fields(
    globe: "World",
    region_id: int,
    params: Optional[RegionUpsampleParams] = None,
) -> RegionFields:
    """Refine the global fields inside one region onto its hex-texture grid.

    Deterministic: the noise streams are seeded with ``world.seed XOR region_id``,
    so the same region always regenerates identically.
    """
    params = params or RegionUpsampleParams()
    mesh = globe.mesh
    if "elevation" not in mesh.face_textures:
        raise ValueError("global mesh has no 'elevation' texture — run the elevation stage first")

    tex_size = int(params.tex_size)
    geometry = RegionGeometry(region_id, globe.antimeridian)
    grid = RegionHexGrid(geometry, int(params.n_cols))
    positions = grid.texture_positions(tex_size)                 # (M, T, T, 3)
    in_hex = grid.texture_in_hexagon(tex_size)                   # (M, T, T)
    m = len(grid)
    flat_pos = positions.reshape(-1, 3)
    weights = in_hex.reshape(-1).astype(np.float64)

    sampler = GlobalSampler(mesh)
    faces, u, v = sampler.locate(flat_pos)

    elev_tex = np.asarray(mesh.face_textures["elevation"], dtype=np.float64)
    n_faces, gh, gw, _ = elev_tex.shape
    n_anchors = n_faces * gh * gw
    anchors = sampler.anchors(faces, u, v, gh, gw)

    # Detail band: from the global grid's Nyquist up to the regional grid's.
    global_spacing = float(np.sqrt(np.mean(mesh.face_areas))) / max(gw - 1, 1)
    region_spacing = grid.texel_spacing_rad(tex_size)
    base_frequency = 0.5 / max(global_spacing, 1e-12)
    max_frequency = (
        float(params.nyquist_safety) / region_spacing if region_spacing > 0 else 0.0
    )
    freqs, amps = _octave_bands(base_frequency, max_frequency, params)
    feature_length_km = EARTH_RADIUS_KM / base_frequency
    seed = int(globe.seed or 0) ^ int(region_id)

    slope = sampler.sample(global_slope(mesh, elev_tex), faces, u, v)[:, 0]
    eff_slope = _effective_slope(slope, params)
    water = _anchor_water(mesh, sampler, anchors)

    textures: dict[str, np.ndarray] = {}

    # ── elevation ────────────────────────────────────────────────────────────
    elev_coarse = sampler.sample(elev_tex, faces, u, v)                     # (P, 1)
    noise = _normalized_fbm(flat_pos, seed ^ _SEED_SALT_ELEVATION, freqs, amps)
    amplitude = np.clip(
        float(params.relief_factor) * feature_length_km * eff_slope,
        0.0, float(params.max_relief_km),
    ) * _shallow_water_damping(water.depth(elev_coarse[:, 0]), params)
    elevation = elev_coarse + (noise * amplitude)[:, None]
    elevation = conserve_additive(
        elevation, anchors, sampler.anchor_values(elev_tex, anchors), weights, n_anchors
    )
    textures["elevation"] = elevation.reshape(m, tex_size, tex_size, 1).astype(np.float32)

    # The sub-grid relief, in km: the one signal every derived field reads from.
    # ~Zero-mean inside each anchor group, so it perturbs without biasing.
    relief_anomaly = elevation[:, 0] - elev_coarse[:, 0]

    # ── water masks: the water surface is global, the shoreline is refined ───
    textures.update(
        _refine_water_masks(
            water, elevation[:, 0].reshape(m, tex_size, tex_size), params
        )
    )

    # ── temperature: lapse rate on the refined relief ────────────────────────
    if "temperature" in mesh.face_textures:
        temp_tex = np.asarray(mesh.face_textures["temperature"], dtype=np.float64)
        t_coarse = sampler.sample(temp_tex, faces, u, v)                    # (P, K)
        # Sub-sea-level ground is pinned to sea level for lapse, as in the
        # global temperature stage.
        delta_elev = np.maximum(elevation[:, 0], 0.0) - np.maximum(elev_coarse[:, 0], 0.0)
        temperature = t_coarse - float(params.lapse_rate_c_per_km) * delta_elev[:, None]
        temperature = conserve_additive(
            temperature, anchors, sampler.anchor_values(temp_tex, anchors), weights, n_anchors
        )
        textures["temperature"] = temperature.reshape(
            m, tex_size, tex_size, temp_tex.shape[3]
        ).astype(np.float32)

    # ── precipitation: orographic response to the refined relief ─────────────
    # Deliberately *not* an independent noise stream: precipitation feeds a
    # threshold classifier (biome), and uncorrelated noise there turns every
    # near-threshold area into speckle.  Sharing the relief instead puts the
    # moisture bands on the contours.
    if "precipitation" in mesh.face_textures:
        precip_tex = np.asarray(mesh.face_textures["precipitation"], dtype=np.float64)
        p_coarse = sampler.sample(precip_tex, faces, u, v)                  # (P, K)
        wobble = np.clip(float(params.orographic_gain) * relief_anomaly, -0.9, 0.9)
        precipitation = p_coarse * (1.0 + wobble)[:, None]
        precipitation = conserve_multiplicative(
            precipitation, anchors, sampler.anchor_values(precip_tex, anchors),
            weights, n_anchors,
        )
        textures["precipitation"] = np.maximum(precipitation, 0.0).reshape(
            m, tex_size, tex_size, precip_tex.shape[3]
        ).astype(np.float32)

    # ── biome: reclassify from the area-averaged refined climate ─────────────
    if "temperature" in textures and "precipitation" in textures:
        from worldgen.climate.classification import classify_textures
        k = int(params.biome_smoothing)
        textures["biome"] = classify_textures(
            _box_filter(textures["temperature"], k),
            _box_filter(textures["precipitation"], k),
            np.zeros(m),
        )

    region_area_km2 = 4.0 * np.pi * EARTH_RADIUS_KM ** 2 / N_REGIONS
    covering = int(in_hex.sum())
    return RegionFields(
        region_id=int(region_id),
        geometry=geometry,
        grid=grid,
        tex_size=tex_size,
        positions=positions,
        in_hex=in_hex,
        anchors=anchors.reshape(m, tex_size, tex_size),
        faces=faces.reshape(m, tex_size, tex_size),
        face_uv=np.stack([u, v], axis=-1).reshape(m, tex_size, tex_size, 2),
        textures=textures,
        detail_octaves=int(freqs.size),
        texel_area_km2=float(region_area_km2 / max(covering, 1)),
    )


def conservation_report(fields: RegionFields, globe: "World") -> dict[str, dict[str, float]]:
    """Residuals of the critical invariant: does downsampling recover the globe?

    For every global texel that received in-hexagon regional samples, compare the
    regional mean against the global value.  Returns ``{field: {"max": …, "mean": …,
    "n_anchors": …}}`` in each field's own units.
    """
    mesh = globe.mesh
    anchors = fields.anchors.reshape(-1)
    weights = fields.in_hex.reshape(-1).astype(np.float64)
    report: dict[str, dict[str, float]] = {}
    for key in ("elevation", "temperature", "precipitation"):
        if key not in fields.textures or key not in mesh.face_textures:
            continue
        global_tex = np.asarray(mesh.face_textures[key], dtype=np.float64)
        n_anchors = global_tex.shape[0] * global_tex.shape[1] * global_tex.shape[2]
        values = fields.textures[key].reshape(-1, fields.textures[key].shape[3]).astype(np.float64)
        means, counts = anchor_means(values, anchors, weights, n_anchors)
        hit = counts > 0.0
        residual = np.abs(means[hit] - global_tex.reshape(n_anchors, -1)[hit])
        report[key] = {
            "max": float(residual.max()) if residual.size else 0.0,
            "mean": float(residual.mean()) if residual.size else 0.0,
            "n_anchors": int(hit.sum()),
        }
    return report
