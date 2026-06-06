"""Climate stage orchestrator — ClimateParams and ClimateBuilder."""
from __future__ import annotations

from typing import Callable, Mapping, Optional

import numpy as np

import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra as sp_dijkstra

from core.display import DisplayRegistry
from core.pipeline import BuildContext, Stage
from core.world import World

from worldgen.climate.classification import (
    N_BIOMES,
    classify_textures,
)
from worldgen.climate.channels import (
    BIOME,
    DISTANCE_TO_COAST,
    PRECIPITATION,
    PRECIPITATION_ANNUAL,
    TEMPERATURE,
    TEMPERATURE_ANNUAL,
    TEMPERATURE_COLDEST,
    TEMPERATURE_WARMEST,
)
from worldgen.climate.currents import CurrentsResult, compute_currents
from worldgen.geology.channels import ELEVATION, OCEAN_MASK
from worldgen.hydrology.channels import LAKE_MASK
from worldgen.climate.insolation import (
    SEASON_DECLINATIONS_DEG,
    face_latitudes,
    seasonal_insolation,
)
from worldgen.climate.humidity import (
    advect_moisture,
    evaporation_source,
    per_pixel_humidity,
    smooth_humidity,
)
from worldgen.climate.precipitation import per_pixel_precipitation
from worldgen.climate.temperature import (
    per_pixel_temperatures,
    surface_temperatures,
)
from worldgen.climate.winds import pressure_winds, zonal_winds
from worldgen.climate.display import (
    biome_color,
    biome_legend_entries,
    build_biome_display_texture,
    build_precipitation_display_texture,
    build_temperature_display_texture,
    humidity_color,
    precipitation_color,
    precipitation_legend_entries,
    temperature_color,
    temperature_legend_entries,
)
from core.mesh import PolygonMesh
from core.params import field, param_field as _f, paramclass


SEASON_LABELS = ["Spring", "Summer", "Autumn", "Winter"]


EARTH_RADIUS_KM = 6371.0
N_SEASONS = 4


@paramclass
class TemperatureParams:
    equator_annual_c: float = _f(27.0, label="Equator annual mean (°C)", min=-10.0, max=40.0)
    pole_annual_c: float = _f(-25.0, label="Pole annual mean (°C)", min=-60.0, max=10.0)
    anomaly_scale: float = _f(0.05, label="Seasonal anomaly scale", min=0.0, max=0.5)
    lapse_rate_c_per_km: float = _f(6.5, label="Lapse rate (°C/km)", min=0.0, max=15.0)
    ocean_inertia: float = _f(0.30, label="Ocean inertia", min=0.0, max=1.0)
    land_inertia: float = _f(1.0, label="Land inertia", min=0.0, max=2.0)
    inertia_distance_km: float = _f(1500.0, label="Inertia saturation (km)", min=100.0, max=5000.0)


@paramclass
class WindParams:
    trade_speed: float = _f(7.0, label="Trade winds (m/s)", min=0.0, max=30.0)
    westerly_speed: float = _f(10.0, label="Westerlies (m/s)", min=0.0, max=30.0)
    polar_speed: float = _f(5.0, label="Polar easterlies (m/s)", min=0.0, max=30.0)
    itcz_shift_deg: float = _f(10.0, label="ITCZ shift (°)", min=0.0, max=30.0)
    pressure_wind_scale: float = _f(
        600.0, label="Pressure wind scale (m·s⁻¹·km/°C)", min=0.0, max=1000.0
    )


@paramclass
class PrecipitationParams:
    evaporation_scale: float = _f(1.0, label="Evaporation scale", min=0.1, max=5.0)
    precipitation_scale: float = _f(2000.0, label="Precipitation scale (mm/yr)", min=100.0, max=10000.0)
    moisture_advection_iter: int = _f(200, label="Moisture transport steps", min=50, max=600)
    moisture_decay_rate: float = _f(0.03, label="Moisture decay rate", min=0.005, max=0.2)
    orographic_k: float = _f(0.4, label="Orographic K", min=0.0, max=2.0)
    humidity_smoothing_passes: int = _f(10, label="Humidity smoothing passes", min=0, max=30, hidden=True)
    humidity_smoothing_alpha: float = _f(0.4, label="Humidity smoothing α", min=0.0, max=1.0, hidden=True)
    lake_evap_scale: float = _f(1.0, label="Lake evap scale", min=0.0, max=5.0, hidden=True)
    lake_feedback_iter: int = _f(100, label="Lake feedback steps", min=0, max=600, hidden=True)


@paramclass
class CurrentsParams:
    enabled: bool = _f(True, label="Enable currents", min=0, max=1)
    beta_scale: float = _f(1.0, label="β scale (western intensification)", min=0.0, max=5.0)
    advection_alpha: float = _f(1.0, label="Advection mixing α", min=0.0, max=1.0)
    advection_iterations: int = _f(400, label="Advection iterations", min=10, max=2000)
    temperature_scale: float = _f(0.06, label="Current ΔT scale (°C / (W/m²))", min=0.0, max=0.5)
    precipitation_scale: float = _f(0.03, label="Current wet scale (per °C)", min=0.0, max=0.2)
    inland_decay_km: float = _f(600.0, label="Inland decay (km)", min=50.0, max=5000.0)


@paramclass
class ClimateParams:
    temperature: TemperatureParams = field(
        default_factory=TemperatureParams, metadata={"label": "Temperature"}
    )
    wind: WindParams = field(default_factory=WindParams, metadata={"label": "Winds"})
    precipitation: PrecipitationParams = field(
        default_factory=PrecipitationParams, metadata={"label": "Precipitation"}
    )
    currents: CurrentsParams = field(
        default_factory=CurrentsParams, metadata={"label": "Currents"}
    )
    seed: Optional[int] = field(
        default=None,
        metadata={"label": "Seed", "kind": "optional_int", "min": 0, "max": 2**31 - 1},
    )


def _distance_to_coast_km(mesh: PolygonMesh, face_elev_km: np.ndarray,
                          sea_level: float = 0.0,
                          is_ocean: np.ndarray | None = None) -> np.ndarray:
    """Dijkstra on the face graph from ocean seeds; edge weight = great-circle km."""
    n = len(mesh.faces)
    centroids = mesh.face_centroids.astype(np.float64)
    if is_ocean is None:
        is_ocean = face_elev_km <= sea_level

    # Build face-distance graph + virtual multi-source node at index n (weight 0
    # to every ocean face) for single-call multi-source Dijkstra.
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for u in range(n):
        for v in mesh.face_adjacency[u]:
            cos_ang = float(np.clip(centroids[u] @ centroids[v], -1.0, 1.0))
            rows.append(u)
            cols.append(int(v))
            data.append(EARTH_RADIUS_KM * np.arccos(cos_ang))
    ocean_idx = np.flatnonzero(is_ocean)
    virt = n
    for i in ocean_idx:
        rows.append(virt)
        cols.append(int(i))
        data.append(0.0)

    graph = sp.csr_matrix((data, (rows, cols)), shape=(n + 1, n + 1))
    dist = sp_dijkstra(graph, directed=True, indices=virt)[:n]
    dist[~np.isfinite(dist)] = 0.0  # land with no ocean reachable: treat as coast
    return dist.astype(np.float32)


def _dominant_biome_per_face(biome_tex: np.ndarray) -> np.ndarray:
    """Mode of the (N, H, W, 1) uint8 biome texture over each face."""
    n = biome_tex.shape[0]
    flat = biome_tex.reshape(n, -1)
    result = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        counts = np.bincount(flat[i], minlength=N_BIOMES)
        result[i] = counts.argmax()
    return result


class ClimateBuilder(Stage):
    """Runs the first climate pass on a built elevation mesh (a pipeline ``Stage``).

    ``build(ctx)`` reads the mesh and its ``"elevation"`` texture from the world.
    Attaches `temperature` (N,H,W,4 float32 °C), `precipitation` (N,H,W,4
    float32 mm/season), and `biome` (N,H,W,1 uint8) textures, and writes
    per-face aggregates into `mesh.face_metadata`. Intermediate per-face
    results are kept as attributes so downstream code (display, future
    currents stage) can read them without re-running anything.
    """

    name = "climate"

    def __init__(self,
                 params: ClimateParams | None = None,
                 classifier: Callable[..., np.ndarray] = classify_textures) -> None:
        self.params: ClimateParams = params or ClimateParams()
        self.classifier = classifier

        self.insolation: Optional[np.ndarray] = None             # (N, K) W/m²
        self.distance_to_coast: Optional[np.ndarray] = None      # (N,) km
        self.temperature_face: Optional[np.ndarray] = None       # (N, K) °C
        self.precipitation_face: Optional[np.ndarray] = None     # (N, K) mm/season
        self.humidity_face: Optional[np.ndarray] = None          # (N, K) dimensionless q_air
        self.winds: Optional[np.ndarray] = None                  # (N, K, 3) m/s
        self.currents: Optional[CurrentsResult] = None           # populated when params.currents.enabled

    def build(self, ctx: BuildContext) -> None:
        mesh = ctx.mesh
        elevation_textures = ELEVATION.read(mesh)
        _report = ctx.report

        p = self.params
        tp, wp, pp, cp = p.temperature, p.wind, p.precipitation, p.currents

        elev_tex = elevation_textures[:, :, :, 0]                              # (N, H, W) km
        face_elev = elev_tex.mean(axis=(1, 2)).astype(np.float64)              # (N,) km
        latitudes_rad = face_latitudes(mesh.face_centroids).astype(np.float64)  # (N,)

        _report("Computing insolation…")
        decls = np.deg2rad(SEASON_DECLINATIONS_DEG)
        insolation = seasonal_insolation(latitudes_rad, decls)                 # (N, K)

        _report("Computing distance to coast…")
        is_ocean_face: np.ndarray | None = None
        if OCEAN_MASK.present(mesh):
            omask = OCEAN_MASK.read(mesh)
            is_ocean_face = omask[:, :, :, 0].mean(axis=(1, 2)) > 0.5
        dist_coast = _distance_to_coast_km(mesh, face_elev, is_ocean=is_ocean_face)  # (N,)

        _report("Computing surface temperatures…")
        t_face = surface_temperatures(
            latitudes_rad,
            insolation,
            face_elev,
            dist_coast.astype(np.float64),
            params=tp,
        )

        _report("Computing prevailing winds…")
        winds = zonal_winds(
            mesh.face_centroids.astype(np.float64),
            latitudes_rad,
            params=wp,
        )
        # Land/sea thermal contrast → seasonal pressure systems → cross-isobar
        # flow. Added on top of the zonal base; ocean inertia damps the anomaly
        # so the contribution is largest on continental coasts.
        winds = winds + pressure_winds(
            mesh.face_centroids.astype(np.float64),
            mesh.face_adjacency,
            t_face,
            pressure_wind_scale=wp.pressure_wind_scale,
            earth_radius_km=EARTH_RADIUS_KM,
        )

        _report("Computing atmospheric moisture transport…")
        evap = evaporation_source(t_face, face_elev, pp.evaporation_scale, is_ocean=is_ocean_face)
        q_air = advect_moisture(
            evap,
            mesh.face_centroids.astype(np.float64),
            mesh.face_adjacency,
            winds.astype(np.float64),
            n_iter=pp.moisture_advection_iter,
            decay_rate=pp.moisture_decay_rate,
        )
        q_air = smooth_humidity(q_air, mesh.face_adjacency,
                                n_passes=pp.humidity_smoothing_passes,
                                alpha=pp.humidity_smoothing_alpha)

        currents_result: Optional[CurrentsResult] = None
        if cp.enabled:
            _report("Computing ocean currents…")
            currents_result = compute_currents(
                mesh,
                face_elev,
                mesh.face_centroids.astype(np.float64),
                winds,
                insolation.mean(axis=1),
                params=cp,
                is_ocean=is_ocean_face,
            )

            # Second-pass T and P with the current-driven deltas broadcast across seasons.
            temp_delta = currents_result.temperature_delta[:, None].astype(np.float64)
            t_face = surface_temperatures(
                latitudes_rad,
                insolation,
                face_elev,
                dist_coast.astype(np.float64),
                params=tp,
                coastal_current_temp_delta=temp_delta,
            )
            # Second-pass humidity: updated SST → more evaporation from warm currents.
            evap = evaporation_source(t_face, face_elev, pp.evaporation_scale, is_ocean=is_ocean_face)
            q_air = advect_moisture(
                evap,
                mesh.face_centroids.astype(np.float64),
                mesh.face_adjacency,
                winds.astype(np.float64),
                n_iter=pp.moisture_advection_iter,
                decay_rate=pp.moisture_decay_rate,
            )
            q_air = smooth_humidity(q_air, mesh.face_adjacency,
                                    n_passes=pp.humidity_smoothing_passes,
                                    alpha=pp.humidity_smoothing_alpha)

        n_h = elev_tex.shape[1]

        _report("Rasterising temperature texture…")
        elev_pixel = elev_tex.astype(np.float64)                                # (N, H, W) km
        t_pixel = per_pixel_temperatures(
            t_face, face_elev, elev_pixel, tp.lapse_rate_c_per_km, mesh,
        )                                                                       # (N, H, W, K)

        if currents_result is not None:
            q_air = q_air * (1.0 + currents_result.precipitation_delta[:, None])

        _report("Rasterising humidity texture…")
        h_pixel = per_pixel_humidity(q_air.astype(np.float32), mesh, n_h)      # (N, H, W, K)

        _report("Rasterising precipitation texture…")
        u_axes, v_axes, _, _, u_ranges, v_ranges, _ = mesh.face_tangent_frames()
        p_pixel = per_pixel_precipitation(
            h_pixel,
            winds,
            u_axes.astype(np.float64),
            v_axes.astype(np.float64),
            u_ranges.astype(np.float64) * EARTH_RADIUS_KM,
            v_ranges.astype(np.float64) * EARTH_RADIUS_KM,
            elev_pixel,
            orographic_k=pp.orographic_k,
            precipitation_scale=pp.precipitation_scale,
        )                                                                       # (N, H, W, K)

        _report("Classifying biomes…")
        biome_tex = self.classifier(
            t_pixel, p_pixel, np.rad2deg(latitudes_rad)
        )                                                                       # (N, H, W, 1) uint8
        if biome_tex.dtype != np.uint8:
            biome_tex = biome_tex.astype(np.uint8)

        TEMPERATURE.write(mesh, t_pixel.astype(np.float16))
        PRECIPITATION.write(mesh, p_pixel.astype(np.float16))
        mesh.set_face_textures("humidity", h_pixel.astype(np.float16), n_channels=N_SEASONS)
        mesh.set_face_textures("biome", biome_tex, n_channels=1)

        # Per-face metadata aggregates — p_face derived from pixel averages for consistency.
        p_face = p_pixel.mean(axis=(1, 2))                                      # (N, K) mm/season
        t_annual = t_face.mean(axis=1)
        t_warmest = t_face.max(axis=1)
        t_coldest = t_face.min(axis=1)
        p_annual = p_face.sum(axis=1)
        biome_face = _dominant_biome_per_face(biome_tex)

        TEMPERATURE_ANNUAL.write(mesh, t_annual)
        TEMPERATURE_WARMEST.write(mesh, t_warmest)
        TEMPERATURE_COLDEST.write(mesh, t_coldest)
        PRECIPITATION_ANNUAL.write(mesh, p_annual)
        BIOME.write(mesh, biome_face)
        DISTANCE_TO_COAST.write(mesh, dist_coast)
        mesh.apply_metadata_channel("humidity_annual", q_air.mean(axis=1).tolist())

        self.insolation = insolation
        self.distance_to_coast = dist_coast
        self.temperature_face = t_face
        self.precipitation_face = p_face
        self.humidity_face = q_air
        self.winds = winds
        self.currents = currents_result

        centroids = mesh.face_centroids.astype(np.float32)
        if self.winds is not None:
            ctx.world.add_feature(
                "winds",
                [
                    (centroids.copy(), self.winds[:, s, :].astype(np.float32))
                    for s in range(self.winds.shape[1])
                ],
            )
        if self.currents is not None:
            ctx.world.add_feature(
                "currents",
                [(
                    centroids.copy(),
                    self.currents.currents.astype(np.float32),
                    self.currents.temperature_delta.astype(np.float32),
                )],
            )

        _report("Climate done.")

    def lake_feedback_stage(self) -> "ClimateLakeFeedbackStage":
        """The second climate pass as its own pipeline node (runs after hydrology)."""
        return ClimateLakeFeedbackStage(self)

    def apply_lake_humidity_feedback(
        self,
        mesh: PolygonMesh,
        lake_mask_tex: np.ndarray,
        progress: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Add lake evaporation as an extra moisture source, then refresh precipitation and biomes.

        Lakes are open-water evaporation sources proportional to their area and
        temperature. This single correction pass approximates the hydrology→climate
        feedback without re-running the full pipeline.

        Call after build() and after HydrologyBuilder.build() has attached the
        lake_mask texture to the mesh.
        """
        assert self.humidity_face is not None, "ClimateBuilder.build() must be called first"
        pp = self.params.precipitation

        if pp.lake_feedback_iter <= 0 or pp.lake_evap_scale <= 0.0:
            return

        # Per-face lake area fraction.
        lake_frac = lake_mask_tex[:, :, :, 0].mean(axis=(1, 2)).astype(np.float32)  # (N,)
        if lake_frac.max() < 1e-6:
            return

        def _report(msg: str) -> None:
            if progress is not None:
                progress(msg)

        _report("Applying lake humidity feedback…")

        # Lake evaporation: same SST-driven formula as ocean, scaled by lake area fraction.
        t_factor = np.clip(
            (self.temperature_face.astype(np.float32) + 25.0) / 52.0, 0.0, 1.0
        )                                                              # (N, K)
        lake_evap = (
            lake_frac[:, None] * t_factor * (pp.lake_evap_scale * pp.evaporation_scale)
        ).astype(np.float32)                                           # (N, K)

        # Advect lake-sourced moisture; fewer iterations than the main pass since
        # lake sources are small and the effect is localized.
        dq_lake = advect_moisture(
            lake_evap,
            mesh.face_centroids.astype(np.float64),
            mesh.face_adjacency,
            self.winds.astype(np.float64),
            n_iter=pp.lake_feedback_iter,
            decay_rate=pp.moisture_decay_rate,
        ).astype(np.float32)                                           # (N, K)

        # Smooth just the delta to soften face-boundary artefacts before adding.
        dq_lake = smooth_humidity(dq_lake, mesh.face_adjacency,
                                  n_passes=pp.humidity_smoothing_passes,
                                  alpha=pp.humidity_smoothing_alpha)

        q_updated = (self.humidity_face + dq_lake).astype(np.float32)
        self.humidity_face = q_updated

        # Re-rasterize humidity → precipitation → biomes.
        elev_tex = ELEVATION.read(mesh)                                # (N, H, W, 1)
        n_h = elev_tex.shape[1]

        if self.currents is not None:
            q_updated = q_updated * (1.0 + self.currents.precipitation_delta[:, None])

        h_pixel = per_pixel_humidity(q_updated.astype(np.float32), mesh, n_h)  # (N, H, W, K)

        u_axes, v_axes, _, _, u_ranges, v_ranges, _ = mesh.face_tangent_frames()
        p_pixel = per_pixel_precipitation(
            h_pixel,
            self.winds,
            u_axes.astype(np.float64),
            v_axes.astype(np.float64),
            u_ranges.astype(np.float64) * EARTH_RADIUS_KM,
            v_ranges.astype(np.float64) * EARTH_RADIUS_KM,
            elev_tex[:, :, :, 0].astype(np.float64),
            orographic_k=pp.orographic_k,
            precipitation_scale=pp.precipitation_scale,
        )                                                              # (N, H, W, K)

        latitudes_rad = face_latitudes(mesh.face_centroids).astype(np.float64)
        biome_tex = self.classifier(
            mesh.face_textures["temperature"], p_pixel, np.rad2deg(latitudes_rad)
        )
        if biome_tex.dtype != np.uint8:
            biome_tex = biome_tex.astype(np.uint8)

        PRECIPITATION.write(mesh, p_pixel.astype(np.float16))
        mesh.set_face_textures("humidity", h_pixel.astype(np.float16), n_channels=N_SEASONS)
        mesh.set_face_textures("biome", biome_tex, n_channels=1)

        # Refresh per-face metadata aggregates.
        p_face = p_pixel.mean(axis=(1, 2))                             # (N, K)
        p_annual = p_face.sum(axis=1)
        biome_face = _dominant_biome_per_face(biome_tex)
        PRECIPITATION_ANNUAL.write(mesh, p_annual)
        BIOME.write(mesh, biome_face)
        mesh.apply_metadata_channel("humidity_annual", q_updated.mean(axis=1).tolist())
        self.precipitation_face = p_face

        _report("Lake humidity feedback applied.")

    # ── Display ──────────────────────────────────────────────────────────────

    def register_display(
        self, world: World, reg: DisplayRegistry, *, debug: bool = False
    ) -> None:
        mesh = world.mesh
        if "temperature" not in mesh.face_textures:
            return
        elev_tex = mesh.face_textures["elevation"]

        mesh.set_face_textures("biome_rgba", build_biome_display_texture(mesh))
        reg.add_scheme(
            "Biome", biome_color,
            texture="biome_rgba", legend=("categorical", biome_legend_entries()),
        )
        if not debug:
            return

        temp_tex = mesh.face_textures["temperature"]
        precip_tex = mesh.face_textures["precipitation"]
        ocean_mask_tex = mesh.face_textures.get("ocean_mask")
        mesh.set_face_textures(
            "temperature_annual_rgba",
            build_temperature_display_texture(temp_tex, elevation_tex=elev_tex, ocean_mask_tex=ocean_mask_tex),
        )
        mesh.set_face_textures(
            "temperature_summer_rgba",
            build_temperature_display_texture(
                temp_tex, season=1, elevation_tex=elev_tex, ocean_mask_tex=ocean_mask_tex
            ),
        )
        mesh.set_face_textures(
            "temperature_winter_rgba",
            build_temperature_display_texture(
                temp_tex, season=3, elevation_tex=elev_tex, ocean_mask_tex=ocean_mask_tex
            ),
        )
        mesh.set_face_textures(
            "precipitation_rgba",
            build_precipitation_display_texture(precip_tex, elevation_tex=elev_tex, ocean_mask_tex=ocean_mask_tex),
        )
        reg.add_scheme(
            "Annual Temperature", temperature_color,
            texture="temperature_annual_rgba", legend=("gradient", temperature_legend_entries()),
        )
        reg.add_scheme(
            "Summer Temperature", lambda meta: temperature_color(meta, key="temperature_warmest"),
            texture="temperature_summer_rgba", legend=("gradient", temperature_legend_entries()),
        )
        reg.add_scheme(
            "Winter Temperature", lambda meta: temperature_color(meta, key="temperature_coldest"),
            texture="temperature_winter_rgba", legend=("gradient", temperature_legend_entries()),
        )
        reg.add_scheme(
            "Annual Precipitation", precipitation_color,
            texture="precipitation_rgba", legend=("gradient", precipitation_legend_entries()),
        )
        reg.add_scheme("Humidity", humidity_color)

        if self.winds is not None:
            for i, label in enumerate(SEASON_LABELS):
                reg.add_overlay(f"winds_season_{i}", f"Winds — {label}", checked=False)
        if self.currents is not None:
            reg.add_overlay("currents", "Currents", checked=False)

        from worldgen.climate.classification import BIOME_NAMES

        def _climate_hover(face: int, _uv, _w) -> list:
            meta = mesh.face_metadata[face]
            parts: list = []
            if "biome" in meta:
                parts.append(BIOME_NAMES[int(meta["biome"])])
            if "temperature_annual" in meta:
                parts.append(f"T: {meta['temperature_annual']:.1f}°C")
            if "precipitation_annual" in meta:
                parts.append(f"P: {meta['precipitation_annual']:.0f} mm/yr")
            return parts

        reg.add_hover(_climate_hover)

    # ── Persistence ──────────────────────────────────────────────────────────

    def serialize(self) -> dict:
        arrays: dict = {}
        if self.winds is not None:
            arrays["cb_winds"] = self.winds.astype(np.float32)
        if self.currents is not None:
            arrays["cb_currents_vectors"] = self.currents.currents.astype(np.float32)
            arrays["cb_currents_streamfunction"] = self.currents.streamfunction.astype(np.float32)
            arrays["cb_currents_temp_delta"] = self.currents.temperature_delta.astype(np.float32)
            arrays["cb_currents_precip_delta"] = self.currents.precipitation_delta.astype(np.float32)
            arrays["cb_currents_advected_insolation"] = self.currents.advected_insolation.astype(np.float32)
        return arrays

    def load(self, arrays: Mapping[str, np.ndarray], ctx: BuildContext) -> None:
        centroids = ctx.mesh.face_centroids.astype(np.float32)
        if "cb_winds" in arrays:
            self.winds = arrays["cb_winds"]
            ctx.world.add_feature(
                "winds",
                [
                    (centroids.copy(), self.winds[:, s, :].astype(np.float32))
                    for s in range(self.winds.shape[1])
                ],
            )
        if "cb_currents_vectors" in arrays:
            self.currents = CurrentsResult(
                currents=arrays["cb_currents_vectors"],
                streamfunction=arrays["cb_currents_streamfunction"],
                temperature_delta=arrays["cb_currents_temp_delta"],
                precipitation_delta=arrays["cb_currents_precip_delta"],
                advected_insolation=arrays["cb_currents_advected_insolation"],
            )
            ctx.world.add_feature(
                "currents",
                [(
                    centroids.copy(),
                    self.currents.currents.astype(np.float32),
                    self.currents.temperature_delta.astype(np.float32),
                )],
            )


class ClimateLakeFeedbackStage(Stage):
    """Second climate pass: lake evaporation feedback (runs after hydrology).

    Shares the :class:`ClimateBuilder` instance so it reuses the per-face climate
    state computed in the first pass.
    """

    name = "climate_lake_feedback"

    def __init__(self, builder: ClimateBuilder) -> None:
        self.builder = builder

    def build(self, ctx: BuildContext) -> None:
        mesh = ctx.mesh
        if LAKE_MASK.present(mesh):
            self.builder.apply_lake_humidity_feedback(
                mesh, LAKE_MASK.read(mesh), progress=ctx.report
            )


