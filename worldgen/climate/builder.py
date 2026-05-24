"""Climate stage orchestrator — ClimateParams and ClimateBuilder."""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from worldgen.climate.classification import (
    N_BIOMES,
    classify_textures,
    default_biome,
)
from worldgen.climate.currents import CurrentsResult, compute_currents
from worldgen.climate.insolation import (
    SEASON_DECLINATIONS_DEG,
    face_latitudes,
    seasonal_insolation,
)
from worldgen.climate.precipitation import (
    per_pixel_precipitation,
    surface_precipitation,
)
from worldgen.climate.temperature import (
    per_pixel_temperatures,
    surface_temperatures,
)
from worldgen.climate.winds import SEASON_PHASES, pressure_winds, zonal_winds
from worldgen.mesh.mymesh import PolygonMesh


EARTH_RADIUS_KM = 6371.0
N_SEASONS = 4


def _f(default, *, label: str, min: float, max: float, step: float | None = None,
       hidden: bool = False):
    meta = {"label": label, "min": min, "max": max}
    if step is not None:
        meta["step"] = step
    if hidden:
        meta["hidden"] = True
    return field(default=default, metadata=meta)


@dataclass
class TemperatureParams:
    equator_annual_c: float = _f(27.0, label="Equator annual mean (°C)", min=-10.0, max=40.0)
    pole_annual_c: float = _f(-25.0, label="Pole annual mean (°C)", min=-60.0, max=10.0)
    anomaly_scale: float = _f(0.05, label="Seasonal anomaly scale", min=0.0, max=0.5)
    lapse_rate_c_per_km: float = _f(6.5, label="Lapse rate (°C/km)", min=0.0, max=15.0)
    ocean_inertia: float = _f(0.30, label="Ocean inertia", min=0.0, max=1.0)
    land_inertia: float = _f(1.0, label="Land inertia", min=0.0, max=2.0)
    inertia_distance_km: float = _f(1500.0, label="Inertia saturation (km)", min=100.0, max=5000.0)


@dataclass
class WindParams:
    trade_speed: float = _f(7.0, label="Trade winds (m/s)", min=0.0, max=30.0)
    westerly_speed: float = _f(10.0, label="Westerlies (m/s)", min=0.0, max=30.0)
    polar_speed: float = _f(5.0, label="Polar easterlies (m/s)", min=0.0, max=30.0)
    itcz_shift_deg: float = _f(10.0, label="ITCZ shift (°)", min=0.0, max=30.0)
    pressure_wind_scale: float = _f(
        600.0, label="Pressure wind scale (m·s⁻¹·km/°C)", min=0.0, max=1000.0
    )


@dataclass
class PrecipitationParams:
    itcz_mm_annual: float = _f(2200.0, label="ITCZ baseline (mm/yr)", min=0.0, max=5000.0)
    horse_mm_annual: float = _f(250.0, label="Horse latitudes (mm/yr)", min=0.0, max=2000.0)
    midlat_mm_annual: float = _f(900.0, label="Mid-latitude (mm/yr)", min=0.0, max=3000.0)
    polar_mm_annual: float = _f(200.0, label="Polar (mm/yr)", min=0.0, max=2000.0)
    orographic_k: float = _f(0.4, label="Orographic K", min=0.0, max=2.0)
    continentality_decay_km: float = _f(1200.0, label="Continentality decay (km)", min=100.0, max=5000.0)


@dataclass
class CurrentsParams:
    enabled: bool = _f(True, label="Enable currents", min=0, max=1)
    beta_scale: float = _f(1.0, label="β scale (western intensification)", min=0.0, max=5.0)
    advection_alpha: float = _f(1.0, label="Advection mixing α", min=0.0, max=1.0)
    advection_iterations: int = _f(400, label="Advection iterations", min=10, max=2000)
    temperature_scale: float = _f(0.06, label="Current ΔT scale (°C / (W/m²))", min=0.0, max=0.5)
    precipitation_scale: float = _f(0.03, label="Current wet scale (per °C)", min=0.0, max=0.2)
    inland_decay_km: float = _f(600.0, label="Inland decay (km)", min=50.0, max=5000.0)


@dataclass
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
                          sea_level: float = 0.0) -> np.ndarray:
    """Dijkstra on the face graph from ocean seeds; edge weight = great-circle km."""
    n = len(mesh.faces)
    centroids = mesh.face_centroids.astype(np.float64)
    is_ocean = face_elev_km <= sea_level
    dist = np.full(n, np.inf, dtype=np.float64)

    heap: list[tuple[float, int]] = []
    for i in np.flatnonzero(is_ocean):
        dist[i] = 0.0
        heapq.heappush(heap, (0.0, int(i)))

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v in mesh.face_adjacency[u]:
            cos_ang = float(np.clip(centroids[u] @ centroids[v], -1.0, 1.0))
            edge = EARTH_RADIUS_KM * np.arccos(cos_ang)
            nd = d + edge
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, int(v)))

    dist[~np.isfinite(dist)] = 0.0  # land with no ocean reachable: treat as coast
    return dist.astype(np.float32)


def _face_elevation_gradient(mesh: PolygonMesh,
                             face_elev_km: np.ndarray) -> np.ndarray:
    """Least-squares 3-D elevation gradient (km/km) projected onto each face's tangent plane."""
    n = len(mesh.faces)
    centroids = mesh.face_centroids.astype(np.float64)
    grad = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        neighbours = mesh.face_adjacency[i]
        if not neighbours:
            continue
        nbrs = np.asarray(neighbours, dtype=np.int64)
        delta_pos = (centroids[nbrs] - centroids[i]) * EARTH_RADIUS_KM      # (k, 3) km
        delta_h = face_elev_km[nbrs] - face_elev_km[i]                       # (k,) km
        g, *_ = np.linalg.lstsq(delta_pos, delta_h, rcond=None)
        n_hat = centroids[i]
        grad[i] = g - (g @ n_hat) * n_hat
    return grad.astype(np.float64)


def _dominant_biome_per_face(biome_tex: np.ndarray) -> np.ndarray:
    """Mode of the (N, H, W, 1) uint8 biome texture over each face."""
    n = biome_tex.shape[0]
    flat = biome_tex.reshape(n, -1)
    result = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        counts = np.bincount(flat[i], minlength=N_BIOMES)
        result[i] = counts.argmax()
    return result


class ClimateBuilder:
    """Runs the climate stage on a built elevation mesh.

    Call `build(mesh, elevation_textures)` after `ElevationBuilder.build()`.
    Attaches `temperature` (N,H,W,4 float32 °C), `precipitation` (N,H,W,4
    float32 mm/season), and `biome` (N,H,W,1 uint8) textures, and writes
    per-face aggregates into `mesh.face_metadata`. Intermediate per-face
    results are kept as attributes so downstream code (display, future
    currents stage) can read them without re-running anything.
    """

    def __init__(self,
                 params: ClimateParams | None = None,
                 classifier: Callable[..., np.ndarray] = classify_textures) -> None:
        self.params: ClimateParams = params or ClimateParams()
        self.classifier = classifier

        self.insolation: Optional[np.ndarray] = None             # (N, K) W/m²
        self.distance_to_coast: Optional[np.ndarray] = None      # (N,) km
        self.temperature_face: Optional[np.ndarray] = None       # (N, K) °C
        self.precipitation_face: Optional[np.ndarray] = None     # (N, K) mm/season
        self.winds: Optional[np.ndarray] = None                  # (N, K, 3) m/s
        self.currents: Optional[CurrentsResult] = None           # populated when params.currents.enabled

    def build(self,
              mesh: PolygonMesh,
              elevation_textures: np.ndarray,
              progress: Optional[Callable[[str], None]] = None) -> None:
        def _report(msg: str) -> None:
            if progress is not None:
                progress(msg)

        p = self.params
        tp, wp, pp, cp = p.temperature, p.wind, p.precipitation, p.currents

        elev_tex = elevation_textures[:, :, :, 0]                              # (N, H, W) km
        face_elev = elev_tex.mean(axis=(1, 2)).astype(np.float64)              # (N,) km
        latitudes_rad = face_latitudes(mesh.face_centroids).astype(np.float64)  # (N,)

        _report("Computing insolation…")
        decls = np.deg2rad(SEASON_DECLINATIONS_DEG)
        insolation = seasonal_insolation(latitudes_rad, decls)                 # (N, K)

        _report("Computing distance to coast…")
        dist_coast = _distance_to_coast_km(mesh, face_elev)                    # (N,)

        _report("Computing surface temperatures…")
        t_face = surface_temperatures(
            latitudes_rad,
            insolation,
            face_elev,
            dist_coast.astype(np.float64),
            equator_annual_c=tp.equator_annual_c,
            pole_annual_c=tp.pole_annual_c,
            anomaly_scale=tp.anomaly_scale,
            lapse_rate_c_per_km=tp.lapse_rate_c_per_km,
            ocean_inertia=tp.ocean_inertia,
            land_inertia=tp.land_inertia,
            inertia_distance_km=tp.inertia_distance_km,
        )

        _report("Computing prevailing winds…")
        winds = zonal_winds(
            mesh.face_centroids.astype(np.float64),
            latitudes_rad,
            trade_speed=wp.trade_speed,
            westerly_speed=wp.westerly_speed,
            polar_speed=wp.polar_speed,
            itcz_shift_deg=wp.itcz_shift_deg,
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

        _report("Computing precipitation…")
        face_grad = _face_elevation_gradient(mesh, face_elev)
        p_face = surface_precipitation(
            latitudes_rad,
            winds.astype(np.float64),
            face_grad,
            dist_coast.astype(np.float64),
            itcz_mm_annual=pp.itcz_mm_annual,
            horse_mm_annual=pp.horse_mm_annual,
            midlat_mm_annual=pp.midlat_mm_annual,
            polar_mm_annual=pp.polar_mm_annual,
            orographic_k=pp.orographic_k,
            continentality_decay_km=pp.continentality_decay_km,
            itcz_shift_deg=wp.itcz_shift_deg,
        )

        currents_result: Optional[CurrentsResult] = None
        if cp.enabled:
            _report("Computing ocean currents…")
            currents_result = compute_currents(
                mesh,
                face_elev,
                mesh.face_centroids.astype(np.float64),
                winds,
                insolation.mean(axis=1),
                beta_scale=cp.beta_scale,
                advection_alpha=cp.advection_alpha,
                advection_iterations=cp.advection_iterations,
                temperature_scale=cp.temperature_scale,
                precipitation_scale=cp.precipitation_scale,
                inland_decay_km=cp.inland_decay_km,
            )

            # Second-pass T and P with the current-driven deltas broadcast across seasons.
            temp_delta = currents_result.temperature_delta[:, None].astype(np.float64)
            wet_delta = currents_result.precipitation_delta[:, None].astype(np.float64)
            t_face = surface_temperatures(
                latitudes_rad,
                insolation,
                face_elev,
                dist_coast.astype(np.float64),
                equator_annual_c=tp.equator_annual_c,
                pole_annual_c=tp.pole_annual_c,
                anomaly_scale=tp.anomaly_scale,
                lapse_rate_c_per_km=tp.lapse_rate_c_per_km,
                ocean_inertia=tp.ocean_inertia,
                land_inertia=tp.land_inertia,
                inertia_distance_km=tp.inertia_distance_km,
                coastal_current_temp_delta=temp_delta,
            )
            p_face = surface_precipitation(
                latitudes_rad,
                winds.astype(np.float64),
                face_grad,
                dist_coast.astype(np.float64),
                itcz_mm_annual=pp.itcz_mm_annual,
                horse_mm_annual=pp.horse_mm_annual,
                midlat_mm_annual=pp.midlat_mm_annual,
                polar_mm_annual=pp.polar_mm_annual,
                orographic_k=pp.orographic_k,
                continentality_decay_km=pp.continentality_decay_km,
                itcz_shift_deg=wp.itcz_shift_deg,
                current_wet_delta=wet_delta,
            )

        _report("Rasterising temperature texture…")
        elev_pixel = elev_tex.astype(np.float64)                                # (N, H, W) km
        t_pixel = per_pixel_temperatures(
            t_face, face_elev, elev_pixel, tp.lapse_rate_c_per_km
        )                                                                       # (N, H, W, K)

        _report("Rasterising precipitation texture…")
        u_axes, v_axes, _, _, u_ranges, v_ranges, _ = mesh.face_tangent_frames()
        p_pixel = per_pixel_precipitation(
            p_face,
            winds,
            u_axes.astype(np.float64),
            v_axes.astype(np.float64),
            u_ranges.astype(np.float64) * EARTH_RADIUS_KM,
            v_ranges.astype(np.float64) * EARTH_RADIUS_KM,
            elev_pixel,
            orographic_k=pp.orographic_k,
        )                                                                       # (N, H, W, K)

        _report("Classifying biomes…")
        biome_tex = self.classifier(
            t_pixel, p_pixel, np.rad2deg(latitudes_rad)
        )                                                                       # (N, H, W, 1) uint8
        if biome_tex.dtype != np.uint8:
            biome_tex = biome_tex.astype(np.uint8)

        mesh.set_face_textures("temperature", t_pixel, n_channels=N_SEASONS)
        mesh.set_face_textures("precipitation", p_pixel, n_channels=N_SEASONS)
        mesh.set_face_textures("biome", biome_tex, n_channels=1)

        # Per-face metadata aggregates.
        t_annual = t_face.mean(axis=1)
        t_warmest = t_face.max(axis=1)
        t_coldest = t_face.min(axis=1)
        p_annual = p_face.sum(axis=1)
        p_range = p_face.max(axis=1) - p_face.min(axis=1)
        p_seasonality = p_range / np.maximum(p_annual, 1e-6)
        biome_face = _dominant_biome_per_face(biome_tex)

        mesh.apply_metadata_channel("temperature_annual", t_annual.tolist())
        mesh.apply_metadata_channel("temperature_warmest", t_warmest.tolist())
        mesh.apply_metadata_channel("temperature_coldest", t_coldest.tolist())
        mesh.apply_metadata_channel("precipitation_annual", p_annual.tolist())
        mesh.apply_metadata_channel("precipitation_seasonality", p_seasonality.tolist())
        mesh.apply_metadata_channel("biome", [int(b) for b in biome_face])
        mesh.apply_metadata_channel("distance_to_coast", dist_coast.tolist())

        self.insolation = insolation
        self.distance_to_coast = dist_coast
        self.temperature_face = t_face
        self.precipitation_face = p_face
        self.winds = winds
        self.currents = currents_result

        _report("Climate done.")


# Silence the unused-import warning while keeping the constant accessible.
_ = (SEASON_PHASES, default_biome)
