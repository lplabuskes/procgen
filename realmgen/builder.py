"""RealmBuilder — orchestrator for the human geography layer.

Downstream of the full worldgen pipeline. Takes a fully-built :class:`Globe`
and attaches realm features. Stage 1 is complete: suitability scorer and static
situational value (see docs/settlement_generation_design.md).

Usage::

    from realmgen.builder import RealmBuilder, RealmParams

    params = RealmParams()
    builder = RealmBuilder(params)
    builder.build(globe)
    # globe.mesh.face_metadata[i]["suitability"] and ["situational_value"] are set
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from realmgen.suitability import (
    _positions_to_face_ids,
    combine_suitability,
    compute_climate_habitability,
    compute_resource_proximity,
    compute_ruggedness_penalty,
    compute_water_proximity,
)
from realmgen.situational_value import (
    combine_situational_value,
    compute_harbor_quality,
    compute_mountain_passes,
    compute_navigable_heads,
    compute_river_confluences,
    compute_strait_adjacency,
)
from worldgen.geology.display import OCEAN_COLOR
from worldgen.globe.globe import Globe
from worldgen.resources.terrain import face_elevation_range
from core.params import field, param_field as _f, paramclass


# ── Params ────────────────────────────────────────────────────────────────────


@paramclass
class WaterProximityParams:
    freshwater_decay: float = _f(0.4, label="Freshwater hop decay", min=0.05, max=3.0)
    freshwater_max_hops: int = _f(12, label="Freshwater max hops", min=1, max=32)
    coast_scale_km: float = _f(150.0, label="Coastal scale (km)", min=10.0, max=2000.0)
    coast_weight: float = _f(0.4, label="Coastal weight", min=0.0, max=1.0)


@paramclass
class ClimateHabitabilityParams:
    t_grow_min_c: float = _f(0.0, label="Min growing T (°C)", min=-10.0, max=20.0)
    t_grow_opt_c: float = _f(18.0, label="Opt growing T (°C)", min=5.0, max=35.0)
    t_cold_min_c: float = _f(-30.0, label="Cold survival T (°C)", min=-60.0, max=0.0)
    p_dry_mm: float = _f(100.0, label="Dry threshold (mm/yr)", min=0.0, max=500.0)
    p_opt_mm: float = _f(800.0, label="Opt precip (mm/yr)", min=100.0, max=3000.0)
    p_wet_mm: float = _f(3000.0, label="Wet limit (mm/yr)", min=500.0, max=10000.0)


@paramclass
class RuggednessParams:
    penalty_scale_km: float = _f(1.0, label="Ruggedness penalty scale (km)", min=0.05, max=10.0)


@paramclass
class ResourceAccessParams:
    decay: float = _f(0.2, label="Resource hop decay", min=0.05, max=2.0)
    max_hops: int = _f(16, label="Resource max hops", min=1, max=32)


@paramclass
class SuitabilityWeightParams:
    water_weight: float = _f(0.30, label="Water weight", min=0.0, max=1.0)
    climate_weight: float = _f(0.30, label="Climate weight", min=0.0, max=1.0)
    fertility_weight: float = _f(0.25, label="Fertility weight", min=0.0, max=1.0)
    resource_weight: float = _f(0.15, label="Resource weight", min=0.0, max=1.0)


@paramclass
class HarborQualityParams:
    depth_scale_km: float = _f(0.5, label="Harbor depth scale (km)", min=0.05, max=5.0)


@paramclass
class MountainPassParams:
    min_neighbor_cost: float = _f(3.0, label="Min neighbor cost for pass", min=0.5, max=10.0)
    cost_scale: float = _f(1.5, label="Pass contrast scale", min=0.1, max=5.0)


@paramclass
class SituationalValueWeightParams:
    harbor_weight: float = _f(0.25, label="Harbor weight", min=0.0, max=1.0)
    confluence_weight: float = _f(0.20, label="Confluence weight", min=0.0, max=1.0)
    nav_head_weight: float = _f(0.20, label="Nav. river head weight", min=0.0, max=1.0)
    pass_weight: float = _f(0.20, label="Pass weight", min=0.0, max=1.0)
    strait_weight: float = _f(0.15, label="Strait weight", min=0.0, max=1.0)


@paramclass
class SituationalValueParams:
    harbor: HarborQualityParams = field(
        default_factory=HarborQualityParams, metadata={"label": "Harbor quality"}
    )
    passes: MountainPassParams = field(
        default_factory=MountainPassParams, metadata={"label": "Mountain passes"}
    )
    weights: SituationalValueWeightParams = field(
        default_factory=SituationalValueWeightParams, metadata={"label": "Weights"}
    )


@paramclass
class RealmParams:
    water: WaterProximityParams = field(
        default_factory=WaterProximityParams, metadata={"label": "Water proximity"}
    )
    climate: ClimateHabitabilityParams = field(
        default_factory=ClimateHabitabilityParams, metadata={"label": "Climate"}
    )
    ruggedness: RuggednessParams = field(
        default_factory=RuggednessParams, metadata={"label": "Ruggedness"}
    )
    resource: ResourceAccessParams = field(
        default_factory=ResourceAccessParams, metadata={"label": "Resource access"}
    )
    weights: SuitabilityWeightParams = field(
        default_factory=SuitabilityWeightParams, metadata={"label": "Weights"}
    )
    situational: SituationalValueParams = field(
        default_factory=SituationalValueParams, metadata={"label": "Situational value"}
    )


# ── Builder ───────────────────────────────────────────────────────────────────


def _read_meta_floats(face_metadata: list[dict], key: str, default: float = 0.0) -> np.ndarray:
    return np.array([float(m.get(key, default)) for m in face_metadata], dtype=np.float64)


def _read_meta_bools(face_metadata: list[dict], key: str, default: bool = False) -> np.ndarray:
    return np.array([bool(m.get(key, default)) for m in face_metadata], dtype=bool)


class RealmBuilder:
    """Produces human geography features from a fully-built ``Globe``.

    Call :meth:`build` after ``GlobeBuilder.build()`` so the ``biome``,
    ``fertility_mean``, ``temperature_*``, ``precipitation_annual``, and
    ``is_ocean`` / ``is_water`` per-face metadata and the ``elevation`` texture
    are present, along with the ``cost_surfaces`` and ``minerals`` / ``rare_bio``
    globe features.

    Stores:

    * ``self.suitability`` — ``(N,)`` float32 in [0, 1].
    * ``self.situational_value`` — ``(N,)`` float32 in [0, 1].

    Writes per-face metadata keys ``"suitability"`` and ``"situational_value"``.
    """

    def __init__(self, params: RealmParams | None = None) -> None:
        self.params: RealmParams = params or RealmParams()
        self.suitability: Optional[np.ndarray] = None
        self.situational_value: Optional[np.ndarray] = None

    def build(
        self,
        globe: Globe,
        progress: Optional[Callable[[str], None]] = None,
    ) -> np.ndarray:
        """Compute suitability for every face and write it to face metadata.

        Args:
            globe:    Fully-built Globe (all worldgen stages done).
            progress: Optional callback for progress messages.

        Returns:
            ``(N,)`` float32 suitability array (also stored as
            ``self.suitability`` and in ``mesh.face_metadata[i]["suitability"]``).
        """
        def _report(msg: str) -> None:
            if progress is not None:
                progress(msg)

        mesh = globe.mesh
        n = len(mesh.faces)
        p = self.params

        # ── Open-water mask (impassable for settlement) ──────────────────────
        _report("Reading physical layers…")
        # Prefer the cost stage's is_water metadata (ocean ∪ lake); fall back
        # to is_ocean if the cost stage has not been run.
        if "is_water" in mesh.face_metadata[0]:
            is_water = _read_meta_bools(mesh.face_metadata, "is_water")
        else:
            is_water = _read_meta_bools(mesh.face_metadata, "is_ocean")

        # ── Freshwater seed mask (rivers + lakes) ───────────────────────────
        # Rivers: use cost_surfaces.river_flow_mag > 0 which covers all
        # non-in_lake river segments (the most accurate signal available).
        river_mask = np.zeros(n, dtype=bool)
        if globe.has_feature("cost_surfaces"):
            cs = globe.get_feature("cost_surfaces")
            river_mask = cs.river_flow_mag > 0.0

        # Lakes: non-ocean water faces (coast stage adds is_water but not all
        # callers will have the cost stage — use the texture directly if present).
        lake_mask = np.zeros(n, dtype=bool)
        if "lake_mask" in mesh.face_textures:
            lake_tex = mesh.face_textures["lake_mask"]
            lake_mask = lake_tex[..., 0].mean(axis=(1, 2)) >= 0.5
        else:
            # Fall back: any face flagged is_water but not is_ocean is a lake.
            is_ocean = _read_meta_bools(mesh.face_metadata, "is_ocean")
            lake_mask = is_water & ~is_ocean

        freshwater_mask = river_mask | lake_mask

        # ── Per-face climate and fertility ───────────────────────────────────
        dist_coast_km = _read_meta_floats(mesh.face_metadata, "distance_to_coast")
        fertility = _read_meta_floats(mesh.face_metadata, "fertility_mean").clip(0.0, 1.0)
        t_warmest = _read_meta_floats(mesh.face_metadata, "temperature_warmest")
        t_coldest = _read_meta_floats(mesh.face_metadata, "temperature_coldest")
        p_annual = _read_meta_floats(mesh.face_metadata, "precipitation_annual")

        # ── Elevation range for ruggedness penalty ───────────────────────────
        elev_range = face_elevation_range(mesh.face_textures["elevation"])

        # ── Resource deposit positions → face seeds ─────────────────────────
        _report("Mapping resource deposits to faces…")
        resource_mask = np.zeros(n, dtype=bool)
        centroids = np.asarray(mesh.face_centroids, dtype=np.float64)
        for feature_name in ("minerals", "rare_bio"):
            if globe.has_feature(feature_name):
                positions, _, _ = globe.get_feature(feature_name)
                if len(positions) > 0:
                    fids = _positions_to_face_ids(positions, centroids)
                    resource_mask[fids] = True

        # ── Factor computation ───────────────────────────────────────────────
        _report("Computing water proximity…")
        water_score = compute_water_proximity(
            mesh,
            freshwater_mask,
            dist_coast_km,
            freshwater_decay=p.water.freshwater_decay,
            freshwater_max_hops=p.water.freshwater_max_hops,
            coast_scale_km=p.water.coast_scale_km,
            coast_weight=p.water.coast_weight,
        )

        _report("Computing climate habitability…")
        climate_score = compute_climate_habitability(
            t_warmest,
            t_coldest,
            p_annual,
            t_grow_min_c=p.climate.t_grow_min_c,
            t_grow_opt_c=p.climate.t_grow_opt_c,
            t_cold_min_c=p.climate.t_cold_min_c,
            p_dry_mm=p.climate.p_dry_mm,
            p_opt_mm=p.climate.p_opt_mm,
            p_wet_mm=p.climate.p_wet_mm,
        )

        _report("Computing resource access…")
        resource_score = compute_resource_proximity(
            mesh,
            resource_mask,
            resource_decay=p.resource.decay,
            resource_max_hops=p.resource.max_hops,
        )

        _report("Combining suitability…")
        suitability = combine_suitability(
            is_water,
            water_score,
            climate_score,
            fertility.astype(np.float32),
            resource_score,
            water_weight=p.weights.water_weight,
            climate_weight=p.weights.climate_weight,
            fertility_weight=p.weights.fertility_weight,
            resource_weight=p.weights.resource_weight,
        )

        ruggedness_mult = compute_ruggedness_penalty(
            elev_range,
            penalty_scale_km=p.ruggedness.penalty_scale_km,
        )
        suitability = (suitability * ruggedness_mult).astype(np.float32)

        self.suitability = suitability
        mesh.apply_metadata_channel("suitability", suitability.astype(float).tolist())

        # ── Situational value ────────────────────────────────────────────────
        _report("Computing situational value…")
        sp = p.situational

        # Harbor quality: needs per-face mean elevation and ocean mask.
        is_ocean = _read_meta_bools(mesh.face_metadata, "is_ocean")
        face_mean_elev_km = mesh.face_textures["elevation"][..., 0].mean(axis=(1, 2))
        harbor_score = compute_harbor_quality(
            mesh, is_ocean, face_mean_elev_km,
            depth_scale_km=sp.harbor.depth_scale_km,
        )

        # River confluences and navigable heads use the globe's rivers feature
        # (positions/flows/in_lake tuples) with a centroid lookup.
        rivers = globe.get_feature("rivers") if globe.has_feature("rivers") else None
        confluence_score = compute_river_confluences(rivers, centroids, n)

        # Navigable river heads: fall-line faces per river.
        river_navigability = np.zeros(n, dtype=np.float32)
        if globe.has_feature("cost_surfaces"):
            river_navigability = globe.get_feature("cost_surfaces").river_navigability
        nav_head_score = compute_navigable_heads(rivers, centroids, river_navigability, n)

        # Mountain passes: local land-cost minima in high-cost terrain.
        land_base_cost = np.zeros(n, dtype=np.float32)
        if globe.has_feature("cost_surfaces"):
            land_base_cost = globe.get_feature("cost_surfaces").land_base_cost
        pass_score = compute_mountain_passes(
            mesh, land_base_cost, is_water,
            min_neighbor_cost=sp.passes.min_neighbor_cost,
            cost_scale=sp.passes.cost_scale,
        )

        # Strait adjacency: land faces bordering tagged strait ocean faces.
        is_strait = _read_meta_bools(mesh.face_metadata, "is_strait")
        strait_score = compute_strait_adjacency(mesh, is_strait, is_water)

        sw = sp.weights
        situational_value = combine_situational_value(
            is_water,
            harbor_score,
            confluence_score,
            nav_head_score,
            pass_score,
            strait_score,
            harbor_weight=sw.harbor_weight,
            confluence_weight=sw.confluence_weight,
            nav_head_weight=sw.nav_head_weight,
            pass_weight=sw.pass_weight,
            strait_weight=sw.strait_weight,
        )

        self.situational_value = situational_value
        mesh.apply_metadata_channel(
            "situational_value", situational_value.astype(float).tolist()
        )

        _report("Situational value done.")
        return suitability

    def apply_display(self, window) -> None:
        """Register realmgen color schemes into a MapWindow.

        Call after :meth:`build` and after ``GlobeBuilder.apply_display`` so
        the schemes are appended alongside the worldgen schemes.  Only adds
        schemes for results that have actually been computed.

        Args:
            window: ``MapWindow`` instance.
        """
        schemes: dict = {}

        if self.suitability is not None:
            def _suitability_color(meta: dict) -> tuple:
                if meta.get("is_water", meta.get("is_ocean", False)):
                    return OCEAN_COLOR
                s = min(max(float(meta.get("suitability", 0.0)), 0.0), 1.0)
                if s < 0.5:
                    t = s / 0.5
                    return (0.45 + 0.20 * t, 0.38 + 0.42 * t, 0.28 - 0.08 * t, 1.0)
                else:
                    t = (s - 0.5) / 0.5
                    return (0.65 - 0.55 * t, 0.80 - 0.20 * t, 0.20, 1.0)
            schemes["Suitability"] = _suitability_color

        if self.situational_value is not None:
            def _situational_color(meta: dict) -> tuple:
                if meta.get("is_water", meta.get("is_ocean", False)):
                    return OCEAN_COLOR
                s = min(max(float(meta.get("situational_value", 0.0)), 0.0), 1.0)
                if s < 0.5:
                    t = s / 0.5
                    return (0.35 + 0.50 * t, 0.33 + 0.42 * t, 0.30 + 0.00 * t, 1.0)
                else:
                    t = (s - 0.5) / 0.5
                    return (0.85 + 0.05 * t, 0.75 - 0.20 * t, 0.30 - 0.20 * t, 1.0)
            schemes["Situational Value"] = _situational_color

        if not schemes:
            return

        # Merge into the existing scheme dict so the combo box keeps all entries.
        merged = dict(window._map._color_schemes)
        merged.update(schemes)
        window.set_color_schemes(merged)
