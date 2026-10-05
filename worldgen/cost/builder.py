"""Cost-surface stage: the physical base land + sea traversal costs realmgen builds on.

The model lives in :mod:`worldgen.cost.surfaces`; see docs/cost.md.
"""
from __future__ import annotations

from typing import Mapping, Optional

import numpy as np

from worldgen.cost.surfaces import (
    CostSurfaces,
    compute_land_base_cost,
    compute_river_flow_field,
    compute_river_navigability,
    compute_sea_cost,
    compute_sea_distance_to_coast,
    compute_wind_mean_dir,
    detect_straits,
    navigable_river_mask,
)
from core.display import DisplayRegistry
from core.mesh import PolygonMesh
from core.params import field, param_field as _f, paramclass
from core.pipeline import BuildContext, Stage
from core.world import World
from worldgen.climate.channels import BIOME, TEMPERATURE
from worldgen.cost.channels import COST_SURFACES, IS_STRAIT, IS_WATER
from worldgen.delta.channels import MARSH_FRACTION
from worldgen.geology.channels import ELEVATION, IS_OCEAN
from worldgen.geology.display import LAND_COLOR, OCEAN_COLOR, is_water_meta
from worldgen.hydrology.channels import LAKE_MASK, RIVER_OBJECTS


@paramclass
class LandCostParams:
    biome_cost_scale: float = _f(1.0, label="Biome cost scale", min=0.0, max=5.0)
    ruggedness_cost_per_km: float = _f(1.5, label="Ruggedness cost/km", min=0.0, max=10.0)
    river_ford_strength: float = _f(2.0, label="River ford penalty", min=0.0, max=10.0)
    river_corridor_strength: float = _f(0.5, label="River corridor discount", min=0.0, max=1.0)
    river_flow_reference: float = _f(2000.0, label="River flow reference", min=1.0, max=20000.0)
    river_min_mult: float = _f(0.4, label="River min multiplier", min=0.05, max=1.0)
    river_max_mult: float = _f(5.0, label="River max multiplier", min=1.0, max=20.0)
    navigable_ruggedness_km: float = _f(
        0.6, label="Navigable ruggedness (km)", min=0.05, max=5.0
    )
    forest_cost_strength: float = _f(1.0, label="Forest cost strength", min=0.0, max=5.0)
    # Delta marsh: land cost × (1 + strength · the face's marsh fraction).
    marsh_cost_strength: float = _f(3.0, label="Marsh cost strength", min=0.0, max=10.0)


@paramclass
class SeaCostParams:
    coastal_cost: float = _f(0.5, label="Coastal water cost", min=0.01, max=5.0)
    open_ocean_cost: float = _f(2.0, label="Open ocean cost", min=0.1, max=10.0)
    open_ocean_distance_km: float = _f(2000.0, label="Open ocean distance (km)", min=100.0, max=10000.0)
    strait_discount: float = _f(0.5, label="Strait discount", min=0.05, max=1.0)
    navigable_flow_min: float = _f(
        500.0, label="Navigable river flow min", min=1.0, max=20000.0
    )
    ice_penalty_strength: float = _f(5.0, label="Ice penalty strength", min=0.0, max=20.0)
    ice_freeze_threshold: float = _f(0.0, label="Ice freeze threshold (°C)", min=-10.0, max=10.0)
    wind_optimal_speed: float = _f(7.0, label="Wind optimal speed (m/s)", min=0.0, max=30.0)
    wind_speed_width: float = _f(5.0, label="Wind speed bell width (m/s)", min=0.5, max=20.0)
    wind_penalty_strength: float = _f(1.0, label="Wind speed penalty", min=0.0, max=5.0)
    wind_direction_strength: float = _f(0.4, label="Wind direction strength", min=0.0, max=1.0)
    wind_coast_blend_km: float = _f(300.0, label="Wind coast blend (km)", min=0.0, max=2000.0)


@paramclass
class StraitParams:
    opposite_dot_threshold: float = _f(-0.3, label="Strait opposite-dot threshold", min=-1.0, max=0.0)
    min_water_neighbors: int = _f(1, label="Strait min water neighbors", min=0, max=6)


@paramclass
class CostParams:
    land: LandCostParams = field(default_factory=LandCostParams, metadata={"label": "Land"})
    sea: SeaCostParams = field(default_factory=SeaCostParams, metadata={"label": "Sea"})
    strait: StraitParams = field(default_factory=StraitParams, metadata={"label": "Straits"})


def _face_elevation_range_km(elevation_tex: np.ndarray) -> np.ndarray:
    """Per-face pixel elevation span (km) — the ruggedness proxy."""
    elev = elevation_tex[..., 0]
    return (elev.max(axis=(1, 2)) - elev.min(axis=(1, 2))).astype(np.float32)


def _face_lake_mask(mesh: PolygonMesh, threshold: float = 0.5) -> np.ndarray:
    """Per-face lake flag: at least ``threshold`` of its pixels are lake (a shoreline sliver doesn't count)."""
    n = len(mesh.faces)
    if not LAKE_MASK.present(mesh):
        return np.zeros(n, dtype=bool)
    tex = LAKE_MASK.read(mesh)
    return tex[..., 0].mean(axis=(1, 2)) >= threshold


class CostBuilder(Stage):
    """Builds the base land + sea cost surfaces.

    The full :class:`CostSurfaces` go to ``self.surfaces`` and the ``"cost_surfaces"`` feature; per-face
    metadata carries a displayable summary (``land_base_cost``, ``sea_base_cost``, ``is_strait``,
    ``is_water``, ``is_navigable_river``, ``river_navigability``).
    """

    name = "cost"

    def __init__(self, params: CostParams | None = None) -> None:
        self.params: CostParams = params or CostParams()
        self.surfaces: Optional[CostSurfaces] = None

    def build(self, ctx: BuildContext) -> None:
        mesh = ctx.mesh
        rivers = RIVER_OBJECTS.read(ctx.world)
        _report = ctx.report

        p = self.params

        _report("Reading physical layers for cost…")
        face_biome = BIOME.read(mesh)
        is_ocean = IS_OCEAN.read(mesh)
        is_lake = _face_lake_mask(mesh)
        is_water = is_ocean | is_lake          # sailable, impassable on land
        elev_range = _face_elevation_range_km(ELEVATION.read(mesh))
        temperature_tex = TEMPERATURE.read(mesh) if TEMPERATURE.present(mesh) else None

        _report("Computing land base cost…")
        face_forest_density: np.ndarray | None = None
        if mesh.face_metadata and "forest_density_mean" in mesh.face_metadata[0]:
            face_forest_density = np.array(
                [m.get("forest_density_mean", 0.0) for m in mesh.face_metadata], dtype=np.float32
            )
        land_cost, land_passable = compute_land_base_cost(
            face_biome,
            elev_range,
            is_water,
            biome_cost_scale=p.land.biome_cost_scale,
            ruggedness_cost_per_km=p.land.ruggedness_cost_per_km,
            face_forest_density=face_forest_density,
            forest_cost_strength=p.land.forest_cost_strength,
        )
        if MARSH_FRACTION.present(mesh):
            marsh = np.asarray(MARSH_FRACTION.read(mesh), dtype=np.float32)
            land_cost = (land_cost * (1.0 + p.land.marsh_cost_strength * marsh)).astype(np.float32)

        _report("Computing river flow field…")
        # Mouth segments touch water faces; keep the land river term off them.
        river_flow_dir, river_flow_mag = compute_river_flow_field(
            mesh, rivers, exclude_mask=is_water
        )
        river_navigability = compute_river_navigability(
            river_flow_mag,
            elev_range,
            navigable_ruggedness_km=p.land.navigable_ruggedness_km,
        )
        river_navigable = navigable_river_mask(
            river_flow_mag,
            river_navigability,
            is_water,
            navigable_flow_min=p.sea.navigable_flow_min,
        )

        is_sailable = is_water | river_navigable

        _report("Detecting straits…")
        strait_mask = detect_straits(mesh, is_water, params=p.strait)

        _report("Computing sea distance to coast…")
        sea_dist = compute_sea_distance_to_coast(mesh, is_sailable)

        # Climate winds feature: [(centroids, (N,3)), ...] per season.
        winds_arr: Optional[np.ndarray] = None
        wind_dir: Optional[np.ndarray] = None
        winds_feature = ctx.world.get_feature("winds")
        if winds_feature is not None:
            winds_arr = np.stack([w for _, w in winds_feature], axis=1)  # (N, K, 3)

        _report("Computing sea cost…")
        sea_cost, sea_passable = compute_sea_cost(
            is_sailable,
            sea_dist,
            strait_mask,
            params=p.sea,
            temperature_tex=temperature_tex,
            winds=winds_arr,
        )

        if winds_arr is not None:
            wind_dir = compute_wind_mean_dir(
                winds_arr,
                sea_dist,
                coast_blend_km=p.sea.wind_coast_blend_km,
            )

        surfaces = CostSurfaces(
            land_base_cost=land_cost,
            land_passable=land_passable,
            river_flow_dir=river_flow_dir,
            river_flow_mag=river_flow_mag,
            river_navigability=river_navigability,
            sea_base_cost=sea_cost,
            sea_passable=sea_passable,
            strait_mask=strait_mask,
            wind_dir=wind_dir,
        )
        self.surfaces = surfaces

        mesh.apply_metadata_channel("land_base_cost", land_cost.astype(float).tolist())
        mesh.apply_metadata_channel("sea_base_cost", sea_cost.astype(float).tolist())
        IS_STRAIT.write(mesh, strait_mask)
        IS_WATER.write(mesh, is_water)
        mesh.apply_metadata_channel("is_navigable_river", river_navigable.tolist())
        mesh.apply_metadata_channel(
            "river_navigability", river_navigability.astype(float).tolist()
        )

        COST_SURFACES.write(ctx.world, surfaces)

        _report("Cost surfaces done.")

    # ── Display ──────────────────────────────────────────────────────────────

    def register_display(
        self, world: World, reg: DisplayRegistry, *, debug: bool = False
    ) -> None:
        mesh = world.mesh
        if not (debug and mesh.face_metadata and "land_base_cost" in mesh.face_metadata[0]):
            return

        land_costs = np.array(
            [m.get("land_base_cost", 0.0) for m in mesh.face_metadata], dtype=np.float64
        )
        # Normalize against land faces only (water carries a meaningless cost value).
        is_water_arr = np.array(
            [is_water_meta(m) for m in mesh.face_metadata],
            dtype=bool,
        )
        land_ref = (
            float(np.percentile(land_costs[~is_water_arr], 95)) if (~is_water_arr).any() else 1.0
        )
        land_ref = max(land_ref, 1e-6)

        def _land_cost_color(meta: dict, _ref=land_ref) -> tuple:
            if is_water_meta(meta):
                return OCEAN_COLOR
            c = min(max(float(meta.get("land_base_cost", 0.0)) / _ref, 0.0), 1.0)
            return (0.15 + 0.80 * c, 0.75 - 0.55 * c, 0.20, 1.0)

        reg.add_scheme("Land Cost", _land_cost_color)

        # Sailable faces (water ∪ navigable rivers) carry the sea cost.
        is_sailable_arr = is_water_arr | np.array(
            [bool(m.get("is_navigable_river", False)) for m in mesh.face_metadata], dtype=bool
        )
        sea_costs = np.array(
            [m.get("sea_base_cost", 0.0) for m in mesh.face_metadata], dtype=np.float64
        )
        sea_ref = max(
            float(sea_costs[is_sailable_arr].max()) if is_sailable_arr.any() else 1.0, 1e-6
        )

        def _sea_cost_color(meta: dict, _ref=sea_ref) -> tuple:
            sailable = is_water_meta(meta) or meta.get("is_navigable_river", False)
            if not sailable:
                return LAND_COLOR
            if meta.get("is_strait", False):
                return (1.0, 0.85, 0.1, 1.0)  # highlight chokepoints
            if meta.get("is_navigable_river", False):
                return (0.30, 0.85, 0.95, 1.0)  # navigable rivers stand out
            c = min(max(float(meta.get("sea_base_cost", 0.0)) / _ref, 0.0), 1.0)
            return (0.1, 0.20 + 0.30 * (1.0 - c), 0.50 + 0.40 * c, 1.0)

        reg.add_scheme("Sea Cost", _sea_cost_color)

    # ── Persistence ──────────────────────────────────────────────────────────

    def serialize(self) -> dict:
        cs = self.surfaces
        if cs is None:
            return {}
        out = {
            "cost_land_base_cost": cs.land_base_cost.astype(np.float32),
            "cost_land_passable": cs.land_passable.astype(np.bool_),
            "cost_river_flow_dir": cs.river_flow_dir.astype(np.float32),
            "cost_river_flow_mag": cs.river_flow_mag.astype(np.float32),
            "cost_river_navigability": cs.river_navigability.astype(np.float32),
            "cost_sea_base_cost": cs.sea_base_cost.astype(np.float32),
            "cost_sea_passable": cs.sea_passable.astype(np.bool_),
            "cost_strait_mask": cs.strait_mask.astype(np.bool_),
        }
        if cs.wind_dir is not None:
            out["cost_wind_dir"] = cs.wind_dir.astype(np.float32)
        return out

    def load(self, arrays: Mapping[str, np.ndarray], ctx: BuildContext) -> None:
        if "cost_land_base_cost" not in arrays:
            return
        river_flow_mag = arrays["cost_river_flow_mag"]
        river_navigability = arrays["cost_river_navigability"]
        cs = CostSurfaces(
            land_base_cost=arrays["cost_land_base_cost"],
            land_passable=arrays["cost_land_passable"],
            river_flow_dir=arrays["cost_river_flow_dir"],
            river_flow_mag=river_flow_mag,
            river_navigability=river_navigability,
            sea_base_cost=arrays["cost_sea_base_cost"],
            sea_passable=arrays["cost_sea_passable"],
            strait_mask=arrays["cost_strait_mask"],
            wind_dir=arrays.get("cost_wind_dir"),
        )
        self.surfaces = cs
        COST_SURFACES.write(ctx.world, cs)
