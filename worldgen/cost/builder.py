"""Cost-surface stage — the final worldgen step producing base traversal costs.

Reads the `biome` and `is_ocean` per-face metadata and the `elevation` +
`lake_mask` face textures, plus the hydrology rivers, and produces the physical
base land + sea cost layers. See `worldgen/cost/surfaces.py` for the model and
docs/settlement_generation_design.md (Stage 0) for the rationale: these layers
are consumed by realmgen, which adds the danger field on top.

Lakes count as sailable water (impassable on land like the ocean), and navigable
rivers — flat enough and carrying enough flow, up to the fall line — extend the
sailable sea network inland so a river can connect to the sea directly.
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
from worldgen.geology.channels import ELEVATION, IS_OCEAN
from worldgen.geology.display import LAND_COLOR, OCEAN_COLOR
from worldgen.hydrology.channels import LAKE_MASK, RIVER_OBJECTS


@paramclass
class LandCostParams:
    biome_cost_scale: float = _f(1.0, label="Biome cost scale", min=0.0, max=5.0)
    ruggedness_cost_per_km: float = _f(1.5, label="Ruggedness cost/km", min=0.0, max=10.0)
    water_crossing_cost: float = _f(20.0, label="Water crossing cost", min=0.0, max=100.0)
    river_ford_strength: float = _f(2.0, label="River ford penalty", min=0.0, max=10.0)
    river_corridor_strength: float = _f(0.5, label="River corridor discount", min=0.0, max=1.0)
    river_flow_reference: float = _f(2000.0, label="River flow reference", min=1.0, max=20000.0)
    river_min_mult: float = _f(0.4, label="River min multiplier", min=0.05, max=1.0)
    river_max_mult: float = _f(5.0, label="River max multiplier", min=1.0, max=20.0)
    navigable_ruggedness_km: float = _f(
        0.6, label="Navigable ruggedness (km)", min=0.05, max=5.0
    )


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
    """Per-face lake flag — True where the face is mostly lake.

    Reads the hydrology ``lake_mask`` texture (1 inside the lake footprint). A
    face counts as water when at least ``threshold`` of its pixels are lake, so a
    sliver of shoreline lake does not turn a land face impassable. Returns all
    False when no lake texture is present.
    """
    n = len(mesh.faces)
    if not LAKE_MASK.present(mesh):
        return np.zeros(n, dtype=bool)
    tex = LAKE_MASK.read(mesh)
    return tex[..., 0].mean(axis=(1, 2)) >= threshold


class CostBuilder(Stage):
    """Builds the base land + sea cost surfaces (a pipeline ``Stage``).

    ``build(ctx)`` runs after elevation + climate + hydrology so the
    ``biome`` / ``is_ocean`` metadata and the ``elevation`` / ``lake_mask``
    textures are present; it reads the rivers from the ``"river_objects"`` world
    feature. Stores ``self.surfaces: CostSurfaces`` and writes
    ``land_base_cost`` / ``sea_base_cost`` / ``is_strait`` / ``is_water`` /
    ``is_navigable_river`` / ``river_navigability`` per-face metadata (the
    displayable, serializable summary; the full surfaces — including the river
    flow field — live in ``self.surfaces`` and the ``"cost_surfaces"`` globe
    feature).
    """

    name = "cost"

    def __init__(self, params: CostParams | None = None) -> None:
        self.params: CostParams = params or CostParams()
        self.surfaces: Optional[CostSurfaces] = None

    def build(self, ctx: BuildContext) -> CostSurfaces:
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
        land_cost, land_passable = compute_land_base_cost(
            face_biome,
            elev_range,
            is_water,
            biome_cost_scale=p.land.biome_cost_scale,
            ruggedness_cost_per_km=p.land.ruggedness_cost_per_km,
            water_crossing_cost=p.land.water_crossing_cost,
        )

        _report("Computing river flow field…")
        # A river's mouth segment touches a water face; exclude water faces so the
        # land river term never lands on impassable water (in-lake stretches are
        # already skipped inside compute_river_flow_field).
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

        # Sailable network = open water plus navigable rivers (which stay
        # land-passable too — the one overlap between the two surfaces).
        is_sailable = is_water | river_navigable

        _report("Detecting straits…")
        strait_mask = detect_straits(mesh, is_water, params=p.strait)

        _report("Computing sea distance to coast…")
        sea_dist = compute_sea_distance_to_coast(mesh, is_sailable)

        _report("Computing sea cost…")
        sea_cost, sea_passable = compute_sea_cost(
            is_sailable,
            sea_dist,
            strait_mask,
            params=p.sea,
            temperature_tex=temperature_tex,
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
        return surfaces

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
        # Normalize against land faces only (water crossing cost would skew the scale).
        # Fall back to is_ocean for worlds saved before the cost stage wrote is_water.
        is_water_arr = np.array(
            [bool(m.get("is_water", m.get("is_ocean", False))) for m in mesh.face_metadata],
            dtype=bool,
        )
        land_ref = (
            float(np.percentile(land_costs[~is_water_arr], 95)) if (~is_water_arr).any() else 1.0
        )
        land_ref = max(land_ref, 1e-6)

        def _land_cost_color(meta: dict, _ref=land_ref) -> tuple:
            if meta.get("is_water", meta.get("is_ocean", False)):
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
            sailable = meta.get("is_water", meta.get("is_ocean", False)) or meta.get(
                "is_navigable_river", False
            )
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
        return {
            "cost_land_base_cost": cs.land_base_cost.astype(np.float32),
            "cost_land_passable": cs.land_passable.astype(np.bool_),
            "cost_river_flow_dir": cs.river_flow_dir.astype(np.float32),
            "cost_river_flow_mag": cs.river_flow_mag.astype(np.float32),
            "cost_river_navigability": cs.river_navigability.astype(np.float32),
            "cost_sea_base_cost": cs.sea_base_cost.astype(np.float32),
            "cost_sea_passable": cs.sea_passable.astype(np.bool_),
            "cost_strait_mask": cs.strait_mask.astype(np.bool_),
        }

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
        )
        self.surfaces = cs
        COST_SURFACES.write(ctx.world, cs)
