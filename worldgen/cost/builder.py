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

from typing import Callable, Optional

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
from core.mesh import PolygonMesh
from utils.params import field, param_field as _f, paramclass


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


@paramclass
class SeaCostParams:
    coastal_cost: float = _f(0.5, label="Coastal water cost", min=0.01, max=5.0)
    open_ocean_cost: float = _f(2.0, label="Open ocean cost", min=0.1, max=10.0)
    open_ocean_distance_km: float = _f(2000.0, label="Open ocean distance (km)", min=100.0, max=10000.0)
    strait_discount: float = _f(0.5, label="Strait discount", min=0.05, max=1.0)
    navigable_flow_min: float = _f(
        500.0, label="Navigable river flow min", min=1.0, max=20000.0
    )


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
    tex = mesh.face_textures.get("lake_mask")
    n = len(mesh.faces)
    if tex is None:
        return np.zeros(n, dtype=bool)
    return tex[..., 0].mean(axis=(1, 2)) >= threshold


class CostBuilder:
    """Builds the base land + sea cost surfaces from a fully-built mesh.

    Call ``build(mesh, rivers=...)`` after elevation + climate + hydrology so the
    ``biome`` / ``is_ocean`` metadata and the ``elevation`` / ``lake_mask``
    textures are present. Stores ``self.surfaces: CostSurfaces`` and writes
    ``land_base_cost`` / ``sea_base_cost`` / ``is_strait`` / ``is_water`` /
    ``is_navigable_river`` / ``river_navigability`` per-face metadata (the
    displayable, serializable summary; the full surfaces — including the river
    flow field — live in ``self.surfaces`` and the ``"cost_surfaces"`` globe
    feature).
    """

    def __init__(self, params: CostParams | None = None) -> None:
        self.params: CostParams = params or CostParams()
        self.surfaces: Optional[CostSurfaces] = None

    def build(
        self,
        mesh: PolygonMesh,
        rivers=None,
        progress: Optional[Callable[[str], None]] = None,
    ) -> CostSurfaces:
        def _report(msg: str) -> None:
            if progress is not None:
                progress(msg)

        p = self.params
        n = len(mesh.faces)

        _report("Reading physical layers for cost…")
        face_biome = np.array(
            [int(mesh.face_metadata[i].get("biome", 0)) for i in range(n)],
            dtype=np.int32,
        )
        is_ocean = np.array(
            [bool(mesh.face_metadata[i].get("is_ocean", False)) for i in range(n)],
            dtype=bool,
        )
        is_lake = _face_lake_mask(mesh)
        is_water = is_ocean | is_lake          # sailable, impassable on land
        elev_range = _face_elevation_range_km(mesh.face_textures["elevation"])

        _report("Computing land base cost…")
        land_cost, land_passable = compute_land_base_cost(
            face_biome,
            elev_range,
            is_water,
            biome_cost_scale=p.land.biome_cost_scale,
            ruggedness_cost_per_km=p.land.ruggedness_cost_per_km,
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
        strait_mask = detect_straits(
            mesh,
            is_water,
            opposite_dot_threshold=p.strait.opposite_dot_threshold,
            min_water_neighbors=p.strait.min_water_neighbors,
        )

        _report("Computing sea distance to coast…")
        sea_dist = compute_sea_distance_to_coast(mesh, is_sailable)

        _report("Computing sea cost…")
        sea_cost, sea_passable = compute_sea_cost(
            is_sailable,
            sea_dist,
            strait_mask,
            coastal_cost=p.sea.coastal_cost,
            open_ocean_cost=p.sea.open_ocean_cost,
            open_ocean_distance_km=p.sea.open_ocean_distance_km,
            strait_discount=p.sea.strait_discount,
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
        mesh.apply_metadata_channel("is_strait", strait_mask.tolist())
        mesh.apply_metadata_channel("is_water", is_water.tolist())
        mesh.apply_metadata_channel("is_navigable_river", river_navigable.tolist())
        mesh.apply_metadata_channel(
            "river_navigability", river_navigability.astype(float).tolist()
        )

        _report("Cost surfaces done.")
        return surfaces
