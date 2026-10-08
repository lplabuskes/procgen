"""Stage 6 — texel-level refinement of realm geometry.

Purely geometric; no population/polity recomputation. Mutates world features:

1. **Settlements**: sited on the best texel of their face, at least
   ``city_min_spacing_km`` from a bigger city.
2. **Trade routes**: endpoints remapped to moved settlements.
3. **Polity borders**: smoothed polylines plus the texel "Polities" fill texture.
"""
from __future__ import annotations

import numpy as np

from core.params import param_field as _f, paramclass
from core.pipeline import BuildContext, Stage
from realmgen.channels import AUTHORITY, POLITY_ID
from realmgen.persistence import pack_borders, pack_routes, unpack_borders, unpack_routes
from realmgen.refine.borders import plain_border_chains, smooth_polity_borders
from realmgen.refine.fill import PolityLabelField, colorize_labels
from realmgen.refine.river_follow import RiverFollowParams, build_river_swaps
from realmgen.refine.settlements import refine_settlement_positions, remap_routes
from realmgen.refine.sites import SiteField
from worldgen.cost.surfaces import EARTH_RADIUS_KM
from worldgen.cost.channels import IS_WATER
from worldgen.geology.channels import OCEAN_MASK
from worldgen.hydrology.channels import LAKE_MASK
from worldgen.resources.channels import FERTILITY_MEAN


@paramclass
class RefineParams:
    river_threshold_factor: float = _f(
        1.5, label="River snap distance (× texel spacing)", min=0.5, max=5.0
    )
    city_min_spacing_km: float = _f(
        120.0, label="Minimum distance between cities (km)", min=0.0, max=300.0, step=10.0
    )
    border_smoothing_iterations: int = _f(
        2, label="Border smoothing (Chaikin passes)", min=0, max=5, step=1
    )
    border_lattice_offset: int = _f(
        0, label="Border lattice depth offset (vs texel-fine)", min=-3, max=1, step=1, hidden=True
    )
    border_control_floor: float = _f(
        1e-3, label="Border control floor", min=1e-6, max=1.0, hidden=True
    )
    boundary_warp_amplitude: float = _f(
        0.012, label="Boundary erosion strength", min=0.0, max=0.05, step=0.001
    )
    boundary_warp_frequency: float = _f(
        45.0, label="Boundary erosion frequency", min=5.0, max=200.0, step=5.0
    )
    boundary_warp_octaves: int = _f(
        3, label="Boundary erosion octaves", min=1, max=5, step=1, hidden=True
    )
    border_river_reach_km: float = _f(
        120.0, label="Border river follow reach (km, 0=off)", min=0.0, max=300.0, step=10.0
    )
    border_river_min_flow: float = _f(
        300.0, label="Border river: min flow", min=0.0, max=50000.0, step=100.0
    )
    border_river_full_flow: float = _f(
        4000.0, label="Border river: flow at full reach", min=1.0, max=100000.0, step=500.0
    )
    border_river_min_run_km: float = _f(
        40.0, label="Border river: shortest followed stretch (km)", min=5.0, max=300.0, step=5.0
    )


class RefineStage(Stage):
    """Texel-level refinement of settlements, trade routes, and polity borders."""

    name = "refine"
    requires = (IS_WATER, OCEAN_MASK, FERTILITY_MEAN)

    def __init__(self, params: RefineParams | None = None) -> None:
        self.params = params or RefineParams()
        # Final routes/borders, cached for serialization.
        self._routes: list | None = None
        self._borders: dict | None = None

    def build(self, ctx: BuildContext) -> None:
        world = ctx.world
        mesh = world.mesh
        p = self.params

        ctx.report("Refining settlement positions and borders…")
        settlements = world.get_feature("settlements") or []
        if settlements and OCEAN_MASK.present(mesh):
            ocean_tex = OCEAN_MASK.read(mesh)
            lake_tex = LAKE_MASK.read(mesh) if LAKE_MASK.present(mesh) else None
            sites = SiteField(
                mesh, ocean_tex, lake_tex, mesh.face_textures.get("fertility"),
                world.get_feature("rivers"),
                river_threshold_factor=float(p.river_threshold_factor),
            )
            moved = refine_settlement_positions(
                settlements,
                sites=sites,
                min_spacing=float(p.city_min_spacing_km) / EARTH_RADIUS_KM,
                seed=int(world.seed or 0),
            )

            routes = world.get_feature("trade_routes")
            if routes and moved:
                world.add_feature("trade_routes", remap_routes(routes, moved))

            field = self._label_field(world)
            polities = world.get_feature("polities") or []
            if world.has_feature("polity_borders") and field is not None:
                # One label field feeds both fill and borders, so the lines bound the fill.
                H, W = int(ocean_tex.shape[1]), int(ocean_tex.shape[2])
                labels = field.texel_labels(
                    mesh.face_texture_positions(H, W), ocean_tex, lake_tex)
                world.add_feature("polity_borders", smooth_polity_borders(
                    mesh, polities, field, ocean_tex, lake_tex,
                    iterations=int(p.border_smoothing_iterations),
                    lattice_offset=int(p.border_lattice_offset),
                ))
                rgba = colorize_labels(labels, polities, ocean_tex, lake_tex)
                mesh.set_face_textures("polity_fill", rgba.astype(np.float16), n_channels=4)

            ctx.report(f"  ✦ Refined {len(moved)} settlement position(s) to texel level.")

        # Captured even if refinement didn't run (raw routes/borders then).
        self._routes = world.get_feature("trade_routes")
        self._borders = world.get_feature("polity_borders")

    def _label_field(self, world) -> PolityLabelField | None:
        """Build and store the ``polity_label_field`` feature; None before polities."""
        mesh = world.mesh
        polities = world.get_feature("polities") or []
        if not (polities and POLITY_ID.present(mesh) and AUTHORITY.present(mesh)
                and IS_WATER.present(mesh)):
            return None
        p = self.params
        field = PolityLabelField(
            mesh,
            np.asarray(POLITY_ID.read(mesh), dtype=np.int32),
            np.asarray(IS_WATER.read(mesh), dtype=bool),
            np.asarray(AUTHORITY.read(mesh), dtype=np.float64),
            control_floor=float(p.border_control_floor),
            warp_amplitude=float(p.boundary_warp_amplitude),
            warp_frequency=float(p.boundary_warp_frequency),
            warp_octaves=int(p.boundary_warp_octaves),
            warp_seed=int(world.seed or 0),
        )
        if p.border_river_reach_km > 0.0 and OCEAN_MASK.present(mesh):
            field = field.with_swaps(self._river_swaps(world, field, polities))
        world.add_feature("polity_label_field", field)
        return field

    def _river_swaps(self, world, field: PolityLabelField, polities):
        """Border runs moved onto big rivers (``river_follow``), from the field's plain borders."""
        mesh, p = world.mesh, self.params
        ocean_tex = OCEAN_MASK.read(mesh)
        lake_tex = LAKE_MASK.read(mesh) if LAKE_MASK.present(mesh) else None
        chains = plain_border_chains(mesh, polities, field, ocean_tex, lake_tex,
                                     lattice_offset=int(p.border_lattice_offset))
        settlements = world.get_feature("settlements") or []
        positions = (np.array([np.asarray(s.site_position, dtype=np.float64) for s in settlements])
                     if settlements else np.zeros((0, 3)))
        follow = RiverFollowParams(reach_km=float(p.border_river_reach_km),
                                   min_flow=float(p.border_river_min_flow),
                                   full_flow=float(p.border_river_full_flow),
                                   min_run_km=float(p.border_river_min_run_km))
        return build_river_swaps(chains, world.get_feature("rivers"), field.plain_land_labels,
                                 field.is_water_at, positions, field.centroids, follow,
                                 labels_without=field.labels_without)

    # ── Persistence ───────────────────────────────────────────────────────────
    # Owns the route/border round-trip; polity_fill persists via mesh textures.

    def serialize(self) -> dict:
        arrays: dict = {}
        if self._routes:
            arrays.update(pack_routes(self._routes))
        if self._borders:
            arrays.update(pack_borders(self._borders))
        return arrays

    def load(self, arrays, ctx: BuildContext) -> None:
        world = ctx.world
        if "route_flow" in arrays:
            self._routes = unpack_routes(arrays)
            world.add_feature("trade_routes", self._routes)
        if "border_segs" in arrays:
            self._borders = unpack_borders(arrays)
            world.add_feature("polity_borders", self._borders)
        # Not serialized: rebuilt from the persisted polity channels.
        self._label_field(world)
