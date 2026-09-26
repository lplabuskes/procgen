"""Stage 6 — texel-level refinement of realm geometry.

Runs last in :class:`~realmgen.builder.RealmPipeline`.  Purely a geometric
refinement pass over render outputs (no population/polity recomputation):

1. **Settlements** — site each settlement on the best texel of its face (harbour,
   confluence, river, fertility), no closer than ``city_min_spacing_km`` to a
   bigger city, mutating ``Settlement.position``.
2. **Trade routes** — remap any route endpoint that coincided with a moved
   settlement to its refined position (same connections, adjusted geometry).
3. **Polity borders** — replace the angular per-edge border soup with continuous,
   Chaikin-smoothed polylines, and build a texel-resolution, authority-weighted
   polity *fill* texture (attached to the "Polities" map mode).

All outputs are mutated in place on the existing world features.
"""
from __future__ import annotations

import numpy as np

from core.params import param_field as _f, paramclass
from core.pipeline import BuildContext, Stage
from realmgen.channels import AUTHORITY, POLITY_ID
from realmgen.persistence import pack_borders, pack_routes, unpack_borders, unpack_routes
from realmgen.refine.borders import smooth_polity_borders
from realmgen.refine.fill import PolityLabelField, colorize_labels
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


class RefineStage(Stage):
    """Texel-level refinement of settlements, trade routes, and polity borders."""

    name = "refine"
    requires = (IS_WATER, OCEAN_MASK, FERTILITY_MEAN)

    def __init__(self, params: RefineParams | None = None) -> None:
        self.params = params or RefineParams()
        # Final realm geometry, cached for serialization (refine is the last stage,
        # so these are the post-refine route/border features).
        self._routes: list | None = None
        self._borders: dict | None = None

    def build(self, ctx: BuildContext) -> None:
        world = ctx.world
        mesh = world.mesh
        p = self.params

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
                # One label field — the polity at each point of the sphere — feeds
                # both the fill (sampled at every texel) and the borders (contoured
                # on a seamless lattice), so the lines bound the fill.
                H, W = int(ocean_tex.shape[1]), int(ocean_tex.shape[2])
                labels = field.texel_labels(
                    mesh.face_texture_positions(H, W), ocean_tex, lake_tex)
                world.add_feature("polity_borders", smooth_polity_borders(
                    mesh, polities, field, ocean_tex, lake_tex,
                    iterations=int(p.border_smoothing_iterations),
                ))
                rgba = colorize_labels(labels, polities, ocean_tex, lake_tex)
                mesh.set_face_textures("polity_fill", rgba.astype(np.float16), n_channels=4)

            ctx.report(f"  ✦ Refined {len(moved)} settlement position(s) to texel level.")

        # Capture the final realm geometry for serialization regardless of whether
        # refinement ran (these may be the raw routes/borders from earlier stages).
        self._routes = world.get_feature("trade_routes")
        self._borders = world.get_feature("polity_borders")

    def _label_field(self, world) -> PolityLabelField | None:
        """The polity label field, also kept as the ``polity_label_field`` feature
        so the display can redraw the fill at any resolution; None before polities."""
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
        world.add_feature("polity_label_field", field)
        return field

    # ── Persistence ───────────────────────────────────────────────────────────
    # Trade-route and polity-border geometry reach their final form here (routes
    # remapped, borders smoothed), so this stage owns their round-trip. The
    # polity_fill RGBA texture persists generically via mesh textures.

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
