"""Stage 6 — texel-level refinement of realm geometry.

Runs last in :class:`~realmgen.builder.RealmPipeline`.  Purely a geometric
refinement pass over render outputs (no population/polity recomputation):

1. **Settlements** — snap each settlement from its face centroid to the best
   in-face texel (coast → river → fertility), mutating ``Settlement.position``.
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
from realmgen.refine.borders import smooth_polity_borders
from realmgen.refine.fill import build_polity_fill_texture, compute_texel_labels
from realmgen.refine.settlements import refine_settlement_positions, remap_routes
from worldgen.cost.channels import IS_WATER
from worldgen.geology.channels import OCEAN_MASK
from worldgen.hydrology.channels import LAKE_MASK
from worldgen.resources.channels import FERTILITY_MEAN


@paramclass
class RefineParams:
    river_threshold_factor: float = _f(
        1.5, label="River snap distance (× texel spacing)", min=0.5, max=5.0
    )
    border_smoothing_iterations: int = _f(
        2, label="Border smoothing (Chaikin passes)", min=0, max=5, step=1
    )
    border_control_floor: float = _f(
        1e-3, label="Border control floor", min=1e-6, max=1.0, hidden=True
    )


class RefineStage(Stage):
    """Texel-level refinement of settlements, trade routes, and polity borders."""

    name = "refine"
    requires = (IS_WATER, OCEAN_MASK, FERTILITY_MEAN)

    def __init__(self, params: RefineParams | None = None) -> None:
        self.params = params or RefineParams()

    def build(self, ctx: BuildContext) -> None:
        world = ctx.world
        mesh = world.mesh
        p = self.params

        settlements = world.get_feature("settlements") or []
        if not settlements or not OCEAN_MASK.present(mesh):
            return

        ocean_tex = OCEAN_MASK.read(mesh)
        lake_tex = LAKE_MASK.read(mesh) if LAKE_MASK.present(mesh) else None
        fertility = FERTILITY_MEAN.read(mesh)
        rivers = world.get_feature("rivers")

        moved = refine_settlement_positions(
            settlements,
            mesh=mesh,
            ocean_tex=ocean_tex,
            lake_tex=lake_tex,
            fertility=fertility,
            rivers=rivers,
            river_threshold_factor=float(p.river_threshold_factor),
        )

        routes = world.get_feature("trade_routes")
        if routes and moved:
            world.add_feature("trade_routes", remap_routes(routes, moved))

        if world.has_feature("polity_borders") and POLITY_ID.present(mesh):
            polities = world.get_feature("polities") or []
            pof = np.asarray(POLITY_ID.read(mesh), dtype=np.int32)
            is_water = np.asarray(IS_WATER.read(mesh), dtype=bool)
            authority = (np.asarray(AUTHORITY.read(mesh), dtype=np.float64)
                         if AUTHORITY.present(mesh) else None)
            if polities and authority is not None:
                # Single texel classification feeds both the fill and the borders,
                # so the lines fall exactly on the fill's region edges.
                labels = compute_texel_labels(
                    mesh, pof, is_water, authority, ocean_tex, lake_tex,
                    control_floor=float(p.border_control_floor),
                )
                world.add_feature("polity_borders", smooth_polity_borders(
                    mesh, polities, labels,
                    iterations=int(p.border_smoothing_iterations),
                ))
                rgba = build_polity_fill_texture(
                    mesh, polities, pof, is_water,
                    authority, ocean_tex, lake_tex,
                    control_floor=float(p.border_control_floor),
                    labels=labels,
                )
                mesh.set_face_textures("polity_fill", rgba, n_channels=4)

        ctx.report(f"  ✦ Refined {len(moved)} settlement position(s) to texel level.")
