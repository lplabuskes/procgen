"""Delta stage — the land large rivers build at their mouths.

Runs right after hydrology (which finds the rivers) and before the climate's
lake-feedback pass (which re-derives precipitation and biomes from the edited
relief, so the new land gets its own).  For each big river reaching the sea it
deposits a delta (:mod:`worldgen.delta.deposit`): a low plain over the river's
lower course and built out into the shallow sea, with the rivers carried across
it to the new front.

Writes back the ``elevation``, ``ocean_mask`` and ``filled_height`` textures,
the per-face ``is_ocean`` / ``elevation`` aggregates, the extended river objects,
river polylines and stream network; adds the ``delta`` texture, the per-face
``delta_fraction`` / ``marsh_fraction``, and the ``"delta_fans"`` and
``"deltas"`` (new mouths) features.  Resources (fertility), danger and cost
(marsh) and realm scoring (the apex as a city site) read them.
"""
from __future__ import annotations

from typing import List, Mapping, Optional

import numpy as np

from core.display import DisplayRegistry
from core.params import param_field as _f, paramclass
from core.pipeline import BuildContext, Stage
from core.world import World
from worldgen.delta.channels import DELTA, DELTA_FANS, DELTA_FRACTION, DELTA_MOUTHS, MARSH_FRACTION
from worldgen.delta.deposit import DeltaFan, build_deltas
from worldgen.geology.channels import ELEVATION, ELEVATION_FACE, IS_OCEAN, OCEAN_MASK
from worldgen.hydrology.channels import LAKE_MASK, RIVER_OBJECTS, RIVERS, STREAMS
from worldgen.hydrology.rivers import River


@paramclass
class DeltaParams:
    # Which rivers get a delta: mouth flow and length, reaching the sea.
    min_flow: float = _f(1000.0, label="Min mouth flow", min=0.0, max=10000.0)
    min_length_km: float = _f(500.0, label="Min river length (km)", min=0.0, max=5000.0)
    # Size: area_km2 · (flow / 1000) ** area_exponent.
    area_km2: float = _f(1500.0, label="Area at flow 1000 (km²)", min=0.0, max=20000.0, step=100.0)
    area_exponent: float = _f(0.7, label="Area exponent", min=0.0, max=1.5, step=0.05)
    # Shape.  The apex (head of the delta) lies this far up the river from the
    # mouth, and the flood looks this far past the mouth, both × √area.
    apex_distance: float = _f(0.5, label="Apex distance (× √area)", min=0.0, max=3.0, step=0.05)
    reach: float = _f(2.5, label="Reach (× √area)", min=0.5, max=6.0, step=0.1)
    # Priority multipliers: off the axis (1 − cos), on land, per depth.
    lateral_cost: float = _f(0.6, label="Lateral cost", min=0.0, max=5.0, step=0.05)
    land_cost: float = _f(3.0, label="Land cost", min=0.1, max=5.0, step=0.05)
    depth_scale_km: float = _f(0.1, label="Depth scale (km)", min=0.01, max=1.0, step=0.01)
    # Never taken: hills and deep water.
    max_land_elevation_km: float = _f(0.12, label="Max land elevation (km)", min=0.0, max=1.0, step=0.01)
    max_depth_km: float = _f(0.4, label="Max sea depth (km)", min=0.0, max=3.0, step=0.05)
    # The plain's surface, apex to front, and the shoaled sea in front.
    apex_height_km: float = _f(0.02, label="Apex height (km)", min=0.0, max=0.2, step=0.005)
    front_height_km: float = _f(0.002, label="Front height (km)", min=0.0, max=0.05, step=0.001)
    skirt_rings: int = _f(2, label="Prodelta rings (texels)", min=0, max=6)
    skirt_depth_km: float = _f(0.03, label="Prodelta depth (km)", min=0.0, max=0.5, step=0.01)
    # Marsh: from this fraction of the way from apex to front, rising to 1 at the front.
    marsh_start: float = _f(0.6, label="Marsh start (apex→front)", min=0.0, max=1.0, step=0.05)


class DeltaBuilder(Stage):
    """Deposits river deltas onto the global relief — a pipeline ``Stage``."""

    name = "delta"

    def __init__(self, params: DeltaParams | None = None) -> None:
        self.params: DeltaParams = params or DeltaParams()
        self.fans: List[DeltaFan] = []

    def build(self, ctx: BuildContext) -> None:
        mesh = ctx.mesh
        world = ctx.world
        self.fans = []
        if not RIVER_OBJECTS.present(world):
            return
        ctx.report("Depositing river deltas…")
        rivers: List[River] = RIVER_OBJECTS.read(world)
        elevation = np.array(ELEVATION.read(mesh), dtype=np.float32)
        ocean = np.array(OCEAN_MASK.read(mesh))
        lake = LAKE_MASK.read(mesh) if LAKE_MASK.present(mesh) else None
        filled = (np.array(mesh.face_textures["filled_height"], dtype=np.float32)
                  if "filled_height" in mesh.face_textures else None)
        streams = STREAMS.read(world) if STREAMS.present(world) else None

        result = build_deltas(mesh, rivers, elevation, ocean, lake, filled, streams, self.params)
        self.fans = result.fans

        ELEVATION.write(mesh, elevation)
        OCEAN_MASK.write(mesh, ocean)
        if filled is not None:
            mesh.set_face_textures("filled_height", filled.astype(np.float16), n_channels=1)
        IS_OCEAN.write(mesh, ocean[..., 0].mean(axis=(1, 2)) > 0.5)
        from worldgen.geology.elevation import face_elevations_from_texture
        ELEVATION_FACE.write(mesh, face_elevations_from_texture(elevation))

        DELTA.write(mesh, result.delta)
        DELTA_FRACTION.write(mesh, result.plain_fraction)
        MARSH_FRACTION.write(mesh, result.marsh_fraction)

        # The rivers were extended in place; refresh their flattened polylines.
        RIVERS.write(world, _flatten(rivers))
        self._publish(world)

    def _publish(self, world: World) -> None:
        DELTA_FANS.write(world, self.fans)
        if self.fans:
            DELTA_MOUTHS.write(world, np.stack([f.mouth for f in self.fans]).astype(np.float32))

    # ── Display ──────────────────────────────────────────────────────────────

    def register_display(self, world: World, reg: DisplayRegistry, *, debug: bool = False) -> None:
        if not debug:
            return
        mesh = world.mesh
        if world.has_feature("deltas"):
            reg.add_overlay("deltas", "River Deltas", checked=False)
        if "delta" not in mesh.face_textures:
            return
        from worldgen.delta.display import DELTA_RECIPE, delta_color, delta_legend_entries

        mesh.set_face_textures("delta_rgba", DELTA_RECIPE(mesh.face_textures))
        reg.add_scheme(
            "Delta", delta_color, texture="delta_rgba",
            legend=("gradient", delta_legend_entries()), recipe=DELTA_RECIPE,
        )

        def _delta_hover(face: int, _uv, _w) -> list:
            meta = mesh.face_metadata[face]
            plain = float(meta.get("delta_fraction", 0.0))
            if plain <= 0.0:
                return []
            return [f"Delta plain: {plain:.0%} (marsh {float(meta.get('marsh_fraction', 0.0)):.0%})"]

        reg.add_hover(_delta_hover)

    # ── Persistence ──────────────────────────────────────────────────────────
    # The textures and per-face fractions are saved with the mesh; the edited
    # rivers and streams with the hydrology stage.

    def serialize(self) -> dict:
        if not self.fans:
            return {}
        return {
            "delta_apex": np.stack([f.apex for f in self.fans]).astype(np.float32),
            "delta_mouth": np.stack([f.mouth for f in self.fans]).astype(np.float32),
            "delta_axis": np.stack([f.axis for f in self.fans]).astype(np.float32),
            "delta_scalars": np.array(
                [[f.radius_km, f.flow, f.area_km2] for f in self.fans], dtype=np.float64
            ),
        }

    def load(self, arrays: Mapping[str, np.ndarray], ctx: BuildContext) -> None:
        self.fans = []
        if "delta_apex" in arrays:
            for a, m, x, (r, q, ar) in zip(arrays["delta_apex"], arrays["delta_mouth"], arrays["delta_axis"],
                                           arrays["delta_scalars"]):
                self.fans.append(DeltaFan(apex=np.asarray(a), mouth=np.asarray(m), axis=np.asarray(x),
                                          radius_km=float(r), flow=float(q), area_km2=float(ar)))
        self._publish(ctx.world)


def _flatten(rivers: Optional[List[River]]) -> list:
    """``(positions, flows, in_lake)`` polyline tuples, as the hydrology stage registers them."""
    paths = []
    for river in rivers or []:
        polyline = river.as_polyline()
        if len(polyline) < 2:
            continue
        paths.append((
            np.stack(polyline).astype(np.float32),
            np.asarray(river.flow_rates(), dtype=np.float32),
            np.array([s.in_lake for s in river.segments], dtype=bool),
        ))
    return paths
