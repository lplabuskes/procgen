"""Delta stage — the land large rivers build at their mouths.

Runs right after hydrology (which finds the rivers) and before the climate's
lake-feedback pass (which re-derives precipitation and biomes from the edited
relief, so the new land gets its own).  For each big river reaching the sea it
deposits a delta (:mod:`worldgen.delta.deposit`): a low plain over the river's
lower course and built out into the shallow sea, with the rivers carried across
it to the new front.  Big rivers entering a lake build one there too.

Writes back the ``elevation``, ``ocean_mask``, ``lake_mask`` and ``filled_height`` textures,
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
from worldgen.delta.shape import ARCUATE, BIRDFOOT, CUSPATE
from worldgen.geology.channels import ELEVATION, ELEVATION_FACE, IS_OCEAN, OCEAN_MASK
from worldgen.hydrology.channels import LAKE_MASK, RIVER_OBJECTS, RIVERS, STREAMS
from worldgen.hydrology.rivers import River


@paramclass
class DeltaParams:
    # Which rivers get a delta: mouth flow and length, reaching the sea or a lake.
    min_flow: float = _f(1000.0, label="Min mouth flow", min=0.0, max=10000.0)
    min_length_km: float = _f(500.0, label="Min river length (km)", min=0.0, max=5000.0)
    # Size: area_km2 · (flow / 1000) ** area_exponent.
    area_km2: float = _f(1500.0, label="Area at flow 1000 (km²)", min=0.0, max=20000.0, step=100.0)
    area_exponent: float = _f(0.7, label="Area exponent", min=0.0, max=1.5, step=0.05)
    # The share of the sediment reaching a lake that settles there (the outflow
    # carries the rest on); a river's supply is its flow × the share still in it.
    lake_trap_efficiency: float = _f(0.2, label="Lake sediment trapping", min=0.0, max=1.0, step=0.05)
    # The share of a lake's area its deltas may fill, all together.
    max_lake_share: float = _f(0.25, label="Max lake share filled", min=0.0, max=1.0, step=0.05)
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

    # ── Shape: river power against wave power (see worldgen/delta/shape.py) ──
    # Wave power: onshore wind³ + swell³, × fetch over open sea (capped).
    fetch_cap_km: float = _f(1500.0, label="Fetch cap (km)", min=100.0, max=5000.0, step=100.0)
    swell_speed: float = _f(3.0, label="Swell (m/s equivalent)", min=0.0, max=20.0, step=0.5)
    # River power: river_weight · (flow / 1000) ** river_exponent.
    river_weight: float = _f(0.08, label="River weight", min=0.0, max=10.0, step=0.05)
    river_exponent: float = _f(0.5, label="River exponent", min=0.0, max=1.5, step=0.05)
    # Wave dominance below / above which a delta is bird's-foot / cuspate, blended over this width.
    birdfoot_below: float = _f(0.3, label="Bird's-foot below (wave dominance)", min=0.0, max=1.0, step=0.05)
    cuspate_above: float = _f(0.7, label="Cuspate above (wave dominance)", min=0.0, max=1.0, step=0.05)
    shape_blend: float = _f(0.1, label="Shape blend width", min=0.0, max=0.5, step=0.05)
    # Bird's-foot fingers: how many (at flow 1000, more per tenfold flow), fanned
    # how wide, how long (× √area, the longest; the rest up to ``finger_jitter``
    # shorter), how wide (texels either side) and how dear it is to leave them.
    finger_base: float = _f(2.0, label="Fingers at flow 1000", min=2.0, max=8.0, step=1.0)
    finger_per_decade: float = _f(1.0, label="More fingers per 10× flow", min=0.0, max=4.0, step=0.5)
    finger_max: int = _f(5, label="Max fingers", min=2, max=8)
    finger_spread_deg: float = _f(45.0, label="Finger spread (°)", min=10.0, max=85.0, step=5.0)
    finger_length: float = _f(1.4, label="Finger length (× √area)", min=0.1, max=3.0, step=0.05)
    finger_jitter: float = _f(0.5, label="Finger length jitter", min=0.0, max=0.9, step=0.05)
    finger_half_width: float = _f(0.4, label="Finger half-width (texels)", min=0.0, max=3.0, step=0.1)
    finger_cost: float = _f(8.0, label="Finger cost", min=1.0, max=20.0, step=0.5)
    # Cuspate: the angle between the two beaches meeting at the mouth, and the
    # marsh left (× the arcuate marsh) on its sandy, wave-built front.
    cusp_angle_deg: float = _f(140.0, label="Cusp angle (°)", min=60.0, max=175.0, step=5.0)
    cuspate_marsh: float = _f(0.4, label="Cuspate marsh (×)", min=0.0, max=1.0, step=0.05)


# Saved shape codes.
_SHAPES = (BIRDFOOT, ARCUATE, CUSPATE)


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
        lake = np.array(LAKE_MASK.read(mesh)) if LAKE_MASK.present(mesh) else None
        filled = (np.array(mesh.face_textures["filled_height"], dtype=np.float32)
                  if "filled_height" in mesh.face_textures else None)
        streams = STREAMS.read(world) if STREAMS.present(world) else None
        winds = world.get_feature("winds") if world.has_feature("winds") else None
        winds = np.stack([w for _, w in winds], axis=1) if winds else None

        result = build_deltas(mesh, rivers, elevation, ocean, lake, filled, streams, self.params,
                              winds=winds, seed=int(world.seed or 0))
        self.fans = result.fans

        ELEVATION.write(mesh, elevation)
        OCEAN_MASK.write(mesh, ocean)
        if lake is not None:
            LAKE_MASK.write(mesh, lake)
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
        from worldgen.delta.shape import SHAPE_NAMES

        mesh.set_face_textures("delta_rgba", DELTA_RECIPE(mesh.face_textures))
        reg.add_scheme(
            "Delta", delta_color, texture="delta_rgba",
            legend=("gradient", delta_legend_entries()), recipe=DELTA_RECIPE,
        )

        centroids = np.asarray(mesh.face_centroids, dtype=np.float64)

        def _delta_hover(face: int, _uv, _w) -> list:
            meta = mesh.face_metadata[face]
            plain = float(meta.get("delta_fraction", 0.0))
            if plain <= 0.0:
                return []
            line = f"Delta plain: {plain:.0%} (marsh {float(meta.get('marsh_fraction', 0.0)):.0%})"
            if self.fans:
                gap = [float(np.linalg.norm(np.asarray(f.apex) - centroids[face])) for f in self.fans]
                fan = self.fans[int(np.argmin(gap))]
                line += f" — {SHAPE_NAMES.get(fan.shape, fan.shape)}{' lake' if fan.lake else ''}, " \
                        f"wave dominance {fan.wave_dominance:.2f}"
                if fan.sediment < 0.995:
                    line += f", sediment {fan.sediment:.0%} of flow"
            return [line]

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
            "delta_head": np.stack([f.head if f.head is not None else f.mouth for f in self.fans]).astype(np.float32),
            "delta_scalars": np.array(
                [[f.radius_km, f.flow, f.area_km2, f.wave_dominance, f.sediment] for f in self.fans],
                dtype=np.float64
            ),
            "delta_shape": np.array([_SHAPES.index(f.shape) for f in self.fans], dtype=np.int8),
            "delta_lake": np.array([f.lake for f in self.fans], dtype=np.bool_),
            "delta_finger_count": np.array([len(f.fingers) for f in self.fans], dtype=np.int32),
            "delta_fingers": np.concatenate(
                [np.asarray(f.fingers, dtype=np.float32).reshape(-1, 3) for f in self.fans]),
        }

    def load(self, arrays: Mapping[str, np.ndarray], ctx: BuildContext) -> None:
        self.fans = []
        if "delta_apex" in arrays:
            ends = np.cumsum(arrays["delta_finger_count"])
            fingers = np.split(np.asarray(arrays["delta_fingers"]).reshape(-1, 3), ends[:-1])
            for a, m, x, hd, (r, q, ar, wd, sd), sh, fg, lk in zip(
                    arrays["delta_apex"], arrays["delta_mouth"], arrays["delta_axis"], arrays["delta_head"],
                    arrays["delta_scalars"], arrays["delta_shape"], fingers, arrays["delta_lake"]):
                self.fans.append(DeltaFan(apex=np.asarray(a), mouth=np.asarray(m), axis=np.asarray(x),
                                          radius_km=float(r), flow=float(q), area_km2=float(ar),
                                          head=np.asarray(hd), wave_dominance=float(wd),
                                          shape=_SHAPES[int(sh)], fingers=np.asarray(fg), lake=bool(lk),
                                          sediment=float(sd)))
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
