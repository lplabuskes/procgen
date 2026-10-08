"""Wetland stage: the wetness score texture and the Wetland biome.

Runs after the climate lake feedback (which reclassifies biomes) on the final relief,
water and rivers (deltas included).  Writes the ``wetland`` score texture, sets the
``biome`` texture to Wetland wherever the score reaches ``threshold`` (land, not ice),
and the per-face ``wetland_fraction``.  The face ``biome`` keeps its climate biome.
See ``score.py`` for the score.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Optional

import numpy as np

from core.display import DisplayRegistry
from core.params import param_field as _f, paramclass
from core.pipeline import BuildContext, Stage
from core.world import World
from worldgen.climate.channels import PRECIPITATION, TEMPERATURE
from worldgen.climate.classification import WETLAND_BIOME
from worldgen.geology.channels import ELEVATION, OCEAN_MASK
from worldgen.hydrology.channels import LAKE_MASK, STREAMS
from worldgen.hydrology.rivers import inter_face_adjacency, potential_evapotranspiration, texel_neighbor_csr
from worldgen.resources.terrain import global_slope
from worldgen.wetland.channels import WETLAND, WETLAND_FRACTION
from worldgen.wetland.score import combine, drainage_targets, flatness, flood_height_above, flood_term, rain_term

if TYPE_CHECKING:
    from core.mesh import PolygonMesh
    from worldgen.hydrology.builder import HydrologyParams


@paramclass
class WetlandParams:
    # The globe marks whole texels (~14 km) Wetland at or above this score; the finer
    # tiers use their own, lower, thresholds.
    threshold: float = _f(0.5, label="Wetland threshold", min=0.0, max=1.0, step=0.05)
    # Flatness halves at this slope.
    slope_scale_m_per_km: float = _f(0.4, label="Flatness slope scale (m/km)", min=0.05, max=10.0, step=0.05)
    # Rain-fed wetness ramps over this range of precipitation / PET.  PET bottoms out at
    # hydrology's baseline in the cold, so boreal ratios run high: these sit well above 1.
    rain_low: float = _f(4.0, label="Rain-fed onset (P/PET)", min=0.0, max=20.0, step=0.1)
    rain_high: float = _f(12.0, label="Rain-fed full (P/PET)", min=0.0, max=40.0, step=0.1)
    # Water held above a channel texel: height · (flow / REFERENCE_FLOW) ** exponent.
    flood_height_m: float = _f(4.0, label="Flood height at reference flow (m)", min=0.0, max=30.0, step=0.5)
    flood_exponent: float = _f(0.45, label="Flood height exponent", min=0.0, max=1.0, step=0.05)
    # Water held above a lake's or the sea's surface along its shore.
    shore_flood_height_m: float = _f(3.0, label="Shore flood height (m)", min=0.0, max=20.0, step=0.5)
    # Flood term: 0 this far above the water, 1 this deep under it.
    bank_m: float = _f(3.0, label="Flooded bank height (m)", min=0.0, max=20.0, step=0.5)
    full_depth_m: float = _f(3.0, label="Full flood depth (m)", min=0.1, max=20.0, step=0.5)


@paramclass
class _PetParams:
    pet_baseline_mm_yr: float = 100.0
    pet_per_degree_mm_yr: float = 60.0
    pet_max_mm_yr: float = 2500.0


def in_polygon_fraction(mesh: "PolygonMesh", mask: np.ndarray) -> np.ndarray:
    """(N,) share of each face's in-polygon texels where ``(N, H, W)`` ``mask`` is set."""
    _, h, w = mask.shape
    inside = mesh.face_texture_in_polygon(h, w)
    return (mask & inside).sum(axis=(1, 2)) / np.maximum(inside.sum(axis=(1, 2)), 1)


def wetland_score(
    mesh: "PolygonMesh",
    params: WetlandParams,
    pet: "HydrologyParams | _PetParams",
    *,
    elevation: np.ndarray,
    water: np.ndarray,
    surface_km: np.ndarray,
    temperature: Optional[np.ndarray],
    precipitation: Optional[np.ndarray],
    channel_nodes: Optional[np.ndarray] = None,
    channel_flow: Optional[np.ndarray] = None,
) -> dict[str, np.ndarray]:
    """The score and its terms, each ``(N, H, W)``; 0 on water.

    ``elevation`` (km) and ``water`` (bool) are ``(N, H, W)``; ``surface_km`` the
    water surface on water texels (sea level, the lake level).
    """
    n, h, w = elevation.shape
    land = ~water.reshape(-1)
    ground_m = elevation.reshape(-1).astype(np.float64) * 1000.0
    surface_m = surface_km.reshape(-1).astype(np.float64) * 1000.0

    flat = flatness(global_slope(mesh, elevation[..., None])[..., 0], params).reshape(-1)
    if temperature is not None and precipitation is not None:
        pet_mm = potential_evapotranspiration(
            np.asarray(temperature, dtype=np.float64).mean(axis=-1),
            float(pet.pet_baseline_mm_yr), float(pet.pet_per_degree_mm_yr), float(pet.pet_max_mm_yr),
        )
        rain = rain_term(np.asarray(precipitation, dtype=np.float64).sum(axis=-1), pet_mm, params).reshape(-1)
    else:
        rain = np.zeros(n * h * w)

    indptr, nbrs = texel_neighbor_csr(n, h, inter_face_adjacency(mesh, h))
    water_flat = ~land
    channel = np.zeros(n * h * w, dtype=bool)
    if channel_nodes is not None:
        channel[np.asarray(channel_nodes, dtype=np.int64)] = True
    target = drainage_targets(elevation.reshape(-1), water_flat, channel, indptr, nbrs)
    depth = flood_height_above(ground_m, water_flat, surface_m, target, channel_nodes, channel_flow, params)
    flood = flood_term(depth, params)

    score = np.where(land, combine(flat, rain, flood), 0.0)
    shape = (n, h, w)
    return {
        "score": score.reshape(shape), "flat": flat.reshape(shape),
        "rain": rain.reshape(shape), "flood": np.where(land, flood, 0.0).reshape(shape),
    }


def stamp_wetland(biome: np.ndarray, score: np.ndarray, threshold: float, land: np.ndarray,
                  frozen: Optional[np.ndarray] = None) -> np.ndarray:
    """Biome indices with Wetland where ``score ≥ threshold`` on unfrozen land (same shape as ``biome``)."""
    wet = (score >= threshold) & land
    if frozen is not None:
        wet &= ~frozen
    return np.where(wet, np.uint8(WETLAND_BIOME), biome).astype(biome.dtype)


class WetlandBuilder(Stage):
    """Scores wetness and stamps the Wetland biome where it is high."""

    name = "wetland"

    def __init__(self, params: WetlandParams | None = None, pet: "HydrologyParams | None" = None) -> None:
        self.params: WetlandParams = params or WetlandParams()
        self.pet = pet if pet is not None else _PetParams()

    def build(self, ctx: BuildContext) -> None:
        mesh = ctx.mesh
        ctx.report("Wetland: scoring wetness…")
        elevation = np.asarray(ELEVATION.read(mesh), dtype=np.float64)[..., 0]
        shape = elevation.shape
        ocean = OCEAN_MASK.read(mesh)[..., 0] > 0.5 if OCEAN_MASK.present(mesh) else elevation <= 0.0
        lake = LAKE_MASK.read(mesh)[..., 0] > 0.5 if LAKE_MASK.present(mesh) else np.zeros(shape, dtype=bool)
        surface = np.zeros(shape)
        if lake.any() and "filled_height" in mesh.face_textures:
            surface = np.where(lake, np.asarray(mesh.face_textures["filled_height"], dtype=np.float64)[..., 0], 0.0)
        streams = STREAMS.read(ctx.world)
        nodes = flow = None
        if streams is not None:
            nodes, flow = streams.nodes[streams.channel], streams.flow[streams.channel]
        temperature = TEMPERATURE.read(mesh) if TEMPERATURE.present(mesh) else None
        terms = wetland_score(
            mesh, self.params, self.pet,
            elevation=elevation, water=ocean | lake, surface_km=surface,
            temperature=temperature,
            precipitation=PRECIPITATION.read(mesh) if PRECIPITATION.present(mesh) else None,
            channel_nodes=nodes, channel_flow=flow,
        )
        WETLAND.write(mesh, terms["score"][..., None].astype(np.float16))

        if "biome" in mesh.face_textures:
            frozen = None
            if temperature is not None:
                frozen = np.asarray(temperature, dtype=np.float64).max(axis=-1) < 0.0
            biome = np.asarray(mesh.face_textures["biome"])[..., 0]
            biome = stamp_wetland(biome, terms["score"], float(self.params.threshold), ~(ocean | lake), frozen)
            mesh.set_face_textures("biome", biome[..., None].astype(np.uint8), n_channels=1)
            WETLAND_FRACTION.write(mesh, in_polygon_fraction(mesh, biome == WETLAND_BIOME))
        ctx.report("Wetland done.")

    # ── Display ───────────────────────────────────────────────────────────────

    def register_display(self, world: World, reg: DisplayRegistry, *, debug: bool = False) -> None:
        mesh = world.mesh
        if not debug or "wetland" not in mesh.face_textures:
            return
        from worldgen.wetland.display import WETLAND_RECIPE, wetland_color, wetland_legend_entries

        mesh.set_face_textures("wetland_rgba", WETLAND_RECIPE(mesh.face_textures))
        reg.add_scheme(
            "Wetland Score", wetland_color, texture="wetland_rgba",
            legend=("gradient", wetland_legend_entries()), recipe=WETLAND_RECIPE,
        )

        def _wetland_hover(face: int, _uv, _w) -> list:
            share = float(mesh.face_metadata[face].get("wetland_fraction", 0.0))
            return [f"Wetland: {share:.0%} of face"] if share > 0.0 else []

        reg.add_hover(_wetland_hover)

    # ── Persistence ───────────────────────────────────────────────────────────
    # The score and biome textures go with the mesh; the face fraction is re-derived.

    def serialize(self) -> dict:
        return {}

    def load(self, arrays: Mapping[str, np.ndarray], ctx: BuildContext) -> None:
        mesh = ctx.mesh
        if "biome" in mesh.face_textures:
            biome = np.asarray(mesh.face_textures["biome"])[..., 0]
            WETLAND_FRACTION.write(mesh, in_polygon_fraction(mesh, biome == WETLAND_BIOME))
