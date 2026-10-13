"""Ruggedness stage: the elevation stage's tectonic seed refined by what came after it.

The seed (:mod:`worldgen.geology.ruggedness`) knows plates, volcanism and height.
This stage, once climate, rivers, deltas and wetlands exist, adds:

- **glacial** — cold ground was carved by ice (fjords, cirques, knobbly shields):
  a ramp on annual mean temperature;
- **delta** — subtracted at river mouths by ``sqrt(flow / largest flow)``, spread
  over a few faces: sediment-fed coasts and their plains are smooth;
- **wetland** — subtracted by the face's Wetland share: standing water means flat ground;
- **floodplain** — subtracted per texel along the rivers (:func:`floodplain_strength`):
  big, gently falling rivers lay down flat alluvial valleys, a few to tens of km
  wide — far below a face, so this term works on the texture.

The face terms are diffused over the face graph (the seed already is), added to
the seed, clipped to [0, 1] and rasterised seam-free; the floodplain comes off the
texture; ``RUGGEDNESS`` / ``RUGGEDNESS_FACE`` (its face means) are rewritten in
place.  Runs after the wetland stage and before resources, danger and cost, so
every later consumer reads the final field.  The region and
local tiers style their shores by it (``region/upsample.py``).

It refines the **jaggedness** seed (:mod:`worldgen.geology.jaggedness`) alongside,
with climate (:func:`refine_jaggedness`):

- **aridity** — dry ground is jagged (badlands, mesas, sharp scarps: little soil
  and no creep to round it), humid ground rolling: a ramp on precipitation / PET;
- **glacial** — on cold ground (the glacial ramp), rough ground goes jagged
  (arêtes, horns, cirques) and smooth ground rolling (ice-scoured, hummocky
  lowlands): signed by a ramp on the refined ruggedness.

``JAGGEDNESS`` / ``JAGGEDNESS_FACE`` are rewritten in place the same way.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Optional

import numpy as np
from scipy.spatial import KDTree

from core.display import DisplayRegistry
from core.params import field, param_field as _f, paramclass
from core.pipeline import BuildContext, Stage
from core.world import World
from worldgen.climate.channels import PRECIPITATION_ANNUAL, TEMPERATURE_ANNUAL
from worldgen.geology.channels import ELEVATION, JAGGEDNESS, JAGGEDNESS_FACE, RUGGEDNESS, RUGGEDNESS_FACE
from worldgen.geology.ruggedness import decayed_max, diffuse, ruggedness_texture
from worldgen.hydrology.channels import RIVER_OBJECTS, STREAMS
from worldgen.hydrology.rivers import potential_evapotranspiration
from worldgen.resources.terrain import global_slope
from worldgen.wetland.builder import _PetParams
from worldgen.wetland.channels import WETLAND_FRACTION

if TYPE_CHECKING:
    from core.mesh import PolygonMesh
    from worldgen.hydrology.builder import HydrologyParams


@paramclass
class JaggednessRefineParams:
    # Aridity: 0 at or above aridity_humid (precipitation / PET), full at or below aridity_arid.
    # Seed 4's dry biomes (steppe, shrubland, desert) sit at 0.2–0.6, the rest at ~1.5–2 and above.
    aridity_weight: float = _f(0.4, label="Aridity weight", min=0.0, max=1.0, step=0.05)
    aridity_humid: float = _f(1.0, label="Aridity onset (P/PET)", min=0.0, max=10.0, step=0.05)
    aridity_arid: float = _f(0.3, label="Aridity full (P/PET)", min=0.0, max=10.0, step=0.05)
    # Glacial: the cold ramp (annual mean, °C) times ±weight, jagged on rough ground and
    # rolling on smooth, from −1 at alpine_low ruggedness to +1 at alpine_high.
    glacial_weight: float = _f(0.3, label="Glacial weight", min=0.0, max=1.0, step=0.05)
    glacial_warm_c: float = _f(4.0, label="Glacial onset (°C)", min=-30.0, max=30.0, step=0.5)
    glacial_cold_c: float = _f(-6.0, label="Glacial full (°C)", min=-40.0, max=20.0, step=0.5)
    alpine_low: float = _f(0.35, label="Ice-scoured below ruggedness", min=0.0, max=1.0, step=0.05)
    alpine_high: float = _f(0.65, label="Alpine above ruggedness", min=0.0, max=1.0, step=0.05)
    # Neighbour-mean diffusion of the refinement terms (the seed is diffused already).
    smoothing_passes: int = _f(2, label="Smoothing passes", min=0, max=20)
    smoothing_alpha: float = _f(0.5, label="Smoothing strength", min=0.0, max=1.0, step=0.05)


@paramclass
class RuggednessRefineParams:
    # Glacial ruggedness ramps from 0 at the warm limit to full at the cold one (annual mean, °C).
    glacial_weight: float = _f(0.3, label="Glacial weight", min=0.0, max=1.0, step=0.05)
    glacial_warm_c: float = _f(4.0, label="Glacial onset (°C)", min=-30.0, max=30.0, step=0.5)
    glacial_cold_c: float = _f(-6.0, label="Glacial full (°C)", min=-40.0, max=20.0, step=0.5)
    # River mouths smooth by sqrt(flow / largest flow), e-folding over this many faces (~220 km each).
    delta_weight: float = _f(0.6, label="Delta weight", min=0.0, max=2.0, step=0.05)
    delta_decay_hops: float = _f(1.5, label="Delta decay (faces)", min=0.1, max=10.0, step=0.1)
    # Subtracted × the face's Wetland share.
    wetland_weight: float = _f(0.4, label="Wetland weight", min=0.0, max=1.0, step=0.05)
    # Floodplains: the stream network's channel texels at ≥ floodplain_min_flow (the
    # rivers) smooth the texels around them by weight × strength × exp(−distance / width),
    # strength = sqrt(flow / floodplain_full_flow) (≤ 1) × flatness, flatness =
    # 1 / (1 + (slope / floodplain_slope)²) on the globe's slope at the river (gorges
    # through a range stay rough); width = floodplain_width_km × (flow / min flow) ** exponent.
    floodplain_weight: float = _f(0.35, label="Floodplain weight", min=0.0, max=1.0, step=0.05)
    floodplain_min_flow: float = _f(500.0, label="Floodplain min river flow", min=1.0, max=100_000.0, step=50.0)
    floodplain_full_flow: float = _f(20_000.0, label="Floodplain full-strength flow", min=1.0, max=1_000_000.0,
                                     step=500.0)
    floodplain_width_km: float = _f(6.0, label="Floodplain width at min flow (km)", min=0.5, max=100.0, step=0.5)
    floodplain_width_exponent: float = _f(0.4, label="Floodplain width exponent", min=0.0, max=1.0, step=0.05)
    floodplain_slope: float = _f(0.001, label="Floodplain flatness slope (km/km)", min=0.0001, max=0.05,
                                 step=0.0005)
    # Neighbour-mean diffusion of the refinement terms (the seed is diffused already).
    smoothing_passes: int = _f(2, label="Smoothing passes", min=0, max=20)
    smoothing_alpha: float = _f(0.5, label="Smoothing strength", min=0.0, max=1.0, step=0.05)
    jaggedness: JaggednessRefineParams = field(
        default_factory=JaggednessRefineParams, metadata={"label": "Jaggedness"})


def _ramp(x: np.ndarray, zero_at: float, one_at: float) -> np.ndarray:
    """Linear 0 → 1 between two thresholds, either order."""
    span = one_at - zero_at
    if abs(span) < 1e-12:
        return (x >= one_at).astype(np.float64)
    return np.clip((x - zero_at) / span, 0.0, 1.0)


def refine_ruggedness(
    mesh: "PolygonMesh",
    seed_values: np.ndarray,
    params: RuggednessRefineParams,
    *,
    temperature_annual: Optional[np.ndarray] = None,
    mouth_strength: Optional[np.ndarray] = None,
    wetland_fraction: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Per-face refined ruggedness in [0, 1] and each term's (undiffused) contribution.

    Missing inputs contribute nothing.  ``mouth_strength`` is per-face in [0, 1], non-zero at river mouths.
    """
    p = params
    adjacency = mesh.face_adjacency
    terms: dict[str, np.ndarray] = {}
    if temperature_annual is not None:
        terms["glacial"] = float(p.glacial_weight) * _ramp(
            np.asarray(temperature_annual, dtype=np.float64), float(p.glacial_warm_c), float(p.glacial_cold_c)
        )
    if mouth_strength is not None and np.any(mouth_strength > 0):
        terms["delta"] = -float(p.delta_weight) * decayed_max(adjacency, mouth_strength, float(p.delta_decay_hops))
    if wetland_fraction is not None:
        terms["wetland"] = -float(p.wetland_weight) * np.clip(
            np.asarray(wetland_fraction, dtype=np.float64), 0.0, 1.0)
    seed_values = np.asarray(seed_values, dtype=np.float64)
    if not terms:
        return np.clip(seed_values, 0.0, 1.0), terms
    delta = diffuse(np.sum(list(terms.values()), axis=0), adjacency, p.smoothing_passes, float(p.smoothing_alpha))
    return np.clip(seed_values + delta, 0.0, 1.0), terms


def refine_jaggedness(
    mesh: "PolygonMesh",
    seed_values: np.ndarray,
    params: JaggednessRefineParams,
    *,
    ruggedness: np.ndarray,
    temperature_annual: Optional[np.ndarray] = None,
    aridity_index: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Per-face refined jaggedness in [0, 1] and each term's (undiffused) contribution.

    ``ruggedness`` per face (the refined field); ``aridity_index`` per face
    precipitation / PET.  Missing climate inputs contribute nothing.
    """
    p = params
    adjacency = mesh.face_adjacency
    terms: dict[str, np.ndarray] = {}
    if aridity_index is not None:
        terms["aridity"] = float(p.aridity_weight) * _ramp(
            np.asarray(aridity_index, dtype=np.float64), float(p.aridity_humid), float(p.aridity_arid))
    if temperature_annual is not None:
        cold = _ramp(np.asarray(temperature_annual, dtype=np.float64), float(p.glacial_warm_c), float(p.glacial_cold_c))
        alpine = 2.0 * _ramp(np.asarray(ruggedness, dtype=np.float64), float(p.alpine_low), float(p.alpine_high)) - 1.0
        terms["glacial"] = float(p.glacial_weight) * cold * alpine
    seed_values = np.asarray(seed_values, dtype=np.float64)
    if not terms:
        return np.clip(seed_values, 0.0, 1.0), terms
    delta = diffuse(np.sum(list(terms.values()), axis=0), adjacency, p.smoothing_passes, float(p.smoothing_alpha))
    return np.clip(seed_values + delta, 0.0, 1.0), terms


_EARTH_RADIUS_KM = 6371.0
# Floodplain reach, in widths: exp(-3) ≈ 5% beyond it.
_FLOODPLAIN_REACH = 3.0


def floodplain_strength(mesh: "PolygonMesh", streams, params: RuggednessRefineParams) -> np.ndarray:
    """``(N, H, W)`` in [0, 1]: how much each texel is a river's floodplain (see the params).

    ``streams`` is the hydrology stage's :class:`~worldgen.hydrology.types.StreamNetwork`
    (its nodes are texel centres of the elevation texture).
    """
    p = params
    elev = ELEVATION.read(mesh)
    n, h, w = elev.shape[0], elev.shape[1], elev.shape[2]
    out = np.zeros(n * h * w, dtype=np.float64)
    if streams is None or len(streams) == 0:
        return out.reshape(n, h, w)
    flow = np.asarray(streams.flow, dtype=np.float64)
    src = np.flatnonzero(np.asarray(streams.channel) & (flow >= float(p.floodplain_min_flow)))
    if src.size == 0:
        return out.reshape(n, h, w)
    flow = flow[src]
    nodes = np.asarray(streams.nodes, dtype=np.int64)[src]
    slope = global_slope(mesh, elev)[..., 0].reshape(-1)[np.clip(nodes, 0, n * h * w - 1)]
    flat = 1.0 / (1.0 + (slope / max(float(p.floodplain_slope), 1e-9)) ** 2)
    strength = np.minimum(np.sqrt(flow / max(float(p.floodplain_full_flow), 1e-9)), 1.0) * flat
    width_rad = (float(p.floodplain_width_km) * (flow / max(float(p.floodplain_min_flow), 1e-9))
                 ** float(p.floodplain_width_exponent)) / _EARTH_RADIUS_KM
    keep = strength > 1e-3
    if not keep.any():
        return out.reshape(n, h, w)
    pos = np.asarray(streams.positions, dtype=np.float64)[src][keep]
    strength, width_rad = strength[keep], width_rad[keep]
    texels = np.asarray(mesh.face_texture_positions(h, w), dtype=np.float64).reshape(-1, 3)
    texels /= np.maximum(np.linalg.norm(texels, axis=1, keepdims=True), 1e-12)
    tree = KDTree(texels)
    hits = tree.query_ball_point(pos, _FLOODPLAIN_REACH * width_rad)
    counts = np.fromiter((len(x) for x in hits), dtype=np.int64, count=len(hits))
    if counts.sum() == 0:
        return out.reshape(n, h, w)
    which = np.repeat(np.arange(len(hits)), counts)
    idx = np.concatenate([np.asarray(x, dtype=np.int64) for x in hits if x])
    d = np.linalg.norm(texels[idx] - pos[which], axis=1)          # chord ≈ angle at these scales
    np.maximum.at(out, idx, strength[which] * np.exp(-d / width_rad[which]))
    return out.reshape(n, h, w)


def river_mouth_strength(rivers, n_faces: int) -> np.ndarray:
    """Per-face delta strength: ``sqrt(flow / largest flow)`` at each river mouth."""
    strength = np.zeros(n_faces, dtype=np.float64)
    mouths: list[tuple[int, float]] = []
    for river in rivers or []:
        if river.mouth is None or not river.segments:
            continue
        mouths.append((int(river.mouth.face_id), float(river.segments[-1].flow_rate)))
    if not mouths:
        return strength
    peak = max(flow for _, flow in mouths)
    if peak <= 0:
        return strength
    for face, flow in mouths:
        strength[face] = max(strength[face], float(np.sqrt(max(flow, 0.0) / peak)))
    return strength


class RuggednessBuilder(Stage):
    """Refines the ruggedness seed in place with glacial, deltaic and wetland causes, and jaggedness with climate."""

    name = "ruggedness"

    def __init__(
        self, params: RuggednessRefineParams | None = None, pet: "HydrologyParams | None" = None,
    ) -> None:
        self.params: RuggednessRefineParams = params or RuggednessRefineParams()
        # PET for the aridity index, as the hydrology and wetland stages compute it.
        self.pet: "HydrologyParams | _PetParams" = pet if pet is not None else _PetParams()

    def build(self, ctx: BuildContext) -> None:
        mesh = ctx.mesh
        n = len(mesh.faces)
        if not RUGGEDNESS_FACE.present(mesh):
            raise ValueError("no ruggedness seed — run the elevation stage first")
        ctx.report("Ruggedness: refining…")
        t_annual = TEMPERATURE_ANNUAL.read(mesh) if TEMPERATURE_ANNUAL.present(mesh) else None
        mouths = (
            river_mouth_strength(RIVER_OBJECTS.read(ctx.world), n)
            if RIVER_OBJECTS.present(ctx.world) else None
        )
        wetland = WETLAND_FRACTION.read(mesh) if WETLAND_FRACTION.present(mesh) else None

        rho, _ = refine_ruggedness(
            mesh, RUGGEDNESS_FACE.read(mesh), self.params,
            temperature_annual=t_annual, mouth_strength=mouths, wetland_fraction=wetland,
        )
        tex = ruggedness_texture(mesh, rho, ELEVATION.read(mesh).shape[1])
        if float(self.params.floodplain_weight) > 0.0 and STREAMS.present(ctx.world):
            plain = floodplain_strength(mesh, STREAMS.read(ctx.world), self.params)
            tex = np.clip(tex - float(self.params.floodplain_weight) * plain[..., None], 0.0, 1.0).astype(np.float32)
        RUGGEDNESS.write(mesh, tex)
        # Per face: the texture's face means, as load restores them, so a saved world reads alike.
        RUGGEDNESS_FACE.write(mesh, tex[..., 0].mean(axis=(1, 2)).astype(np.float64))
        if JAGGEDNESS_FACE.present(mesh):
            self._refine_jaggedness(mesh, rho, t_annual)
        ctx.report("Ruggedness done.")

    def _refine_jaggedness(self, mesh: "PolygonMesh", rho: np.ndarray, t_annual: Optional[np.ndarray]) -> None:
        aridity = None
        if t_annual is not None and PRECIPITATION_ANNUAL.present(mesh):
            pet = self.pet
            pet_mm = potential_evapotranspiration(
                np.asarray(t_annual, dtype=np.float64),
                float(pet.pet_baseline_mm_yr), float(pet.pet_per_degree_mm_yr), float(pet.pet_max_mm_yr),
            )
            aridity = np.asarray(PRECIPITATION_ANNUAL.read(mesh), dtype=np.float64) / np.maximum(pet_mm, 1e-9)
        jag, _ = refine_jaggedness(
            mesh, JAGGEDNESS_FACE.read(mesh), self.params.jaggedness,
            ruggedness=rho, temperature_annual=t_annual, aridity_index=aridity,
        )
        tex = ruggedness_texture(mesh, jag, ELEVATION.read(mesh).shape[1])
        JAGGEDNESS.write(mesh, tex)
        JAGGEDNESS_FACE.write(mesh, tex[..., 0].mean(axis=(1, 2)).astype(np.float64))

    # ── Display ───────────────────────────────────────────────────────────────

    def register_display(
        self, world: World, reg: DisplayRegistry, *, debug: bool = False
    ) -> None:
        mesh = world.mesh
        if not debug:
            return
        if "jaggedness" in mesh.face_textures:
            self._register_jaggedness(mesh, reg)
        if "ruggedness" not in mesh.face_textures:
            return
        from worldgen.geology.display import RUGGEDNESS_RECIPE, ruggedness_color, ruggedness_legend_entries

        mesh.set_face_textures("ruggedness_rgba", RUGGEDNESS_RECIPE(mesh.face_textures))
        reg.add_scheme(
            "Ruggedness", ruggedness_color, texture="ruggedness_rgba",
            legend=("gradient", ruggedness_legend_entries()), recipe=RUGGEDNESS_RECIPE,
        )

        def _ruggedness_hover(face: int, _uv, local) -> list:
            refined = local(("ruggedness",)) if local is not None else None
            if refined:
                return refined
            meta = mesh.face_metadata[face]
            return [f"Ruggedness: {meta['ruggedness']:.2f}"] if "ruggedness" in meta else []

        reg.add_hover(_ruggedness_hover, refined=("ruggedness",))

    @staticmethod
    def _register_jaggedness(mesh: "PolygonMesh", reg: DisplayRegistry) -> None:
        from worldgen.geology.display import JAGGEDNESS_RECIPE, jaggedness_color, jaggedness_legend_entries

        mesh.set_face_textures("jaggedness_rgba", JAGGEDNESS_RECIPE(mesh.face_textures))
        reg.add_scheme(
            "Jaggedness", jaggedness_color, texture="jaggedness_rgba",
            legend=("gradient", jaggedness_legend_entries()), recipe=JAGGEDNESS_RECIPE,
        )

        def _jaggedness_hover(face: int, _uv, local) -> list:
            refined = local(("jaggedness",)) if local is not None else None
            if refined:
                return refined
            meta = mesh.face_metadata[face]
            return [f"Jaggedness: {meta['jaggedness']:.2f}"] if "jaggedness" in meta else []

        reg.add_hover(_jaggedness_hover, refined=("jaggedness",))

    # ── Persistence ───────────────────────────────────────────────────────────
    # The texture is saved with the mesh; per-face values are its face means, at build and on load.

    def serialize(self) -> dict:
        return {}

    def load(self, arrays: Mapping[str, np.ndarray], ctx: BuildContext) -> None:
        mesh = ctx.mesh
        if "ruggedness" in mesh.face_textures:
            tex = np.asarray(mesh.face_textures["ruggedness"], dtype=np.float64)
            RUGGEDNESS_FACE.write(mesh, tex[..., 0].mean(axis=(1, 2)))
        if "jaggedness" in mesh.face_textures:
            tex = np.asarray(mesh.face_textures["jaggedness"], dtype=np.float64)
            JAGGEDNESS_FACE.write(mesh, tex[..., 0].mean(axis=(1, 2)))
