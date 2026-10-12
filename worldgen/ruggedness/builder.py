"""Ruggedness stage: the elevation stage's tectonic seed refined by what came after it.

The seed (:mod:`worldgen.geology.ruggedness`) knows plates, volcanism and height.
This stage, once climate, rivers, deltas and wetlands exist, adds:

- **glacial** — cold ground was carved by ice (fjords, cirques, knobbly shields):
  a ramp on annual mean temperature;
- **delta** — subtracted at river mouths by ``sqrt(flow / largest flow)``, spread
  over a few faces: sediment-fed coasts and their plains are smooth;
- **wetland** — subtracted by the face's Wetland share: standing water means flat ground.

The terms are diffused over the face graph (the seed already is), added to the
seed, clipped to [0, 1] and rasterised seam-free; ``RUGGEDNESS`` / ``RUGGEDNESS_FACE``
are rewritten in place.  Runs after the wetland stage and before resources,
danger and cost, so every later consumer reads the final field.  The region and
local tiers style their shores by it (``region/upsample.py``).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Optional

import numpy as np

from core.display import DisplayRegistry
from core.params import param_field as _f, paramclass
from core.pipeline import BuildContext, Stage
from core.world import World
from worldgen.climate.channels import TEMPERATURE_ANNUAL
from worldgen.geology.channels import ELEVATION, RUGGEDNESS, RUGGEDNESS_FACE
from worldgen.geology.ruggedness import decayed_max, diffuse, ruggedness_texture
from worldgen.hydrology.channels import RIVER_OBJECTS
from worldgen.wetland.channels import WETLAND_FRACTION

if TYPE_CHECKING:
    from core.mesh import PolygonMesh


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
    # Neighbour-mean diffusion of the refinement terms (the seed is diffused already).
    smoothing_passes: int = _f(2, label="Smoothing passes", min=0, max=20)
    smoothing_alpha: float = _f(0.5, label="Smoothing strength", min=0.0, max=1.0, step=0.05)


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
    """Refines the ruggedness seed in place with glacial, deltaic and wetland causes."""

    name = "ruggedness"

    def __init__(self, params: RuggednessRefineParams | None = None) -> None:
        self.params: RuggednessRefineParams = params or RuggednessRefineParams()

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
        RUGGEDNESS.write(mesh, tex)
        # Per face: the texture's face means, as load restores them, so a saved world reads alike.
        RUGGEDNESS_FACE.write(mesh, tex[..., 0].mean(axis=(1, 2)).astype(np.float64))
        ctx.report("Ruggedness done.")

    # ── Display ───────────────────────────────────────────────────────────────

    def register_display(
        self, world: World, reg: DisplayRegistry, *, debug: bool = False
    ) -> None:
        mesh = world.mesh
        if not debug or "ruggedness" not in mesh.face_textures:
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

    # ── Persistence ───────────────────────────────────────────────────────────
    # The texture is saved with the mesh; per-face values are its face means, at build and on load.

    def serialize(self) -> dict:
        return {}

    def load(self, arrays: Mapping[str, np.ndarray], ctx: BuildContext) -> None:
        mesh = ctx.mesh
        if "ruggedness" in mesh.face_textures:
            tex = np.asarray(mesh.face_textures["ruggedness"], dtype=np.float64)
            RUGGEDNESS_FACE.write(mesh, tex[..., 0].mean(axis=(1, 2)))
