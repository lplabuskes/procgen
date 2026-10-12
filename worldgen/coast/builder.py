"""Coast stage: per-face *coast smoothness* in [0, 1] and its continuous texture.

The globe's coast is unchanged (no shoreline shape at ~14 km texels); the field steers the region and
local tiers' shoreline noise (``region/upsample.py``).  Ruggedness sums tectonic (convergent margins,
volcanism), glacial (cold coasts), minus deltaic (large river mouths), plus low-frequency noise.
Smoothness = ``1 − ruggedness``, clipped, diffused over the face graph and rasterised seam-free.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Optional

import numpy as np

from core.display import DisplayRegistry
from core.params import param_field as _f, paramclass
from core.pipeline import BuildContext, Stage
from core.world import World
from utils.simplex import fbm_sum, make_perm
from worldgen.climate.channels import TEMPERATURE_ANNUAL
from worldgen.coast.channels import COAST_SMOOTHNESS, COAST_SMOOTHNESS_FACE
from worldgen.geology.channels import BOUNDARY_TYPE, ELEVATION, VOLCANIC_ACTIVITY
from worldgen.geology.ruggedness import decayed_max, diffuse
from worldgen.hydrology.channels import RIVER_OBJECTS
from worldgen.resources.terrain import face_hops_from_seeds

if TYPE_CHECKING:
    from core.mesh import PolygonMesh

_MAX_HOPS = 12


@paramclass
class CoastParams:
    # Ruggedness before any cause: 0.4 → smoothness 0.6 on a quiet coast.
    base_ruggedness: float = _f(0.4, label="Base ruggedness", min=0.0, max=1.0, step=0.05)
    # Low-frequency variety: noise in ~[-1, 1] times this weight.
    noise_weight: float = _f(0.35, label="Noise weight", min=0.0, max=1.0, step=0.05)
    # Simplex frequency on the unit sphere; 5 → features ~1,000–2,000 km across.
    noise_frequency: float = _f(5.0, label="Noise frequency", min=0.5, max=30.0, step=0.5)
    # Convergent boundaries: full weight on the boundary, e-folding over this many faces (~220 km each).
    tectonic_weight: float = _f(0.4, label="Tectonic weight", min=0.0, max=1.0, step=0.05)
    tectonic_decay_hops: float = _f(2.0, label="Tectonic decay (faces)", min=0.1, max=10.0, step=0.1)
    volcanic_weight: float = _f(0.3, label="Volcanic weight", min=0.0, max=1.0, step=0.05)
    # Glacial ruggedness ramps from 0 at the warm limit to full at the cold one (annual mean, °C).
    glacial_weight: float = _f(0.5, label="Glacial weight", min=0.0, max=1.0, step=0.05)
    glacial_warm_c: float = _f(4.0, label="Glacial onset (°C)", min=-30.0, max=30.0, step=0.5)
    glacial_cold_c: float = _f(-6.0, label="Glacial full (°C)", min=-40.0, max=20.0, step=0.5)
    # River mouths smooth by sqrt(flow / largest flow), e-folding over this many faces.
    delta_weight: float = _f(0.6, label="Delta weight", min=0.0, max=2.0, step=0.05)
    delta_decay_hops: float = _f(1.5, label="Delta decay (faces)", min=0.1, max=10.0, step=0.1)
    # Neighbour-mean diffusion over the face graph before rasterising.
    smoothing_passes: int = _f(3, label="Smoothing passes", min=0, max=20)
    smoothing_alpha: float = _f(0.5, label="Smoothing strength", min=0.0, max=1.0, step=0.05)


def _ramp(x: np.ndarray, zero_at: float, one_at: float) -> np.ndarray:
    """Linear 0 → 1 between two thresholds, either order."""
    span = one_at - zero_at
    if abs(span) < 1e-12:
        return (x >= one_at).astype(np.float64)
    return np.clip((x - zero_at) / span, 0.0, 1.0)


def coast_smoothness(
    mesh: "PolygonMesh",
    params: CoastParams,
    seed: int,
    *,
    convergent: Optional[np.ndarray] = None,
    volcanic: Optional[np.ndarray] = None,
    temperature_annual: Optional[np.ndarray] = None,
    mouth_strength: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Per-face smoothness in [0, 1] and each term's ruggedness contribution.

    Missing inputs contribute nothing.  ``mouth_strength`` is per-face in [0, 1], non-zero at river mouths.
    """
    n, adjacency = len(mesh.faces), mesh.face_adjacency
    p = params
    pos = np.asarray(mesh.face_centroids, dtype=np.float64)
    pos = pos / np.maximum(np.linalg.norm(pos, axis=1, keepdims=True), 1e-12)
    f0 = float(p.noise_frequency)
    noise = fbm_sum(pos, make_perm(seed), np.array([f0, 2.0 * f0]), np.array([1.0, 0.5])) / 1.5

    terms: dict[str, np.ndarray] = {"noise": float(p.noise_weight) * noise}
    if convergent is not None and convergent.any():
        hops = face_hops_from_seeds(mesh, convergent, max_hops=_MAX_HOPS).astype(np.float64)
        terms["tectonic"] = float(p.tectonic_weight) * np.exp(-hops / max(float(p.tectonic_decay_hops), 1e-6))
    if volcanic is not None:
        terms["volcanic"] = float(p.volcanic_weight) * np.clip(volcanic, 0.0, 1.0)
    if temperature_annual is not None:
        terms["glacial"] = float(p.glacial_weight) * _ramp(
            np.asarray(temperature_annual, dtype=np.float64), float(p.glacial_warm_c), float(p.glacial_cold_c)
        )
    if mouth_strength is not None and np.any(mouth_strength > 0):
        terms["delta"] = -float(p.delta_weight) * decayed_max(
            adjacency, mouth_strength, float(p.delta_decay_hops)
        )

    ruggedness = np.full(n, float(p.base_ruggedness)) + sum(terms.values())
    smooth = np.clip(1.0 - ruggedness, 0.0, 1.0)
    smooth = np.clip(diffuse(smooth, adjacency, p.smoothing_passes, float(p.smoothing_alpha)), 0.0, 1.0)
    return smooth, terms


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


class CoastBuilder(Stage):
    """Computes the coast-smoothness field the finer tiers style shorelines by."""

    name = "coast"

    def __init__(self, params: CoastParams | None = None) -> None:
        self.params: CoastParams = params or CoastParams()

    def build(self, ctx: BuildContext) -> None:
        mesh = ctx.mesh
        n = len(mesh.faces)
        ctx.report("Coast: computing shoreline style…")

        convergent = np.array([b == "convergent" for b in BOUNDARY_TYPE.read(mesh)], dtype=bool) \
            if BOUNDARY_TYPE.present(mesh) else None
        volcanic = VOLCANIC_ACTIVITY.read(mesh) if VOLCANIC_ACTIVITY.present(mesh) else None
        t_annual = TEMPERATURE_ANNUAL.read(mesh) if TEMPERATURE_ANNUAL.present(mesh) else None
        mouths = (
            river_mouth_strength(RIVER_OBJECTS.read(ctx.world), n)
            if RIVER_OBJECTS.present(ctx.world) else None
        )
        seed = int(ctx.rng("coast").integers(0, 2**31 - 1))

        smooth, _ = coast_smoothness(
            mesh, self.params, seed,
            convergent=convergent, volcanic=volcanic,
            temperature_annual=t_annual, mouth_strength=mouths,
        )
        COAST_SMOOTHNESS_FACE.write(mesh, smooth)

        from worldgen.geology.elevation import rasterize_face_values, smooth_texture_seams
        tex_size = ELEVATION.read(mesh).shape[1]
        tex = rasterize_face_values(mesh, smooth, tex_size)
        smooth_texture_seams(mesh, tex, n_passes=1)
        COAST_SMOOTHNESS.write(mesh, np.clip(tex, 0.0, 1.0).astype(np.float32))
        ctx.report("Coast done.")

    # ── Display ───────────────────────────────────────────────────────────────

    def register_display(
        self, world: World, reg: DisplayRegistry, *, debug: bool = False
    ) -> None:
        mesh = world.mesh
        if not debug or "coast_smoothness" not in mesh.face_textures:
            return
        from worldgen.coast.display import (
            COAST_RECIPE, coast_color, coast_legend_entries,
        )

        mesh.set_face_textures("coast_smoothness_rgba", COAST_RECIPE(mesh.face_textures))
        reg.add_scheme(
            "Coast Smoothness", coast_color, texture="coast_smoothness_rgba",
            legend=("gradient", coast_legend_entries()), recipe=COAST_RECIPE,
        )

        def _coast_hover(face: int, _uv, _w) -> list:
            meta = mesh.face_metadata[face]
            if "coast_smoothness" not in meta:
                return []
            return [f"Coast smoothness: {meta['coast_smoothness']:.2f}"]

        reg.add_hover(_coast_hover)

    # ── Persistence ───────────────────────────────────────────────────────────
    # The texture is saved with the mesh; per-face values are restored as its face means.

    def serialize(self) -> dict:
        return {}

    def load(self, arrays: Mapping[str, np.ndarray], ctx: BuildContext) -> None:
        mesh = ctx.mesh
        if "coast_smoothness" in mesh.face_textures:
            tex = np.asarray(mesh.face_textures["coast_smoothness"], dtype=np.float64)
            COAST_SMOOTHNESS_FACE.write(mesh, tex[..., 0].mean(axis=(1, 2)))
