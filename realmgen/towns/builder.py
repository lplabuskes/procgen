"""Town stage — the settlements between a village and the city floor.

Runs after :class:`~realmgen.refine.builder.RefineStage`, once the cities have
their final sites. Towns are **display-only**: nothing downstream reads them.

1. **Rural density** — each face's rural population is spread over its texels
   by the rural model's own food base (``fertility ** γ``), conserved per face
   (``compute_rural_density``).
2. **Town sizes** — a share (``town_fraction``) of each face's rural people is
   spent on towns of ``town_min`` … ``PopulationParams.city_min`` people,
   truncated-Pareto sized, from an RNG keyed on ``(seed, face)``.
3. **Town sites** — towns are sited largest first on owned land texels,
   weighted by density and site bonuses, clear of each other and of cities
   (``place_towns``).
4. **Accounting** — town people come *out of* the rural population: the face's
   ``rural_population`` and its density texture are reduced by exactly the
   towns sited there, so rural + towns is the countryside the population stage
   computed. A town that found no site stays rural.

Everything is per global face and independent of which region is viewed, so a
town is the same whichever neighbouring region the view entered from.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from core.display import DisplayRegistry
from core.params import param_field as _f, paramclass
from core.pipeline import BuildContext, Stage
from realmgen.channels import HINTERLAND_ID, RURAL_POPULATION, TOWN_POPULATION
from realmgen.placement.types import format_population
from realmgen.population.builder import PopulationParams
from realmgen.population.calibration import KM2_PER_STERADIAN
from realmgen.refine.builder import RefineParams
from realmgen.refine.sites import SiteField, densify_polylines
from realmgen.towns.placement import (
    SPACING_EXPONENT,
    compute_rural_density,
    draw_town_sizes,
    place_towns,
    texel_owners,
)
from realmgen.towns.types import TOWN_KIND_COLORS, TOWN_KIND_NAMES, Towns
from viz.color_stops import gradient_texture
from worldgen.cost.surfaces import EARTH_RADIUS_KM
from worldgen.geology.channels import OCEAN_MASK
from worldgen.geology.display import OCEAN_COLOR
from worldgen.hydrology.channels import LAKE_MASK

RURAL_DENSITY_TEXTURE = "rural_density"
_SIZE_STREAM = 1   # RNG stream tag for town sizes (sites use the bare (seed, face) stream)


@paramclass
class TownParams:
    town_fraction: float = _f(
        0.10, label="Share of rural people living in towns", min=0.0, max=0.5, step=0.01)
    town_min: float = _f(1_000.0, label="Smallest town (people)", min=100.0, max=5_000.0, step=100.0)
    size_exponent: float = _f(1.0, label="Town size exponent α (1 = rank-size)", min=0.3, max=3.0)
    town_spacing_km: float = _f(
        20.0, label="Town exclusion radius at the smallest size (km)", min=0.0, max=60.0)
    city_clearance_km: float = _f(
        25.0, label="City exclusion radius at the city floor (km)", min=0.0, max=60.0)
    port_bonus: float = _f(2.0, label="Site bonus: harbour", min=0.0, max=10.0)
    confluence_bonus: float = _f(2.0, label="Site bonus: river confluence", min=0.0, max=10.0)
    river_bonus: float = _f(1.0, label="Site bonus: river", min=0.0, max=10.0)
    # A stream: a channel of the hydrology stream network, too small for the
    # world map's rivers (a region draws it as a tributary).
    stream_bonus: float = _f(1.0, label="Site bonus: stream", min=0.0, max=10.0)
    route_bonus: float = _f(1.0, label="Site bonus: trade route", min=0.0, max=10.0)
    mining_bonus: float = _f(1.5, label="Site bonus: mineral deposit", min=0.0, max=10.0)
    route_radius_km: float = _f(15.0, label="Trade-route reach (km)", min=1.0, max=60.0)
    mining_radius_km: float = _f(25.0, label="Mineral-deposit reach (km)", min=1.0, max=60.0)
    site_jitter: float = _f(
        0.4, label="Site jitter (× texel spacing)", min=0.0, max=0.5, hidden=True)


# ── Display ───────────────────────────────────────────────────────────────────

_DENSITY_STOPS = [
    (0.0, (0.16, 0.14, 0.12, 1.0)),   # empty
    (0.35, (0.45, 0.40, 0.20, 1.0)),
    (0.70, (0.85, 0.65, 0.25, 1.0)),
    (1.0, (1.00, 0.95, 0.60, 1.0)),   # dense agricultural core
]


def build_rural_density_texture(
    density: np.ndarray, water: np.ndarray, d_max: float,
) -> np.ndarray:
    """``(N, H, W, 1)`` people/km² → RGBA, sqrt-scaled to ``d_max``; water flat."""
    s = np.sqrt(np.clip(density[..., 0] / max(d_max, 1e-9), 0.0, 1.0))
    rgba = gradient_texture(s[..., None], _DENSITY_STOPS)
    rgba[water] = OCEAN_COLOR
    return rgba.astype(np.float16)


def _density_legend(d_max: float) -> list[tuple[str, tuple]]:
    labels = ["0", f"{0.35 ** 2 * d_max:.1f}", f"{0.7 ** 2 * d_max:.1f}", f"{d_max:.0f}/km²"]
    return [(lbl, c) for lbl, (_, c) in zip(labels, _DENSITY_STOPS)]


@dataclass
class TownSiteContext:
    """What town siting reads from the rest of the world, in unit-sphere chords:
    the cities' exclusion discs, trade-route and mineral points, and the radii."""

    city_positions: np.ndarray       # (C, 3)
    city_faces: np.ndarray           # (C,)
    city_radii: np.ndarray           # (C,)
    route_points: Optional[np.ndarray]
    mineral_points: Optional[np.ndarray]
    town_spacing: float
    route_radius: float
    mining_radius: float


def town_site_context(world, p: TownParams, pp: PopulationParams) -> TownSiteContext:
    """The :class:`TownSiteContext` of ``world`` — shared by the stage and the
    region view's re-siting (``region/towns.py``)."""
    settlements = world.get_feature("settlements") or []
    city_pop = np.array([s.population for s in settlements], dtype=np.float64)
    minerals = world.get_feature("minerals")
    routes = world.get_feature("trade_routes") or []
    route_radius = float(p.route_radius_km) / EARTH_RADIUS_KM
    return TownSiteContext(
        city_positions=np.array([s.position for s in settlements], dtype=np.float64).reshape(-1, 3),
        city_faces=np.array([s.face_id for s in settlements], dtype=np.int32),
        city_radii=(
            float(p.city_clearance_km) / EARTH_RADIUS_KM
            * (np.maximum(city_pop, 1.0) / float(pp.city_min)) ** SPACING_EXPONENT
        ),
        route_points=densify_polylines(
            [np.stack([np.asarray(a), np.asarray(b)]) for a, b, _ in routes], 0.5 * route_radius,
        ),
        mineral_points=None if minerals is None else np.asarray(minerals[0], dtype=np.float64),
        town_spacing=float(p.town_spacing_km) / EARTH_RADIUS_KM,
        route_radius=route_radius,
        mining_radius=float(p.mining_radius_km) / EARTH_RADIUS_KM,
    )


# ── TownStage ─────────────────────────────────────────────────────────────────


class TownStage(Stage):
    """Sub-city towns, carved out of the rural population (display-only)."""

    name = "realm_towns"
    requires = (RURAL_POPULATION, HINTERLAND_ID, OCEAN_MASK)
    provides = (TOWN_POPULATION,)

    def __init__(
        self,
        params: TownParams | None = None,
        population_params: PopulationParams | None = None,
        refine_params: RefineParams | None = None,
    ) -> None:
        self.params = params or TownParams()
        self.population_params = population_params or PopulationParams()
        self.refine_params = refine_params or RefineParams()
        self.towns: Optional[Towns] = None

    def build(self, ctx: BuildContext) -> None:
        world, mesh = ctx.world, ctx.mesh
        p, pp = self.params, self.population_params
        if not (RURAL_POPULATION.present(mesh) and OCEAN_MASK.present(mesh)):
            return
        seed = int(world.seed or 0)
        n = len(mesh.faces)

        # The countryside before towns: rural + any towns an earlier run carved out
        # (PopulationStage zeroes town_population when it writes a fresh rural_population).
        rural = np.asarray(RURAL_POPULATION.read(mesh), dtype=np.float64)
        if TOWN_POPULATION.present(mesh):
            rural = rural + np.asarray(TOWN_POPULATION.read(mesh), dtype=np.float64)
        hinterland = (
            np.asarray(HINTERLAND_ID.read(mesh), dtype=np.int32)
            if HINTERLAND_ID.present(mesh) else np.full(n, -1, dtype=np.int32)
        )
        ocean_tex = OCEAN_MASK.read(mesh)
        lake_tex = LAKE_MASK.read(mesh) if LAKE_MASK.present(mesh) else None
        fert_tex = mesh.face_textures.get("fertility")
        H, W = int(ocean_tex.shape[1]), int(ocean_tex.shape[2])

        water = ocean_tex[..., 0] > 0.5
        if lake_tex is not None:
            water |= lake_tex[..., 0] > 0.5
        positions = mesh.face_texture_positions(H, W)
        owner = texel_owners(mesh.face_centroids, positions)
        areas_km2 = np.asarray(mesh.face_areas, dtype=np.float64) * KM2_PER_STERADIAN
        density = compute_rural_density(
            rural, areas_km2,
            None if fert_tex is None else np.asarray(fert_tex[..., 0], dtype=np.float64),
            ~water, owner, float(pp.fertility_exponent),
        )

        # ── Town sizes, per face ─────────────────────────────────────────────
        town_max = max(float(pp.city_min), float(p.town_min))
        requests = []
        for f in np.flatnonzero(rural > 0):
            sizes = draw_town_sizes(
                float(p.town_fraction) * rural[f],
                np.random.default_rng([seed, int(f), _SIZE_STREAM]),
                town_min=float(p.town_min), town_max=town_max,
                alpha=float(p.size_exponent), cap=float(rural[f]),
            )
            if sizes.size:
                requests.append((int(f), sizes))

        # ── Sites ────────────────────────────────────────────────────────────
        sites = SiteField(
            mesh, ocean_tex, lake_tex, fert_tex, world.get_feature("rivers"),
            river_threshold_factor=float(self.refine_params.river_threshold_factor),
            positions=positions,
            streams=world.get_feature("streams"),
        )
        sc = town_site_context(world, p, pp)

        ctx.report(f"Siting {sum(len(s) for _, s in requests)} towns…")
        towns, unplaced = place_towns(
            requests,
            sites=sites,
            density=density,
            adjacency=mesh.face_adjacency,
            city_positions=sc.city_positions,
            city_faces=sc.city_faces,
            city_radii=sc.city_radii,
            hinterland=hinterland,
            params=p,
            town_spacing=sc.town_spacing,
            mining_radius=sc.mining_radius,
            route_radius=sc.route_radius,
            mineral_points=sc.mineral_points,
            route_points=sc.route_points,
            seed=seed,
        )

        # ── Carve the towns out of the countryside ───────────────────────────
        town_pop = np.bincount(towns.face_id, weights=towns.population, minlength=n)
        rural_left = np.maximum(rural - town_pop, 0.0)
        keep = np.where(rural > 0, rural_left / np.maximum(rural, 1e-300), 0.0)
        density *= keep[owner]

        RURAL_POPULATION.write(mesh, rural_left)
        TOWN_POPULATION.write(mesh, town_pop)
        mesh.set_face_textures(
            RURAL_DENSITY_TEXTURE, density[..., None].astype(np.float32), n_channels=1)
        self.towns = towns
        world.add_feature("towns", towns)
        ctx.report(
            f"Towns done: {len(towns)} towns ({town_pop.sum() / 1e6:.1f} M people), "
            f"{unplaced.sum() / 1e6:.2f} M left rural for want of a site."
        )

    # ── Display ───────────────────────────────────────────────────────────────

    def register_display(self, world, reg: DisplayRegistry, *, debug: bool = False) -> None:
        mesh = world.mesh
        towns = self.towns
        if towns is not None and len(towns):
            reg.add_overlay(
                "towns", "Towns", checked=True,
                legend=("categorical", list(zip(TOWN_KIND_NAMES, TOWN_KIND_COLORS))),
            )
            n = len(mesh.faces)
            count = np.bincount(towns.face_id, minlength=n)
            pop = np.bincount(towns.face_id, weights=towns.population, minlength=n)
            n_kinds = len(TOWN_KIND_NAMES)
            kinds = np.bincount(
                towns.face_id.astype(np.int64) * n_kinds + towns.kind, minlength=n * n_kinds,
            ).reshape(n, n_kinds)

            def _town_hover(face: int, _uv, _w) -> list:
                if face >= n or count[face] == 0:
                    return []
                return [
                    f"Towns: {count[face]} ({format_population(pop[face])})",
                    ", ".join(
                        f"{k} {TOWN_KIND_NAMES[i].split()[0].lower()}"
                        for i, k in enumerate(kinds[face]) if k
                    ),
                ]

            reg.add_hover(_town_hover)

        if debug and RURAL_DENSITY_TEXTURE in mesh.face_textures:
            density = np.asarray(mesh.face_textures[RURAL_DENSITY_TEXTURE], dtype=np.float32)
            water = mesh.face_textures["ocean_mask"][..., 0] > 0.5
            if "lake_mask" in mesh.face_textures:
                water |= mesh.face_textures["lake_mask"][..., 0] > 0.5
            land_vals = density[..., 0][~water]
            d_max = float(np.percentile(land_vals, 99)) if land_vals.size else 1.0
            mesh.set_face_textures(
                "rural_density_rgba", build_rural_density_texture(density, water, d_max))
            reg.add_scheme(
                "Rural Density", lambda meta: OCEAN_COLOR,
                texture="rural_density_rgba", legend=("gradient", _density_legend(d_max)),
            )

    # ── Persistence ───────────────────────────────────────────────────────────
    # The towns feature is the only thing this stage owns outright; the reduced
    # rural_population / town_population channels round-trip as metadata and the
    # rural_density texture as a mesh texture.

    def serialize(self) -> dict:
        return self.towns.pack() if self.towns is not None else {}

    def load(self, arrays, ctx: BuildContext) -> None:
        if "town_population" in arrays:
            self.towns = Towns.unpack(arrays)
            ctx.world.add_feature("towns", self.towns)
