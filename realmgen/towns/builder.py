"""Town stage — the settlements between a village and the city floor.

Runs after refine, once cities have their final sites. Display-only: nothing
downstream reads towns.

1. **Rural density**: farmers spread by ``fertility ** γ``, fishers on the shore,
   miners on their deposit's texel, conserved per face.
2. **Sizes**: ``town_fraction`` of farmers+fishers and ``mining_town_fraction``
   of miners become truncated-Pareto towns of ``town_min`` …
   ``PopulationParams.city_min`` people, RNG keyed on ``(seed, face)``.
3. **Sites**: largest first on owned land texels, clear of towns and cities;
   mining towns only near worked deposits.
4. **Accounting**: towns are carved out of ``rural_population`` and the density
   texture, so rural + towns is unchanged; unsited towns stay rural.
5. **Walls**: rolled per town, likelier in danger and for big towns, rarer than
   a city's (data only).

All per global face, so a town doesn't depend on which region is viewed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from core.display import DisplayRegistry, TextureRecipe
from core.params import param_field as _f, paramclass
from core.pipeline import BuildContext, Stage
from realmgen.channels import (
    FISHING_POPULATION, HINTERLAND_ID, MINING_POPULATION, RURAL_POPULATION, TOWN_POPULATION,
)
from realmgen.polity.seeding import roll_walls
from realmgen.population.builder import PopulationParams
from realmgen.population.calibration import KM2_PER_STERADIAN
from realmgen.refine.builder import RefineParams
from realmgen.refine.sites import SiteField, densify_polylines
from realmgen.scoring.suitability import _positions_to_face_ids
from realmgen.towns.placement import (
    SPACING_EXPONENT,
    compute_mining_density,
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
from worldgen.resources.minerals import mineral_workings, site_workers

RURAL_DENSITY_TEXTURE = "rural_density"
# Fishers' / miners' parts of ``rural_density``, so finer tiers keep them on the shore / by the deposit.
FISHING_DENSITY_TEXTURE = "fishing_density"
MINING_DENSITY_TEXTURE = "mining_density"
_SIZE_STREAM = 1   # RNG stream tag for town sizes (sites use the bare (seed, face) stream)
_MINING_SIZE_STREAM = 2   # ... and for mining-town sizes
_WALL_STREAM = 3   # ... and for town walls (with the town's rank on its face)


@paramclass
class TownParams:
    town_fraction: float = _f(
        0.12, label="Share of rural people living in towns", min=0.0, max=0.5, step=0.01)
    # The rest of the miners live in camps and hamlets by the mine (villages on a local map).
    mining_town_fraction: float = _f(
        0.6, label="Share of miners living in mining towns", min=0.0, max=1.0, step=0.05)
    town_min: float = _f(1_000.0, label="Smallest town (people)", min=100.0, max=5_000.0, step=100.0)
    size_exponent: float = _f(1.0, label="Town size exponent α (1 = rank-size)", min=0.3, max=3.0)
    town_spacing_km: float = _f(
        20.0, label="Town exclusion radius at the smallest size (km)", min=0.0, max=60.0)
    city_clearance_km: float = _f(
        25.0, label="City exclusion radius at the city floor (km)", min=0.0, max=60.0)
    port_bonus: float = _f(2.0, label="Site bonus: harbour", min=0.0, max=10.0)
    confluence_bonus: float = _f(2.0, label="Site bonus: river confluence", min=0.0, max=10.0)
    river_bonus: float = _f(1.0, label="Site bonus: river", min=0.0, max=10.0)
    # Stream: a hydrology stream-network channel too small for the world map's rivers.
    stream_bonus: float = _f(1.0, label="Site bonus: stream", min=0.0, max=10.0)
    route_bonus: float = _f(1.0, label="Site bonus: trade route", min=0.0, max=10.0)
    mining_bonus: float = _f(1.5, label="Site bonus: mineral deposit", min=0.0, max=10.0)
    route_radius_km: float = _f(15.0, label="Trade-route reach (km)", min=1.0, max=60.0)
    mining_radius_km: float = _f(25.0, label="Mineral-deposit reach (km)", min=1.0, max=60.0)
    site_jitter: float = _f(
        0.4, label="Site jitter (× texel spacing)", min=0.0, max=0.5, hidden=True)
    # Rarer and weaker than a city's (PolityParams.defensibility: 0.3 + 0.3 danger + 0.3 wealth).
    wall_base_prob: float = _f(0.05, label="Town walls: base probability", min=0.0, max=1.0)
    wall_danger_boost: float = _f(0.15, label="Town walls: danger boost to prob", min=0.0, max=1.0)
    wall_size_boost: float = _f(0.15, label="Town walls: size boost to prob (at the city floor)", min=0.0, max=1.0)
    wall_min_strength: float = _f(0.2, label="Town walls: min strength", min=0.0, max=5.0)
    wall_max_strength: float = _f(0.8, label="Town walls: max strength", min=0.0, max=5.0)


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


def rural_density_recipe(d_max: float) -> TextureRecipe:
    """The Rural Density layer over any texture set holding ``rural_density`` (globe or region)."""
    def fn(tex) -> np.ndarray:
        water = np.asarray(tex["ocean_mask"])[..., 0] > 0.5
        if "lake_mask" in tex:
            water = water | (np.asarray(tex["lake_mask"])[..., 0] > 0.5)
        return build_rural_density_texture(np.asarray(tex[RURAL_DENSITY_TEXTURE], dtype=np.float32), water, d_max)
    return TextureRecipe(fn, sources=(RURAL_DENSITY_TEXTURE, "ocean_mask"))


def _density_legend(d_max: float) -> list[tuple[str, tuple]]:
    labels = ["0", f"{0.35 ** 2 * d_max:.1f}", f"{0.7 ** 2 * d_max:.1f}", f"{d_max:.0f}/km²"]
    return [(lbl, c) for lbl, (_, c) in zip(labels, _DENSITY_STOPS)]


@dataclass
class TownSiteContext:
    """Town-siting inputs from the rest of the world; distances in unit-sphere chords."""

    city_positions: np.ndarray       # (C, 3)
    city_faces: np.ndarray           # (C,)
    city_radii: np.ndarray           # (C,)
    route_points: Optional[np.ndarray]
    mineral_points: Optional[np.ndarray]
    town_spacing: float
    route_radius: float
    mining_radius: float


def town_site_context(world, p: TownParams, pp: PopulationParams) -> TownSiteContext:
    """The :class:`TownSiteContext` of ``world``; shared with ``region/towns.py``."""
    settlements = world.get_feature("settlements") or []
    city_pop = np.array([s.population for s in settlements], dtype=np.float64)
    workings = mineral_workings(world)
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
        mineral_points=None if workings is None else workings[0],
        town_spacing=float(p.town_spacing_km) / EARTH_RADIUS_KM,
        route_radius=route_radius,
        mining_radius=float(p.mining_radius_km) / EARTH_RADIUS_KM,
    )


# ── TownStage ─────────────────────────────────────────────────────────────────


def _face_danger(world, n: int) -> np.ndarray:
    from worldgen.danger.channels import DANGER_ARRAY
    if DANGER_ARRAY.present(world):
        return np.asarray(DANGER_ARRAY.read(world), dtype=np.float64)
    return np.zeros(n, dtype=np.float64)


def roll_town_walls(
    towns: Towns, danger: np.ndarray, seed: int, p: TownParams, town_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """``(has_walls, wall_strength)`` per town, keyed ``(seed, face, rank on face)``.

    ``prob = base + danger_boost · danger + size_boost · size`` with danger over its
    land p90 and size from ``town_min`` (0) to the city floor (1); quality is biased
    by both, as a city's is by danger and wealth.
    """
    k = len(towns)
    if k == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=np.float64)
    pos = danger[danger > 0.0]
    d_ref = float(np.percentile(pos, 90.0)) if pos.size else 1.0
    d = np.clip(danger[towns.face_id] / max(d_ref, 1e-9), 0.0, 1.0)
    size = np.clip((towns.population - p.town_min) / max(town_max - p.town_min, 1e-9), 0.0, 1.0)
    prob = np.clip(p.wall_base_prob + p.wall_danger_boost * d + p.wall_size_boost * size, 0.0, 1.0)
    rank: dict = {}
    keys = []
    for f in towns.face_id.tolist():
        r = rank.get(f, 0)
        rank[f] = r + 1
        keys.append([seed, int(f), _WALL_STREAM, r])
    return roll_walls(prob, 0.5 * (d + size), keys, p.wall_min_strength, p.wall_max_strength)


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

        # Countryside before towns: rural + any towns an earlier run carved out
        # (PopulationStage zeroes town_population when it rewrites rural_population).
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
        fishing = (
            np.asarray(FISHING_POPULATION.read(mesh), dtype=np.float64)
            if FISHING_POPULATION.present(mesh) else None
        )
        # ``minerals`` is index-aligned with the population stage's ``mine_workers``.
        minerals = world.get_feature("minerals")
        workings = mineral_workings(world)
        workers = world.get_feature("mine_workers")
        mining = np.zeros(n)
        mine_points = mine_faces = None
        mining_density = np.zeros(water.shape)
        if MINING_POPULATION.present(mesh) and minerals is not None and workings is not None and workers is not None:
            mining = np.clip(np.asarray(MINING_POPULATION.read(mesh), dtype=np.float64), 0.0, rural)
            mine_points, site_deposit, site_share = workings
            # Same face assignment as the population stage, to match its channel.
            deposit_faces = _positions_to_face_ids(np.asarray(minerals[0]), np.asarray(mesh.face_centroids))
            mine_faces = deposit_faces[site_deposit]
            workers = site_workers(site_deposit, site_share, workers)
            mining_density, unsited = compute_mining_density(
                mine_points, mine_faces, workers, positions, areas_km2, ~water, owner)
            mining = mining - unsited
        countryside = rural - mining
        density, fishing_density = compute_rural_density(
            countryside, areas_km2,
            None if fert_tex is None else np.asarray(fert_tex[..., 0], dtype=np.float64),
            ~water, owner, float(pp.fertility_exponent), fishing,
        )

        # ── Town sizes, per face ─────────────────────────────────────────────
        town_max = max(float(pp.city_min), float(p.town_min))

        def _requests(budget: np.ndarray, stream: int) -> list:
            out = []
            for f in np.flatnonzero(budget > 0):
                sizes = draw_town_sizes(
                    float(budget[f]), np.random.default_rng([seed, int(f), stream]),
                    town_min=float(p.town_min), town_max=town_max,
                    alpha=float(p.size_exponent), cap=float(rural[f]),
                )
                if sizes.size:
                    out.append((int(f), sizes))
            return out

        requests = _requests(float(p.town_fraction) * countryside, _SIZE_STREAM)
        mining_requests = _requests(float(p.mining_town_fraction) * mining, _MINING_SIZE_STREAM)

        # ── Sites ────────────────────────────────────────────────────────────
        sites = SiteField(
            mesh, ocean_tex, lake_tex, fert_tex, world.get_feature("rivers"),
            river_threshold_factor=float(self.refine_params.river_threshold_factor),
            positions=positions,
            streams=world.get_feature("streams"),
        )
        sc = town_site_context(world, p, pp)

        ctx.report(f"Siting {sum(len(s) for _, s in requests + mining_requests)} towns…")
        towns, unplaced, from_mines = place_towns(
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
            mining_requests=mining_requests,
            mine_points=mine_points,
            mine_faces=mine_faces,
            mine_workers=workers,
        )

        towns.has_walls, towns.wall_strength = roll_town_walls(
            towns, _face_danger(world, n), seed, p, town_max)

        # ── Carve the towns out of the countryside ───────────────────────────
        # A mining town takes miners first; any excess comes from farmers and fishers.
        town_pop = np.bincount(towns.face_id, weights=towns.population, minlength=n)
        mine_town_pop = np.bincount(
            towns.face_id[from_mines], weights=towns.population[from_mines], minlength=n)
        miners_left = np.maximum(mining - mine_town_pop, 0.0)
        countryside_left = np.maximum(countryside - (town_pop - (mining - miners_left)), 0.0)
        keep = np.where(countryside > 0, countryside_left / np.maximum(countryside, 1e-300), 0.0)
        keep_mine = np.where(mining > 0, miners_left / np.maximum(mining, 1e-300), 0.0)
        mining_density = mining_density * keep_mine[owner]
        density = density * keep[owner] + mining_density
        fishing_density *= keep[owner]
        rural_left = countryside_left + miners_left

        RURAL_POPULATION.write(mesh, rural_left)
        TOWN_POPULATION.write(mesh, town_pop)
        mesh.set_face_textures(
            RURAL_DENSITY_TEXTURE, density[..., None].astype(np.float32), n_channels=1)
        mesh.set_face_textures(
            FISHING_DENSITY_TEXTURE, fishing_density[..., None].astype(np.float32), n_channels=1)
        mesh.set_face_textures(
            MINING_DENSITY_TEXTURE, mining_density[..., None].astype(np.float32), n_channels=1)
        self.towns = towns
        world.add_feature("towns", towns)
        ctx.report(
            f"Towns done: {len(towns)} towns ({town_pop.sum() / 1e6:.1f} M people, "
            f"{int(towns.has_walls.sum())} walled), "
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

        if debug and RURAL_DENSITY_TEXTURE in mesh.face_textures:
            density = np.asarray(mesh.face_textures[RURAL_DENSITY_TEXTURE], dtype=np.float32)
            water = mesh.face_textures["ocean_mask"][..., 0] > 0.5
            if "lake_mask" in mesh.face_textures:
                water |= mesh.face_textures["lake_mask"][..., 0] > 0.5
            land_vals = density[..., 0][~water]
            d_max = float(np.percentile(land_vals, 99)) if land_vals.size else 1.0
            recipe = rural_density_recipe(d_max)
            mesh.set_face_textures("rural_density_rgba", recipe(mesh.face_textures))
            reg.add_scheme(
                "Rural Density", lambda meta: OCEAN_COLOR,
                texture="rural_density_rgba", legend=("gradient", _density_legend(d_max)), recipe=recipe,
            )

    # ── Persistence ───────────────────────────────────────────────────────────
    # Only the towns feature; channels and density textures persist generically.

    def serialize(self) -> dict:
        return self.towns.pack() if self.towns is not None else {}

    def load(self, arrays, ctx: BuildContext) -> None:
        if "town_population" in arrays:
            self.towns = Towns.unpack(arrays)
            ctx.world.add_feature("towns", self.towns)
