"""Danger stage — DangerParams and DangerBuilder.

Runs after the resources stage. Seeds per-face danger values from terrain
proxies (geology, climate, hydrology, resources) and writes a clamped [0, 1]
float32 danger field per face. Consumed by realmgen and display.
"""
from __future__ import annotations

from typing import Mapping, Optional

import numpy as np

from core.display import DisplayRegistry
from core.params import field, param_field as _f, paramclass
from core.pipeline import BuildContext, Stage
from core.world import World
from worldgen.climate.channels import BIOME
from worldgen.danger.channels import DANGER, DANGER_ARRAY
from worldgen.danger.sources import (
    ACTIVE_BOUNDARY_TYPES,
    AIR_RIFT_ELEV_MIN,
    BARROW_ELEV_MAX,
    BARROW_ISOLATION_MIN,
    BARROW_RUGG_MAX,
    CONTINENTAL_TYPES,
    CURSED_FOREST_DENSITY_MIN,
    CURSED_FOREST_ISOLATION_MIN,
    DEADWATER_DEPTH_MIN,
    DRAGON_VOLCANIC_MIN,
    FIRE_RIFT_VOLCANIC_MIN,
    GIANT_ELEV_MIN,
    GOBLIN_FOREST_MIN,
    GRASSLAND_BIOMES,
    HARPY_ELEV_MIN,
    HAUNT_COAST_MAX,
    HAUNT_FERTILITY_MAX,
    KRAKEN_DEPTH_MIN,
    LICH_ELEV_MIN,
    LICH_FERTILITY_MAX,
    LICH_FOREST_MAX,
    PASS_RUGG_MAX,
    PASS_RUGG_MIN,
    SALAMANDER_VOLCANIC_MIN,
    SCRUB_BIOMES,
    SEA_SERPENT_COAST_MAX,
    WATER_RIFT_DEPTH_MIN,
    WEREWOLF_FOREST_MIN,
    coastal_land_faces,
    source_contribution,
)
from worldgen.geology.channels import (
    BOUNDARY_TYPE,
    ELEVATION,
    IS_OCEAN,
    PLATE_TYPE,
    VOLCANIC_ACTIVITY,
)
from worldgen.hydrology.channels import LAKE_MASK, RIVER_OBJECTS
from worldgen.resources.channels import FERTILITY_MEAN
from worldgen.resources.terrain import (
    coastal_ocean_faces,
    face_elevation_range,
    face_hops_from_seeds,
    lake_containing_faces,
    river_traversed_faces,
)


def _pos_to_face(mesh, pos: np.ndarray) -> Optional[int]:
    """Return the face index whose centroid is closest to ``pos``."""
    diffs = mesh.face_centroids - pos.astype(np.float64)
    idx = int(np.argmin((diffs * diffs).sum(axis=1)))
    return idx


# Display name for each source, in the same order they are added in build().
SOURCE_NAMES: list[str] = [
    "Dragon Lair",
    "Fire Elemental Rift",
    "Salamander Colony",
    "Kraken Domain",
    "Deadwater Zone",
    "Sea Serpent Territory",
    "Harpy Peak",
    "Troll Hold",
    "Giant Lair",
    "Goblin Warband",
    "Werewolf Range",
    "Basilisk Territory",
    "Cursed Forest",
    "Fire Rift",
    "Earth Rift",
    "Air Rift",
    "Water Rift",
    "Lich Stronghold",
    "Barrow Field",
    "Haunt",
]

# Category color per source (R, G, B, A) — grouped into six threat categories.
_C_VOL = (0.95, 0.25, 0.05, 1.0)   # volcanic
_C_SEA = (0.10, 0.35, 0.80, 1.0)   # ocean / coastal
_C_HIG = (0.60, 0.45, 0.20, 1.0)   # highland
_C_FOR = (0.15, 0.50, 0.10, 1.0)   # forest / wilderness
_C_PLN = (0.70, 0.10, 0.85, 1.0)   # planar rift
_C_UND = (0.35, 0.05, 0.45, 1.0)   # undead / necromantic
SOURCE_COLORS: list[tuple] = [
    _C_VOL, _C_VOL, _C_VOL,          # 0-2  dragon, fire rift, salamander
    _C_SEA, _C_SEA, _C_SEA,          # 3-5  kraken, deadwater, sea serpent
    _C_HIG, _C_HIG, _C_HIG,          # 6-8  harpy, troll, giant
    _C_FOR, _C_FOR, _C_FOR, _C_FOR,  # 9-12 goblin, werewolf, basilisk, cursed
    _C_PLN, _C_PLN, _C_PLN, _C_PLN,  # 13-16 fire/earth/air/water rifts
    _C_UND, _C_UND, _C_UND,          # 17-19 lich, barrow, haunt
]
_SOURCE_CAT_LEGEND: list[tuple[str, tuple]] = [
    ("Volcanic",    _C_VOL),
    ("Ocean",       _C_SEA),
    ("Highland",    _C_HIG),
    ("Forest",      _C_FOR),
    ("Planar Rift", _C_PLN),
    ("Undead",      _C_UND),
]


@paramclass
class DangerSourceParams:
    count: int = field(default=4, metadata={"label": "Count", "min": 0, "max": 50})
    hop_radius: int = field(default=1, metadata={"label": "Hop radius", "min": 0, "max": 10})
    intensity: float = _f(0.7, label="Intensity", min=0.0, max=1.0)
    decay_per_hop: float = _f(0.3, label="Decay/hop", min=0.0, max=3.0)
    enabled: bool = field(default=True, metadata={"label": "Enabled"})


@paramclass
class DangerParams:
    # Volcanic / Tectonic
    dragon_lairs: DangerSourceParams = field(
        default_factory=lambda: DangerSourceParams(count=4, hop_radius=1, intensity=0.8, decay_per_hop=0.3),
        metadata={"label": "Dragon Lairs"},
    )
    fire_elemental_rifts: DangerSourceParams = field(
        default_factory=lambda: DangerSourceParams(count=6, hop_radius=0, intensity=0.9, decay_per_hop=0.0),
        metadata={"label": "Fire Elemental Rifts"},
    )
    salamander_colonies: DangerSourceParams = field(
        default_factory=lambda: DangerSourceParams(count=8, hop_radius=0, intensity=0.5, decay_per_hop=0.0),
        metadata={"label": "Salamander Colonies"},
    )
    # Coastal / Ocean
    kraken_domains: DangerSourceParams = field(
        default_factory=lambda: DangerSourceParams(count=4, hop_radius=2, intensity=0.7, decay_per_hop=0.15),
        metadata={"label": "Kraken Domains"},
    )
    deadwater_zones: DangerSourceParams = field(
        default_factory=lambda: DangerSourceParams(count=7, hop_radius=2, intensity=0.4, decay_per_hop=0.15),
        metadata={"label": "Deadwater Zones"},
    )
    sea_serpent_territories: DangerSourceParams = field(
        default_factory=lambda: DangerSourceParams(count=11, hop_radius=1, intensity=0.6, decay_per_hop=0.4),
        metadata={"label": "Sea Serpent Territories"},
    )
    # Highland / Pass
    harpy_peaks: DangerSourceParams = field(
        default_factory=lambda: DangerSourceParams(count=9, hop_radius=0, intensity=0.8, decay_per_hop=0.0),
        metadata={"label": "Harpy Peaks"},
    )
    troll_holds: DangerSourceParams = field(
        default_factory=lambda: DangerSourceParams(count=6, hop_radius=0, intensity=0.95, decay_per_hop=0.0),
        metadata={"label": "Troll Holds"},
    )
    giant_lairs: DangerSourceParams = field(
        default_factory=lambda: DangerSourceParams(count=7, hop_radius=1, intensity=0.6, decay_per_hop=0.25),
        metadata={"label": "Giant Lairs"},
    )
    # Forest / Wilderness
    goblin_warbands: DangerSourceParams = field(
        default_factory=lambda: DangerSourceParams(count=20, hop_radius=2, intensity=0.35, decay_per_hop=0.2),
        metadata={"label": "Goblin Warbands"},
    )
    werewolf_ranges: DangerSourceParams = field(
        default_factory=lambda: DangerSourceParams(count=11, hop_radius=2, intensity=0.35, decay_per_hop=0.2),
        metadata={"label": "Werewolf Ranges"},
    )
    basilisk_territories: DangerSourceParams = field(
        default_factory=lambda: DangerSourceParams(count=9, hop_radius=1, intensity=0.55, decay_per_hop=0.3),
        metadata={"label": "Basilisk Territories"},
    )
    cursed_forests: DangerSourceParams = field(
        default_factory=lambda: DangerSourceParams(count=6, hop_radius=2, intensity=0.6, decay_per_hop=0.2),
        metadata={"label": "Cursed Forests"},
    )
    # Planar Intrusions
    fire_rifts: DangerSourceParams = field(
        default_factory=lambda: DangerSourceParams(count=4, hop_radius=0, intensity=0.7, decay_per_hop=0.0),
        metadata={"label": "Fire Rifts"},
    )
    earth_rifts: DangerSourceParams = field(
        default_factory=lambda: DangerSourceParams(count=6, hop_radius=0, intensity=0.7, decay_per_hop=0.0),
        metadata={"label": "Earth Rifts"},
    )
    air_rifts: DangerSourceParams = field(
        default_factory=lambda: DangerSourceParams(count=6, hop_radius=0, intensity=0.6, decay_per_hop=0.0),
        metadata={"label": "Air Rifts"},
    )
    water_rifts: DangerSourceParams = field(
        default_factory=lambda: DangerSourceParams(count=6, hop_radius=0, intensity=0.6, decay_per_hop=0.0),
        metadata={"label": "Water Rifts"},
    )
    # Undead / Necromantic
    lich_strongholds: DangerSourceParams = field(
        default_factory=lambda: DangerSourceParams(count=3, hop_radius=0, intensity=0.9, decay_per_hop=0.0),
        metadata={"label": "Lich Strongholds"},
    )
    barrow_fields: DangerSourceParams = field(
        default_factory=lambda: DangerSourceParams(count=11, hop_radius=1, intensity=0.6, decay_per_hop=0.5),
        metadata={"label": "Barrow Fields"},
    )
    haunt_concentrations: DangerSourceParams = field(
        default_factory=lambda: DangerSourceParams(count=15, hop_radius=0, intensity=0.5, decay_per_hop=0.5),
        metadata={"label": "Haunt Concentrations"},
    )


class DangerBuilder(Stage):
    """Computes per-face danger values from terrain proxies.

    Slots into the globe pipeline between resources and cost. Writes the
    ``danger`` MetaChannel (per-face metadata) and ``danger`` FeatureChannel
    (numpy array) for realmgen consumption.
    """

    name = "danger"

    def __init__(self, params: DangerParams | None = None) -> None:
        self.params: DangerParams = params or DangerParams()
        self.danger: Optional[np.ndarray] = None   # (N,) float32
        self.sources: Optional[tuple] = None        # (positions (M,3), kinds (M,) int8)

    def build(self, ctx: BuildContext) -> None:
        mesh = ctx.mesh
        rng = ctx.rng("danger")
        n = len(mesh.faces)
        p = self.params
        _report = ctx.report

        _report("Danger: reading inputs…")
        volcanic_activity = VOLCANIC_ACTIVITY.read(mesh)
        is_ocean = IS_OCEAN.read(mesh)
        biome = BIOME.read(mesh)
        elevation_tex = ELEVATION.read(mesh)
        lake_mask_tex = LAKE_MASK.read(mesh) if LAKE_MASK.present(mesh) else None
        rivers = RIVER_OBJECTS.read(ctx.world)
        fertility_mean = FERTILITY_MEAN.read(mesh).astype(np.float32)
        forest_density_mean = np.array(
            [m.get("forest_density_mean", 0.0) for m in mesh.face_metadata],
            dtype=np.float32,
        )
        boundary_type = BOUNDARY_TYPE.read(mesh)
        plate_type = PLATE_TYPE.read(mesh)

        # Derived terrain inputs
        is_lake_face = (
            lake_containing_faces(lake_mask_tex) if lake_mask_tex is not None
            else np.zeros(n, dtype=bool)
        )
        is_water = is_ocean | is_lake_face
        elev_mean = elevation_tex[..., 0].mean(axis=(1, 2)).astype(np.float32)
        elev_range = face_elevation_range(elevation_tex)

        _report("Danger: computing ocean depth hops…")
        _max_ocean_depth = max(KRAKEN_DEPTH_MIN, DEADWATER_DEPTH_MIN, WATER_RIFT_DEPTH_MIN) + 2
        ocean_depth_hops = face_hops_from_seeds(
            mesh, coastal_ocean_faces(mesh, is_ocean), max_hops=_max_ocean_depth
        )

        _report("Danger: computing coast hops…")
        _max_coast = max(SEA_SERPENT_COAST_MAX, HAUNT_COAST_MAX) + 1
        coast_hops = face_hops_from_seeds(
            mesh, coastal_land_faces(mesh, is_ocean), max_hops=_max_coast
        )

        _report("Danger: computing water proximity hops…")
        water_seed_mask = river_traversed_faces(rivers, n_faces=n)
        if lake_mask_tex is not None:
            water_seed_mask |= lake_containing_faces(lake_mask_tex)
        _max_water = max(CURSED_FOREST_ISOLATION_MIN, BARROW_ISOLATION_MIN) + 1
        water_hops = face_hops_from_seeds(mesh, water_seed_mask, max_hops=_max_water)

        # Shared candidate mask reused by fire elemental rifts + fire rifts
        fire_volcanic_mask = volcanic_activity >= FIRE_RIFT_VOLCANIC_MIN

        danger = np.zeros(n, dtype=np.float32)

        # Track (face_idx, source_kind) for the overlay.
        _seed_records: list[tuple[int, int]] = []
        _kind_counter = [0]  # mutable int for closure

        def _add(candidate_mask: np.ndarray, sp: DangerSourceParams) -> None:
            kind = _kind_counter[0]
            _kind_counter[0] += 1
            if sp.enabled:
                contrib, seeds = source_contribution(mesh, rng, candidate_mask, sp)
                danger.__iadd__(contrib)
                for s in seeds:
                    _seed_records.append((int(s), kind))

        _report("Danger: volcanic / tectonic sources…")
        _add(volcanic_activity >= DRAGON_VOLCANIC_MIN, p.dragon_lairs)
        _add(fire_volcanic_mask, p.fire_elemental_rifts)
        _add(volcanic_activity >= SALAMANDER_VOLCANIC_MIN, p.salamander_colonies)

        _report("Danger: coastal / ocean sources…")
        _add(is_ocean & (ocean_depth_hops >= KRAKEN_DEPTH_MIN), p.kraken_domains)
        _add(is_ocean & (ocean_depth_hops >= DEADWATER_DEPTH_MIN), p.deadwater_zones)
        _add(~is_ocean & (coast_hops <= SEA_SERPENT_COAST_MAX), p.sea_serpent_territories)

        _report("Danger: highland / pass sources…")
        _add(
            (elev_mean >= HARPY_ELEV_MIN) & (elev_range >= PASS_RUGG_MIN) & (elev_range <= PASS_RUGG_MAX),
            p.harpy_peaks,
        )
        river_mask = river_traversed_faces(rivers, n_faces=n)
        _add(river_mask & ~is_water, p.troll_holds)
        _add(elev_mean >= GIANT_ELEV_MIN, p.giant_lairs)

        _report("Danger: forest / wilderness sources…")
        _add((forest_density_mean >= GOBLIN_FOREST_MIN) & ~is_ocean, p.goblin_warbands)
        grassland_mask = np.isin(biome, list(GRASSLAND_BIOMES))
        _add((forest_density_mean >= WEREWOLF_FOREST_MIN) & grassland_mask, p.werewolf_ranges)
        scrub_mask = np.isin(biome, list(SCRUB_BIOMES))
        _add(scrub_mask & ~is_ocean, p.basilisk_territories)
        _add(
            (forest_density_mean >= CURSED_FOREST_DENSITY_MIN) & (water_hops >= CURSED_FOREST_ISOLATION_MIN),
            p.cursed_forests,
        )

        _report("Danger: planar intrusions…")
        _add(fire_volcanic_mask, p.fire_rifts)
        active_boundary = np.isin(boundary_type, list(ACTIVE_BOUNDARY_TYPES))
        continental = np.isin(plate_type, list(CONTINENTAL_TYPES))
        _add(active_boundary & continental, p.earth_rifts)
        _add(elev_mean >= AIR_RIFT_ELEV_MIN, p.air_rifts)
        _add(is_ocean & (ocean_depth_hops >= WATER_RIFT_DEPTH_MIN), p.water_rifts)

        _report("Danger: undead / necromantic sources…")
        _add(
            (elev_mean >= LICH_ELEV_MIN)
            & (fertility_mean <= LICH_FERTILITY_MAX)
            & (forest_density_mean <= LICH_FOREST_MAX),
            p.lich_strongholds,
        )
        _add(
            (elev_mean <= BARROW_ELEV_MAX)
            & (elev_range <= BARROW_RUGG_MAX)
            & (water_hops >= BARROW_ISOLATION_MIN)
            & ~is_ocean,
            p.barrow_fields,
        )
        _add(
            (coast_hops <= HAUNT_COAST_MAX)
            & (fertility_mean <= HAUNT_FERTILITY_MAX)
            & ~is_ocean,
            p.haunt_concentrations,
        )

        danger = np.clip(danger, 0.0, 1.0).astype(np.float32)
        self.danger = danger

        DANGER.write(mesh, danger)
        DANGER_ARRAY.write(ctx.world, danger)

        # Build per-seed positions + kinds array for the overlay.
        centroids = mesh.face_centroids.astype(np.float32)
        if _seed_records:
            rec_faces = np.array([f for f, _ in _seed_records], dtype=np.intp)
            rec_kinds = np.array([k for _, k in _seed_records], dtype=np.int8)
            positions = centroids[rec_faces]
        else:
            positions = np.empty((0, 3), dtype=np.float32)
            rec_kinds = np.empty(0, dtype=np.int8)

        self.sources = (positions, rec_kinds)
        ctx.world.add_feature("danger_sources", self.sources)

        # Write source names into face metadata for hover.
        for face_idx, kind in _seed_records:
            meta = mesh.face_metadata[face_idx]
            srcs = meta.get("danger_sources")
            if srcs is None:
                meta["danger_sources"] = [SOURCE_NAMES[kind]]
            elif SOURCE_NAMES[kind] not in srcs:
                srcs.append(SOURCE_NAMES[kind])

        _report("Danger done.")

    # ── Display ───────────────────────────────────────────────────────────────

    def register_display(
        self, world: World, reg: DisplayRegistry, *, debug: bool = False
    ) -> None:
        if not debug:
            return
        from worldgen.danger.display import danger_color, danger_legend_entries

        reg.add_scheme(
            "Danger", danger_color,
            legend=("gradient", danger_legend_entries()),
        )

        if world.has_feature("danger_sources"):
            reg.add_overlay(
                "danger_sources", "Danger Sources",
                legend=("categorical", _SOURCE_CAT_LEGEND),
            )

        mesh = world.mesh

        def _danger_hover(face: int, _uv, _w) -> list:
            meta = mesh.face_metadata[face]
            parts: list = []
            if "danger" in meta:
                parts.append(f"Danger: {meta['danger']:.2f}")
            for src in meta.get("danger_sources", []):
                parts.append(src)
            return parts

        reg.add_hover(_danger_hover)

    # ── Persistence ───────────────────────────────────────────────────────────

    def serialize(self) -> dict:
        arrays: dict = {}
        if self.danger is not None:
            arrays["danger"] = self.danger.astype(np.float32)
        if self.sources is not None:
            positions, kinds = self.sources
            if len(positions):
                arrays["danger_source_positions"] = positions.astype(np.float32)
                arrays["danger_source_kinds"] = kinds.astype(np.int8)
        return arrays

    def load(self, arrays: Mapping[str, np.ndarray], ctx: BuildContext) -> None:
        if "danger" in arrays:
            danger = arrays["danger"].astype(np.float32)
            self.danger = danger
            DANGER.write(ctx.mesh, danger)
            DANGER_ARRAY.write(ctx.world, danger)

        if "danger_source_positions" in arrays:
            positions = arrays["danger_source_positions"].astype(np.float32)
            kinds = arrays["danger_source_kinds"].astype(np.int8)
            self.sources = (positions, kinds)
            ctx.world.add_feature("danger_sources", self.sources)

            mesh = ctx.mesh
            for pos, kind in zip(positions, kinds):
                face_idx = _pos_to_face(mesh, pos)
                if face_idx is None:
                    continue
                meta = mesh.face_metadata[face_idx]
                srcs = meta.get("danger_sources")
                name = SOURCE_NAMES[int(kind)]
                if srcs is None:
                    meta["danger_sources"] = [name]
                elif name not in srcs:
                    srcs.append(name)
