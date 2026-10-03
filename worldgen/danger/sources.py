"""Danger source definitions — params, terrain container, and source registry."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TYPE_CHECKING

import numpy as np

from core.mesh import PolygonMesh
from core.params import field, param_field as _f, paramclass
from worldgen.resources.terrain import face_hops_from_seeds

if TYPE_CHECKING:
    pass  # keep for future TYPE_CHECKING imports


# ── Per-source tunable params ─────────────────────────────────────────────────


@paramclass
class DangerSourceParams:
    count: int = field(default=4, metadata={"label": "Count", "min": 0, "max": 50})
    hop_radius: int = field(default=1, metadata={"label": "Hop radius", "min": 0, "max": 10})
    intensity: float = _f(0.7, label="Intensity", min=0.0, max=1.0)
    decay_per_hop: float = _f(0.3, label="Decay/hop", min=0.0, max=3.0)
    enabled: bool = field(default=True, metadata={"label": "Enabled"})


# ── Threshold constants ───────────────────────────────────────────────────────

# Volcanic / Tectonic
FIRE_RIFT_VOLCANIC_MIN: float = 0.7

# Coastal / Ocean
KRAKEN_DEPTH_MIN: int = 3
DEADWATER_DEPTH_MIN: int = 2
SEA_SERPENT_COAST_MAX: int = 2

# Highland / Pass
HARPY_ELEV_MIN: float = 2.0
PASS_RUGG_MIN: float = 0.5
PASS_RUGG_MAX: float = 2.5
GIANT_ELEV_MIN: float = 1.5

# Forest / Wilderness
GOBLIN_FOREST_MIN: float = 0.4
WEREWOLF_FOREST_MIN: float = 0.3
CURSED_FOREST_DENSITY_MIN: float = 0.6
CURSED_FOREST_ISOLATION_MIN: int = 3
LIZARDFOLK_MARSH_MIN: float = 0.05

# Planar Intrusions
AIR_RIFT_ELEV_MIN: float = 3.0
WATER_RIFT_DEPTH_MIN: int = 2

# Undead / Necromantic
LICH_ELEV_MIN: float = 1.5
LICH_FERTILITY_MAX: float = 0.25
LICH_FOREST_MAX: float = 0.2
BARROW_ELEV_MAX: float = 0.5
BARROW_RUGG_MAX: float = 0.5
BARROW_ISOLATION_MIN: int = 2
HAUNT_COAST_MAX: int = 2
HAUNT_FERTILITY_MAX: float = 0.3

# Biome / boundary classification sets
# Biome indices from worldgen.climate.classification:
#   Steppe=4, Savannah=5
GRASSLAND_BIOMES: frozenset[int] = frozenset({4, 5})
# Semi-Arid=7, Desert=8
SCRUB_BIOMES: frozenset[int] = frozenset({7, 8})

ACTIVE_BOUNDARY_TYPES: frozenset[str] = frozenset({"convergent", "divergent"})
CONTINENTAL_TYPES: frozenset[str] = frozenset({"continental"})

# Aggregate BFS search bounds for builder.py — include the slack margin.
MAX_OCEAN_DEPTH_HOPS: int = max(KRAKEN_DEPTH_MIN, DEADWATER_DEPTH_MIN, WATER_RIFT_DEPTH_MIN) + 2
MAX_COAST_HOPS: int = max(SEA_SERPENT_COAST_MAX, HAUNT_COAST_MAX) + 1
MAX_WATER_HOPS: int = max(CURSED_FOREST_ISOLATION_MIN, BARROW_ISOLATION_MIN) + 1


# ── All-source params ─────────────────────────────────────────────────────────


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
    lizardfolk_fens: DangerSourceParams = field(
        default_factory=lambda: DangerSourceParams(count=4, hop_radius=1, intensity=0.6, decay_per_hop=0.3),
        metadata={"label": "Lizardfolk Fens"},
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


# ── Category colors ───────────────────────────────────────────────────────────

_C_VOL = (0.95, 0.25, 0.05, 1.0)   # volcanic
_C_SEA = (0.10, 0.35, 0.80, 1.0)   # ocean / coastal
_C_HIG = (0.60, 0.45, 0.20, 1.0)   # highland
_C_FOR = (0.15, 0.50, 0.10, 1.0)   # forest / wilderness
_C_PLN = (0.70, 0.10, 0.85, 1.0)   # planar rift
_C_UND = (0.35, 0.05, 0.45, 1.0)   # undead / necromantic

SOURCE_CAT_LEGEND: list[tuple[str, tuple]] = [
    ("Volcanic",    _C_VOL),
    ("Ocean",       _C_SEA),
    ("Highland",    _C_HIG),
    ("Forest",      _C_FOR),
    ("Planar Rift", _C_PLN),
    ("Undead",      _C_UND),
]


# ── Terrain container ─────────────────────────────────────────────────────────


@dataclass
class DangerTerrain:
    """All derived terrain inputs consumed by danger source mask functions."""
    n: int
    is_water: np.ndarray
    is_ocean: np.ndarray
    elev_mean: np.ndarray
    elev_range: np.ndarray
    volcanic_activity: np.ndarray
    biome: np.ndarray
    fertility_mean: np.ndarray
    forest_density_mean: np.ndarray
    ocean_depth_hops: np.ndarray
    coast_hops: np.ndarray
    water_hops: np.ndarray
    river_mask: np.ndarray
    boundary_type: np.ndarray
    plate_type: np.ndarray
    # Share of the face in a river delta's marsh (0 without the delta stage).
    marsh_fraction: np.ndarray | None = None


def _marsh(t: DangerTerrain) -> np.ndarray:
    """Faces with a river delta's marsh on them."""
    if t.marsh_fraction is None:
        return np.zeros(t.n, dtype=bool)
    return np.asarray(t.marsh_fraction) >= LIZARDFOLK_MARSH_MIN


# ── DangerSource and registry ─────────────────────────────────────────────────

_MaskFn = Callable[["DangerTerrain"], np.ndarray]
_ParamFn = Callable[["DangerParams"], DangerSourceParams]


class DangerSource:
    """One danger threat: display metadata, tunable params, and a mask function."""

    __slots__ = ("name", "color", "params", "_mask_fn")

    def __init__(
        self,
        name: str,
        color: tuple,
        params: DangerSourceParams,
        mask_fn: _MaskFn,
    ) -> None:
        self.name = name
        self.color = color
        self.params = params
        self._mask_fn = mask_fn

    def mask(self, terrain: DangerTerrain) -> np.ndarray:
        return self._mask_fn(terrain)


# Master list: (name, color, params_getter, mask_fn) — order defines source indices.
_SOURCE_DEFS: list[tuple[str, tuple, _ParamFn, _MaskFn]] = [
    # Volcanic / Tectonic
    ("Dragon Lair",           _C_VOL, lambda p: p.dragon_lairs,
        lambda t: ~t.is_water),
    ("Fire Elemental Rift",   _C_VOL, lambda p: p.fire_elemental_rifts,
        lambda t: (t.volcanic_activity >= FIRE_RIFT_VOLCANIC_MIN) & ~t.is_water),
    # Coastal / Ocean
    ("Kraken Domain",         _C_SEA, lambda p: p.kraken_domains,
        lambda t: t.is_ocean & (t.ocean_depth_hops >= KRAKEN_DEPTH_MIN)),
    ("Deadwater Zone",        _C_SEA, lambda p: p.deadwater_zones,
        lambda t: t.is_ocean & (t.ocean_depth_hops >= DEADWATER_DEPTH_MIN)),
    ("Sea Serpent Territory", _C_SEA, lambda p: p.sea_serpent_territories,
        lambda t: t.is_ocean & (t.coast_hops <= SEA_SERPENT_COAST_MAX)),
    # Highland / Pass
    ("Harpy Peak",            _C_HIG, lambda p: p.harpy_peaks,
        lambda t: (
            (t.elev_mean >= HARPY_ELEV_MIN) & (t.elev_range >= PASS_RUGG_MIN)
            & (t.elev_range <= PASS_RUGG_MAX) & ~t.is_water)),
    ("Troll Hold",            _C_HIG, lambda p: p.troll_holds,
        lambda t: t.river_mask & ~t.is_water),
    ("Giant Lair",            _C_HIG, lambda p: p.giant_lairs,
        lambda t: (t.elev_mean >= GIANT_ELEV_MIN) & ~t.is_water),
    # Forest / Wilderness
    ("Goblin Warband",        _C_FOR, lambda p: p.goblin_warbands,
        lambda t: (t.forest_density_mean >= GOBLIN_FOREST_MIN) & ~t.is_water),
    ("Werewolf Range",        _C_FOR, lambda p: p.werewolf_ranges,
        lambda t: (
            (t.forest_density_mean >= WEREWOLF_FOREST_MIN)
            & np.isin(t.biome, list(GRASSLAND_BIOMES)) & ~t.is_water)),
    ("Basilisk Territory",    _C_FOR, lambda p: p.basilisk_territories,
        lambda t: np.isin(t.biome, list(SCRUB_BIOMES)) & ~t.is_water),
    ("Cursed Forest",         _C_FOR, lambda p: p.cursed_forests,
        lambda t: (
            (t.forest_density_mean >= CURSED_FOREST_DENSITY_MIN)
            & (t.water_hops >= CURSED_FOREST_ISOLATION_MIN) & ~t.is_water)),
    # Planar Intrusions
    ("Fire Rift",             _C_PLN, lambda p: p.fire_rifts,
        lambda t: t.volcanic_activity >= FIRE_RIFT_VOLCANIC_MIN),
    ("Earth Rift",            _C_PLN, lambda p: p.earth_rifts,
        lambda t: (
            np.isin(t.boundary_type, list(ACTIVE_BOUNDARY_TYPES))
            & np.isin(t.plate_type, list(CONTINENTAL_TYPES)))),
    ("Air Rift",              _C_PLN, lambda p: p.air_rifts,
        lambda t: t.elev_mean >= AIR_RIFT_ELEV_MIN),
    ("Water Rift",            _C_PLN, lambda p: p.water_rifts,
        lambda t: t.is_ocean & (t.ocean_depth_hops >= WATER_RIFT_DEPTH_MIN)),
    # Undead / Necromantic
    ("Lich Stronghold",       _C_UND, lambda p: p.lich_strongholds,
        lambda t: (
            (t.elev_mean >= LICH_ELEV_MIN) & (t.fertility_mean <= LICH_FERTILITY_MAX)
            & (t.forest_density_mean <= LICH_FOREST_MAX))),
    ("Barrow Field",          _C_UND, lambda p: p.barrow_fields,
        lambda t: (
            (t.elev_mean <= BARROW_ELEV_MAX) & (t.elev_range <= BARROW_RUGG_MAX)
            & (t.water_hops >= BARROW_ISOLATION_MIN) & ~t.is_water)),
    ("Haunt",                 _C_UND, lambda p: p.haunt_concentrations,
        lambda t: (t.coast_hops <= HAUNT_COAST_MAX) & (t.fertility_mean <= HAUNT_FERTILITY_MAX) & ~t.is_water),
    # Wilderness, appended last so the other sources keep their kind indices and draws.
    ("Lizardfolk Fen",        _C_FOR, lambda p: p.lizardfolk_fens,
        lambda t: _marsh(t) & ~t.is_water),
]

SOURCE_NAMES: list[str] = [d[0] for d in _SOURCE_DEFS]
SOURCE_COLORS: list[tuple] = [d[1] for d in _SOURCE_DEFS]


def build_sources(params: DangerParams) -> list[DangerSource]:
    """Instantiate all danger sources with resolved params."""
    return [
        DangerSource(name, color, get_sp(params), mask_fn)
        for name, color, get_sp, mask_fn in _SOURCE_DEFS
    ]


# ── Placement helpers ─────────────────────────────────────────────────────────


def coastal_land_faces(mesh: PolygonMesh, is_ocean: np.ndarray) -> np.ndarray:
    """Land faces that border at least one ocean face — seeds for coast BFS."""
    n = len(mesh.faces)
    mask = np.zeros(n, dtype=bool)
    for i in range(n):
        if is_ocean[i]:
            continue
        for nb in mesh.face_adjacency[i]:
            if is_ocean[nb]:
                mask[i] = True
                break
    return mask


def source_contribution(
    mesh: PolygonMesh,
    rng: np.random.Generator,
    candidate_mask: np.ndarray,
    params: DangerSourceParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Pick seed faces from candidates and compute radial falloff contribution.

    Returns ``(contrib, seeds)`` where ``contrib`` is a (N,) float32 danger
    contribution array and ``seeds`` is a 1-D int array of chosen face indices.
    """
    n_faces = len(mesh.faces)
    candidates = np.flatnonzero(candidate_mask)
    if candidates.size == 0:
        return np.zeros(n_faces, dtype=np.float32), np.empty(0, dtype=np.intp)

    n = min(params.count, candidates.size)
    seeds = rng.choice(candidates, size=n, replace=False)

    seed_mask = np.zeros(n_faces, dtype=bool)
    seed_mask[seeds] = True

    hops = face_hops_from_seeds(mesh, seed_mask, max_hops=params.hop_radius)
    contrib = np.where(
        hops <= params.hop_radius,
        params.intensity * np.exp(-hops.astype(np.float32) * params.decay_per_hop),
        0.0,
    )
    return contrib.astype(np.float32), seeds
