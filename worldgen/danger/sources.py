"""Pure placement helpers and threshold constants for the danger stage."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from core.mesh import PolygonMesh
from worldgen.resources.terrain import face_hops_from_seeds

if TYPE_CHECKING:
    from worldgen.danger.builder import DangerSourceParams

# ── Volcanic / Tectonic ───────────────────────────────────────────────────────
DRAGON_VOLCANIC_MIN: float = 0.6
FIRE_RIFT_VOLCANIC_MIN: float = 0.7
SALAMANDER_VOLCANIC_MIN: float = 0.3

# ── Coastal / Ocean ───────────────────────────────────────────────────────────
KRAKEN_DEPTH_MIN: int = 3
DEADWATER_DEPTH_MIN: int = 2
SEA_SERPENT_COAST_MAX: int = 2

# ── Highland / Pass ───────────────────────────────────────────────────────────
HARPY_ELEV_MIN: float = 2.0
PASS_RUGG_MIN: float = 0.5
PASS_RUGG_MAX: float = 2.5
GIANT_ELEV_MIN: float = 1.5

# ── Forest / Wilderness ───────────────────────────────────────────────────────
GOBLIN_FOREST_MIN: float = 0.4
WEREWOLF_FOREST_MIN: float = 0.3
CURSED_FOREST_DENSITY_MIN: float = 0.6
CURSED_FOREST_ISOLATION_MIN: int = 3

# ── Planar Intrusions ─────────────────────────────────────────────────────────
AIR_RIFT_ELEV_MIN: float = 3.0
WATER_RIFT_DEPTH_MIN: int = 2

# ── Undead / Necromantic ──────────────────────────────────────────────────────
LICH_ELEV_MIN: float = 1.5
LICH_FERTILITY_MAX: float = 0.25
LICH_FOREST_MAX: float = 0.2
BARROW_ELEV_MAX: float = 0.5
BARROW_RUGG_MAX: float = 0.5
BARROW_ISOLATION_MIN: int = 2
HAUNT_COAST_MAX: int = 2
HAUNT_FERTILITY_MAX: float = 0.3

# ── Biome / boundary classification sets ─────────────────────────────────────
# Biome indices from worldgen.climate.classification:
#   Steppe=4, Savannah=5
GRASSLAND_BIOMES: frozenset[int] = frozenset({4, 5})
# Semi-Arid=7, Desert=8
SCRUB_BIOMES: frozenset[int] = frozenset({7, 8})

ACTIVE_BOUNDARY_TYPES: frozenset[str] = frozenset({"convergent", "divergent"})
CONTINENTAL_TYPES: frozenset[str] = frozenset({"continental"})


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
    params: "DangerSourceParams",
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
