"""Polity stage — territory growth and border-tension scoring.

Params and stage for the fourth phase of the realm pipeline.  Runs after
population & calibration so it can seed polities from the final, populated
settlements.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from core.display import DisplayRegistry
from core.params import param_field as _f, paramclass
from core.pipeline import BuildContext, Stage
from realmgen.channels import BORDER_TENSION, POLITY_ID, POPULATION, TRADE_FLOW
from realmgen.placement.seeder import build_cost_graph
from realmgen.placement.types import Settlement
from realmgen.polity.formation import (
    compute_border_tension,
    compute_polity_power,
    grow_territory,
    seed_capitals,
)
from realmgen.polity.types import POLITY_COLORS, Polity, UNCLAIMED_COLOR
from viz.color_stops import interp_stops_scalar
from worldgen.cost.channels import COST_SURFACES, IS_WATER
from worldgen.geology.display import OCEAN_COLOR


# ── Params ────────────────────────────────────────────────────────────────────


@paramclass
class PolityParams:
    capital_min_separation: float = _f(
        60.0, label="Capital min separation (cost-dist)", min=5.0, max=400.0
    )
    capital_min_pop_fraction: float = _f(
        0.15, label="Capital min pop fraction", min=0.0, max=1.0
    )
    power_bias: float = _f(
        25.0, label="Power territory bias (cost-dist / e-fold)", min=0.0, max=200.0
    )
    growth_max_reach: float = _f(
        250.0, label="Territory max reach (cost-dist)", min=10.0, max=1000.0
    )
    tension_barrier_scale: float = _f(
        3.0, label="Tension barrier scale (edge cost)", min=0.1, max=50.0
    )
    tension_flow_ref: float = _f(
        0.4, label="Tension flow reference (norm. to busy-flow)", min=0.01, max=5.0
    )
    tension_artery_weight: float = _f(
        0.6, label="Tension trade weight (rest is baseline)", min=0.0, max=1.0
    )


# ── Display ───────────────────────────────────────────────────────────────────

_TENSION_STOPS = [
    (0.0, (0.30, 0.33, 0.36, 1.0)),  # calm interior — slate
    (0.4, (0.85, 0.70, 0.25, 1.0)),  # warming — amber
    (1.0, (0.90, 0.20, 0.15, 1.0)),  # contested — red
]


def _polity_color(meta: dict) -> tuple:
    if meta.get("is_water", meta.get("is_ocean", False)):
        return OCEAN_COLOR
    pid = meta.get("polity_id", -1)
    if pid is None or int(pid) < 0:
        return UNCLAIMED_COLOR
    return POLITY_COLORS[int(pid) % len(POLITY_COLORS)]


def _tension_color(meta: dict) -> tuple:
    if meta.get("is_water", meta.get("is_ocean", False)):
        return OCEAN_COLOR
    t = min(max(float(meta.get("border_tension", 0.0)), 0.0), 1.0)
    return interp_stops_scalar(t, _TENSION_STOPS)


# ── PolityStage ───────────────────────────────────────────────────────────────


class PolityStage(Stage):
    """Grows territorial polities and scores inter-polity border tension."""

    name = "realm_polity"
    requires = (POPULATION, IS_WATER, COST_SURFACES)
    provides = (POLITY_ID, BORDER_TENSION)

    def __init__(self, params: PolityParams) -> None:
        self.params = params
        self.polities: Optional[List[Polity]] = None
        self.polity_of_face: Optional[np.ndarray] = None
        self.face_tension: Optional[np.ndarray] = None
        self.pair_tension: Optional[dict] = None

    def build(self, ctx: BuildContext) -> None:
        world = ctx.world
        mesh = ctx.mesh
        n = len(mesh.faces)
        p = self.params

        is_water = IS_WATER.read(mesh)
        centroids = np.asarray(mesh.face_centroids, dtype=np.float64)

        settlements: List[Settlement] = list(world.get_feature("settlements") or [])
        if not settlements:
            ctx.report("No settlements — skipping polity formation.")
            self._finalize(mesh, world, [], np.full(n, -1, dtype=np.int32),
                           np.zeros(n, dtype=np.float64), {})
            return

        # ── Land-only cost graph ─────────────────────────────────────────────
        # Polities grow over land only (design: ocean is impassable, so
        # coastlines are natural borders and islands stay outside territory). The
        # land surface treats water as impassable (``land_passable == ~is_water``),
        # so ``build_land_edge_costs`` already gives every water-incident edge inf.
        if COST_SURFACES.present(world):
            from worldgen.cost.surfaces import build_land_edge_costs
            cs = COST_SURFACES.read(world)
            edge_costs = build_land_edge_costs(
                mesh,
                cs.land_base_cost,
                cs.land_passable,
                cs.river_flow_dir,
                cs.river_flow_mag,
                cs.river_navigability,
            )
        else:
            base = np.where(is_water, np.inf, 1.5)
            edge_costs = [
                np.array([0.5 * (base[i] + base[j]) for j in adj], dtype=np.float64)
                for i, adj in enumerate(mesh.face_adjacency)
            ]
        cost_graph = build_cost_graph(mesh.face_adjacency, edge_costs)

        settlement_faces = [s.face_id for s in settlements]
        populations = np.array([s.population for s in settlements], dtype=np.float64)

        # ── Seed capitals ─────────────────────────────────────────────────────
        ctx.report("Seeding capitals…")
        capital_idx = seed_capitals(
            settlement_faces, populations, cost_graph,
            min_separation=p.capital_min_separation,
            min_pop_fraction=p.capital_min_pop_fraction,
        )
        if not capital_idx:
            ctx.report("No capitals qualified — skipping polity formation.")
            self._finalize(mesh, world, [], np.full(n, -1, dtype=np.int32),
                           np.zeros(n, dtype=np.float64), {})
            return

        capital_faces = [settlement_faces[k] for k in capital_idx]
        capital_powers = populations[capital_idx]

        # ── Grow territory ────────────────────────────────────────────────────
        ctx.report(f"Growing {len(capital_faces)} polities…")
        polity_of_face = grow_territory(
            n, capital_faces, capital_powers, cost_graph,
            power_bias=p.power_bias, max_reach=p.growth_max_reach,
        )

        # ── Power + tension ───────────────────────────────────────────────────
        n_polities = len(capital_faces)
        polity_power = compute_polity_power(
            polity_of_face, settlement_faces, populations, n_polities,
        )

        face_flow = (
            TRADE_FLOW.read(mesh).astype(np.float64)
            if TRADE_FLOW.present(mesh)
            else np.zeros(n, dtype=np.float64)
        )
        # Normalise flow by a robust "busy-flow" reference (90th percentile of
        # positive flow) rather than the global max — the max is dominated by the
        # single largest hub and would make every border crossing read as quiet.
        pos = face_flow[face_flow > 0.0]
        flow_ref = float(np.quantile(pos, 0.90)) if pos.size > 0 else 0.0
        norm_flow = face_flow / flow_ref if flow_ref > 0.0 else face_flow
        pair_tension, face_tension = compute_border_tension(
            mesh.face_adjacency, polity_of_face, polity_power,
            edge_costs, norm_flow,
            barrier_scale=p.tension_barrier_scale,
            flow_ref=p.tension_flow_ref,
            artery_weight=p.tension_artery_weight,
        )

        # ── Assemble Polity objects ───────────────────────────────────────────
        members: List[List[int]] = [[] for _ in range(n_polities)]
        for s in settlements:
            pid = int(polity_of_face[s.face_id])
            if pid >= 0:
                members[pid].append(s.face_id)
        face_counts = np.bincount(
            polity_of_face[polity_of_face >= 0], minlength=n_polities
        )
        polities = [
            Polity(
                polity_id=pid,
                capital_face=capital_faces[pid],
                capital_position=centroids[capital_faces[pid]].copy(),
                power=float(polity_power[pid]),
                n_faces=int(face_counts[pid]),
                member_settlement_faces=members[pid],
            )
            for pid in range(n_polities)
        ]

        self._finalize(mesh, world, polities, polity_of_face, face_tension, pair_tension)
        ctx.report(
            f"Polity formation done: {n_polities} polities, "
            f"{len(pair_tension)} shared borders."
        )

    def _finalize(self, mesh, world, polities, polity_of_face, face_tension, pair_tension):
        self.polities = polities
        self.polity_of_face = polity_of_face
        self.face_tension = face_tension
        self.pair_tension = pair_tension

        POLITY_ID.write(mesh, polity_of_face)
        BORDER_TENSION.write(mesh, face_tension)
        world.add_feature("polities", polities)
        # Tensions as a sorted list of (a, b, score) for inspection / Stage 6.
        world.add_feature(
            "polity_tensions",
            [(a, b, float(t)) for (a, b), t in sorted(pair_tension.items())],
        )

    def register_display(self, world, reg: DisplayRegistry, *, debug: bool = False) -> None:
        if self.polities is not None and len(self.polities) > 0:
            reg.add_scheme("Polities", _polity_color)
        if self.face_tension is not None:
            reg.add_scheme("Border Tension", _tension_color)
