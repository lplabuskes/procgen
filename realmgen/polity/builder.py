"""Polity stage — seeds capitals, runs the forest algorithm, scores tension.

The tree model (``docs/plans/polity_generation_brief.md``, Stages 0-3): seed
sovereign capitals, project authority over the land + sea cost surfaces, claim
faces, classify each settlement direct/vassal/independent, recurse for vassal
sub-territories. The heavy lifting lives in :mod:`realmgen.polity.forest`; this
stage handles IO, cost graphs, border tension, and display.

Runs last in :class:`~realmgen.builder.RealmPipeline`.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from core.display import DisplayRegistry
from core.params import field, param_field as _f, paramclass
from core.pipeline import BuildContext, Stage
from realmgen.channels import (
    BORDER_TENSION,
    CONTROL_CLASS,
    POLITY_ID,
    POPULATION,
    SITUATIONAL_VALUE,
    SOVEREIGN_ID,
    TRADE_FLOW,
)
from realmgen.placement.seeder import build_cost_graph
from realmgen.polity import display as disp
from realmgen.polity.forest import (
    ClaimingParams,
    ForestBuilder,
    HierarchyParams,
    ProjectionParams,
    _Ctx,
)
from realmgen.polity.seeding import compute_defensibility, seed_capitals
from realmgen.polity.tension import compute_border_tension
from realmgen.polity.types import Polity
from realmgen.trade.network import build_base_hybrid_graph
from worldgen.cost.channels import COST_SURFACES, IS_WATER


@paramclass
class CapitalSeedingParams:
    """Stage 0 — which settlements found sovereign polities (consumed here)."""
    min_separation: float = _f(
        20.0, label="Capital min separation (cost-dist)", min=5.0, max=400.0
    )
    min_pop_fraction: float = _f(0.1, label="Capital min pop fraction", min=0.0, max=1.0)
    trade_weight: float = _f(1.0, label="Capital seed trade weight", min=0.0, max=5.0)


@paramclass
class DefensibilityParams:
    """Terrain-derived local-power multiplier (consumed here)."""
    rugged_weight: float = _f(0.6, label="Defensibility: ruggedness", min=0.0, max=3.0)
    situational_weight: float = _f(0.8, label="Defensibility: situational", min=0.0, max=3.0)


@paramclass
class BorderTensionParams:
    """Inter-polity border-crossing tension on the sovereign partition (consumed here)."""
    barrier_scale: float = _f(3.0, label="Tension barrier scale (edge cost)", min=0.1, max=50.0)
    flow_ref: float = _f(0.4, label="Tension flow reference (norm. to busy-flow)", min=0.01, max=5.0)
    artery_weight: float = _f(0.6, label="Tension trade weight (rest is baseline)", min=0.0, max=1.0)


@paramclass
class PolityParams:
    seeding: CapitalSeedingParams = field(
        default_factory=CapitalSeedingParams, metadata={"label": "Capital seeding"}
    )
    defensibility: DefensibilityParams = field(
        default_factory=DefensibilityParams, metadata={"label": "Defensibility"}
    )
    projection: ProjectionParams = field(
        default_factory=ProjectionParams, metadata={"label": "Authority projection"}
    )
    claiming: ClaimingParams = field(
        default_factory=ClaimingParams, metadata={"label": "Claiming / competition"}
    )
    hierarchy: HierarchyParams = field(
        default_factory=HierarchyParams, metadata={"label": "Hierarchy classification"}
    )
    tension: BorderTensionParams = field(
        default_factory=BorderTensionParams, metadata={"label": "Border tension"}
    )


class PolityStage(Stage):
    """Authority-field territory growth and tree (sovereign/vassal) classification."""

    name = "realm_polity"
    requires = (POPULATION, SITUATIONAL_VALUE, IS_WATER, COST_SURFACES)
    provides = (POLITY_ID, SOVEREIGN_ID, CONTROL_CLASS, BORDER_TENSION)

    def __init__(self, params: PolityParams, danger_params=None) -> None:
        self.params = params
        self.danger_params = danger_params
        self.polities: Optional[List[Polity]] = None
        self.polity_of_face: Optional[np.ndarray] = None
        self.sovereign_of_face: Optional[np.ndarray] = None
        self.control_of_face: Optional[np.ndarray] = None
        self.authority_of_face: Optional[np.ndarray] = None
        self.face_tension: Optional[np.ndarray] = None
        self.pair_tension: Optional[dict] = None

    def build(self, ctx: BuildContext) -> None:
        world, mesh = ctx.world, ctx.mesh
        n = len(mesh.faces)
        p = self.params

        is_water = np.asarray(IS_WATER.read(mesh), dtype=bool)
        land_mask = ~is_water

        settlements = list(world.get_feature("settlements") or [])
        if not settlements:
            ctx.report("No settlements — skipping polity formation.")
            self._finalize_empty(mesh, world)
            return

        settlement_faces = np.array([s.face_id for s in settlements], dtype=np.int64)
        populations = np.array([s.population for s in settlements], dtype=np.float64)
        situational = (
            SITUATIONAL_VALUE.read(mesh).astype(np.float64)
            if SITUATIONAL_VALUE.present(mesh) else np.zeros(n)
        )
        trade_flow = (
            TRADE_FLOW.read(mesh).astype(np.float64)
            if TRADE_FLOW.present(mesh) else np.zeros(n)
        )
        trade_wealth = trade_flow[settlement_faces]

        danger, danger_scale = self._read_danger(world, n)
        land_base_cost, land_edge_costs, hybrid_graph = self._build_cost_graphs(
            mesh, world, is_water, danger, danger_scale,
        )
        land_graph = build_cost_graph(mesh.face_adjacency, land_edge_costs)
        defensibility = compute_defensibility(
            land_base_cost, situational, land_mask,
            rugged_weight=p.defensibility.rugged_weight,
            situational_weight=p.defensibility.situational_weight,
        )

        # Stage 0 — seed sovereign capitals (weighted).
        trade_ref = float(trade_wealth.max())
        trade_norm = trade_wealth / trade_ref if trade_ref > 0.0 else np.zeros_like(trade_wealth)
        seed_weights = (
            populations * (1.0 + p.seeding.trade_weight * trade_norm) * defensibility[settlement_faces]
        )
        ctx.report("Seeding sovereign capitals…")
        capital_sidx = seed_capitals(
            settlement_faces.tolist(), populations, land_graph, weights=seed_weights,
            min_separation=p.seeding.min_separation, min_pop_fraction=p.seeding.min_pop_fraction,
        )
        if not capital_sidx:
            ctx.report("No capitals qualified — skipping polity formation.")
            self._finalize_empty(mesh, world)
            return

        sctx = _Ctx(
            n=n, centroids=np.asarray(mesh.face_centroids, dtype=np.float64),
            settlement_faces=settlement_faces, populations=populations,
            pop_norm=populations / float(populations.max()), trade_norm=trade_norm,
            defensibility=defensibility, land_graph=land_graph, hybrid_graph=hybrid_graph,
            base_seed=int(world.seed) if getattr(world, "seed", None) is not None else 0, p=p,
            face_to_settlement={int(f): k for k, f in enumerate(settlement_faces.tolist())},
        )

        # Stages 1-3 — grow the forest.
        forest = ForestBuilder(sctx)
        forest.build(capital_sidx, land_mask, ctx.report)
        self.polities = forest.polities
        self.polity_of_face = forest.polity_of_face
        self.sovereign_of_face = forest.sovereign_of_face
        self.control_of_face = forest.control_of_face
        self.authority_of_face = forest.authority_of_face

        self._compute_tension(mesh, land_edge_costs, trade_flow)
        self._write_outputs(mesh, world)

        n_sov = sum(1 for pol in self.polities if pol.liege is None)
        ctx.report(
            f"Polity formation done: {n_sov} sovereigns, "
            f"{len(self.polities) - n_sov} vassals, {len(self.pair_tension)} shared borders."
        )
        self._dump_tree(ctx)

    # ── inputs ────────────────────────────────────────────────────────────────

    def _read_danger(self, world, n):
        danger = np.zeros(n, dtype=np.float32)
        scale = 0.0
        if self.danger_params is not None:
            from worldgen.danger.channels import DANGER_ARRAY
            if DANGER_ARRAY.present(world):
                danger = DANGER_ARRAY.read(world).astype(np.float32)
            scale = float(self.danger_params.movement_scale)
        return danger, scale

    def _build_cost_graphs(self, mesh, world, is_water, danger, danger_scale):
        """Return ``(land_base_cost (N,), land_edge_costs, hybrid_graph or None)``."""
        n = len(mesh.faces)
        if COST_SURFACES.present(world):
            from worldgen.cost.surfaces import build_land_edge_costs, build_sea_edge_costs
            cs = COST_SURFACES.read(world)
            land_base_cost = np.asarray(cs.land_base_cost, dtype=np.float64) + danger_scale * danger
            land_edge_costs = build_land_edge_costs(
                mesh, land_base_cost, cs.land_passable,
                cs.river_flow_dir, cs.river_flow_mag, cs.river_navigability,
            )
            sea_edge_costs = build_sea_edge_costs(
                mesh, cs.sea_base_cost, cs.sea_passable, wind_dir=cs.wind_dir,
            )
            hybrid_graph = build_base_hybrid_graph(
                n, mesh.face_adjacency,
                land_edge_costs, cs.land_passable, sea_edge_costs, cs.sea_passable,
            )
            return land_base_cost, land_edge_costs, hybrid_graph

        # Fallback when no cost surfaces were built (land-only, no sea channel).
        base = np.where(is_water, np.inf, 1.5) + danger_scale * danger
        land_edge_costs = [
            np.array([0.5 * (base[i] + base[j]) for j in adj], dtype=np.float64)
            for i, adj in enumerate(mesh.face_adjacency)
        ]
        land_base_cost = np.where(is_water, 1.5, 1.5 + danger_scale * danger).astype(np.float64)
        return land_base_cost, land_edge_costs, None

    # ── tension + outputs ───────────────────────────────────────────────────────

    def _compute_tension(self, mesh, land_edge_costs, trade_flow) -> None:
        p = self.params
        pos = trade_flow[trade_flow > 0.0]
        flow_ref = float(np.quantile(pos, 0.90)) if pos.size > 0 else 0.0
        norm_flow = trade_flow / flow_ref if flow_ref > 0.0 else trade_flow
        # Scored on the sovereign partition (top-level borders only).
        self.pair_tension, self.face_tension = compute_border_tension(
            mesh.face_adjacency, self.sovereign_of_face,
            _root_power(self.polities), land_edge_costs, norm_flow,
            barrier_scale=p.tension.barrier_scale,
            flow_ref=p.tension.flow_ref,
            artery_weight=p.tension.artery_weight,
        )

    def _write_outputs(self, mesh, world) -> None:
        POLITY_ID.write(mesh, self.polity_of_face)
        SOVEREIGN_ID.write(mesh, self.sovereign_of_face)
        CONTROL_CLASS.write(mesh, self.control_of_face)
        BORDER_TENSION.write(mesh, self.face_tension)
        world.add_feature("polities", self.polities)
        world.add_feature(
            "polity_tensions",
            [(a, b, float(t)) for (a, b), t in sorted(self.pair_tension.items())],
        )

    def _finalize_empty(self, mesh, world) -> None:
        n = len(mesh.faces)
        self.polities = []
        self.polity_of_face = np.full(n, -1, dtype=np.int32)
        self.sovereign_of_face = np.full(n, -1, dtype=np.int32)
        self.control_of_face = np.full(n, -1, dtype=np.int32)
        self.authority_of_face = np.zeros(n, dtype=np.float64)
        self.face_tension = np.zeros(n, dtype=np.float64)
        self.pair_tension = {}
        self._write_outputs(mesh, world)

    def _dump_tree(self, ctx: BuildContext) -> None:
        for pol in (self.polities or []):
            if pol.liege is not None:
                continue
            ctx.report(
                f"  ⬢ Sovereign {pol.polity_id}: domain {pol.domain_power / 1e3:.0f}k people, "
                f"{pol.n_faces} faces, cohesion {pol.cohesion:.2f}, infra {pol.infrastructure:.2f}, "
                f"{len(pol.vassals)} vassal(s)."
            )

    # ── display ───────────────────────────────────────────────────────────────

    def register_display(self, world, reg: DisplayRegistry, *, debug: bool = False) -> None:
        if self.polities:
            reg.add_scheme("Polities", disp.polity_color)
        if self.face_tension is not None:
            reg.add_scheme("Border Tension", disp.tension_color)
        if debug:
            if self.sovereign_of_face is not None:
                reg.add_scheme("Sovereigns", disp.sovereign_color)
            if self.control_of_face is not None:
                reg.add_scheme("Control Class", disp.control_color)
            if self.authority_of_face is not None:
                mx = float(self.authority_of_face.max()) if self.authority_of_face.size else 0.0
                world.mesh.apply_metadata_channel("authority", self.authority_of_face.tolist())
                reg.add_scheme("Authority Field", disp.make_authority_color(mx))


def _root_power(polities: List[Polity]) -> np.ndarray:
    """Per-id power for border tension: sovereigns use domain power, vassals own."""
    out = np.zeros(len(polities), dtype=np.float64)
    for pol in polities:
        out[pol.polity_id] = pol.domain_power if pol.liege is None else pol.power
    return out
