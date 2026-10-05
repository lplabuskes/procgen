"""Polity stage — seeds capitals, runs the forest algorithm, scores tension.

Tree model per ``docs/plans/polity_generation_brief.md``. The algorithm lives in
:mod:`realmgen.polity.forest`; this stage handles IO, cost graphs, tension and display.

Runs after population and before refine in :class:`~realmgen.builder.RealmPipeline`.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from core.display import DisplayRegistry
from core.params import field, param_field as _f, paramclass
from core.pipeline import BuildContext, Stage
from realmgen.channels import (
    AUTHORITY,
    BORDER_TENSION,
    CONTROL_CLASS,
    POLITY_ID,
    POPULATION,
    SOVEREIGN_ID,
    TRADE_FLOW,
)
from realmgen.placement.seeder import build_cost_graph
from realmgen.polity import display as disp
from realmgen.polity.epochs import FragmentationEvent, FragmentationParams, run_fragmentation_epochs
from realmgen.polity.forest import (
    ClaimingParams,
    HierarchyParams,
    ProjectionParams,
    _Ctx,
)
from realmgen.polity.loyalty import LoyaltyParams
from realmgen.persistence import pack_polities, unpack_polities
from realmgen.polity.seeding import seed_capitals
from realmgen.polity.tension import compute_border_tension
from realmgen.polity.types import Polity
from realmgen.refine.fill import polity_fill_recipe
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
    """Per-settlement defensibility multiplier — computed in PopulationStage and stored on Settlement."""
    rugged_weight: float = _f(0.6, label="Defensibility: ruggedness", min=0.0, max=3.0)
    situational_weight: float = _f(0.8, label="Defensibility: situational", min=0.0, max=3.0)
    river_weight: float = _f(0.4, label="Defensibility: river barrier", min=0.0, max=3.0)
    wall_base_prob: float = _f(0.3, label="Walls: base probability", min=0.0, max=1.0)
    wall_danger_boost: float = _f(0.3, label="Walls: danger boost to prob", min=0.0, max=1.0)
    wall_wealth_boost: float = _f(0.3, label="Walls: wealth boost to prob", min=0.0, max=1.0)
    wall_min_strength: float = _f(0.3, label="Walls: min strength bonus", min=0.0, max=5.0)
    wall_max_strength: float = _f(1.5, label="Walls: max strength bonus", min=0.0, max=5.0)


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
    loyalty: LoyaltyParams = field(
        default_factory=LoyaltyParams, metadata={"label": "Loyalty modifiers"}
    )
    fragmentation: FragmentationParams = field(
        default_factory=FragmentationParams, metadata={"label": "Fragmentation epochs"}
    )
    tension: BorderTensionParams = field(
        default_factory=BorderTensionParams, metadata={"label": "Border tension"}
    )


class PolityStage(Stage):
    """Authority-field territory growth and tree (sovereign/vassal) classification."""

    name = "realm_polity"
    requires = (POPULATION, IS_WATER, COST_SURFACES)
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
        self.polity_history: List[FragmentationEvent] = []

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
        trade_flow = (
            TRADE_FLOW.read(mesh).astype(np.float64)
            if TRADE_FLOW.present(mesh) else np.zeros(n)
        )
        trade_wealth = trade_flow[settlement_faces]
        # (S, S) trade matrix for the Stage-4 loyalty levers; absent => levers no-op.
        n_settle = len(settlements)
        trade_attribution = world.get_feature("trade_attribution")
        if trade_attribution is None or np.shape(trade_attribution) != (n_settle, n_settle):
            trade_attribution = np.zeros((0, 0), dtype=np.float64)
        else:
            trade_attribution = np.asarray(trade_attribution, dtype=np.float64)

        danger, danger_scale = self._read_danger(world, n)
        land_base_cost, land_edge_costs, hybrid_graph = self._build_cost_graphs(
            mesh, world, is_water, danger, danger_scale,
        )
        land_graph = build_cost_graph(mesh.face_adjacency, land_edge_costs)

        # Only settlement faces are ever read; the rest stay 1.0.
        defensibility = np.ones(n, dtype=np.float64)
        for s in settlements:
            defensibility[s.face_id] = s.defensibility

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
            trade_wealth=trade_wealth, trade_attribution=trade_attribution,
            defensibility=defensibility, land_graph=land_graph, hybrid_graph=hybrid_graph,
            base_seed=int(world.seed) if world.seed is not None else 0, p=p,
            face_to_settlement={int(f): k for k, f in enumerate(settlement_faces.tolist())},
        )

        # Stages 1-3 — grow the forest; Stage 5 — fragmentation epochs on top.
        forest, self.polity_history = run_fragmentation_epochs(
            sctx, capital_sidx, land_mask, ctx.report, params=p.fragmentation
        )
        polities = forest.polities
        polity_of_face = forest.polity_of_face
        sovereign_of_face = forest.sovereign_of_face
        control_of_face = forest.control_of_face
        self.polities = polities
        self.polity_of_face = polity_of_face
        self.sovereign_of_face = sovereign_of_face
        self.control_of_face = control_of_face
        self.authority_of_face = forest.authority_of_face

        pair_tension, face_tension = self._compute_tension(
            mesh, land_edge_costs, trade_flow, sovereign_of_face, polities
        )
        self.pair_tension = pair_tension
        self.face_tension = face_tension
        self._write_outputs(mesh, world, polities, polity_of_face, sovereign_of_face, control_of_face, face_tension,
                            pair_tension, is_water)

        n_sov = sum(1 for pol in polities if pol.liege is None)
        ctx.report(
            f"Polity formation done: {n_sov} sovereigns, "
            f"{len(polities) - n_sov} vassals, {len(pair_tension)} shared borders."
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

        # No cost surfaces: land-only, no sea channel.
        base = np.where(is_water, np.inf, 1.5) + danger_scale * danger
        land_edge_costs = [
            np.array([0.5 * (base[i] + base[j]) for j in adj], dtype=np.float64)
            for i, adj in enumerate(mesh.face_adjacency)
        ]
        land_base_cost = np.where(is_water, 1.5, 1.5 + danger_scale * danger).astype(np.float64)
        return land_base_cost, land_edge_costs, None

    # ── tension + outputs ───────────────────────────────────────────────────────

    def _compute_tension(
        self, mesh, land_edge_costs, trade_flow,
        sovereign_of_face: np.ndarray, polities: List[Polity],
    ) -> tuple[dict, np.ndarray]:
        p = self.params
        pos = trade_flow[trade_flow > 0.0]
        flow_ref = float(np.quantile(pos, 0.90)) if pos.size > 0 else 0.0
        norm_flow = trade_flow / flow_ref if flow_ref > 0.0 else trade_flow
        # Sovereign partition only (top-level borders).
        return compute_border_tension(
            mesh.face_adjacency, sovereign_of_face,
            _root_power(polities), land_edge_costs, norm_flow,
            barrier_scale=p.tension.barrier_scale,
            flow_ref=p.tension.flow_ref,
            artery_weight=p.tension.artery_weight,
        )

    def _write_outputs(
        self, mesh, world,
        polities: List[Polity], polity_of_face: np.ndarray,
        sovereign_of_face: np.ndarray, control_of_face: np.ndarray,
        face_tension: np.ndarray, pair_tension: dict,
        is_water: np.ndarray,
    ) -> None:
        POLITY_ID.write(mesh, polity_of_face)
        SOVEREIGN_ID.write(mesh, sovereign_of_face)
        CONTROL_CLASS.write(mesh, control_of_face)
        BORDER_TENSION.write(mesh, face_tension)
        if self.authority_of_face is not None:
            AUTHORITY.write(mesh, np.asarray(self.authority_of_face, dtype=np.float64))
        world.add_feature("polities", polities)
        world.add_feature("polity_history", list(self.polity_history))
        world.add_feature(
            "polity_tensions",
            [(a, b, float(t)) for (a, b), t in sorted(pair_tension.items())],
        )
        world.add_feature(
            "polity_borders",
            self._compute_polity_borders(mesh, polities, polity_of_face, is_water),
        )

    def _compute_polity_borders(
        self, mesh, polities: List[Polity],
        polity_of_face: np.ndarray, is_water: np.ndarray,
    ) -> dict:
        """``{depth: (M, 6) float32}`` border segments ``(xyz, xyz)``.

        Depth 0 = sovereign borders (incl. vs unclaimed); depth d = between depth-d
        subtrees of the same parent.
        """
        if not polities:
            return {}

        pol_by_id = {p.polity_id: p for p in polities}
        max_depth = max(p.depth for p in polities)
        n = len(polity_of_face)

        pol_chain: dict[int, list[int]] = {}
        for pol in polities:
            chain: list[int] = []
            cur: Optional[Polity] = pol
            while cur is not None:
                chain.append(cur.polity_id)
                cur = pol_by_id.get(cur.liege) if cur.liege is not None else None
            chain.reverse()
            pol_chain[pol.polity_id] = chain

        # ancestor_at[d, face]: ancestor polity at depth d, clamped to the deepest.
        ancestor_at = np.full((max_depth + 1, n), -1, dtype=np.int32)
        for face_idx in range(n):
            if is_water[face_idx]:
                continue
            pid = int(polity_of_face[face_idx])
            if pid < 0:
                continue
            chain = pol_chain.get(pid, [pid])
            for d in range(max_depth + 1):
                ancestor_at[d, face_idx] = chain[min(d, len(chain) - 1)]

        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = mesh.faces

        by_depth: dict[int, list] = {}
        for face_a, neighbors in enumerate(mesh.face_adjacency):
            if is_water[face_a]:
                continue
            verts_a = set(faces[face_a])
            for face_b in neighbors:
                if face_b <= face_a:
                    continue
                if is_water[face_b]:
                    continue
                if ancestor_at[0, face_a] < 0 and ancestor_at[0, face_b] < 0:
                    continue
                border_depth = -1
                for d in range(max_depth + 1):
                    if ancestor_at[d, face_a] != ancestor_at[d, face_b]:
                        border_depth = d
                        break
                if border_depth < 0:
                    continue
                shared = list(verts_a & set(faces[face_b]))
                if len(shared) != 2:
                    continue
                seg = np.concatenate([vertices[shared[0]], vertices[shared[1]]])
                by_depth.setdefault(border_depth, []).append(seg)

        return {d: np.array(segs, dtype=np.float32) for d, segs in by_depth.items()}

    def _finalize_empty(self, mesh, world) -> None:
        n = len(mesh.faces)
        polities: List[Polity] = []
        polity_of_face = np.full(n, -1, dtype=np.int32)
        sovereign_of_face = np.full(n, -1, dtype=np.int32)
        control_of_face = np.full(n, -1, dtype=np.int32)
        face_tension = np.zeros(n, dtype=np.float64)
        pair_tension: dict = {}
        self.polities = polities
        self.polity_of_face = polity_of_face
        self.sovereign_of_face = sovereign_of_face
        self.control_of_face = control_of_face
        self.authority_of_face = np.zeros(n, dtype=np.float64)
        self.face_tension = face_tension
        self.pair_tension = pair_tension
        self._write_outputs(mesh, world, polities, polity_of_face, sovereign_of_face, control_of_face, face_tension,
                            pair_tension, np.zeros(n, dtype=bool))

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
            tex = ("polity_fill" if world is not None
                   and "polity_fill" in world.mesh.face_textures else None)
            # Lets a region redraw the fill at its own resolution.
            field = world.get_feature("polity_label_field") if world is not None else None
            recipe = (polity_fill_recipe(field, self.polities)
                      if tex is not None and field is not None else None)
            reg.add_scheme("Polities", disp.make_polity_tree_color(self.polities),
                           texture=tex, recipe=recipe)
            pol_by_id = {pol.polity_id: pol for pol in self.polities}

            def _polity_hover(face: int, _uv, w) -> list:
                meta = w.mesh.face_metadata[face]
                pid = meta.get("polity_id")
                if pid is None or int(pid) < 0:
                    return []
                chain: List[int] = []
                node = int(pid)
                while node >= 0:
                    chain.append(node)
                    node_pol = pol_by_id.get(node)
                    if node_pol is None or node_pol.liege is None:
                        break
                    node = node_pol.liege
                chain.reverse()
                parts = [
                    f"Sovereign {chain[0]}" if len(chain) == 1
                    else "Pol: " + " › ".join(str(p) for p in chain)
                ]
                cc = meta.get("control_class")
                if cc is not None and int(cc) >= 0:
                    parts.append(["Direct", "Vassal", "Independent"][int(cc)])
                tension = meta.get("border_tension")
                if debug and tension is not None and float(tension) > 0.01:
                    parts.append(f"Tension: {float(tension):.2f}")
                return parts

            reg.add_hover(_polity_hover)
        if debug and self.face_tension is not None:
            reg.add_scheme("Border Tension", disp.tension_color)
        if world is not None and world.has_feature("polity_borders"):
            reg.add_overlay("polity_borders", "Polity Borders")
        if debug:
            if self.sovereign_of_face is not None:
                reg.add_scheme("Sovereigns", disp.sovereign_color)
            if self.control_of_face is not None:
                reg.add_scheme("Control Class", disp.control_color)
            if self.authority_of_face is not None:
                mx = float(self.authority_of_face.max()) if self.authority_of_face.size else 0.0
                world.mesh.apply_metadata_channel("authority", self.authority_of_face.tolist())
                reg.add_scheme("Authority Field", disp.make_authority_color(mx))

    # ── Persistence ───────────────────────────────────────────────────────────
    # Per-face fields round-trip via metadata; load() rebuilds only the Polity objects.

    def serialize(self) -> dict:
        if self.polities:
            return pack_polities(self.polities)
        return {}

    def load(self, arrays, ctx: BuildContext) -> None:
        world, mesh = ctx.world, ctx.mesh
        if "pol_id" in arrays:
            self.polities = unpack_polities(arrays)
            world.add_feature("polities", self.polities)
        if POLITY_ID.present(mesh):
            self.polity_of_face = np.asarray(POLITY_ID.read(mesh), dtype=np.int32)
        if SOVEREIGN_ID.present(mesh):
            self.sovereign_of_face = np.asarray(SOVEREIGN_ID.read(mesh), dtype=np.int32)
        if CONTROL_CLASS.present(mesh):
            self.control_of_face = np.asarray(CONTROL_CLASS.read(mesh), dtype=np.int32)
        if BORDER_TENSION.present(mesh):
            self.face_tension = np.asarray(BORDER_TENSION.read(mesh), dtype=np.float64)
        if AUTHORITY.present(mesh):
            self.authority_of_face = np.asarray(AUTHORITY.read(mesh), dtype=np.float64)


def _root_power(polities: List[Polity]) -> np.ndarray:
    """Per-id power for border tension: sovereigns use domain power, vassals own."""
    out = np.zeros(len(polities), dtype=np.float64)
    for pol in polities:
        out[pol.polity_id] = pol.domain_power if pol.liege is None else pol.power
    return out
