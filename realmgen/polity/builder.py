"""Polity stage — seeds capitals, runs the forest algorithm, places forts, scores tension.

Tree model per ``docs/plans/polity_generation_brief.md``. The algorithm lives in
:mod:`realmgen.polity.forest`; this stage handles IO, cost graphs, tension and display.
With forts or marches on, the forest grows twice: forts are placed on the first pass's
borders (:mod:`realmgen.polity.forts`) and make their crossings dearer for the second,
and the first pass's border tension raises frontier vassals' autonomy in the second
(:mod:`realmgen.polity.marches`).

Runs after population and before refine in :class:`~realmgen.builder.RealmPipeline`.
"""
from __future__ import annotations

import dataclasses
from typing import List, Optional

import numpy as np

from core.display import DisplayRegistry
from core.params import field, param_field as _f, paramclass
from core.pipeline import BuildContext, Stage
from realmgen.channels import (
    AUTHORITY,
    BORDER_TENSION,
    CONTROL_CLASS,
    FISHING_POPULATION,
    FORT_LEVY,
    FORT_POPULATION,
    MINING_POPULATION,
    POLITY_ID,
    POPULATION,
    RURAL_POPULATION,
    SOVEREIGN_ID,
    TRADE_FLOW,
)
from realmgen.placement.seeder import build_cost_graph
from realmgen.polity import display as disp
from realmgen.polity.authority import land_edges
from realmgen.polity.epochs import FragmentationEvent, FragmentationParams, run_fragmentation_epochs
from realmgen.polity.forest import (
    ClaimingParams,
    HierarchyParams,
    ProjectionParams,
    _Ctx,
)
from realmgen.polity.forts import (
    FORT_BORDER,
    FORT_FRONTIER,
    FORT_STRAIT,
    FortParams,
    Forts,
    FortSite,
    border_candidates,
    find_passages,
    fortify_land_costs,
    fortify_landings,
    frontier_candidates,
    garrisons,
    select_sites,
    site_forts,
    strait_candidates,
)
from realmgen.polity.loyalty import LoyaltyParams
from realmgen.polity.marches import MarchParams, frontier_exposure
from realmgen.persistence import pack_polities, unpack_polities
from realmgen.polity.seeding import seed_capitals
from realmgen.polity.tension import border_crossings, compute_border_tension
from realmgen.polity.types import Polity
from realmgen.refine.fill import polity_fill_recipe
from realmgen.towns.placement import texel_owners
from realmgen.trade.network import build_base_hybrid_graph
from worldgen.cost.channels import COST_SURFACES, IS_WATER
from worldgen.geology.channels import OCEAN_MASK
from worldgen.hydrology.channels import LAKE_MASK


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
    marches: MarchParams = field(
        default_factory=MarchParams, metadata={"label": "Marcher autonomy"}
    )
    fragmentation: FragmentationParams = field(
        default_factory=FragmentationParams, metadata={"label": "Fragmentation epochs"}
    )
    tension: BorderTensionParams = field(
        default_factory=BorderTensionParams, metadata={"label": "Border tension"}
    )
    forts: FortParams = field(
        default_factory=FortParams, metadata={"label": "Forts"}
    )


class PolityStage(Stage):
    """Authority-field territory growth and tree (sovereign/vassal) classification."""

    name = "realm_polity"
    requires = (POPULATION, IS_WATER, COST_SURFACES)
    provides = (POLITY_ID, SOVEREIGN_ID, CONTROL_CLASS, BORDER_TENSION, FORT_POPULATION, FORT_LEVY)

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
        self.forts: Optional[Forts] = None

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
            land_edges=land_edges(mesh.face_adjacency, land_mask),
        )

        # Stages 1-3 — grow the forest; Stage 5 — fragmentation epochs on top.
        forest, self.polity_history = run_fragmentation_epochs(
            sctx, capital_sidx, land_mask, ctx.report, params=p.fragmentation
        )

        # Forts and marches on the first pass's frontiers, then the forest regrown around them.
        sites: List[FortSite] = []
        regrow: dict = {}
        if p.forts.enabled:
            sites = self._choose_fort_sites(
                mesh, is_water, forest.sovereign_of_face, forest.polities, land_base_cost,
                land_edge_costs, trade_flow, danger, settlement_faces,
            )
            fp = p.forts
            if sites and fp.resistance > 0.0:
                fort_costs = fortify_land_costs(
                    mesh.face_adjacency, land_edge_costs, forest.sovereign_of_face,
                    [s.face for s in sites if s.kind == FORT_BORDER], fp.resistance,
                )
                hybrid = self._hybrid_graph(mesh, world, is_water, fort_costs)
                if hybrid is not None:
                    hybrid = fortify_landings(
                        hybrid, [s.face for s in sites if s.kind == FORT_STRAIT], n, fp.resistance)
                regrow.update(land_graph=build_cost_graph(mesh.face_adjacency, fort_costs), hybrid_graph=hybrid)
        if p.marches.weight > 0.0:
            _, first_tension = self._compute_tension(
                mesh, land_edge_costs, trade_flow, forest.sovereign_of_face, forest.polities)
            regrow["frontier"] = frontier_exposure(
                land_graph, first_tension, reach=p.marches.reach, tension_quantile=p.marches.tension_quantile)
        if regrow:
            ctx.report(f"Regrowing polities around {len(sites)} forts and the first-pass marches…")
            forest, self.polity_history = run_fragmentation_epochs(
                dataclasses.replace(sctx, **regrow), capital_sidx, land_mask, ctx.report, params=p.fragmentation
            )
        self.forts = self._build_forts(mesh, is_water, sites, forest.sovereign_of_face)
        world.add_feature("forts", self.forts)
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
            f"{len(polities) - n_sov} vassals, {len(pair_tension)} shared borders, "
            f"{len(self.forts)} forts ({self.forts.garrison.sum() / 1e3:.0f}k garrison)."
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
        if COST_SURFACES.present(world):
            from worldgen.cost.surfaces import build_land_edge_costs
            cs = COST_SURFACES.read(world)
            land_base_cost = np.asarray(cs.land_base_cost, dtype=np.float64) + danger_scale * danger
            land_edge_costs = build_land_edge_costs(
                mesh, land_base_cost, cs.land_passable,
                cs.river_flow_dir, cs.river_flow_mag, cs.river_navigability,
            )
            return land_base_cost, land_edge_costs, self._hybrid_graph(mesh, world, is_water, land_edge_costs)

        # No cost surfaces: land-only, no sea channel.
        base = np.where(is_water, np.inf, 1.5) + danger_scale * danger
        land_edge_costs = [
            np.array([0.5 * (base[i] + base[j]) for j in adj], dtype=np.float64)
            for i, adj in enumerate(mesh.face_adjacency)
        ]
        land_base_cost = np.where(is_water, 1.5, 1.5 + danger_scale * danger).astype(np.float64)
        return land_base_cost, land_edge_costs, None

    def _hybrid_graph(self, mesh, world, is_water, land_edge_costs):
        """The land/sea graph over ``land_edge_costs``; None without cost surfaces."""
        if not COST_SURFACES.present(world):
            return None
        from worldgen.cost.surfaces import build_sea_edge_costs
        cs = COST_SURFACES.read(world)
        sea_edge_costs = build_sea_edge_costs(
            mesh, cs.sea_base_cost, cs.sea_passable, wind_dir=cs.wind_dir,
        )
        return build_base_hybrid_graph(
            len(mesh.faces), mesh.face_adjacency,
            land_edge_costs, cs.land_passable, sea_edge_costs, cs.sea_passable,
        )

    # ── forts ───────────────────────────────────────────────────────────────────

    def _choose_fort_sites(
        self, mesh, is_water, sovereign_of_face, polities, land_base_cost,
        land_edge_costs, trade_flow, danger, settlement_faces,
    ) -> List[FortSite]:
        """Border, strait and frontier sites on this partition (best first, spaced, off cities)."""
        p, fp = self.params, self.params.forts
        norm_flow, flow_ref = self._norm_flow(trade_flow)
        power = _root_power(polities)
        land = ~is_water
        cost = np.asarray(land_base_cost, dtype=np.float64)
        valid = cost[land & np.isfinite(cost)]
        lo = float(np.percentile(valid, 50.0)) if valid.size else 0.0
        hi = float(np.percentile(valid, 90.0)) if valid.size else 1.0
        rugged = np.clip((cost - lo) / max(hi - lo, 1e-9), 0.0, 1.0)

        src, dst, crossing = border_crossings(
            mesh.face_adjacency, sovereign_of_face, power, land_edge_costs, norm_flow,
            barrier_scale=p.tension.barrier_scale, flow_ref=p.tension.flow_ref,
            artery_weight=p.tension.artery_weight,
        )
        border = border_candidates(
            src, dst, crossing, rugged, rugged_weight=fp.rugged_weight, min_score=fp.border_min_score)
        sea = is_water
        if OCEAN_MASK.present(mesh):   # lake narrows aren't straits
            sea = is_water & (OCEAN_MASK.read(mesh)[..., 0].mean(axis=(1, 2)) > 0.5)
        strait = strait_candidates(
            mesh.face_adjacency, sovereign_of_face, is_water,
            find_passages(mesh.face_adjacency, sea, max_radius=fp.strait_max_radius), norm_flow,
            flow_ref=p.tension.flow_ref, artery_weight=p.tension.artery_weight,
            uncontested_weight=fp.strait_uncontested_weight, min_score=fp.strait_min_score,
        )
        frontier: List[FortSite] = []
        if self.danger_params is not None and RURAL_POPULATION.present(mesh):
            frontier = frontier_candidates(
                mesh.face_adjacency, sovereign_of_face, is_water, danger, self._rural_before_forts(mesh),
                danger_threshold=float(self.danger_params.block_threshold), min_score=fp.frontier_min_score,
            )

        # One spacing field, filled by kind: border forts first, then straits, then outposts.
        blocked = np.zeros(len(mesh.faces), dtype=bool)
        cities = {int(f) for f in settlement_faces}
        chosen: List[FortSite] = []
        for cands in (border, strait, frontier):
            chosen += select_sites(
                cands, mesh.face_adjacency, spacing_hops=fp.spacing_hops, excluded=cities, blocked=blocked)
        return chosen

    @staticmethod
    def _rural_before_forts(mesh) -> np.ndarray:
        """Rural people with any earlier run's levies put back."""
        rural = np.asarray(RURAL_POPULATION.read(mesh), dtype=np.float64)
        if FORT_LEVY.present(mesh):
            rural = rural + np.asarray(FORT_LEVY.read(mesh), dtype=np.float64)
        return rural

    def _build_forts(self, mesh, is_water, sites: List[FortSite], sovereign_of_face: np.ndarray) -> Forts:
        """Site the forts and levy their garrisons from each sovereign's farmers."""
        n = len(mesh.faces)
        fort_pop = np.zeros(n, dtype=np.float64)
        levy = np.zeros(n, dtype=np.float64)
        forts = Forts.empty()
        have_rural = RURAL_POPULATION.present(mesh)
        if sites:
            faces = np.array([s.face for s in sites], dtype=np.int64)
            positions = self._site_positions(mesh, is_water, sites, faces)
            fp = self.params.forts
            garrison = np.zeros(len(sites))
            if have_rural:
                rural = self._rural_before_forts(mesh)
                farmers = rural.copy()
                for ch in (FISHING_POPULATION, MINING_POPULATION):
                    if ch.present(mesh):
                        farmers -= np.asarray(ch.read(mesh), dtype=np.float64)
                garrison, levy = garrisons(sites, farmers, {
                    FORT_BORDER: fp.border_garrison, FORT_STRAIT: fp.strait_garrison,
                    FORT_FRONTIER: fp.frontier_garrison,
                }, sovereign_of_face)
                fort_pop = np.bincount(faces, weights=garrison, minlength=n)
            forts = Forts(
                positions=positions, face_id=faces.astype(np.int32),
                facing=np.array([s.facing for s in sites], dtype=np.int32),
                kind=np.array([s.kind for s in sites], dtype=np.int8),
                score=np.array([s.score for s in sites], dtype=np.float64), garrison=garrison,
            )
        if have_rural:
            RURAL_POPULATION.write(mesh, self._rural_before_forts(mesh) - levy)
            FORT_POPULATION.write(mesh, fort_pop)
            FORT_LEVY.write(mesh, levy)
        return forts

    @staticmethod
    def _site_positions(mesh, is_water, sites: List[FortSite], faces: np.ndarray) -> np.ndarray:
        centroids = np.asarray(mesh.face_centroids, dtype=np.float64)
        if not OCEAN_MASK.present(mesh):
            return centroids[faces] / np.linalg.norm(centroids[faces], axis=1, keepdims=True)
        ocean = OCEAN_MASK.read(mesh)[faces, ..., 0] > 0.5
        water = ocean | (LAKE_MASK.read(mesh)[faces, ..., 0] > 0.5) if LAKE_MASK.present(mesh) else ocean
        h, w = ocean.shape[1:]
        pos = np.asarray(mesh.face_texture_positions(h, w, faces), dtype=np.float64)
        owned = texel_owners(centroids, pos) == faces[:, None, None]
        elev = mesh.face_textures.get("elevation")
        elev = None if elev is None else np.asarray(elev[faces, ..., 0], dtype=np.float64)
        return site_forts(sites, centroids, pos, ~water, owned, elev)

    # ── tension + outputs ───────────────────────────────────────────────────────

    def _norm_flow(self, trade_flow: np.ndarray) -> tuple[np.ndarray, float]:
        pos = trade_flow[trade_flow > 0.0]
        flow_ref = float(np.quantile(pos, 0.90)) if pos.size > 0 else 0.0
        return (trade_flow / flow_ref if flow_ref > 0.0 else trade_flow), flow_ref

    def _compute_tension(
        self, mesh, land_edge_costs, trade_flow,
        sovereign_of_face: np.ndarray, polities: List[Polity],
    ) -> tuple[dict, np.ndarray]:
        p = self.params
        norm_flow, _ = self._norm_flow(trade_flow)
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
        self.forts = self._build_forts(mesh, np.zeros(n, dtype=bool), [], sovereign_of_face)
        world.add_feature("forts", self.forts)
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
        forts = self.forts
        if forts is not None and len(forts):
            reg.add_overlay("forts", "Forts", checked=True)
            by_face: dict = {}
            for k in range(len(forts)):
                by_face.setdefault(int(forts.face_id[k]), []).append(f"Fort ({forts.garrison[k]:.0f} men)")

            def _fort_hover(face: int, _uv, _w) -> list:
                return by_face.get(face, [])

            reg.add_hover(_fort_hover, world_only=True)
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
        out = pack_polities(self.polities) if self.polities else {}
        if self.forts is not None:
            out.update(self.forts.pack())
        return out

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
        if "fort_face_id" in arrays:
            self.forts = Forts.unpack(arrays)
            world.add_feature("forts", self.forts)


def _root_power(polities: List[Polity]) -> np.ndarray:
    """Per-id power for border tension: sovereigns use domain power, vassals own."""
    out = np.zeros(len(polities), dtype=np.float64)
    for pol in polities:
        out[pol.polity_id] = pol.domain_power if pol.liege is None else pol.power
    return out
