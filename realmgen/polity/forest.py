"""The sovereign/vassal forest algorithm — Stages 1-3 orchestration.

``ForestBuilder`` threads the pure functions (seeding, authority, claiming,
classification) into the tree: resolve sovereigns (claim + promote independents),
then recurse each into its directly-held core plus vassal sub-territories.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from core.params import param_field as _f, paramclass
from realmgen.polity.authority import (
    ProjectionDistances,
    claim_faces,
    classify_settlements,
    compute_authority_fields,
    merge_exclaves,
)
from realmgen.polity.loyalty import (
    compute_internal_trade_fraction,
    economic_grip_multiplier,
    peripheral_wealth_breakaways,
)
from realmgen.polity.marches import march_autonomy
from realmgen.polity.seeding import cohesion_to_gain, roll_polity_scalars
from realmgen.polity.types import ControlClass, Polity, Scalars

if TYPE_CHECKING:
    from realmgen.polity.builder import PolityParams


# ── Params (consumed only in this module) ──────────────────────────────────────


@paramclass
class ProjectionParams:
    """Authority field reach + the per-polity scalar rolls that modulate it."""
    base_range: float = _f(
        14.0, label="Projection base range (cost-dist / e-fold)", min=5.0, max=400.0
    )
    range_floor: float = _f(
        0.5, label="Projection range floor (+ infrastructure)", min=0.0, max=2.0
    )
    sea_range_factor: float = _f(1.1, label="Sea projection range factor (× land)", min=1.0, max=10.0)
    sea_weight: float = _f(1.0, label="Sea channel magnitude weight (vs land)", min=0.0, max=1.0)
    legitimacy_spread: float = _f(0.3, label="Legitimacy spread (±)", min=0.0, max=1.0)
    maritime_scale: float = _f(1.5, label="Maritime reach scale (× trade)", min=0.0, max=5.0)
    cohesion_min: float = _f(0.4, label="Cohesion floor", min=0.0, max=1.0)
    infrastructure_min: float = _f(0.4, label="Infrastructure floor", min=0.0, max=1.0)
    infrastructure_emergent: float = _f(
        0.5, label="Infrastructure emergent weight (pop+trade)", min=0.0, max=1.0
    )


@paramclass
class ClaimingParams:
    """Face-claim competition and the independence-relaxation outer loop."""
    hysteresis: float = _f(0.15, label="Claim hysteresis (incumbent bonus)", min=0.0, max=1.0)
    max_iters: int = _f(8, label="Claim max iterations", min=1, max=50)
    outer_max_iters: int = _f(6, label="Independence relaxation iterations", min=1, max=20)
    exclave_max_faces: int = _f(
        40, label="Merge seatless land pieces up to (faces, 0=off)", min=0, max=500, step=1
    )


@paramclass
class HierarchyParams:
    """Direct/vassal/independent cutoffs and the cohesion→authority-gain map."""
    direct_threshold: float = _f(0.8, label="Direct-rule authority ratio", min=0.5, max=20.0)
    vassal_floor: float = _f(0.2, label="Vassal authority ratio floor", min=0.05, max=10.0)
    power_exponent: float = _f(
        0.5, label="Authority population exponent (1 = raw ratio)", min=0.0, max=1.0
    )
    cohesion_gain_floor: float = _f(0.5, label="Cohesion gain floor", min=0.0, max=2.0)
    cohesion_gain_span: float = _f(1.0, label="Cohesion gain span", min=0.0, max=2.0)
    max_recursion_depth: int = _f(2, label="Vassal recursion depth", min=0, max=3)
    vassal_cluster_distance: float = _f(
        15.0, label="Vassal cluster distance (cost-dist, 0=off)", min=0.0, max=200.0
    )


@dataclass
class _Ctx:
    """Read-only inputs shared by the recursive builder."""
    n: int
    centroids: np.ndarray
    settlement_faces: np.ndarray        # (S,)
    populations: np.ndarray             # (S,) raw people
    pop_norm: np.ndarray                # (S,) populations / peak
    trade_norm: np.ndarray              # (S,) trade wealth / peak
    trade_wealth: np.ndarray            # (S,) raw trade wealth (loyalty levers)
    trade_attribution: np.ndarray       # (S, S) directional pair flow (loyalty levers)
    defensibility: np.ndarray           # (N,)
    land_graph: csr_matrix
    hybrid_graph: Optional[csr_matrix]
    base_seed: int
    p: "PolityParams"
    face_to_settlement: Dict[int, int]
    frontier: Optional[np.ndarray] = None  # (N,) march exposure from a previous pass; None = off
    # Directed land–land adjacency (authority.land_edges); empty => no exclave merging.
    land_edges: Tuple[np.ndarray, np.ndarray] = field(
        default_factory=lambda: (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)))
    distances: ProjectionDistances = field(init=False)

    def __post_init__(self) -> None:
        self.distances = ProjectionDistances(self.land_graph, self.hybrid_graph, self.n)


class ForestBuilder:
    """Grows the polity forest from seeded capitals into per-face assignments."""

    def __init__(self, sctx: _Ctx) -> None:
        self.s = sctx
        n = sctx.n
        self.polities: List[Polity] = []
        self.polity_of_face = np.full(n, -1, dtype=np.int32)    # directly-governing id
        self.sovereign_of_face = np.full(n, -1, dtype=np.int32)  # root id
        self.control_of_face = np.full(n, -1, dtype=np.int32)    # ControlClass at settlements
        self.authority_of_face = np.zeros(n, dtype=np.float64)   # winning field value (debug)

    def build(self, capital_sidx: List[int], land_mask: np.ndarray, report: Callable[[str], None]) -> None:
        partition, fields, scalars = self._resolve_sovereigns(capital_sidx, land_mask, report)
        for k, sidx in enumerate(capital_sidx):
            territory = np.flatnonzero(partition == k)
            if territory.size == 0:
                territory = np.array([int(self.s.settlement_faces[sidx])], dtype=np.int64)
            self._build_subtree(
                sidx, scalars.take(k), territory, self._members_on(territory),
                liege_id=None, root_id=None, depth=0, autonomy=0.0,
            )
        self.authority_of_face = fields.max(axis=0) if fields.size else np.zeros(self.s.n)
        self._renumber_sovereign_first()

    def _renumber_sovereign_first(self) -> None:
        """Renumber so sovereigns take the lowest ids (consecutive POLITY_COLORS entries)."""
        old = self.polities
        sov_ids = [p.polity_id for p in old if p.liege is None]
        vas_ids = [p.polity_id for p in old if p.liege is not None]
        remap = {old_id: new_id for new_id, old_id in enumerate(sov_ids + vas_ids)}

        new_polities: List[Optional[Polity]] = [None] * len(old)
        for pol in old:
            new_pid = remap[pol.polity_id]
            new_polities[new_pid] = Polity(
                polity_id=new_pid,
                capital_face=pol.capital_face,
                capital_position=pol.capital_position,
                cohesion=pol.cohesion,
                infrastructure=pol.infrastructure,
                legitimacy=pol.legitimacy,
                maritime_reach=pol.maritime_reach,
                liege=remap[pol.liege] if pol.liege is not None else None,
                autonomy=pol.autonomy,
                depth=pol.depth,
                power=pol.power,
                domain_power=pol.domain_power,
                n_faces=pol.n_faces,
                territory_faces=pol.territory_faces,
                member_settlement_faces=pol.member_settlement_faces,
                vassals=[remap[v] for v in pol.vassals],
            )
        self.polities = new_polities  # type: ignore[assignment]

        remap_arr = np.array([remap[i] for i in range(len(old))], dtype=np.int32)
        for arr in (self.polity_of_face, self.sovereign_of_face):
            mask = arr >= 0
            arr[mask] = remap_arr[arr[mask]]

    # ── sovereign resolution ──────────────────────────────────────────────────

    def _resolve_sovereigns(self, capital_sidx, land_mask, report):
        """Claim → promote-independents loop; mutates ``capital_sidx`` to the
        settled sovereign set. Returns ``(partition (N,), fields (C,N), scalars)``."""
        s, p = self.s, self.s.p
        prev = None
        partition = np.full(s.n, -1, dtype=np.int32)
        fields = np.zeros((0, s.n))
        scalars = Scalars(*(np.zeros(0) for _ in range(4)))

        n_outer = max(1, int(p.claiming.outer_max_iters))
        for outer in range(n_outer):
            capital_faces = [int(s.settlement_faces[i]) for i in capital_sidx]
            scalars = self._roll(capital_sidx, capital_faces)
            fields = self._project(capital_faces, scalars)
            partition, n_iters = claim_faces(
                fields, capital_faces, land_mask,
                hysteresis=p.claiming.hysteresis, max_iters=p.claiming.max_iters, prev=prev,
            )
            partition = self._merge_exclaves(partition, capital_faces, fields)
            prev = partition

            # Capital's raw power, not the domain sum (which would let a big empire hold everything).
            cap_power = s.populations[np.asarray(capital_sidx, dtype=np.int64)]
            gain = cohesion_to_gain(
                scalars.cohesion,
                gain_floor=p.hierarchy.cohesion_gain_floor, gain_span=p.hierarchy.cohesion_gain_span,
            )
            is_cap = np.zeros(s.settlement_faces.size, dtype=bool)
            is_cap[capital_sidx] = True
            loyalty_mult, force_ind = self._loyalty_modifiers(partition, capital_sidx)
            control, _ = classify_settlements(
                s.settlement_faces.tolist(), s.populations, partition, fields,
                cap_power, s.defensibility, gain, is_cap,
                direct_threshold=p.hierarchy.direct_threshold, vassal_floor=p.hierarchy.vassal_floor,
                power_exponent=p.hierarchy.power_exponent,
                loyalty_mult=loyalty_mult, force_independent=force_ind,
            )
            existing = set(capital_sidx)
            new_caps = [
                int(x) for x in np.flatnonzero(control == int(ControlClass.INDEPENDENT)).tolist()
                if int(x) not in existing
            ]
            # Don't extend on the final pass, or the returned scalars/fields would be stale.
            last_pass = outer == n_outer - 1
            if not new_caps or last_pass:
                report(
                    f"  Sovereigns settled after {outer + 1} pass(es): "
                    f"{len(capital_sidx)} capitals ({n_iters} claim iters)."
                )
                break
            capital_sidx.extend(sorted(new_caps))
            report(f"  +{len(new_caps)} seceded → {len(capital_sidx)} sovereigns.")

        return partition, fields, scalars

    def _merge_exclaves(self, assign, capital_faces, fields, eligible=None) -> np.ndarray:
        max_faces = int(self.s.p.claiming.exclave_max_faces)
        if max_faces <= 0 or self.s.land_edges[0].size == 0:
            return assign
        return merge_exclaves(assign, capital_faces, fields, self.s.land_edges,
                              max_faces=max_faces, eligible_mask=eligible)

    # ── Stage-4 loyalty levers ─────────────────────────────────────────────────

    def _loyalty_modifiers(self, partition: np.ndarray, capital_sidx: List[int]):
        """``(loyalty_mult, force_independent)`` for the current claim partition.

        Either is ``None`` when its lever is disabled or no trade attribution exists.
        """
        s, lp = self.s, self.s.p.loyalty
        grip_on = lp.economic_grip > 0.0
        peripheral_on = lp.peripheral_wealth_ratio > 0.0
        if (not grip_on and not peripheral_on) or s.trade_attribution.size == 0:
            return None, None

        frac = compute_internal_trade_fraction(
            s.settlement_faces, partition, s.trade_attribution
        )
        mult = economic_grip_multiplier(frac, lp.economic_grip) if grip_on else None
        force = (
            peripheral_wealth_breakaways(
                s.settlement_faces, partition, capital_sidx, s.trade_wealth, frac,
                wealth_ratio=lp.peripheral_wealth_ratio,
                external_floor=lp.peripheral_external_floor,
            )
            if peripheral_on else None
        )
        return mult, force

    # ── projection / scalar helpers ───────────────────────────────────────────

    def _project(self, capital_faces: Sequence[int], sc: Scalars) -> np.ndarray:
        """Authority fields ``(C, N)``."""
        p = self.s.p
        return compute_authority_fields(
            self.s.n, capital_faces, sc.legitimacy, sc.infrastructure, sc.maritime_reach,
            self.s.land_graph, self.s.hybrid_graph,
            base_range=p.projection.base_range, range_floor=p.projection.range_floor,
            sea_range_factor=p.projection.sea_range_factor, sea_weight=p.projection.sea_weight,
            distances=self.s.distances,
        )

    def _roll(self, capital_sidx, capital_faces) -> Scalars:
        s, p = self.s, self.s.p
        sidx = np.asarray(capital_sidx, dtype=np.int64)
        return roll_polity_scalars(
            capital_faces, s.pop_norm[sidx], s.trade_norm[sidx], s.base_seed,
            cohesion_min=p.projection.cohesion_min, infrastructure_min=p.projection.infrastructure_min,
            infrastructure_emergent=p.projection.infrastructure_emergent,
            legitimacy_spread=p.projection.legitimacy_spread, maritime_scale=p.projection.maritime_scale,
        )

    def _members_on(self, faces: np.ndarray) -> List[int]:
        lookup = self.s.face_to_settlement
        return [lookup[int(f)] for f in faces.tolist() if int(f) in lookup]

    def _cluster_vassal_members(self, vassal_members: List[int], threshold: float) -> List[List[int]]:
        """Greedily group vassal members within `threshold` cost distance of one another."""
        s = self.s
        source_faces = [int(s.settlement_faces[m]) for m in vassal_members]
        dists = dijkstra(s.land_graph, directed=False, indices=source_faces)
        n = len(vassal_members)
        assigned = np.zeros(n, dtype=bool)
        clusters: List[List[int]] = []
        for i in range(n):
            if assigned[i]:
                continue
            cluster = [vassal_members[i]]
            assigned[i] = True
            for j in range(i + 1, n):
                if assigned[j]:
                    continue
                if dists[i, source_faces[j]] <= threshold:
                    cluster.append(vassal_members[j])
                    assigned[j] = True
            clusters.append(cluster)
        return clusters

    # ── recursive subtree builder ─────────────────────────────────────────────

    def _build_subtree(
        self, cap_sidx: int, sc: Scalars, territory: np.ndarray, members: List[int],
        *, liege_id: Optional[int], root_id: Optional[int], depth: int, autonomy: float,
    ) -> int:
        """Create one polity over ``territory``; recurse to carve vassals.

        ``sc`` is this polity's 1-element :class:`Scalars`. Returns the new id.
        """
        s, p = self.s, self.s.p
        pid = len(self.polities)
        if root_id is None:
            root_id = pid
        cap_face = int(s.settlement_faces[cap_sidx])
        domain_raw = float(s.populations[members].sum()) if members else 0.0

        # Members not firmly held (authority vs local power) become vassals.
        vassal_members: List[int] = []
        vassal_autonomy: Dict[int, float] = {}
        if depth < int(p.hierarchy.max_recursion_depth) and len(members) > 1:
            field_self = self._project([cap_face], sc)[0]
            e = float(p.hierarchy.power_exponent)
            cap_power = float(s.populations[cap_sidx]) ** e
            gain = float(cohesion_to_gain(
                sc.cohesion,
                gain_floor=p.hierarchy.cohesion_gain_floor, gain_span=p.hierarchy.cohesion_gain_span,
            )[0])
            for m in members:
                if m == cap_sidx:
                    continue
                face = int(s.settlement_faces[m])
                eff = cap_power * float(field_self[face]) * gain
                loc = float(s.populations[m]) ** e * float(s.defensibility[face])
                r = eff / max(loc, 1e-9)
                if r < p.hierarchy.direct_threshold:
                    vassal_members.append(m)
                    vassal_autonomy[m] = float(
                        np.clip(1.0 - r / max(p.hierarchy.direct_threshold, 1e-9), 0.0, 1.0)
                    )

            cluster_dist = float(p.hierarchy.vassal_cluster_distance)
            if vassal_members and cluster_dist > 0.0:
                clusters = self._cluster_vassal_members(vassal_members, cluster_dist)
                vassal_members = []
                new_autonomy: Dict[int, float] = {}
                for cluster in clusters:
                    cap_m = max(cluster, key=lambda m: float(s.populations[m]))
                    vassal_members.append(cap_m)
                    new_autonomy[cap_m] = vassal_autonomy[cap_m]
                vassal_autonomy = new_autonomy

        self.polities.append(None)  # type: ignore[arg-type]  # reserve slot for child ids
        own_faces = territory
        child_ids: List[int] = []
        if vassal_members:
            sub_sidx = [cap_sidx] + vassal_members
            sub_caps = [cap_face] + [int(s.settlement_faces[m]) for m in vassal_members]
            sub = self._roll(sub_sidx, sub_caps)
            for parent_val, arr in zip(sc, sub):  # parent keeps its established scalars at idx 0
                arr[0] = float(parent_val[0])
            terr_mask = np.zeros(s.n, dtype=bool)
            terr_mask[territory] = True
            sub_field = self._project(sub_caps, sub)
            sub_assign, _ = claim_faces(
                sub_field, sub_caps, terr_mask,
                hysteresis=p.claiming.hysteresis, max_iters=p.claiming.max_iters,
            )
            sub_assign = self._merge_exclaves(sub_assign, sub_caps, sub_field, terr_mask)
            own_faces = np.flatnonzero(sub_assign == 0)
            for j, vm in enumerate(vassal_members, start=1):
                v_terr = np.flatnonzero(sub_assign == j)
                if v_terr.size == 0:
                    v_terr = np.array([int(s.settlement_faces[vm])], dtype=np.int64)
                v_autonomy = vassal_autonomy.get(vm, 0.0)
                if s.frontier is not None:  # a march: territory reaching a hot frontier
                    v_autonomy = march_autonomy(v_autonomy, float(s.frontier[v_terr].max()), p.marches.weight)
                child_ids.append(self._build_subtree(
                    vm, sub.take(j), v_terr, self._members_on(v_terr),
                    liege_id=pid, root_id=root_id, depth=depth + 1, autonomy=v_autonomy,
                ))
                self.control_of_face[int(s.settlement_faces[vm])] = int(ControlClass.VASSAL)

        self.polity_of_face[own_faces] = pid
        self.sovereign_of_face[own_faces] = root_id
        own_face_set = set(own_faces.tolist())
        own_members = [m for m in members if int(s.settlement_faces[m]) in own_face_set]
        for m in own_members:
            face = int(s.settlement_faces[m])
            if self.control_of_face[face] < 0:
                self.control_of_face[face] = int(ControlClass.DIRECT)

        self.polities[pid] = Polity(
            polity_id=pid,
            capital_face=cap_face,
            capital_position=s.centroids[cap_face].copy(),
            cohesion=float(sc.cohesion[0]),
            infrastructure=float(sc.infrastructure[0]),
            legitimacy=float(sc.legitimacy[0]),
            maritime_reach=float(sc.maritime_reach[0]),
            liege=liege_id,
            autonomy=float(autonomy),
            depth=depth,
            power=float(s.populations[own_members].sum()) if own_members else 0.0,
            domain_power=domain_raw,
            n_faces=int(own_faces.size),
            territory_faces=own_faces.astype(int).tolist(),
            member_settlement_faces=[int(s.settlement_faces[m]) for m in own_members],
            vassals=child_ids,
        )
        return pid
