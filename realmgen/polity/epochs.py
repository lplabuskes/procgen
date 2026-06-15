"""Stage 5 — fragmentation epochs.

Runs a few "historical epochs" on top of the resolved forest. Each epoch rolls a
cohesion shock per sovereign; any sovereign whose shocked cohesion falls below
what its **territory size demands** sheds one peripheral seat as a new sovereign —
its weakest-held vassal (highest ``autonomy``) if it has vassals, otherwise the
lowest-authority directly-held settlement (the interior **saddle**). The shed seat
is added to the capital set and Stages 2-4 are re-run (a fresh
:class:`~realmgen.polity.forest.ForestBuilder` build), so the empire retreats to
its strong core while the breakaway holds the periphery beyond the saddle.

This is a *modifier* over the base mechanism, never a rewrite: with
``n_epochs == 0`` the forest is the plain Stage 0-3 result.

Each shed is recorded as a :class:`FragmentationEvent` so the run's sequence of
secessions survives as a ``polity_history`` feature — a chronicle a later
history-generation pass can narrate (who broke away from whom, in which epoch).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.params import param_field as _f, paramclass
from realmgen.polity.forest import ForestBuilder
from realmgen.polity.types import Polity

if TYPE_CHECKING:
    from realmgen.polity.forest import _Ctx

# Distinct RNG salt so epoch shock rolls never collide with the scalar rolls
# (which key on ``SeedSequence([base_seed, face])``).
_EPOCH_SALT = 0x_F2A6

# How a seat broke away (``FragmentationEvent.kind``).
VASSAL_SECESSION = "vassal_secession"
SADDLE_SPLIT = "saddle_split"


@dataclass
class FragmentationEvent:
    """One secession during a fragmentation epoch — a unit of generated history.

    Faces are stable identifiers (a settlement's face never moves), so both
    ``parent_capital_face`` and ``breakaway_capital_face`` are sovereign capitals
    in the final forest and can be mapped to ``Polity`` records by capital face.
    """

    epoch: int                      # 0-based epoch in which the secession occurred
    parent_capital_face: int        # capital of the sovereign that fragmented
    breakaway_capital_face: int     # capital of the seat that seceded
    kind: str                       # VASSAL_SECESSION | SADDLE_SPLIT
    parent_cohesion: float          # parent cohesion at the time
    shock: float                    # rolled cohesion shock
    demand: float                   # cohesion the parent's size demanded
    parent_domain_faces: int        # parent domain size (faces) when it fragmented


@paramclass
class FragmentationParams:
    """Stage 5 — successor-state fragmentation over a few historical epochs."""
    n_epochs: int = _f(2, label="Fragmentation epochs (0 = off)", min=0, max=8)
    shock_scale: float = _f(
        0.4, label="Cohesion shock scale (max downward roll)", min=0.0, max=1.0
    )
    size_reference: float = _f(
        400.0, label="Domain size demanding full cohesion (faces)", min=20.0, max=3000.0
    )


def _breakaway_seat(
    pol: Polity, by_id: Dict[int, Polity], authority_of_face: np.ndarray
) -> Optional[Tuple[int, str]]:
    """``(capital face, kind)`` of the seat ``pol`` sheds when it fragments, or ``None``.

    Prefer the weakest-held vassal (highest ``autonomy``); failing that, the
    lowest-authority directly-held settlement (the deepest interior saddle).
    """
    if pol.vassals:
        vassals = [by_id[v] for v in pol.vassals if v in by_id]
        if vassals:
            return int(max(vassals, key=lambda q: q.autonomy).capital_face), VASSAL_SECESSION
    members = [f for f in pol.member_settlement_faces if f != pol.capital_face]
    if not members:
        return None
    auth = np.asarray(authority_of_face)
    return int(min(members, key=lambda f: float(auth[f]))), SADDLE_SPLIT


def select_fragmentations(
    polities: Sequence[Polity],
    sovereign_of_face: np.ndarray,
    authority_of_face: np.ndarray,
    base_seed: int,
    epoch: int,
    *,
    shock_scale: float,
    size_reference: float,
) -> List[FragmentationEvent]:
    """Secessions this epoch — one :class:`FragmentationEvent` per fragmenting sovereign.

    A sovereign fragments when ``cohesion - shock < demand``, where
    ``demand = min(domain_face_count / size_reference, 1)`` (bigger domains demand
    more cohesion) and ``shock ~ U[0, shock_scale]`` is rolled from a per-(seed,
    capital, epoch) substream. Over-extended low-cohesion realms (Mongol) shed
    every epoch; tight realms (Song) never do; the shock decides the borderline.
    """
    n = len(polities)
    if n == 0:
        return []
    sof = np.asarray(sovereign_of_face)
    claimed = sof[sof >= 0]
    domain_faces = (
        np.bincount(claimed, minlength=n) if claimed.size else np.zeros(n, dtype=int)
    )
    by_id = {p.polity_id: p for p in polities}
    ref = max(float(size_reference), 1.0)

    events: List[FragmentationEvent] = []
    for pol in polities:
        if pol.liege is not None:
            continue  # only sovereigns fragment
        dom_faces = int(domain_faces[pol.polity_id])
        demand = min(dom_faces / ref, 1.0)
        rng = np.random.default_rng(
            np.random.SeedSequence([int(base_seed), int(pol.capital_face), _EPOCH_SALT, int(epoch)])
        )
        shock = float(shock_scale) * float(rng.random())
        if pol.cohesion - shock >= demand:
            continue  # holds together this epoch
        seat = _breakaway_seat(pol, by_id, authority_of_face)
        if seat is None:
            continue
        face, kind = seat
        events.append(FragmentationEvent(
            epoch=int(epoch), parent_capital_face=int(pol.capital_face),
            breakaway_capital_face=face, kind=kind,
            parent_cohesion=float(pol.cohesion), shock=shock,
            demand=float(demand), parent_domain_faces=dom_faces,
        ))
    return events


def _sovereign_faces(forest: ForestBuilder) -> set:
    """Capital faces of the forest's current sovereigns."""
    return {int(p.capital_face) for p in forest.polities if p.liege is None}


def run_fragmentation_epochs(
    sctx: "_Ctx",
    capital_sidx: List[int],
    land_mask: np.ndarray,
    report: Callable[[str], None],
    *,
    params: FragmentationParams,
) -> Tuple[ForestBuilder, List[FragmentationEvent]]:
    """Build the forest, then run ``params.n_epochs`` fragmentation epochs.

    Returns ``(final ForestBuilder, history)`` where ``history`` is the chronicle
    of secessions in order. Each epoch re-runs Stages 2-4 with the shed seats
    added to the capital set; stops early once an epoch is stable.
    """
    forest = ForestBuilder(sctx)
    forest.build(list(capital_sidx), land_mask, report)

    history: List[FragmentationEvent] = []
    n_epochs = int(params.n_epochs)
    for epoch in range(n_epochs):
        candidates = select_fragmentations(
            forest.polities, forest.sovereign_of_face, forest.authority_of_face,
            sctx.base_seed, epoch,
            shock_scale=params.shock_scale, size_reference=params.size_reference,
        )
        # Keep only secessions whose breakaway becomes a genuinely new sovereign.
        existing = _sovereign_faces(forest)
        fresh = [
            ev for ev in candidates
            if ev.breakaway_capital_face not in existing
            and ev.breakaway_capital_face in sctx.face_to_settlement
        ]
        if not fresh:
            report(f"  Fragmentation epoch {epoch + 1}/{n_epochs}: stable.")
            break
        history.extend(fresh)
        new_sidx = {sctx.face_to_settlement[ev.breakaway_capital_face] for ev in fresh}
        existing_sidx = {
            sctx.face_to_settlement[f] for f in existing if f in sctx.face_to_settlement
        }
        caps = sorted(existing_sidx | new_sidx)
        report(
            f"  Fragmentation epoch {epoch + 1}/{n_epochs}: "
            f"{len(fresh)} breakaway(s) → rebuild with {len(caps)} sovereigns."
        )
        forest = ForestBuilder(sctx)
        forest.build(caps, land_mask, report)

    return forest, history
