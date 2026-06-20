"""Stage 4 loyalty modifiers — economic grip and peripheral-wealth secession.

Both levers key on the per-settlement, per-partner trade-attribution matrix
(``trade_attribution`` world feature, ``(S, S)`` aligned with ``settlements``)
and the current claim **partition**. They sit on opposite ends of one question:
*who does this city's economy actually depend on?*

- **Economic grip** raises a settlement's secession-resistance when its trade is
  dominated by *intra-polity* partners (a trade bloc holds together even when the
  authority field is weak). It scales the direct/vassal/independent ratio up.
- **Peripheral-wealth secession** does the inverse: a rich settlement whose trade
  flows through *external* partners and which substantially out-earns its own
  claiming capital is flagged to break away as a new sovereign.

Neither touches the authority *field* — they are loyalty modifiers applied at the
classification step, consistent with the brief's "levers are modifiers, never
rewrites" guideline.
"""
from __future__ import annotations

from typing import Sequence, Union

import numpy as np

from core.params import param_field as _f, paramclass


@paramclass
class LoyaltyParams:
    """Stage 4 trade-attribution loyalty levers (each independently toggleable).

    Both default **off** (``0.0``): they are opt-in modifiers over the tuned base
    forest, not part of it. The internal-trade fraction is measured against the
    *current* claim partition each relaxation pass, so as a realm fragments the grip
    holding its periphery weakens — a deliberate feedback, not a bug.
    """

    economic_grip: float = _f(
        0.0, label="Economic grip (intra-polity trade resistance, 0 = off)", min=0.0, max=3.0
    )
    peripheral_wealth_ratio: float = _f(
        0.0, label="Peripheral secession wealth ratio (0 = off)", min=0.0, max=10.0
    )
    peripheral_external_floor: float = _f(
        0.5, label="Peripheral secession min external trade fraction", min=0.0, max=1.0
    )


def compute_internal_trade_fraction(
    settlement_faces: Union[Sequence[int], np.ndarray],
    partition: np.ndarray,
    trade_attribution: np.ndarray,
) -> np.ndarray:
    """Per-settlement fraction of bilateral trade with partners in the same cell.

    ``trade_attribution[i, j]`` is the volume exporter *i* sends importer *j*; the
    bilateral volume between two settlements is ``attr[i, j] + attr[j, i]``. For
    each settlement, returns ``internal / total`` where *internal* sums only over
    partners sharing its (claimed, ``>= 0``) ``partition`` cell. A settlement with
    no trade — or one in an unclaimed cell — gets ``0.0`` (no grip to apply).

    Returns ``(S,)`` float64 in [0, 1].
    """
    faces = np.asarray(settlement_faces, dtype=np.int64)
    s = faces.size
    attr = np.asarray(trade_attribution, dtype=np.float64)
    if s == 0 or attr.shape != (s, s):
        return np.zeros(s, dtype=np.float64)

    sym = attr + attr.T              # bilateral volume per pair
    np.fill_diagonal(sym, 0.0)       # ignore any self-trade
    cell = np.asarray(partition)[faces]                     # (S,) claim cell per settlement
    total = sym.sum(axis=1)
    same = (cell[:, None] == cell[None, :]) & (cell[None, :] >= 0)
    internal = (sym * same).sum(axis=1)

    frac = np.zeros(s, dtype=np.float64)
    nz = total > 0.0
    frac[nz] = internal[nz] / total[nz]
    return frac


def economic_grip_multiplier(internal_fraction: np.ndarray, grip_strength: float) -> np.ndarray:
    """Authority-ratio multiplier ``1 + grip_strength * internal_fraction`` (``>= 1``).

    A fully intra-polity settlement (``internal_fraction == 1``) gets the maximum
    ``1 + grip_strength`` resistance bonus; an outward-facing one (``0``) is
    unaffected.
    """
    return 1.0 + float(grip_strength) * np.asarray(internal_fraction, dtype=np.float64)


def peripheral_wealth_breakaways(
    settlement_faces: Union[Sequence[int], np.ndarray],
    partition: np.ndarray,
    capital_sidx: Union[Sequence[int], np.ndarray],
    trade_wealth: np.ndarray,
    internal_fraction: np.ndarray,
    *,
    wealth_ratio: float,
    external_floor: float,
) -> np.ndarray:
    """Flag settlements that should break away as new sovereigns (``(S,)`` bool).

    A non-capital settlement is flagged when **all** hold:

    - it is claimed (``partition >= 0``);
    - its external trade fraction (``1 - internal_fraction``) is at least
      ``external_floor`` — its economy faces *out* of the polity;
    - its trade wealth is at least ``wealth_ratio`` times its claiming capital's —
      a prosperous edge that does not need its liege.

    ``wealth_ratio <= 0`` disables the lever (returns all-False).
    """
    faces = np.asarray(settlement_faces, dtype=np.int64)
    s = faces.size
    mask = np.zeros(s, dtype=bool)
    if s == 0 or float(wealth_ratio) <= 0.0:
        return mask

    cell = np.asarray(partition)[faces]
    caps = np.asarray(capital_sidx, dtype=np.int64)
    tw = np.asarray(trade_wealth, dtype=np.float64)
    external = 1.0 - np.asarray(internal_fraction, dtype=np.float64)

    valid = cell >= 0
    cap_wealth_of_cell = tw[caps]                           # (C,) capital trade wealth per cell
    claim_capital_wealth = np.zeros(s, dtype=np.float64)
    claim_capital_wealth[valid] = cap_wealth_of_cell[cell[valid]]

    is_capital = np.zeros(s, dtype=bool)
    is_capital[caps] = True

    mask = (
        valid
        & ~is_capital
        & (external >= float(external_floor))
        & (tw >= float(wealth_ratio) * claim_capital_wealth)
    )
    return mask
