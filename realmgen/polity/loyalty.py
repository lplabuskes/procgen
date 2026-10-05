"""Stage 4 loyalty modifiers — economic grip and peripheral-wealth secession.

Both read the ``trade_attribution`` world feature (``(S, S)``, aligned with
``settlements``) against the current claim partition, and act only at
classification, never on the authority field.

- **Economic grip** scales up the authority ratio of settlements trading mostly
  within their own polity.
- **Peripheral-wealth secession** flags rich, outward-trading settlements that
  out-earn their claiming capital to break away as new sovereigns.
"""
from __future__ import annotations

from typing import Sequence, Union

import numpy as np

from core.params import param_field as _f, paramclass


@paramclass
class LoyaltyParams:
    """Stage 4 loyalty levers; both default off (``0.0``).

    The internal-trade fraction is re-measured each relaxation pass, so as a realm
    fragments the grip on its periphery weakens (deliberate feedback).
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

    ``trade_attribution[i, j]`` is the volume exporter *i* sends importer *j*;
    bilateral volume is ``attr[i, j] + attr[j, i]``. No trade or an unclaimed
    cell gives ``0.0``.

    Returns ``(S,)`` float64 in [0, 1].
    """
    faces = np.asarray(settlement_faces, dtype=np.int64)
    s = faces.size
    attr = np.asarray(trade_attribution, dtype=np.float64)
    if s == 0 or attr.shape != (s, s):
        return np.zeros(s, dtype=np.float64)

    sym = attr + attr.T
    np.fill_diagonal(sym, 0.0)
    cell = np.asarray(partition)[faces]
    total = sym.sum(axis=1)
    same = (cell[:, None] == cell[None, :]) & (cell[None, :] >= 0)
    internal = (sym * same).sum(axis=1)

    frac = np.zeros(s, dtype=np.float64)
    nz = total > 0.0
    frac[nz] = internal[nz] / total[nz]
    return frac


def economic_grip_multiplier(internal_fraction: np.ndarray, grip_strength: float) -> np.ndarray:
    """Authority-ratio multiplier ``1 + grip_strength * internal_fraction`` (``>= 1``)."""
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

    A claimed non-capital settlement is flagged when its external trade fraction
    is ``>= external_floor`` and its trade wealth is ``>= wealth_ratio`` times its
    claiming capital's. ``wealth_ratio <= 0`` disables the lever.
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
    cap_wealth_of_cell = tw[caps]
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
