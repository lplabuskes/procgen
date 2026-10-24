"""Who lives in a settlement: households by what they do, and how much ground each holds.

Not every household farms.  Every place keeps its everyday tradesmen — smiths,
carpenters, alehouses, bakers — at about one per ``people_per`` people (none
below ``min_people``), more where the profile's industries lean on the trade;
a Poisson draw makes the count.  A village's other households farm, or (its
``1 − farming_share``) fish or mine and live in cottages.  A farm toft is solved
so the village's plots keep their total ground; a trade's plot is
``craft_plot_scale`` of it, a cottage's ``cottage_plot_scale``.

Towns and cities spread their everyday trades (``Trade.urban``) through their
houses (:func:`settlegen.layout.districts.disperse_trades`); the noxious and wet
trades keep to their quarters there.  Geometry-free.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from settlegen.layout.params import SettlementLayoutParams

if TYPE_CHECKING:
    from settlegen.profile.builder import SettlementProfile


@dataclass(frozen=True)
class Trade:
    """One everyday trade."""

    label: str                       # what its plot reads as on hover
    people_per: float                # one practitioner per this many people
    min_people: float                # none in a place smaller
    industries: tuple[str, ...] = ()  # profile industries that raise the count (× 1 + weight)
    urban: bool = True               # spread through a town's or city's houses


TRADES: dict[str, Trade] = {
    "smith": Trade("Smithy", 200.0, 120.0, ("smithing", "mining")),
    "carpenter": Trade("Carpenter's yard", 250.0, 150.0, ("timber",)),
    "alehouse": Trade("Alehouse", 200.0, 120.0, ("brewing", "hostelry")),
    "baker": Trade("Bakehouse", 350.0, 250.0, ("milling",)),
    "miller": Trade("Mill", 500.0, 150.0, ("milling",), urban=False),
    "weaver": Trade("Weaver's", 250.0, 150.0, ("weaving",), urban=False),
    "cobbler": Trade("Cobbler's", 400.0, 300.0, ("tanning",)),
    "butcher": Trade("Butcher's", 500.0, 350.0, ("herding",), urban=False),
    "cooper": Trade("Cooper's", 700.0, 400.0, ("brewing", "fishing", "shipping"), urban=False),
    "potter": Trade("Pottery", 700.0, 400.0, ("quarrying",), urban=False),
}

# Households that are not trades: what their plots read as on hover.
HOUSEHOLD_LABELS = {"farm": "Farmstead", "fisher": "Fisher's cottage", "miner": "Miner's cottage"}


def use_label(use: str) -> str:
    """How a lot of ``use`` reads ("" when it is just its district's)."""
    if use in TRADES:
        return TRADES[use].label
    return HOUSEHOLD_LABELS.get(use, "")


@dataclass
class Household:
    """One household: what it does and its plot's area relative to a farm toft."""

    use: str          # "farm", "fisher", "miner", or a trade
    scale: float

    @property
    def trade(self) -> bool:
        return self.use in TRADES


def trade_counts(profile: "SettlementProfile", params: SettlementLayoutParams, rng: np.random.Generator,
                 urban_only: bool = False) -> dict[str, int]:
    """Practitioners of each everyday trade (``urban_only``: just those a town spreads)."""
    pop = float(profile.population)
    weight = {i.name: float(i.weight) for i in profile.industries}
    out: dict[str, int] = {}
    for name, t in TRADES.items():
        if (urban_only and not t.urban) or pop < t.min_people:
            continue
        boost = 1.0 + max([weight.get(i, 0.0) for i in t.industries] or [0.0])
        n = int(rng.poisson(pop / t.people_per * boost * float(params.trade_people_mult)))
        if n:
            out[name] = n
    return out


def households(profile: "SettlementProfile", params: SettlementLayoutParams,
               rng: np.random.Generator) -> list[Household]:
    """A village's households: trades first, then cottages (fishers, miners), then farms.
    At least 3; trades at most ``max_trade_share`` of them."""
    n = max(int(round(float(profile.population) / float(params.household_people))), 3)
    trades = trade_counts(profile, params, rng)
    names = [name for name, k in trades.items() for _ in range(k)]
    cap = int(float(params.max_trade_share) * n)
    if len(names) > cap:
        names = [names[i] for i in np.sort(rng.choice(len(names), size=cap, replace=False))]
    craft = float(params.craft_plot_scale)
    out = [Household(name, craft) for name in names]
    rest = n - len(out)
    share = float(np.clip(profile.farming_share, 0.0, 1.0))
    cottagers = int(round((1.0 - share) * rest))
    cottage = "miner" if profile.kind == "mining" else "fisher"
    out += [Household(cottage, float(params.cottage_plot_scale)) for _ in range(cottagers)]
    out += [Household("farm", 1.0) for _ in range(rest - cottagers)]
    return out


def farm_toft_m2(households_: list[Household], ground_m2: float, lo: float = 300.0, hi: float = 3000.0) -> float:
    """The farm toft's area: every household's plot (``scale`` × it) together cover ``ground_m2``."""
    total = sum(h.scale for h in households_)
    return float(np.clip(ground_m2 / max(total, 1e-9), lo, hi))
