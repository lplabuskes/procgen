"""A settlement's profile: what kind of place it is, for the layout to read (plan phase 5).

:func:`build_profile` derives it from the world records, the realm channels and the
settlement map's terrain, **never writing back**:

- **who**: ``population``, ``tier`` (cities: :func:`population_display_tier`
  against the world's largest city — ``Settlement.tier`` is only a seeding label;
  then town, village) and ``kind`` (a town's or village's own; a city's from its
  site flags: ``port`` on sea or lake, ``river`` on a river or confluence, else
  ``inland``; every flag in ``site_kinds``);
- **walls**: the record's ``has_walls`` / ``wall_strength`` (villages: none);
- **polity** (:class:`PolityRelation`): the polity governing its face and its
  sovereign; ``control`` the city's ``control_class`` (a town or village: direct
  under a sovereign, vassal under a vassal); ``capital`` where a polity's
  ``capital_face`` is the city's face; ``march`` the frontier proxy — the face's
  ``border_tension`` against ``march_tension`` (clipped to 1), scaled by
  ``0.6 + 0.4 · autonomy`` of its vassal governor (direct rule: 0.6), and
  ``fort`` where the polity stage garrisoned the face;
- **wealth**: trade per head, as a percentile among its peers — a city's
  exports + imports (``trade_attribution`` row + column) per person among
  cities; a town's ``trade_flow`` at its face per person among towns; a
  village's face ``trade_flow`` among land faces;
- **industries** (:class:`Industry`, ranked): the town / village kind; the
  share of its countryside that mines or fishes (``mining_density`` /
  ``fishing_density`` over ``rural_density``); its crop (mills, breweries,
  sugar, weaving) by fertility and the share that farms; timber by forest;
  herding on pasture biomes; shipping on water; commerce by wealth; mineral
  workings and rare biological resources within ``resource_reach_km``, by
  nearness.  Each in [0, 1]; same-named entries merge; at most
  ``max_industries`` at or above ``min_industry``;
- **wind**: the annual-mean prevailing wind at its face, in frame axes (m/s),
  with its steadiness (|mean| / mean |wind|);
- **materials**: shares of stone / timber / brick / adobe from ruggedness (the
  world's ``ruggedness`` at the settlement, from ``stone_ruggedness_low`` to
  ``_high``), forest, aridity (biome), fertile clay plains, nearby marble and
  (cities) wealth;
- **danger**: the terrain's danger at the settlement;
- **stubs for the history layer**: ``founding_age`` (years; a seeded draw,
  older for more people and for capitals) and ``planned`` (seeded, likelier
  for waypoint and mining towns and marches).

Every input is optional: a world without polities, trade or winds gives the
neutral value.  Qt-free.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

from core.params import param_field as _f, paramclass
from localmap.villages import VILLAGE_KIND_NAMES, VILLAGE_MINING
from realmgen.channels import (
    BORDER_TENSION, CONTROL_CLASS, FORT_POPULATION, POLITY_ID, SOVEREIGN_ID, TRADE_FLOW,
)
from realmgen.placement.types import (
    SITE_KIND_CONFLUENCE, SITE_KIND_LAKE_COAST, SITE_KIND_OCEAN_COAST, SITE_KIND_RIVER, TIER_NAMES,
    population_display_tier,
)
from realmgen.polity.types import ControlClass
from realmgen.towns.types import TOWN_KIND_NAMES, TOWN_MINING
from region.upsample import RegionFields
from settlegen.frame import Frame
from settlegen.hinterland import village_farming
from worldgen.climate.classification import BIOME_NAMES
from worldgen.resources.crops import CROP_NAMES
from worldgen.resources.minerals import MARBLE, MINERAL_NAMES, SALT, mineral_workings
from worldgen.resources.rare_bio import RARE_BIO_NAMES

if TYPE_CHECKING:
    from core.world import World
    from settlegen.chain import SettlementMaps, SettlementRef

_EARTH_RADIUS_KM = 6371.0
# RNG stream tags for the stubs.
_FOUNDING_STREAM, _PLANNED_STREAM = 51, 52
_KIND_CODE = {"city": 0, "town": 1, "village": 2}

# Biome → aridity (building in earth) and pasture (herding).
_ARID = {"Desert": 1.0, "Semi-Arid": 0.8, "Steppe": 0.4, "Savannah": 0.3}
_PASTURE = {"Steppe": 0.6, "Savannah": 0.5, "Semi-Arid": 0.4, "Tundra": 0.3}
# Crop → (industry, weight) it supports.
_CROP_INDUSTRIES = {
    "Rice": [("milling", 1.0)],
    "Maize": [("milling", 1.0)],
    "Wheat": [("milling", 1.0), ("brewing", 0.4)],
    "Barley": [("brewing", 1.0), ("milling", 0.6)],
    "Millet": [("milling", 1.0), ("brewing", 0.5)],
    "Sugarcane": [("sugar", 1.0)],
    "Cotton": [("weaving", 1.0)],
    "Coffee": [("coffee", 1.0)],
}
_RARE_INDUSTRIES = {
    "Spices": "spice trade", "Tea": "tea", "Silk": "weaving", "Ivory": "carving", "Furs": "tanning",
    "Incense": "incense", "Dyes": "dyeing", "Amber": "carving", "Rubber": "rubber", "Pearls": "pearling",
}
# Planned-layout chance by kind (the history layer will decide; this is a stub).
_PLANNED_CHANCE = {"waypoint": 0.5, "mining": 0.35, "market": 0.15, "port": 0.15, "river": 0.15}
_PLANNED_DEFAULT = {"city": 0.1, "town": 0.15, "village": 0.05}


@paramclass
class SettlementProfileParams:
    """Tuning for the settlement profile."""

    resource_reach_km: float = _f(10.0, label="Resource reach (km)", min=1.0, max=50.0, step=1.0)
    # Border tension at which a face counts as a full march (seed 4 land: p90 ~0.11, max ~0.5).
    march_tension: float = _f(0.2, label="March border tension", min=0.01, max=1.0, step=0.01)
    min_industry: float = _f(0.1, label="Industry floor", min=0.0, max=1.0, step=0.05)
    max_industries: int = _f(8, label="Industries kept", min=1, max=20)
    # Ruggedness at the settlement from which building turns to stone (none at low, full at high).
    stone_ruggedness_low: float = _f(0.25, label="Stone from ruggedness", min=0.0, max=1.0, step=0.05)
    stone_ruggedness_high: float = _f(0.65, label="All-stone ruggedness", min=0.0, max=1.0, step=0.05)


@dataclass
class PolityRelation:
    """How the settlement stands in its polity."""

    polity_id: int = -1           # governing polity (-1: none)
    sovereign_id: int = -1
    control: str = "none"         # "direct" / "vassal" / "independent" / "none"
    capital: bool = False         # a polity's seat
    sovereign_capital: bool = False
    depth: int = 0                # of the governing polity (0: sovereign)
    autonomy: float = 0.0         # of the governing polity (0: sovereign)
    march: float = 0.0            # [0, 1] frontier proxy
    fort: bool = False            # a garrison on its face


@dataclass
class Industry:
    """One trade the settlement leans on."""

    name: str
    weight: float                                  # [0, 1]
    goods: tuple[str, ...] = ()                    # what feeds it (crop, minerals, …)


@dataclass
class SettlementProfile:
    """See the module docstring."""

    population: float
    tier: int                     # 0–3 city display tiers, 4 town, 5 village
    tier_name: str
    kind: str
    site_kinds: tuple[str, ...]
    walls: bool
    wall_strength: float
    polity: PolityRelation
    wealth: float                 # [0, 1] percentile among peers
    trade: float                  # the raw measure behind it
    industries: list[Industry]
    wind_xy: np.ndarray           # (2,) m/s in frame axes, toward which it blows
    wind_steadiness: float        # [0, 1]
    materials: dict[str, float]   # stone / timber / brick / adobe shares, sum 1
    danger: float
    biome: str
    founding_age: int             # STUB (history layer): years
    planned: bool                 # STUB (history layer)
    face_id: int = -1
    detail: dict = field(default_factory=dict)

    @property
    def material(self) -> str:
        return max(self.materials, key=lambda k: self.materials[k])

    def describe(self) -> list[str]:
        """Short lines for hover and debugging."""
        pol = self.polity
        wind = float(np.hypot(*self.wind_xy))
        heading = (np.degrees(np.arctan2(self.wind_xy[0], self.wind_xy[1])) + 360.0) % 360.0
        lines = [
            f"{self.tier_name} ({self.kind}), {self.population:,.0f} people",
            ("walls %.2f" % self.wall_strength) if self.walls else "no walls",
            f"{pol.control}" + (" capital" if pol.capital else "") + (f", march {pol.march:.2f}" if pol.march else "")
            + (", fort" if pol.fort else ""),
            f"wealth {self.wealth:.2f}, danger {self.danger:.2f}",
            "industries: " + ", ".join(f"{i.name} {i.weight:.2f}" for i in self.industries),
            f"wind {wind:.1f} m/s toward {heading:.0f}° (frame), steadiness {self.wind_steadiness:.2f}",
            f"{self.biome}; builds in " + ", ".join(f"{k} {v:.2f}" for k, v in
                                                    sorted(self.materials.items(), key=lambda kv: -kv[1])),
            f"founded ~{self.founding_age} years ago (stub), " + ("planned" if self.planned else "organic") + " (stub)",
        ]
        return lines


# ── pieces (pure, for tests) ─────────────────────────────────────────────────

def percentile_of(value: float, peers: np.ndarray) -> float:
    """Share of ``peers`` strictly below ``value``, plus half the ties: [0, 1]."""
    peers = np.asarray(peers, dtype=np.float64)
    if peers.size == 0:
        return 0.5
    return float((np.sum(peers < value) + 0.5 * np.sum(peers == value)) / peers.size)


def city_kind(site_kind: int) -> tuple[str, tuple[str, ...]]:
    """``(kind, every site flag)`` of a city's site flags."""
    flags = tuple(name for bit, name in ((SITE_KIND_OCEAN_COAST, "sea"), (SITE_KIND_LAKE_COAST, "lake"),
                                         (SITE_KIND_RIVER, "river"), (SITE_KIND_CONFLUENCE, "confluence"))
                  if site_kind & bit)
    if site_kind & (SITE_KIND_OCEAN_COAST | SITE_KIND_LAKE_COAST):
        return "port", flags
    if site_kind & (SITE_KIND_RIVER | SITE_KIND_CONFLUENCE):
        return "river", flags
    return "inland", flags


def march_score(tension: float, autonomy: float, vassal: bool, march_tension: float) -> float:
    """The frontier proxy: tension against ``march_tension``, a vassal's autonomy raising it."""
    frontier = float(np.clip(tension / max(march_tension, 1e-9), 0.0, 1.0))
    return frontier * (0.6 + 0.4 * float(np.clip(autonomy, 0.0, 1.0)) if vassal else 0.6)


def merge_industries(entries: list[Industry], min_weight: float, max_count: int) -> list[Industry]:
    """Same-named entries merged (strongest weight, every good), ranked, floored and capped."""
    merged: dict[str, Industry] = {}
    for e in entries:
        w = float(np.clip(e.weight, 0.0, 1.0))
        if e.name in merged:
            m = merged[e.name]
            m.weight = max(m.weight, w)
            m.goods = tuple(dict.fromkeys(m.goods + e.goods))
        else:
            merged[e.name] = Industry(e.name, w, tuple(e.goods))
    ranked = sorted(merged.values(), key=lambda i: (-i.weight, i.name))
    return [i for i in ranked if i.weight >= min_weight and i.weight > 0.0][:max_count]


def material_shares(rugged: float, forest: float, arid: float, fertility: float, marble: float,
                    wealth: float) -> dict[str, float]:
    """Stone / timber / brick / adobe shares (sum 1) from [0, 1] inputs."""
    rugged, forest, arid, fertility, marble, wealth = (
        float(np.clip(v, 0.0, 1.0)) for v in (rugged, forest, arid, fertility, marble, wealth))
    score = {
        "stone": 0.1 + rugged + 0.5 * marble + 0.3 * wealth,
        "timber": 1.5 * forest * (1.0 - arid),
        "brick": 0.5 * (1.0 - rugged) * (1.0 - forest) * (1.0 - arid) + 0.3 * fertility * (1.0 - arid),
        "adobe": 1.5 * arid * (1.0 - forest),
    }
    total = sum(score.values())
    return {k: v / total for k, v in score.items()}


def wind_in_frame(frame: Frame, at: np.ndarray, vectors: np.ndarray) -> tuple[np.ndarray, float]:
    """``(mean (2,) m/s in frame axes, steadiness)`` of seasonal ``(K, 3)`` tangent wind vectors at ``at``."""
    vectors = np.asarray(vectors, dtype=np.float64).reshape(-1, 3)
    if vectors.size == 0:
        return np.zeros(2), 0.0
    mean = vectors.mean(axis=0)
    speed = float(np.linalg.norm(vectors, axis=1).mean())
    at = np.asarray(at, dtype=np.float64).reshape(3)
    at = at / np.linalg.norm(at)
    tangent = mean - at * float(mean @ at)
    norm = float(np.linalg.norm(tangent))
    if norm <= 1e-9:
        return np.zeros(2), 0.0
    step = 1e-5                                       # radians: ~64 m, well inside the frame
    tip = at + step * tangent / norm
    xy = frame.to_xy(np.stack([at, tip / np.linalg.norm(tip)]))
    heading = xy[1] - xy[0]
    heading = heading / max(float(np.linalg.norm(heading)), 1e-12)
    return heading * norm, float(np.clip(norm / max(speed, 1e-9), 0.0, 1.0))


def founding_age(rng: np.random.Generator, population: float, capital: bool) -> int:
    """STUB: years since founding — median ~100 at 500 people, rising as people^0.35, ×1.5 for a capital."""
    median = 100.0 * (max(population, 1.0) / 500.0) ** 0.35 * (1.5 if capital else 1.0)
    return int(np.clip(round(median * float(rng.lognormal(0.0, 0.5))), 10, 3000))


def settlement_key(globe: "World", ref: "SettlementRef") -> tuple[int, ...]:
    """The settlement's RNG key: (world seed, kind, index[, region, hex])."""
    return (int(globe.seed or 0) & 0xFFFFFFFF, _KIND_CODE[ref.kind], int(ref.index), *(ref.map or ()))


# ── the profile ──────────────────────────────────────────────────────────────

def _meta(globe: "World", face: int, key: str, default: float) -> float:
    meta = globe.mesh.face_metadata
    if face < 0 or face >= len(meta):
        return default
    return float(meta[face].get(key, default))


def _at_center(fields: RegionFields, key: str, default: float = 0.0) -> float:
    tex = fields.textures.get(key)
    if tex is None:
        return default
    t = fields.tex_size // 2
    return float(np.asarray(tex)[0, t, t, 0])


def _nearness(center: np.ndarray, positions: np.ndarray, reach_km: float) -> np.ndarray:
    """``1 − distance / reach`` (≥ 0) of each point from ``center``."""
    if len(positions) == 0:
        return np.zeros(0)
    p = np.asarray(positions, dtype=np.float64)
    p = p / np.linalg.norm(p, axis=1, keepdims=True)
    c = np.asarray(center, dtype=np.float64) / np.linalg.norm(center)
    d_km = np.arccos(np.clip(p @ c, -1.0, 1.0)) * _EARTH_RADIUS_KM
    return np.clip(1.0 - d_km / reach_km, 0.0, 1.0)


def _polity(globe: "World", face: int, is_city: bool, params: SettlementProfileParams) -> PolityRelation:
    meta = globe.mesh.face_metadata
    if face < 0 or face >= len(meta) or POLITY_ID.name not in meta[face]:
        return PolityRelation()
    pid = int(meta[face].get(POLITY_ID.name, -1))
    polities = globe.get_feature("polities") or []
    by_id = {p.polity_id: p for p in polities}
    gov = by_id.get(pid)
    depth = int(gov.depth) if gov is not None else 0
    autonomy = float(gov.autonomy) if gov is not None else 0.0
    if pid < 0:
        control = "none"
    elif is_city and int(meta[face].get(CONTROL_CLASS.name, -1)) >= 0:
        control = ControlClass(int(meta[face][CONTROL_CLASS.name])).name.lower()
    else:
        control = "vassal" if depth > 0 else "direct"
    seats = [p for p in polities if int(p.capital_face) == face] if is_city else []
    return PolityRelation(
        polity_id=pid, sovereign_id=int(meta[face].get(SOVEREIGN_ID.name, -1)), control=control,
        capital=bool(seats), sovereign_capital=any(p.depth == 0 for p in seats), depth=depth,
        autonomy=autonomy,
        march=march_score(float(meta[face].get(BORDER_TENSION.name, 0.0)), autonomy, depth > 0,
                          float(params.march_tension)),
        fort=float(meta[face].get(FORT_POPULATION.name, 0.0)) > 0.0,
    )


def _wealth(globe: "World", kind: str, index: int, face: int) -> tuple[float, float]:
    """``(percentile among peers, trade per head)``.  See the module docstring."""
    mesh = globe.mesh
    if kind == "city":
        cities = globe.get_feature("settlements") or []
        ta = globe.get_feature("trade_attribution")
        if ta is None or np.shape(ta) != (len(cities), len(cities)) or not cities:
            return 0.5, 0.0
        ta = np.asarray(ta, dtype=np.float64)
        pops = np.maximum(np.array([c.population for c in cities], dtype=np.float64), 1.0)
        per_head = (ta.sum(axis=0) + ta.sum(axis=1)) / pops
        return percentile_of(per_head[index], per_head), float(per_head[index])
    if not TRADE_FLOW.present(mesh):
        return 0.5, 0.0
    flow = TRADE_FLOW.read(mesh).astype(np.float64)
    if kind == "town":
        towns = globe.get_feature("towns")
        per_head = flow[np.asarray(towns.face_id)] / np.maximum(np.asarray(towns.population, dtype=np.float64), 1.0)
        return percentile_of(per_head[index], per_head), float(per_head[index])
    land = np.array([not m.get("is_water", False) for m in mesh.face_metadata])
    value = float(flow[face]) if 0 <= face < len(flow) else 0.0
    return percentile_of(value, flow[land]), value


def _industries(
    globe: "World", fields: RegionFields, center: np.ndarray, kind: str, sub_kind: str, site_kinds: tuple[str, ...],
    wealth: float, farming_share: float, params: SettlementProfileParams,
) -> list[Industry]:
    out: list[Industry] = []
    rural = _at_center(fields, "rural_density")
    mining = _at_center(fields, "mining_density") / rural if rural > 0 else 0.0
    fishing = _at_center(fields, "fishing_density") / rural if rural > 0 else 0.0
    farm = float(np.clip(min(farming_share, 1.0 - mining - fishing), 0.0, 1.0))
    if sub_kind == "mining":
        out.append(Industry("mining", 1.0))
    elif mining > 0:
        out.append(Industry("mining", min(1.0, 2.0 * mining)))
    if sub_kind in ("port", "fishing") or fishing > 0:
        out.append(Industry("fishing", max(0.8 if sub_kind in ("port", "fishing") else 0.0, min(1.0, 2.0 * fishing))))
    if "sea" in site_kinds or "lake" in site_kinds or sub_kind == "port":
        out.append(Industry("shipping", 0.8))
    elif "river" in site_kinds or "confluence" in site_kinds or sub_kind == "river":
        out.append(Industry("shipping", 0.5))
    if sub_kind == "waypoint":
        out.append(Industry("hostelry", 0.8))
    elif sub_kind == "roadside":
        out.append(Industry("hostelry", 0.4))
    commerce = wealth if kind != "village" else 0.5 * wealth
    out.append(Industry("commerce", max(commerce, 0.8 if sub_kind == "market" else 0.0)))
    crop_idx = int(_at_center(fields, "dominant_crop"))
    crop = CROP_NAMES[crop_idx] if 0 <= crop_idx < len(CROP_NAMES) else "None"
    fert = float(np.clip(_at_center(fields, "fertility"), 0.0, 1.0))
    for name, w in _CROP_INDUSTRIES.get(crop, []):
        out.append(Industry(name, w * fert * farm, (crop,)))
    forest = float(np.clip(_at_center(fields, "forest_density"), 0.0, 1.0))
    out.append(Industry("timber", forest))
    biome = _biome(fields)
    if biome in _PASTURE:
        out.append(Industry("herding", _PASTURE[biome] * farm))
        out.append(Industry("tanning", 0.5 * _PASTURE[biome] * farm, ("hides",)))
        out.append(Industry("weaving", 0.5 * _PASTURE[biome] * farm, ("wool",)))
    reach = float(params.resource_reach_km)
    workings = mineral_workings(globe)
    minerals = globe.get_feature("minerals")
    if workings is not None and minerals is not None:
        pos, dep, _ = workings
        near = _nearness(center, pos, reach)
        kinds = np.asarray(minerals[1], dtype=np.int64)[dep]
        for k in np.unique(kinds[near > 0]):
            w = 0.3 + 0.7 * float(near[kinds == k].max())
            name = MINERAL_NAMES[int(k)]
            industry = "salt works" if k == SALT else "quarrying" if k == MARBLE else "mining"
            out.append(Industry(industry, w, (name,)))
            if name in ("Iron", "Coal"):
                out.append(Industry("smithing", 0.7 * w, (name,)))
    rare = globe.get_feature("rare_bio")
    if rare is not None and len(rare[0]):
        near = _nearness(center, np.asarray(rare[0]), reach)
        kinds = np.asarray(rare[1], dtype=np.int64)
        for k in np.unique(kinds[near > 0]):
            name = RARE_BIO_NAMES[int(k)]
            out.append(Industry(_RARE_INDUSTRIES[name], 0.3 + 0.7 * float(near[kinds == k].max()), (name,)))
    return merge_industries(out, float(params.min_industry), int(params.max_industries))


def _biome(fields: RegionFields) -> str:
    b = int(_at_center(fields, "biome", -1))
    return BIOME_NAMES[b] if 0 <= b < len(BIOME_NAMES) else "Unknown"


def _stone_ruggedness(fields: RegionFields, params: SettlementProfileParams) -> float:
    """The materials' ruggedness in [0, 1]: the world's ``ruggedness`` at the settlement, ramped."""
    lo, hi = float(params.stone_ruggedness_low), float(params.stone_ruggedness_high)
    rho = _at_center(fields, "ruggedness", lo)
    return float(np.clip((rho - lo) / max(hi - lo, 1e-9), 0.0, 1.0))


def _relief_m(fields: RegionFields) -> float:
    """The frame's land relief: 2nd–98th percentile of elevation (m)."""
    elev = fields.textures["elevation"][0, ..., 0].astype(np.float64) * 1000.0
    wet = np.zeros(elev.shape, dtype=bool)
    for key in ("ocean_mask", "lake_mask"):
        if key in fields.textures:
            wet |= fields.textures[key][0, ..., 0] > 0.5
    land = elev[~wet]
    if land.size < 2:
        return 0.0
    lo, hi = np.percentile(land, [2.0, 98.0])
    return float(hi - lo)


def build_profile(
    globe: "World", fields: RegionFields, frame: Frame, chain: "SettlementMaps",
    params: Optional[SettlementProfileParams] = None,
) -> SettlementProfile:
    """The profile of ``chain``'s settlement.  See the module docstring."""
    params = params or SettlementProfileParams()
    ref = chain.ref
    center = np.asarray(chain.center, dtype=np.float64)
    t = fields.tex_size // 2
    centre_face = int(fields.faces[0, t, t])
    walls, wall_strength, farming_share = False, 0.0, 1.0
    site_kinds: tuple[str, ...] = ()
    if ref.kind == "city":
        cities = globe.get_feature("settlements")
        city = cities[ref.index]
        face = int(city.face_id)
        top = max(float(c.population) for c in cities)
        tier = population_display_tier(float(city.population), top)
        tier_name = TIER_NAMES[tier]
        kind, site_kinds = city_kind(int(city.site_kind))
        walls, wall_strength = bool(city.has_walls), float(city.wall_strength)
    elif ref.kind == "town":
        towns = globe.get_feature("towns")
        face = int(towns.face_id[ref.index])
        tier, tier_name = 4, "Town"
        kind = TOWN_KIND_NAMES[int(towns.kind[ref.index])].split()[0].lower()
        walls, wall_strength = bool(towns.has_walls[ref.index]), float(towns.wall_strength[ref.index])
        if int(towns.kind[ref.index]) == TOWN_MINING:
            farming_share = 0.0
    else:
        villages = chain.owner.villages
        face = centre_face
        tier, tier_name = 5, "Village"
        code = int(villages.kind[ref.index])
        kind = {"Farming": "farming", "River": "river", "Fishing": "fishing", "Roadside": "roadside",
                "Mining": "mining"}[VILLAGE_KIND_NAMES[code].split()[0]]
        farming_share = float(village_farming(villages)[ref.index]) if code != VILLAGE_MINING else 0.0
    population = float(chain.population)
    polity = _polity(globe, face, ref.kind == "city", params)
    wealth, trade = _wealth(globe, ref.kind, ref.index, face)
    biome = _biome(fields)
    industries = _industries(globe, fields, center, ref.kind, kind, site_kinds, wealth, farming_share, params)

    winds = globe.get_feature("winds")
    if winds and 0 <= face < len(winds[0][1]):
        wind_xy, steadiness = wind_in_frame(frame, center, np.stack([np.asarray(v)[face] for _, v in winds]))
    else:
        wind_xy, steadiness = np.zeros(2), 0.0

    reach = float(params.resource_reach_km)
    marble = 0.0
    workings, minerals = mineral_workings(globe), globe.get_feature("minerals")
    if workings is not None and minerals is not None:
        is_marble = np.asarray(minerals[1], dtype=np.int64)[workings[1]] == MARBLE
        near = _nearness(center, workings[0][is_marble], reach)
        marble = float(near.max()) if near.size else 0.0
    materials = material_shares(
        rugged=_stone_ruggedness(fields, params),
        forest=_at_center(fields, "forest_density"), arid=_ARID.get(biome, 0.0),
        fertility=_at_center(fields, "fertility"), marble=marble,
        wealth=wealth if ref.kind == "city" else 0.0,
    )

    key = settlement_key(globe, ref)
    age = founding_age(np.random.default_rng((*key, _FOUNDING_STREAM)), population, polity.capital)
    chance = _PLANNED_CHANCE.get(kind, _PLANNED_DEFAULT[ref.kind])
    if polity.march >= 0.5:
        chance = max(chance, 0.4)
    planned = bool(np.random.default_rng((*key, _PLANNED_STREAM)).random() < chance)

    return SettlementProfile(
        population=population, tier=tier, tier_name=tier_name, kind=kind, site_kinds=site_kinds,
        walls=walls, wall_strength=wall_strength, polity=polity, wealth=wealth, trade=trade,
        industries=industries, wind_xy=wind_xy, wind_steadiness=steadiness, materials=materials,
        danger=_at_center(fields, "danger"), biome=biome, founding_age=age, planned=planned, face_id=face,
        detail={"relief_m": _relief_m(fields), "marble": marble, "planned_chance": chance},
    )
