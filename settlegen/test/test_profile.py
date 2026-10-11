"""The settlement profile: its pieces, and a whole profile on the synthetic globe."""
import numpy as np
import pytest

from localmap.test.test_store import REGION_ID, small_store
from realmgen.placement.types import SITE_KIND_CONFLUENCE, SITE_KIND_LAKE_COAST, SITE_KIND_RIVER
from region.geometry import ecef_to_cell
from region.test.test_builder import land_towns
from region.test.test_resources import globe  # noqa: F401  (fixture)
from settlegen.chain import SettlementRef
from settlegen.profile import (
    Industry, city_kind, founding_age, march_score, material_shares, merge_industries, percentile_of, wind_in_frame,
)


def test_percentile_counts_half_the_ties():
    peers = np.array([1.0, 2.0, 2.0, 3.0])
    assert percentile_of(0.0, peers) == 0.0
    assert percentile_of(2.0, peers) == pytest.approx(0.5)
    assert percentile_of(9.0, peers) == 1.0
    assert percentile_of(1.0, np.zeros(0)) == 0.5


def test_city_kind_prefers_the_shore_and_keeps_every_flag():
    assert city_kind(SITE_KIND_LAKE_COAST | SITE_KIND_RIVER) == ("port", ("lake", "river"))
    assert city_kind(SITE_KIND_CONFLUENCE) == ("river", ("confluence",))
    assert city_kind(0) == ("inland", ())


def test_march_rises_with_tension_and_a_vassals_autonomy():
    assert march_score(0.0, 1.0, True, 0.2) == 0.0
    assert march_score(1.0, 0.0, False, 0.2) == pytest.approx(0.6)
    assert march_score(1.0, 1.0, True, 0.2) == pytest.approx(1.0)
    assert march_score(0.1, 0.5, True, 0.2) == pytest.approx(0.5 * 0.8)


def test_industries_merge_rank_floor_and_cap():
    entries = [Industry("mining", 0.4, ("Iron",)), Industry("mining", 0.7, ("Coal",)), Industry("timber", 0.05),
               Industry("milling", 0.5), Industry("brewing", 0.5), Industry("fishing", 1.5)]
    out = merge_industries(entries, 0.1, 3)
    assert [i.name for i in out] == ["fishing", "mining", "brewing"]
    assert out[0].weight == 1.0
    assert out[1].weight == pytest.approx(0.7) and out[1].goods == ("Iron", "Coal")


def test_materials_follow_the_land():
    def top(**kw):
        base = dict(rugged=0.0, forest=0.0, arid=0.0, fertility=0.0, marble=0.0, wealth=0.0)
        shares = material_shares(**{**base, **kw})
        assert sum(shares.values()) == pytest.approx(1.0)
        return max(shares, key=lambda k: shares[k])

    assert top(rugged=1.0) == "stone"
    assert top(forest=0.9) == "timber"
    assert top(arid=1.0) == "adobe"
    assert top(fertility=0.8) == "brick"


def test_founding_age_is_older_for_bigger_places_and_capitals():
    def median(pop, capital):
        return np.median([founding_age(np.random.default_rng(k), pop, capital) for k in range(400)])

    assert median(50_000, False) > 2.0 * median(500, False)
    assert median(5_000, True) > 1.3 * median(5_000, False)


class _Frame:
    """East is +x, north +y, at the north pole's tangent plane along x = ECEF x."""

    def to_xy(self, ecef):
        return np.asarray(ecef)[:, :2] * 6.371e6


def test_wind_points_where_it_blows_with_its_steadiness():
    at = np.array([0.0, 0.0, 1.0])
    steady, _ = wind_in_frame(_Frame(), at, np.array([[3.0, 0.0, 0.0], [5.0, 0.0, 0.0]]))  # type: ignore[arg-type]
    assert steady == pytest.approx([4.0, 0.0], abs=1e-6)
    xy, s = wind_in_frame(_Frame(), at, np.array([[0.0, 4.0, 0.0], [0.0, -2.0, 0.0]]))  # type: ignore[arg-type]
    assert xy == pytest.approx([0.0, 1.0], abs=1e-6)
    assert s == pytest.approx(1.0 / 3.0)
    assert wind_in_frame(_Frame(), at, np.zeros((0, 3)))[1] == 0.0  # type: ignore[arg-type]


@pytest.fixture
def towns(globe):  # noqa: F811
    towns = land_towns(globe)
    globe.add_feature("towns", towns)
    yield towns
    globe.remove_feature("towns")


def test_a_town_gets_a_whole_profile_without_realm_features(globe, towns):  # noqa: F811
    from settlegen.builder import build_settlement_map
    from settlegen.params import SettlementMapParams
    from settlegen.upsample import SettlementUpsampleParams

    k = int(np.flatnonzero(ecef_to_cell(towns.site_positions, globe.antimeridian) == REGION_ID)[0])
    built = build_settlement_map(small_store(globe), SettlementRef("town", k),
                                 SettlementMapParams(upsample=SettlementUpsampleParams(texel_m=200.0)))
    profile = built.profile
    assert profile.population == pytest.approx(float(towns.population[k]))
    assert profile.tier_name == "Town"
    assert profile.polity.control == "none" and profile.polity.march == 0.0
    assert sum(profile.materials.values()) == pytest.approx(1.0)
    assert all(0.0 <= i.weight <= 1.0 for i in profile.industries)
    assert profile.founding_age >= 10
    assert profile.describe()
    again = build_settlement_map(small_store(globe), SettlementRef("town", k),
                                 SettlementMapParams(upsample=SettlementUpsampleParams(texel_m=200.0))).profile
    assert (again.founding_age, again.planned) == (profile.founding_age, profile.planned)
