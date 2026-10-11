"""Which water a face's texels reach, and the water links cities are tagged with."""
import numpy as np

from realmgen.trade.water_links import (
    TAG_LAKE, TAG_OCEAN, WaterAccess, settlement_water_links, water_link_mask,
)


def _access() -> WaterAccess:
    """0: land, coast on the ocean; 1: river face (river 5); 2: river face (rivers 5, 6);
    3: ocean; 4: lake; 5: land, no coast; 6: river face (river 7)."""
    return WaterAccess(
        rivers=[frozenset(), frozenset({5}), frozenset({5, 6}), frozenset({5}), frozenset(),
                frozenset(), frozenset({7})],
        coast_ocean=np.array([True, False, False, False, False, False, False]),
        coast_lake=np.array([False, False, True, False, False, False, False]),
        ocean=np.array([False, False, False, True, False, False, False]),
        land_passable=np.array([True, True, True, False, False, True, True]),
        sea_passable=np.array([False, True, True, True, True, False, True]),
    )


def test_a_port_needs_coast_of_the_waters_kind():
    a = _access()
    assert a.link_tags(0, 3) == (TAG_OCEAN,)
    assert a.link_tags(0, 4) == ()            # ocean coast, lake neighbour
    assert a.link_tags(5, 3) == ()            # no coast at all


def test_a_river_face_reaches_on_by_a_shared_river_or_coast():
    a = _access()
    assert a.link_tags(1, 2) == (5,)
    assert a.link_tags(1, 3) == (5,)          # the river flows on into the sea face
    assert a.link_tags(2, 4) == (TAG_LAKE,)
    assert a.link_tags(1, 6) == ()
    assert a.sea_edge_ok(2, 1) and a.sea_edge_ok(3, 1) and a.sea_edge_ok(3, 4)
    assert not a.sea_edge_ok(1, 6) and not a.sea_edge_ok(4, 1)


def test_settlement_links_merge_by_water_strongest_first_and_drop_unreachable():
    a = _access()
    tags = settlement_water_links(a, [0, 2, 5], [
        [(3, 1.0)],
        [(1, 1.0), (4, 3.0), (3, 0.5)],      # 1 and 3 are both river 5: 1.5 < 3.0
        [(3, 9.0)],
    ])
    assert tags == [[(TAG_OCEAN,)], [(TAG_LAKE,), (5,)], []]


FLOWS = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 40.0, 20.0])


def _mask(links, ocean, lake, rivers):
    return water_link_mask(links, np.array(ocean, dtype=bool), np.array(lake, dtype=bool),
                           lambda r: np.array(rivers[r], dtype=bool), FLOWS)


def test_mask_takes_candidates_meeting_every_link_when_some_do():
    m = _mask([(TAG_OCEAN,), (5,)], [1, 1, 0, 0], [0, 0, 0, 0], {5: [0, 1, 1, 0]})
    np.testing.assert_array_equal(m, [False, True, False, False])
    # Any tag of a link meets it.
    m = _mask([(5, TAG_LAKE)], [0, 0, 0, 0], [0, 0, 0, 1], {5: [0, 1, 1, 0]})
    np.testing.assert_array_equal(m, [False, True, True, True])
    assert _mask([(6,)], [0, 0], [0, 0], {6: [0, 0]}) is None
    assert _mask([], [0, 0], [0, 0], {}) is None


def test_mask_falls_back_confluence_then_mouth_then_coast_then_river():
    # Three links nothing meets together; candidates: coast, mouth of 5, confluence 5+6, river 7.
    links = [(5,), (TAG_OCEAN,), (6,), (7,)]
    ocean = [1, 1, 0, 0]
    rivers = {5: [0, 1, 1, 0], 6: [0, 0, 1, 0], 7: [0, 0, 0, 1]}
    np.testing.assert_array_equal(_mask(links, ocean, [0] * 4, rivers), [False, False, True, False])
    rivers[6] = [0, 0, 0, 0]                           # no confluence: the mouth
    np.testing.assert_array_equal(_mask(links, ocean, [0] * 4, rivers), [False, True, False, False])
    rivers[5] = [0, 0, 1, 0]                           # no mouth: the coast
    np.testing.assert_array_equal(_mask(links, ocean, [0] * 4, rivers), [True, True, False, False])
    # No coast: the biggest river (7, flow 20 > 10).
    np.testing.assert_array_equal(_mask(links, [0] * 4, [0] * 4, rivers), [False, False, False, True])


def test_mask_prefers_the_mouth_of_the_bigger_river():
    links = [(5,), (6,), (TAG_OCEAN,)]
    m = _mask(links, [1, 1, 1], [0, 0, 0], {5: [1, 0, 0], 6: [0, 1, 0]})
    np.testing.assert_array_equal(m, [False, True, False])
