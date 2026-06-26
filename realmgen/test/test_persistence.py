"""Round-trip checks for the realm pack/unpack serialization helpers."""
import numpy as np

from realmgen.persistence import (
    pack_borders,
    pack_polities,
    pack_routes,
    pack_settlements,
    unpack_borders,
    unpack_polities,
    unpack_routes,
    unpack_settlements,
)
from realmgen.placement.types import Settlement
from realmgen.polity.types import Polity


def test_settlements_round_trip():
    settlements = [
        Settlement(
            face_id=3, position=np.array([0.0, 0.0, 1.0]), tier=0, score=0.9,
            population=12000.0, defensibility=1.4, has_walls=True, wall_strength=0.8,
        ),
        Settlement(
            face_id=7, position=np.array([1.0, 0.0, 0.0]), tier=2, score=0.3,
            population=400.0,
        ),
    ]
    out = unpack_settlements(pack_settlements(settlements))
    assert len(out) == 2
    for a, b in zip(settlements, out):
        assert a.face_id == b.face_id
        assert a.tier == b.tier
        assert a.score == b.score
        assert a.population == b.population
        assert a.defensibility == b.defensibility
        assert a.has_walls == b.has_walls
        assert a.wall_strength == b.wall_strength
        assert np.allclose(a.position, b.position)


def test_polities_round_trip_preserves_tree_and_lists():
    polities = [
        Polity(
            polity_id=0, capital_face=1, capital_position=np.array([0.0, 1.0, 0.0]),
            cohesion=0.7, infrastructure=1.2, legitimacy=0.9, maritime_reach=0.3,
            liege=None, autonomy=0.0, depth=0, power=50.0, domain_power=80.0, n_faces=10,
            territory_faces=[1, 2, 3], member_settlement_faces=[1], vassals=[1],
        ),
        Polity(
            polity_id=1, capital_face=4, capital_position=np.array([0.0, 0.0, 1.0]),
            liege=0, autonomy=0.4, depth=1, power=30.0, domain_power=30.0, n_faces=5,
            territory_faces=[4, 5], member_settlement_faces=[4], vassals=[],
        ),
    ]
    out = unpack_polities(pack_polities(polities))
    assert [p.polity_id for p in out] == [0, 1]
    assert out[0].liege is None
    assert out[1].liege == 0
    assert out[0].depth == 0 and out[1].depth == 1
    assert out[0].territory_faces == [1, 2, 3]
    assert out[0].vassals == [1]
    assert out[1].territory_faces == [4, 5]
    assert out[1].vassals == []
    assert np.allclose(out[0].capital_position, [0.0, 1.0, 0.0])
    assert out[0].infrastructure == 1.2


def test_routes_round_trip():
    routes = [
        (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), 5.0),
        (np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0]), 2.5),
    ]
    out = unpack_routes(pack_routes(routes))
    assert len(out) == 2
    for a, b in zip(routes, out):
        assert np.allclose(a[0], b[0])
        assert np.allclose(a[1], b[1])
        assert a[2] == b[2]


def test_borders_round_trip_groups_by_depth():
    borders = {
        0: np.array([[1.0, 0, 0, 0, 1.0, 0], [0, 1.0, 0, 0, 0, 1.0]]),
        1: np.array([[1.0, 0, 0, 0, 0, 1.0]]),
        2: np.zeros((0, 6)),  # empty depth is dropped
    }
    out = unpack_borders(pack_borders(borders))
    assert set(out.keys()) == {0, 1}
    assert out[0].shape == (2, 6)
    assert out[1].shape == (1, 6)
    assert np.allclose(out[0], borders[0])


def test_empty_borders_pack_is_empty():
    assert pack_borders({2: np.zeros((0, 6))}) == {}
