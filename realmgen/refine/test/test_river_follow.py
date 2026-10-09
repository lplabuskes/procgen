"""Unit tests for realmgen.refine.river_follow — borders moved onto rivers run by run."""
from __future__ import annotations

import numpy as np
import pytest

from core.mesh import Mesh
from realmgen.polity.types import Polity
from realmgen.refine.barriers import pack_river_segments
from realmgen.refine.borders import _icosphere_lattice, trace_lattice_chains
from realmgen.refine.fill import PolityLabelField
from realmgen.refine.river_follow import RiverFollowParams, _even_odd, build_river_swaps
from worldgen.cost.surfaces import EARTH_RADIUS_KM


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _pol(pid):
    return Polity(polity_id=pid, capital_face=pid, capital_position=np.array([1.0, 0.0, 0.0]),
                  power=10.0, n_faces=1, member_settlement_faces=[pid])


def _river_at(x, phi0=0.1, phi1=1.4, flow=10000.0, n=200):
    """A river on the small circle ``x = const`` (the polities split at ``x = 0``)."""
    phi = np.linspace(phi0, phi1, n)
    r = np.sqrt(1.0 - x ** 2)
    return (np.column_stack([np.full(n, x), r * np.cos(phi), r * np.sin(phi)]), np.full(n - 1, flow))


# Coarse world: faces ~2000 km apart, so reach and sampling are scaled up to match.
_PARAMS = RiverFollowParams(reach_km=0.3 * EARTH_RADIUS_KM, min_flow=1000.0, full_flow=8000.0,
                            min_run_km=200.0, sample_km=40.0, border_step_km=20.0, settlement_clear_km=300.0,
                            view_reach_km=100.0)


@pytest.fixture(scope="module")
def split():
    dual = Mesh.icosphere(2).mesh_dual()
    n = len(dual.faces)
    c = np.asarray(dual.face_centroids, dtype=np.float64)
    pof = np.where(c[:, 0] > 0, 0, 1)
    field = PolityLabelField(dual, pof, np.zeros(n, bool), np.ones(n))
    verts, tris, edges, tri_edges = _icosphere_lattice(5)
    chains = trace_lattice_chains(verts, tris, edges, tri_edges, field.plain_land_labels(verts),
                                  [_pol(0), _pol(1)], label_fn=field.plain_land_labels)
    return field, chains


def _swapped(field, chains, rivers, settlements=np.zeros((0, 3)), stats=None):
    swaps = build_river_swaps(chains, rivers, field.plain_land_labels, field.is_water_at, settlements,
                              field.centroids, _PARAMS, stats=stats)
    return field.with_swaps(swaps), swaps


def _flips_along_x(field, y=0.6, z=0.5):
    xs = np.linspace(-0.3, 0.3, 1201)
    path = _unit(np.column_stack([xs, np.full(xs.size, y), np.full(xs.size, z)]))
    lab = field.land_labels(path)
    return path[np.flatnonzero(np.diff(lab) != 0), 0], lab


def test_even_odd_counts_crossings():
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    pts = np.array([[0.5, 0.5], [1.5, 0.5], [-0.1, 0.2]])
    assert _even_odd(pts, square).tolist() == [True, False, False]


def test_chains_carry_their_label_pair(split):
    _, chains = split
    assert chains and all(pair == (0, 1) for _, pair, _ in chains)


def test_border_moves_onto_a_nearby_river(split):
    field, chains = split
    moved, swaps = _swapped(field, chains, [_river_at(0.05)])
    assert len(swaps) >= 1
    flips, lab = _flips_along_x(moved)
    assert len(flips) == 1
    assert lab[0] == 1 and lab[-1] == 0                       # sides kept
    assert flips[0] == pytest.approx(0.05 / np.sqrt(1 + 0.05 ** 2), abs=4e-3)
    plain_flips, _ = _flips_along_x(field)
    assert abs(plain_flips[0] - flips[0]) > 0.01


def test_parallel_rivers_one_is_picked_for_the_whole_run(split):
    """Two rivers either side of the border: it follows one along its length, never both."""
    field, chains = split
    moved, _ = _swapped(field, chains, [_river_at(0.04, flow=9000.0), _river_at(-0.04, flow=8500.0)])
    at = []
    for z in (0.3, 0.5, 0.7):
        flips, _ = _flips_along_x(moved, y=np.sqrt(1 - z ** 2), z=z)
        assert len(flips) == 1
        at.append(np.sign(flips[0]))
    assert len(set(at)) == 1


def test_short_contact_is_left_alone(split):
    field, chains = split
    stats: dict = {}
    _, swaps = _swapped(field, chains, [_river_at(0.05, phi0=0.6, phi1=0.62, n=5)], stats=stats)
    assert len(swaps) == 0


def test_settlement_is_never_handed_over(split):
    field, chains = split
    # Polity 0's settlement between the plain border and a river west of it.
    town = _unit(np.array([0.02, 0.6, 0.5]))
    assert field.plain_land_labels(town[None])[0] == 0
    moved, swaps = _swapped(field, chains, [_river_at(-0.05)], settlements=town[None])
    assert moved.land_labels(town[None])[0] == 0
    # Elsewhere along the river the border still moves.
    assert len(swaps) >= 1


def test_view_rivers_move_the_swap_edge(split):
    """A tier's drawing of the same world river (by id) stands in for it, however far it wandered."""
    field, chains = split
    moved, _ = _swapped(field, chains, [_river_at(0.05)])
    drawn = _river_at(0.06, n=800)                             # world river 0, re-routed ~64 km off
    ids = [np.zeros(800, dtype=np.int64)]
    view = moved.with_rivers(pack_river_segments([drawn], river_ids=ids))
    flips, _ = _flips_along_x(view)
    assert len(flips) == 1
    assert flips[0] == pytest.approx(0.06 / np.sqrt(1 + 0.06 ** 2), abs=4e-3)
    world_x = 0.05 / np.sqrt(1 + 0.05 ** 2)
    # Another river (id 1) drawn there, or rivers without ids: the world line stays.
    other = moved.with_rivers(pack_river_segments([drawn], river_ids=[np.ones(800, dtype=np.int64)]))
    assert _flips_along_x(other)[0][0] == pytest.approx(world_x, abs=4e-3)
    bare = moved.with_rivers(pack_river_segments([drawn]))
    assert _flips_along_x(bare)[0][0] == pytest.approx(world_x, abs=4e-3)


def test_partly_drawn_river_keeps_the_world_line_beyond_the_view(split):
    field, chains = split
    moved, _ = _swapped(field, chains, [_river_at(0.05)])
    drawn = _river_at(0.06, phi0=0.6, phi1=1.4, n=400)          # only the downstream half is on this map
    view = moved.with_rivers(pack_river_segments([drawn], river_ids=[np.zeros(400, dtype=np.int64)]))
    (on_drawn,), _ = _flips_along_x(view, y=np.cos(0.9), z=np.sin(0.9))
    (on_world,), _ = _flips_along_x(view, y=np.cos(0.3), z=np.sin(0.3))
    assert on_drawn == pytest.approx(0.06 / np.sqrt(1 + 0.06 ** 2), abs=4e-3)
    assert on_world == pytest.approx(0.05 / np.sqrt(1 + 0.05 ** 2), abs=4e-3)


# ── junctions ───────────────────────────────────────────────────────────────


def test_junction_swap_moves_the_junction_onto_the_river():
    """Polity 0 takes its neighbours' land on its bank and gives its own across the river away."""
    from realmgen.refine.river_follow import OTHER, JunctionSwap
    # River along y = 0 (x from -0.2 to 0.2), polity 0's old boundary bulging north to y = 0.1.
    xs = np.linspace(-0.2, 0.2, 81)
    river = _unit(np.column_stack([np.ones(xs.size), xs, np.zeros(xs.size)]))
    bulge = 0.1 * np.cos(np.pi * xs / 0.4) ** 2
    border = _unit(np.column_stack([np.ones(xs.size), xs, bulge]))
    sw = JunctionSwap(OTHER, 0, 0.01, 0.005, 0, border, np.ones(xs.size), border.copy(), 0, xs.size - 1,
                      river, 0, polity=0, others=(1, 2),
                      without=lambda pts, pol: np.where(pts[:, 1] < 0.0, 1, 2))
    pts = _unit(np.array([[1.0, -0.05, 0.05],     # polity 0 north of the river, west half → 1
                          [1.0, 0.05, 0.05],      # … east half → 2
                          [1.0, 0.0, -0.005],     # polity 1 just south of the river (in the band) → 0
                          [1.0, 0.0, 0.05],       # polity 3, a stranger: untouched
                          [1.0, 0.0, -0.3]]))     # polity 0 far south, outside: untouched
    labels = np.array([0, 0, 1, 3, 0])
    new, claimed = sw.relabel(pts, labels)
    assert new.tolist() == [1, 2, 0, 3, 0]
    assert claimed.tolist() == [True, True, True, False, False]


def test_walk_follows_a_polity_boundary_through_a_junction():
    from realmgen.refine.river_follow import _Run, _end_key, _walk
    a = np.array([[0.0, 0.0, 1.0], [0.0, 0.1, 1.0], [0.0, 0.2, 1.0]])          # 0 | 1, ending at the junction
    b = np.array([[0.0, 0.2, 1.0], [0.1, 0.3, 1.0], [0.2, 0.4, 1.0]])          # 0 | 2, from the junction
    c = np.array([[0.0, 0.2, 1.0], [-0.1, 0.3, 1.0]])                          # 1 | 2, from the junction
    chains = [(a, (0, 1), 0), (b, (0, 2), 0), (c, (1, 2), 0)]
    ends: dict = {}
    for k, (pts, _, _) in enumerate(chains):
        ends.setdefault(_end_key(pts[0]), []).append((k, 0))
        ends.setdefault(_end_key(pts[-1]), []).append((k, 1))
    run_b = _Run(0, 1, 2, 1, 2, 1.0)
    walk = _walk(chains, ends, 0, 0, True, 1, 1, run_b, 1)
    assert walk is not None
    path, met, spans = walk
    assert np.allclose(path, _unit(np.vstack([a[1:], b[1:2]])))
    assert met == {0, 1, 2}
    assert spans == [(0, 1, 2), (1, 0, 1)]
    assert _walk(chains, ends, 2, 0, True, 1, 1, run_b, 1) is None     # polity 2 isn't on chain 0's end


def test_labels_without_hands_a_polity_to_its_neighbours(split):
    field, _ = split
    pts = _unit(np.array([[0.3, 0.6, 0.5], [-0.3, 0.6, 0.5]]))
    assert field.plain_land_labels(pts).tolist() == [0, 1]
    assert field.labels_without(pts, 0).tolist() == [1, 1]
    assert field.labels_without(pts, 1).tolist() == [0, 0]


def test_junction_swap_keeps_a_bridgehead_round_a_settlement():
    from realmgen.refine.river_follow import OTHER, JunctionSwap
    xs = np.linspace(-0.2, 0.2, 81)
    river = _unit(np.column_stack([np.ones(xs.size), xs, np.zeros(xs.size)]))
    border = _unit(np.column_stack([np.ones(xs.size), xs, 0.1 * np.cos(np.pi * xs / 0.4) ** 2]))
    city = _unit(np.array([[1.0, 0.0, 0.002]]))                # polity 0's city on the north bank
    sw = JunctionSwap(OTHER, 0, 0.01, 0.005, 0, border, np.ones(xs.size), border.copy(), 0, xs.size - 1,
                      river, 0, polity=0, others=(1,), without=lambda pts, pol: np.ones(len(pts), dtype=np.int64),
                      keep_pos=city, keep_lab=np.array([0]), keep_radius=0.02)
    pts = _unit(np.array([[1.0, 0.0, 0.002],     # the city
                          [1.0, 0.01, 0.015],    # beside it, within its pocket
                          [1.0, 0.1, 0.03]]))    # further along the river: handed over
    new, _ = sw.relabel(pts, np.array([0, 0, 0]))
    assert new.tolist() == [0, 0, 1]


def test_walk_to_river_finds_where_the_boundary_returns_to_the_river():
    from realmgen.refine.river_follow import _Rivers, _end_key, _walk_to_river
    # River along y = 0 (x from -0.3 to 0.3); polity 0's border leaves it at a junction
    # (chain a ends at x = 0, y = 0.05) and its border with polity 2 comes back down to it.
    xs = np.linspace(-0.3, 0.3, 121)
    river = (_unit(np.column_stack([np.ones(xs.size), xs, np.zeros(xs.size)])), np.full(xs.size - 1, 10000.0))
    riv = _Rivers([river], RiverFollowParams(reach_km=0.1 * EARTH_RADIUS_KM, min_flow=1000.0, full_flow=8000.0))
    a = _unit(np.array([[1.0, -0.2, 0.01], [1.0, -0.1, 0.02], [1.0, 0.0, 0.05]]))          # 0 | 1
    b = _unit(np.array([[1.0, 0.0, 0.05], [1.0, 0.05, 0.08], [1.0, 0.1, 0.03], [1.0, 0.15, 0.002]]))  # 0 | 2
    c = _unit(np.array([[1.0, 0.0, 0.05], [1.0, 0.0, 0.2]]))                               # 1 | 2
    chains = [(a, (0, 1), 0), (b, (0, 2), 0), (c, (1, 2), 0)]
    ends: dict = {}
    for k, (pts, _, _) in enumerate(chains):
        ends.setdefault(_end_key(pts[0]), []).append((k, 0))
        ends.setdefault(_end_key(pts[-1]), []).append((k, 1))
    s_from = 0.2                                     # arc of the river point beside a's core end
    walk = _walk_to_river(chains, ends, 0, 0, True, 1, riv, 0, s_from, 1.0, 0.01)
    assert walk is not None
    path, met, spans, s_end = walk
    assert np.allclose(path[-1], b[-1]) and met == {0, 1, 2}
    assert spans[-1] == (1, 0, 3)
    assert s_end == pytest.approx(0.45, abs=0.01)    # x = 0.15 along a river starting at x = -0.3
    assert _walk_to_river(chains, ends, 1, 0, True, 1, riv, 0, s_from, 1.0, 0.01) is None   # 1's border leaves


def test_border_moves_onto_a_nearby_ridge(split):
    """A ridge is a line like a river (relief for flow); its swap id comes after the rivers'."""
    field, chains = split
    pos, _ = _river_at(0.05)
    ridge = (pos, np.full(len(pos) - 1, 2.0))                    # 2 km relief: full reach
    params = RiverFollowParams(**{**_PARAMS.__dict__, "ridge_reach_km": _PARAMS.reach_km})
    swaps = build_river_swaps(chains, [], field.plain_land_labels, field.is_water_at, np.zeros((0, 3)),
                              field.centroids, params, ridges=[ridge])
    assert len(swaps) >= 1 and all(sw.river_id == 0 for sw in swaps.swaps)
    flips, _ = _flips_along_x(field.with_swaps(swaps))
    assert len(flips) == 1
    assert flips[0] == pytest.approx(0.05 / np.sqrt(1 + 0.05 ** 2), abs=4e-3)
    river = _river_at(-0.04, flow=8500.0)
    both = build_river_swaps(chains, [river], field.plain_land_labels, field.is_water_at, np.zeros((0, 3)),
                             field.centroids, params, ridges=[ridge])
    assert {sw.river_id for sw in both.swaps} <= {0, 1}         # river 0, the ridge 1


def test_low_ridge_holds_no_border(split):
    field, chains = split
    pos, _ = _river_at(0.05)
    params = RiverFollowParams(**{**_PARAMS.__dict__, "ridge_reach_km": _PARAMS.reach_km})
    swaps = build_river_swaps(chains, [], field.plain_land_labels, field.is_water_at, np.zeros((0, 3)),
                              field.centroids, params, ridges=[(pos, np.full(len(pos) - 1, 0.1))])
    assert len(swaps) == 0


def test_connector_wavers_between_its_ends():
    from realmgen.refine.river_follow import _connector
    p, q = _unit(np.array([1.0, 0.0, 0.0])), _unit(np.array([1.0, 0.01, 0.0]))     # ~64 km apart
    c = _connector(p, q)
    gap = np.linalg.norm(q - p)
    assert np.allclose(c[0], p) and len(c) >= 30
    straight = _unit(np.outer(1.0 - np.arange(len(c)) / len(c), p) + np.outer(np.arange(len(c)) / len(c), q))
    off = np.linalg.norm(c - straight, axis=1)
    assert off.max() <= 0.25 * gap * 1.01 and off.max() > 0.01 * gap      # sways, within bounds
    assert np.array_equal(c, _connector(p, q))                             # deterministic


def test_run_starting_deep_in_reach_joins_the_old_border_end(split):
    """A swap whose blend starts at full weight runs its line from the old border's end."""
    field, chains = split
    _, swaps = _swapped(field, chains, [_river_at(0.05)])
    for sw in swaps.swaps:
        for end, joined in ((0, sw.join_start), (-1, sw.join_end)):
            if joined and not (sw.open_start if end == 0 else sw.open_end):
                assert np.allclose(sw.line[end], sw.border[end])
        assert all(j >= 0.0 for j in sw.jumps)


def test_points_past_a_rivers_end_are_not_beside_it():
    """A border point beyond a river's end (its foot the end point) is out of its reach."""
    from realmgen.refine.river_follow import _Rivers
    river = _river_at(0.0, phi0=0.1, phi1=0.5)
    riv = _Rivers([river], _PARAMS)
    phi = np.array([0.3, 0.6])
    beside = _unit(np.column_stack([np.full(2, 0.05), np.cos(phi), np.sin(phi)]))
    nr = riv.near(beside)[0]
    assert nr.mask.tolist() == [True, False]


def test_a_far_closer_line_shadows_the_core():
    from realmgen.refine.river_follow import _Near, _shadow
    k = 3
    far = _Near(np.ones(k, bool), np.zeros(k), np.array([0.010, 0.010, 0.010]), np.ones(k), np.zeros((k, 3)))
    near = _Near(np.ones(k, bool), np.zeros(k), np.array([0.009, 0.004, 0.001]), np.ones(k), np.zeros((k, 3)))
    _shadow({0: far, 1: near})
    assert far.shadowed is not None and near.shadowed is not None
    assert far.shadowed.tolist() == [False, True, True]       # similar distance shadows nothing
    assert near.shadowed.tolist() == [False, False, False]
