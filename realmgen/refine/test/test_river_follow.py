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
