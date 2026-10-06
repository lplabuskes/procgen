"""Forts: candidates, spacing, siting, garrisons, fortified costs, stage accounting."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from core.mesh import Mesh
from core.world import World
from realmgen.channels import FISHING_POPULATION, FORT_POPULATION, RURAL_POPULATION
from realmgen.polity.builder import PolityParams, PolityStage
from realmgen.polity.forts import (
    FORT_BORDER,
    FORT_FRONTIER,
    FORT_STRAIT,
    Forts,
    FortSite,
    border_candidates,
    find_passages,
    fortify_land_costs,
    fortify_landings,
    frontier_candidates,
    garrisons,
    select_sites,
    site_forts,
    strait_candidates,
)

# A path 0—1—2—3—4 (index-aligned adjacency).
LINE = [[1], [0, 2], [1, 3], [2, 4], [3]]


@pytest.fixture(scope="module")
def dual():
    return Mesh.icosphere(2).mesh_dual()


# ── candidates ───────────────────────────────────────────────────────────────


def test_border_candidates_threshold_and_ruggedness():
    src, dst = np.array([1, 2]), np.array([2, 1])
    crossing = np.array([0.5, 0.5])
    rugged = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    sites = border_candidates(src, dst, crossing, rugged, rugged_weight=1.0, min_score=0.3)
    # The rugged side keeps the full score; the flat side is halved below the cut.
    assert [(s.face, s.facing, s.kind) for s in sites] == [(2, 1, FORT_BORDER)]
    assert sites[0].score == pytest.approx(0.5)


def _two_seas(dual, channel_to: float):
    """Polar seas split by an equatorial land band; a channel of sea faces cut through
    it at longitude ~0 from the north down to ``z > channel_to``."""
    c = np.asarray(dual.face_centroids)
    c = c / np.linalg.norm(c, axis=1, keepdims=True)
    lon = np.degrees(np.arctan2(c[:, 1], c[:, 0]))
    band = np.abs(c[:, 2]) < 0.3
    channel = band & (np.abs(lon) < 14.0) & (c[:, 2] > channel_to)
    return ~band | channel, channel


def test_find_passages_marks_a_channel_between_two_seas(dual):
    sea, channel = _two_seas(dual, -1.0)
    passage = find_passages(dual.face_adjacency, sea, max_radius=1)
    hits = np.flatnonzero(passage >= 0)
    assert hits.size > 0 and channel[hits].any()
    near = channel.copy()
    for f in np.flatnonzero(channel):
        near[list(dual.face_adjacency[f])] = True
    assert near[hits].all()   # the narrows, never the open seas


def test_find_passages_ignores_a_dead_end_arm(dual):
    sea, _ = _two_seas(dual, 0.0)   # the channel stops halfway: an inlet, not a strait
    assert (find_passages(dual.face_adjacency, sea, max_radius=1) < 0).all()


# Shores 0 (A) and 1 (B) on strait face 10 (radius 0); shore 2 (A) on strait face 11 (radius 1).
STRAIT_ADJ = [[10], [10], [11]] + [[] for _ in range(7)] + [[0, 1, 11], [10, 2]]
STRAIT_WATER = np.array([False] * 3 + [True] * 9)
STRAIT_PASSAGE = np.array([-1] * 10 + [0, 1])


def test_strait_candidates_hold_the_narrows_on_both_shores():
    sov = np.array([0, 1, 0] + [-1] * 9)
    kw = dict(flow_ref=1.0, artery_weight=0.0, uncontested_weight=0.5, min_score=0.0)
    out = strait_candidates(STRAIT_ADJ, sov, STRAIT_WATER, STRAIT_PASSAGE, np.zeros(12), **kw)
    assert [(s.face, s.facing, s.via, s.kind) for s in out] == [(0, 1, 10, FORT_STRAIT), (1, 0, 10, FORT_STRAIT)]
    assert all(s.score == pytest.approx(1.0) for s in out)
    # One sovereign on every shore: weaker, and it watches the water.
    sov[1] = 0
    out = strait_candidates(STRAIT_ADJ, sov, STRAIT_WATER, STRAIT_PASSAGE, np.zeros(12), **kw)
    assert [(s.face, s.facing) for s in out] == [(0, 10), (1, 10)]
    assert out[0].score == pytest.approx(0.5)
    kw["min_score"] = 0.6
    assert strait_candidates(STRAIT_ADJ, sov, STRAIT_WATER, STRAIT_PASSAGE, np.zeros(12), **kw) == []


def test_frontier_candidates_guard_settled_faces_beside_danger():
    sov = np.array([0, 0, 0, 0, -1])
    water = np.zeros(5, dtype=bool)
    danger = np.array([0.0, 0.0, 0.1, 0.9, 0.0])
    rural = np.array([100.0, 100.0, 100.0, 0.0, 0.0])
    out = frontier_candidates(LINE, sov, water, danger, rural, danger_threshold=0.65, min_score=0.0)
    assert [(s.face, s.facing, s.kind) for s in out] == [(2, 3, FORT_FRONTIER)]
    assert out[0].score == pytest.approx(0.9)
    # Nobody to guard: no outpost.
    rural[2] = 0.0
    assert frontier_candidates(LINE, sov, water, danger, rural, danger_threshold=0.65, min_score=0.01) == []


# ── selection, siting, garrisons ─────────────────────────────────────────────


def _site(face, score, kind=FORT_BORDER, facing=None):
    facing = face + 1 if facing is None else facing
    return FortSite(face=face, facing=facing, via=facing, kind=kind, score=score)


def test_select_sites_spaces_best_first_and_avoids_cities():
    cands = [_site(1, 0.5), _site(2, 0.9), _site(4, 0.4, facing=3), _site(0, 0.8)]
    chosen = select_sites(cands, LINE, spacing_hops=2, excluded={0})
    assert [s.face for s in chosen] == [2, 4]
    # A shared field carries one kind's exclusion zones into the next.
    blocked = np.zeros(5, dtype=bool)
    select_sites([_site(2, 0.9)], LINE, spacing_hops=2, blocked=blocked)
    later = select_sites([_site(3, 0.1, FORT_FRONTIER, 4)], LINE, spacing_hops=2, blocked=blocked)
    assert later == []


def test_garrisons_scale_by_score_and_stop_at_the_farmers():
    sites = [_site(0, 1.0), _site(0, 0.5), _site(2, 0.2, FORT_FRONTIER)]
    g = garrisons(sites, np.array([200.0, 0.0, 1e4]), {FORT_BORDER: 100.0, FORT_FRONTIER: 50.0})
    np.testing.assert_allclose(g, [150.0, 50.0, 75.0])   # second border fort gets what's left


def test_site_forts_on_owned_land_near_the_target(dual):
    face = 5
    via = int(dual.face_adjacency[face][0])
    pos = dual.face_texture_positions(8, 8, [face])
    owned = np.ones((1, 8, 8), dtype=bool)
    land = np.ones((1, 8, 8), dtype=bool)
    land[0, :, :4] = False
    elev = np.zeros((1, 8, 8))
    for kind in (FORT_BORDER, FORT_STRAIT):
        out = site_forts([FortSite(face, via, via, kind, 1.0)], dual.face_centroids, pos, land, owned, elev)
        hit = np.argmin(np.linalg.norm(pos[0].reshape(-1, 3) - out[0], axis=1))
        assert land[0].reshape(-1)[hit]
        assert np.linalg.norm(out[0]) == pytest.approx(1.0)
    # No land anywhere: the lean point itself.
    out = site_forts([FortSite(face, via, via, FORT_BORDER, 1.0)], dual.face_centroids, pos,
                     np.zeros_like(land), owned, elev)
    c = np.asarray(dual.face_centroids)
    assert np.dot(out[0], c[via]) > np.dot(c[face] / np.linalg.norm(c[face]), c[via])


# ── fortified costs ──────────────────────────────────────────────────────────


def test_fortify_land_costs_only_cross_border_edges_both_ways():
    costs = [np.ones(len(a)) for a in LINE]
    sov = np.array([0, 0, 0, 1, 1])
    out = fortify_land_costs(LINE, costs, sov, [2], 5.0)
    assert out[2].tolist() == [1.0, 6.0]   # 2→1 own side; 2→3 across
    assert out[3].tolist() == [6.0, 1.0]   # and back
    assert costs[2].tolist() == [1.0, 1.0]  # input untouched


def test_fortify_landings_adds_to_transitions_at_the_fort():
    n = 2
    # land 0 — land 1; land 1 ↔ sea 3 (face 1's water side); sea 2 — sea 3.
    rows = [0, 1, 1, 3, 2, 3]
    cols = [1, 0, 3, 1, 3, 2]
    g = csr_matrix((np.ones(6), (rows, cols)), shape=(4, 4))
    out = fortify_landings(g, [1], n, 7.0).toarray()
    assert out[1, 3] == 8.0 and out[3, 1] == 8.0
    assert out[0, 1] == 1.0 and out[2, 3] == 1.0


def test_forts_pack_round_trip():
    forts = Forts(
        positions=np.eye(3), face_id=np.array([1, 2, 3], dtype=np.int32),
        facing=np.array([4, 5, 6], dtype=np.int32), kind=np.array([0, 1, 2], dtype=np.int8),
        score=np.array([0.1, 0.2, 0.3]), garrison=np.array([300.0, 250.0, 120.0]),
    )
    back = Forts.unpack(forts.pack())
    np.testing.assert_allclose(back.positions, forts.positions, atol=1e-7)
    for name in ("face_id", "facing", "kind", "score", "garrison"):
        np.testing.assert_array_equal(getattr(back, name), getattr(forts, name))
    assert len(Forts.empty()) == 0


# ── stage accounting ─────────────────────────────────────────────────────────


def test_stage_carves_garrisons_out_of_the_farmers(dual):
    world = World.from_mesh(dual)
    mesh = world.mesh
    n = len(mesh.faces)
    rural0 = np.full(n, 1000.0)
    RURAL_POPULATION.write(mesh, rural0)
    FISHING_POPULATION.write(mesh, np.full(n, 900.0))
    stage = PolityStage(PolityParams())
    sites = [_site(3, 1.0, facing=int(mesh.face_adjacency[3][0])),
             _site(7, 1.0, FORT_FRONTIER, facing=int(mesh.face_adjacency[7][0]))]
    is_water = np.zeros(n, dtype=bool)
    forts = stage._build_forts(mesh, is_water, sites)
    # 100 farmers each: border wants 1.5 × 300, frontier 1.5 × 120.
    np.testing.assert_allclose(forts.garrison, [100.0, 100.0])
    rural, fort = RURAL_POPULATION.read(mesh), FORT_POPULATION.read(mesh)
    np.testing.assert_allclose(rural + fort, rural0)
    assert fort[3] == 100.0 and fort[7] == 100.0

    # Idempotent: a re-run puts the old garrisons back first.
    stage._build_forts(mesh, is_water, sites)
    np.testing.assert_allclose(RURAL_POPULATION.read(mesh) + FORT_POPULATION.read(mesh), rural0)
    np.testing.assert_allclose(FORT_POPULATION.read(mesh)[[3, 7]], [100.0, 100.0])
