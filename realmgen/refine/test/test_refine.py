"""Unit tests for realmgen.refine — pure-function / stub-mesh tests.

These avoid a full worldgen build; integration is exercised by the pipeline
smoke test elsewhere.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from realmgen.polity.types import Polity
from realmgen.test.conftest import _settlement
from core.display import POSITIONS_KEY
from core.mesh import Mesh
from realmgen.refine.borders import (
    _chaikin,
    _chain_edges,
    _icosphere_lattice,
    lattice_depth_for,
    smooth_polity_borders,
    trace_lattice_borders,
)
from realmgen.refine.fill import (
    UNCLAIMED,
    WATER,
    PolityLabelField,
    build_polity_fill_texture,
    domain_warp,
    polity_fill_recipe,
)
from realmgen.refine.settlements import (
    position_key,
    refine_settlement_positions,
    remap_routes,
)
from realmgen.refine.sites import SiteField, _neighbor_any, densify_polylines


# ── Stub mesh ──────────────────────────────────────────────────────────────────


class _StubMesh:
    """4×4-texel faces laid side by side: face f's texel (i, j) sits at [4f + j, i, 0]."""

    def __init__(self, *centroids):
        self.face_centroids = np.array(centroids, dtype=np.float64)

    def face_texture_positions(self, h, w):
        n = len(self.face_centroids)
        pos = np.zeros((n, h, w, 3), dtype=np.float64)
        for f in range(n):
            for i in range(h):
                for j in range(w):
                    pos[f, i, j] = [4 * f + j, i, 0.0]
        return pos


def _refine(settlements, mesh, ocean, lake=None, rivers=None, **kw):
    sites = SiteField(mesh, ocean, lake, None, rivers)
    return refine_settlement_positions(settlements, sites=sites, **kw)


_REFINE_POS = np.array([10.0, 10.0, 10.0])


# ── _neighbor_any ────────────────────────────────────────────────────────────


def test_neighbor_any_orthogonal_only():
    m = np.zeros((3, 3), dtype=bool)
    m[1, 1] = True
    out = _neighbor_any(m)
    assert out[0, 1] and out[2, 1] and out[1, 0] and out[1, 2]
    assert not out[0, 0] and not out[1, 1]  # diagonals / self excluded


# ── settlement snapping ──────────────────────────────────────────────────────


def test_snap_to_ocean_coast():
    # Columns 2-3 are ocean → land cols 0-1; coast = col 1 (adjacent to ocean).
    mesh = _StubMesh([1.0, 1.5, 0.0])
    ocean = np.zeros((1, 4, 4, 1), dtype=np.float64)
    ocean[0, :, 2:, 0] = 1.0
    s = _settlement(pop=1000.0, position=_REFINE_POS)
    moved = _refine([s], mesh, ocean)
    assert s.position[0] == 1.0          # snapped onto the coast column
    assert position_key(np.array([10.0, 10.0, 10.0])) in moved


def test_ocean_preferred_over_lake():
    # Col 0 borders a lake (col... actually make col 3 ocean, col 0 lake-side).
    mesh = _StubMesh([1.5, 1.5, 0.0])
    ocean = np.zeros((1, 4, 4, 1), dtype=np.float64)
    ocean[0, :, 3, 0] = 1.0              # ocean on col 3 → coast col 2
    lake = np.zeros((1, 4, 4, 1), dtype=np.float64)
    lake[0, :, 0, 0] = 1.0               # lake on col 0 → coast col 1
    s = _settlement(pop=1000.0, position=_REFINE_POS)
    _refine([s], mesh, ocean, lake)
    assert s.position[0] == 2.0          # ocean coast wins over lake coast


def test_snap_to_river_when_no_coast():
    mesh = _StubMesh([0.0, 0.0, 0.0])
    ocean = np.zeros((1, 4, 4, 1), dtype=np.float64)  # all land
    river_pts = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    rivers = [(river_pts, np.array([1.0]), None)]
    s = _settlement(pop=1000.0, position=_REFINE_POS)
    _refine([s], mesh, ocean, rivers=rivers)
    assert np.allclose(s.position, [0.0, 0.0, 0.0])   # snapped onto the river texel


def _two_cities_on_one_river(min_spacing):
    # Two all-land faces; a river runs along their shared edge (x = 3.5), so both
    # cities' best site hugs the same stretch of it.
    mesh = _StubMesh([1.5, 1.5, 0.0], [5.5, 1.5, 0.0])
    ocean = np.zeros((2, 4, 4, 1), dtype=np.float64)
    rivers = [(np.array([[3.5, 0.0, 0.0], [3.5, 3.0, 0.0]]), np.array([1.0, 1.0]), None)]
    small = _settlement(face_id=0, pop=1000.0, position=_REFINE_POS.copy())
    big = _settlement(face_id=1, pop=5000.0, position=_REFINE_POS + 1.0)
    _refine([small, big], mesh, ocean, rivers=rivers, min_spacing=min_spacing)
    return small.position, big.position


def test_adjacent_cities_crowd_the_shared_river_without_spacing():
    small, big = _two_cities_on_one_river(0.0)
    assert np.linalg.norm(small - big) < 3.0


def test_min_spacing_keeps_adjacent_cities_apart_bigger_city_first():
    small, big = _two_cities_on_one_river(3.0)
    assert np.linalg.norm(small - big) >= 3.0
    assert big[0] in (4.0, 5.0)           # the bigger city kept its river site
    assert small[0] < 4.0                 # and each stays inside its own face


def test_city_stays_on_owned_texels():
    # Face 0's grid over-hangs face 1: its texels at x = 3 are nearer face 1's
    # centroid, so they are not face 0's to build on even though they are coast.
    mesh = _StubMesh([0.5, 1.5, 0.0], [3.0, 1.5, 0.0])
    ocean = np.zeros((2, 4, 4, 1), dtype=np.float64)
    ocean[1] = 1.0
    ocean[0, :, 3, 0] = 1.0               # coast at face 0's col 2 — owned by face 1
    s = _settlement(face_id=0, pop=1000.0, position=_REFINE_POS.copy())
    _refine([s], mesh, ocean)
    assert s.position[0] < 1.75           # nearer centroid 0 than centroid 1


def test_densify_polylines_bounds_the_gap():
    pts = densify_polylines([np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])], 0.3)
    assert pts is not None
    xs = np.sort(pts[:, 0])
    assert xs[0] == 0.0 and xs[-1] == 1.0
    assert np.diff(xs).max() <= 0.3 + 1e-12


# ── route remapping ──────────────────────────────────────────────────────────


def test_remap_routes_moves_matching_endpoints():
    old = np.array([1.0, 2.0, 3.0])
    new = np.array([1.1, 2.1, 3.1])
    other = np.array([9.0, 9.0, 9.0])
    moved = {position_key(old): new}
    routes = [(old.copy(), other.copy(), 5.0), (other.copy(), old.copy(), 2.0)]
    out = remap_routes(routes, moved)
    assert np.allclose(out[0][0], new) and np.allclose(out[0][1], other)
    assert np.allclose(out[1][0], other) and np.allclose(out[1][1], new)
    assert out[0][2] == 5.0


# ── border chaining + Chaikin smoothing ──────────────────────────────────────


def test_chaikin_cuts_corner_keeps_open_endpoints():
    pts = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([1.0, 1.0, 0.0])]
    out = _chaikin(pts, 1, closed=False)
    assert np.allclose(out[0], pts[0]) and np.allclose(out[-1], pts[-1])  # endpoints fixed
    assert len(out) > len(pts)                                            # corner cut in two
    assert not any(np.allclose(o, pts[1]) for o in out)                   # sharp corner gone


def test_chain_edges_open_and_closed():
    open_chains = _chain_edges({(0, 1), (1, 2), (2, 3)})
    assert len(open_chains) == 1 and len(open_chains[0]) == 4
    loop = _chain_edges({(0, 1), (1, 2), (2, 3), (3, 0)})
    assert len(loop) == 1 and loop[0][0] == loop[0][-1]                   # closed ring


def test_chain_edges_stops_at_given_vertices():
    # 0-1-2 is a degree-2 run, but a stop at 1 splits it (a junction held fixed
    # by another depth's line) and keeps 1 as a shared chain end.
    chains = _chain_edges({(0, 1), (1, 2)}, stops=frozenset({1}))
    assert sorted(len(c) for c in chains) == [2, 2]
    assert all(1 in (c[0], c[-1]) for c in chains)


# ── lattice ──────────────────────────────────────────────────────────────────


def test_icosphere_lattice_is_a_closed_unit_mesh():
    d = 3
    verts, tris, edges, tri_edges = _icosphere_lattice(d)
    assert len(verts) == 10 * 4 ** d + 2
    assert len(tris) == 20 * 4 ** d
    assert len(edges) == 30 * 4 ** d
    assert np.allclose(np.linalg.norm(verts, axis=1), 1.0)
    # Closed 2-manifold: every edge borders exactly two triangles.
    assert np.all(np.bincount(tri_edges.ravel(), minlength=len(edges)) == 2)
    # tri_edges[:, k] really is the edge (corner k, corner k+1).
    a, b = tris[:, 0], tris[:, 1]
    e = edges[tri_edges[:, 0]]
    assert np.all(np.minimum(a, b) == e[:, 0]) and np.all(np.maximum(a, b) == e[:, 1])


def test_lattice_depth_matches_texel_spacing():
    # Default world: 10,242 faces at 16 texels → ~14 km texels → depth 9 (~15 km).
    assert lattice_depth_for(10242, 16) == 9
    assert lattice_depth_for(10242, 32) == 10


# ── lattice border tracing ───────────────────────────────────────────────────


def _lattice(depth=3):
    return _icosphere_lattice(depth)


def _degrees(segs):
    from collections import Counter
    c = Counter()
    for row in np.round(segs, 6):
        c[tuple(row[:3])] += 1
        c[tuple(row[3:])] += 1
    return c


def test_trace_lattice_two_hemispheres_single_closed_loop():
    verts, tris, edges, te = _lattice()
    labels = np.where(verts[:, 0] > 0, 0, 1)
    out = trace_lattice_borders(verts, tris, edges, te, labels, [_pol(0), _pol(1)],
                                iterations=0)
    segs = out[0]
    assert set(_degrees(segs).values()) == {2}                    # closed: no loose ends
    edge_len = np.linalg.norm(verts[edges[0, 0]] - verts[edges[0, 1]])
    assert np.all(np.abs(segs[:, [0, 3]]) < edge_len)             # hugs the x = 0 plane


def test_trace_lattice_bisection_lands_on_the_true_boundary():
    verts, tris, edges, te = _lattice()

    def label_fn(p):
        return np.where(p[:, 0] > 0.1, 0, 1)

    labels = label_fn(verts)
    out = trace_lattice_borders(verts, tris, edges, te, labels, [_pol(0), _pol(1)],
                                iterations=0, label_fn=label_fn, bisect_steps=8)
    x = out[0][:, [0, 3]]
    edge_len = np.linalg.norm(verts[edges[0, 0]] - verts[edges[0, 1]])
    assert np.max(np.abs(x - 0.1)) < edge_len / 50


def test_trace_lattice_ignores_coastline():
    verts, tris, edges, te = _lattice()
    labels = np.where(verts[:, 0] > 0, 0, WATER)
    assert trace_lattice_borders(verts, tris, edges, te, labels, [_pol(0), _pol(1)]) == {}


def test_trace_lattice_border_ends_at_the_coast():
    # Two polities split by x = 0 on land; everything below z = -0.5 is sea. The
    # border is one open line whose two ends both sit on the coast.
    verts, tris, edges, te = _lattice()
    labels = np.where(verts[:, 0] > 0, 0, 1)
    labels[verts[:, 2] < -0.5] = WATER
    segs = trace_lattice_borders(verts, tris, edges, te, labels, [_pol(0), _pol(1)],
                                 iterations=2)[0]
    deg = _degrees(segs)
    ends = np.array([k for k, v in deg.items() if v == 1])
    assert len(ends) == 2 and set(deg.values()) <= {1, 2}
    assert np.all(np.abs(ends[:, 2] + 0.5) < 0.1)


def test_trace_lattice_three_way_junctions_join():
    # Three sectors of longitude meet at both poles: every line runs pole to pole,
    # so the only odd vertices are the two degree-3 junctions.
    verts, tris, edges, te = _lattice()
    lon = np.arctan2(verts[:, 1], verts[:, 0])
    labels = np.digitize(lon, [-np.pi / 3, np.pi / 3])
    segs = trace_lattice_borders(verts, tris, edges, te, labels,
                                 [_pol(0), _pol(1), _pol(2)], iterations=2)[0]
    deg = _degrees(segs)
    assert sorted(v for v in deg.values() if v != 2) == [3, 3]


def test_trace_lattice_vassal_border_meets_sovereign_border():
    # Sovereign 0 holds x > 0; inside it, vassal 1 holds y > 0. Sovereign 2 holds
    # x < 0. The depth-1 line (0 | 1) must end exactly on the depth-0 line.
    verts, tris, edges, te = _lattice()
    labels = np.where(verts[:, 0] > 0, np.where(verts[:, 1] > 0, 1, 0), 2)
    pols = [_pol(0), dataclasses.replace(_pol(1), liege=0, depth=1), _pol(2)]
    out = trace_lattice_borders(verts, tris, edges, te, labels, pols, iterations=2)
    assert set(out) == {0, 1}
    d0 = _degrees(out[0])
    d1 = _degrees(out[1])
    ends1 = [k for k, v in d1.items() if v == 1]
    assert len(ends1) == 2                                         # both ends at junctions
    assert all(k in d0 for k in ends1)                             # ...on the depth-0 line


def test_domain_warp_identity_and_continuity():
    rng = np.random.default_rng(0)
    pos = rng.normal(size=(50, 3))
    pos /= np.linalg.norm(pos, axis=1, keepdims=True)
    # amplitude 0 is a no-op (warp off → bit-identical positions).
    assert np.array_equal(domain_warp(pos, amplitude=0.0, base_frequency=40.0,
                                      octaves=3), pos)
    # Pure function of position: coincident points get an identical displacement.
    dup = np.vstack([pos, pos[:5]])
    w = domain_warp(dup, amplitude=0.02, base_frequency=40.0, octaves=3, seed=7)
    assert np.allclose(w[:5], w[50:55])
    # Displacement is bounded by the amplitude (fBm is normalised to ~[-1, 1]).
    assert np.max(np.linalg.norm(w[:50] - pos, axis=1)) < 0.02 * 3


# ── label field ──────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def dual():
    return Mesh.icosphere(2).mesh_dual()


def _field(dual, pof, water=None, authority=None, **kw):
    n = len(dual.faces)
    return PolityLabelField(
        dual, np.asarray(pof), np.zeros(n, bool) if water is None else water,
        np.ones(n) if authority is None else authority, **kw)


def test_label_field_interior_unclaimed_and_orphan(dual):
    n = len(dual.faces)
    c = np.asarray(dual.face_centroids, dtype=np.float64)
    pof = np.where(c[:, 0] > 0, 0, 1)
    pof[5] = -1                                        # an unclaimed land face
    water = np.zeros(n, bool)
    w = next(f for f in range(n) if f != 5 and 5 not in dual.face_adjacency[f]
             and all(g != 5 for g in dual.face_adjacency[f]))
    water[w] = True                                    # a face-level water face
    field = _field(dual, pof, water=water)
    lab = field.land_labels(c)
    assert lab[5] == UNCLAIMED
    # Orphan land in the water face goes to one of its claimed neighbours.
    assert lab[w] in {int(pof[g]) for g in dual.face_adjacency[w]}
    # Faces whose whole neighbourhood is one polity are that polity.
    for f in range(n):
        nb = [f, *dual.face_adjacency[f]]
        if f != 5 and not water[f] and len({int(pof[g]) for g in nb}) == 1 \
                and not any(water[g] or g == 5 for g in nb):
            assert lab[f] == pof[f]


def test_label_field_stronger_polity_bulges(dual):
    c = np.asarray(dual.face_centroids, dtype=np.float64)
    a = 0
    b = dual.face_adjacency[0][0]
    pof = np.full(len(dual.faces), 0)
    pof[b] = 1
    mid = (c[a] + c[b]) / np.linalg.norm(c[a] + c[b])
    auth = np.ones(len(dual.faces))
    auth[a] = 3.0
    assert _field(dual, pof, authority=auth).land_labels(mid[None])[0] == 0
    auth[a], auth[b] = 1.0, 3.0
    assert _field(dual, pof, authority=auth).land_labels(mid[None])[0] == 1


def test_texel_label_ignores_which_face_holds_the_texel(dual):
    # The label is a function of position alone, so a face's over-hanging texture
    # and its neighbour's agree wherever they sample the same ground — no seam in
    # the fill. Put face a's texel grid in face b's slot: the labels must not move.
    c = np.asarray(dual.face_centroids, dtype=np.float64)
    pof = np.digitize(np.arctan2(c[:, 1], c[:, 0]), [-1.0, 1.0])
    field = _field(dual, pof, warp_amplitude=0.05, warp_frequency=8.0, warp_seed=3)
    pos = np.asarray(dual.face_texture_positions(8, 8), dtype=np.float64)
    ocean = np.zeros((len(dual.faces), 8, 8, 1))
    labels = field.texel_labels(pos, ocean, None)
    a = int(np.argmax([len(set(labels[f].ravel())) for f in range(len(dual.faces))]))
    b = dual.face_adjacency[a][0]
    assert len(set(labels[a].ravel())) > 1                         # a genuine border face
    swapped = pos.copy()
    swapped[b] = pos[a]
    assert np.array_equal(field.texel_labels(swapped, ocean, None)[b], labels[a])


def test_smooth_polity_borders_on_a_real_mesh(dual):
    c = np.asarray(dual.face_centroids, dtype=np.float64)
    pof = np.where(c[:, 0] > 0, 0, 1)
    field = _field(dual, pof)
    ocean = np.zeros((len(dual.faces), 8, 8, 1))
    out = smooth_polity_borders(dual, [_pol(0), _pol(1)], field, ocean, None)
    assert set(out) == {0}
    assert set(_degrees(out[0]).values()) == {2}                 # one closed loop


def test_polity_fill_recipe_colours_any_texel_grid(dual):
    from worldgen.geology.display import OCEAN_COLOR

    c = np.asarray(dual.face_centroids, dtype=np.float64)
    field = _field(dual, np.where(c[:, 0] > 0, 0, 1))
    recipe = polity_fill_recipe(field, [_pol(0), _pol(1)])
    pts = np.array([[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])   # (1, 3, 3)
    ocean = np.zeros((1, 3, 1))
    ocean[0, 2, 0] = 1.0
    textures = {POSITIONS_KEY: pts, "ocean_mask": ocean}
    assert recipe.available(textures)
    rgba = recipe(textures)
    assert rgba.shape == (1, 3, 4)
    assert not np.allclose(rgba[0, 0], rgba[0, 1])              # two polities, two colours
    assert np.allclose(rgba[0, 2, :3], OCEAN_COLOR[:3])


# ── border / fill stub mesh ──────────────────────────────────────────────────


class _BorderMesh:
    """Two faces sharing edge (1, 2); texels collapse to their face centroid."""

    def __init__(self):
        v = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
        self.vertices = v / np.linalg.norm(v, axis=1, keepdims=True)
        self.faces = [(0, 1, 2), (1, 2, 3)]
        self.face_adjacency = [[1], [0]]
        c = np.array([self.vertices[list(f)].mean(0) for f in self.faces])
        self.face_centroids = c / np.linalg.norm(c, axis=1, keepdims=True)

    def face_texture_positions(self, h, w):
        pos = np.zeros((2, h, w, 3))
        for f in range(2):
            pos[f, :, :] = self.face_centroids[f]
        return pos


def _pol(pid):
    return Polity(polity_id=pid, capital_face=pid, capital_position=np.array([1.0, 0.0, 0.0]),
                  power=10.0, n_faces=1, member_settlement_faces=[pid])


def test_fill_texture_shape_and_coastline():
    from worldgen.geology.display import OCEAN_COLOR

    mesh = _BorderMesh()
    ocean = np.zeros((2, 2, 2, 1))
    ocean[0, 0, 0, 0] = 1.0                       # one water texel on face 0
    rgba = build_polity_fill_texture(
        mesh, [_pol(0), _pol(1)], np.array([0, 1]), np.array([False, False]),
        authority=np.array([1.0, 1.0]), ocean_tex=ocean, lake_tex=None,
    )
    assert rgba.shape == (2, 2, 2, 4) and rgba.dtype == np.float16
    assert np.allclose(rgba[..., 3], 1.0)
    assert np.allclose(rgba[0, 0, 0, :3].astype(float), OCEAN_COLOR[:3], atol=1e-2)  # coast
    assert not np.allclose(rgba[1, 0, 0, :3].astype(float), OCEAN_COLOR[:3], atol=1e-2)


def test_fill_covers_orphan_land_in_water_face():
    """A face that is water at *face* level (>50% texels water) but holds genuine
    land texels: those land texels belong to the claimed-land neighbour, not the
    ocean — the polity colour must cover them at texel level."""
    from realmgen.refine.fill import _polity_color_map

    mesh = _BorderMesh()
    ocean = np.zeros((2, 2, 2, 1))               # all texels land per the mask
    rgba = build_polity_fill_texture(
        mesh, [_pol(0), _pol(1)],
        polity_of_face=np.array([-1, 1]),        # face 0 unclaimed + water, face 1 polity 1
        is_water=np.array([True, False]),
        authority=np.array([1.0, 1.0]), ocean_tex=ocean, lake_tex=None,
    )
    pid_rgb = _polity_color_map([_pol(0), _pol(1)])
    # Face 0's land texels take the neighbouring polity colour, not ocean/grey.
    assert np.allclose(rgba[0, 0, 0, :3].astype(float), pid_rgb[1], atol=1e-2)


def test_fill_texture_lake_distinct_from_ocean():
    from worldgen.geology.display import OCEAN_COLOR
    from worldgen.hydrology.builder import LAKE_COLOR

    mesh = _BorderMesh()
    ocean = np.zeros((2, 2, 2, 1))
    lake = np.zeros((2, 2, 2, 1))
    lake[0, 0, 0, 0] = 1.0                        # one lake texel on face 0
    rgba = build_polity_fill_texture(
        mesh, [_pol(0), _pol(1)], np.array([0, 1]), np.array([False, False]),
        authority=np.array([1.0, 1.0]), ocean_tex=ocean, lake_tex=lake,
    )
    got = rgba[0, 0, 0, :3].astype(float)
    assert np.allclose(got, LAKE_COLOR[:3], atol=1e-2)            # lake colour, not ocean
    assert not np.allclose(got, OCEAN_COLOR[:3], atol=1e-2)
