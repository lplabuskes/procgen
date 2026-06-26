"""Unit tests for realmgen.refine — pure-function / stub-mesh tests.

These avoid a full worldgen build; integration is exercised by the pipeline
smoke test elsewhere.
"""
from __future__ import annotations

import numpy as np

from realmgen.placement.types import Settlement
from realmgen.polity.types import Polity
from realmgen.refine.borders import (
    _chaikin,
    _chain_edges,
    _weld_seam_endpoints,
    smooth_polity_borders,
    trace_label_borders,
)
from realmgen.refine.fill import (
    _despeckle_interior_exclaves,
    build_polity_fill_texture,
    domain_warp,
)
from realmgen.refine.settlements import (
    _neighbor_any,
    position_key,
    refine_settlement_positions,
    remap_routes,
)


# ── Stub mesh ──────────────────────────────────────────────────────────────────


class _StubMesh:
    """One 4×4-texel face whose texel (i, j) sits at [j, i, 0]."""

    def __init__(self, centroid):
        self.face_centroids = np.array([centroid], dtype=np.float64)

    def face_texture_positions(self, h, w):
        pos = np.zeros((1, h, w, 3), dtype=np.float64)
        for i in range(h):
            for j in range(w):
                pos[0, i, j] = [j, i, 0.0]
        return pos


def _settlement(face_id=0):
    return Settlement(face_id=face_id, position=np.array([10.0, 10.0, 10.0]),
                      tier=0, score=1.0, population=1000.0)


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
    mesh = _StubMesh(centroid=[1.0, 1.5, 0.0])
    ocean = np.zeros((1, 4, 4, 1), dtype=np.float64)
    ocean[0, :, 2:, 0] = 1.0
    s = _settlement()
    moved = refine_settlement_positions(
        [s], mesh=mesh, ocean_tex=ocean, lake_tex=None,
        fertility=np.array([1.0]), rivers=None,
    )
    assert s.position[0] == 1.0          # snapped onto the coast column
    assert position_key(np.array([10.0, 10.0, 10.0])) in moved


def test_ocean_preferred_over_lake():
    # Col 0 borders a lake (col... actually make col 3 ocean, col 0 lake-side).
    mesh = _StubMesh(centroid=[1.5, 1.5, 0.0])
    ocean = np.zeros((1, 4, 4, 1), dtype=np.float64)
    ocean[0, :, 3, 0] = 1.0              # ocean on col 3 → coast col 2
    lake = np.zeros((1, 4, 4, 1), dtype=np.float64)
    lake[0, :, 0, 0] = 1.0               # lake on col 0 → coast col 1
    s = _settlement()
    refine_settlement_positions(
        [s], mesh=mesh, ocean_tex=ocean, lake_tex=lake,
        fertility=np.array([1.0]), rivers=None,
    )
    assert s.position[0] == 2.0          # ocean coast wins over lake coast


def test_snap_to_river_when_no_coast():
    mesh = _StubMesh(centroid=[0.0, 0.0, 0.0])
    ocean = np.zeros((1, 4, 4, 1), dtype=np.float64)  # all land
    river_pts = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    rivers = [(river_pts, np.array([1.0]), None)]
    s = _settlement()
    refine_settlement_positions(
        [s], mesh=mesh, ocean_tex=ocean, lake_tex=None,
        fertility=np.array([1.0]), rivers=rivers,
    )
    assert np.allclose(s.position, [0.0, 0.0, 0.0])   # snapped onto the river texel


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


# ── texel border tracing ─────────────────────────────────────────────────────


def _polar_grid(h, w):
    """One face's (1, h, w, 3) texel positions on a small patch near +z."""
    pos = np.zeros((1, h, w, 3), dtype=np.float64)
    for i in range(h):
        for j in range(w):
            v = np.array([(j - (w - 1) / 2) * 0.1, (i - (h - 1) / 2) * 0.1, 1.0])
            pos[0, i, j] = v / np.linalg.norm(v)
    return pos


def test_trace_label_borders_splits_two_polities():
    # Left half polity 0, right half polity 1 → a single depth-0 border down the
    # column seam between j=1 and j=2.
    pos = _polar_grid(4, 4)
    labels = np.array([[[0, 0, 1, 1]] * 4], dtype=np.int32)
    out = trace_label_borders(pos, labels, [_pol(0), _pol(1)], iterations=0)
    assert 0 in out and out[0].shape[1] == 6 and len(out[0]) >= 1
    # The border runs vertically: endpoints span a range of y but a narrow x.
    segs = out[0]
    xs = np.concatenate([segs[:, 0], segs[:, 3]])
    ys = np.concatenate([segs[:, 1], segs[:, 4]])
    assert np.ptp(ys) > np.ptp(xs)


def _overlap_pair():
    """Two faces whose square textures overlap near +z, both carrying the same
    polity-0/polity-1 boundary down the j=1|j=2 seam — the configuration the
    over-hang doubling arises from. The two grids are slightly *offset* (as real
    neighbour faces are), so the two boundary copies have distinct vertex keys."""
    def grid(dx):
        g = np.zeros((4, 4, 3), dtype=np.float64)
        for i in range(4):
            for j in range(4):
                v = np.array([(j - 1.5) * 0.2 + dx, (i - 1.5) * 0.2, 1.0])
                g[i, j] = v / np.linalg.norm(v)
        return g
    pos = np.stack([grid(0.0), grid(0.07)])            # (2, 4, 4, 3) offset grids
    labels = np.array([[[0, 0, 1, 1]] * 4] * 2, dtype=np.int32)
    c = np.array([[-0.1, 0.0, 1.0], [0.1, 0.0, 1.0]])  # centroids straddle x=0
    centroids = c / np.linalg.norm(c, axis=1, keepdims=True)
    return pos, labels, centroids


def test_texel_owner_assigns_by_nearest_centroid():
    from realmgen.refine.borders import _texel_owner
    pos, _, centroids = _overlap_pair()
    owner = _texel_owner(pos, centroids, adj=[[1], [0]])
    assert owner.shape == (2, 4, 4)
    # Midline is x=0: the two left columns go to face 0, the two right to face 1,
    # for *both* faces' (identical) grids.
    assert (owner[:, :, :2] == 0).all()
    assert (owner[:, :, 2:] == 1).all()


def test_texel_owner_polygon_assigns_overhang_to_neighbour():
    from realmgen.refine.borders import _texel_owner
    # With a polygon mask, a texel inside its own face's polygon is owned by that
    # face *definitively* (no centroid-bisector sliver); an over-hang texel goes to
    # the neighbour. Mark face 0's right half as over-hang.
    pos, _, centroids = _overlap_pair()
    in_poly = np.ones((2, 4, 4), dtype=bool)
    in_poly[0, :, 2:] = False
    owner = _texel_owner(pos, centroids, adj=[[1], [0]], in_poly=in_poly)
    assert (owner[0, :, :2] == 0).all()                # in-polygon → own face
    assert (owner[0, :, 2:] == 1).all()                # over-hang → neighbour


def test_trace_label_borders_dedoubles_overlap():
    # Both faces' over-hangs carry the same boundary. Without centroids each face
    # draws it → a doubled line. Voronoi ownership makes only the lower-id face-cell
    # draw each edge, halving the geometry to a single line.
    pos, labels, centroids = _overlap_pair()
    pols = [_pol(0), _pol(1)]
    doubled = trace_label_borders(pos, labels, pols, iterations=0, weld_factor=0.0)
    single = trace_label_borders(pos, labels, pols, iterations=0, weld_factor=0.0,
                                 adj=[[1], [0]], centroids=centroids)
    n_doubled = sum(len(v) for v in doubled.values())
    n_single = sum(len(v) for v in single.values())
    assert n_single >= 1
    assert n_doubled == 2 * n_single                   # exactly the duplicate removed


def test_domain_warp_identity_and_continuity():
    rng = np.random.default_rng(0)
    pos = rng.normal(size=(50, 3))
    pos /= np.linalg.norm(pos, axis=1, keepdims=True)
    # amplitude 0 is a no-op (warp off → bit-identical positions).
    assert np.array_equal(domain_warp(pos, amplitude=0.0, base_frequency=40.0,
                                      octaves=3), pos)
    # Pure function of position: coincident points (the basis for seam continuity)
    # get an identical displacement.
    dup = np.vstack([pos, pos[:5]])
    w = domain_warp(dup, amplitude=0.02, base_frequency=40.0, octaves=3, seed=7)
    assert np.allclose(w[:5], w[50:55])
    # Displacement is bounded by the amplitude (fBm is normalised to ~[-1, 1]).
    assert np.max(np.linalg.norm(w[:50] - pos, axis=1)) < 0.02 * 3


def test_trace_label_borders_resolves_saddle_without_center_x():
    # A 2x2 checkerboard makes a single cell whose four edges are all borders.
    # Routing every crossing to the cell centre would draw a self-crossing X
    # (the visible knot); it must instead emit two non-crossing links that skip
    # the centre point entirely.
    pos = _polar_grid(2, 2)
    labels = np.array([[[0, 1], [1, 0]]], dtype=np.int32)
    out = trace_label_borders(pos, labels, [_pol(0), _pol(1)], iterations=0)
    segs = out[0]
    assert len(segs) == 2                                    # two links, not four
    center = pos[0].reshape(-1, 3).mean(axis=0)
    pts = np.concatenate([segs[:, :3], segs[:, 3:]], axis=0)
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    center = center / np.linalg.norm(center)
    assert np.all(np.linalg.norm(pts - center, axis=1) > 1e-6)  # never the centre


def test_despeckle_dissolves_single_pixel_exclave():
    # A lone polity-7 texel surrounded by polity 0 is reassigned to 0; an island
    # texel surrounded only by water (-2) is left alone (no land neighbour).
    labels = np.zeros((1, 5, 5), dtype=np.int32)
    labels[0, 2, 2] = 7                      # interior exclave
    out = _despeckle_interior_exclaves(labels)
    assert out[0, 2, 2] == 0
    water = np.full((1, 5, 5), -2, dtype=np.int32)
    water[0, 2, 2] = 7                        # claimed island in water
    assert _despeckle_interior_exclaves(water)[0, 2, 2] == 7


def test_despeckle_dissolves_diagonal_only_exclave():
    # A polity-7 texel touching another 7 only diagonally has no orthogonal (D4)
    # same-label neighbour, so it is dissolved into the surrounding 0.
    labels = np.zeros((1, 5, 5), dtype=np.int32)
    labels[0, 2, 2] = 7
    labels[0, 3, 3] = 7                       # diagonal partner only
    out = _despeckle_interior_exclaves(labels)
    assert out[0, 2, 2] == 0


def test_despeckle_leaves_edge_ring_untouched():
    # An exclave on the outer ring is preserved (shared-edge reconciliation owns it).
    labels = np.zeros((1, 5, 5), dtype=np.int32)
    labels[0, 0, 2] = 7
    assert _despeckle_interior_exclaves(labels)[0, 0, 2] == 7


def test_weld_seam_endpoints_joins_cross_face_dangles():
    from realmgen.refine.settlements import position_key
    a = np.array([2.0, 0.0, 0.0])       # face 0 interior endpoint
    b = np.array([1.0, 0.0, 0.0])       # face 0 seam endpoint
    c = np.array([1.0008, 0.0, 0.0])    # face 1 seam endpoint (near b)
    d = np.array([0.0, 0.0, 0.0])       # face 1 interior endpoint
    ka, kb, kc, kd = (position_key(p) for p in (a, b, c, d))
    pt_of = {ka: a, kb: b, kc: c, kd: d}
    edges = {tuple(sorted((ka, kb))), tuple(sorted((kc, kd)))}
    key_face = {kb: 0, kc: 1}                 # b and c are outer-ring crossings

    def deg1(e):
        d_: dict = {}
        for x, y in e:
            d_[x] = d_.get(x, 0) + 1
            d_[y] = d_.get(y, 0) + 1
        return sum(1 for v in d_.values() if v == 1)

    # Adjacent faces within tolerance → b and c weld, leaving only a and d dangling.
    welded, _ = _weld_seam_endpoints(set(edges), dict(pt_of), key_face,
                                     tol=0.01, adj=[[1], [0]])
    assert deg1(welded) == 2
    # Non-adjacent faces → no weld (all four endpoints stay dangling pairs).
    same = _weld_seam_endpoints(set(edges), dict(pt_of), key_face,
                                tol=0.01, adj=[[], []])[0]
    assert same == edges


def test_trace_label_borders_ignores_coastline():
    # Land polity 0 against water (-2): a coastline, not a border.
    pos = _polar_grid(4, 4)
    labels = np.array([[[0, 0, -2, -2]] * 4], dtype=np.int32)
    out = trace_label_borders(pos, labels, [_pol(0), _pol(1)], iterations=0)
    assert all(seg.shape[0] == 0 for seg in out.values()) or out == {}


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


class _GridMesh:
    """One face whose texel grid is a small patch near +z."""

    def face_texture_positions(self, h, w):
        return _polar_grid(h, w)


def test_smooth_polity_borders_emits_depth0_segments():
    mesh = _GridMesh()
    labels = np.array([[[0, 0, 1, 1]] * 4], dtype=np.int32)
    out = smooth_polity_borders(mesh, [_pol(0), _pol(1)], labels)
    assert 0 in out and out[0].shape[1] == 6 and len(out[0]) >= 1


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
