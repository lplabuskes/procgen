"""Unit tests for `worldgen.resources.terrain`."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from worldgen.resources.terrain import (
    face_elevation_range,
    face_hops_from_seeds,
    lake_containing_faces,
    river_traversed_faces,
    volcanic_faces,
)


@dataclass
class _StubMesh:
    """Minimal stand-in for PolygonMesh — only attributes used by tested functions."""
    faces: List[tuple]
    face_adjacency: List[List[int]]
    face_metadata: List[dict] = field(default_factory=list)


class TestFaceElevationRange:
    def test_constant_texture_returns_zero(self):
        tex = np.full((3, 4, 4, 1), 1.5, dtype=np.float32)
        assert np.allclose(face_elevation_range(tex), 0.0)

    def test_varying_texture_returns_max_minus_min(self):
        tex = np.zeros((2, 2, 2, 1), dtype=np.float32)
        tex[0, 0, 0, 0] = 2.0
        tex[0, 1, 1, 0] = -1.0
        tex[1, :, :, 0] = 0.5
        rng = face_elevation_range(tex)
        assert rng[0] == np.float32(3.0)
        assert rng[1] == np.float32(0.0)

    def test_dtype_float32(self):
        tex = np.zeros((1, 2, 2, 1), dtype=np.float32)
        assert face_elevation_range(tex).dtype == np.float32


class TestFaceHopsFromSeeds:
    def _line_mesh(self, n: int) -> _StubMesh:
        adj = [[] for _ in range(n)]
        for i in range(n - 1):
            adj[i].append(i + 1)
            adj[i + 1].append(i)
        return _StubMesh(faces=[(0,)] * n, face_adjacency=adj)

    def test_seed_face_has_zero_hops(self):
        mesh = self._line_mesh(5)
        mask = np.array([True, False, False, False, False])
        h = face_hops_from_seeds(mesh, mask, max_hops=8)
        assert h[0] == 0

    def test_distance_grows_with_chain(self):
        mesh = self._line_mesh(5)
        mask = np.array([True, False, False, False, False])
        h = face_hops_from_seeds(mesh, mask, max_hops=8)
        assert list(h) == [0, 1, 2, 3, 4]

    def test_multiple_seeds_take_min(self):
        mesh = self._line_mesh(5)
        mask = np.array([True, False, False, False, True])
        h = face_hops_from_seeds(mesh, mask, max_hops=8)
        assert list(h) == [0, 1, 2, 1, 0]

    def test_unreachable_returns_sentinel(self):
        # Two disconnected components.
        adj = [[1], [0], [3], [2]]
        mesh = _StubMesh(faces=[(0,)] * 4, face_adjacency=adj)
        mask = np.array([True, False, False, False])
        h = face_hops_from_seeds(mesh, mask, max_hops=3)
        assert h[0] == 0
        assert h[1] == 1
        assert h[2] == 4  # max_hops + 1 sentinel
        assert h[3] == 4

    def test_cap_truncates(self):
        mesh = self._line_mesh(10)
        mask = np.zeros(10, dtype=bool); mask[0] = True
        h = face_hops_from_seeds(mesh, mask, max_hops=3)
        assert list(h[:4]) == [0, 1, 2, 3]
        assert all(h[4:] == 4)

    def test_no_seeds_returns_sentinel(self):
        mesh = self._line_mesh(3)
        mask = np.zeros(3, dtype=bool)
        h = face_hops_from_seeds(mesh, mask, max_hops=2)
        assert all(h == 3)

    def test_shape_check_raises(self):
        mesh = self._line_mesh(3)
        try:
            face_hops_from_seeds(mesh, np.zeros(4, dtype=bool), max_hops=2)
        except ValueError:
            return
        assert False, "expected ValueError on mismatched seed_mask"


class TestRiverTraversedFaces:
    def test_none_returns_empty(self):
        m = river_traversed_faces(None, n_faces=5)
        assert m.shape == (5,)
        assert not m.any()

    def test_empty_list(self):
        m = river_traversed_faces([], n_faces=3)
        assert not m.any()

    def test_marks_segment_endpoints(self):
        class _Node:
            def __init__(self, fid): self.face_id = fid
        class _Seg:
            def __init__(self, a, b, in_lake=False):
                self.start = _Node(a)
                self.end = _Node(b)
                self.in_lake = in_lake
        class _River:
            def __init__(self, segs): self.segments = segs
        rivers = [_River([_Seg(1, 2), _Seg(2, 4)])]
        m = river_traversed_faces(rivers, n_faces=6)
        assert list(m) == [False, True, True, False, True, False]

    def test_skips_in_lake_segments(self):
        class _Node:
            def __init__(self, fid): self.face_id = fid
        class _Seg:
            def __init__(self, a, b, in_lake=False):
                self.start = _Node(a)
                self.end = _Node(b)
                self.in_lake = in_lake
        class _River:
            def __init__(self, segs): self.segments = segs
        rivers = [_River([_Seg(1, 2, in_lake=True), _Seg(2, 3)])]
        m = river_traversed_faces(rivers, n_faces=5)
        # 1 should not be marked (only appeared on in_lake segment); 2 and 3 marked.
        assert not m[1]
        assert m[2]
        assert m[3]


class TestLakeContainingFaces:
    def test_threshold(self):
        tex = np.zeros((3, 4, 4, 1), dtype=np.float32)
        tex[1, :2, :2, 0] = 1.0  # 4/16 = 0.25 fraction
        m = lake_containing_faces(tex, threshold=0.05)
        assert not m[0]
        assert m[1]
        assert not m[2]

    def test_threshold_excludes_tiny(self):
        tex = np.zeros((2, 8, 8, 1), dtype=np.float32)
        tex[0, 0, 0, 0] = 1.0  # 1/64 ≈ 0.015 fraction
        m = lake_containing_faces(tex, threshold=0.05)
        assert not m[0]


class TestVolcanicFaces:
    def test_threshold(self):
        mesh = _StubMesh(
            faces=[(0,)] * 3,
            face_adjacency=[[], [], []],
            face_metadata=[
                {"volcanic_activity": 0.1},
                {"volcanic_activity": 0.5},
                {},
            ],
        )
        m = volcanic_faces(mesh, threshold=0.2)
        assert list(m) == [False, True, False]
