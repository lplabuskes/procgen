"""Tests for realmgen.situational_value and the situational-value portion of RealmPipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pytest

from realmgen.scoring.situational_value import (
    combine_situational_value,
    compute_delta_apexes,
    compute_harbor_quality,
    compute_mountain_passes,
    compute_navigable_heads,
    compute_river_confluences,
    compute_strait_adjacency,
)
from realmgen.scoring.builder import (
    MountainPassParams,
    SituationalValueWeightParams,
)


def _norm(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


@dataclass
class _StubMesh:
    faces: List[tuple]
    face_adjacency: List[List[int]]
    face_centroids: np.ndarray


def _star_mesh() -> _StubMesh:
    """5-face star: 0=MID adjacent to all arms 1–4; arms touch only MID."""
    centroids = np.array([
        _norm((0.0, 0.0, 1.0)),
        _norm((0.25, 0.0, 1.0)),
        _norm((0.0, 0.25, 1.0)),
        _norm((-0.25, 0.0, 1.0)),
        _norm((0.0, -0.25, 1.0)),
    ])
    adj = [[1, 2, 3, 4], [0], [0], [0], [0]]
    return _StubMesh(faces=[(i,) for i in range(5)], face_adjacency=adj, face_centroids=centroids)


def _make_river_tuple(mesh: _StubMesh, *face_ids: int, in_lake: bool = False):
    """Build a ``(positions, flows, in_lake)`` tuple using face centroids as positions."""
    positions = mesh.face_centroids[np.array(face_ids, dtype=np.intp)].astype(np.float32)
    n_segs = len(face_ids) - 1
    flows = [float(i + 1) for i in range(n_segs)]
    lake_arr = np.full(n_segs, in_lake, dtype=bool)
    return (positions, flows, lake_arr)


# ── Harbor quality ────────────────────────────────────────────────────────────


class TestHarborQuality:
    def test_no_ocean_neighbors_scores_zero(self):
        mesh = _star_mesh()
        is_ocean = np.zeros(5, dtype=bool)  # no ocean faces
        elev = np.zeros(5)
        score = compute_harbor_quality(mesh, is_ocean, elev)
        assert np.allclose(score, 0.0)

    def test_coastal_face_scores_positive(self):
        mesh = _star_mesh()
        is_ocean = np.array([False, True, False, False, False])  # face 1 is ocean
        elev = np.array([0.1, -1.0, 0.1, 0.1, 0.1])  # face 1 at -1 km depth
        score = compute_harbor_quality(mesh, is_ocean, elev, depth_scale_km=1.0)
        # MID (face 0) has 1/4 ocean neighbors × depth_factor(-1.0/1.0)=1.0
        assert score[0] == pytest.approx(0.25, abs=1e-5)

    def test_more_ocean_neighbors_scores_higher(self):
        mesh = _star_mesh()
        # Two arm faces (1 and 2) are ocean — MID has 2/4 ocean neighbors.
        is_ocean = np.array([False, True, True, False, False])
        elev = np.array([0.1, -1.0, -1.0, 0.1, 0.1])
        score = compute_harbor_quality(mesh, is_ocean, elev, depth_scale_km=1.0)
        # 2/4 × 1.0 = 0.5
        assert score[0] == pytest.approx(0.5, abs=1e-5)

    def test_shallow_ocean_lowers_score(self):
        mesh = _star_mesh()
        is_ocean = np.array([False, True, False, False, False])
        shallow = np.array([0.1, -0.1, 0.1, 0.1, 0.1])  # only 0.1 km depth
        deep = np.array([0.1, -1.0, 0.1, 0.1, 0.1])
        s_shallow = compute_harbor_quality(mesh, is_ocean, shallow, depth_scale_km=1.0)
        s_deep = compute_harbor_quality(mesh, is_ocean, deep, depth_scale_km=1.0)
        assert s_deep[0] > s_shallow[0]

    def test_ocean_face_itself_scores_zero(self):
        mesh = _star_mesh()
        is_ocean = np.array([True, False, False, False, False])  # MID is ocean
        elev = np.full(5, -1.0)
        score = compute_harbor_quality(mesh, is_ocean, elev)
        assert score[0] == 0.0


# ── River confluences ─────────────────────────────────────────────────────────


class TestRiverConfluences:
    def test_no_rivers_returns_zeros(self):
        mesh = _star_mesh()
        score = compute_river_confluences(None, mesh.face_centroids, 5)
        assert np.allclose(score, 0.0)

    def test_single_river_scores_zero(self):
        mesh = _star_mesh()
        r = _make_river_tuple(mesh, 0, 1, 2)
        score = compute_river_confluences([r], mesh.face_centroids, 5)
        assert score[0] == pytest.approx(0.0)
        assert score[1] == pytest.approx(0.0)

    def test_two_rivers_at_same_face_scores_positive(self):
        mesh = _star_mesh()
        r1 = _make_river_tuple(mesh, 3, 2, 0)  # flows into face 0
        r2 = _make_river_tuple(mesh, 4, 0)     # also flows through face 0
        score = compute_river_confluences([r1, r2], mesh.face_centroids, 5)
        assert score[0] > 0.0

    def test_three_rivers_score_higher_than_two(self):
        mesh = _star_mesh()
        r1 = _make_river_tuple(mesh, 1, 0)
        r2 = _make_river_tuple(mesh, 2, 0)
        r3 = _make_river_tuple(mesh, 3, 0)
        s2 = compute_river_confluences([r1, r2], mesh.face_centroids, 5)
        s3 = compute_river_confluences([r1, r2, r3], mesh.face_centroids, 5)
        assert s3[0] > s2[0]

    def test_in_lake_segments_not_counted(self):
        # Two rivers both touch face 0, but river1's segment is in_lake — only
        # river2 counts, so face 0 has count=1 (no confluence, score=0).
        mesh = _star_mesh()
        r1 = _make_river_tuple(mesh, 1, 0, in_lake=True)   # in_lake
        r2 = _make_river_tuple(mesh, 2, 0, in_lake=False)  # not in_lake
        score = compute_river_confluences([r1, r2], mesh.face_centroids, 5)
        assert score[0] == pytest.approx(0.0)  # count=1 → tanh(0)=0


# ── Navigable river heads ─────────────────────────────────────────────────────


class TestNavigableHeads:
    def test_no_rivers_returns_zeros(self):
        mesh = _star_mesh()
        nav = np.zeros(5, dtype=np.float32)
        score = compute_navigable_heads(None, mesh.face_centroids, nav, 5)
        assert np.allclose(score, 0.0)

    def test_head_at_first_navigable_face(self):
        # River flows 3→2→1→0; navigability non-zero only at faces 0 and 1.
        mesh = _star_mesh()
        river = _make_river_tuple(mesh, 3, 2, 1, 0)
        nav = np.array([1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        score = compute_navigable_heads([river], mesh.face_centroids, nav, 5)
        # Walking source(3)→mouth(0): first navigable face = 1
        assert score[1] == pytest.approx(1.0)
        assert score[2] == pytest.approx(0.0)
        assert score[3] == pytest.approx(0.0)

    def test_all_navigable_head_at_source(self):
        # Entire river navigable — head is the source face.
        mesh = _star_mesh()
        river = _make_river_tuple(mesh, 3, 2, 1)
        nav = np.ones(5, dtype=np.float32)
        score = compute_navigable_heads([river], mesh.face_centroids, nav, 5)
        assert score[3] == pytest.approx(1.0)  # source face

    def test_fully_non_navigable_river_skipped(self):
        mesh = _star_mesh()
        river = _make_river_tuple(mesh, 0, 1, 2)
        nav = np.zeros(5, dtype=np.float32)
        score = compute_navigable_heads([river], mesh.face_centroids, nav, 5)
        assert np.allclose(score, 0.0)

    def test_in_lake_segments_skipped(self):
        # River: face3(non-nav) → face2(in_lake) → face1(nav) → face0
        mesh = _star_mesh()
        pos = mesh.face_centroids[[3, 2, 1, 0]].astype(np.float32)
        in_lake = np.array([False, True, False], dtype=bool)
        river = (pos, [1.0, 2.0, 3.0], in_lake)
        nav = np.array([1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        score = compute_navigable_heads([river], mesh.face_centroids, nav, 5)
        # face 3 non-navigable; face 2 in_lake (skipped); face 1 = first navigable
        assert score[1] == pytest.approx(1.0)


# ── Mountain passes ───────────────────────────────────────────────────────────


class TestMountainPasses:
    def _run(self, cost, is_water=None, **kw):
        mesh = _star_mesh()
        cost = np.array(cost, dtype=np.float32)
        if is_water is None:
            is_water = np.zeros(5, dtype=bool)
        return compute_mountain_passes(mesh, cost, is_water, params=MountainPassParams(**kw))

    def test_low_cost_amid_high_cost_scores_positive(self):
        # MID=1.0 surrounded by arms at 4.0 (above threshold)
        cost = np.array([1.0, 4.0, 4.0, 4.0, 4.0])
        score = self._run(cost, min_neighbor_cost=3.0)
        assert score[0] > 0.0

    def test_high_cost_face_surrounded_by_high_cost_scores_zero(self):
        # All equal — no contrast
        cost = np.full(5, 4.0, dtype=np.float32)
        score = self._run(cost, min_neighbor_cost=3.0)
        assert score[0] == pytest.approx(0.0)

    def test_low_cost_in_low_cost_neighborhood_scores_zero(self):
        # Neighbors below min_neighbor_cost threshold
        cost = np.array([0.5, 1.0, 1.0, 1.0, 1.0])
        score = self._run(cost, min_neighbor_cost=3.0)
        assert score[0] == pytest.approx(0.0)

    def test_water_face_scores_zero(self):
        cost = np.array([0.5, 4.0, 4.0, 4.0, 4.0])
        is_water = np.array([True, False, False, False, False])
        score = self._run(cost, is_water=is_water, min_neighbor_cost=3.0)
        assert score[0] == pytest.approx(0.0)

    def test_higher_contrast_scores_higher(self):
        low = np.array([1.0, 4.0, 4.0, 4.0, 4.0])
        high = np.array([0.5, 4.0, 4.0, 4.0, 4.0])  # greater contrast
        s_low = self._run(low, min_neighbor_cost=3.0)
        s_high = self._run(high, min_neighbor_cost=3.0)
        assert s_high[0] > s_low[0]


# ── Strait adjacency ──────────────────────────────────────────────────────────


class TestStraitAdjacency:
    def test_adjacent_to_strait_scores_one(self):
        mesh = _star_mesh()
        is_strait = np.array([False, True, False, False, False])  # face 1 is strait
        is_water = np.zeros(5, dtype=bool)
        score = compute_strait_adjacency(mesh, is_strait, is_water)
        # MID (face 0) is adjacent to face 1
        assert score[0] == pytest.approx(1.0)

    def test_not_adjacent_to_strait_scores_zero(self):
        mesh = _star_mesh()
        # Only face 3 is a strait; face 1 is not adjacent to face 3.
        is_strait = np.array([False, False, False, True, False])
        is_water = np.zeros(5, dtype=bool)
        score = compute_strait_adjacency(mesh, is_strait, is_water)
        assert score[1] == pytest.approx(0.0)

    def test_water_face_adjacent_to_strait_scores_zero(self):
        mesh = _star_mesh()
        is_strait = np.array([False, True, False, False, False])
        is_water = np.array([True, False, False, False, False])  # MID is water
        score = compute_strait_adjacency(mesh, is_strait, is_water)
        assert score[0] == pytest.approx(0.0)

    def test_no_straits_returns_zeros(self):
        mesh = _star_mesh()
        is_strait = np.zeros(5, dtype=bool)
        is_water = np.zeros(5, dtype=bool)
        score = compute_strait_adjacency(mesh, is_strait, is_water)
        assert np.allclose(score, 0.0)


# ── Combination ───────────────────────────────────────────────────────────────


class TestCombineSituationalValue:
    def _ones(self, n=5):
        return np.ones(n, dtype=np.float32)

    def _zeros(self, n=5):
        return np.zeros(n, dtype=np.float32)

    def test_water_face_forced_zero(self):
        is_water = np.array([True, False, False, False, False])
        out = combine_situational_value(
            is_water,
            self._ones(), self._ones(), self._ones(), self._ones(), self._ones(),
            params=SituationalValueWeightParams(),
        )
        assert out[0] == 0.0
        assert out[1] > 0.0

    def test_all_ones_gives_one(self):
        is_water = np.zeros(5, dtype=bool)
        out = combine_situational_value(
            is_water,
            self._ones(), self._ones(), self._ones(), self._ones(), self._ones(),
            params=SituationalValueWeightParams(),
        )
        assert np.allclose(out, 1.0)

    def test_bounded_in_zero_one(self):
        rng = np.random.default_rng(1)
        n = 20
        out = combine_situational_value(
            rng.random(n) < 0.3,
            rng.random(n).astype(np.float32),
            rng.random(n).astype(np.float32),
            rng.random(n).astype(np.float32),
            rng.random(n).astype(np.float32),
            rng.random(n).astype(np.float32),
            params=SituationalValueWeightParams(),
        )
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_weight_proportionality(self):
        # Only harbor = 1, rest = 0. Result = harbor_weight / total_weight.
        n = 3
        hw = 0.25
        total = 0.25 + 0.20 + 0.20 + 0.20 + 0.15
        out = combine_situational_value(
            np.zeros(n, dtype=bool),
            self._ones(n), self._zeros(n), self._zeros(n), self._zeros(n), self._zeros(n),
            params=SituationalValueWeightParams(
                harbor_weight=hw, confluence_weight=0.20, nav_head_weight=0.20,
                pass_weight=0.20, strait_weight=0.15),
        )
        assert out[0] == pytest.approx(hw / total, abs=1e-5)


# ── Builder integration ───────────────────────────────────────────────────────


class TestRealmPipelineSituational:
    def test_situational_value_shape_and_range(self, scoring_globe, scoring_realm):
        sv = scoring_realm.situational_value
        n = len(scoring_globe.mesh.faces)
        assert sv is not None
        assert sv.shape == (n,)
        assert sv.min() >= 0.0
        assert sv.max() <= 1.0

    def test_situational_value_metadata_written(self, scoring_globe, scoring_realm):
        assert "situational_value" in scoring_globe.mesh.face_metadata[0]

    def test_water_faces_score_zero(self, scoring_globe, scoring_realm):
        sv = scoring_realm.situational_value
        assert sv is not None
        for i, m in enumerate(scoring_globe.mesh.face_metadata):
            if m.get("is_water", m.get("is_ocean", False)):
                assert sv[i] == 0.0, f"face {i} is water but situational_value = {sv[i]}"


# ── Delta apexes ──────────────────────────────────────────────────────────────


class TestDeltaApexes:
    def test_marks_the_apex_faces(self):
        centroids = np.eye(3)
        out = compute_delta_apexes(np.array([[0.9, 0.1, 0.0]]), centroids, 3)
        assert out.tolist() == [1.0, 0.0, 0.0]
        assert compute_delta_apexes(None, centroids, 3).tolist() == [0.0, 0.0, 0.0]

    def test_weight_counts_only_with_deltas(self):
        n = 2
        ones, zeros = np.ones(n, dtype=np.float32), np.zeros(n, dtype=np.float32)
        p = SituationalValueWeightParams()
        without = combine_situational_value(np.zeros(n, dtype=bool), ones, ones, ones, ones, ones, params=p)
        assert np.allclose(without, 1.0)
        with_none = combine_situational_value(
            np.zeros(n, dtype=bool), ones, ones, ones, ones, ones, params=p, delta=zeros)
        total = p.harbor_weight + p.confluence_weight + p.nav_head_weight + p.pass_weight + p.strait_weight
        assert np.allclose(with_none, total / (total + p.delta_weight))
        apex = combine_situational_value(
            np.zeros(n, dtype=bool), zeros, zeros, zeros, zeros, zeros, params=p, delta=ones)
        assert np.allclose(apex, p.delta_weight / (total + p.delta_weight))
