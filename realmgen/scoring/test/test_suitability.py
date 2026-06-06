"""Tests for realmgen.suitability and RealmPipeline.

Each pure-function test uses a lightweight stub mesh so there is no dependency
on a full worldgen run.  The integration tests at the bottom use a real (small)
dual mesh built by the full GlobePipeline pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pytest

from core.pipeline import BuildContext
from core.world import World
from realmgen.scoring.suitability import (
    _positions_to_face_ids,
    combine_suitability,
    compute_climate_habitability,
    compute_resource_proximity,
    compute_ruggedness_penalty,
    compute_water_proximity,
)
from realmgen.builder import (
    ClimateHabitabilityParams,
    RealmPipeline,
    SuitabilityWeightParams,
    WaterProximityParams,
)
from worldgen import GlobePipeline, GlobeParams


def _norm(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


@dataclass
class _StubMesh:
    faces: List[tuple]
    face_adjacency: List[List[int]]
    face_centroids: np.ndarray


def _star_mesh() -> _StubMesh:
    """5-face star at the +Z pole, same layout used in cost tests.

    Index map: 0=MID, 1=+x, 2=+y, 3=-x, 4=-y. MID is adjacent to all arms;
    each arm touches only MID.
    """
    centroids = np.array([
        _norm((0.0, 0.0, 1.0)),
        _norm((0.25, 0.0, 1.0)),
        _norm((0.0, 0.25, 1.0)),
        _norm((-0.25, 0.0, 1.0)),
        _norm((0.0, -0.25, 1.0)),
    ])
    adj = [[1, 2, 3, 4], [0], [0], [0], [0]]
    return _StubMesh(faces=[(i,) for i in range(5)], face_adjacency=adj, face_centroids=centroids)


# ── Water proximity ───────────────────────────────────────────────────────────


class TestWaterProximity:
    def _run(self, freshwater_mask, dist_coast=None, **kw):
        mesh = _star_mesh()
        n = 5
        if dist_coast is None:
            dist_coast = np.full(n, 10000.0)  # far from coast by default
        return compute_water_proximity(
            mesh, freshwater_mask, dist_coast, params=WaterProximityParams(**kw)
        )

    def test_seed_face_scores_one(self):
        mask = np.array([True, False, False, False, False])
        score = self._run(mask, freshwater_decay=0.4, coast_weight=0.0)
        assert score[0] == pytest.approx(1.0)

    def test_decays_monotonically_with_hops(self):
        # River at MID (face 0); arm faces are 1 hop away.
        mask = np.array([True, False, False, False, False])
        score = self._run(mask, freshwater_decay=0.4, coast_weight=0.0)
        assert score[0] > score[1]
        assert score[1] == pytest.approx(score[2], abs=1e-5)  # all arms equidistant

    def test_coastal_contributes_when_no_freshwater(self):
        mask = np.zeros(5, dtype=bool)
        dist = np.full(5, 0.0)  # right at the coast
        score = self._run(mask, dist_coast=dist, coast_weight=0.4)
        assert score[0] > 0.0
        assert score[0] == pytest.approx(0.4, abs=1e-5)  # 0 + 0.4 * exp(0) = 0.4

    def test_far_from_both_scores_near_zero(self):
        mask = np.zeros(5, dtype=bool)
        dist = np.full(5, 1e9)   # very far from coast
        score = self._run(mask, dist_coast=dist, coast_scale_km=150.0)
        assert score.max() < 0.01

    def test_combined_freshwater_and_coast_clips_at_one(self):
        mask = np.array([True, False, False, False, False])
        dist = np.zeros(5)        # also at coast
        score = self._run(mask, dist_coast=dist, coast_weight=1.0)
        assert score[0] == pytest.approx(1.0)  # clipped to 1


# ── Climate habitability ──────────────────────────────────────────────────────


class TestClimateHabitability:
    def _run(self, tw, tc, pa, **kw):
        return compute_climate_habitability(
            np.atleast_1d(tw), np.atleast_1d(tc), np.atleast_1d(pa),
            params=ClimateHabitabilityParams(**kw),
        )

    def test_optimal_climate_scores_one(self):
        score = self._run(20.0, 5.0, 800.0)
        assert score[0] == pytest.approx(1.0, abs=1e-5)

    def test_frozen_growing_season_scores_zero(self):
        score = self._run(-5.0, 5.0, 800.0, t_grow_min_c=0.0)
        assert score[0] == 0.0

    def test_lethal_cold_winter_scores_zero(self):
        score = self._run(20.0, -40.0, 800.0, t_cold_min_c=-30.0)
        assert score[0] == 0.0

    def test_desert_precip_scores_zero(self):
        score = self._run(20.0, 5.0, 50.0, p_dry_mm=100.0)
        assert score[0] == 0.0

    def test_partial_cold_tolerance(self):
        # T_coldest at the midpoint of the survival range → cold score ≈ 0.5
        score = self._run(25.0, -15.0, 800.0, t_cold_min_c=-30.0)
        # growing = 1, precip = 1, cold ≈ 0.5 → product ≈ 0.5
        assert score[0] == pytest.approx(0.5, abs=0.02)

    def test_wet_precip_stays_nonzero(self):
        # Above p_wet_mm → floors at 0.5, rest of climate perfect → 0.5
        score = self._run(20.0, 5.0, 5000.0, p_wet_mm=3000.0)
        assert score[0] == pytest.approx(0.5, abs=1e-5)

    def test_output_shape_matches_input(self):
        tw = np.array([20.0, -5.0, 30.0])
        tc = np.array([5.0, -35.0, 10.0])
        pa = np.array([800.0, 200.0, 50.0])
        out = compute_climate_habitability(tw, tc, pa, params=ClimateHabitabilityParams())
        assert out.shape == (3,)
        assert out[0] == pytest.approx(1.0, abs=1e-5)  # optimal
        assert out[1] == 0.0   # cold winter kills it
        assert out[2] == 0.0   # desert kills it


# ── Ruggedness penalty ────────────────────────────────────────────────────────


class TestRuggedness:
    def test_flat_terrain_has_no_penalty(self):
        mult = compute_ruggedness_penalty(np.array([0.0]), penalty_scale_km=1.0)
        assert mult[0] == pytest.approx(1.0)

    def test_penalty_at_one_scale_length(self):
        mult = compute_ruggedness_penalty(np.array([1.0]), penalty_scale_km=1.0)
        assert mult[0] == pytest.approx(np.exp(-1.0), abs=1e-5)

    def test_penalty_increases_with_relief(self):
        mults = compute_ruggedness_penalty(np.array([0.0, 0.5, 1.0, 2.0]), penalty_scale_km=1.0)
        assert np.all(np.diff(mults) < 0)  # strictly decreasing

    def test_penalty_never_reaches_zero(self):
        mult = compute_ruggedness_penalty(np.array([100.0]), penalty_scale_km=1.0)
        assert mult[0] > 0.0

    def test_multiplied_into_suitability(self):
        # A perfectly suitable flat face (mult=1) vs a rugged copy (mult<1).
        n = 2
        suit = np.ones(n, dtype=np.float32)
        mults = compute_ruggedness_penalty(np.array([0.0, 2.0]), penalty_scale_km=1.0)
        result = (suit * mults).astype(np.float32)
        assert result[0] == pytest.approx(1.0)
        assert result[1] == pytest.approx(float(np.exp(-2.0)), abs=1e-5)


# ── Resource access ───────────────────────────────────────────────────────────


class TestResourceProximity:
    def test_deposit_face_scores_one(self):
        mesh = _star_mesh()
        mask = np.array([True, False, False, False, False])
        score = compute_resource_proximity(mesh, mask, resource_decay=0.2)
        assert score[0] == pytest.approx(1.0)

    def test_decays_with_hops(self):
        mesh = _star_mesh()
        mask = np.array([True, False, False, False, False])
        score = compute_resource_proximity(mesh, mask, resource_decay=0.5)
        assert score[0] > score[1]  # seed > 1 hop away

    def test_no_resources_returns_zeros(self):
        mesh = _star_mesh()
        mask = np.zeros(5, dtype=bool)
        score = compute_resource_proximity(mesh, mask)
        assert np.allclose(score, 0.0)


class TestPositionsToFaceIds:
    def test_returns_nearest_centroid(self):
        centroids = np.array([
            _norm((1.0, 0.0, 0.0)),
            _norm((0.0, 1.0, 0.0)),
            _norm((0.0, 0.0, 1.0)),
        ])
        pos = np.array([_norm((1.0, 0.1, 0.0))])   # closest to face 0
        fids = _positions_to_face_ids(pos, centroids)
        assert fids[0] == 0

    def test_scalar_position(self):
        centroids = np.array([
            _norm((1.0, 0.0, 0.0)),
            _norm((0.0, 0.0, 1.0)),
        ])
        pos = _norm((0.1, 0.0, 1.0))   # scalar (3,) — close to face 1
        fid = _positions_to_face_ids(pos, centroids)
        assert int(fid[0]) == 1


# ── Combination ───────────────────────────────────────────────────────────────


class TestCombineSuitability:
    def _ones(self, n=5):
        return np.ones(n, dtype=np.float32)

    def _zeros(self, n=5):
        return np.zeros(n, dtype=np.float32)

    def test_water_face_forced_zero(self):
        is_water = np.array([True, False, False, False, False])
        out = combine_suitability(is_water,
                                  self._ones(), self._ones(),
                                  self._ones(), self._ones(),
                                  params=SuitabilityWeightParams())
        assert out[0] == 0.0
        assert out[1] > 0.0

    def test_all_ones_gives_one(self):
        n = 5
        out = combine_suitability(np.zeros(n, dtype=bool),
                                  self._ones(), self._ones(),
                                  self._ones(), self._ones(),
                                  params=SuitabilityWeightParams())
        assert np.allclose(out, 1.0)

    def test_bounded_in_zero_one(self):
        rng = np.random.default_rng(0)
        n = 20
        out = combine_suitability(
            rng.random(n) < 0.3,
            rng.random(n).astype(np.float32),
            rng.random(n).astype(np.float32),
            rng.random(n).astype(np.float32),
            rng.random(n).astype(np.float32),
            params=SuitabilityWeightParams(),
        )
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_weight_proportionality(self):
        # Only water_proximity = 1, others = 0, no water.  Result = water_weight / total.
        n = 3
        is_water = np.zeros(n, dtype=bool)
        ww = 0.30
        total = 0.30 + 0.30 + 0.25 + 0.15
        out = combine_suitability(is_water,
                                  self._ones(n), self._zeros(n),
                                  self._zeros(n), self._zeros(n),
                                  params=SuitabilityWeightParams(
                                      water_weight=ww, climate_weight=0.30,
                                      fertility_weight=0.25, resource_weight=0.15))
        assert out[0] == pytest.approx(ww / total, abs=1e-5)


# ── Builder integration ───────────────────────────────────────────────────────


class TestRealmPipeline:
    def _small_globe(self):
        p = GlobeParams()
        p.seed = 7
        ctx = BuildContext(world=World(p.mesh, seed=p.seed))
        GlobePipeline(p).build(ctx)
        return ctx.world

    def _run_realm(self, globe) -> "tuple[RealmPipeline, np.ndarray]":
        realm = RealmPipeline()
        realm.build(BuildContext(world=globe))
        return realm, realm.suitability

    def test_build_output_shape_and_range(self):
        globe = self._small_globe()
        n = len(globe.mesh.faces)
        _, suit = self._run_realm(globe)
        assert suit.shape == (n,)
        assert suit.min() >= 0.0
        assert suit.max() <= 1.0

    def test_suitability_metadata_written(self):
        globe = self._small_globe()
        self._run_realm(globe)
        assert "suitability" in globe.mesh.face_metadata[0]

    def test_water_faces_score_zero(self):
        globe = self._small_globe()
        _, suit = self._run_realm(globe)
        mesh = globe.mesh
        for i, m in enumerate(mesh.face_metadata):
            if m.get("is_water", m.get("is_ocean", False)):
                assert suit[i] == 0.0, f"face {i} is water but suitability = {suit[i]}"

    def test_land_face_suitability_nonzero_in_habitable_world(self):
        globe = self._small_globe()
        _, suit = self._run_realm(globe)
        mesh = globe.mesh
        land_suit = [suit[i] for i, m in enumerate(mesh.face_metadata)
                     if not m.get("is_water", m.get("is_ocean", False))]
        assert len(land_suit) > 0
        assert max(land_suit) > 0.1
