"""Unit tests for `worldgen.cost.surfaces` and `CostBuilder`.

The river-aware land edge cost is the subtle part, so most of the file exercises
it: direction attribution, the along-valley discount vs. across-river ford
penalty, flow-magnitude scaling, impassability, and clamping.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pytest

from worldgen.cost.surfaces import (
    BIOME_TRAVERSAL_COST,
    build_land_edge_costs,
    build_sea_edge_costs,
    compute_ice_penalty,
    compute_land_base_cost,
    compute_river_flow_field,
    compute_river_navigability,
    compute_sea_cost,
    compute_sea_distance_to_coast,
    detect_straits,
    navigable_river_mask,
)
from worldgen.hydrology.types import River, RiverNode, RiverSegment
from core.mesh import Mesh
from core.pipeline import BuildContext
from core.world import World
from worldgen.cost.builder import CostBuilder, SeaCostParams, StraitParams
from worldgen.climate.classification import N_BIOMES


def _norm(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def _run_cost(dual, rivers=None):
    """Run the CostBuilder stage against a stub mesh and return its surfaces.

    Rivers (River objects) are supplied via the ``"river_objects"`` world
    feature, the same channel the hydrology stage registers them on.
    """
    world = World.from_mesh(dual)
    if rivers is not None:
        world.add_feature("river_objects", rivers)
    return CostBuilder().build(BuildContext(world=world))


@dataclass
class _StubMesh:
    """Minimal stand-in — only the attributes the cost functions touch."""

    faces: List[tuple]
    face_adjacency: List[List[int]]
    face_centroids: np.ndarray


def _star_mesh() -> _StubMesh:
    """5-face star centred on the +Z pole.

    Index map: 0=MID (at the pole), 1=+x, 2=+y, 3=-x, 4=-y. MID is adjacent to
    all four arms; each arm touches only MID. The tangent plane at MID is the
    x-y plane, so a +x river flow is exactly parallel to the MID→(+x) step and
    perpendicular to the MID→(+y) step — letting us test "along" vs "across"
    cleanly. `face_adjacency[0]` is `[1, 2, 3, 4]`, i.e. [+x, +y, -x, -y].
    """
    centroids = np.array([
        _norm((0.0, 0.0, 1.0)),    # 0 MID
        _norm((0.25, 0.0, 1.0)),   # 1 +x
        _norm((0.0, 0.25, 1.0)),   # 2 +y
        _norm((-0.25, 0.0, 1.0)),  # 3 -x
        _norm((0.0, -0.25, 1.0)),  # 4 -y
    ])
    adj = [[1, 2, 3, 4], [0], [0], [0], [0]]
    faces = [(i,) for i in range(5)]
    return _StubMesh(faces=faces, face_adjacency=adj, face_centroids=centroids)


def _seg(p_start, fid_start, p_end, fid_end, flow, in_lake=False) -> RiverSegment:
    a = RiverNode(position=np.asarray(p_start, dtype=np.float32), elevation=0.0,
                  face_id=fid_start, uv=(0.0, 0.0))
    b = RiverNode(position=np.asarray(p_end, dtype=np.float32), elevation=0.0,
                  face_id=fid_end, uv=(0.0, 0.0))
    return RiverSegment(start=a, end=b, flow_rate=float(flow), in_lake=in_lake)


# ── Land base cost ───────────────────────────────────────────────────────────


class TestComputeLandBaseCost:
    def test_biome_plus_ruggedness(self):
        biome = np.array([3, 8, 6])          # Temperate, Desert, Tropical
        elev = np.array([0.0, 1.0, 2.0])
        ocean = np.array([False, False, True])
        cost, passable = compute_land_base_cost(
            biome, elev, ocean, biome_cost_scale=1.0, ruggedness_cost_per_km=1.5
        )
        assert cost[0] == pytest.approx(1.0)              # Temperate, flat
        assert cost[1] == pytest.approx(3.0 + 1.5)        # Desert + 1 km range
        assert cost[2] == pytest.approx(2.5 + 3.0)        # Tropical + 2 km range
        assert passable.tolist() == [True, True, False]

    def test_scale_zero_is_pure_ruggedness(self):
        cost, _ = compute_land_base_cost(
            np.array([8]), np.array([2.0]), np.array([False]),
            biome_cost_scale=0.0, ruggedness_cost_per_km=1.0,
        )
        assert cost[0] == pytest.approx(2.0)

    def test_biome_index_clamped(self):
        cost, _ = compute_land_base_cost(
            np.array([99]), np.array([0.0]), np.array([False])
        )
        assert cost[0] == pytest.approx(BIOME_TRAVERSAL_COST[-1])

    def test_negative_elev_range_floored(self):
        cost, _ = compute_land_base_cost(
            np.array([3]), np.array([-5.0]), np.array([False]),
            biome_cost_scale=1.0, ruggedness_cost_per_km=1.0,
        )
        assert cost[0] == pytest.approx(1.0)  # ruggedness clamped at 0


# ── River flow field ─────────────────────────────────────────────────────────


class TestRiverFlowField:
    def test_direction_and_magnitude(self):
        mesh = _star_mesh()
        river = River(segments=[_seg((-0.1, 0, 1), 0, (0.1, 0, 1), 0, 500.0)])
        fdir, fmag = compute_river_flow_field(mesh, [river])
        assert fmag[0] == pytest.approx(500.0)
        assert np.allclose(fdir[0], [1, 0, 0], atol=1e-5)
        assert fmag[1] == 0.0
        assert np.allclose(fdir[1], 0.0)

    def test_attributes_to_both_touched_faces(self):
        mesh = _star_mesh()
        river = River(segments=[_seg((0, 0, 1), 0, (0.25, 0, 1), 1, 700.0)])
        _, fmag = compute_river_flow_field(mesh, [river])
        assert fmag[0] == pytest.approx(700.0)
        assert fmag[1] == pytest.approx(700.0)

    def test_magnitude_is_max_of_segments(self):
        mesh = _star_mesh()
        river = River(segments=[
            _seg((-0.1, 0, 1), 0, (0, 0, 1), 0, 300.0),
            _seg((0, 0, 1), 0, (0.1, 0, 1), 0, 900.0),
        ])
        _, fmag = compute_river_flow_field(mesh, [river])
        assert fmag[0] == pytest.approx(900.0)

    def test_in_lake_segments_skipped_by_default(self):
        mesh = _star_mesh()
        river = River(segments=[_seg((-0.1, 0, 1), 0, (0.1, 0, 1), 0, 500.0, in_lake=True)])
        _, fmag = compute_river_flow_field(mesh, [river])
        assert fmag[0] == 0.0
        _, fmag2 = compute_river_flow_field(mesh, [river], include_in_lake=True)
        assert fmag2[0] == pytest.approx(500.0)

    def test_tangent_projection_removes_radial(self):
        # Face centroid on +x; a flow with a radial (+x) component must project
        # to the pure tangent (+y) direction.
        mesh = _StubMesh(faces=[(0,)], face_adjacency=[[]],
                         face_centroids=np.array([_norm((1.0, 0.0, 0.0))]))
        river = River(segments=[_seg((1.0, 0.0, 0.0), 0, (1.2, 0.1, 0.0), 0, 400.0)])
        fdir, _ = compute_river_flow_field(mesh, [river])
        assert np.allclose(fdir[0], [0, 1, 0], atol=1e-6)
        assert abs(float(fdir[0] @ np.array([1.0, 0.0, 0.0]))) < 1e-6  # tangent

    def test_zero_length_segment_ignored(self):
        mesh = _star_mesh()
        river = River(segments=[_seg((0.1, 0, 1), 0, (0.1, 0, 1), 0, 500.0)])
        fdir, fmag = compute_river_flow_field(mesh, [river])
        assert fmag[0] == 0.0
        assert np.allclose(fdir[0], 0.0)

    def test_no_rivers(self):
        mesh = _star_mesh()
        fdir, fmag = compute_river_flow_field(mesh, None)
        assert np.allclose(fmag, 0.0)
        assert np.allclose(fdir, 0.0)

    def test_exclude_mask_zeros_water_faces(self):
        # A river mouth segment touches a water face (the +x arm here); with that
        # face in exclude_mask it must carry no flow, while the land face still does.
        mesh = _star_mesh()
        river = River(segments=[_seg((0, 0, 1), 0, (0.25, 0, 1), 1, 700.0)])
        exclude = np.array([False, True, False, False, False])  # +x arm is water
        fdir, fmag = compute_river_flow_field(mesh, [river], exclude_mask=exclude)
        assert fmag[0] == pytest.approx(700.0)   # land face keeps the flow
        assert fmag[1] == 0.0                     # excluded water face stays zero
        assert np.allclose(fdir[1], 0.0)


# ── River navigability — the fall line ───────────────────────────────────────


class TestRiverNavigability:
    def test_flat_river_fully_navigable(self):
        nav = compute_river_navigability(
            np.array([1000.0]), np.array([0.0]), navigable_ruggedness_km=0.6
        )
        assert nav[0] == pytest.approx(1.0)

    def test_taper_linear_with_relief(self):
        # Half the limiting relief → halfway-navigable.
        nav = compute_river_navigability(
            np.array([1000.0]), np.array([0.3]), navigable_ruggedness_km=0.6
        )
        assert nav[0] == pytest.approx(0.5, abs=1e-6)

    def test_rugged_river_is_zero(self):
        nav = compute_river_navigability(
            np.array([1000.0]), np.array([1.0]), navigable_ruggedness_km=0.6
        )
        assert nav[0] == 0.0

    def test_no_river_is_zero_even_if_flat(self):
        nav = compute_river_navigability(
            np.array([0.0]), np.array([0.0]), navigable_ruggedness_km=0.6
        )
        assert nav[0] == 0.0


class TestNavigableRiverMask:
    def test_requires_flow_navigability_and_not_water(self):
        flow = np.array([1000.0, 1000.0, 100.0, 1000.0])
        nav = np.array([1.0, 0.0, 1.0, 1.0])              # face 1 is at the fall line
        is_water = np.array([False, False, False, True])  # face 3 already water
        mask = navigable_river_mask(flow, nav, is_water, navigable_flow_min=500.0)
        assert mask.tolist() == [True, False, False, False]

    def test_flow_threshold(self):
        flow = np.array([400.0, 600.0])
        nav = np.array([1.0, 1.0])
        is_water = np.array([False, False])
        mask = navigable_river_mask(flow, nav, is_water, navigable_flow_min=500.0)
        assert mask.tolist() == [False, True]


# ── Land edge cost — the river anisotropy ────────────────────────────────────


class TestLandEdgeCosts:
    def _costs(self, *, flow_mag_mid=4000.0, corridor=0.5, ford=2.0,
               min_mult=0.4, max_mult=5.0, ref=2000.0, land=None, passable=None):
        mesh = _star_mesh()
        n = 5
        land = np.ones(n) if land is None else land
        passable = np.ones(n, dtype=bool) if passable is None else passable
        fdir = np.zeros((n, 3))
        fmag = np.zeros(n)
        fdir[0] = [1, 0, 0]               # MID river flows +x
        fmag[0] = flow_mag_mid
        costs = build_land_edge_costs(
            mesh, land, passable, fdir, fmag,
            river_ford_strength=ford, river_corridor_strength=corridor,
            river_flow_reference=ref, river_min_mult=min_mult, river_max_mult=max_mult,
        )
        return mesh, costs

    def test_edge_list_mirrors_adjacency(self):
        mesh, costs = self._costs()
        assert len(costs) == len(mesh.face_adjacency)
        for c, nbrs in zip(costs, mesh.face_adjacency):
            assert len(c) == len(nbrs)

    def test_along_valley_discounted_ford_penalized(self):
        # adjacency[0] order = [+x, +y, -x, -y]
        _, costs = self._costs()
        along_px, ford_py, along_nx, ford_ny = costs[0]
        assert along_px < 1.0 < ford_py                     # the core property
        assert along_px == pytest.approx(0.5, abs=0.03)     # base·(1 - corridor)
        assert ford_py == pytest.approx(3.0, abs=0.05)      # base·(1 + ford)
        assert along_nx == pytest.approx(along_px, abs=1e-6)  # ±x symmetric
        assert ford_ny == pytest.approx(ford_py, abs=1e-6)

    def test_directed_edges_are_symmetric(self):
        _, costs = self._costs()
        assert costs[0][0] == pytest.approx(costs[1][0])  # MID→+x == +x→MID
        assert costs[0][1] == pytest.approx(costs[2][0])  # MID→+y == +y→MID

    def test_flow_magnitude_scales_effect(self):
        # Half the reference flow → w = 0.5 → milder multipliers both ways.
        _, costs = self._costs(flow_mag_mid=1000.0, ref=2000.0)
        along_px, ford_py = costs[0][0], costs[0][1]
        assert ford_py == pytest.approx(2.0, abs=0.05)    # 1 + 0.5·2
        assert along_px == pytest.approx(0.75, abs=0.03)  # 1 - 0.5·0.5

    def test_no_river_is_plain_base(self):
        mesh = _star_mesh()
        n = 5
        costs = build_land_edge_costs(
            mesh, np.full(n, 2.0), np.ones(n, bool),
            np.zeros((n, 3)), np.zeros(n),
        )
        for c in costs[0]:
            assert c == pytest.approx(2.0)

    def test_impassable_endpoint_is_inf(self):
        mesh = _star_mesh()
        n = 5
        passable = np.ones(n, bool)
        passable[2] = False  # +y face (adjacency[0] index 1) is ocean
        costs = build_land_edge_costs(
            mesh, np.ones(n), passable, np.zeros((n, 3)), np.zeros(n),
        )
        assert np.isinf(costs[0][1])   # MID→+y
        assert np.isinf(costs[2][0])   # +y→MID (reverse)
        assert np.isfinite(costs[0][0])

    def test_base_cost_is_endpoint_mean(self):
        mesh = _star_mesh()
        n = 5
        land = np.array([1.0, 3.0, 1.0, 1.0, 1.0])  # +x face costs 3
        costs = build_land_edge_costs(
            mesh, land, np.ones(n, bool), np.zeros((n, 3)), np.zeros(n),
        )
        assert costs[0][0] == pytest.approx(2.0)  # mean(1, 3)

    def test_two_aligned_river_faces_clamped_to_min(self):
        # Both endpoints river faces, both aligned → product below min → clamp.
        mesh = _star_mesh()
        n = 5
        fdir = np.zeros((n, 3))
        fmag = np.zeros(n)
        fdir[0] = [1, 0, 0]
        fdir[1] = [1, 0, 0]
        fmag[0] = 4000.0
        fmag[1] = 4000.0
        costs = build_land_edge_costs(
            mesh, np.ones(n), np.ones(n, bool), fdir, fmag,
            river_corridor_strength=0.5, river_min_mult=0.4,
        )
        # factor ≈ 0.5 · 0.5 = 0.25, clamped up to 0.4.
        assert costs[0][0] == pytest.approx(0.4, abs=0.01)

    def test_chain_cheaper_to_follow_than_to_cross(self):
        # A straight river through MID: following it (±x) is cheaper than
        # crossing it (±y), and crossing exceeds the no-river base.
        _, costs = self._costs()
        follow = min(costs[0][0], costs[0][2])
        cross = max(costs[0][1], costs[0][3])
        assert follow < 1.0 < cross

    def test_navigability_tapers_corridor_discount(self):
        # Fall line: where the river is unnavigable (nav=0) the along-valley
        # discount disappears, but the ford penalty across it remains.
        mesh = _star_mesh()
        n = 5
        fdir = np.zeros((n, 3))
        fmag = np.zeros(n)
        fdir[0] = [1, 0, 0]
        fmag[0] = 4000.0

        nav1 = np.zeros(n)
        nav1[0] = 1.0      # placid, fully navigable
        nav0 = np.zeros(n)  # rapids at the fall line

        costs_nav = build_land_edge_costs(mesh, np.ones(n), np.ones(n, bool), fdir, fmag, nav1)
        costs_rap = build_land_edge_costs(mesh, np.ones(n), np.ones(n, bool), fdir, fmag, nav0)

        # along +x (adjacency index 0), ford +y (index 1)
        assert costs_nav[0][0] < 1.0                      # navigable → discounted
        assert costs_rap[0][0] > costs_nav[0][0]          # rapids → discount removed
        assert costs_rap[0][0] == pytest.approx(1.0, abs=0.03)  # ~ base, no discount
        assert costs_rap[0][1] == pytest.approx(costs_nav[0][1], abs=1e-6)  # ford unchanged
        assert costs_rap[0][1] > 1.0                      # still costly to ford

    def test_default_navigability_is_full_discount(self):
        # Omitting river_navigability behaves like nav=1 everywhere.
        mesh = _star_mesh()
        n = 5
        fdir = np.zeros((n, 3))
        fmag = np.zeros(n)
        fdir[0] = [1, 0, 0]
        fmag[0] = 4000.0
        nav1 = np.zeros(n)
        nav1[0] = 1.0
        c_default = build_land_edge_costs(mesh, np.ones(n), np.ones(n, bool), fdir, fmag)
        c_nav1 = build_land_edge_costs(mesh, np.ones(n), np.ones(n, bool), fdir, fmag, nav1)
        assert c_default[0][0] == pytest.approx(c_nav1[0][0], abs=1e-9)


# ── Straits ──────────────────────────────────────────────────────────────────


class TestDetectStraits:
    def _mesh(self, centroids, adj):
        return _StubMesh([(i,) for i in range(len(centroids))], adj, np.array(centroids))

    def test_pinched_channel_is_strait(self):
        # Ocean MID with land on ±x, ocean on ±y → a through-channel.
        centroids = [_norm((0, 0, 1)), _norm((0.25, 0, 1)), _norm((-0.25, 0, 1)),
                     _norm((0, 0.25, 1)), _norm((0, -0.25, 1))]
        adj = [[1, 2, 3, 4], [0], [0], [0], [0]]
        mesh = self._mesh(centroids, adj)
        is_ocean = np.array([True, False, False, True, True])
        strait = detect_straits(mesh, is_ocean, params=StraitParams())
        assert strait[0]
        assert not strait[1:].any()

    def test_bay_not_strait(self):
        # Both land neighbours on the same (+x) side → not opposite → not a strait.
        centroids = [_norm((0, 0, 1)), _norm((0.25, 0.1, 1)), _norm((0.25, -0.1, 1)),
                     _norm((-0.25, 0, 1)), _norm((0, 0.25, 1))]
        adj = [[1, 2, 3, 4], [0], [0], [0], [0]]
        mesh = self._mesh(centroids, adj)
        is_ocean = np.array([True, False, False, True, True])
        assert not detect_straits(mesh, is_ocean, params=StraitParams())[0]

    def test_open_ocean_not_strait(self):
        centroids = [_norm((0, 0, 1)), _norm((0.25, 0, 1)), _norm((-0.25, 0, 1))]
        adj = [[1, 2], [0], [0]]
        mesh = self._mesh(centroids, adj)
        assert not detect_straits(mesh, np.array([True, True, True]), params=StraitParams()).any()

    def test_land_face_is_never_strait(self):
        centroids = [_norm((0, 0, 1)), _norm((0.25, 0, 1)), _norm((-0.25, 0, 1))]
        adj = [[1, 2], [0], [0]]
        mesh = self._mesh(centroids, adj)
        assert not detect_straits(mesh, np.array([False, False, False]), params=StraitParams()).any()

    def test_min_water_neighbors_required(self):
        # Pinched by land on both sides but no water neighbour → enclosed, not a
        # strait when min_water_neighbors=1.
        centroids = [_norm((0, 0, 1)), _norm((0.25, 0, 1)), _norm((-0.25, 0, 1))]
        adj = [[1, 2], [0], [0]]
        mesh = self._mesh(centroids, adj)
        is_water = np.array([True, False, False])
        assert not detect_straits(mesh, is_water, params=StraitParams(min_water_neighbors=1))[0]
        assert detect_straits(mesh, is_water, params=StraitParams(min_water_neighbors=0))[0]


# ── Sea cost ─────────────────────────────────────────────────────────────────


class TestSeaCost:
    def test_ramp_strait_discount_and_land(self):
        is_ocean = np.array([True, True, True, False])
        dist = np.array([0.0, 1000.0, 4000.0, 0.0])
        strait = np.array([False, False, True, False])
        cost, passable = compute_sea_cost(
            is_ocean, dist, strait,
            params=SeaCostParams(
                coastal_cost=0.5, open_ocean_cost=2.0,
                open_ocean_distance_km=2000.0, strait_discount=0.5,
            ),
        )
        assert cost[0] == pytest.approx(0.5)    # at the shore
        assert cost[1] == pytest.approx(1.25)   # halfway to open ocean
        assert cost[2] == pytest.approx(1.0)    # t clamped to 1 → 2.0, strait ×0.5
        assert cost[3] == pytest.approx(0.0)    # land
        assert passable.tolist() == [True, True, True, False]


class TestIcePenalty:
    def test_no_freezing_returns_one(self):
        temp = np.full((3, 4, 4, 4), 10.0, dtype=np.float32)
        penalty = compute_ice_penalty(temp, freeze_threshold_celsius=0.0)
        assert np.allclose(penalty, 1.0)

    def test_always_freezing_returns_max_penalty(self):
        temp = np.full((3, 4, 4, 4), -10.0, dtype=np.float32)
        penalty = compute_ice_penalty(temp, freeze_threshold_celsius=0.0, penalty_strength=5.0)
        assert np.allclose(penalty, 6.0)

    def test_half_year_freezing_returns_half_penalty(self):
        temp = np.zeros((3, 4, 4, 4), dtype=np.float32)
        temp[:, :, :, :2] = -10.0  # 2 cold seasons
        temp[:, :, :, 2:] = 10.0   # 2 warm seasons
        penalty = compute_ice_penalty(temp, freeze_threshold_celsius=0.0, penalty_strength=5.0)
        assert np.allclose(penalty, 3.5)  # 1 + 0.5 * 5

    def test_penalty_strength_scales_effect(self):
        temp = np.full((3, 4, 4, 4), -5.0, dtype=np.float32)
        p1 = compute_ice_penalty(temp, freeze_threshold_celsius=0.0, penalty_strength=2.0)
        p2 = compute_ice_penalty(temp, freeze_threshold_celsius=0.0, penalty_strength=4.0)
        assert np.allclose(p1, 3.0)  # 1 + 1 * 2
        assert np.allclose(p2, 5.0)  # 1 + 1 * 4

    def test_different_freeze_threshold(self):
        temp = np.full((3, 4, 4, 4), 5.0, dtype=np.float32)
        p_above = compute_ice_penalty(temp, freeze_threshold_celsius=10.0, penalty_strength=5.0)
        p_below = compute_ice_penalty(temp, freeze_threshold_celsius=0.0, penalty_strength=5.0)
        assert np.allclose(p_above, 6.0)  # all seasons below 10°C
        assert np.allclose(p_below, 1.0)  # no seasons below 0°C


class TestSeaDistanceToCoast:
    def _chain(self, is_ocean):
        # Four faces in a row along a meridian; adjacency 0-1-2-3.
        centroids = np.array([
            _norm((1.0, 0.0, 0.0)),
            _norm((1.0, 0.1, 0.0)),
            _norm((1.0, 0.2, 0.0)),
            _norm((1.0, 0.3, 0.0)),
        ])
        adj = [[1], [0, 2], [1, 3], [2]]
        mesh = _StubMesh([(i,) for i in range(4)], adj, centroids)
        return mesh

    def test_increases_away_from_coast(self):
        mesh = self._chain([False, True, True, True])
        dist = compute_sea_distance_to_coast(mesh, np.array([False, True, True, True]))
        assert dist[0] == 0.0                 # land
        assert dist[1] == pytest.approx(0.0)  # coastal water (adjacent to land)
        assert dist[2] > dist[1]
        assert dist[3] > dist[2]

    def test_distance_in_km_scale(self):
        # Adjacent ocean faces ~0.1 rad apart → ~637 km per hop on a 6371 km sphere.
        mesh = self._chain([False, True, True, True])
        dist = compute_sea_distance_to_coast(mesh, np.array([False, True, True, True]))
        assert dist[2] == pytest.approx(6371.0 * np.arccos(np.clip(
            mesh.face_centroids[1] @ mesh.face_centroids[2], -1.0, 1.0)), rel=1e-4)

    def test_no_land_returns_zero(self):
        mesh = self._chain([True, True, True, True])
        dist = compute_sea_distance_to_coast(mesh, np.array([True, True, True, True]))
        assert np.allclose(dist, 0.0)


class TestSeaEdgeCosts:
    def test_mean_and_inf(self):
        mesh = _star_mesh()
        sea = np.array([0.5, 1.5, 0.5, 0.5, 0.5])
        passable = np.array([True, True, False, True, True])  # face 2 (+y) is land
        costs = build_sea_edge_costs(mesh, sea, passable)
        assert costs[0][0] == pytest.approx(1.0)   # MID→+x, mean(0.5, 1.5)
        assert np.isinf(costs[0][1])               # MID→+y (land)


# ── Builder integration ──────────────────────────────────────────────────────


class TestCostBuilder:
    def _small_mesh(self):
        dual = Mesh.icosphere(1).mesh_dual()
        n = len(dual.faces)
        rng = np.random.default_rng(0)
        biome = rng.integers(0, N_BIOMES, size=n)
        is_ocean = rng.random(n) < 0.4
        dist = rng.random(n) * 3000.0
        dual.apply_metadata_channel("biome", biome.tolist())
        dual.apply_metadata_channel("is_ocean", is_ocean.astype(bool).tolist())
        dual.apply_metadata_channel("distance_to_coast", dist.tolist())
        elev = rng.standard_normal((n, 4, 4, 1)).astype(np.float32)
        dual.set_face_textures("elevation", elev, n_channels=1)
        return dual, n, is_ocean

    def _flat_mesh(self, seed=1):
        """Like `_small_mesh` but with zero (flat) elevation, so any river face is
        fully navigable — lets us exercise lakes-as-water and navigable rivers."""
        dual = Mesh.icosphere(1).mesh_dual()
        n = len(dual.faces)
        rng = np.random.default_rng(seed)
        biome = rng.integers(0, N_BIOMES, size=n)
        is_ocean = rng.random(n) < 0.4
        dual.apply_metadata_channel("biome", biome.tolist())
        dual.apply_metadata_channel("is_ocean", is_ocean.astype(bool).tolist())
        dual.set_face_textures("elevation", np.zeros((n, 4, 4, 1), np.float32), n_channels=1)
        return dual, n, is_ocean

    def _tangent_river(self, dual, f, flow):
        c = np.asarray(dual.face_centroids[f], dtype=np.float64)
        ref = np.array([0.0, 0.0, 1.0]) if abs(c[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        tangent = np.cross(ref, c)
        tangent /= np.linalg.norm(tangent)
        return River(segments=[_seg(c - 0.05 * tangent, f, c + 0.05 * tangent, f, flow)])

    def test_build_outputs_and_metadata(self):
        dual, n, is_ocean = self._small_mesh()
        surf = _run_cost(dual)
        assert surf.land_base_cost.shape == (n,)
        assert surf.land_passable.tolist() == (~is_ocean).tolist()
        assert surf.sea_passable.tolist() == is_ocean.tolist()
        assert np.allclose(surf.river_flow_mag, 0.0)         # no rivers supplied
        assert np.allclose(surf.river_navigability, 0.0)
        assert np.all(surf.sea_base_cost >= 0.0)
        for key in ("land_base_cost", "sea_base_cost", "is_strait",
                    "is_water", "is_navigable_river", "river_navigability"):
            assert key in dual.face_metadata[0]

    def test_build_with_river_populates_flow_field(self):
        dual, n, is_ocean = self._small_mesh()             # rugged (random) elevation
        f = int(np.argmax(~is_ocean))                        # any land face
        surf = _run_cost(dual, rivers=[self._tangent_river(dual, f, 3000.0)])
        assert surf.river_flow_mag[f] == pytest.approx(3000.0)
        assert np.linalg.norm(surf.river_flow_dir[f]) == pytest.approx(1.0, abs=1e-4)
        # Rugged terrain → past the fall line → not navigable, land-only.
        assert surf.river_navigability[f] == 0.0
        assert not surf.sea_passable[f]

    def test_lake_face_is_sailable_water_not_land(self):
        dual, n, is_ocean = self._flat_mesh()
        f = int(np.argmax(~is_ocean))                        # a land face
        lake = np.zeros((n, 4, 4, 1), np.float32)
        lake[f] = 1.0                                        # flood it
        dual.set_face_textures("lake_mask", lake, n_channels=1)
        surf = _run_cost(dual)
        assert not surf.land_passable[f]                     # impassable overland
        assert surf.sea_passable[f]                          # sailable like ocean
        assert bool(dual.face_metadata[f]["is_water"])

    def test_navigable_river_joins_sea_network(self):
        dual, n, is_ocean = self._flat_mesh()                # flat → fully navigable
        f = int(np.argmax(~is_ocean))
        surf = _run_cost(dual, rivers=[self._tangent_river(dual, f, 3000.0)])
        assert surf.river_navigability[f] == pytest.approx(1.0)
        assert surf.sea_passable[f]                          # amphibious: joins the sea
        assert surf.land_passable[f]                         # …but still walkable
        assert surf.sea_base_cost[f] > 0.0
        assert bool(dual.face_metadata[f]["is_navigable_river"])

    def test_low_flow_river_is_not_navigable(self):
        dual, n, is_ocean = self._flat_mesh()
        f = int(np.argmax(~is_ocean))
        surf = _run_cost(dual, rivers=[self._tangent_river(dual, f, 100.0)])
        assert surf.river_navigability[f] == pytest.approx(1.0)  # flat…
        assert not surf.sea_passable[f]                          # …but too little flow
