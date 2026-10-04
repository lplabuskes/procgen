"""Sediment trapping: lakes keep the silt, so rivers leaving them build smaller deltas."""
from __future__ import annotations

import numpy as np

from worldgen.delta.builder import DeltaParams
from worldgen.delta.sediment import mouth_node, sediment_load
from worldgen.delta.test.test_lake import FLOW, _run
from worldgen.hydrology.types import StreamNetwork


def _network() -> tuple[StreamNetwork, np.ndarray]:
    """Headwater 0 → lake 1 → lake 2 → outlet 3 → 4 (sea), with a fresh tributary 5 → 3.

    Own runoff: 100 at the headwater, 10 per lake node, 20 at the outlet, 50 on
    the tributary.  Nodes sit on flat texels 10…15.
    """
    flow = np.array([100, 110, 120, 190, 190, 50], dtype=np.float32)
    receiver = np.array([1, 2, 3, 4, -1, 3], dtype=np.int64)
    nodes = np.arange(10, 16, dtype=np.int64)
    lake = np.zeros(6, dtype=np.int64)     # per node: lake 1 on nodes 1 and 2
    lake[[1, 2]] = 1
    net = StreamNetwork(
        nodes=nodes, positions=np.eye(3, dtype=np.float32)[np.arange(6) % 3], receiver=receiver, flow=flow,
        channel=np.array([1, 1, 1, 1, 0, 1], dtype=bool), stop=np.array([0, 0, 0, 0, 1, 0], dtype=bool),
        resolution=16,
    )
    return net, lake


def test_without_trapping_the_load_is_the_flow():
    net, lake = _network()
    assert np.allclose(sediment_load(net, lake, 0.0), net.flow)


def test_a_lake_outlet_passes_only_the_untrapped_share():
    net, lake = _network()
    load = sediment_load(net, lake, 0.9)
    # Into and through the lake the load is whole; past its outlet a tenth goes
    # on, plus the outlet's own runoff and the fresh tributary.
    assert np.allclose(load[:3], [100, 110, 120])
    assert np.isclose(load[3], 0.1 * 120 + 20 + 50)
    assert np.isclose(load[4], load[3])
    assert np.all(load <= net.flow + 1e-6)


def test_the_mouth_node_is_the_biggest_on_the_mouth_texel():
    net, _ = _network()
    assert mouth_node(net, 14, np.zeros(3), 0.0) == 4
    # Off the network's texels: the biggest node near the point.
    assert mouth_node(net, 99, np.array([1.0, 0.0, 0.0]), 0.1) == 3
    assert mouth_node(net, 99, np.array([0.0, 0.0, -1.0]), 0.1) is None


def test_a_river_leaving_a_lake_builds_a_smaller_delta(mesh, index):
    # The test river gathers all its water above the lake, so past it only the
    # untrapped share carries silt to the sea.
    big = 20 * FLOW
    clear, *_ = _run(mesh, index, DeltaParams(lake_trap_efficiency=0.0), streams=True, flow=big)
    trapped, *_ = _run(mesh, index, DeltaParams(lake_trap_efficiency=0.9), streams=True, flow=big)
    sea = {name: next(f for f in r.fans if not f.lake) for name, r in (("clear", clear), ("trapped", trapped))}
    lake = {name: next(f for f in r.fans if f.lake) for name, r in (("clear", clear), ("trapped", trapped))}
    assert np.isclose(sea["clear"].sediment, 1.0) and np.isclose(sea["trapped"].sediment, 0.1, atol=0.02)
    assert sea["trapped"].area_km2 < 0.5 * sea["clear"].area_km2
    assert sea["trapped"].flow == sea["clear"].flow            # the water is the same
    # The lake delta gets the full load arriving at the lake either way.
    assert np.isclose(lake["trapped"].sediment, 1.0) and np.isclose(lake["trapped"].area_km2, lake["clear"].area_km2)


def test_too_little_silt_left_means_no_delta(mesh, index):
    # 10% of FLOW is below the threshold: the lake keeps its delta, the sea gets none.
    result, *_ = _run(mesh, index, DeltaParams(lake_trap_efficiency=0.9), streams=True)
    assert [f.lake for f in result.fans] == [True]


def test_a_full_lake_spills_what_it_cannot_hold():
    net, lake = _network()
    load = sediment_load(net, lake, 0.9, spill={1: 50.0})
    # The spilled 50 passes whole; the other 70 reaching the outlet is trapped as usual.
    assert np.isclose(load[3], 50.0 + 0.1 * 70.0 + 20 + 50)
    # Never more than arrives: a spill beyond the load passes just the load.
    assert np.isclose(sediment_load(net, lake, 0.9, spill={1: 1e9})[3], 120 + 20 + 50)
    # Another lake's spill changes nothing here.
    assert np.allclose(sediment_load(net, lake, 0.9, spill={2: 50.0}), sediment_load(net, lake, 0.9))


def test_capping_a_lake_delta_lets_its_silt_through(mesh, index):
    # The same river, the same trapping; only how much of the lake its delta may fill differs.
    big = 20 * FLOW
    sea = {}
    for share in (0.25, 0.005, 0.0):
        params = DeltaParams(lake_trap_efficiency=0.9, max_lake_share=share)
        result, *_ = _run(mesh, index, params, streams=True, flow=big)
        sea[share] = next(f for f in result.fans if not f.lake)
    # Room to spare: trapped as usual.  Cut short by the cap: part of its silt
    # passes on.  No room at all: all of it does.
    assert np.isclose(sea[0.25].sediment, 0.1, atol=0.02)
    assert sea[0.25].sediment + 0.05 < sea[0.005].sediment < sea[0.0].sediment
    assert np.isclose(sea[0.0].sediment, 1.0, atol=0.02)
    assert sea[0.0].area_km2 > sea[0.005].area_km2 > sea[0.25].area_km2
