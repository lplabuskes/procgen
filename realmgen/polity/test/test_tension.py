"""Unit tests for realmgen.polity.tension — border-crossing tension."""
from __future__ import annotations

import numpy as np

from realmgen.polity.tension import compute_border_tension


def _line_adjacency(n: int):
    """Adjacency list + uniform-cost edge arrays for an n-face line graph."""
    adj, costs = [], []
    for i in range(n):
        neigh = []
        if i > 0:
            neigh.append(i - 1)
        if i < n - 1:
            neigh.append(i + 1)
        adj.append(neigh)
        costs.append(np.ones(len(neigh), dtype=np.float64))
    return adj, costs


class TestComputeBorderTension:
    def test_no_borders_when_single_polity(self):
        adj, costs = _line_adjacency(4)
        pair, face_t = compute_border_tension(
            adj, np.zeros(4, dtype=np.int32), np.array([1.0]), costs, np.zeros(4),
            barrier_scale=1.0, flow_ref=1.0,
        )
        assert pair == {}
        assert (face_t == 0.0).all()

    def test_single_border_pair_detected(self):
        adj, costs = _line_adjacency(4)
        polity_of_face = np.array([0, 0, 1, 1], dtype=np.int32)
        flow = np.array([0.0, 1.0, 1.0, 0.0])
        pair, face_t = compute_border_tension(
            adj, polity_of_face, np.array([1.0, 1.0]), costs, flow,
            barrier_scale=1.0, flow_ref=1.0,
        )
        assert set(pair.keys()) == {(0, 1)}
        assert pair[(0, 1)] > 0.0
        assert face_t[1] > 0.0 and face_t[2] > 0.0
        assert face_t[0] == 0.0 and face_t[3] == 0.0

    def test_strong_barrier_lowers_tension(self):
        adj, costs_low = _line_adjacency(4)
        _, costs_high = _line_adjacency(4)
        for c in costs_high:
            c[:] = 10.0
        polity_of_face = np.array([0, 0, 1, 1], dtype=np.int32)
        flow = np.ones(4)
        power = np.array([1.0, 1.0])
        low, _ = compute_border_tension(adj, polity_of_face, power, costs_low, flow, barrier_scale=1.0, flow_ref=1.0)
        high, _ = compute_border_tension(adj, polity_of_face, power, costs_high, flow, barrier_scale=1.0, flow_ref=1.0)
        assert high[(0, 1)] < low[(0, 1)]

    def test_power_imbalance_lowers_tension(self):
        adj, costs = _line_adjacency(4)
        polity_of_face = np.array([0, 0, 1, 1], dtype=np.int32)
        flow = np.ones(4)
        balanced, _ = compute_border_tension(adj, polity_of_face, np.array([1.0, 1.0]), costs, flow, barrier_scale=1.0, flow_ref=1.0)
        lopsided, _ = compute_border_tension(adj, polity_of_face, np.array([100.0, 1.0]), costs, flow, barrier_scale=1.0, flow_ref=1.0)
        assert lopsided[(0, 1)] < balanced[(0, 1)]

    def test_unclaimed_faces_create_no_border(self):
        adj, costs = _line_adjacency(4)
        polity_of_face = np.array([0, 0, -1, -1], dtype=np.int32)
        pair, _ = compute_border_tension(
            adj, polity_of_face, np.array([1.0]), costs, np.ones(4), barrier_scale=1.0, flow_ref=1.0,
        )
        assert pair == {}
