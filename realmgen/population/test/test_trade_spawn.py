"""Port spawning: a port town is founded where trade lands at a non-port."""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from realmgen.population.trade_spawn import spawn_trade_settlements
from realmgen.test.conftest import _settlement

N = 10
TIERS = [(5.0, 0.0, 100.0)] * 4


def _line_graph(n: int, cost: float = 1.0) -> csr_matrix:
    rows = np.arange(n - 1)
    return csr_matrix((np.full(n - 1, cost), (rows, rows + 1)), shape=(n, n))


def _spawn(landing: np.ndarray | None, **kw):
    # Flat flow; hubs (suitability 1 < 2) and waypoints off.
    flow = np.ones(N)
    centroids = np.eye(3)[np.arange(N) % 3]
    return spawn_trade_settlements(
        N, [_settlement(0)], flow, np.ones(N), np.zeros(N, dtype=bool), _line_graph(N),
        centroids, TIERS, hub_min_suitability=2.0, waypoint_min_cost=1e9,
        landing_flow=landing, **kw,
    )


def test_a_busy_landing_founds_a_port():
    landing = np.zeros(N)
    landing[7] = 5.0
    spawned = _spawn(landing)
    assert [s.face_id for s in spawned] == [7]
    assert spawned[0].tier == 2


def test_quiet_landings_found_nothing():
    landing = np.zeros(N)
    landing[7] = 0.5          # below the median face flow (1.0)
    assert _spawn(landing) == []


def test_ports_keep_their_distance():
    landing = np.zeros(N)
    landing[1] = 9.0          # too close to the settlement at 0
    landing[6] = 5.0
    landing[7] = 4.0          # too close to the new port at 6
    spawned = _spawn(landing, port_exclusion=2.0)
    assert [s.face_id for s in spawned] == [6]


def test_no_landing_flow_no_ports():
    assert _spawn(None) == []
