"""Carving Priority-Flood (``priority_flood_tree``)."""
import numpy as np

from worldgen.hydrology import routing
from worldgen.hydrology.routing import priority_flood_tree


def _grid_csr(n: int) -> tuple[np.ndarray, np.ndarray]:
    """4-connected CSR adjacency of an ``n × n`` grid."""
    indptr, flat = [0], []
    for node in range(n * n):
        r, c = divmod(node, n)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            if 0 <= r + dr < n and 0 <= c + dc < n:
                flat.append((r + dr) * n + c + dc)
        indptr.append(len(flat))
    return np.array(indptr, dtype=np.int64), np.array(flat, dtype=np.int64)


def _descend(parent: np.ndarray, node: int) -> list[int]:
    path = [node]
    while parent[path[-1]] >= 0:
        path.append(int(parent[path[-1]]))
    return path


def test_every_node_drains_to_an_outlet_through_settled_parents():
    rng = np.random.default_rng(0)
    n = 12
    elev = rng.random(n * n).astype(np.float32)
    outlets = np.zeros(n * n, dtype=bool)
    outlets[:n] = True
    indptr, flat = _grid_csr(n)
    parent, rank = priority_flood_tree(elev, outlets, indptr, flat)
    assert np.all(parent[outlets] == -1) and np.all(parent[~outlets] >= 0)
    child = np.flatnonzero(~outlets)
    assert np.all(rank[parent[child]] < rank[child])       # reversed rank is topological
    for node in child.tolist():
        assert outlets[_descend(parent, node)[-1]]


def test_a_pit_drains_along_its_lowest_route():
    """Water leaves a basin along the low channel, not straight across the
    filled flat the way an epsilon-filled surface would route it."""
    n = 9
    elev = np.full((n, n), 5.0, dtype=np.float32)
    elev[4, 1:8] = 1.0                    # the pit: a row of low ground ...
    elev[4, 0] = 0.0                      # ... spilling west through one outlet
    elev[5:8, 7] = 1.0                    # plus a low arm hanging off its east end
    outlets = np.zeros(n * n, dtype=bool)
    outlets[4 * n + 0] = True
    indptr, flat = _grid_csr(n)
    parent, _ = priority_flood_tree(elev.ravel(), outlets, indptr, flat)
    path = _descend(parent, 7 * n + 7)
    assert all(elev.ravel()[node] <= 1.0 for node in path)


def test_numba_matches_pure_python():
    rng = np.random.default_rng(1)
    n = 15
    elev = rng.random(n * n).astype(np.float32)
    elev[rng.integers(0, n * n, 20)] = 0.5    # ties exercise the index tie-break
    outlets = np.zeros(n * n, dtype=bool)
    outlets[[0, n * n - 1]] = True
    indptr, flat = _grid_csr(n)
    fast = priority_flood_tree(elev, outlets, indptr, flat)
    slow = routing._priority_flood_tree_py(elev, outlets, indptr, flat)
    np.testing.assert_array_equal(fast[0], slow[0])
    np.testing.assert_array_equal(fast[1], slow[1])
