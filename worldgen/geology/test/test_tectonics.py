import numpy as np
import pytest

from core.mesh import Mesh
from worldgen.geology.tectonics import (
    _tarjan_aps_iterative,
    assign_plate_types,
    generate_plate_velocities,
    generate_plates,
    relax_plates,
)


def _connected_components(faces: set, adjacency: list) -> list:
    """Return list of connected components (each a set of face indices)."""
    components = []
    unvisited = set(faces)
    while unvisited:
        start = next(iter(unvisited))
        comp: set = {start}
        stack = [start]
        while stack:
            node = stack.pop()
            for nb in adjacency[node]:
                if nb in unvisited and nb not in comp:
                    comp.add(nb)
                    stack.append(nb)
        unvisited -= comp
        components.append(comp)
    return components


def _plate_faces(plate_ids: list) -> dict:
    result: dict = {}
    for i, pid in enumerate(plate_ids):
        result.setdefault(pid, set()).add(i)
    return result


def _assert_no_disjoint_plates(plate_ids: list, adjacency: list, label: str = "") -> None:
    for pid, faces in _plate_faces(plate_ids).items():
        comps = _connected_components(faces, adjacency)
        assert len(comps) == 1, (
            f"{label}plate {pid} has {len(comps)} disjoint components "
            f"(sizes: {sorted(len(c) for c in comps)})"
        )


@pytest.fixture(scope="module")
def dual_d2():
    mesh = Mesh.icosphere(2)
    mesh.deregularize(n_swaps=40, n_relax=3, stiffness=0.5, outer_iters=2)
    return mesh.mesh_dual()


# ---------------------------------------------------------------------------
# _tarjan_aps_iterative
# ---------------------------------------------------------------------------

class TestTarjanAPs:
    def _adj(self, n: int, edges: list) -> list:
        adj: list = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def test_triangle_no_aps(self):
        adj = self._adj(3, [(0, 1), (1, 2), (0, 2)])
        assert _tarjan_aps_iterative({0, 1, 2}, adj) == set()

    def test_path_middle_is_ap(self):
        # 0 - 1 - 2: node 1 is the only AP
        adj = self._adj(3, [(0, 1), (1, 2)])
        assert _tarjan_aps_iterative({0, 1, 2}, adj) == {1}

    def test_star_center_is_ap(self):
        adj = self._adj(4, [(0, 1), (0, 2), (0, 3)])
        assert _tarjan_aps_iterative({0, 1, 2, 3}, adj) == {0}

    def test_chain_internal_nodes_are_aps(self):
        # 0 - 1 - 2 - 3: nodes 1 and 2 are APs
        adj = self._adj(4, [(0, 1), (1, 2), (2, 3)])
        assert _tarjan_aps_iterative({0, 1, 2, 3}, adj) == {1, 2}

    def test_single_node_no_aps(self):
        assert _tarjan_aps_iterative({0}, [[]]) == set()

    def test_two_nodes_no_aps(self):
        assert _tarjan_aps_iterative({0, 1}, [[1], [0]]) == set()

    def test_subgraph_restriction(self):
        # Path 0-1-2-3 restricted to {0, 1, 2}: node 1 is still an AP.
        adj = self._adj(4, [(0, 1), (1, 2), (2, 3)])
        aps = _tarjan_aps_iterative({0, 1, 2}, adj)
        assert aps == {1}


# ---------------------------------------------------------------------------
# generate_plates
# ---------------------------------------------------------------------------

class TestGeneratePlates:
    def test_all_faces_assigned(self, dual_d2):
        plate_ids = generate_plates(dual_d2, n_plates=8, seed=42)
        assert len(plate_ids) == len(dual_d2.faces)
        assert all(p >= 0 for p in plate_ids)

    def test_plate_ids_in_range(self, dual_d2):
        n_plates = 8
        plate_ids = generate_plates(dual_d2, n_plates=n_plates, seed=42)
        assert all(0 <= p < n_plates for p in plate_ids)

    def test_no_disjoint_plates(self, dual_d2):
        plate_ids = generate_plates(dual_d2, n_plates=10, seed=0)
        _assert_no_disjoint_plates(plate_ids, dual_d2.face_adjacency)

    def test_no_disjoint_plates_many_seeds(self, dual_d2):
        for seed in range(20):
            plate_ids = generate_plates(dual_d2, n_plates=12, seed=seed)
            _assert_no_disjoint_plates(
                plate_ids, dual_d2.face_adjacency, label=f"seed={seed}: "
            )

    def test_single_plate_all_same(self, dual_d2):
        plate_ids = generate_plates(dual_d2, n_plates=1, seed=0)
        assert len(set(plate_ids)) == 1

    def test_deterministic_with_seed(self, dual_d2):
        a = generate_plates(dual_d2, n_plates=8, seed=99)
        b = generate_plates(dual_d2, n_plates=8, seed=99)
        assert a == b

    def test_different_seeds_differ(self, dual_d2):
        a = generate_plates(dual_d2, n_plates=8, seed=1)
        b = generate_plates(dual_d2, n_plates=8, seed=2)
        assert a != b

    def test_many_plates_no_disjoint(self, dual_d2):
        # Push to a high plate count where the flood fill is crowded.
        n_faces = len(dual_d2.faces)
        plate_ids = generate_plates(dual_d2, n_plates=n_faces // 4, seed=7)
        _assert_no_disjoint_plates(plate_ids, dual_d2.face_adjacency)


# ---------------------------------------------------------------------------
# relax_plates
# ---------------------------------------------------------------------------

class TestRelaxPlates:
    def test_all_faces_remain_assigned(self, dual_d2):
        plate_ids = generate_plates(dual_d2, n_plates=10, seed=42)
        relaxed = relax_plates(dual_d2, plate_ids, max_passes=5)
        assert len(relaxed) == len(plate_ids)
        assert all(p >= 0 for p in relaxed)

    def test_no_disjoint_plates_after_relax(self, dual_d2):
        plate_ids = generate_plates(dual_d2, n_plates=10, seed=7)
        relaxed = relax_plates(dual_d2, plate_ids, max_passes=10)
        _assert_no_disjoint_plates(relaxed, dual_d2.face_adjacency)

    def test_no_disjoint_plates_many_seeds(self, dual_d2):
        for seed in range(15):
            plate_ids = generate_plates(dual_d2, n_plates=12, seed=seed)
            relaxed = relax_plates(dual_d2, plate_ids, max_passes=10)
            _assert_no_disjoint_plates(
                relaxed, dual_d2.face_adjacency, label=f"seed={seed}: "
            )

    def test_idempotent(self, dual_d2):
        plate_ids = generate_plates(dual_d2, n_plates=8, seed=5)
        once = relax_plates(dual_d2, plate_ids, max_passes=10)
        twice = relax_plates(dual_d2, once[:], max_passes=10)
        assert once == twice

    def test_plate_count_does_not_increase(self, dual_d2):
        plate_ids = generate_plates(dual_d2, n_plates=10, seed=3)
        relaxed = relax_plates(dual_d2, plate_ids, max_passes=10)
        assert len(set(relaxed)) <= len(set(plate_ids))


# ---------------------------------------------------------------------------
# assign_plate_types
# ---------------------------------------------------------------------------

class TestAssignPlateTypes:
    def test_length_matches_n_plates(self):
        assert len(assign_plate_types(10, seed=0)) == 10

    def test_valid_values(self):
        types = assign_plate_types(200, seed=0)
        assert all(t in ("oceanic", "continental") for t in types)

    def test_oceanic_fraction(self):
        n = 10_000
        types = assign_plate_types(n, oceanic_fraction=0.7, seed=42)
        ratio = sum(1 for t in types if t == "oceanic") / n
        assert abs(ratio - 0.7) < 0.02

    def test_deterministic(self):
        assert assign_plate_types(20, seed=7) == assign_plate_types(20, seed=7)

    def test_all_oceanic(self):
        types = assign_plate_types(10, oceanic_fraction=1.0, seed=0)
        assert all(t == "oceanic" for t in types)

    def test_all_continental(self):
        types = assign_plate_types(10, oceanic_fraction=0.0, seed=0)
        assert all(t == "continental" for t in types)


# ---------------------------------------------------------------------------
# generate_plate_velocities — NNR correction
# ---------------------------------------------------------------------------

class TestGeneratePlateVelocities:
    N_PLATES = 6
    _DENSITY = {"oceanic": 3.0, "continental": 2.7}

    @pytest.fixture
    def dual(self):
        # Function-scoped so each test starts with clean face metadata.
        mesh = Mesh.icosphere(2)
        mesh.deregularize(n_swaps=40, n_relax=3, stiffness=0.5, outer_iters=2)
        return mesh.mesh_dual()

    def test_output_shape(self, dual):
        plate_ids = generate_plates(dual, n_plates=self.N_PLATES, seed=0)
        omegas = generate_plate_velocities(dual, plate_ids, self.N_PLATES, seed=0)
        assert omegas.shape == (self.N_PLATES, 3)

    def test_nnr_without_metadata(self, dual):
        """Area-weighted sum of omegas is zero when no plate_type metadata present."""
        plate_ids = generate_plates(dual, n_plates=self.N_PLATES, seed=1)
        omegas = generate_plate_velocities(dual, plate_ids, self.N_PLATES, seed=1)
        ids = np.array(plate_ids, dtype=np.int32)
        weighted_sum = (dual.face_areas[:, np.newaxis] * omegas[ids]).sum(axis=0)
        np.testing.assert_allclose(weighted_sum, 0.0, atol=1e-12)

    def test_nnr_with_density_weighting(self, dual):
        """(area × density)-weighted sum of omegas is zero when plate_type is set."""
        plate_ids = generate_plates(dual, n_plates=self.N_PLATES, seed=2)
        plate_types_per_plate = assign_plate_types(self.N_PLATES, seed=2)
        plate_types = [plate_types_per_plate[p] for p in plate_ids]
        dual.apply_metadata_channel("plate_type", plate_types)

        omegas = generate_plate_velocities(dual, plate_ids, self.N_PLATES, seed=2)
        ids = np.array(plate_ids, dtype=np.int32)
        densities = np.array([self._DENSITY[t] for t in plate_types], dtype=np.float64)
        weighted_sum = (dual.face_areas[:, np.newaxis] * densities[:, np.newaxis] * omegas[ids]).sum(axis=0)
        np.testing.assert_allclose(weighted_sum, 0.0, atol=1e-12)

    def test_density_weighting_changes_result(self, dual):
        """Density-weighted NNR produces different omegas than area-only NNR."""
        plate_ids = generate_plates(dual, n_plates=self.N_PLATES, seed=3)
        # Guaranteed alternating mix so the two corrections differ.
        plate_types_per_plate = ["oceanic", "continental"] * (self.N_PLATES // 2)
        plate_types = [plate_types_per_plate[p] for p in plate_ids]

        omegas_uniform = generate_plate_velocities(dual, plate_ids, self.N_PLATES, seed=3)

        dual.apply_metadata_channel("plate_type", plate_types)
        omegas_density = generate_plate_velocities(dual, plate_ids, self.N_PLATES, seed=3)

        assert not np.allclose(omegas_uniform, omegas_density)

    def test_deterministic_with_seed(self, dual):
        plate_ids = generate_plates(dual, n_plates=self.N_PLATES, seed=4)
        a = generate_plate_velocities(dual, plate_ids, self.N_PLATES, seed=4)
        b = generate_plate_velocities(dual, plate_ids, self.N_PLATES, seed=4)
        np.testing.assert_array_equal(a, b)
