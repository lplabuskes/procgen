"""Tests for river flow routing and path extraction.

Covers three categories:
- compute_river_flow: structural validity of the flow field.
- extract_river_paths: path well-formedness and invariants.
- Land masking: rivers should not extend into ocean faces (elevation < 0).
"""
from __future__ import annotations

import numpy as np
from worldgen.geology.rivers import compute_river_flow, extract_river_paths
from worldgen.mesh.mymesh import Mesh


def _dual(depth: int = 2) -> Mesh:
    return Mesh.icosphere(depth).mesh_dual()


def _gradient_elevations(mesh) -> np.ndarray:
    """z-coordinate of each face centroid: north hemisphere > 0, south < 0."""
    return mesh.face_centroids[:, 2].copy()


class TestFlowRouting:
    def test_output_shapes_and_dtypes(self):
        mesh = _dual()
        N = len(mesh.faces)
        flow, flow_to = compute_river_flow(mesh, _gradient_elevations(mesh))
        assert flow.shape == (N,)
        assert flow_to.shape == (N,)
        assert flow_to.dtype == np.int32

    def test_flow_accumulation_at_least_one(self):
        mesh = _dual()
        flow, _ = compute_river_flow(mesh, _gradient_elevations(mesh))
        assert np.all(flow >= 1.0), "every face should accumulate at least its own unit of flow"

    def test_flow_to_targets_are_adjacent_or_sink(self):
        """flow_to[i] must be an edge-adjacent neighbour of i, or -1 (sink)."""
        mesh = _dual()
        _, flow_to = compute_river_flow(mesh, _gradient_elevations(mesh))
        adjacency = mesh.face_adjacency
        for i, dst in enumerate(flow_to):
            dst = int(dst)
            if dst >= 0:
                assert dst in adjacency[i], (
                    f"face {i} routes to non-adjacent face {dst}"
                )

    def test_flow_to_routes_downhill(self):
        """Each non-sink face must route to a neighbour with lower or equal elevation."""
        mesh = _dual()
        elevations = _gradient_elevations(mesh)
        _, flow_to = compute_river_flow(mesh, elevations)
        for i, dst in enumerate(flow_to):
            dst = int(dst)
            if dst >= 0:
                assert elevations[dst] <= elevations[i] + 1e-9, (
                    f"face {i} (elev={elevations[i]:.3f}) routes uphill to "
                    f"face {dst} (elev={elevations[dst]:.3f})"
                )

    def test_downstream_face_receives_upstream_flow(self):
        """For every non-sink face i, flow[flow_to[i]] must be >= flow[i].

        Each face passes its accumulated flow to its downstream neighbour, so the
        downstream face can never have *less* flow than any of its upstream faces.
        """
        mesh = _dual()
        elevations = _gradient_elevations(mesh)
        flow, flow_to = compute_river_flow(mesh, elevations)
        for i, dst in enumerate(flow_to):
            dst = int(dst)
            if dst >= 0:
                assert flow[dst] >= flow[i], (
                    f"downstream face {dst} (flow={flow[dst]:.1f}) has less flow "
                    f"than upstream face {i} (flow={flow[i]:.1f})"
                )


class TestRiverPaths:
    _THRESHOLD = 5.0

    def _paths(self, mesh=None, elevations=None):
        if mesh is None:
            mesh = _dual()
        if elevations is None:
            elevations = _gradient_elevations(mesh)
        flow, flow_to = compute_river_flow(mesh, elevations)
        paths = extract_river_paths(flow, flow_to, threshold=self._THRESHOLD, elevations=elevations)
        return mesh, elevations, flow, flow_to, paths

    def test_face_indices_in_range(self):
        mesh, _, _, _, paths = self._paths()
        N = len(mesh.faces)
        for path in paths:
            for idx in path:
                assert 0 <= idx < N, f"face index {idx} out of range [0, {N})"

    def test_minimum_path_length(self):
        _, _, _, _, paths = self._paths()
        for i, path in enumerate(paths):
            assert len(path) >= 2, f"path {i} has length {len(path)}, expected >= 2"

    def test_all_path_faces_meet_threshold(self):
        _, _, flow, _, paths = self._paths()
        for path in paths:
            for idx in path:
                assert flow[idx] >= self._THRESHOLD, (
                    f"face {idx} is in a path but has flow {flow[idx]:.1f} < "
                    f"threshold {self._THRESHOLD}"
                )

    def test_paths_follow_flow_to(self):
        """Every consecutive pair in a path must match flow_to."""
        _, _, _, flow_to, paths = self._paths()
        for path in paths:
            for j in range(len(path) - 1):
                expected = int(flow_to[path[j]])
                actual = path[j + 1]
                assert expected == actual, (
                    f"path step {path[j]} → {actual} does not match "
                    f"flow_to[{path[j]}]={expected}"
                )

    def test_sources_have_no_upstream_river_face(self):
        """A source face must not be the flow_to target of any other river face."""
        _, _, flow, flow_to, paths = self._paths()
        if not paths:
            return
        river_set = set(int(i) for i in np.where(flow >= self._THRESHOLD)[0])
        sources = {path[0] for path in paths}
        targets_of_river_faces = {
            int(flow_to[i])
            for i in river_set
            if int(flow_to[i]) in river_set
        }
        bad = sources & targets_of_river_faces
        assert not bad, (
            f"the following source faces are also downstream targets of river "
            f"faces: {bad}"
        )

    def test_paths_stay_on_land(self):
        """Rivers should not include ocean faces (elevation < 0).

        With z-coordinate elevations, the northern hemisphere is "land"
        (elevation > 0) and the southern hemisphere is "ocean" (elevation < 0).
        Rivers should stop at the coastline and not continue across the ocean floor.
        """
        mesh = _dual()
        elevations = _gradient_elevations(mesh)
        flow, flow_to = compute_river_flow(mesh, elevations)
        paths = extract_river_paths(flow, flow_to, threshold=self._THRESHOLD, elevations=elevations)

        ocean_in_paths = [
            (path_idx, step, face_idx, float(elevations[face_idx]))
            for path_idx, path in enumerate(paths)
            for step, face_idx in enumerate(path)
            if elevations[face_idx] < 0.0
        ]
        assert len(ocean_in_paths) == 0, (
            f"{len(ocean_in_paths)} ocean faces appear in river paths "
            f"(showing first 3): {ocean_in_paths[:3]}"
        )
