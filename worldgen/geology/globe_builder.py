from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from geology.elevation import compute_elevation_delta_textures, compute_face_elevations
from geology.tectonics import (
    assign_plate_types,
    classify_plate_boundaries,
    compute_face_velocities,
    compute_plate_centroids,
    generate_plate_velocities,
    generate_plates,
    relax_plates,
)
from mesh.mymesh import Mesh, PolygonMesh


@dataclass
class GlobeParams:
    # Mesh generation
    icosphere_depth: int = 4
    n_swaps: int = 1000
    n_relax: int = 10
    stiffness: float = 0.1
    outer_iters: int = 3
    # Plate layout
    n_plates: int = 12
    plate_compactness: float = 0.9
    relax_max_passes: int = 10
    oceanic_fraction: float = 0.7
    # Plate motion
    drift_scale: float = 1.0
    spin_scale: float = 0.1
    # Elevation
    elevation_rate_norm: float = 1.75
    elevation_decay_rate: float = 0.5
    elevation_max_hops: int = 8
    # Reproducibility — None means random each run
    seed: Optional[int] = None


class GlobeBuilder:
    """Orchestrates the full globe-generation pipeline.

    Call build() to run every stage. Intermediate results are stored as
    attributes so callers (e.g. visualization code) can read them without
    re-running any work.
    """

    def __init__(self, params: GlobeParams | None = None) -> None:
        self.params: GlobeParams = params or GlobeParams()
        self.mesh: Optional[PolygonMesh] = None
        self.plate_ids: Optional[List[int]] = None
        self.plate_omegas: Optional[np.ndarray] = None
        self.face_velocities: Optional[np.ndarray] = None
        self.boundary_types: Optional[List[str]] = None
        self.boundary_data: Optional[List[dict]] = None
        self.elevations: Optional[np.ndarray] = None
        self.max_speed: float = 1.0

    def _sub_seed(self, offset: int) -> Optional[int]:
        return None if self.params.seed is None else self.params.seed + offset

    def build(self) -> PolygonMesh:
        """Run the full pipeline and return the dual mesh with all metadata applied."""
        p = self.params

        tri = Mesh.icosphere(p.icosphere_depth).deregularize(
            n_swaps=p.n_swaps,
            n_relax=p.n_relax,
            stiffness=p.stiffness,
            outer_iters=p.outer_iters,
        )
        dual = tri.mesh_dual()

        plate_ids = generate_plates(
            dual,
            n_plates=p.n_plates,
            seed=self._sub_seed(0),
            compactness=p.plate_compactness,
        )
        plate_ids = relax_plates(dual, plate_ids, max_passes=p.relax_max_passes)

        plate_types_by_plate = assign_plate_types(
            p.n_plates, oceanic_fraction=p.oceanic_fraction, seed=self._sub_seed(1)
        )
        plate_types = [plate_types_by_plate[pid] for pid in plate_ids]
        dual.apply_metadata_channel("plate_type", plate_types)

        plate_omegas = generate_plate_velocities(
            dual,
            plate_ids,
            p.n_plates,
            drift_scale=p.drift_scale,
            spin_scale=p.spin_scale,
            seed=self._sub_seed(2),
        )
        face_velocities = compute_face_velocities(dual, plate_ids, plate_omegas)
        boundary_data = classify_plate_boundaries(dual, plate_ids, plate_omegas)
        boundary_types = [d["boundary_type"] for d in boundary_data]

        speeds = np.linalg.norm(face_velocities, axis=1)
        dual.apply_metadata_channel("plate_id", plate_ids)
        dual.apply_metadata_channel("boundary_type", boundary_types)
        dual.apply_metadata_channel("surface_velocity", [tuple(v) for v in face_velocities])
        dual.apply_metadata_channel("surface_speed", speeds.tolist())

        elevations = compute_face_elevations(
            dual,
            plate_types,
            boundary_data,
            decay_rate=p.elevation_decay_rate,
            max_hops=p.elevation_max_hops,
            rate_norm=p.elevation_rate_norm,
        )
        dual.apply_metadata_channel("elevation", elevations.tolist())

        elevation_delta_textures = compute_elevation_delta_textures(dual, elevations)
        dual.set_face_textures("elevation_delta", elevation_delta_textures)

        self.mesh = dual
        self.plate_ids = plate_ids
        self.plate_omegas = plate_omegas
        self.face_velocities = face_velocities
        self.boundary_types = boundary_types
        self.boundary_data = boundary_data
        self.elevations = elevations
        self.max_speed = float(speeds.max()) or 1.0

        return dual
