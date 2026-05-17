from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

from worldgen.geology.elevation import (
    compute_elevation_delta_textures,
    compute_face_elevations,
    compute_noise_elevation_deltas,
)
from worldgen.geology.volcanism import compute_volcanism
from worldgen.geology.tectonics import (
    assign_plate_types,
    classify_plate_boundaries,
    compute_face_velocities,
    generate_plate_velocities,
    generate_plates,
    relax_plates,
)
from worldgen.mesh import cache as mesh_cache
from worldgen.mesh.mymesh import PolygonMesh


@dataclass
class ElevationParams:
    # Mesh generation
    icosphere_depth: int = 5
    n_swaps: int = 4000
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
    elevation_decay_rate: float = 0.25
    elevation_max_hops: int = 16
    elevation_tex_size: int = 16
    # Elevation noise
    noise_amplitude: float = 4.0
    noise_coast_sigma: float = 1.5  # km; controls how far noise extends from sea level
    # Volcanism
    n_hotspots: int = 8
    volcanic_decay_rate: float = 0.3
    volcanic_max_hops: int = 12
    volcanic_elevation_scale: float = 2.0           # km added at full activity (continental)
    volcanic_elevation_scale_oceanic: float = 5.5   # km added at full activity (oceanic)
    # Reproducibility — None means random each run
    seed: Optional[int] = None


class ElevationBuilder:
    """Orchestrates the full elevation/tectonics pipeline.

    Call build() to run every stage. Intermediate results are stored as
    attributes so callers (e.g. visualization code) can read them without
    re-running any work.
    """

    def __init__(self, params: ElevationParams | None = None) -> None:
        self.params: ElevationParams = params or ElevationParams()
        self.mesh: Optional[PolygonMesh] = None
        self.plate_ids: Optional[List[int]] = None
        self.plate_omegas: Optional[np.ndarray] = None
        self.face_velocities: Optional[np.ndarray] = None
        self.boundary_types: Optional[List[str]] = None
        self.boundary_data: Optional[List[dict]] = None
        self.elevations: Optional[np.ndarray] = None
        self.volcanism: Optional[np.ndarray] = None
        self.max_speed: float = 1.0

    def _sub_seed(self, offset: int) -> Optional[int]:
        return None if self.params.seed is None else self.params.seed + offset

    def build(self, progress: Optional[Callable[[str], None]] = None) -> PolygonMesh:
        """Run the full pipeline and return the dual mesh with all metadata applied."""
        def _report(msg: str) -> None:
            if progress is not None:
                progress(msg)

        p = self.params

        if not mesh_cache.is_populated():
            _report("Building mesh cache (first run only)…")
            mesh_cache.build(
                icosphere_depth=p.icosphere_depth,
                n_swaps=p.n_swaps,
                n_relax=p.n_relax,
                stiffness=p.stiffness,
                outer_iters=p.outer_iters,
                progress=_report,
            )
        _report("Loading mesh…")
        dual = mesh_cache.load_random(seed=self._sub_seed(5))

        _report("Generating tectonic plates…")
        plate_ids = generate_plates(
            dual,
            n_plates=p.n_plates,
            seed=self._sub_seed(0),
            compactness=p.plate_compactness,
        )
        _report("Relaxing plate boundaries…")
        plate_ids = relax_plates(dual, plate_ids, max_passes=p.relax_max_passes)

        _report("Assigning plate types and velocities…")
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
        _report("Classifying plate boundaries…")
        boundary_data = classify_plate_boundaries(dual, plate_ids, plate_omegas)
        boundary_types = [d["boundary_type"] for d in boundary_data]

        speeds = np.linalg.norm(face_velocities, axis=1)
        dual.apply_metadata_channel("plate_id", plate_ids)
        dual.apply_metadata_channel("boundary_type", boundary_types)
        dual.apply_metadata_channel("surface_velocity", [tuple(v) for v in face_velocities])
        dual.apply_metadata_channel("surface_speed", speeds.tolist())

        _report("Computing elevations…")
        elevations = compute_face_elevations(
            dual,
            plate_types,
            boundary_data,
            decay_rate=p.elevation_decay_rate,
            max_hops=p.elevation_max_hops,
            rate_norm=p.elevation_rate_norm,
        )

        _report("Computing volcanism…")
        volcanism = compute_volcanism(
            dual,
            plate_types,
            boundary_data,
            n_hotspots=p.n_hotspots,
            decay_rate=p.volcanic_decay_rate,
            max_hops=p.volcanic_max_hops,
            rate_norm=p.elevation_rate_norm,
            seed=self._sub_seed(4),
        )
        oceanic_mask = np.array([t == "oceanic" for t in plate_types], dtype=np.float64)
        elev_scale = (
            oceanic_mask * p.volcanic_elevation_scale_oceanic
            + (1.0 - oceanic_mask) * p.volcanic_elevation_scale
        )
        elevations += volcanism * elev_scale
        dual.apply_metadata_channel("volcanic_activity", volcanism.tolist())
        dual.apply_metadata_channel("elevation", elevations.tolist())

        _report("Generating elevation textures…")
        elevation_delta_textures = compute_elevation_delta_textures(
            dual, elevations, tex_size=p.elevation_tex_size
        )
        _report("Applying elevation noise…")
        noise_seed = self._sub_seed(3)
        noise_deltas = compute_noise_elevation_deltas(
            dual,
            tex_size=p.elevation_tex_size,
            amplitude=p.noise_amplitude,
            seed=noise_seed or 0,
            face_elevations=elevations,
            elevation_deltas=elevation_delta_textures[:, :, :, 0],
            coast_sigma=p.noise_coast_sigma,
        )
        elevation_delta_textures[:, :, :, 0] += noise_deltas[:, :, :, 0]
        dual.set_face_textures("elevation_delta", elevation_delta_textures)

        self.mesh = dual
        self.plate_ids = plate_ids
        self.plate_omegas = plate_omegas
        self.face_velocities = face_velocities
        self.boundary_types = boundary_types
        self.boundary_data = boundary_data
        self.elevations = elevations
        self.volcanism = volcanism
        self.max_speed = float(speeds.max()) or 1.0

        _report("Done.")
        return dual
