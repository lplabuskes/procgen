from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from worldgen.geology.elevation import (
    compute_elevation_textures,
    compute_erosion,
    compute_face_elevations,
    compute_noise_elevation_deltas,
    face_elevations_from_texture,
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


def _f(default, *, label: str, min: float, max: float, step: float | None = None,
       hidden: bool = False):
    """Convenience for declaring a leaf param field with dialog metadata."""
    meta = {"label": label, "min": min, "max": max}
    if step is not None:
        meta["step"] = step
    if hidden:
        meta["hidden"] = True
    return field(default=default, metadata=meta)


@dataclass
class MeshParams:
    icosphere_depth: int = _f(5, label="Icosphere depth", min=1, max=8)
    n_swaps: int = _f(4000, label="Edge swaps", min=0, max=10000)
    n_relax: int = _f(10, label="Relaxation steps", min=0, max=100)
    stiffness: float = _f(0.1, label="Stiffness", min=0.01, max=1.0)
    outer_iters: int = _f(3, label="Outer iterations", min=1, max=20)


@dataclass
class PlateParams:
    n_plates: int = _f(12, label="Number of plates", min=2, max=50)
    plate_compactness: float = _f(0.9, label="Compactness", min=0.0, max=1.0)
    relax_max_passes: int = _f(10, label="Relax max passes", min=0, max=50)
    oceanic_fraction: float = _f(0.7, label="Oceanic fraction", min=0.0, max=1.0)


@dataclass
class PlateMotionParams:
    drift_scale: float = _f(1.0, label="Drift scale", min=0.0, max=10.0)
    spin_scale: float = _f(0.1, label="Spin scale", min=0.0, max=10.0)


@dataclass
class ElevationFieldParams:
    elevation_rate_norm: float = _f(1.75, label="Rate normalization", min=0.0, max=10.0)
    elevation_decay_rate: float = _f(0.25, label="Decay rate", min=0.0, max=5.0)
    elevation_max_hops: int = _f(16, label="Max hops", min=1, max=30)
    elevation_tex_size: int = _f(16, label="Texture size", min=8, max=128)


@dataclass
class NoiseParams:
    noise_amplitude: float = _f(1.5, label="Amplitude (km)", min=0.0, max=20.0)
    noise_coast_sigma: float = _f(3.0, label="Coast sigma (km)", min=0.1, max=10.0)


@dataclass
class FaceErosionParams:
    """Face-level erosion (not currently exposed in dialog)."""
    hydraulic_iterations: int = _f(2, label="Hydraulic iterations", min=0, max=30, hidden=True)
    hydraulic_erosion_rate: float = _f(0.4, label="Hydraulic erosion rate", min=0.0, max=1.0, hidden=True)
    hydraulic_deposition_rate: float = _f(0.3, label="Hydraulic deposition rate", min=0.0, max=1.0, hidden=True)
    thermal_iterations: int = _f(2, label="Thermal iterations", min=0, max=30, hidden=True)
    thermal_talus: float = _f(0.5, label="Thermal talus", min=0.001, max=5.0, hidden=True)
    thermal_erosion_rate: float = _f(0.5, label="Thermal erosion rate", min=0.0, max=1.0, hidden=True)


@dataclass
class VolcanismParams:
    n_hotspots: int = _f(8, label="Hotspots", min=0, max=30)
    volcanic_decay_rate: float = _f(0.3, label="Decay rate", min=0.0, max=5.0)
    volcanic_max_hops: int = _f(12, label="Max hops", min=1, max=20)
    volcanic_elevation_scale: float = _f(2.0, label="Scale continental (km)", min=0.0, max=20.0)
    volcanic_elevation_scale_oceanic: float = _f(5.5, label="Scale oceanic (km)", min=0.0, max=20.0)


@dataclass
class ElevationParams:
    mesh: MeshParams = field(default_factory=MeshParams, metadata={"label": "Mesh Generation"})
    plates: PlateParams = field(default_factory=PlateParams, metadata={"label": "Plate Layout"})
    plate_motion: PlateMotionParams = field(default_factory=PlateMotionParams, metadata={"label": "Plate Motion"})
    elevation: ElevationFieldParams = field(default_factory=ElevationFieldParams, metadata={"label": "Elevation"})
    noise: NoiseParams = field(default_factory=NoiseParams, metadata={"label": "Elevation Noise"})
    face_erosion: FaceErosionParams = field(default_factory=FaceErosionParams, metadata={"label": "Face Erosion", "hidden": True})
    volcanism: VolcanismParams = field(default_factory=VolcanismParams, metadata={"label": "Volcanism"})
    # Reproducibility — None means random each run
    seed: Optional[int] = field(default=None, metadata={"label": "Seed", "kind": "optional_int", "min": 0, "max": 2**31 - 1})


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
        self.elevation_textures: Optional[np.ndarray] = None
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
                icosphere_depth=p.mesh.icosphere_depth,
                n_swaps=p.mesh.n_swaps,
                n_relax=p.mesh.n_relax,
                stiffness=p.mesh.stiffness,
                outer_iters=p.mesh.outer_iters,
                progress=_report,
            )
        _report("Loading mesh…")
        dual = mesh_cache.load_random(seed=self._sub_seed(5))

        _report("Generating tectonic plates…")
        plate_ids = generate_plates(
            dual,
            n_plates=p.plates.n_plates,
            seed=self._sub_seed(0),
            compactness=p.plates.plate_compactness,
        )
        _report("Relaxing plate boundaries…")
        plate_ids = relax_plates(dual, plate_ids, max_passes=p.plates.relax_max_passes)

        _report("Assigning plate types and velocities…")
        plate_types_by_plate = assign_plate_types(
            p.plates.n_plates, oceanic_fraction=p.plates.oceanic_fraction, seed=self._sub_seed(1)
        )
        plate_types = [plate_types_by_plate[pid] for pid in plate_ids]
        dual.apply_metadata_channel("plate_type", plate_types)

        plate_omegas = generate_plate_velocities(
            dual,
            plate_ids,
            p.plates.n_plates,
            drift_scale=p.plate_motion.drift_scale,
            spin_scale=p.plate_motion.spin_scale,
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
            decay_rate=p.elevation.elevation_decay_rate,
            max_hops=p.elevation.elevation_max_hops,
            rate_norm=p.elevation.elevation_rate_norm,
        )

        _report("Computing volcanism…")
        volcanism = compute_volcanism(
            dual,
            plate_types,
            boundary_data,
            n_hotspots=p.volcanism.n_hotspots,
            decay_rate=p.volcanism.volcanic_decay_rate,
            max_hops=p.volcanism.volcanic_max_hops,
            rate_norm=p.elevation.elevation_rate_norm,
            seed=self._sub_seed(4),
        )
        _report("Applying erosion…")
        elevations = compute_erosion(
            dual,
            elevations,
            hydraulic_iterations=p.face_erosion.hydraulic_iterations,
            hydraulic_erosion_rate=p.face_erosion.hydraulic_erosion_rate,
            hydraulic_deposition_rate=p.face_erosion.hydraulic_deposition_rate,
            thermal_iterations=p.face_erosion.thermal_iterations,
            thermal_talus=p.face_erosion.thermal_talus,
            thermal_erosion_rate=p.face_erosion.thermal_erosion_rate,
        )

        oceanic_mask = np.array([t == "oceanic" for t in plate_types], dtype=np.float64)
        elev_scale = (
            oceanic_mask * p.volcanism.volcanic_elevation_scale_oceanic
            + (1.0 - oceanic_mask) * p.volcanism.volcanic_elevation_scale
        )
        elevations += volcanism * elev_scale
        dual.apply_metadata_channel("volcanic_activity", volcanism.tolist())
        dual.apply_metadata_channel("elevation", elevations.tolist())

        _report("Generating elevation textures…")
        elevation_textures = compute_elevation_textures(
            dual, elevations, tex_size=p.elevation.elevation_tex_size
        )
        _report("Applying elevation noise…")
        noise_seed = self._sub_seed(3)
        noise_deltas = compute_noise_elevation_deltas(
            dual,
            tex_size=p.elevation.elevation_tex_size,
            amplitude=p.noise.noise_amplitude,
            seed=noise_seed or 0,
            per_pixel_elevations=elevation_textures[:, :, :, 0],
            coast_sigma=p.noise.noise_coast_sigma,
        )
        elevation_textures[:, :, :, 0] += noise_deltas[:, :, :, 0]

        # Derive per-face elevations from the final texture for land/ocean masking.
        final_elevations = face_elevations_from_texture(elevation_textures)

        dual.set_face_textures("elevation", elevation_textures, n_channels=1)

        self.mesh = dual
        self.plate_ids = plate_ids
        self.plate_omegas = plate_omegas
        self.face_velocities = face_velocities
        self.boundary_types = boundary_types
        self.boundary_data = boundary_data
        self.elevations = final_elevations
        self.elevation_textures = elevation_textures
        self.volcanism = volcanism
        self.max_speed = float(speeds.max()) or 1.0

        _report("Done.")
        return dual
