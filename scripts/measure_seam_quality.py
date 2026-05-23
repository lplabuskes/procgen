"""Measure elevation discontinuities at face-to-face texture boundaries.

Usage:
    python3 scripts/measure_seam_quality.py                  # seeds 42, 137, 999
    python3 scripts/measure_seam_quality.py 1 2 3            # specific seeds
    python3 scripts/measure_seam_quality.py --random 5       # 5 random seeds
    python3 scripts/measure_seam_quality.py --pre-noise 42   # base texture only (no noise)

Outputs a table of mean / median / RMS / max boundary jump (km) per seed.
Useful for gauging how much elevation discontinuity exists at face seams before
and after any changes to the elevation texture pipeline.

Background
----------
Each face has its own elevation texture computed independently via barycentric
interpolation of vertex elevations. Pixels near a shared edge should agree with
their counterpart in the neighbouring face, but in practice the fan-triangulation
apex pulls each side toward a different unshared vertex, creating a small jump.
The jump is also inflated by the coast-proximity noise modulation when a face
boundary happens to straddle a coastline.

`measure_boundary_elevation_jumps` (worldgen.geology.elevation) quantifies this
by matching pixel pairs along every shared edge via their t-parameter and
reporting summary statistics over all ~350 k matched pairs.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np  # noqa: E402

from worldgen.geology.elevation import (  # noqa: E402
    compute_elevation_textures,
    compute_face_elevations,
    compute_erosion,
    measure_boundary_elevation_jumps,
)
from worldgen.geology.elevation_builder import ElevationBuilder, ElevationParams  # noqa: E402
from worldgen.geology.tectonics import (  # noqa: E402
    assign_plate_types,
    classify_plate_boundaries,
    generate_plate_velocities,
    generate_plates,
    relax_plates,
)
from worldgen.geology.volcanism import compute_volcanism  # noqa: E402
from worldgen.mesh import cache as mesh_cache  # noqa: E402


def _build_base_textures(seed: int):
    """Run the pipeline up to elevation textures *before* noise is applied.

    Returns (dual_mesh, base_textures, params) so the caller can measure the
    base texture seam quality independently from the noise contribution.
    """
    p = ElevationParams(seed=seed)

    if not mesh_cache.is_populated():
        mesh_cache.build(
            icosphere_depth=p.mesh.icosphere_depth,
            n_swaps=p.mesh.n_swaps,
            n_relax=p.mesh.n_relax,
            stiffness=p.mesh.stiffness,
            outer_iters=p.mesh.outer_iters,
        )
    dual = mesh_cache.load_random(seed=seed + 5)

    plate_ids = generate_plates(
        dual, n_plates=p.plates.n_plates, seed=seed,
        compactness=p.plates.plate_compactness,
    )
    plate_ids = relax_plates(dual, plate_ids, max_passes=p.plates.relax_max_passes)
    plate_types_by_plate = assign_plate_types(
        p.plates.n_plates, oceanic_fraction=p.plates.oceanic_fraction, seed=seed + 1,
    )
    plate_types = [plate_types_by_plate[pid] for pid in plate_ids]
    plate_omegas = generate_plate_velocities(
        dual, plate_ids, p.plates.n_plates,
        drift_scale=p.plate_motion.drift_scale,
        spin_scale=p.plate_motion.spin_scale,
        seed=seed + 2,
    )
    boundary_data = classify_plate_boundaries(dual, plate_ids, plate_omegas)
    elevations = compute_face_elevations(
        dual, plate_types, boundary_data,
        decay_rate=p.elevation.elevation_decay_rate,
        max_hops=p.elevation.elevation_max_hops,
        rate_norm=p.elevation.elevation_rate_norm,
    )
    volcanism = compute_volcanism(
        dual, plate_types, boundary_data,
        n_hotspots=p.volcanism.n_hotspots,
        decay_rate=p.volcanism.volcanic_decay_rate,
        max_hops=p.volcanism.volcanic_max_hops,
        rate_norm=p.elevation.elevation_rate_norm,
        seed=seed + 4,
    )
    elevations = compute_erosion(
        dual, elevations,
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
        + (1 - oceanic_mask) * p.volcanism.volcanic_elevation_scale
    )
    elevations += volcanism * elev_scale

    base_textures = compute_elevation_textures(
        dual, elevations, tex_size=p.elevation.elevation_tex_size,
    )
    return dual, base_textures, p


def measure_seed(seed: int, pre_noise: bool = False) -> dict:
    if pre_noise:
        dual, textures, _ = _build_base_textures(seed)
    else:
        params = ElevationParams(seed=seed)
        builder = ElevationBuilder(params)
        builder.build(progress=lambda _: None)
        assert builder.mesh is not None
        dual = builder.mesh
        textures = dual.face_textures["elevation"]

    stats = measure_boundary_elevation_jumps(dual, textures)
    stats["seed"] = seed
    return stats


def print_table(rows: list[dict]) -> None:
    header = (
        f"{'seed':>8}  {'mean (m)':>10}  {'median (m)':>10}"
        f"  {'RMS (m)':>10}  {'max (m)':>10}  {'n_pairs':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['seed']:>8}  "
            f"{r['mean_km'] * 1000:>10.1f}  "
            f"{r['median_km'] * 1000:>10.1f}  "
            f"{r['rms_km'] * 1000:>10.1f}  "
            f"{r['max_km'] * 1000:>10.1f}  "
            f"{r['n_pairs']:>8}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "seeds", nargs="*", type=int,
        help="Explicit seed values (default: 42 137 999)",
    )
    parser.add_argument(
        "--random", metavar="N", type=int,
        help="Use N random seeds instead of explicit ones",
    )
    parser.add_argument(
        "--pre-noise", action="store_true",
        help="Measure base textures before noise is added",
    )
    args = parser.parse_args()

    if args.random:
        seeds = [random.randint(0, 2**31 - 1) for _ in range(args.random)]
    elif args.seeds:
        seeds = args.seeds
    else:
        seeds = [42, 137, 999]

    label = "base (pre-noise)" if args.pre_noise else "final (post-noise)"
    print(f"\nBoundary elevation jump statistics — {label}\n")

    rows = []
    for seed in seeds:
        print(f"  building seed={seed}…", end="", flush=True)
        row = measure_seed(seed, pre_noise=args.pre_noise)
        rows.append(row)
        print(f"  max={row['max_km'] * 1000:.0f} m")

    print()
    print_table(rows)
    print()


if __name__ == "__main__":
    main()
