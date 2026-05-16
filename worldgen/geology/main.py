import colorsys
import sys

import numpy as np
from PySide6.QtWidgets import QApplication

from geology.tectonics import (
    assign_plate_types,
    classify_plate_boundaries,
    compute_face_velocities,
    compute_plate_centroids,
    generate_plate_velocities,
    generate_plates,
    relax_plates,
)
from mapview.mapview import MapWindow
from mesh.mymesh import Mesh

N_PLATES = 12
OCEAN_COLOR = (0.25, 0.55, 0.85, 1.0)
LAND_COLOR = (0.72, 0.52, 0.25, 1.0)
BOUNDARY_COLORS = {
    "interior":   (0.70, 0.70, 0.70, 1.0),
    "convergent": (0.85, 0.30, 0.10, 1.0),
    "divergent":  (0.95, 0.90, 0.30, 1.0),
    "transform":  (0.60, 0.20, 0.80, 1.0),
}

def _plate_palette(n: int) -> np.ndarray:
    """Return (n, 4) float32 RGBA colors evenly spaced around the HSV hue wheel."""
    hues = np.linspace(0, 1, n, endpoint=False)
    rgba = np.ones((n, 4), dtype=np.float32)
    for i, h in enumerate(hues):
        r, g, b = colorsys.hsv_to_rgb(h, 0.65, 0.85)
        rgba[i, :3] = r, g, b
    return rgba


if __name__ == "__main__":
    app = QApplication(sys.argv)

    mesh = Mesh.icosphere(4).deregularize(
        n_swaps=1000, n_relax=10, stiffness=0.1, outer_iters=3
    )
    dual = mesh.mesh_dual()

    plate_ids = generate_plates(dual, n_plates=N_PLATES)
    plate_ids = relax_plates(dual, plate_ids)
    plate_types = assign_plate_types(N_PLATES)
    plate_types = [plate_types[i] for i in plate_ids]

    # Apply plate_type before velocity generation so NNR correction can density-weight.
    dual.apply_metadata_channel("plate_type", plate_types)

    _, centroid_faces = compute_plate_centroids(dual, plate_ids, N_PLATES)
    plate_omegas = generate_plate_velocities(dual, plate_ids, N_PLATES)
    face_velocities = compute_face_velocities(dual, plate_ids, plate_omegas)

    boundary_types = classify_plate_boundaries(dual, plate_ids, plate_omegas)
    speeds = np.linalg.norm(face_velocities, axis=1)
    max_speed = float(speeds.max()) or 1.0

    dual.apply_metadata_channel("plate_id", plate_ids)
    dual.apply_metadata_channel("boundary_type", boundary_types)
    dual.apply_metadata_channel("surface_velocity", [tuple(v) for v in face_velocities])
    dual.apply_metadata_channel("surface_speed", speeds.tolist())
    palette = _plate_palette(N_PLATES)
    dual.set_face_colors(palette[np.array(plate_ids, dtype=np.int32)])

    def _speed_color(meta: dict) -> tuple:
        t = meta["surface_speed"] / max_speed
        return (t, 0.2 + 0.3 * (1.0 - t), 1.0 - t, 1.0)

    window = MapWindow(mesh=dual)
    window.setWindowTitle("Plate Tectonics")
    window.set_color_schemes({
        "Plates": lambda meta: tuple(palette[meta["plate_id"]]),
        "Plate Type": lambda meta: OCEAN_COLOR if meta["plate_type"] == "oceanic" else LAND_COLOR,
        "Speed": _speed_color,
        "Boundary": lambda meta: BOUNDARY_COLORS[meta["boundary_type"]],
    })
    window.show()

    sys.exit(app.exec())
