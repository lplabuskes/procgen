import colorsys
import sys

import numpy as np
from PySide6.QtWidgets import QApplication

from geology.tectonics import generate_plate_velocities, generate_plates, relax_plates
from mapview.mapview import MapWindow
from mesh.mymesh import Mesh

N_PLATES = 12


def _plate_palette(n: int) -> np.ndarray:
    """Return (n, 4) float32 RGBA colors evenly spaced around the HSV hue wheel."""
    hues = np.linspace(0, 1, n, endpoint=False)
    rgba = np.ones((n, 4), dtype=np.float32)
    for i, h in enumerate(hues):
        r, g, b = colorsys.hsv_to_rgb(h, 0.65, 0.85)
        rgba[i, :3] = r, g, b
    return rgba


def _face_colors(plate_ids: list[int], palette: np.ndarray) -> np.ndarray:
    ids = np.array(plate_ids, dtype=np.int32)
    return palette[ids]


if __name__ == "__main__":
    app = QApplication(sys.argv)

    mesh = Mesh.icosphere(3).deregularize(
        n_swaps=1000, n_relax=10, stiffness=0.1, outer_iters=3
    )
    dual = mesh.mesh_dual()

    plate_ids = generate_plates(dual, n_plates=N_PLATES)
    plate_omegas = generate_plate_velocities(dual, plate_ids, n_plates=N_PLATES)
    plate_ids = relax_plates(dual, plate_ids, plate_omegas)
    palette = _plate_palette(N_PLATES)
    colors = _face_colors(plate_ids, palette)

    window = MapWindow(mesh=dual)
    window.set_face_colors(colors)
    window.setWindowTitle("Plate Tectonics")
    window.show()

    sys.exit(app.exec())
