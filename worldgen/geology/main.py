import colorsys
import sys

import numpy as np
from PySide6.QtWidgets import QApplication

from geology.globe_builder import GlobeBuilder, GlobeParams
from mapview.mapview import MapWindow

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

    builder = GlobeBuilder(GlobeParams())
    gmesh = builder.build()

    n_plates = builder.params.n_plates
    palette = _plate_palette(n_plates)
    max_speed = builder.max_speed
    gmesh.set_face_colors(palette[np.array(builder.plate_ids, dtype=np.int32)])

    def _speed_color(meta: dict) -> tuple:
        t = meta["surface_speed"] / max_speed
        return (t, 0.2 + 0.3 * (1.0 - t), 1.0 - t, 1.0)

    window = MapWindow(mesh=gmesh)
    window.setWindowTitle("Plate Tectonics")
    window.set_color_schemes({
        "Plates": lambda meta: tuple(palette[meta["plate_id"]]),
        "Plate Type": lambda meta: OCEAN_COLOR if meta["plate_type"] == "oceanic" else LAND_COLOR,
        "Speed": _speed_color,
        "Boundary": lambda meta: BOUNDARY_COLORS[meta["boundary_type"]],
    })
    window.show()

    sys.exit(app.exec())
