import sys

import numpy as np
from PySide6.QtWidgets import QApplication

from utils.mapview.mapview import MapWindow
from core.mesh import Mesh

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mesh = Mesh.icosphere(4).deregularize(
        n_swaps=1000, n_relax=10, stiffness=0.1, outer_iters=3
    )
    dual = mesh.mesh_dual()

    N = len(dual.faces)
    H, W = 32, 32

    u = np.linspace(0.0, 1.0, W)
    v = np.linspace(0.0, 1.0, H)
    uu, vv = np.meshgrid(u, v)
    dist = np.sqrt((uu - 0.5) ** 2 + (vv - 0.5) ** 2)
    brightness = (0.15 + 0.85 * (1.0 - np.clip(dist / (0.5 * np.sqrt(2)), 0.0, 1.0))).astype(np.float32)

    rgb = dual.face_colors[:, :3]
    color_data = (rgb[:, np.newaxis, np.newaxis, :] * brightness[np.newaxis, :, :, np.newaxis]).astype(np.float32)
    alpha = np.ones((N, H, W, 1), dtype=np.float32)
    dual.set_face_textures("default", np.concatenate([color_data, alpha], axis=3))

    window = MapWindow(mesh=dual)
    window.show()
    sys.exit(app.exec())
