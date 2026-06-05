import sys

import numpy as np
from PySide6.QtWidgets import QApplication

from core.mesh import Mesh
from worldgen.mesh.gui import MainWindow


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mesh = Mesh.icosphere(4).deregularize(n_swaps=1000, n_relax=10, stiffness=0.1, outer_iters=3)
    dual = mesh.mesh_dual()

    N = len(dual.faces)
    H, W = 32, 32

    # Build a radial gradient mask: 1.0 at center (0.5, 0.5), fading toward 0.15 at corners.
    u = np.linspace(0.0, 1.0, W)
    v = np.linspace(0.0, 1.0, H)
    uu, vv = np.meshgrid(u, v)                            # (H, W)
    dist = np.sqrt((uu - 0.5) ** 2 + (vv - 0.5) ** 2)   # 0 at center, ~0.707 at corners
    brightness = (0.15 + 0.85 * (1.0 - np.clip(dist / (0.5 * np.sqrt(2)), 0.0, 1.0))).astype(np.float32)

    # Per-face radial gradient tinted by the face's random color.
    rgb = dual.face_colors[:, :3]                          # (N, 3)
    rgb_4d = rgb[:, np.newaxis, np.newaxis, :]             # (N, 1, 1, 3)
    bright_4d = brightness[np.newaxis, :, :, np.newaxis]   # (1, H, W, 1)
    color_data = (rgb_4d * bright_4d).astype(np.float32)   # (N, H, W, 3)
    alpha = np.ones((N, H, W, 1), dtype=np.float32)
    textures = np.concatenate([color_data, alpha], axis=3)  # (N, H, W, 4)

    dual.set_face_textures("default", textures)

    window = MainWindow(mesh=dual)
    window.show()
    sys.exit(app.exec())
