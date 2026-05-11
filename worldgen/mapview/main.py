import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mesh"))

from PySide6.QtWidgets import QApplication

from mapview import MapWindow
from mymesh import Mesh

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mesh = Mesh.icosphere(4).deregularize(
        n_swaps=1000, n_relax=10, stiffness=0.1, outer_iters=3
    )
    dual = mesh.mesh_dual()
    window = MapWindow(mesh=dual)
    window.show()
    sys.exit(app.exec())
