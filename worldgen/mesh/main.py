import sys

from PySide6.QtWidgets import QApplication

from mymesh import Mesh, Polytope
from gui import MainWindow


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mesh = Mesh.icosphere(4).deregularize(n_swaps=1000, n_relax=10, stiffness=0.1, outer_iters=3)
    dual = Polytope.mesh_dual(mesh)
    window = MainWindow(mesh=dual)
    window.show()
    sys.exit(app.exec())
