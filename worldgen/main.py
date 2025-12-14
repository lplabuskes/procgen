import sys

from PySide6.QtWidgets import QApplication

from mymesh import Mesh
from gui import MainWindow


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mesh = Mesh.icosphere(3)
    window = MainWindow(mesh=mesh)
    window.show()
    sys.exit(app.exec())
