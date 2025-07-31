from mymesh import Mesh
from gui import MainWindow

from PySide6.QtWidgets import QApplication
import sys


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mesh = Mesh.icosphere(3)
    window = MainWindow(mesh=mesh)
    window.show()
    sys.exit(app.exec())
