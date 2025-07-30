from mymesh import generate_mesh
from gui import MainWindow

from PySide6.QtWidgets import QApplication
import sys


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mesh = generate_mesh()
    window = MainWindow(mesh=mesh)
    window.show()
    sys.exit(app.exec())
