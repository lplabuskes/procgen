import sys

from PySide6.QtWidgets import QApplication

from worldgen.geology.display import build_elevation_display_texture
from worldgen.globe import GlobeBuilder, GlobeParams
from worldgen.mesh.gui import MainWindow


def main() -> None:
    app = QApplication(sys.argv)

    builder = GlobeBuilder(GlobeParams())
    builder.build()

    mesh = builder.elevation_builder.mesh
    assert mesh is not None
    elevation_rgba = build_elevation_display_texture(mesh)
    mesh.face_textures.clear()
    mesh.set_face_textures("elevation_rgba", elevation_rgba)

    window = MainWindow(mesh=mesh)
    window.setWindowTitle("Procedural World — Mesh View")
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
