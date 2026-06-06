import sys

from PySide6.QtWidgets import QApplication

from core.pipeline import BuildContext
from core.world import World
from worldgen.geology.display import build_elevation_display_texture
from worldgen.globe import GlobePipeline, GlobeParams
from viz.mesh.gui import MainWindow


def main() -> None:
    app = QApplication(sys.argv)

    ctx = BuildContext(world=World())
    builder = GlobePipeline(GlobeParams())
    builder.build(ctx)

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
