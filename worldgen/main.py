import sys

from PySide6.QtWidgets import QApplication

from worldgen.geology.display import apply_geology_display
from worldgen.globe import GlobeBuilder, GlobeParams
from worldgen.mapview.mapview import MapWindow


def _build_and_apply(window: MapWindow) -> None:
    window.set_rebuild_enabled(False)

    def _progress(msg: str) -> None:
        window.set_status(msg)
        QApplication.processEvents()

    builder = GlobeBuilder(GlobeParams())
    globe = builder.build(progress=_progress)

    window.set_globe(globe)
    apply_geology_display(window, builder.elevation_builder)

    window.set_status("")
    window.set_rebuild_enabled(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MapWindow()
    window.setWindowTitle("Procedural World")
    window.set_rebuild_callback(lambda: _build_and_apply(window))
    _build_and_apply(window)
    window.show()

    sys.exit(app.exec())
