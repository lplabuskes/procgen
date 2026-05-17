import dataclasses
import random
import sys

from PySide6.QtWidgets import QApplication

from worldgen.geology.display import apply_geology_display
from worldgen.globe import GlobeBuilder, GlobeParams
from worldgen.mapview.mapview import MapWindow
from worldgen.params_dialog import GlobeParamsDialog


def main() -> None:
    app = QApplication(sys.argv)

    window = MapWindow()
    window.setWindowTitle("Procedural World")

    # Mutable container so closures can reassign current_params.
    current_params: list[GlobeParams] = [GlobeParams()]

    def _build_and_apply(params: GlobeParams) -> None:
        window.set_rebuild_enabled(False)

        def _progress(msg: str) -> None:
            window.set_status(msg)
            QApplication.processEvents()

        # Lock in a concrete seed so the params dialog preserves the same world layout.
        if params.elevation.seed is None:
            ep = dataclasses.replace(params.elevation, seed=random.randint(0, 2**31 - 1))
            params = dataclasses.replace(params, elevation=ep)

        current_params[0] = params

        builder = GlobeBuilder(params)
        globe = builder.build(progress=_progress)

        window.set_globe(globe)
        apply_geology_display(window, builder.elevation_builder)

        window.set_status("")
        window.set_rebuild_enabled(True)

    def _rerun() -> None:
        # Generate a fresh world with a new random seed.
        fresh = dataclasses.replace(current_params[0], elevation=dataclasses.replace(
            current_params[0].elevation, seed=None
        ))
        _build_and_apply(fresh)

    def _open_params() -> None:
        dialog = GlobeParamsDialog(current_params[0], parent=window)
        if dialog.exec() == GlobeParamsDialog.DialogCode.Accepted:
            new_params = dialog.result_params(current_params[0])
            if new_params != current_params[0]:
                _build_and_apply(new_params)

    window.set_rebuild_callback(_rerun)
    window.set_params_callback(_open_params)
    _build_and_apply(GlobeParams())
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
